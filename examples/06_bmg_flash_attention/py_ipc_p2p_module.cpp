// Cross-process IPC-handle based P2P copy for XPU tensors, using the
// Level-Zero copy engine (zeCommandListAppendMemoryCopy on an async immediate
// command list). This bypasses torch.distributed for K/V data movement in ring
// attention so the transfer truly overlaps the attention compute kernel.
//
// Route (recommended, minimal change):
//   sender : zeMemGetIpcHandle(local_buffer)  -> 64B handle blob
//   (exchange handle blobs via the existing process group, done in Python)
//   receiver: zeMemOpenIpcHandle(peer_handle) -> device pointer into peer mem
//   each round: zeCommandListAppendMemoryCopy(dst_local, peer_src, bytes)
//               on a copy-engine immediate command list -> async event.
//
// The Python side owns the ring buffers, calls export/open once, then issues
// one async copy per round and waits on the returned event handle.

#include <torch/extension.h>
#include <pybind11/stl.h>

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace {

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------
inline void zeCheck(ze_result_t r, const char *what) {
  TORCH_CHECK(r == ZE_RESULT_SUCCESS, what, " failed: ze_result_t=0x",
              std::hex, static_cast<uint32_t>(r));
}

// ---------------------------------------------------------------------------
// Level-Zero context/device/queue plumbing derived from the current XPU tensor.
//
// We fetch the native L0 handles that back the SYCL runtime PyTorch is using,
// so IPC handles and pointers live in the exact same L0 context PyTorch owns.
// ---------------------------------------------------------------------------
struct L0Env {
  ze_context_handle_t context = nullptr;
  ze_device_handle_t device = nullptr;
  // Async immediate command list backed by a copy-engine ordinal.
  ze_command_list_handle_t copy_cmdlist = nullptr;
  uint32_t copy_ordinal = 0;
};

// One L0Env per (device_index) so multi-XPU still works.
std::mutex g_env_mutex;
std::unordered_map<int, std::shared_ptr<L0Env>> g_env_by_device;

// Cache of opened peer pointers keyed by the raw 64B IPC handle bytes, so we
// only zeMemOpenIpcHandle once per unique peer buffer.
std::mutex g_peer_mutex;
std::unordered_map<std::string, void *> g_opened_peers;

uint32_t find_copy_ordinal(ze_device_handle_t device) {
  uint32_t count = 0;
  zeCheck(zeDeviceGetCommandQueueGroupProperties(device, &count, nullptr),
          "zeDeviceGetCommandQueueGroupProperties(count)");
  std::vector<ze_command_queue_group_properties_t> props(count);
  for (auto &p : props) {
    p.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    p.pNext = nullptr;
  }
  zeCheck(zeDeviceGetCommandQueueGroupProperties(device, &count, props.data()),
          "zeDeviceGetCommandQueueGroupProperties(props)");

  // Prefer a pure copy engine (has COPY, lacks COMPUTE); fall back to any group
  // that supports copy.
  uint32_t fallback = UINT32_MAX;
  for (uint32_t i = 0; i < count; ++i) {
    const bool can_copy =
        props[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY;
    const bool can_compute =
        props[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE;
    if (can_copy && !can_compute) {
      return i; // dedicated copy engine
    }
    if (can_copy && fallback == UINT32_MAX) {
      fallback = i;
    }
  }
  TORCH_CHECK(fallback != UINT32_MAX,
              "no command queue group with COPY capability on this device");
  return fallback;
}

// 从 torch 传入的 sycl::queue* 地址反查 L0 context/device。
// 这是与 tensor 分配严格同一个 L0 context 的唯一可靠来源。
static void l0_from_queue_ptr(uintptr_t queue_addr, ze_context_handle_t &out_ctx,
                              ze_device_handle_t &out_dev) {
  TORCH_CHECK(queue_addr != 0, "null sycl_queue pointer from torch");
  sycl::queue *q = reinterpret_cast<sycl::queue *>(queue_addr);
  sycl::context ctx = q->get_context();
  sycl::device dev = q->get_device();
  out_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
  out_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
}

// env 现在按 (device_index) 缓存，但构造时需要 torch 的 queue 指针。
std::shared_ptr<L0Env> env_for_tensor(const at::Tensor &t, uintptr_t queue_addr) {
  TORCH_CHECK(t.device().type() == c10::DeviceType::XPU,
              "tensor must be an XPU tensor");
  TORCH_CHECK(t.device().has_index(), "tensor must have a device index");
  const int dev_index = t.device().index();

  {
    std::lock_guard<std::mutex> lock(g_env_mutex);
    auto it = g_env_by_device.find(dev_index);
    if (it != g_env_by_device.end()) {
      return it->second;
    }
  }

  zeCheck(zeInit(0), "zeInit");

  auto env = std::make_shared<L0Env>();
  l0_from_queue_ptr(queue_addr, env->context, env->device);
  env->copy_ordinal = find_copy_ordinal(env->device);

  ze_command_queue_desc_t q_desc = {};
  q_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  q_desc.ordinal = env->copy_ordinal;
  q_desc.index = 0;
  q_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  q_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
  zeCheck(zeCommandListCreateImmediate(env->context, env->device, &q_desc,
                                       &env->copy_cmdlist),
          "zeCommandListCreateImmediate(copy)");

  {
    std::lock_guard<std::mutex> lock(g_env_mutex);
    g_env_by_device[dev_index] = env;
  }
  return env;
}

// A small event pool so each in-flight copy gets its own event.
struct EventPool {
  ze_event_pool_handle_t pool = nullptr;
  std::vector<ze_event_handle_t> events;
  std::vector<bool> in_use;
  ze_context_handle_t context = nullptr;
  std::mutex mtx;

  void init(ze_context_handle_t ctx, ze_device_handle_t dev, uint32_t capacity) {
    context = ctx;
    ze_event_pool_desc_t pool_desc = {};
    pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    pool_desc.count = capacity;
    zeCheck(zeEventPoolCreate(context, &pool_desc, 1, &dev, &pool),
            "zeEventPoolCreate");
    events.resize(capacity, nullptr);
    in_use.assign(capacity, false);
    for (uint32_t i = 0; i < capacity; ++i) {
      ze_event_desc_t ed = {};
      ed.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
      ed.index = i;
      ed.signal = ZE_EVENT_SCOPE_FLAG_HOST;
      ed.wait = ZE_EVENT_SCOPE_FLAG_HOST;
      zeCheck(zeEventCreate(pool, &ed, &events[i]), "zeEventCreate");
    }
  }

  ze_event_handle_t acquire(uint32_t &out_slot) {
    std::lock_guard<std::mutex> lock(mtx);
    for (uint32_t i = 0; i < events.size(); ++i) {
      if (!in_use[i]) {
        in_use[i] = true;
        out_slot = i;
        zeEventHostReset(events[i]);
        return events[i];
      }
    }
    TORCH_CHECK(false, "IPC copy event pool exhausted; increase pool size");
    return nullptr;
  }

  void release(uint32_t slot) {
    std::lock_guard<std::mutex> lock(mtx);
    in_use[slot] = false;
  }
};

std::mutex g_evpool_mutex;
std::unordered_map<int, std::shared_ptr<EventPool>> g_evpool_by_device;

std::shared_ptr<EventPool> evpool_for_env(int dev_index,
                                          const std::shared_ptr<L0Env> &env) {
  std::lock_guard<std::mutex> lock(g_evpool_mutex);
  auto it = g_evpool_by_device.find(dev_index);
  if (it != g_evpool_by_device.end()) {
    return it->second;
  }
  auto ep = std::make_shared<EventPool>();
  // 64 in-flight copies is plenty for a ring (world <= handful, 2 buffers).
  ep->init(env->context, env->device, 64);
  g_evpool_by_device[dev_index] = ep;
  return ep;
}

// Opaque per-round handle returned to Python so it can wait later.
struct PendingCopy {
  std::shared_ptr<L0Env> env;
  std::shared_ptr<EventPool> pool;
  ze_event_handle_t event = nullptr;
  uint32_t slot = 0;
  bool waited = false;
};

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Export the IPC handle of a tensor's storage. Returns raw handle bytes as a
// Python bytes object suitable for exchange over torch.distributed.
py::bytes ipc_get_handle(const at::Tensor &t, uintptr_t queue_ptr) {
  TORCH_CHECK(t.is_contiguous(), "tensor must be contiguous to export IPC");
  auto env = env_for_tensor(t, queue_ptr);
  ze_ipc_mem_handle_t handle = {};
  zeCheck(zeMemGetIpcHandle(env->context, t.data_ptr(), &handle),
          "zeMemGetIpcHandle");
  return py::bytes(reinterpret_cast<const char *>(handle.data),
                   sizeof(handle.data));
}

uintptr_t ipc_open_handle(const at::Tensor &like, const py::bytes &handle_bytes,
                          uintptr_t queue_ptr) {
  auto env = env_for_tensor(like, queue_ptr);
  std::string key = handle_bytes;
  {
    std::lock_guard<std::mutex> lock(g_peer_mutex);
    auto it = g_opened_peers.find(key);
    if (it != g_opened_peers.end()) {
      return reinterpret_cast<uintptr_t>(it->second);
    }
  }
  ze_ipc_mem_handle_t handle = {};
  std::memcpy(handle.data, key.data(),
              std::min(sizeof(handle.data), key.size()));
  void *peer_ptr = nullptr;
  zeCheck(zeMemOpenIpcHandle(env->context, env->device, handle,
                             ZE_IPC_MEMORY_FLAG_BIAS_CACHED, &peer_ptr),
          "zeMemOpenIpcHandle");
  {
    std::lock_guard<std::mutex> lock(g_peer_mutex);
    g_opened_peers[key] = peer_ptr;
  }
  return reinterpret_cast<uintptr_t>(peer_ptr);
}

void ipc_close_handle(const at::Tensor &like, uintptr_t peer_ptr,
                      uintptr_t queue_ptr) {
  auto env = env_for_tensor(like, queue_ptr);
  void *p = reinterpret_cast<void *>(peer_ptr);
  {
    std::lock_guard<std::mutex> lock(g_peer_mutex);
    for (auto it = g_opened_peers.begin(); it != g_opened_peers.end();) {
      if (it->second == p) it = g_opened_peers.erase(it);
      else ++it;
    }
  }
  zeCheck(zeMemCloseIpcHandle(env->context, p), "zeMemCloseIpcHandle");
}

uintptr_t ipc_copy_from_peer_async(at::Tensor &dst_local, uintptr_t peer_src,
                                   int64_t nbytes, uintptr_t queue_ptr) {
  TORCH_CHECK(dst_local.is_contiguous(), "dst_local must be contiguous");
  TORCH_CHECK(nbytes > 0, "nbytes must be positive");
  TORCH_CHECK(static_cast<size_t>(nbytes) <= dst_local.nbytes(),
              "nbytes (", nbytes, ") exceeds dst tensor size (",
              dst_local.nbytes(), ")");
  const int dev_index = dst_local.device().index();
  auto env = env_for_tensor(dst_local, queue_ptr);
  auto pool = evpool_for_env(dev_index, env);
  auto *pending = new PendingCopy();
  pending->env = env;
  pending->pool = pool;
  pending->event = pool->acquire(pending->slot);
  zeCheck(zeCommandListAppendMemoryCopy(
              env->copy_cmdlist, dst_local.data_ptr(),
              reinterpret_cast<const void *>(peer_src),
              static_cast<size_t>(nbytes), pending->event, 0, nullptr),
          "zeCommandListAppendMemoryCopy");
  return reinterpret_cast<uintptr_t>(pending);
}

// Block until an async copy completes and recycle its event.
void ipc_wait(uintptr_t pending_handle) {
  auto *pending = reinterpret_cast<PendingCopy *>(pending_handle);
  TORCH_CHECK(pending != nullptr, "null pending handle");
  if (!pending->waited) {
    zeCheck(zeEventHostSynchronize(pending->event, UINT64_MAX),
            "zeEventHostSynchronize");
    pending->pool->release(pending->slot);
    pending->waited = true;
  }
  delete pending;
}

// Query completion without blocking. Returns true if done. Does NOT free the
// handle; call ipc_wait() to free.
bool ipc_query(uintptr_t pending_handle) {
  auto *pending = reinterpret_cast<PendingCopy *>(pending_handle);
  TORCH_CHECK(pending != nullptr, "null pending handle");
  return zeEventQueryStatus(pending->event) == ZE_RESULT_SUCCESS;
}

PYBIND11_MODULE(sycl_tla_ipc_p2p, m) {
  m.doc() = "XPU Level-Zero IPC-handle P2P copy engine transfers";
  m.def("ipc_get_handle", &ipc_get_handle,
        py::arg("tensor"), py::arg("queue_ptr"));
  m.def("ipc_open_handle", &ipc_open_handle,
        py::arg("like_tensor"), py::arg("handle_bytes"), py::arg("queue_ptr"));
  m.def("ipc_close_handle", &ipc_close_handle,
        py::arg("like_tensor"), py::arg("peer_ptr"), py::arg("queue_ptr"));
  m.def("ipc_copy_from_peer_async", &ipc_copy_from_peer_async,
        py::arg("dst_local"), py::arg("peer_src"), py::arg("nbytes"),
        py::arg("queue_ptr"));
  m.def("ipc_wait", &ipc_wait, py::arg("pending_handle"));
  m.def("ipc_query", &ipc_query, py::arg("pending_handle"));
}
