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

#include <linux/futex.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <atomic>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace {

// ---------------------------------------------------------------------------
// Linux cross-process control plane.
//
// Ring data still moves through Level-Zero IPC CE writes. This control plane
// only transports small host-side epochs:
//
//   handle_ready[rank]        IPC arena handle published during bootstrap
//   ready[rank][slot]         K/V slot has been completely written
//   free[rank][slot]          rank has finished consuming the old slot value
//   barrier_epoch[rank]       infrequent non-hot-path rendezvous
//   work_done[rank]           all local compute/copies stopped at teardown
//   mapping_closed[rank]      rank closed its mapping to next_rank
//   elapsed_ready[rank]       benchmark elapsed_ns value is available
//
// FUTEX_WAIT, rather than FUTEX_WAIT_PRIVATE, is required because the mapping
// is shared between processes.
// ---------------------------------------------------------------------------
constexpr uint64_t kRingControlMagic = 0x5359434c52494e47ULL; // "SYCLRING"
constexpr uint32_t kRingControlVersion = 2;

inline size_t align_up_size(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

struct alignas(64) RingControlHeader {
  std::atomic<uint32_t> initialized;
  uint32_t version;
  uint32_t world;
  uint32_t slots;
  uint32_t handle_bytes;
  uint32_t reserved;
  uint64_t magic;
};

struct RingControlLayout {
  size_t total_size = 0;
  size_t handle_ready_off = 0;
  size_t handles_off = 0;
  size_t ready_off = 0;
  size_t free_off = 0;
  size_t barrier_epoch_off = 0;
  size_t work_done_off = 0;
  size_t mapping_closed_off = 0;
  size_t elapsed_ready_off = 0;
  size_t elapsed_ns_off = 0;
};

RingControlLayout ring_control_layout(uint32_t world, uint32_t slots,
                                      uint32_t handle_bytes) {
  RingControlLayout l;
  size_t off = align_up_size(sizeof(RingControlHeader), 64);

  l.handle_ready_off = off;
  off += sizeof(std::atomic<uint32_t>) * world;
  off = align_up_size(off, 64);

  l.handles_off = off;
  off += static_cast<size_t>(world) * handle_bytes;
  off = align_up_size(off, 64);

  l.ready_off = off;
  off += sizeof(std::atomic<uint32_t>) *
         static_cast<size_t>(world) * slots;
  off = align_up_size(off, 64);

  l.free_off = off;
  off += sizeof(std::atomic<uint32_t>) *
         static_cast<size_t>(world) * slots;
  off = align_up_size(off, 64);

  l.barrier_epoch_off = off;
  off += sizeof(std::atomic<uint32_t>) * world;
  off = align_up_size(off, 64);

  l.work_done_off = off;
  off += sizeof(std::atomic<uint32_t>) * world;
  off = align_up_size(off, 64);

  l.mapping_closed_off = off;
  off += sizeof(std::atomic<uint32_t>) * world;
  off = align_up_size(off, 64);

 l.elapsed_ready_off = off;
  off += sizeof(std::atomic<uint32_t>) * world;
  off = align_up_size(off, 64);

  l.elapsed_ns_off = off;
  off += sizeof(uint64_t) * world;

  l.total_size = align_up_size(off, 4096);
  return l;
}

timespec duration_to_timespec(std::chrono::steady_clock::duration duration) {
  if (duration <= std::chrono::steady_clock::duration::zero()) {
    return timespec{0, 1};
  }

  const auto ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
  timespec ts{};
  ts.tv_sec = static_cast<time_t>(ns / 1000000000LL);
  ts.tv_nsec = static_cast<long>(ns % 1000000000LL);
  return ts;
}

int futex_wait_shared(std::atomic<uint32_t> *address, uint32_t expected,
                      const timespec *timeout) {
  return static_cast<int>(
      syscall(SYS_futex, reinterpret_cast<uint32_t *>(address),
              FUTEX_WAIT, expected, timeout, nullptr, 0));
}

void futex_wake_shared(std::atomic<uint32_t> *address) {
  syscall(SYS_futex, reinterpret_cast<uint32_t *>(address),
          FUTEX_WAKE, INT_MAX, nullptr, nullptr, 0);
}

void publish_epoch(std::atomic<uint32_t> *address, uint32_t epoch) {
  address->store(epoch, std::memory_order_release);
  futex_wake_shared(address);
}

void wait_epoch(std::atomic<uint32_t> *address, uint32_t target,
                int64_t timeout_ms, const std::string &description) {
  TORCH_CHECK(timeout_ms > 0, "timeout_ms must be positive");

  const auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::milliseconds(timeout_ms);

  // Short adaptive spin avoids a syscall when the CE completion notification
  // is only a few microseconds away.
  for (int i = 0; i < 256; ++i) {
    if (address->load(std::memory_order_acquire) >= target) {
      return;
    }
  }

  for (;;) {
    const uint32_t observed = address->load(std::memory_order_acquire);
    if (observed >= target) {
      return;
    }

    const auto now = std::chrono::steady_clock::now();
    TORCH_CHECK(
        now < deadline,
        "timed out waiting for ", description,
        ": target epoch=", target, ", observed epoch=", observed,
        ", timeout_ms=", timeout_ms);

    const timespec relative_timeout =
        duration_to_timespec(deadline - now);

    errno = 0;
    const int rc =
        futex_wait_shared(address, observed, &relative_timeout);
    if (rc == 0 || errno == EAGAIN || errno == EINTR) {
      continue;
    }

    TORCH_CHECK(
        errno != ETIMEDOUT,
        "timed out waiting for ", description,
        ": target epoch=", target, ", observed epoch=", observed,
        ", timeout_ms=", timeout_ms);

    TORCH_CHECK(false,
                "futex wait failed while waiting for ", description,
                ": errno=", errno, " (", std::strerror(errno), ")");
  }
}

class RingControl {
public:
  RingControl(std::string name, int rank, int world, int slots,
              int handle_bytes, int64_t timeout_ms)
      : name_(std::move(name)),
        rank_(rank),
        world_(world),
        slots_(slots),
        handle_bytes_(handle_bytes),
        timeout_ms_(timeout_ms),
        layout_(ring_control_layout(
            static_cast<uint32_t>(world),
            static_cast<uint32_t>(slots),
            static_cast<uint32_t>(handle_bytes))) {
    TORCH_CHECK(!name_.empty() && name_[0] == '/',
                "shared-memory name must start with '/'");
    TORCH_CHECK(rank_ >= 0 && rank_ < world_,
                "rank out of range: ", rank_, " for world ", world_);
    TORCH_CHECK(world_ >= 2, "RingControl requires world >= 2");
    TORCH_CHECK(slots_ >= 1 && slots_ <= world_ - 1,
                "slots must be in [1, world - 1]; got slots=",
                slots_, ", world=", world_);
    TORCH_CHECK(world_ == 2 || slots_ >= 2,
                "world > 2 requires at least two ring buffers");
    TORCH_CHECK(slots_ <= 3,
                "this ring implementation supports at most three buffers; "
                "got ", slots_);
    TORCH_CHECK(handle_bytes_ > 0, "handle_bytes must be positive");
    TORCH_CHECK(timeout_ms_ > 0, "timeout_ms must be positive");

    if (rank_ == 0) {
      create_mapping();
      initialize_mapping();
    } else {
      open_mapping();
      wait_initialized();
    }

    validate_header();
  }

  ~RingControl() {
    try {
      close(false);
    } catch (...) {
    }
  }

  RingControl(const RingControl &) = delete;
  RingControl &operator=(const RingControl &) = delete;

  void publish_handle(const py::bytes &handle) {
    std::string bytes = handle;
    TORCH_CHECK(
        static_cast<int>(bytes.size()) == handle_bytes_,
        "bad IPC handle size: expected ", handle_bytes_,
        ", got ", bytes.size());

    std::memcpy(handle_ptr(rank_), bytes.data(), bytes.size());
    publish_epoch(handle_ready(rank_), 1);
  }

  py::bytes wait_handle(int peer_rank) {
    check_rank(peer_rank);
    wait_epoch(handle_ready(peer_rank), 1, timeout_ms_,
               "IPC handle from rank " + std::to_string(peer_rank));
    return py::bytes(
        reinterpret_cast<const char *>(handle_ptr(peer_rank)),
        static_cast<py::ssize_t>(handle_bytes_));
  }

  void publish_ready(int receiver_rank, int slot, uint32_t epoch) {
    check_rank(receiver_rank);
    check_slot(slot);
    TORCH_CHECK(epoch > 0, "ready epoch must be positive");
    publish_epoch(ready(receiver_rank, slot), epoch);
  }

  void wait_ready(int receiver_rank, int slot, uint32_t epoch) {
    check_rank(receiver_rank);
    check_slot(slot);
    TORCH_CHECK(epoch > 0, "ready epoch must be positive");
    wait_epoch(
        ready(receiver_rank, slot), epoch, timeout_ms_,
        "ready[rank=" + std::to_string(receiver_rank) +
            "][slot=" + std::to_string(slot) + "]");
  }

  void publish_free(int owner_rank, int slot, uint32_t epoch) {
    check_rank(owner_rank);
    check_slot(slot);
    TORCH_CHECK(epoch > 0, "free epoch must be positive");
    publish_epoch(free_slot(owner_rank, slot), epoch);
  }

  void wait_free(int owner_rank, int slot, uint32_t epoch) {
    check_rank(owner_rank);
    check_slot(slot);
    TORCH_CHECK(epoch > 0, "free epoch must be positive");
    wait_epoch(
        free_slot(owner_rank, slot), epoch, timeout_ms_,
        "free[rank=" + std::to_string(owner_rank) +
            "][slot=" + std::to_string(slot) + "]");
  }

  void barrier(uint32_t epoch) {
    TORCH_CHECK(epoch > 0, "barrier epoch must be positive");

    // Publish first, then wait. This avoids a circular wait where every rank
    // waits before making its own arrival visible.
    publish_epoch(barrier_epoch(rank_), epoch);
    for (int r = 0; r < world_; ++r) {
      wait_epoch(
          barrier_epoch(r), epoch, timeout_ms_,
          "barrier[rank=" + std::to_string(r) + "]");
    }
  }

  void publish_work_done(int rank) {
    check_rank(rank);
    publish_epoch(work_done(rank), 1);
  }

  void wait_work_done(int rank) {
    check_rank(rank);
    wait_epoch(
        work_done(rank), 1, timeout_ms_,
        "work_done[rank=" + std::to_string(rank) + "]");
  }

  void publish_mapping_closed(int rank) {
    check_rank(rank);
    publish_epoch(mapping_closed(rank), 1);
  }

  void wait_mapping_closed(int rank) {
    check_rank(rank);
    wait_epoch(
        mapping_closed(rank), 1, timeout_ms_,
        "mapping_closed[rank=" + std::to_string(rank) + "]");
  }

  void publish_elapsed_ns(int rank, uint64_t elapsed_ns,
                          uint32_t epoch) {
    check_rank(rank);
    TORCH_CHECK(epoch > 0, "elapsed epoch must be positive");

    elapsed_ns_ptr()[rank] = elapsed_ns;
    publish_epoch(elapsed_ready(rank), epoch);
  }

  uint64_t wait_max_elapsed_ns(uint32_t epoch) {
    TORCH_CHECK(epoch > 0, "elapsed epoch must be positive");

    uint64_t maximum = 0;
    for (int r = 0; r < world_; ++r) {
      wait_epoch(
          elapsed_ready(r), epoch, timeout_ms_,
          "elapsed_ns[rank=" + std::to_string(r) + "]");
      maximum = std::max(maximum, elapsed_ns_ptr()[r]);
    }
    return maximum;
  }

  void close(bool unlink_name) {
    if (mapping_ != MAP_FAILED) {
      munmap(mapping_, layout_.total_size);
      mapping_ = MAP_FAILED;
    }
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    if (unlink_name && !unlinked_) {
      if (shm_unlink(name_.c_str()) != 0 && errno != ENOENT) {
        TORCH_CHECK(false,
                    "shm_unlink(", name_, ") failed: errno=", errno,
                    " (", std::strerror(errno), ")");
      }
      unlinked_ = true;
    }
  }

private:
  void create_mapping() {
    // The name contains a launch-specific digest. Unlinking first also cleans
    // up a stale object left by an aborted run using the same torchrun port.
    if (shm_unlink(name_.c_str()) != 0 && errno != ENOENT) {
      TORCH_CHECK(false,
                  "shm_unlink(", name_, ") failed: errno=", errno,
                  " (", std::strerror(errno), ")");
    }

    fd_ = shm_open(name_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    TORCH_CHECK(fd_ >= 0,
                "shm_open(create ", name_, ") failed: errno=", errno,
                " (", std::strerror(errno), ")");

    TORCH_CHECK(
        ftruncate(fd_, static_cast<off_t>(layout_.total_size)) == 0,
        "ftruncate(", name_, ") failed: errno=", errno,
        " (", std::strerror(errno), ")");

    map_fd();
  }

  void open_mapping() {
    const auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::milliseconds(timeout_ms_);

    for (;;) {
      fd_ = shm_open(name_.c_str(), O_RDWR, 0600);
      if (fd_ >= 0) {
        struct stat st {};
        if (fstat(fd_, &st) == 0 &&
            static_cast<size_t>(st.st_size) == layout_.total_size) {
          break;
        }
        ::close(fd_);
        fd_ = -1;
      } else {
        TORCH_CHECK(
            errno == ENOENT,
            "shm_open(open ", name_, ") failed: errno=", errno,
            " (", std::strerror(errno), ")");
      }

      TORCH_CHECK(
          std::chrono::steady_clock::now() < deadline,
          "timed out opening shared ring control ", name_);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    map_fd();
  }

  void map_fd() {
    mapping_ = mmap(nullptr, layout_.total_size,
                    PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    TORCH_CHECK(mapping_ != MAP_FAILED,
                "mmap(", name_, ") failed: errno=", errno,
                " (", std::strerror(errno), ")");
  }

  void initialize_mapping() {
    std::memset(mapping_, 0, layout_.total_size);

    auto *h = header();
    new (&h->initialized) std::atomic<uint32_t>(0);
    h->version = kRingControlVersion;
    h->world = static_cast<uint32_t>(world_);
    h->slots = static_cast<uint32_t>(slots_);
    h->handle_bytes = static_cast<uint32_t>(handle_bytes_);
    h->reserved = 0;
    h->magic = kRingControlMagic;

    initialize_atomic_array(layout_.handle_ready_off, world_);
    initialize_atomic_array(
        layout_.ready_off,
        static_cast<size_t>(world_) * slots_);
    initialize_atomic_array(
        layout_.free_off,
        static_cast<size_t>(world_) * slots_);
    initialize_atomic_array(layout_.barrier_epoch_off, world_);
    initialize_atomic_array(layout_.work_done_off, world_);
    initialize_atomic_array(layout_.mapping_closed_off, world_);
    initialize_atomic_array(layout_.elapsed_ready_off, world_);

    publish_epoch(&h->initialized, 1);
  }

  void wait_initialized() {
    wait_epoch(&header()->initialized, 1, timeout_ms_,
               "ring-control initialization");
  }

  void validate_header() const {
    const auto *h = header();
    TORCH_CHECK(h->magic == kRingControlMagic,
                "bad ring-control shared-memory magic");
    TORCH_CHECK(h->version == kRingControlVersion,
                "ring-control version mismatch: expected ",
                kRingControlVersion, ", got ", h->version);
    TORCH_CHECK(static_cast<int>(h->world) == world_,
                "ring-control world mismatch: expected ", world_,
                ", got ", h->world);
    TORCH_CHECK(static_cast<int>(h->slots) == slots_,
                "ring-control slot mismatch: expected ", slots_,
                ", got ", h->slots);
    TORCH_CHECK(static_cast<int>(h->handle_bytes) == handle_bytes_,
                "ring-control handle size mismatch");
  }

  void initialize_atomic_array(size_t offset, size_t count) {
    auto *array = atomic_array(offset);
    for (size_t i = 0; i < count; ++i) {
      new (&array[i]) std::atomic<uint32_t>(0);
      TORCH_CHECK(array[i].is_lock_free(),
                  "shared uint32_t atomics must be lock-free");
    }
  }

  RingControlHeader *header() const {
    return reinterpret_cast<RingControlHeader *>(mapping_);
  }

  void *at_offset(size_t offset) const {
    return static_cast<char *>(mapping_) + offset;
  }

  std::atomic<uint32_t> *atomic_array(size_t offset) const {
    return reinterpret_cast<std::atomic<uint32_t> *>(at_offset(offset));
  }

  std::atomic<uint32_t> *handle_ready(int rank) const {
    return atomic_array(layout_.handle_ready_off) + rank;
  }

  void *handle_ptr(int rank) const {
    return static_cast<char *>(at_offset(layout_.handles_off)) +
           static_cast<size_t>(rank) * handle_bytes_;
  }

  std::atomic<uint32_t> *ready(int rank, int slot) const {
    return atomic_array(layout_.ready_off) +
           static_cast<size_t>(rank) * slots_ + slot;
  }

  std::atomic<uint32_t> *free_slot(int rank, int slot) const {
    return atomic_array(layout_.free_off) +
           static_cast<size_t>(rank) * slots_ + slot;
  }

  std::atomic<uint32_t> *barrier_epoch(int rank) const {
    return atomic_array(layout_.barrier_epoch_off) + rank;
  }

  std::atomic<uint32_t> *work_done(int rank) const {
    return atomic_array(layout_.work_done_off) + rank;
  }

  std::atomic<uint32_t> *mapping_closed(int rank) const {
    return atomic_array(layout_.mapping_closed_off) + rank;
  }

  std::atomic<uint32_t> *elapsed_ready(int rank) const {
    return atomic_array(layout_.elapsed_ready_off) + rank;
  }

  uint64_t *elapsed_ns_ptr() const {
    return reinterpret_cast<uint64_t *>(at_offset(layout_.elapsed_ns_off));
  }

  void check_rank(int rank) const {
    TORCH_CHECK(rank >= 0 && rank < world_,
                "rank out of range: ", rank, " for world ", world_);
  }

  void check_slot(int slot) const {
    TORCH_CHECK(slot >= 0 && slot < slots_,
                "slot out of range: ", slot, " for slots ", slots_);
  }

  std::string name_;
  int rank_;
  int world_;
  int slots_;
  int handle_bytes_;
  int64_t timeout_ms_;
  RingControlLayout layout_;
  int fd_ = -1;
  void *mapping_ = MAP_FAILED;
  bool unlinked_ = false;
};

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------
inline void zeCheck(ze_result_t r, const char *what) {
  TORCH_CHECK(r == ZE_RESULT_SUCCESS, what, " failed: ze_result_t=0x",
              std::hex, static_cast<uint32_t>(r));
}

inline size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

// ---------------------------------------------------------------------------
// Level-Zero context/device/queue plumbing derived from the current XPU tensor.
// ---------------------------------------------------------------------------
struct L0Env {
  ze_context_handle_t context = nullptr;
  ze_device_handle_t device = nullptr;
  ze_command_list_handle_t copy_cmdlist = nullptr;
  uint32_t copy_ordinal = 0;
  // 记录建立本 env 时使用的 sycl::queue 地址，用于检测跨 stream 误用。
  uintptr_t queue_addr = 0;
  int device_index = -1;
};

// 本端导出过的 IPC handle：zeMemGetIpcHandle 在 Linux 上返回 dma-buf fd，
// 必须配对 zeMemPutIpcHandle，否则每次导出泄漏一个 fd（vllm 每层导出 → EMFILE）。
struct ExportedHandle {
  ze_ipc_mem_handle_t handle{};
  void *base = nullptr;
  int refcount = 0;
};

// 打开的对端映射：按 handle 字节 key 引用计数，close 到 0 才真正 CloseIpcHandle。
struct OpenedPeer {
  void *base = nullptr;
  int refcount = 0;
};

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

  void destroy() {
    std::lock_guard<std::mutex> lock(mtx);
    for (auto e : events) {
      if (e) zeEventDestroy(e);
    }
    events.clear();
    in_use.clear();
    if (pool) { zeEventPoolDestroy(pool); pool = nullptr; }
  }
};

// ---------------------------------------------------------------------------
// 全局 Registry：故意泄漏的单例。
//
// 之前所有状态都是文件级 static，析构顺序与 PyTorch XPU runtime / SYCL runtime /
// Python 解释器卸载的顺序不确定，进程退出时容易 segfault。这里用 new 出来且永不
// delete 的单例彻底规避静态析构，资源回收改为显式 shutdown()（幂等）。
// ---------------------------------------------------------------------------
struct Registry {
  std::mutex mtx;
  bool ze_inited = false;
  bool shut_down = false;
  std::unordered_map<int, std::shared_ptr<L0Env>> env_by_device;
  std::unordered_map<int, std::shared_ptr<EventPool>> evpool_by_device;
  std::unordered_map<std::string, ExportedHandle> exported;   // key: handle bytes
  std::unordered_map<std::string, OpenedPeer> opened_peers;   // key: handle bytes
  std::atomic<int64_t> inflight{0};
};

Registry &registry() {
  static Registry *g = new Registry();  // 故意不释放
  return *g;
}

std::shared_ptr<L0Env> env_for_device(int dev_index, uintptr_t queue_addr) {
  auto &R = registry();
  std::lock_guard<std::mutex> lock(R.mtx);
  TORCH_CHECK(!R.shut_down, "sycl_tla_ipc_p2p has already been shut down");
  auto it = R.env_by_device.find(dev_index);
  if (it != R.env_by_device.end()) {
    // 同一 device 上换了另一个 sycl::queue 意味着可能换了 L0 context，
    // 之前是静默走错 context，这里直接报错。
    TORCH_CHECK(it->second->queue_addr == queue_addr,
                "device ", dev_index,
                " was initialized with a different sycl queue (0x",
                std::hex, it->second->queue_addr, " vs 0x", queue_addr,
                "); all IPC calls for a device must use one stream");
    return it->second;
  }
  if (!R.ze_inited) {
    zeCheck(zeInit(0), "zeInit");
    R.ze_inited = true;
  }
  auto env = std::make_shared<L0Env>();
  env->device_index = dev_index;
  env->queue_addr = queue_addr;
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
  R.env_by_device[dev_index] = env;
  return env;
}

std::shared_ptr<L0Env> env_for_tensor(const at::Tensor &t, uintptr_t queue_addr) {
  TORCH_CHECK(t.device().type() == c10::DeviceType::XPU,
              "tensor must be an XPU tensor");
  TORCH_CHECK(t.device().has_index(), "tensor must have a device index");
  return env_for_device(t.device().index(), queue_addr);
}

std::shared_ptr<EventPool> evpool_for_env(const std::shared_ptr<L0Env> &env) {
  auto &R = registry();
  std::lock_guard<std::mutex> lock(R.mtx);
  auto it = R.evpool_by_device.find(env->device_index);
  if (it != R.evpool_by_device.end()) return it->second;
  auto ep = std::make_shared<EventPool>();
  ep->init(env->context, env->device, 256);
  R.evpool_by_device[env->device_index] = ep;
  return ep;
}

} // namespace

// ---------------------------------------------------------------------------
// PendingCopy：改成 pybind11 类，析构自动回收 event slot。
//
// 旧版是 `new PendingCopy` + 裸 uintptr_t，只有 ipc_wait 才 delete；任何异常路径
// 或忘记 wait 都会永久占用 slot，最终 "event pool exhausted"。
// ---------------------------------------------------------------------------
class PendingCopy {
public:
  PendingCopy(std::shared_ptr<L0Env> env, std::shared_ptr<EventPool> pool)
      : env_(std::move(env)), pool_(std::move(pool)) {
    event_ = pool_->acquire(slot_);
    registry().inflight.fetch_add(1);
  }
  ~PendingCopy() { release_(); }

  PendingCopy(const PendingCopy &) = delete;
  PendingCopy &operator=(const PendingCopy &) = delete;

  ze_event_handle_t event() const { return event_; }

  void wait() {
    if (done_) return;
    zeCheck(zeEventHostSynchronize(event_, UINT64_MAX),
            "zeEventHostSynchronize");
    done_ = true;
    release_();
  }

  bool query() {
    if (done_) return true;
    if (zeEventQueryStatus(event_) == ZE_RESULT_SUCCESS) {
      done_ = true;
      release_();
      return true;
    }
    return false;
  }

private:
  void release_() {
    if (released_) return;
    released_ = true;
    if (!done_) {
      // 未完成就析构：必须先等，否则 copy engine 还在写目标 buffer。
      zeEventHostSynchronize(event_, UINT64_MAX);
      done_ = true;
    }
    if (pool_) pool_->release(slot_);
    registry().inflight.fetch_sub(1);
  }

  std::shared_ptr<L0Env> env_;
  std::shared_ptr<EventPool> pool_;
  ze_event_handle_t event_ = nullptr;
  uint32_t slot_ = 0;
  bool done_ = false;
  bool released_ = false;
};

// ---------------------------------------------------------------------------
// IpcArena：一块由本模块自己拥有、进程生命周期内绝不释放的裸 L0 显存。
//
// 关键点：PyTorch caching allocator 的 slab 会被回收/复用，对它导出的 IPC handle
// 天然会变野指针（对端 free 之后本端映射仍在；slab 被复用后 handle 字节相同但内容
// 已换）。vllm-omni 每层新建 k/v tensor 时这个洞必踩。改为：所有 ring buffer 都是
// 这块 arena 的 view，整个进程只导出 1 个 handle、只 open 1 次。
// ---------------------------------------------------------------------------
class IpcArena {
public:
  IpcArena(int device_index, int64_t nbytes, uintptr_t queue_ptr)
      : env_(env_for_device(device_index, queue_ptr)),
        device_index_(device_index),
        nbytes_(static_cast<size_t>(nbytes)) {
    TORCH_CHECK(nbytes > 0, "arena nbytes must be positive");
    ze_device_mem_alloc_desc_t d = {};
    d.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    d.ordinal = 0;
    zeCheck(zeMemAllocDevice(env_->context, &d, nbytes_, 4096, env_->device,
                             &base_),
            "zeMemAllocDevice(arena)");
    zeCheck(zeMemGetIpcHandle(env_->context, base_, &handle_),
            "zeMemGetIpcHandle(arena)");
    handle_bytes_.assign(reinterpret_cast<const char *>(handle_.data),
                         sizeof(handle_.data));
  }

  // 不提供析构释放：arena 与导出的 handle 必须活到进程结束，交由 shutdown() 处理。
  ~IpcArena() = default;

  py::bytes export_handle() const { return py::bytes(handle_bytes_); }
  uintptr_t base_ptr() const { return reinterpret_cast<uintptr_t>(base_); }
  int64_t nbytes() const { return static_cast<int64_t>(nbytes_); }
  int device_index() const { return device_index_; }

  // 把 arena 的一段包成不拥有内存的 torch tensor。
  at::Tensor view(int64_t offset, std::vector<int64_t> shape,
                  at::ScalarType dtype) const {
    int64_t n = 1;
    for (auto s : shape) n *= s;
    const int64_t need = n * static_cast<int64_t>(c10::elementSize(dtype));
    TORCH_CHECK(offset >= 0 && offset + need <= static_cast<int64_t>(nbytes_),
                "view [", offset, ", +", need, ") out of arena of size ",
                nbytes_);
    TORCH_CHECK(offset % 256 == 0, "arena view offset must be 256B aligned");
    void *p = static_cast<char *>(base_) + offset;
    auto opts = at::TensorOptions()
                    .dtype(dtype)
                    .device(at::kXPU, static_cast<c10::DeviceIndex>(device_index_));
    return at::from_blob(p, shape, [](void *) {}, opts);
  }

  void destroy() {
    if (base_ == nullptr) return;
#ifndef SYCL_TLA_NO_ZE_MEM_PUT_IPC
    zeMemPutIpcHandle(env_->context, handle_);
#endif
    zeMemFree(env_->context, base_);
    base_ = nullptr;
  }

private:
  std::shared_ptr<L0Env> env_;
  int device_index_;
  size_t nbytes_;
  void *base_ = nullptr;
  ze_ipc_mem_handle_t handle_{};
  std::string handle_bytes_;
};

// arena 由 Registry 持有一份强引用，保证 Python 端对象被 GC 也不会提前释放显存。
std::mutex g_arena_mutex;
std::vector<std::shared_ptr<IpcArena>> g_arenas;

std::shared_ptr<IpcArena> make_arena(int device_index, int64_t nbytes,
                                     uintptr_t queue_ptr) {
  auto a = std::make_shared<IpcArena>(device_index, nbytes, queue_ptr);
  std::lock_guard<std::mutex> lock(g_arena_mutex);
  g_arenas.push_back(a);
  return a;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Export the IPC handle of a tensor's storage, together with the tensor's byte
// offset inside its L0 allocation.
//
// zeMemGetIpcHandle/zeMemOpenIpcHandle operate on whole allocations: the peer
// always receives the *base* address of the slab, never the address we passed
// in. PyTorch's caching allocator sub-allocates, so several ring buffers can
// share one slab and thus one identical handle blob. Without the offset the
// receiver reads from the slab head instead of the tensor.
std::pair<py::bytes, int64_t> ipc_get_handle(const at::Tensor &t,
                                             uintptr_t queue_ptr) {
  TORCH_CHECK(t.is_contiguous(), "tensor must be contiguous to export IPC");
  TORCH_WARN_ONCE(
      "ipc_get_handle() exports a caching-allocator slab; the handle becomes a "
      "dangling reference once the tensor is freed or its slab is reused. "
      "Prefer IpcArena for anything with a dynamic tensor lifetime.");
  auto env = env_for_tensor(t, queue_ptr);

  void *ptr = t.data_ptr();
  void *base = nullptr;
  size_t alloc_size = 0;
  zeCheck(zeMemGetAddressRange(env->context, ptr, &base, &alloc_size),
          "zeMemGetAddressRange");

  const int64_t offset =
      static_cast<int64_t>(reinterpret_cast<uintptr_t>(ptr) -
                           reinterpret_cast<uintptr_t>(base));
  TORCH_CHECK(offset >= 0 &&
                  static_cast<size_t>(offset) + t.nbytes() <= alloc_size,
              "tensor [", offset, ", +", t.nbytes(),
              ") does not lie inside its L0 allocation of size ", alloc_size);

  ze_ipc_mem_handle_t handle = {};
  zeCheck(zeMemGetIpcHandle(env->context, base, &handle),
          "zeMemGetIpcHandle");

  std::string key(reinterpret_cast<const char *>(handle.data),
                  sizeof(handle.data));
  {
    // 登记以便 shutdown() 时 Put 回去；同一 slab 多次导出只保留一份。
    auto &R = registry();
    std::lock_guard<std::mutex> lock(R.mtx);
    auto it = R.exported.find(key);
    if (it == R.exported.end()) {
      ExportedHandle e;
      e.handle = handle;
      e.base = base;
      e.refcount = 1;
      R.exported.emplace(key, e);
    } else {
      it->second.refcount += 1;
#ifndef SYCL_TLA_NO_ZE_MEM_PUT_IPC
      zeMemPutIpcHandle(env->context, handle);  // 立刻归还重复的 fd
#endif
    }
  }
  return {py::bytes(key), offset};
}

// 按 handle 字节引用计数打开对端 slab；返回 slab 基址（offset 由调用方自己加）。
uintptr_t ipc_open_peer(int device_index, const py::bytes &handle_bytes,
                        uintptr_t queue_ptr) {
  auto env = env_for_device(device_index, queue_ptr);
  std::string key = handle_bytes;
  auto &R = registry();
  std::lock_guard<std::mutex> lock(R.mtx);
  auto it = R.opened_peers.find(key);
  if (it != R.opened_peers.end()) {
    it->second.refcount += 1;
    return reinterpret_cast<uintptr_t>(it->second.base);
  }
  ze_ipc_mem_handle_t handle = {};
  TORCH_CHECK(key.size() == sizeof(handle.data),
              "bad IPC handle size: ", key.size());
  std::memcpy(handle.data, key.data(), sizeof(handle.data));
  void *peer_base = nullptr;
  // UNCACHED: 缓存过的远端行会在对端下一轮覆写后返回陈旧数据。
  zeCheck(zeMemOpenIpcHandle(env->context, env->device, handle,
                             ZE_IPC_MEMORY_FLAG_BIAS_UNCACHED, &peer_base),
          "zeMemOpenIpcHandle");
  OpenedPeer p;
  p.base = peer_base;
  p.refcount = 1;
  R.opened_peers.emplace(key, p);
  return reinterpret_cast<uintptr_t>(peer_base);
}

void ipc_close_peer(int device_index, const py::bytes &handle_bytes,
                    uintptr_t queue_ptr) {
  auto env = env_for_device(device_index, queue_ptr);
  std::string key = handle_bytes;
  void *base = nullptr;
  {
    auto &R = registry();
    std::lock_guard<std::mutex> lock(R.mtx);
    auto it = R.opened_peers.find(key);
    if (it == R.opened_peers.end()) return;   // 幂等
    if (--it->second.refcount > 0) return;
    base = it->second.base;
    R.opened_peers.erase(it);
  }
  TORCH_CHECK(registry().inflight.load() == 0,
              "cannot close peer mapping while copies are still in flight");
  zeCheck(zeMemCloseIpcHandle(env->context, base), "zeMemCloseIpcHandle");
}

// 兼容旧签名：内部转成 open_peer + offset。
uintptr_t ipc_open_handle(const at::Tensor &like, const py::bytes &handle_bytes,
                          int64_t offset, uintptr_t queue_ptr) {
  return ipc_open_peer(like.device().index(), handle_bytes, queue_ptr) +
         static_cast<uintptr_t>(offset);
}

// 裸指针版：dst 也可能是 arena 里的地址，不必是 tensor。
std::shared_ptr<PendingCopy> ipc_copy_async(int device_index, uintptr_t dst,
                                            uintptr_t src, int64_t nbytes,
                                            uintptr_t queue_ptr) {
  TORCH_CHECK(nbytes > 0, "nbytes must be positive");
  TORCH_CHECK(dst != 0 && src != 0, "null src/dst pointer");
  auto env = env_for_device(device_index, queue_ptr);
  auto pool = evpool_for_env(env);
  auto pending = std::make_shared<PendingCopy>(env, pool);
  zeCheck(zeCommandListAppendMemoryCopy(
              env->copy_cmdlist, reinterpret_cast<void *>(dst),
              reinterpret_cast<const void *>(src),
              static_cast<size_t>(nbytes), pending->event(), 0, nullptr),
          "zeCommandListAppendMemoryCopy");
  return pending;
}

std::shared_ptr<PendingCopy> ipc_copy_from_peer_async(at::Tensor &dst_local,
                                                      uintptr_t peer_src,
                                                      int64_t nbytes,
                                                      uintptr_t queue_ptr) {
  TORCH_CHECK(dst_local.is_contiguous(), "dst_local must be contiguous");
  TORCH_CHECK(static_cast<size_t>(nbytes) <= dst_local.nbytes(),
              "nbytes (", nbytes, ") exceeds dst tensor size (",
              dst_local.nbytes(), ")");
  return ipc_copy_async(dst_local.device().index(),
                        reinterpret_cast<uintptr_t>(dst_local.data_ptr()),
                        peer_src, nbytes, queue_ptr);
}

// 显式关停：必须在 dist.destroy_process_group() 之前、解释器退出之前调用。
// 幂等，可以由 atexit 再兜一次。
void ipc_shutdown() {
  auto &R = registry();
  {
    std::lock_guard<std::mutex> lock(R.mtx);
    if (R.shut_down) return;
    R.shut_down = true;
  }

  std::vector<std::shared_ptr<IpcArena>> arenas;
  {
    std::lock_guard<std::mutex> lock(g_arena_mutex);
    arenas.swap(g_arenas);
  }

  std::lock_guard<std::mutex> lock(R.mtx);
  for (auto &kv : R.opened_peers) {
    auto env_it = R.env_by_device.begin();
    if (env_it != R.env_by_device.end() && kv.second.base) {
      zeMemCloseIpcHandle(env_it->second->context, kv.second.base);
    }
  }
  R.opened_peers.clear();

#ifndef SYCL_TLA_NO_ZE_MEM_PUT_IPC
  for (auto &kv : R.exported) {
    auto env_it = R.env_by_device.begin();
    if (env_it != R.env_by_device.end()) {
      zeMemPutIpcHandle(env_it->second->context, kv.second.handle);
    }
  }
#endif
  R.exported.clear();

  for (auto &a : arenas) a->destroy();

  for (auto &kv : R.evpool_by_device) kv.second->destroy();
  R.evpool_by_device.clear();

  for (auto &kv : R.env_by_device) {
    if (kv.second->copy_cmdlist) {
      zeCommandListDestroy(kv.second->copy_cmdlist);
      kv.second->copy_cmdlist = nullptr;
    }
  }
  R.env_by_device.clear();
}

PYBIND11_MODULE(sycl_tla_ipc_p2p, m) {
  m.doc() = "XPU Level-Zero IPC-handle P2P copy engine transfers";

  py::class_<RingControl, std::shared_ptr<RingControl>>(m, "RingControl")
      .def(py::init<std::string, int, int, int, int, int64_t>(),
           py::arg("name"),
           py::arg("rank"),
           py::arg("world"),
           py::arg("slots"),
           py::arg("handle_bytes") = 64,
           py::arg("timeout_ms") = 30000)
      .def("publish_handle", &RingControl::publish_handle,
           py::arg("handle"))
      .def("wait_handle", &RingControl::wait_handle,
           py::arg("peer_rank"))
      .def("publish_ready", &RingControl::publish_ready,
           py::arg("receiver_rank"), py::arg("slot"), py::arg("epoch"))
      .def("wait_ready", &RingControl::wait_ready,
           py::arg("receiver_rank"), py::arg("slot"), py::arg("epoch"),
           py::call_guard<py::gil_scoped_release>())
      .def("publish_free", &RingControl::publish_free,
           py::arg("owner_rank"), py::arg("slot"), py::arg("epoch"))
      .def("wait_free", &RingControl::wait_free,
           py::arg("owner_rank"), py::arg("slot"), py::arg("epoch"),
           py::call_guard<py::gil_scoped_release>())
      .def("barrier", &RingControl::barrier,
           py::arg("epoch"),
           py::call_guard<py::gil_scoped_release>())
      .def("publish_work_done", &RingControl::publish_work_done,
           py::arg("rank"))
      .def("wait_work_done", &RingControl::wait_work_done,
           py::arg("rank"),
           py::call_guard<py::gil_scoped_release>())
      .def("publish_mapping_closed", &RingControl::publish_mapping_closed,
           py::arg("rank"))
      .def("wait_mapping_closed", &RingControl::wait_mapping_closed,
           py::arg("rank"),
           py::call_guard<py::gil_scoped_release>())
      .def("publish_elapsed_ns", &RingControl::publish_elapsed_ns,
           py::arg("rank"), py::arg("elapsed_ns"), py::arg("epoch"))
      .def("wait_max_elapsed_ns", &RingControl::wait_max_elapsed_ns,
           py::arg("epoch"),
           py::call_guard<py::gil_scoped_release>())
      .def("close", &RingControl::close,
           py::arg("unlink_name") = false);

  py::class_<PendingCopy, std::shared_ptr<PendingCopy>>(m, "PendingCopy")
      .def("wait", &PendingCopy::wait,
           py::call_guard<py::gil_scoped_release>())
      .def("query", &PendingCopy::query);

  py::class_<IpcArena, std::shared_ptr<IpcArena>>(m, "IpcArena")
      .def("export_handle", &IpcArena::export_handle)
      .def("view", &IpcArena::view, py::arg("offset"), py::arg("shape"),
           py::arg("dtype"))
      .def("base_ptr", &IpcArena::base_ptr)
      .def("nbytes", &IpcArena::nbytes)
      .def("device_index", &IpcArena::device_index);

  m.def("make_arena", &make_arena, py::arg("device_index"), py::arg("nbytes"),
        py::arg("queue_ptr"));
  m.def("open_peer", &ipc_open_peer, py::arg("device_index"),
        py::arg("handle_bytes"), py::arg("queue_ptr"));
  m.def("close_peer", &ipc_close_peer, py::arg("device_index"),
        py::arg("handle_bytes"), py::arg("queue_ptr"));
  m.def("copy_async", &ipc_copy_async, py::arg("device_index"), py::arg("dst"),
        py::arg("src"), py::arg("nbytes"), py::arg("queue_ptr"));
  m.def("shutdown", &ipc_shutdown);

  // ---- 兼容旧接口 ----
  m.def("ipc_get_handle", &ipc_get_handle,
        py::arg("tensor"), py::arg("queue_ptr"));
  m.def("ipc_open_handle", &ipc_open_handle,
        py::arg("like_tensor"), py::arg("handle_bytes"), py::arg("offset"),
        py::arg("queue_ptr"));
  m.def("ipc_copy_from_peer_async", &ipc_copy_from_peer_async,
        py::arg("dst_local"), py::arg("peer_src"), py::arg("nbytes"),
        py::arg("queue_ptr"));
}

