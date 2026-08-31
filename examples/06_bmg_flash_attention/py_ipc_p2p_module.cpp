// Cross-process IPC-handle based P2P copy for XPU tensors, using experimental
// SYCL IPC memory and an independent SYCL memcpy queue. This bypasses
// torch.distributed for K/V data movement in ring attention so transfer can
// overlap the attention compute kernel.
//
// Route:
//   owner   : sycl::aligned_alloc_device -> device USM arena
//   owner   : ipc::memory::get(arena, context) -> IPC handle bytes
//   control : exchange handle bytes through POSIX shared memory
//   sender  : ipc::memory::open(handle, context, device) -> peer USM pointer
//   round   : independent_queue.memcpy(peer_dst, local_src, bytes)
//   teardown: ipc::memory::close / ipc::memory::put / sycl::free
//
// POSIX shared memory and futexes remain the control plane for handle exchange,
// ready/free tickets, barriers, elapsed-time collection, and teardown.

#include <torch/extension.h>
#include <pybind11/stl.h>

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>
#include <sycl/usm.hpp>

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

#ifndef SYCL_EXT_ONEAPI_INTER_PROCESS_COMMUNICATION
#error "sycl_tla_ipc_p2p requires sycl_ext_oneapi_inter_process_communication"
#endif

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
// SYCL IPC and copy-queue environment.
//
// The context/device are taken from the exact SYCL queue backing the current
// PyTorch XPU stream. The memcpy queue is deliberately separate so waiting on
// or submitting work to it does not serialize the PyTorch compute queue.
//
// The SYCL API does not promise a particular physical engine. On Intel GPU
// backends, a pure USM memcpy may be routed to a copy engine, but that remains
// an implementation detail and should be verified with profiling.
// ---------------------------------------------------------------------------
struct SyclIpcEnv {
  SyclIpcEnv(int index, uintptr_t address, const sycl::context &ctx,
             const sycl::device &dev)
      : context(ctx),
        device(dev),
        copy_queue(
            context, device,
            sycl::property_list{sycl::property::queue::in_order{}}),
        queue_addr(address),
        device_index(index) {}

  sycl::context context;
  sycl::device device;
  sycl::queue copy_queue;
  uintptr_t queue_addr = 0;
  int device_index = -1;
};

struct OpenedPeer {
  void *base = nullptr;
  int refcount = 0;
  std::shared_ptr<SyclIpcEnv> env;
};

std::shared_ptr<SyclIpcEnv>
make_sycl_ipc_env(int device_index, uintptr_t queue_addr) {
  TORCH_CHECK(queue_addr != 0, "null sycl_queue pointer from torch");
  auto *torch_queue = reinterpret_cast<sycl::queue *>(queue_addr);
  auto env = std::make_shared<SyclIpcEnv>(
      device_index, queue_addr, torch_queue->get_context(),
      torch_queue->get_device());

  TORCH_CHECK(
      env->device.has(sycl::aspect::ext_oneapi_ipc_memory),
      "XPU device ", device_index,
      " does not support sycl::aspect::ext_oneapi_ipc_memory");

  return env;
}

// ---------------------------------------------------------------------------
// Global registry: deliberately leaked.
//
// The destruction order of the Python interpreter, PyTorch XPU runtime and
// SYCL runtime is not defined. Resources are released by explicit shutdown();
// the registry object itself is intentionally never destroyed.
// ---------------------------------------------------------------------------
struct Registry {
  std::mutex mtx;
  bool shut_down = false;
  std::unordered_map<int, std::shared_ptr<SyclIpcEnv>> env_by_device;
  std::unordered_map<std::string, OpenedPeer> opened_peers;
  std::atomic<int64_t> inflight{0};
};

Registry &registry() {
  static Registry *g = new Registry();
  return *g;
}

std::shared_ptr<SyclIpcEnv>
env_for_device(int dev_index, uintptr_t queue_addr) {
  auto &R = registry();
  std::lock_guard<std::mutex> lock(R.mtx);
  TORCH_CHECK(!R.shut_down, "sycl_tla_ipc_p2p has already been shut down");

  auto it = R.env_by_device.find(dev_index);
  if (it != R.env_by_device.end()) {
    TORCH_CHECK(
        it->second->queue_addr == queue_addr,
        "device ", dev_index,
        " was initialized with a different sycl queue (0x",
        std::hex, it->second->queue_addr, " vs 0x", queue_addr,
        "); all IPC calls for a device must use one stream");
    return it->second;
  }
  auto env = make_sycl_ipc_env(dev_index, queue_addr);
  R.env_by_device.emplace(dev_index, env);
  return env;
}

} // namespace

// ---------------------------------------------------------------------------
// A PendingCopy owns the SYCL event returned by queue::memcpy.
// ---------------------------------------------------------------------------
class PendingCopy {
public:
  PendingCopy(std::shared_ptr<SyclIpcEnv> env, void *dst, const void *src,
              size_t nbytes)
      : env_(std::move(env)),
        event_(env_->copy_queue.memcpy(dst, src, nbytes)) {
    registry().inflight.fetch_add(1);
  }

  ~PendingCopy() { release_(); }

  PendingCopy(const PendingCopy &) = delete;
  PendingCopy &operator=(const PendingCopy &) = delete;

  void wait() {
    if (done_) return;
    event_.wait_and_throw();
    done_ = true;
    release_();
  }

  bool query() {
    if (done_) return true;

    const auto status =
        event_.get_info<sycl::info::event::command_execution_status>();
    if (status == sycl::info::event_command_status::complete) {
      // Surface asynchronous failures before reporting success.
      event_.wait_and_throw();
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
      // A copy may not outlive either USM allocation. Explicit wait()
      // propagates errors; a destructor cannot.
      try {
        event_.wait_and_throw();
      } catch (...) {
      }
      done_ = true;
    }

    registry().inflight.fetch_sub(1);
  }

  std::shared_ptr<SyclIpcEnv> env_;
  sycl::event event_;
  bool done_ = false;
  bool released_ = false;
};

// ---------------------------------------------------------------------------
// Process-lifetime device-USM arena exported through the SYCL IPC API.
//
// The receive buffers are views into one explicitly owned allocation rather
// than PyTorch caching-allocator suballocations. This gives the IPC allocation
// a stable lifetime and avoids exporting unrelated bytes from an allocator
// slab. Each process exports one arena and each predecessor opens it once.
// ---------------------------------------------------------------------------
class IpcArena {
public:
  IpcArena(int device_index, int64_t nbytes, uintptr_t queue_ptr)
      : env_(env_for_device(device_index, queue_ptr)),
        device_index_(device_index),
        nbytes_(static_cast<size_t>(nbytes)) {
    TORCH_CHECK(nbytes > 0, "arena nbytes must be positive");

    base_ = sycl::aligned_alloc_device(
        4096, nbytes_, env_->device, env_->context);
    TORCH_CHECK(base_ != nullptr,
                "sycl::aligned_alloc_device failed for ", nbytes_, " bytes");

    try {
      handle_ =
          std::make_unique<sycl::ext::oneapi::experimental::ipc::handle>(
              sycl::ext::oneapi::experimental::ipc::memory::get(
                  base_, env_->context));

      const auto data = handle_->data();
      TORCH_CHECK(!data.empty(), "SYCL IPC memory returned an empty handle");
      handle_bytes_.assign(
          reinterpret_cast<const char *>(data.data()), data.size());
    } catch (...) {
      sycl::free(base_, env_->context);
      base_ = nullptr;
      throw;
    }
  }

  // Explicit shutdown owns destruction order relative to peer mappings.
  ~IpcArena() = default;

  py::bytes export_handle() const { return py::bytes(handle_bytes_); }
  uintptr_t base_ptr() const { return reinterpret_cast<uintptr_t>(base_); }
  int64_t nbytes() const { return static_cast<int64_t>(nbytes_); }
  int device_index() const { return device_index_; }

  // Wrap a range of device USM in a non-owning XPU tensor.
  at::Tensor view(int64_t offset, std::vector<int64_t> shape,
                  at::ScalarType dtype) const {
    int64_t n = 1;
    for (auto s : shape) {
      TORCH_CHECK(s >= 0, "arena view dimensions must be non-negative");
      n *= s;
    }

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

    if (handle_) {
      sycl::ext::oneapi::experimental::ipc::memory::put(
          *handle_, env_->context);
      handle_.reset();
    }

    sycl::free(base_, env_->context);
    base_ = nullptr;
    handle_bytes_.clear();
  }

private:
  std::shared_ptr<SyclIpcEnv> env_;
  int device_index_;
  size_t nbytes_;
  void *base_ = nullptr;
  std::unique_ptr<sycl::ext::oneapi::experimental::ipc::handle> handle_;
  std::string handle_bytes_;
};

// Keep arenas alive until explicit shutdown even if Python drops its object.
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

uintptr_t ipc_open_peer(int device_index, const py::bytes &handle_bytes,
                        uintptr_t queue_ptr) {
  auto env = env_for_device(device_index, queue_ptr);
  std::string key = handle_bytes;

  TORCH_CHECK(!key.empty(), "IPC handle must not be empty");

  auto &R = registry();
  std::lock_guard<std::mutex> lock(R.mtx);
  auto it = R.opened_peers.find(key);
  if (it != R.opened_peers.end()) {
    TORCH_CHECK(
        it->second.env == env,
        "the same IPC handle was opened with a different SYCL environment");
    it->second.refcount += 1;
    return reinterpret_cast<uintptr_t>(it->second.base);
  }

  sycl::ext::oneapi::experimental::ipc::handle_data_t handle_data(key.size());
  std::memcpy(handle_data.data(), key.data(), key.size());

  void *peer_base =
      sycl::ext::oneapi::experimental::ipc::memory::open(
          handle_data, env->context, env->device);
  TORCH_CHECK(peer_base != nullptr, "SYCL IPC memory open returned null");

  OpenedPeer p;
  p.base = peer_base;
  p.refcount = 1;
  p.env = env;
  R.opened_peers.emplace(key, p);
  return reinterpret_cast<uintptr_t>(peer_base);
}

void ipc_close_peer(int device_index, const py::bytes &handle_bytes,
                    uintptr_t queue_ptr) {
  auto env = env_for_device(device_index, queue_ptr);
  std::string key = handle_bytes;

  void *base = nullptr;
  std::shared_ptr<SyclIpcEnv> opened_env;
  {
    auto &R = registry();
    std::lock_guard<std::mutex> lock(R.mtx);
    auto it = R.opened_peers.find(key);
    if (it == R.opened_peers.end()) return;

    TORCH_CHECK(
        it->second.env == env,
        "IPC peer mapping is being closed with a different SYCL environment");

    if (--it->second.refcount > 0) return;

    base = it->second.base;
    opened_env = it->second.env;
    R.opened_peers.erase(it);
  }

  TORCH_CHECK(registry().inflight.load() == 0,
              "cannot close peer mapping while copies are still in flight");

  sycl::ext::oneapi::experimental::ipc::memory::close(
      base, opened_env->context);
}

std::shared_ptr<PendingCopy> ipc_copy_async(int device_index, uintptr_t dst,
                                            uintptr_t src, int64_t nbytes,
                                            uintptr_t queue_ptr) {
  TORCH_CHECK(nbytes > 0, "nbytes must be positive");
  TORCH_CHECK(dst != 0 && src != 0, "null src/dst pointer");

  auto env = env_for_device(device_index, queue_ptr);
  return std::make_shared<PendingCopy>(
      env,
      reinterpret_cast<void *>(dst),
      reinterpret_cast<const void *>(src),
      static_cast<size_t>(nbytes));
}

void ipc_shutdown() {
  auto &R = registry();
  {
    std::lock_guard<std::mutex> lock(R.mtx);
    if (R.shut_down) return;
    R.shut_down = true;
  }

  TORCH_CHECK(R.inflight.load() == 0,
              "cannot shut down while IPC copies are still in flight");

  std::vector<std::shared_ptr<IpcArena>> arenas;
  {
    std::lock_guard<std::mutex> lock(g_arena_mutex);
    arenas.swap(g_arenas);
  }

  std::vector<OpenedPeer> opened_peers;
  {
    std::lock_guard<std::mutex> lock(R.mtx);
    opened_peers.reserve(R.opened_peers.size());
    for (auto &entry : R.opened_peers) {
      opened_peers.push_back(entry.second);
    }
    R.opened_peers.clear();
  }

  for (auto &peer : opened_peers) {
    if (peer.base != nullptr && peer.env) {
      sycl::ext::oneapi::experimental::ipc::memory::close(
          peer.base, peer.env->context);
    }
  }

  for (auto &arena : arenas) {
    arena->destroy();
  }

  {
    std::lock_guard<std::mutex> lock(R.mtx);
    R.env_by_device.clear();
  }
}

PYBIND11_MODULE(sycl_tla_ipc_p2p, m) {
  m.doc() = "XPU SYCL IPC-memory P2P transfers";

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
}

