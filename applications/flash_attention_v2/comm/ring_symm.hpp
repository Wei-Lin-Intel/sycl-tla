#pragma once

// Ring-Attention symmetric memory manager.
//
// Each rank owns:
//   - k_buf[2], v_buf[2] : double-buffered receive/compute regions for the
//     ring rotation. All four are IPC-exportable sycl::malloc_device blocks.
//   - a signal pad for cheap inter-round barriers.
//
// Layout of each k_buf[b] / v_buf[b] is packed [S_local, H_kv, D] bf16,
// identical to the BSHD (per-batch) tile that the FMHA kernel consumes.
//
// NOTE: This class does NOT call MPI_Init. The caller (e.g. mpi4py on the
// Python side, or the host driver) must initialize MPI before constructing it.

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <mpi.h>

#include "symm.hpp"   // exchange_ipc_ptrs / close_ipc_ptrs / ze helpers (same dir)

class RingSymmMemory {
 public:
  // batch * seq_len_kv_local is the number of KV "rows" this rank holds.
  // We treat the KV region as a flat [rows, h_kv, d] block. For a single
  // batch (typical seq-parallel image/video case) batch == 1.
  RingSymmMemory(int batch,
                 int seq_kv_local,   // per-rank KV sequence length (S / world_size)
                 int h_kv,
                 int d_qk,
                 int d_vo,
                 int rank,
                 int world_size,
                 sycl::queue& q)
      : batch_(batch),
        seq_kv_local_(seq_kv_local),
        h_kv_(h_kv),
        d_qk_(d_qk),
        d_vo_(d_vo),
        rank_(rank),
        world_size_(world_size),
        q_(q) {
    k_elems_ = static_cast<size_t>(batch_) * seq_kv_local_ * h_kv_ * d_qk_;
    v_elems_ = static_cast<size_t>(batch_) * seq_kv_local_ * h_kv_ * d_vo_;

    for (int b = 0; b < 2; ++b) {
      k_buf_[b] = sycl::malloc_device<uint16_t>(k_elems_, q_);
      v_buf_[b] = sycl::malloc_device<uint16_t>(v_elems_, q_);
      if (!k_buf_[b] || !v_buf_[b]) {
        throw std::runtime_error("RingSymmMemory: malloc_device failed");
      }
      q_.memset(k_buf_[b], 0, k_elems_ * sizeof(uint16_t)).wait();
      q_.memset(v_buf_[b], 0, v_elems_ * sizeof(uint16_t)).wait();
    }

    // Signal pad: world_size uint32 per rank is enough for a barrier.
    signal_elems_ = static_cast<size_t>(world_size_) * 2;
    local_signal_ = sycl::malloc_device<uint32_t>(signal_elems_, q_);
    q_.memset(local_signal_, 0, signal_elems_ * sizeof(uint32_t)).wait();

    // Exchange IPC pointers for every buffer.
    for (int b = 0; b < 2; ++b) {
      remote_k_[b] = exchange_ipc_ptrs(k_buf_[b], rank_, world_size_, q_, opened_bases_);
      remote_v_[b] = exchange_ipc_ptrs(v_buf_[b], rank_, world_size_, q_, opened_bases_);

      remote_k_dev_[b] = sycl::malloc_device<void*>(world_size_, q_);
      remote_v_dev_[b] = sycl::malloc_device<void*>(world_size_, q_);
      q_.memcpy(remote_k_dev_[b], remote_k_[b].data(), world_size_ * sizeof(void*)).wait();
      q_.memcpy(remote_v_dev_[b], remote_v_[b].data(), world_size_ * sizeof(void*)).wait();
    }

    remote_signal_ = exchange_ipc_ptrs(local_signal_, rank_, world_size_, q_, opened_bases_);
    remote_signal_dev_ = sycl::malloc_device<uint32_t*>(world_size_, q_);
    std::vector<uint32_t*> pads(world_size_);
    for (int i = 0; i < world_size_; ++i)
      pads[i] = reinterpret_cast<uint32_t*>(remote_signal_[i]);
    q_.memcpy(remote_signal_dev_, pads.data(), world_size_ * sizeof(uint32_t*)).wait();

    make_resident();
    MPI_Barrier(MPI_COMM_WORLD);
  }

  ~RingSymmMemory() {
    close_ipc_ptrs(q_, opened_bases_);
    for (int b = 0; b < 2; ++b) {
      if (k_buf_[b]) sycl::free(k_buf_[b], q_);
      if (v_buf_[b]) sycl::free(v_buf_[b], q_);
      if (remote_k_dev_[b]) sycl::free(remote_k_dev_[b], q_);
      if (remote_v_dev_[b]) sycl::free(remote_v_dev_[b], q_);
    }
    if (local_signal_) sycl::free(local_signal_, q_);
    if (remote_signal_dev_) sycl::free(remote_signal_dev_, q_);
  }

  RingSymmMemory(RingSymmMemory const&) = delete;
  RingSymmMemory& operator=(RingSymmMemory const&) = delete;

  // ---- one-time deep copy of this rank's own K/V into compute buffer 0 ----
  // src_k / src_v are device pointers to the caller's (torch) bf16 tensors,
  // already laid out as packed [rows, h_kv, d].
  void load_local_kv(const void* src_k, const void* src_v) {
    q_.memcpy(k_buf_[0], src_k, k_elems_ * sizeof(uint16_t)).wait();
    q_.memcpy(v_buf_[0], src_v, v_elems_ * sizeof(uint16_t)).wait();
  }

  // ---- accessors used by the host ring driver ----
  uint16_t* local_k(int b) { return k_buf_[b]; }
  uint16_t* local_v(int b) { return v_buf_[b]; }

  // device array<void*>[world_size] holding peers' k_buf[b] / v_buf[b] bases.
  void** remote_k_dev(int b) { return remote_k_dev_[b]; }
  void** remote_v_dev(int b) { return remote_v_dev_[b]; }

  // host-side single peer pointer (convenient for filling mainloop params).
  void* remote_k(int peer, int b) { return remote_k_[b][peer]; }
  void* remote_v(int peer, int b) { return remote_v_[b][peer]; }

  uint32_t** remote_signal_dev() { return remote_signal_dev_; }

  int seq_kv_local() const { return seq_kv_local_; }
  int k_row_stride() const { return h_kv_ * d_qk_; }  // packed row stride (elems)
  int v_row_stride() const { return h_kv_ * d_vo_; }

  // Lightweight inter-round barrier (put/wait on signal pads).
  sycl::event barrier(int channel) {
    int rank = rank_, world = world_size_;
    uint32_t** pads = remote_signal_dev_;
    return q_.submit([&](sycl::handler& h) {
      h.parallel_for(sycl::nd_range<1>(std::max(32, world), std::max(32, world)),
        [=](sycl::nd_item<1> it) {
          auto tid = it.get_local_id(0);
          if (tid < (size_t)world && (int)tid != rank) {
            int target = (int)tid;
            try_put_signal_device(pads[target] + world * channel + rank, 0);
            try_wait_signal_device(pads[rank] + world * channel + target, 0);
          }
        });
    });
  }

 private:
  void make_resident() {
    auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_.get_context());
    auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_.get_device());
    auto resident = [&](void* p, size_t bytes) {
      if (p) ZE_CHECK(zeContextMakeMemoryResident(ze_ctx, ze_dev, p, bytes));
    };
    for (int b = 0; b < 2; ++b) {
      for (int peer = 0; peer < world_size_; ++peer) {
        if (peer == rank_) continue;
        resident(remote_k_[b][peer], k_elems_ * sizeof(uint16_t));
        resident(remote_v_[b][peer], v_elems_ * sizeof(uint16_t));
      }
    }
    for (int peer = 0; peer < world_size_; ++peer) {
      if (peer == rank_) continue;
      resident(remote_signal_[peer], signal_elems_ * sizeof(uint32_t));
    }
  }

  int batch_, seq_kv_local_, h_kv_, d_qk_, d_vo_, rank_, world_size_;
  sycl::queue& q_;
  size_t k_elems_ = 0, v_elems_ = 0, signal_elems_ = 0;

  uint16_t* k_buf_[2] = {nullptr, nullptr};
  uint16_t* v_buf_[2] = {nullptr, nullptr};
  uint32_t* local_signal_ = nullptr;

  std::vector<void*> remote_k_[2];
  std::vector<void*> remote_v_[2];
  std::vector<void*> remote_signal_;
  void** remote_k_dev_[2] = {nullptr, nullptr};
  void** remote_v_dev_[2] = {nullptr, nullptr};
  uint32_t** remote_signal_dev_ = nullptr;

  std::vector<void*> opened_bases_;
};
