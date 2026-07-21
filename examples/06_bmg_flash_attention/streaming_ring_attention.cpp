/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#include "streaming_ring_attention.hpp"

#include "xe_fmha_fwd_runner.hpp"

#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <unordered_map>

namespace {

constexpr int kSubgroupSize = 16;
constexpr int64_t kMaxCommunicationWorkGroups = 32;

struct alignas(16) Vector16 {
  uint32_t value[4];
};

struct alignas(16) Float4 {
  float value[4];
};

class RingCopyKernel;
class RingSignalWaitKernel;
class RingMergeKernel;

class CompatDeviceGuard {
public:
  explicit CompatDeviceGuard(unsigned int device)
      : previous_(compat::get_current_device_id()) {
    if (previous_ != device) {
      compat::select_device(device);
      changed_ = true;
    }
  }

  ~CompatDeviceGuard() {
    if (changed_) {
      compat::select_device(previous_);
    }
  }

private:
  unsigned int previous_;
  bool changed_ = false;
};

void check_data_tensor(const at::Tensor &tensor, const char *name,
                       const c10::Device &device) {
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::XPU, name,
              " must be an XPU tensor");
  TORCH_CHECK(tensor.device() == device, name,
              " must be on the same concrete XPU device as q");
  TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name,
              " must have bfloat16 dtype");
  TORCH_CHECK(tensor.dim() == 4, name, " must be a rank-4 BHSD tensor");
  TORCH_CHECK(tensor.is_contiguous(), name,
              " must have contiguous packed BHSD layout");
}

void check_pointer_tensor(const at::Tensor &tensor, const char *name,
                          const c10::Device &device, int64_t world_size) {
  TORCH_CHECK(tensor.device() == device, name,
              " must be an XPU tensor on the same concrete device as q");
  TORCH_CHECK(tensor.scalar_type() == at::kLong, name,
              " must have torch.int64 dtype");
  TORCH_CHECK(tensor.dim() == 1 && tensor.numel() == world_size, name,
              " must be rank 1 with world_size entries");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_int_range(int64_t value, const char *name) {
  TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(), name,
              " must fit a non-negative 32-bit integer");
}

struct RingInputs {
  int batch;
  int heads;
  int sequence;
  int qk_dim;
  int value_dim;
  int rank;
  int world_size;
  int iteration;
  int work_groups;
  int signal_stride;
  size_t k_slot_bytes;
  size_t v_slot_bytes;
};

RingInputs validate_inputs(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &k_workspace, const at::Tensor &v_workspace,
    const at::Tensor &signal_pad, const at::Tensor &peer_k_workspace_ptrs,
    const at::Tensor &peer_v_workspace_ptrs,
    const at::Tensor &peer_signal_ptrs, int64_t rank, int64_t world_size,
    int64_t iteration, bool is_causal, int64_t work_groups) {
  TORCH_CHECK(!is_causal,
              "streaming_ring_bf16 does not support causal attention");
  TORCH_CHECK(q.device().type() == c10::DeviceType::XPU,
              "q must be an XPU tensor");
  TORCH_CHECK(q.device().has_index() && q.device().index() >= 0,
              "q must have a concrete non-negative XPU device index");
  check_data_tensor(q, "q", q.device());
  check_data_tensor(k, "k", q.device());
  check_data_tensor(v, "v", q.device());

  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0),
              "q, k, and v batch dimensions must be equal");
  TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1),
              "q, k, and v head counts must be equal; GQA/MQA is unsupported");
  TORCH_CHECK(q.size(2) == k.size(2) && q.size(2) == v.size(2),
              "q, k, and v local sequence lengths must be equal");
  TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(2) > 0,
              "batch, head count, and local sequence length must be positive");
  TORCH_CHECK(q.size(3) == k.size(3),
              "k head dimension must equal q head dimension");

  auto supported_dim = [](int64_t dim) {
    return dim == 64 || dim == 96 || dim == 128 || dim == 192;
  };
  TORCH_CHECK(supported_dim(q.size(3)),
              "q/k head dimension must be one of 64, 96, 128, or 192");
  TORCH_CHECK(supported_dim(v.size(3)),
              "v head dimension must be one of 64, 96, 128, or 192");
  TORCH_CHECK(world_size >= 1 && world_size <= std::numeric_limits<int>::max(),
              "world_size must be in [1, INT_MAX]");
  TORCH_CHECK(rank >= 0 && rank < world_size,
              "rank must be in [0, world_size)");
  TORCH_CHECK(iteration > 0 &&
                  iteration <= std::numeric_limits<int>::max(),
              "iteration must be a positive 32-bit tag");
  TORCH_CHECK(work_groups >= 1 &&
                  work_groups <= kMaxCommunicationWorkGroups,
              "work_groups must be in [1, 32]");

  for (int dim = 0; dim < 4; ++dim) {
    check_int_range(q.size(dim), "q shape value");
    check_int_range(k.size(dim), "k shape value");
    check_int_range(v.size(dim), "v shape value");
    check_int_range(q.stride(dim), "q stride");
    check_int_range(k.stride(dim), "k stride");
    check_int_range(v.stride(dim), "v stride");
  }
  const __int128 output_batch_stride =
      static_cast<__int128>(q.size(1)) * q.size(2) * v.size(3);
  const __int128 lse_batch_stride =
      static_cast<__int128>(q.size(1)) * q.size(2);
  TORCH_CHECK(
      output_batch_stride <= std::numeric_limits<int>::max() &&
          lse_batch_stride <= std::numeric_limits<int>::max(),
              "output and LSE strides must fit a 32-bit integer");

  const int64_t slots = world_size - 1;
  TORCH_CHECK(k_workspace.device() == q.device() &&
                  k_workspace.scalar_type() == at::kBFloat16 &&
                  k_workspace.is_contiguous(),
              "k_workspace must be contiguous BF16 on q's XPU device");
  TORCH_CHECK(v_workspace.device() == q.device() &&
                  v_workspace.scalar_type() == at::kBFloat16 &&
                  v_workspace.is_contiguous(),
              "v_workspace must be contiguous BF16 on q's XPU device");
  TORCH_CHECK(k_workspace.dim() == 5 && k_workspace.size(0) == slots &&
                  k_workspace.size(1) == q.size(0) &&
                  k_workspace.size(2) == q.size(1) &&
                  k_workspace.size(3) == q.size(2) &&
                  k_workspace.size(4) == q.size(3),
              "k_workspace must have shape [world_size-1, B, H, S, Dq]");
  TORCH_CHECK(v_workspace.dim() == 5 && v_workspace.size(0) == slots &&
                  v_workspace.size(1) == v.size(0) &&
                  v_workspace.size(2) == v.size(1) &&
                  v_workspace.size(3) == v.size(2) &&
                  v_workspace.size(4) == v.size(3),
              "v_workspace must have shape [world_size-1, B, H, S, Dv]");
  TORCH_CHECK(signal_pad.device() == q.device() &&
                  signal_pad.scalar_type() == at::kInt &&
                  signal_pad.is_contiguous(),
              "signal_pad must be contiguous torch.int32 on q's XPU device");
  TORCH_CHECK(signal_pad.dim() == 2 && signal_pad.size(0) == slots &&
                  signal_pad.size(1) >= work_groups,
              "signal_pad must have shape [world_size-1, >= work_groups]");
  check_int_range(signal_pad.size(1), "signal_pad stride");

  check_pointer_tensor(peer_k_workspace_ptrs, "peer_k_workspace_ptrs",
                       q.device(), world_size);
  check_pointer_tensor(peer_v_workspace_ptrs, "peer_v_workspace_ptrs",
                       q.device(), world_size);
  check_pointer_tensor(peer_signal_ptrs, "peer_signal_ptrs", q.device(),
                       world_size);

  const auto k_bytes = static_cast<size_t>(k.numel()) * k.element_size();
  const auto v_bytes = static_cast<size_t>(v.numel()) * v.element_size();
  TORCH_CHECK(k_bytes % sizeof(Vector16) == 0 &&
                  v_bytes % sizeof(Vector16) == 0,
              "K/V shard byte sizes must be divisible by 16");

  return {static_cast<int>(q.size(0)), static_cast<int>(q.size(1)),
          static_cast<int>(q.size(2)), static_cast<int>(q.size(3)),
          static_cast<int>(v.size(3)), static_cast<int>(rank),
          static_cast<int>(world_size), static_cast<int>(iteration),
          static_cast<int>(work_groups),
          static_cast<int>(signal_pad.size(1)), k_bytes, v_bytes};
}

sycl::queue &communication_queue(sycl::queue &compute_queue, int device) {
  thread_local std::unordered_map<int, sycl::queue> queues;
  auto found = queues.find(device);
  if (found == queues.end()) {
    found = queues
                .emplace(device,
                         sycl::queue(compute_queue.get_context(),
                                     compute_queue.get_device(),
                                     sycl::property::queue::in_order{}))
                .first;
  }
  return found->second;
}

sycl::event launch_ring_copy(
    sycl::queue &queue, const sycl::event &input_ready, const void *local_k,
    const void *local_v, void *local_k_workspace, void *local_v_workspace,
    int *local_signals, const int64_t *peer_k_ptrs,
    const int64_t *peer_v_ptrs, const int64_t *peer_signal_ptrs,
    const RingInputs &inputs) {
  const size_t k_vectors = inputs.k_slot_bytes / sizeof(Vector16);
  const size_t v_vectors = inputs.v_slot_bytes / sizeof(Vector16);
  return queue.submit([&](sycl::handler &handler) {
    handler.depends_on(input_ready);
    handler.parallel_for<RingCopyKernel>(
        sycl::nd_range<1>(
            sycl::range<1>(inputs.work_groups * kSubgroupSize),
            sycl::range<1>(kSubgroupSize)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
          const int group = static_cast<int>(item.get_group_linear_id());
          const int lane = static_cast<int>(item.get_local_linear_id());
          const int right = (inputs.rank + 1) % inputs.world_size;
          auto remote_k_workspace =
              reinterpret_cast<Vector16 *>(peer_k_ptrs[right]);
          auto remote_v_workspace =
              reinterpret_cast<Vector16 *>(peer_v_ptrs[right]);
          auto remote_signals =
              reinterpret_cast<int *>(peer_signal_ptrs[right]);

          for (int phase = 0; phase < inputs.world_size - 1; ++phase) {
            if (phase > 0) {
              sycl::atomic_ref<
                  int, sycl::memory_order::acquire,
                  sycl::memory_scope::system,
                  sycl::access::address_space::global_space>
                  ready(local_signals[(phase - 1) * inputs.signal_stride +
                                      group]);
              while (ready.load() != inputs.iteration) {
              }
            }
            item.barrier(sycl::access::fence_space::global_space);

            const auto *source_k =
                phase == 0
                ? reinterpret_cast<const Vector16 *>(local_k)
                : reinterpret_cast<const Vector16 *>(local_k_workspace) +
                      static_cast<size_t>(phase - 1) * k_vectors;
            const auto *source_v =
                phase == 0
                ? reinterpret_cast<const Vector16 *>(local_v)
                : reinterpret_cast<const Vector16 *>(local_v_workspace) +
                      static_cast<size_t>(phase - 1) * v_vectors;
            auto *destination_k =
                remote_k_workspace + static_cast<size_t>(phase) * k_vectors;
            auto *destination_v =
                remote_v_workspace + static_cast<size_t>(phase) * v_vectors;

            for (size_t index =
                     static_cast<size_t>(group * kSubgroupSize + lane);
                 index < k_vectors;
                 index +=
                     static_cast<size_t>(inputs.work_groups * kSubgroupSize)) {
              destination_k[index] = source_k[index];
            }
            for (size_t index =
                     static_cast<size_t>(group * kSubgroupSize + lane);
                 index < v_vectors;
                 index +=
                     static_cast<size_t>(inputs.work_groups * kSubgroupSize)) {
              destination_v[index] = source_v[index];
            }
            item.barrier(sycl::access::fence_space::global_space);

            if (lane == 0) {
              sycl::atomic_ref<
                  int, sycl::memory_order::release,
                  sycl::memory_scope::system,
                  sycl::access::address_space::global_space>
                  signal(remote_signals[phase * inputs.signal_stride + group]);
              signal.store(inputs.iteration);
            }
          }
        });
  });
}

sycl::event launch_signal_wait(sycl::queue &queue, int *signals,
                               int slot, const RingInputs &inputs) {
  return queue.submit([&](sycl::handler &handler) {
    handler.parallel_for<RingSignalWaitKernel>(
        sycl::nd_range<1>(sycl::range<1>(kSubgroupSize),
                          sycl::range<1>(kSubgroupSize)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
          const int lane = static_cast<int>(item.get_local_linear_id());
          for (int group = lane; group < inputs.work_groups;
               group += kSubgroupSize) {
            sycl::atomic_ref<
                int, sycl::memory_order::acquire,
                sycl::memory_scope::system,
                sycl::access::address_space::global_space>
                ready(signals[slot * inputs.signal_stride + group]);
            while (ready.load() != inputs.iteration) {
            }
          }
        });
  });
}

sycl::event launch_merge(sycl::queue &queue, float *accum_out,
                         float *accum_lse, const float *partial_out,
                         const float *partial_lse, const RingInputs &inputs) {
  const int64_t rows = static_cast<int64_t>(inputs.batch) * inputs.heads *
                       inputs.sequence;
  const int vectors_per_row = inputs.value_dim / 4;
  return queue.submit([&](sycl::handler &handler) {
    handler.parallel_for<RingMergeKernel>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<size_t>(rows) * kSubgroupSize),
            sycl::range<1>(kSubgroupSize)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
          const int64_t row =
              static_cast<int64_t>(item.get_group_linear_id());
          const int lane = static_cast<int>(item.get_local_linear_id());
          float old_lse = 0.0f;
          float new_lse = 0.0f;
          if (lane == 0) {
            old_lse = accum_lse[row];
            new_lse = partial_lse[row];
          }
          const auto subgroup = item.get_sub_group();
          old_lse = sycl::group_broadcast(subgroup, old_lse, 0);
          new_lse = sycl::group_broadcast(subgroup, new_lse, 0);
          const float maximum = sycl::fmax(old_lse, new_lse);
          const float old_weight = sycl::exp(old_lse - maximum);
          const float new_weight = sycl::exp(new_lse - maximum);
          const float denominator = old_weight + new_weight;
          const float old_scale = old_weight / denominator;
          const float new_scale = new_weight / denominator;

          auto *accum_vectors =
              reinterpret_cast<Float4 *>(accum_out) +
              row * vectors_per_row;
          const auto *partial_vectors =
              reinterpret_cast<const Float4 *>(partial_out) +
              row * vectors_per_row;
          for (int vector = lane; vector < vectors_per_row;
               vector += kSubgroupSize) {
            Float4 result;
#pragma unroll
            for (int element = 0; element < 4; ++element) {
              result.value[element] =
                  old_scale * accum_vectors[vector].value[element] +
                  new_scale * partial_vectors[vector].value[element];
            }
            accum_vectors[vector] = result;
          }
          if (lane == 0) {
            accum_lse[row] = maximum + sycl::log(denominator);
          }
        });
  });
}

template <bool Causal, typename ShapeQK, typename ShapePV, typename ShapeOut,
          typename SubgroupLayoutQK>
int run_partial(const Options &options) {
  constexpr int PipelineStages = 2;
  return FMHAConfig<Causal, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void,
                    PipelineStages, false, bfloat16_t, bfloat16_t,
                    bfloat16_t>::run(options);
}

sycl::event launch_partial(const at::Tensor &q, const void *k_ptr,
                           const void *v_ptr, at::Tensor &out,
                           at::Tensor &lse, sycl::queue &queue,
                           const RingInputs &inputs) {
  Options options;
  options.is_causal = false;
  options.varlen = false;
  options.use_paged_kv = false;
  options.scheduler = "Individual";
  options.batch = inputs.batch;
  options.num_heads_q = inputs.heads;
  options.num_heads_kv = inputs.heads;
  options.seq_len_qo = inputs.sequence;
  options.seq_len_kv = inputs.sequence;
  options.seq_len_kv_cache = 0;
  options.head_size_qk = inputs.qk_dim;
  options.head_size_vo = inputs.value_dim;
  options.iterations = 0;
  options.warmup = 0;
  options.verify = 0;
  options.print_performance = false;
  options.softmax_scale =
      1.0f / std::sqrt(static_cast<float>(inputs.qk_dim));
  options.external_q = q.data_ptr();
  options.external_k = k_ptr;
  options.external_v = v_ptr;
  options.external_o = out.data_ptr<float>();
  options.external_lse = lse.data_ptr<float>();
  options.use_external_strides = true;
  options.stride_q_s = static_cast<int>(q.stride(2));
  options.stride_q_h = static_cast<int>(q.stride(1));
  options.stride_q_b = static_cast<int>(q.stride(0));
  options.stride_k_s = inputs.qk_dim;
  options.stride_k_h = inputs.sequence * inputs.qk_dim;
  options.stride_k_b = inputs.heads * inputs.sequence * inputs.qk_dim;
  options.stride_v_s = inputs.value_dim;
  options.stride_v_h = inputs.sequence * inputs.value_dim;
  options.stride_v_b =
      inputs.heads * inputs.sequence * inputs.value_dim;
  options.stride_o_s = static_cast<int>(out.stride(2));
  options.stride_o_h = static_cast<int>(out.stride(1));
  options.stride_o_b = static_cast<int>(out.stride(0));
  options.stride_lse_s = static_cast<int>(lse.stride(2));
  options.stride_lse_h = static_cast<int>(lse.stride(1));
  options.stride_lse_b = static_cast<int>(lse.stride(0));
  options.external_queue = &queue;
  sycl::event completion;
  options.completion_event = &completion;
  options.async_launch = true;

  int status = 0;
  if (inputs.value_dim == 64) {
    using ShapeQK = Shape<_256, _64, _32>;
    using ShapePV = Shape<_256, _32, _64>;
    using ShapeOut = Shape<_256, _64>;
    using SubgroupLayout = Layout<Shape<_16, _1, _1>>;
    status = run_partial<false, ShapeQK, ShapePV, ShapeOut, SubgroupLayout>(
        options);
  } else if (inputs.value_dim == 96) {
    using ShapeQK = Shape<_256, _64, _32>;
    using ShapePV = Shape<_256, _32, _64>;
    using ShapeOut = Shape<_256, _96>;
    using SubgroupLayout = Layout<Shape<_16, _1, _1>>;
    status = run_partial<false, ShapeQK, ShapePV, ShapeOut, SubgroupLayout>(
        options);
  } else if (inputs.value_dim == 128) {
    using ShapeQK = Shape<_256, _32, _32>;
    using ShapePV = Shape<_256, _32, _32>;
    using ShapeOut = Shape<_256, _128>;
    using SubgroupLayout = Layout<Shape<_16, _1, _1>>;
    status = run_partial<false, ShapeQK, ShapePV, ShapeOut, SubgroupLayout>(
        options);
  } else {
    using ShapeQK = Shape<_256, _64, _32>;
    using ShapePV = Shape<_256, _32, _64>;
    using ShapeOut = Shape<_256, _192>;
    using SubgroupLayout = Layout<Shape<_32, _1, _1>>;
    status = run_partial<false, ShapeQK, ShapePV, ShapeOut, SubgroupLayout>(
        options);
  }
  TORCH_CHECK(status == 0, "FA2 partial launch failed");
  return completion;
}

std::tuple<at::Tensor, at::Tensor> streaming_ring_impl(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &k_workspace, const at::Tensor &v_workspace,
    const at::Tensor &signal_pad, const at::Tensor &peer_k_workspace_ptrs,
    const at::Tensor &peer_v_workspace_ptrs,
    const at::Tensor &peer_signal_ptrs, int64_t rank, int64_t world_size,
    int64_t iteration, bool is_causal, int64_t work_groups) {
  const RingInputs inputs = validate_inputs(
      q, k, v, k_workspace, v_workspace, signal_pad,
      peer_k_workspace_ptrs, peer_v_workspace_ptrs, peer_signal_ptrs, rank,
      world_size, iteration, is_causal, work_groups);
  CompatDeviceGuard device_guard(
      static_cast<unsigned int>(q.device().index()));

  auto output = torch::empty(
      {q.size(0), q.size(1), q.size(2), v.size(3)},
      q.options().dtype(at::kFloat));
  auto lse = torch::empty({q.size(0), q.size(1), q.size(2)},
                          q.options().dtype(at::kFloat));

  auto &compute_queue =
      c10::xpu::getCurrentXPUStream(q.device().index()).queue();
  TORCH_CHECK(
      compute_queue.has_property<sycl::property::queue::in_order>(),
      "streaming_ring_bf16 requires an in-order PyTorch XPU stream");

  const sycl::event input_ready =
      compute_queue.ext_oneapi_submit_barrier();

  if (inputs.world_size == 1) {
    sycl::event last_compute =
        launch_partial(q, k.data_ptr(), v.data_ptr(), output, lse,
                       compute_queue, inputs);
    compute_queue.ext_oneapi_submit_barrier({last_compute});
    return {output, lse};
  }

  auto &comm_queue =
      communication_queue(compute_queue, q.device().index());
  const sycl::event transport = launch_ring_copy(
      comm_queue, input_ready, k.data_ptr(), v.data_ptr(),
      k_workspace.data_ptr(), v_workspace.data_ptr(),
      signal_pad.data_ptr<int>(), peer_k_workspace_ptrs.data_ptr<int64_t>(),
      peer_v_workspace_ptrs.data_ptr<int64_t>(),
      peer_signal_ptrs.data_ptr<int64_t>(), inputs);
  sycl::event last_compute =
      launch_partial(q, k.data_ptr(), v.data_ptr(), output, lse,
                     compute_queue, inputs);

  auto partial_output = torch::empty_like(output);
  auto partial_lse = torch::empty_like(lse);
  const auto *k_slots = static_cast<const uint8_t *>(k_workspace.data_ptr());
  const auto *v_slots = static_cast<const uint8_t *>(v_workspace.data_ptr());
  for (int slot = 0; slot < inputs.world_size - 1; ++slot) {
    launch_signal_wait(compute_queue, signal_pad.data_ptr<int>(), slot,
                       inputs);
    launch_partial(q, k_slots + static_cast<size_t>(slot) * inputs.k_slot_bytes,
                   v_slots + static_cast<size_t>(slot) * inputs.v_slot_bytes,
                   partial_output, partial_lse, compute_queue, inputs);
    last_compute = launch_merge(
        compute_queue, output.data_ptr<float>(), lse.data_ptr<float>(),
        partial_output.data_ptr<float>(), partial_lse.data_ptr<float>(),
        inputs);
  }

  compute_queue.ext_oneapi_submit_barrier({last_compute, transport});
  return {output, lse};
}

} // namespace

at::Tensor streaming_ring_bf16(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &k_workspace, const at::Tensor &v_workspace,
    const at::Tensor &signal_pad, const at::Tensor &peer_k_workspace_ptrs,
    const at::Tensor &peer_v_workspace_ptrs,
    const at::Tensor &peer_signal_ptrs, int64_t rank, int64_t world_size,
    int64_t iteration, bool is_causal, int64_t work_groups) {
  return std::get<0>(streaming_ring_impl(
      q, k, v, k_workspace, v_workspace, signal_pad,
      peer_k_workspace_ptrs, peer_v_workspace_ptrs, peer_signal_ptrs, rank,
      world_size, iteration, is_causal, work_groups));
}

std::tuple<at::Tensor, at::Tensor> streaming_ring_bf16_lse(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &k_workspace, const at::Tensor &v_workspace,
    const at::Tensor &signal_pad, const at::Tensor &peer_k_workspace_ptrs,
    const at::Tensor &peer_v_workspace_ptrs,
    const at::Tensor &peer_signal_ptrs, int64_t rank, int64_t world_size,
    int64_t iteration, bool is_causal, int64_t work_groups) {
  return streaming_ring_impl(
      q, k, v, k_workspace, v_workspace, signal_pad,
      peer_k_workspace_ptrs, peer_v_workspace_ptrs, peer_signal_ptrs, rank,
      world_size, iteration, is_causal, work_groups);
}
