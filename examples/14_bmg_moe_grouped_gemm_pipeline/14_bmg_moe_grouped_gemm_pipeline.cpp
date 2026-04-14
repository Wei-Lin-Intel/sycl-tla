/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief MoE Grouped GEMM Pipeline without activation rearrangement.

    This example implements a complete Mixture-of-Experts (MoE) pipeline using
    grouped GEMMs without rearranging (reordering/expanding) the input activation
    matrix hidden_states. There is NO separate gather kernel. Instead, Grouped
    GEMM #1 picks up the selected token rows directly from hidden_states using
    sorted_token_ids as an index array during computation.

    Pipeline stages:
      1. Build routing tables on host: sorted_token_ids (grouped by expert)
         and tokens_per_expert from topk_ids.
      2. Routed Grouped GEMM #1: hidden_states @ w13_weights^T
         Uses K-sliced gather: the GEMM K-loop is inlined, and for each K-tile
         (stride 32), only the needed TILE_M × 32 slice is gathered from the
         scattered hidden_states rows into a tiny per-workgroup scratch buffer,
         followed by a 2D block load + DPAS accumulate. B prefetch is preserved.
         -> intermediate [total_routed_tokens, 2*intermediate_size]  (BF16 compute)
      3. SiLU-gating activation (FP32 compute): for each token row, split
         the 2*intermediate_size columns into gate (first half) and up (second half),
         compute silu(gate) * up, producing [total_routed_tokens, intermediate_size]
      4. Grouped GEMM #2: silu_output @ w2_weights^T
         -> expert_output [total_routed_tokens, hidden_dim]  (BF16 compute)
         (Standard grouped GEMM since silu_output is already contiguous per expert.)
      5. Scatter-reduce (FP32 compute): accumulate expert outputs back into the
         original token order using sorted_token_ids, then cast to BF16

    Design note — K-sliced gather in Grouped GEMM #1:
      Instead of gathering the full TILE_M × K block (e.g. 256 × 2880 ≈ 1.4 MB
      per workgroup) before running the GEMM, the kernel interleaves the gather
      with the GEMM's K-loop:
        for each K-tile (32 columns):
          1. Cooperative gather of TILE_M × 32 from scattered rows → scratch
          2. Workgroup barrier
          3. 2D block load A from scratch, 2D block load B from weights, DPAS
          4. Named barrier
      Scratch per workgroup: TILE_M × K_TILE × 2 = 16 KB (fits in L1).
      Each gather reads one cache line per row per K-tile. Hidden-state rows
      warm in cache across K iterations. K_TILE=32 (power of 2) makes the
      row/col index computation a cheap shift + AND.
      SiLU and scatter-reduce steps use simple SYCL parallel_for kernels
      operating in registers.

    Usage:
      $ ./14_bmg_moe_grouped_gemm_pipeline [options]
      --help                      Display usage
      --num_tokens=<int>          Number of input tokens (default: 512)
      --hidden_dim=<int>          Hidden dimension (default: 2880)
      --intermediate_size=<int>   Intermediate MLP size (default: 1440)
      --num_experts=<int>         Number of experts (default: 8)
      --topk=<int>                Top-K experts per token (default: 2)
      --verify=<int>              Enable verification (0=no, 1=yes, default: 1)
*/

#include <cute/tensor.hpp>
#include <numeric>
#include <random>

#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/initialize_block.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/sycl_event_manager.hpp"

// Reuse MoE grouped GEMM infrastructure from example 12
#include "../12_xe20_moe_gemm_cute_interface/moe_grouped_gemm.hpp"
#include "../12_xe20_moe_gemm_cute_interface/moe_tile_scheduler.hpp"

// Routed GEMM variant that gathers A-rows inside the kernel
#include "moe_routed_gemm.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

using namespace cute;
using namespace MoE;

using ElementCompute = cutlass::bfloat16_t;
using ElementAccumulator = float;

///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  bool error;

  int num_tokens, hidden_dim, intermediate_size, num_experts, topk, verify;

  Options()
      : help(false), error(false), num_tokens(512), hidden_dim(2880),
        intermediate_size(1440), num_experts(8), topk(2), verify(1) {}

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("num_tokens", num_tokens, 512);
    cmd.get_cmd_line_argument("hidden_dim", hidden_dim, 2880);
    cmd.get_cmd_line_argument("intermediate_size", intermediate_size, 1440);
    cmd.get_cmd_line_argument("num_experts", num_experts, 8);
    cmd.get_cmd_line_argument("topk", topk, 2);
    cmd.get_cmd_line_argument("verify", verify, 1);
  }

  /// Prints the usage statement.
  std::ostream &print_usage(std::ostream &out) const {

    out << "MoE Grouped GEMM Pipeline (no activation rearrangement)\n\n"
        << "Options:\n\n"
        << "  --help                      If specified, displays this usage "
           "statement\n\n"
        << "  --num_tokens=<int>          Number of input tokens (default: "
           "512)\n"
        << "  --hidden_dim=<int>          Hidden dimension (default: 2880)\n"
        << "  --intermediate_size=<int>   Intermediate MLP size (default: "
           "1440)\n"
        << "  --num_experts=<int>         Number of experts (default: 8)\n"
        << "  --topk=<int>               Top-K experts per token (default: "
           "2)\n"
        << "  --verify=<int>              Specify whether to verify (0=no, "
           "1=yes, default: 1)\n\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// SYCL kernel name tags
template <typename, typename, typename, char, char, int> class MoEGemmKernelName;
template <typename, typename, typename, char, char, int> class MoERoutedGemmKernelName;
class SiluGatingKernelName;
class ScatterReduceKernelName;
class ScatterReduceCastKernelName;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Verification helper: reference computation on host via per-expert reference GEMMs

struct VerificationHelper {

  // Run the full MoE pipeline on device using reference GEMMs for verification
  bool verify_pipeline(
      const cutlass::bfloat16_t *hidden_states_host,
      const cutlass::bfloat16_t *w13_weights_host,
      const cutlass::bfloat16_t *w2_weights_host,
      const int32_t *topk_ids_host,
      const cutlass::bfloat16_t *final_output_device,
      int num_tokens, int hidden_dim, int intermediate_size,
      int num_experts, int topk) {

    // Compute reference on host in FP32
    int inter2 = 2 * intermediate_size;
    int total_routed = num_tokens * topk;

    // Allocate host buffers
    std::vector<float> hidden_fp32(num_tokens * hidden_dim);
    std::vector<float> w13_fp32(num_experts * hidden_dim * inter2);
    std::vector<float> w2_fp32(num_experts * intermediate_size * hidden_dim);

    // Convert inputs to FP32
    for (int i = 0; i < num_tokens * hidden_dim; i++)
      hidden_fp32[i] = static_cast<float>(hidden_states_host[i]);
    for (int i = 0; i < num_experts * hidden_dim * inter2; i++)
      w13_fp32[i] = static_cast<float>(w13_weights_host[i]);
    for (int i = 0; i < num_experts * intermediate_size * hidden_dim; i++)
      w2_fp32[i] = static_cast<float>(w2_weights_host[i]);

    // Reference output in FP32 then BF16
    std::vector<float> ref_output_fp32(num_tokens * hidden_dim, 0.0f);

    // For each token, for each top-k selection
    for (int t = 0; t < num_tokens; t++) {
      for (int ki = 0; ki < topk; ki++) {
        int expert_id = topk_ids_host[t * topk + ki];
        if (expert_id < 0 || expert_id >= num_experts)
          continue;

        // GEMM1: hidden_states[t] @ w13_weights[expert_id]^T
        // hidden_states[t]: [1, hidden_dim] row-major
        // w13_weights[expert_id]: [hidden_dim, inter2] row-major => A[1,hidden_dim] * B[hidden_dim, inter2] = C[1, inter2]
        std::vector<float> gemm1_out(inter2, 0.0f);
        const float *A_row = &hidden_fp32[t * hidden_dim];
        const float *B_mat = &w13_fp32[expert_id * hidden_dim * inter2];
        // A[1, hidden_dim] * B[hidden_dim, inter2]  (B is row-major: B[r][c] = B_mat[r*inter2 + c])
        for (int j = 0; j < inter2; j++) {
          float sum = 0.0f;
          for (int kk = 0; kk < hidden_dim; kk++) {
            sum += A_row[kk] * B_mat[kk * inter2 + j];
          }
          gemm1_out[j] = sum;
        }

        // SiLU gating: gate = first half, up = second half
        // silu(gate) * up
        std::vector<float> silu_out(intermediate_size);
        for (int j = 0; j < intermediate_size; j++) {
          float gate = gemm1_out[j];
          float up = gemm1_out[intermediate_size + j];
          float silu_gate = gate / (1.0f + std::exp(-gate));
          silu_out[j] = silu_gate * up;
        }

        // GEMM2: silu_out[1, intermediate_size] @ w2_weights[expert_id]^T
        // w2_weights[expert_id]: [intermediate_size, hidden_dim] row-major
        const float *W2_mat =
            &w2_fp32[expert_id * intermediate_size * hidden_dim];
        for (int j = 0; j < hidden_dim; j++) {
          float sum = 0.0f;
          for (int kk = 0; kk < intermediate_size; kk++) {
            sum += silu_out[kk] * W2_mat[kk * hidden_dim + j];
          }
          // Reduce-sum across top-k
          ref_output_fp32[t * hidden_dim + j] += sum;
        }
      }
    }

    // Cast reference to BF16
    std::vector<cutlass::bfloat16_t> ref_output_bf16(num_tokens * hidden_dim);
    for (int i = 0; i < num_tokens * hidden_dim; i++) {
      ref_output_bf16[i] = static_cast<cutlass::bfloat16_t>(ref_output_fp32[i]);
    }

    // Copy device result to host
    std::vector<cutlass::bfloat16_t> device_output_host(num_tokens * hidden_dim);
    cutlass::device_memory::copy_to_host(
        device_output_host.data(), final_output_device, num_tokens * hidden_dim);
    compat::wait();

    // Compare with tolerance
    int mismatches = 0;
    float max_abs_diff = 0.0f;
    for (int i = 0; i < num_tokens * hidden_dim; i++) {
      float ref_val = static_cast<float>(ref_output_bf16[i]);
      float dev_val = static_cast<float>(device_output_host[i]);
      float abs_diff = std::abs(ref_val - dev_val);
      float rel_tol = 0.05f; // 5% relative tolerance for BF16 chain
      float abs_tol = 0.5f;
      if (abs_diff > abs_tol + rel_tol * std::abs(ref_val)) {
        mismatches++;
        if (mismatches <= 5) {
          std::cout << "  Mismatch at index " << i << ": ref=" << ref_val
                    << " dev=" << dev_val << " diff=" << abs_diff << std::endl;
        }
      }
      max_abs_diff = std::max(max_abs_diff, abs_diff);
    }

    std::cout << "  Max absolute diff: " << max_abs_diff << std::endl;
    std::cout << "  Mismatches: " << mismatches << " / "
              << num_tokens * hidden_dim << std::endl;

    return mismatches == 0;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <class TA, class TB> auto choose_tiled_mma(TA * /*A*/, TB * /*B*/) {
  using TA_non_CV = cutlass::platform::remove_cv_t<TA>;
  using TB_non_CV = cutlass::platform::remove_cv_t<TB>;
  auto op = XE_DPAS_TT<8, float, TA_non_CV, TB_non_CV>{};

  using WGTile = Shape<_256, _128, _32>; // 256x128 WG tile size
  using SGLayout =
      Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>; // 8x2 SG tiling, n-major

  using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>,
                                      SGLayout>::TiledMMA;

  return MMA{};
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Compute the total number of workgroups the persistent tile scheduler will use.
// This is needed to allocate the per-workgroup scratch buffer for the routed GEMM.
int compute_total_workgroups() {
  int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  cutlass::KernelHardwareInfo hw_info{0, sm_count};
  auto dummy_problem_shape = cute::Shape<int, int, int>{1, 1, 1};
  auto dummy_group_problem_shape =
      cutlass::gemm::GroupProblemShape<Shape<int, int, int>>{
          1, &dummy_problem_shape, nullptr};
  using TileShape = Shape<_256, _128, _32>;
  using ClusterShape = Shape<_1, _1, _1>;
  auto scheduler_params =
      PersistentTileSchedulerXeMoE<ProblemShape>::to_underlying_arguments(
          dummy_group_problem_shape, TileShape{}, ClusterShape{}, hw_info,
          PersistentTileSchedulerXeMoE<ProblemShape>::Arguments{
              1, RasterOrderOptions::AlongN});
  auto gd = PersistentTileSchedulerXeMoE<ProblemShape>::get_grid_shape(
      scheduler_params, dummy_group_problem_shape, TileShape{}, ClusterShape{},
      hw_info,
      PersistentTileSchedulerXeMoE<ProblemShape>::Arguments{
          1, RasterOrderOptions::AlongN});
  return gd.x * gd.y * gd.z;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Launch a grouped GEMM using the MoE tile scheduler (same as example 12)
template <char layoutA, char layoutB, typename ElementA, typename ElementB,
          typename ElementD, int GemmId>
void launch_grouped_gemm(const ElementA *activations, const ElementB *weights,
                         ElementD *outputs, const int gemm_n, const int gemm_k,
                         const int *num_rows_per_expert_device,
                         const int num_experts) {
  int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  cutlass::KernelHardwareInfo hw_info{0, sm_count};
  auto dummy_problem_shape = cute::Shape<int, int, int>{1, gemm_k, gemm_n};
  auto dummy_group_problem_shape =
      cutlass::gemm::GroupProblemShape<Shape<int, int, int>>{
          1, &dummy_problem_shape, nullptr};
  using TileShape = Shape<_256, _128, _32>;
  using ClusterShape = Shape<_1, _1, _1>;
  auto scheduler_params =
      PersistentTileSchedulerXeMoE<ProblemShape>::to_underlying_arguments(
          dummy_group_problem_shape, TileShape{}, ClusterShape{}, hw_info,
          PersistentTileSchedulerXeMoE<ProblemShape>::Arguments{
              1, RasterOrderOptions::AlongN});
  auto group_distribution =
      PersistentTileSchedulerXeMoE<ProblemShape>::get_grid_shape(
          scheduler_params, dummy_group_problem_shape, TileShape{},
          ClusterShape{}, hw_info,
          PersistentTileSchedulerXeMoE<ProblemShape>::Arguments{
              1, RasterOrderOptions::AlongN});
  auto mma = choose_tiled_mma(activations, weights);
  auto MaxThreadsPerWorkgroup = size(mma);
  dim3 local_range{MaxThreadsPerWorkgroup, 1, 1};

  sycl::range<3> local = {local_range.z, local_range.y, local_range.x};
  sycl::range<3> groups = {group_distribution.z, group_distribution.y,
                           group_distribution.x};
  sycl::range<3> global = {local[0] * groups[0], local[1] * groups[1],
                           local[2] * groups[2]};

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{syclex::sub_group_size<16>,
#if (defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35))
                                  intelex::grf_size<512>
#else
                                  intelex::grf_size<256>
#endif
  };
  sycl::queue Q = compat::get_default_queue();

  auto event = Q.parallel_for<
      MoEGemmKernelName<ElementA, ElementB, ElementD, layoutA, layoutB,
                        GemmId>>(
      sycl::nd_range<3>(global, local), kernel_props, [=](auto) {
        MoE::MoEGEMM<XE_LOAD_2D<16, 32, 32, 16>,
                     XE_LOAD_2D_VNNI<16, 32, 16, 16>, XE_STORE_2D<16, 8, 32>,
                     'R', 'R', 'R'>(activations, weights,
                                    static_cast<void *>(nullptr), outputs, mma,
                                    num_rows_per_expert_device, num_experts,
                                    gemm_n, gemm_k, scheduler_params);
      });
  EventManager::getInstance().addEvent(event);
  Q.wait_and_throw();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Launch a routed grouped GEMM for GEMM #1: gathers A-matrix rows from
// hidden_states inside the kernel using sorted_token_ids.
template <char layoutA, char layoutB, typename ElementA, typename ElementB,
          typename ElementD, int GemmId>
void launch_routed_grouped_gemm(const ElementA *hidden_states,
                                const int32_t *sorted_token_ids,
                                ElementA *scratch_a,
                                const ElementB *weights,
                                ElementD *outputs,
                                const int gemm_n, const int gemm_k,
                                const int *num_rows_per_expert_device,
                                const int num_experts) {
  int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  cutlass::KernelHardwareInfo hw_info{0, sm_count};
  auto dummy_problem_shape = cute::Shape<int, int, int>{1, gemm_k, gemm_n};
  auto dummy_group_problem_shape =
      cutlass::gemm::GroupProblemShape<Shape<int, int, int>>{
          1, &dummy_problem_shape, nullptr};
  using TileShape = Shape<_256, _128, _32>;
  using ClusterShape = Shape<_1, _1, _1>;
  auto scheduler_params =
      PersistentTileSchedulerXeMoE<ProblemShape>::to_underlying_arguments(
          dummy_group_problem_shape, TileShape{}, ClusterShape{}, hw_info,
          PersistentTileSchedulerXeMoE<ProblemShape>::Arguments{
              1, RasterOrderOptions::AlongN});
  auto group_distribution =
      PersistentTileSchedulerXeMoE<ProblemShape>::get_grid_shape(
          scheduler_params, dummy_group_problem_shape, TileShape{},
          ClusterShape{}, hw_info,
          PersistentTileSchedulerXeMoE<ProblemShape>::Arguments{
              1, RasterOrderOptions::AlongN});
  auto mma = choose_tiled_mma(hidden_states, weights);
  auto MaxThreadsPerWorkgroup = size(mma);
  dim3 local_range{MaxThreadsPerWorkgroup, 1, 1};

  sycl::range<3> local = {local_range.z, local_range.y, local_range.x};
  sycl::range<3> groups = {group_distribution.z, group_distribution.y,
                           group_distribution.x};
  sycl::range<3> global = {local[0] * groups[0], local[1] * groups[1],
                           local[2] * groups[2]};

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{syclex::sub_group_size<16>,
#if (defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35))
                                  intelex::grf_size<512>
#else
                                  intelex::grf_size<256>
#endif
  };
  sycl::queue Q = compat::get_default_queue();

  constexpr int32_t TILE_M = 256;

  auto event = Q.parallel_for<
      MoERoutedGemmKernelName<ElementA, ElementB, ElementD, layoutA, layoutB,
                              GemmId>>(
      sycl::nd_range<3>(global, local), kernel_props, [=](auto) {
        MoE::MoEGEMMRouted<XE_LOAD_2D<16, 32, 32, 16>,
                           XE_LOAD_2D_VNNI<16, 32, 16, 16>,
                           XE_STORE_2D<16, 8, 32>,
                           'R', 'R', 'R'>(
            hidden_states, sorted_token_ids, scratch_a,
            weights, static_cast<void *>(nullptr), outputs, mma,
            num_rows_per_expert_device, num_experts,
            gemm_n, gemm_k, TILE_M, scheduler_params);
      });
  EventManager::getInstance().addEvent(event);
  Q.wait_and_throw();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// SiLU gating kernel: silu(gate) * up
// Input:  gemm1_output [total_routed_tokens, 2*intermediate_size] in BF16
// Output: silu_output  [total_routed_tokens, intermediate_size] in BF16
// Compute in FP32
void launch_silu_gating(const cutlass::bfloat16_t *gemm1_output,
                        cutlass::bfloat16_t *silu_output,
                        int total_routed_tokens, int intermediate_size) {
  sycl::queue Q = compat::get_default_queue();
  int inter2 = 2 * intermediate_size;
  auto event = Q.parallel_for<SiluGatingKernelName>(
      sycl::range<1>(total_routed_tokens * intermediate_size),
      [=](sycl::id<1> idx) {
        int linear = idx[0];
        int row = linear / intermediate_size;
        int col = linear % intermediate_size;
        // gate is the first half, up is the second half
        float gate =
            static_cast<float>(gemm1_output[row * inter2 + col]);
        float up = static_cast<float>(
            gemm1_output[row * inter2 + intermediate_size + col]);
        // SiLU(gate) = gate * sigmoid(gate)
        float silu_gate = gate / (1.0f + sycl::exp(-gate));
        silu_output[row * intermediate_size + col] =
            static_cast<cutlass::bfloat16_t>(silu_gate * up);
      });
  EventManager::getInstance().addEvent(event);
  Q.wait_and_throw();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Scatter-reduce kernel: accumulate expert outputs back to original token order
// gemm2_output [total_routed_tokens, hidden_dim] in BF16
// final_output [num_tokens, hidden_dim] in BF16
// sorted_token_ids[i] gives the original token index for sorted position i
// We first zero the output, then atomically add in FP32, then cast to BF16
void launch_scatter_reduce(const cutlass::bfloat16_t *gemm2_output,
                           cutlass::bfloat16_t *final_output,
                           float *reduce_buffer,
                           const int32_t *sorted_token_ids,
                           int total_routed_tokens, int num_tokens,
                           int hidden_dim) {
  sycl::queue Q = compat::get_default_queue();

  // Zero the reduce buffer
  Q.memset(reduce_buffer, 0, num_tokens * hidden_dim * sizeof(float))
      .wait();

  // Accumulate in FP32
  // Use work-items per element; each scatters one value
  auto event1 = Q.parallel_for<ScatterReduceKernelName>(
      sycl::range<1>(total_routed_tokens * hidden_dim),
      [=](sycl::id<1> idx) {
        int linear = idx[0];
        int sorted_idx = linear / hidden_dim;
        int col = linear % hidden_dim;
        int orig_token = sorted_token_ids[sorted_idx];
        float val = static_cast<float>(
            gemm2_output[sorted_idx * hidden_dim + col]);
        // Use atomic add for FP32 reduction
        auto ref = sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(
            reduce_buffer[orig_token * hidden_dim + col]);
        ref.fetch_add(val);
      });
  EventManager::getInstance().addEvent(event1);
  Q.wait_and_throw();

  // Cast FP32 buffer to BF16 output
  auto event2 = Q.parallel_for<ScatterReduceCastKernelName>(
      sycl::range<1>(num_tokens * hidden_dim), [=](sycl::id<1> idx) {
        int i = idx[0];
        final_output[i] =
            static_cast<cutlass::bfloat16_t>(reduce_buffer[i]);
      });
  EventManager::getInstance().addEvent(event2);
  Q.wait_and_throw();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Build routing tables on host from topk_ids
// Returns:
//   sorted_token_ids: for each sorted position, the original token index
//   tokens_per_expert: number of routed tokens for each expert
void build_routing_tables(const int32_t *topk_ids_host, int num_tokens,
                          int topk, int num_experts,
                          std::vector<int32_t> &sorted_token_ids,
                          std::vector<int32_t> &tokens_per_expert) {
  // Count tokens per expert
  tokens_per_expert.assign(num_experts, 0);
  for (int t = 0; t < num_tokens; t++) {
    for (int ki = 0; ki < topk; ki++) {
      int eid = topk_ids_host[t * topk + ki];
      if (eid >= 0 && eid < num_experts)
        tokens_per_expert[eid]++;
    }
  }

  // Compute expert offsets (exclusive prefix sum)
  std::vector<int32_t> expert_offsets(num_experts + 1, 0);
  for (int e = 0; e < num_experts; e++) {
    expert_offsets[e + 1] = expert_offsets[e] + tokens_per_expert[e];
  }

  int total_routed = expert_offsets[num_experts];
  sorted_token_ids.resize(total_routed);

  // Fill sorted_token_ids: tokens are grouped by expert
  std::vector<int32_t> write_pos(num_experts);
  for (int e = 0; e < num_experts; e++)
    write_pos[e] = expert_offsets[e];

  for (int t = 0; t < num_tokens; t++) {
    for (int ki = 0; ki < topk; ki++) {
      int eid = topk_ids_host[t * topk + ki];
      if (eid >= 0 && eid < num_experts) {
        sorted_token_ids[write_pos[eid]] = t;
        write_pos[eid]++;
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void run_moe_pipeline(const Options &options) {
  int num_tokens = options.num_tokens;
  int hidden_dim = options.hidden_dim;
  int intermediate_size = options.intermediate_size;
  int num_experts = options.num_experts;
  int topk = options.topk;
  int inter2 = 2 * intermediate_size;

  std::cout << "\n=== MoE Grouped GEMM Pipeline ===" << std::endl;
  std::cout << "  num_tokens       : " << num_tokens << std::endl;
  std::cout << "  hidden_dim       : " << hidden_dim << std::endl;
  std::cout << "  intermediate_size: " << intermediate_size << std::endl;
  std::cout << "  num_experts      : " << num_experts << std::endl;
  std::cout << "  topk             : " << topk << std::endl;

  // ---- Allocate and initialize inputs ----
  uint64_t seed = 42;

  // hidden_states: [num_tokens, hidden_dim]
  cutlass::DeviceAllocation<cutlass::bfloat16_t> hidden_states;
  hidden_states.reset(num_tokens * hidden_dim);
  initialize_block(hidden_states, seed + 1);

  // w13_weights: [num_experts, hidden_dim, 2*intermediate_size] stored as
  // [num_experts * hidden_dim, 2*intermediate_size] row-major
  cutlass::DeviceAllocation<cutlass::bfloat16_t> w13_weights;
  w13_weights.reset(num_experts * hidden_dim * inter2);
  initialize_block(w13_weights, seed + 2);

  // w2_weights: [num_experts, intermediate_size, hidden_dim] stored as
  // [num_experts * intermediate_size, hidden_dim] row-major
  cutlass::DeviceAllocation<cutlass::bfloat16_t> w2_weights;
  w2_weights.reset(num_experts * intermediate_size * hidden_dim);
  initialize_block(w2_weights, seed + 3);

  // topk_ids: [num_tokens, topk] - generate random expert assignments
  std::vector<int32_t> topk_ids_host(num_tokens * topk);
  std::mt19937 rng(seed + 4);
  for (int t = 0; t < num_tokens; t++) {
    // Pick topk distinct experts for this token
    std::vector<int> available(num_experts);
    std::iota(available.begin(), available.end(), 0);
    for (int ki = 0; ki < topk; ki++) {
      int idx = rng() % available.size();
      topk_ids_host[t * topk + ki] = available[idx];
      available.erase(available.begin() + idx);
    }
  }

  // ---- Build routing tables ----
  std::vector<int32_t> sorted_token_ids;
  std::vector<int32_t> tokens_per_expert;
  build_routing_tables(topk_ids_host.data(), num_tokens, topk, num_experts,
                       sorted_token_ids, tokens_per_expert);

  int total_routed_tokens = static_cast<int>(sorted_token_ids.size());
  std::cout << "  total_routed_tokens: " << total_routed_tokens << std::endl;

  // Print tokens per expert
  std::cout << "  tokens_per_expert: [";
  for (int e = 0; e < num_experts; e++) {
    std::cout << tokens_per_expert[e];
    if (e < num_experts - 1)
      std::cout << ", ";
  }
  std::cout << "]" << std::endl;

  // Copy routing data to device
  cutlass::DeviceAllocation<int32_t> sorted_token_ids_device;
  sorted_token_ids_device.reset(total_routed_tokens);
  sorted_token_ids_device.copy_from_host(sorted_token_ids.data());

  cutlass::DeviceAllocation<int32_t> tokens_per_expert_device;
  tokens_per_expert_device.reset(num_experts);
  tokens_per_expert_device.copy_from_host(tokens_per_expert.data());

  // ---- Allocate intermediate buffers ----

  // Per-workgroup scratch buffer for routed GEMM #1 (K-sliced gather).
  // Each workgroup stages only TILE_M × K_TILE elements at a time
  // (e.g. 256 × 32 × 2 bytes = 16 KB, fits in L1 cache).
  constexpr int TILE_M = 256;
  constexpr int K_TILE = 32;
  int total_workgroups = compute_total_workgroups();
  int64_t scratch_size =
      static_cast<int64_t>(total_workgroups) * TILE_M * K_TILE;
  cutlass::DeviceAllocation<cutlass::bfloat16_t> scratch_a;
  scratch_a.reset(scratch_size);
  std::cout << "  scratch buffer   : " << total_workgroups << " workgroups × "
            << TILE_M << " × " << K_TILE << " = "
            << (scratch_size * 2) / 1024 << " KB" << std::endl;

  // GEMM1 output: [total_routed_tokens, 2*intermediate_size]
  cutlass::DeviceAllocation<cutlass::bfloat16_t> gemm1_output;
  gemm1_output.reset(total_routed_tokens * inter2);

  // SiLU output: [total_routed_tokens, intermediate_size]
  cutlass::DeviceAllocation<cutlass::bfloat16_t> silu_output;
  silu_output.reset(total_routed_tokens * intermediate_size);

  // GEMM2 output: [total_routed_tokens, hidden_dim]
  cutlass::DeviceAllocation<cutlass::bfloat16_t> gemm2_output;
  gemm2_output.reset(total_routed_tokens * hidden_dim);

  // Reduce buffer (FP32): [num_tokens, hidden_dim]
  cutlass::DeviceAllocation<float> reduce_buffer;
  reduce_buffer.reset(num_tokens * hidden_dim);

  // Final output: [num_tokens, hidden_dim] in BF16
  cutlass::DeviceAllocation<cutlass::bfloat16_t> final_output;
  final_output.reset(num_tokens * hidden_dim);

  // ---- Run the pipeline ----
  GPU_Clock timer;
  timer.start();

  // Step 1: Routed Grouped GEMM #1 - hidden_states @ w13_weights^T
  //   K-sliced gather: the kernel fetches A-matrix rows directly from
  //   hidden_states using sorted_token_ids, one K-tile (32 cols) at a time.
  //   A: per-expert rows from hidden_states[sorted_token_ids[...], :hidden_dim]
  //   B: w13_weights[e]: [hidden_dim, 2*intermediate_size]
  //   D: gemm1_output: [total_routed_tokens, 2*intermediate_size] (contiguous per expert)
  launch_routed_grouped_gemm<'R', 'R', cutlass::bfloat16_t, cutlass::bfloat16_t,
                             cutlass::bfloat16_t, 1>(
      hidden_states.get(), sorted_token_ids_device.get(), scratch_a.get(),
      w13_weights.get(), gemm1_output.get(), inter2, hidden_dim,
      tokens_per_expert_device.get(), num_experts);

  // Step 2: SiLU gating activation (FP32 compute)
  launch_silu_gating(gemm1_output.get(), silu_output.get(),
                     total_routed_tokens, intermediate_size);

  // Step 3: Grouped GEMM #2 - silu_output @ w2_weights^T
  //   silu_output is already contiguous per expert, so standard grouped GEMM.
  //   A: [tokens_per_expert[e], intermediate_size] for each expert
  //   B: w2_weights[e]: [intermediate_size, hidden_dim]
  //   D: gemm2_output[e]: [tokens_per_expert[e], hidden_dim]
  launch_grouped_gemm<'R', 'R', cutlass::bfloat16_t, cutlass::bfloat16_t,
                      cutlass::bfloat16_t, 2>(
      silu_output.get(), w2_weights.get(), gemm2_output.get(), hidden_dim,
      intermediate_size, tokens_per_expert_device.get(), num_experts);

  // Step 4: Scatter-reduce (FP32 compute, cast to BF16)
  launch_scatter_reduce(gemm2_output.get(), final_output.get(),
                        reduce_buffer.get(), sorted_token_ids_device.get(),
                        total_routed_tokens, num_tokens, hidden_dim);

  float pipeline_time_ms = timer.seconds() * 1000;
  std::cout << "\n  Pipeline runtime : " << pipeline_time_ms << " ms"
            << std::endl;

  // ---- Verification ----
  if (options.verify) {
    std::cout << "\n  Running verification..." << std::endl;

    // Copy inputs to host for reference computation
    std::vector<cutlass::bfloat16_t> hidden_host(num_tokens * hidden_dim);
    std::vector<cutlass::bfloat16_t> w13_host(num_experts * hidden_dim * inter2);
    std::vector<cutlass::bfloat16_t> w2_host(num_experts * intermediate_size * hidden_dim);

    cutlass::device_memory::copy_to_host(hidden_host.data(),
                                         hidden_states.get(),
                                         num_tokens * hidden_dim);
    cutlass::device_memory::copy_to_host(w13_host.data(), w13_weights.get(),
                                         num_experts * hidden_dim * inter2);
    cutlass::device_memory::copy_to_host(
        w2_host.data(), w2_weights.get(),
        num_experts * intermediate_size * hidden_dim);
    compat::wait();

    VerificationHelper verifier;
    bool passed = verifier.verify_pipeline(
        hidden_host.data(), w13_host.data(), w2_host.data(),
        topk_ids_host.data(), final_output.get(), num_tokens, hidden_dim,
        intermediate_size, num_experts, topk);

    std::cout << "  Disposition: " << (passed ? "Passed" : "Failed")
              << std::endl;
    if (!passed) {
      std::cerr << "\n  Failed accuracy verification :(\n" << std::endl;
    }
  } else {
    std::cout << "  Verification skipped." << std::endl;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv) {
  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  run_moe_pipeline(options);

  return 0;
}
