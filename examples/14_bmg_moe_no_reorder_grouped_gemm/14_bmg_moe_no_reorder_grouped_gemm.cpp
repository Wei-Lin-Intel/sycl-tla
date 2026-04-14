/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
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
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief MoE-style grouped GEMM pipeline without reordering the input activation matrix.

    This example implements a practical Mixture-of-Experts (MoE) FFN computation that avoids
    the common pre-processing step of reordering / expanding the hidden_states matrix.

    Inputs:
      - hidden_states  [num_token, hidden_dim]                       BF16
      - w13_weights    [num_expert, hidden_dim, 2*intermediate_size] BF16  (gate+up projection)
      - w2_weights     [num_expert, intermediate_size, hidden_dim]   BF16  (down projection)
      - topk_ids       [num_token, topk]                             int32 (selected expert per token)

    Pipeline:
      1. Gather  : route token rows from hidden_states to per-expert activation buffers
                   using topk_ids — no reorder of the original hidden_states matrix.
      2. GEMM-1  : per-expert  expert_acts[e] @ w13_weights[e]  →  intermediate13[e]
                   GEMM compute precision: BF16 (FP32 accumulation, BF16 output)
      3. SiLU    : apply SwiGLU gate: silu_out = silu(gate) * up,  computed in FP32
                   where gate = intermediate13[:, :intermediate_size]
                   and   up   = intermediate13[:, intermediate_size:]
                   Result cast back to BF16.
      4. GEMM-2  : per-expert  silu_out[e] @ w2_weights[e]  →  expert_outs[e]
                   GEMM compute precision: BF16 (FP32 accumulation, BF16 output)
      5. Reduce  : for each token, sum the topk expert outputs (FP32 accumulation, BF16 output)

    Design decision — SLM vs registers for the gather step:
      The token-gather kernel uses register-based direct loads: each SYCL work-item reads one
      BF16 element from hidden_states[token_idx, feat] into a register and writes it directly to
      expert_acts[global_pos, feat].  SLM was not used here because:
        • The gather is a pure scatter–gather without data reuse across work-items, so SLM
          staging would only add synchronisation overhead with no bandwidth benefit.
        • Keeping the A-matrix tiles in contiguous global memory lets the existing grouped GEMM
          infrastructure (from example 12) operate unchanged — its 2D-block load atoms expect
          row-major contiguous layouts in global memory.

    Build & run (from your build directory):
      $ ninja 14_bmg_moe_no_reorder_grouped_gemm
      $ ./examples/sycl/14_bmg_moe_no_reorder_grouped_gemm/14_bmg_moe_no_reorder_grouped_gemm
      $ ./examples/sycl/14_bmg_moe_no_reorder_grouped_gemm/14_bmg_moe_no_reorder_grouped_gemm --help
*/

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/initialize_block.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/sycl_event_manager.hpp"

// Reuse the MoE grouped-GEMM kernel and tile-scheduler from example 12.
// The CMakeLists.txt for this example adds example 12's directory to the
// include path so the relative includes inside these headers resolve correctly.
#include "moe_grouped_gemm.hpp"
#include "moe_tile_scheduler.hpp"

#include "helper.h"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

using namespace cute;

// ─────────────────────────────────────────────────────────────────────────────
// Type aliases
// ─────────────────────────────────────────────────────────────────────────────
using ElementAct   = cutlass::bfloat16_t;   ///< hidden_states / per-expert activations
using ElementW13   = cutlass::bfloat16_t;   ///< w13_weights (gate+up projection)
using ElementW2    = cutlass::bfloat16_t;   ///< w2_weights  (down projection)
using ElementInter = cutlass::bfloat16_t;   ///< intermediate buffers (GEMM outputs, SiLU I/O)
using ElementOut   = cutlass::bfloat16_t;   ///< final output after reduce

// ─────────────────────────────────────────────────────────────────────────────
// Command-line options
// ─────────────────────────────────────────────────────────────────────────────
struct Options {
  bool help  = false;
  bool error = false;

  int num_token         = 256;  ///< Number of input tokens
  int hidden_dim        = 1024; ///< Token hidden dimension (K for GEMM-1, N for GEMM-2)
  int intermediate_size = 512;  ///< Expert intermediate size;  w13 output width = 2 * this
  int num_expert        = 4;    ///< Number of experts
  int topk              = 2;    ///< Top-k experts per token
  int iterations        = 1;    ///< Benchmark repetitions (0 = skip timing)
  int verify            = 1;    ///< Run correctness check (0 = skip)

  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);
    if (cmd.check_cmd_line_flag("help")) { help = true; return; }

    cmd.get_cmd_line_argument("num_token",         num_token,         256);
    cmd.get_cmd_line_argument("hidden_dim",        hidden_dim,        1024);
    cmd.get_cmd_line_argument("intermediate_size", intermediate_size, 512);
    cmd.get_cmd_line_argument("num_expert",        num_expert,        4);
    cmd.get_cmd_line_argument("topk",              topk,              2);
    cmd.get_cmd_line_argument("iterations",        iterations,        1);
    cmd.get_cmd_line_argument("verify",            verify,            1);

    if (topk > num_expert) {
      std::cerr << "Error: topk (" << topk << ") must be <= num_expert ("
                << num_expert << ")\n";
      error = true;
    }
    if (num_token <= 0 || hidden_dim <= 0 || intermediate_size <= 0 ||
        num_expert <= 0 || topk <= 0) {
      std::cerr << "Error: all dimension arguments must be positive.\n";
      error = true;
    }
  }

  std::ostream& print_usage(std::ostream& out) const {
    out
      << "14_bmg_moe_no_reorder_grouped_gemm\n\n"
      << "  MoE FFN pipeline: gather → w13 grouped GEMM → SiLU → w2 grouped GEMM → reduce\n"
      << "  The input hidden_states are NOT reordered; a gather kernel routes tokens to\n"
      << "  per-expert activation buffers, then the standard grouped GEMM path is reused.\n\n"
      << "Options:\n\n"
      << "  --help                           Display this help message\n"
      << "  --num_token=<int>                Number of input tokens            (default: 256)\n"
      << "  --hidden_dim=<int>               Hidden dimension                  (default: 1024)\n"
      << "  --intermediate_size=<int>        Intermediate size (w13 width=2x)  (default: 512)\n"
      << "  --num_expert=<int>               Number of experts                 (default: 4)\n"
      << "  --topk=<int>                     Top-k experts selected per token  (default: 2)\n"
      << "  --iterations=<int>               Benchmark repetitions (0=timing off) (default: 1)\n"
      << "  --verify=<int>                   Correctness check (0=skip, 1=on)  (default: 1)\n\n"
      << "Example:\n"
      << "  $ 14_bmg_moe_no_reorder_grouped_gemm"
         " --num_token=512 --hidden_dim=2048 --intermediate_size=1024"
         " --num_expert=8 --topk=2\n\n";
    return out;
  }

  /// Total GEMM flops for both passes (used for TFLOPS reporting)
  double total_gflops(double runtime_s, int total_tokens) const {
    uint64_t flop =
        2ULL * total_tokens * hidden_dim        * 2 * intermediate_size  // GEMM-1
      + 2ULL * total_tokens * intermediate_size * hidden_dim;             // GEMM-2
    return double(flop) / 1e9 / runtime_s;
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// TiledMMA  (same config as example 12: 256×128 WG tile, 8×2 SG layout)
// ─────────────────────────────────────────────────────────────────────────────
auto make_tiled_mma_bf16() {
  using WGTile   = Shape<_256, _128, _32>;
  using SGLayout = Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>;
  auto op = XE_DPAS_TT<8, float, ElementAct, ElementW13>{};
  using MMA = typename TiledMMAHelper<
      MMA_Atom<decltype(op)>, Layout<WGTile>, SGLayout>::TiledMMA;
  return MMA{};
}

// ─────────────────────────────────────────────────────────────────────────────
// Unique SYCL kernel name tags (prevent name collisions across template
// instantiations from different passes of the same GEMM call)
// ─────────────────────────────────────────────────────────────────────────────
template <int Id>               class GatherKernelName;
template <int Id>               class SiluKernelName;
template <int Id>               class ScatterReduceKernelName;
template <int Id>               class RefSiluKernelName;
template <int Id>               class RefScatterReduceKernelName;
template <int Id, typename EA, typename EB> class MoENoReorderGemmName;

// ─────────────────────────────────────────────────────────────────────────────
// Gather kernel
//   Builds per-expert contiguous activation buffers from hidden_states using
//   the precomputed expert_token_indices mapping.
//
//   expert_acts[global_pos * hidden_dim + feat] =
//       hidden_states[expert_token_indices[global_pos] * hidden_dim + feat]
//
//   Implementation: register-based direct loads — no SLM needed.
// ─────────────────────────────────────────────────────────────────────────────
void gather_tokens(sycl::queue&       q,
                   const ElementAct*  hidden_states,        ///< [num_token, hidden_dim]
                   const int32_t*     expert_token_indices, ///< [total_tokens]
                   ElementAct*        expert_acts,          ///< [total_tokens, hidden_dim]
                   int                total_tokens,
                   int                hidden_dim) {
  q.parallel_for<GatherKernelName<0>>(
      sycl::range<1>(total_tokens * hidden_dim),
      [=](sycl::item<1> it) {
        int i    = static_cast<int>(it.get_id(0));
        int pos  = i / hidden_dim;
        int feat = i % hidden_dim;
        int tok  = expert_token_indices[pos];
        expert_acts[i] = hidden_states[tok * hidden_dim + feat];
      }).wait();
}

// ─────────────────────────────────────────────────────────────────────────────
// SiLU activation kernel  (SwiGLU gating, FP32 compute, BF16 I/O)
//
//   input  : intermediate13 [total_tokens, 2 * intermediate_size]
//   output : silu_out       [total_tokens,     intermediate_size]
//
//   For each row r and feature f in [0, intermediate_size):
//     gate = intermediate13[r, f]
//     up   = intermediate13[r, f + intermediate_size]
//     silu_out[r, f] = (gate / (1 + exp(-gate))) * up   (in FP32, stored as BF16)
// ─────────────────────────────────────────────────────────────────────────────
void apply_silu(sycl::queue&        q,
                const ElementInter* intermediate13,  ///< [total_tokens, 2*intermediate_size]
                ElementInter*       silu_out,         ///< [total_tokens,   intermediate_size]
                int                 total_tokens,
                int                 intermediate_size) {
  const int two_is = 2 * intermediate_size;
  q.parallel_for<SiluKernelName<0>>(
      sycl::range<1>(total_tokens * intermediate_size),
      [=](sycl::item<1> it) {
        int i    = static_cast<int>(it.get_id(0));
        int pos  = i / intermediate_size;
        int feat = i % intermediate_size;
        float gate    = static_cast<float>(intermediate13[pos * two_is + feat]);
        float up      = static_cast<float>(intermediate13[pos * two_is + intermediate_size + feat]);
        float gate_sig = gate / (1.0f + sycl::exp(-gate));
        silu_out[i]   = static_cast<ElementInter>(gate_sig * up);
      }).wait();
}

// ─────────────────────────────────────────────────────────────────────────────
// Scatter-reduce kernel
//   For each output token, sum the contributions from its topk expert outputs.
//   Accumulation in FP32; result cast to BF16.
//
//   topk_position[t * topk + k] = the flat position in expert_outs for the k-th
//   expert of token t (precomputed on host during preprocessing).
// ─────────────────────────────────────────────────────────────────────────────
void scatter_reduce(sycl::queue&        q,
                    const ElementInter* expert_outs,   ///< [total_tokens, hidden_dim]
                    const int32_t*      topk_position, ///< [num_token * topk]
                    ElementOut*         final_out,     ///< [num_token, hidden_dim]
                    int                 num_token,
                    int                 topk,
                    int                 hidden_dim) {
  q.parallel_for<ScatterReduceKernelName<0>>(
      sycl::range<1>(num_token * hidden_dim),
      [=](sycl::item<1> it) {
        int i    = static_cast<int>(it.get_id(0));
        int t    = i / hidden_dim;
        int feat = i % hidden_dim;
        float sum = 0.0f;
        for (int k = 0; k < topk; ++k) {
          int pos = topk_position[t * topk + k];
          sum += static_cast<float>(expert_outs[pos * hidden_dim + feat]);
        }
        final_out[i] = static_cast<ElementOut>(sum);
      }).wait();
}

// ─────────────────────────────────────────────────────────────────────────────
// Grouped GEMM launcher
//   Wraps MoE::MoEGEMM (from example 12) for one GEMM pass of the pipeline.
//   GemmId: compile-time integer tag to give each call a unique kernel name.
// ─────────────────────────────────────────────────────────────────────────────
template <int GemmId, typename ElementA, typename ElementB>
void launch_grouped_gemm(sycl::queue&      q,
                         const ElementA*   acts,      ///< [total_tokens, gemm_k]
                         const ElementB*   weights,   ///< [num_expert * gemm_k * gemm_n]
                         ElementInter*     outputs,   ///< [total_tokens, gemm_n]
                         int               gemm_n,
                         int               gemm_k,
                         const int32_t*    num_rows_per_expert_device,
                         int               num_experts) {
  int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  cutlass::KernelHardwareInfo hw_info{0, sm_count};

  // Dummy problem shape — the actual per-expert M values come from
  // num_rows_per_expert_device at runtime (see PersistentTileSchedulerXeMoE).
  auto dummy_ps = cute::Shape<int,int,int>{1, gemm_k, gemm_n};
  auto dummy_group_ps =
      cutlass::gemm::GroupProblemShape<Shape<int,int,int>>{1, &dummy_ps, nullptr};

  using TileShape    = Shape<_256, _128, _32>;
  using ClusterShape = Shape<_1,   _1,   _1>;

  auto scheduler_params =
      MoE::PersistentTileSchedulerXeMoE<MoE::ProblemShape>::to_underlying_arguments(
          dummy_group_ps, TileShape{}, ClusterShape{}, hw_info,
          MoE::PersistentTileSchedulerXeMoE<MoE::ProblemShape>::Arguments{
              1, MoE::RasterOrderOptions::AlongN});

  auto group_distribution =
      MoE::PersistentTileSchedulerXeMoE<MoE::ProblemShape>::get_grid_shape(
          scheduler_params, dummy_group_ps, TileShape{}, ClusterShape{}, hw_info,
          MoE::PersistentTileSchedulerXeMoE<MoE::ProblemShape>::Arguments{
              1, MoE::RasterOrderOptions::AlongN});

  auto mma  = make_tiled_mma_bf16();
  auto MaxThreadsPerWorkgroup = size(mma);
  dim3 local_range{MaxThreadsPerWorkgroup, 1, 1};

  sycl::range<3> local  = {local_range.z, local_range.y, local_range.x};
  sycl::range<3> groups = {group_distribution.z,
                            group_distribution.y,
                            group_distribution.x};
  sycl::range<3> global = {local[0] * groups[0],
                            local[1] * groups[1],
                            local[2] * groups[2]};

  namespace syclex  = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{
      syclex::sub_group_size<16>,
#if (defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35))
      intelex::grf_size<512>
#else
      intelex::grf_size<256>
#endif
  };

  auto event = q.parallel_for<MoENoReorderGemmName<GemmId, ElementA, ElementB>>(
      sycl::nd_range<3>(global, local), kernel_props,
      [=](auto) {
        MoE::MoEGEMM<XE_LOAD_2D<16, 32, 32, 16>,
                     XE_LOAD_2D_VNNI<16, 32, 16, 16>,
                     XE_STORE_2D<16, 8, 32>,
                     'R', 'R', 'R'>(
            acts, weights,
            static_cast<const void*>(nullptr), // no per-channel scales
            outputs,
            mma,
            num_rows_per_expert_device,
            num_experts,
            gemm_n,
            gemm_k,
            scheduler_params);
      });
  EventManager::getInstance().addEvent(event);
  q.wait_and_throw();
}

// ─────────────────────────────────────────────────────────────────────────────
// MoE pipeline runner
// ─────────────────────────────────────────────────────────────────────────────
struct MoENoReorderRunner {

  const Options& opts;
  sycl::queue    Q;
  int            total_tokens; ///< = num_token * topk

  // ── Preprocessing data (host side) ────────────────────────────────────────
  std::vector<int32_t> tokens_per_expert_h;   ///< [num_expert]
  std::vector<int32_t> expert_offsets_h;      ///< [num_expert + 1]  prefix-sum
  std::vector<int32_t> expert_token_indices_h;///< [total_tokens]  token→expert-buffer index
  std::vector<int32_t> topk_position_h;       ///< [num_token * topk]  (t,k)→flat buffer pos

  // ── Device allocations ────────────────────────────────────────────────────
  cutlass::DeviceAllocation<ElementAct>   hidden_states;
  cutlass::DeviceAllocation<ElementW13>   w13_weights;
  cutlass::DeviceAllocation<ElementW2>    w2_weights;
  cutlass::DeviceAllocation<int32_t>      topk_ids;

  cutlass::DeviceAllocation<int32_t>      d_expert_token_indices;
  cutlass::DeviceAllocation<int32_t>      d_topk_position;
  cutlass::DeviceAllocation<int32_t>      d_tokens_per_expert;

  cutlass::DeviceAllocation<ElementAct>   expert_acts;    ///< [total_tokens, hidden_dim]
  cutlass::DeviceAllocation<ElementInter> intermediate13; ///< [total_tokens, 2*intermediate_size]
  cutlass::DeviceAllocation<ElementInter> silu_out;       ///< [total_tokens, intermediate_size]
  cutlass::DeviceAllocation<ElementInter> expert_outs;    ///< [total_tokens, hidden_dim]
  cutlass::DeviceAllocation<ElementOut>   final_out;      ///< [num_token, hidden_dim]
  cutlass::DeviceAllocation<ElementOut>   final_out_ref;  ///< reference output for verification

  explicit MoENoReorderRunner(const Options& o)
      : opts(o),
        Q(compat::get_default_queue()),
        total_tokens(o.num_token * o.topk) {}

  // ── Allocation ────────────────────────────────────────────────────────────
  void allocate() {
    hidden_states.reset(static_cast<size_t>(opts.num_token) * opts.hidden_dim);
    w13_weights.reset(static_cast<size_t>(opts.num_expert) * opts.hidden_dim
                      * (2 * opts.intermediate_size));
    w2_weights.reset(static_cast<size_t>(opts.num_expert) * opts.intermediate_size
                     * opts.hidden_dim);
    topk_ids.reset(static_cast<size_t>(opts.num_token) * opts.topk);

    d_expert_token_indices.reset(total_tokens);
    d_topk_position.reset(static_cast<size_t>(opts.num_token) * opts.topk);
    d_tokens_per_expert.reset(opts.num_expert);

    expert_acts.reset(static_cast<size_t>(total_tokens) * opts.hidden_dim);
    intermediate13.reset(static_cast<size_t>(total_tokens) * 2 * opts.intermediate_size);
    silu_out.reset(static_cast<size_t>(total_tokens) * opts.intermediate_size);
    expert_outs.reset(static_cast<size_t>(total_tokens) * opts.hidden_dim);
    final_out.reset(static_cast<size_t>(opts.num_token) * opts.hidden_dim);
    final_out_ref.reset(static_cast<size_t>(opts.num_token) * opts.hidden_dim);
  }

  // ── Initialization ────────────────────────────────────────────────────────
  void initialize() {
    constexpr uint64_t seed = 2025;
    initialize_block(hidden_states, seed + 1);
    initialize_block(w13_weights,   seed + 2);
    initialize_block(w2_weights,    seed + 3);

    // Generate topk_ids on host: for each token, pick topk distinct experts
    std::vector<int32_t> topk_ids_h(static_cast<size_t>(opts.num_token) * opts.topk);
    std::mt19937 rng(static_cast<uint32_t>(seed + 4));
    std::vector<int32_t> expert_pool(opts.num_expert);
    std::iota(expert_pool.begin(), expert_pool.end(), 0);
    for (int t = 0; t < opts.num_token; ++t) {
      // Sample topk distinct experts without replacement
      std::shuffle(expert_pool.begin(), expert_pool.end(), rng);
      for (int k = 0; k < opts.topk; ++k)
        topk_ids_h[t * opts.topk + k] = expert_pool[k];
    }
    topk_ids.copy_from_host(topk_ids_h.data());

    // Build routing data structures and copy to device
    preprocess_topk_ids(topk_ids_h);
    d_expert_token_indices.copy_from_host(expert_token_indices_h.data());
    d_topk_position.copy_from_host(topk_position_h.data());
    d_tokens_per_expert.copy_from_host(tokens_per_expert_h.data());
  }

  // ── Routing preprocessing (host) ──────────────────────────────────────────
  /// Compute the three routing tables needed by gather and scatter-reduce.
  void preprocess_topk_ids(const std::vector<int32_t>& topk_ids_h) {
    tokens_per_expert_h.assign(opts.num_expert, 0);
    for (int t = 0; t < opts.num_token; ++t)
      for (int k = 0; k < opts.topk; ++k)
        tokens_per_expert_h[topk_ids_h[t * opts.topk + k]]++;

    // Prefix-sum offsets for each expert's slice in the flat expert_acts buffer
    expert_offsets_h.resize(opts.num_expert + 1, 0);
    for (int e = 0; e < opts.num_expert; ++e)
      expert_offsets_h[e + 1] = expert_offsets_h[e] + tokens_per_expert_h[e];

    // expert_token_indices[pos] = original token index for flat position pos
    // topk_position[t*topk+k]  = flat position in expert_acts for (token t, selection k)
    expert_token_indices_h.resize(total_tokens);
    topk_position_h.resize(static_cast<size_t>(opts.num_token) * opts.topk);
    std::vector<int32_t> counters(opts.num_expert, 0);
    for (int t = 0; t < opts.num_token; ++t) {
      for (int k = 0; k < opts.topk; ++k) {
        int e   = topk_ids_h[t * opts.topk + k];
        int pos = expert_offsets_h[e] + counters[e]++;
        expert_token_indices_h[pos]        = t;
        topk_position_h[t * opts.topk + k] = pos;
      }
    }
  }

  // ── Pipeline ──────────────────────────────────────────────────────────────
  void run_pipeline() {
    // Step 1: Gather tokens from hidden_states to per-expert contiguous buffers
    gather_tokens(Q,
                  hidden_states.get(),
                  d_expert_token_indices.get(),
                  expert_acts.get(),
                  total_tokens,
                  opts.hidden_dim);

    // Step 2: First grouped GEMM  —  expert_acts @ w13_weights → intermediate13
    //   A: [total_tokens, hidden_dim]          (K = hidden_dim)
    //   B: [num_expert * hidden_dim * 2*IS]    (each expert: [hidden_dim, 2*IS] row-major)
    //   D: [total_tokens, 2*intermediate_size]
    launch_grouped_gemm<1>(Q,
                            expert_acts.get(),
                            w13_weights.get(),
                            intermediate13.get(),
                            2 * opts.intermediate_size, // N
                            opts.hidden_dim,             // K
                            d_tokens_per_expert.get(),
                            opts.num_expert);

    // Step 3: SiLU gating (FP32 compute, BF16 I/O)
    apply_silu(Q,
               intermediate13.get(),
               silu_out.get(),
               total_tokens,
               opts.intermediate_size);

    // Step 4: Second grouped GEMM  —  silu_out @ w2_weights → expert_outs
    //   A: [total_tokens, intermediate_size]
    //   B: [num_expert * intermediate_size * hidden_dim]
    //   D: [total_tokens, hidden_dim]
    launch_grouped_gemm<2>(Q,
                            silu_out.get(),
                            w2_weights.get(),
                            expert_outs.get(),
                            opts.hidden_dim,              // N
                            opts.intermediate_size,       // K
                            d_tokens_per_expert.get(),
                            opts.num_expert);

    // Step 5: Scatter-reduce  —  sum topk expert outputs per token, cast to BF16
    scatter_reduce(Q,
                   expert_outs.get(),
                   d_topk_position.get(),
                   final_out.get(),
                   opts.num_token,
                   opts.topk,
                   opts.hidden_dim);
  }

  // ── Reference (verification) ──────────────────────────────────────────────
  /// Runs a reference implementation using cutlass::reference::device::GemmComplex
  /// for each expert, plus the same SiLU and scatter-reduce kernels.
  bool run_reference() {
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    const int two_is = 2 * opts.intermediate_size;

    // Reference intermediate13  (BF16, same type as kernel output)
    cutlass::DeviceAllocation<ElementInter> ref_inter13(
        static_cast<size_t>(total_tokens) * two_is);

    // Reference GEMM-1: per expert
    for (int e = 0; e < opts.num_expert; ++e) {
      int M = tokens_per_expert_h[e];
      if (M == 0) continue;

      int64_t off_acts = static_cast<int64_t>(expert_offsets_h[e]) * opts.hidden_dim;
      int64_t off_out  = static_cast<int64_t>(expert_offsets_h[e]) * two_is;
      int64_t off_w13  = static_cast<int64_t>(e) * opts.hidden_dim * two_is;

      // Temporary C matrix (beta = 0, so content does not matter)
      cutlass::DeviceAllocation<ElementInter> tmp_c(M * two_is);

      cutlass::TensorRef<ElementAct,   LayoutA> ref_A(
          expert_acts.get() + off_acts, LayoutA::packed({M, opts.hidden_dim}));
      cutlass::TensorRef<ElementW13,   LayoutB> ref_B(
          w13_weights.get() + off_w13,  LayoutB::packed({opts.hidden_dim, two_is}));
      cutlass::TensorRef<ElementInter, LayoutC> ref_C(
          tmp_c.get(), LayoutC::packed({M, two_is}));
      cutlass::TensorRef<ElementInter, LayoutD> ref_D(
          ref_inter13.get() + off_out, LayoutD::packed({M, two_is}));

      cutlass::reference::device::GemmComplex(
          {M, two_is, opts.hidden_dim},
          1.0f,
          ref_A, cutlass::ComplexTransform::kNone,
          ref_B, cutlass::ComplexTransform::kNone,
          0.0f, ref_C, ref_D,
          float(0),            // initial accumulator
          1,                   // batch_count
          (int64_t)M * opts.hidden_dim,
          (int64_t)opts.hidden_dim * two_is,
          (int64_t)M * two_is,
          (int64_t)M * two_is);
      compat::wait();
    }

    // Reference SiLU  (reuse same kernel — ensures identical precision)
    cutlass::DeviceAllocation<ElementInter> ref_silu(
        static_cast<size_t>(total_tokens) * opts.intermediate_size);
    {
      const int two_is_local   = two_is;
      const int is_local        = opts.intermediate_size;
      const ElementInter* ri13  = ref_inter13.get();
      ElementInter* rs           = ref_silu.get();
      const int tt               = total_tokens;
      Q.parallel_for<RefSiluKernelName<0>>(
          sycl::range<1>(tt * is_local),
          [=](sycl::item<1> it) {
            int i    = static_cast<int>(it.get_id(0));
            int pos  = i / is_local;
            int feat = i % is_local;
            float gate    = static_cast<float>(ri13[pos * two_is_local + feat]);
            float up      = static_cast<float>(ri13[pos * two_is_local + is_local + feat]);
            float gate_sig = gate / (1.0f + sycl::exp(-gate));
            rs[i]  = static_cast<ElementInter>(gate_sig * up);
          }).wait();
    }

    // Reference GEMM-2: per expert
    cutlass::DeviceAllocation<ElementInter> ref_outs(
        static_cast<size_t>(total_tokens) * opts.hidden_dim);

    for (int e = 0; e < opts.num_expert; ++e) {
      int M = tokens_per_expert_h[e];
      if (M == 0) continue;

      int64_t off_silu = static_cast<int64_t>(expert_offsets_h[e]) * opts.intermediate_size;
      int64_t off_out  = static_cast<int64_t>(expert_offsets_h[e]) * opts.hidden_dim;
      int64_t off_w2   = static_cast<int64_t>(e) * opts.intermediate_size * opts.hidden_dim;

      cutlass::DeviceAllocation<ElementInter> tmp_c(M * opts.hidden_dim);

      cutlass::TensorRef<ElementInter, LayoutA> ref_A(
          ref_silu.get() + off_silu, LayoutA::packed({M, opts.intermediate_size}));
      cutlass::TensorRef<ElementW2,    LayoutB> ref_B(
          w2_weights.get() + off_w2,  LayoutB::packed({opts.intermediate_size, opts.hidden_dim}));
      cutlass::TensorRef<ElementInter, LayoutC> ref_C(
          tmp_c.get(), LayoutC::packed({M, opts.hidden_dim}));
      cutlass::TensorRef<ElementInter, LayoutD> ref_D(
          ref_outs.get() + off_out, LayoutD::packed({M, opts.hidden_dim}));

      cutlass::reference::device::GemmComplex(
          {M, opts.hidden_dim, opts.intermediate_size},
          1.0f,
          ref_A, cutlass::ComplexTransform::kNone,
          ref_B, cutlass::ComplexTransform::kNone,
          0.0f, ref_C, ref_D,
          float(0),
          1,
          (int64_t)M * opts.intermediate_size,
          (int64_t)opts.intermediate_size * opts.hidden_dim,
          (int64_t)M * opts.hidden_dim,
          (int64_t)M * opts.hidden_dim);
      compat::wait();
    }

    // Reference scatter-reduce  (reuse same kernel)
    {
      const ElementInter* ro  = ref_outs.get();
      const int32_t*      tp  = d_topk_position.get();
      ElementOut*         rf  = final_out_ref.get();
      const int nt             = opts.num_token;
      const int tk             = opts.topk;
      const int hd             = opts.hidden_dim;
      Q.parallel_for<RefScatterReduceKernelName<0>>(
          sycl::range<1>(nt * hd),
          [=](sycl::item<1> it) {
            int i    = static_cast<int>(it.get_id(0));
            int t    = i / hd;
            int feat = i % hd;
            float sum = 0.0f;
            for (int k = 0; k < tk; ++k) {
              int pos = tp[t * tk + k];
              sum += static_cast<float>(ro[pos * hd + feat]);
            }
            rf[i] = static_cast<ElementOut>(sum);
          }).wait();
    }

    // Compare kernel output against reference
    // Use relative tolerance appropriate for BF16 arithmetic differences
    const ElementOut epsilon       = ElementOut(0.01f);
    const ElementOut nonzero_floor = ElementOut(1e-4f);
    bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(
        final_out.get(),
        final_out_ref.get(),
        static_cast<size_t>(opts.num_token) * opts.hidden_dim,
        epsilon,
        nonzero_floor);
    return passed;
  }

  // ── Top-level entry point ─────────────────────────────────────────────────
  void run() {
    allocate();
    initialize();

    // Warmup run (also used for verification)
    run_pipeline();

    if (opts.verify) {
      bool passed = run_reference();
      std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << "\n";
      if (!passed) return;
    } else {
      std::cout << "Disposition is skipped.\n";
    }

    if (opts.iterations > 0) {
      GPU_Clock timer;
      timer.start();
      for (int it = 0; it < opts.iterations; ++it)
        run_pipeline();
      compat::wait();
      double elapsed_ms  = timer.seconds() * 1000.0;
      double avg_ms      = elapsed_ms / opts.iterations;
      double gflops      = opts.total_gflops(avg_ms / 1000.0, total_tokens);

      std::cout << "\n"
                << "  num_token        : " << opts.num_token         << "\n"
                << "  hidden_dim       : " << opts.hidden_dim        << "\n"
                << "  intermediate_size: " << opts.intermediate_size << "\n"
                << "  num_expert       : " << opts.num_expert        << "\n"
                << "  topk             : " << opts.topk              << "\n"
                << "  total_tokens     : " << total_tokens           << "\n"
                << "  Avg runtime      : " << avg_ms                 << " ms\n"
                << "  GFLOPS (GEMMs)   : " << gflops                 << "\n";
    }
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, const char** argv) {
  Options opts;
  opts.parse(argc, argv);

  if (opts.help) {
    opts.print_usage(std::cout);
    return 0;
  }
  if (opts.error) {
    std::cerr << "Aborting execution.\n";
    return -1;
  }

  MoENoReorderRunner runner(opts);
  runner.run();
  return 0;
}
