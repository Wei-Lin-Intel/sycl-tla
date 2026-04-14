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
#pragma once

// Include moe_grouped_gemm.hpp which transitively includes moe_gemms.hpp
// (the moe_gemm function) and moe_tile_scheduler.hpp. Using this single
// include avoids double-including moe_gemms.hpp which lacks a header guard.
#include "../12_xe20_moe_gemm_cute_interface/moe_grouped_gemm.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace MoE {
using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// MoEGEMMRouted: Grouped GEMM variant that fetches A-matrix rows on the fly
// from the original hidden_states buffer using an index array
// (sorted_token_ids), avoiding a separate gather kernel.
//
// Design:
//   Each workgroup is assigned a per-workgroup scratch region in global
//   memory (scratch_A). For every M-tile the persistent scheduler assigns,
//   the workgroup cooperatively gathers the needed rows from HiddenStates
//   (using sorted_token_ids to map logical row → original token index) into
//   its scratch region. A workgroup barrier ensures the gather is complete,
//   then the standard moe_gemm inner loop operates on the now-contiguous
//   scratch buffer via 2D block loads. A second barrier after the GEMM
//   protects the scratch buffer before the next tile's gather.
//
// Parameters:
//   HiddenStates      - [num_tokens, K] original activation matrix (not reordered)
//   sorted_token_ids  - [total_routed_tokens] original token indices, grouped by expert
//   scratch_A         - pre-allocated global scratch, sized total_workgroups * tile_m * K
//   Weights           - [num_experts * K * N] weight matrices
//   Scales            - unused (pass nullptr)
//   Outputs           - [total_routed_tokens * N] output buffer (contiguous per expert)
//   mma               - CuTe TiledMMA descriptor
//   M_per_group       - [num_experts] tokens routed to each expert (device)
//   num_experts       - number of experts
//   N, K              - GEMM N and K dimensions
//   tile_m            - M-tile size (must match WG tile, typically 256)
//   scheduler_params  - persistent tile scheduler parameters
///////////////////////////////////////////////////////////////////////////////
template <class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyD,
          char LayoutKindA, char LayoutKindB, char LayoutKindD, class TiledMMA,
          typename ElementA, typename ElementB, typename ElementS,
          typename ElementD>
CUTE_DEVICE void
MoEGEMMRouted(const ElementA *HiddenStates,
              const int32_t *sorted_token_ids,
              ElementA *scratch_A,
              const ElementB *Weights,
              const ElementS *Scales,
              ElementD *Outputs,
              TiledMMA const &mma,
              const int32_t *M_per_group,
              const int32_t num_experts,
              const int32_t N, const int32_t K,
              const int32_t tile_m,
              PersistentTileSchedulerSm90GroupParams<ProblemShape> scheduler_params) {

  // Compute workgroup linear ID for scratch buffer indexing
  uint64_t wg_linear_id = uint64_t(BlockIdxX()) +
                           uint64_t(BlockIdxY()) * uint64_t(GridDimX()) +
                           uint64_t(BlockIdxZ()) * uint64_t(GridDimX()) * uint64_t(GridDimY());
  int32_t scratch_stride = tile_m * K;
  ElementA *my_scratch = scratch_A + wg_linear_id * scratch_stride;

  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto local_id = static_cast<int32_t>(item.get_local_linear_id());
  auto local_size = static_cast<int32_t>(item.get_local_range().size());

  TileScheduler scheduler{scheduler_params, const_cast<int32_t *>(M_per_group),
                          N, K, num_experts};

  auto work_tile_info = scheduler.initial_work_tile_info(Shape<_1, _1, _1>{});
  constexpr char actual_layout_of_B = LayoutKindB ^ ('R' ^ 'C');
  bool did_group_change = true;
  int32_t curr_group = 0;
  int32_t prev_group = 0;
  int32_t cumulative_M = 0;
  int32_t M = 0;

  if (work_tile_info.is_valid()) {
    curr_group = work_tile_info.L_idx;
    M = M_per_group[curr_group];
  }

  while (work_tile_info.is_valid()) {
    auto m_coord = work_tile_info.M_idx;
    auto n_coord = work_tile_info.N_idx;

    if (did_group_change) {
      curr_group = work_tile_info.L_idx;
      M = M_per_group[curr_group];
      // Recompute cumulative offset (groups visited in non-decreasing order)
      for (int i = prev_group; i < curr_group; i++) {
        cumulative_M += M_per_group[i];
      }
      prev_group = curr_group;
      did_group_change = false;
    }

    // Compute tile boundaries
    int32_t m_start = m_coord * tile_m;
    int32_t tile_rows = M - m_start;
    if (tile_rows > tile_m) tile_rows = tile_m;

    // --- Per-tile gather: cooperatively fetch rows from HiddenStates ---
    // Each work-item loads a portion of the tile's A data from scattered
    // locations in HiddenStates into the contiguous scratch buffer.
    int32_t total_elements = tile_rows * K;
    for (int32_t elem = local_id; elem < total_elements; elem += local_size) {
      int32_t row = elem / K;
      int32_t col = elem % K;
      int32_t orig_token = sorted_token_ids[cumulative_M + m_start + row];
      my_scratch[row * K + col] = HiddenStates[orig_token * K + col];
    }

    // Synchronize workgroup: ensure all rows are gathered before GEMM reads
    item.barrier(sycl::access::fence_space::global_and_local);

    // Create A tensor from the scratch buffer (now contiguous)
    auto A_tensor = make_moe_tensor<ElementA, LayoutKindA>(my_scratch, tile_rows, K);

    // B tensor for this expert (same as standard MoEGEMM)
    ElementB *ptr_B_curr = const_cast<ElementB *>(Weights) + curr_group * K * N;
    auto B_tensor = make_moe_tensor<ElementB, actual_layout_of_B>(ptr_B_curr, N, K);

    // D tensor: output rows are contiguous per expert in sorted order
    ElementD *ptr_D_tile = Outputs + (cumulative_M + m_start) * N;
    auto D_tensor = make_moe_tensor<ElementD, LayoutKindD>(ptr_D_tile, tile_rows, N);

    // Run GEMM on the gathered tile. m_coord=0 because A is already tile-local.
    auto tile_coord = make_coord(0, n_coord, _, 0);
    moe_gemm<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD>(
        A_tensor, B_tensor, D_tensor, tile_coord, mma);

    // Synchronize workgroup: ensure GEMM reads from scratch are done before
    // the next iteration overwrites the scratch buffer.
    item.barrier(sycl::access::fence_space::global_and_local);

    // Get next work tile
    work_tile_info = scheduler.fetch_next_work(work_tile_info);
    did_group_change = curr_group != work_tile_info.L_idx;
  } // end while loop
}

} // namespace MoE
