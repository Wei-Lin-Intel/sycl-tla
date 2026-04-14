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
// Design — K-sliced gather:
//   Instead of gathering the full TILE_M × K block before running the GEMM,
//   this kernel interleaves the gather with the GEMM's K-loop:
//
//   for each K-tile (stride K_TILE = 32):
//     1. Cooperative gather: each work-item loads its share of the
//        TILE_M × K_TILE slice from scattered rows of HiddenStates into a
//        per-workgroup scratch region (only TILE_M × K_TILE per workgroup).
//     2. Workgroup barrier (memory fence).
//     3. 2D block load the gathered A-tile from scratch, 2D block load the
//        corresponding B K-tile from the expert weight matrix, DPAS
//        accumulate.
//     4. Named barrier ensures all subgroups are done reading scratch
//        before the next iteration overwrites it.
//
//   After all K-tiles, the accumulated result is stored to the output.
//
// Benefits over full-tile gather:
//   - Scratch per workgroup: TILE_M × K_TILE × sizeof(ElementA)
//     (e.g. 256 × 32 × 2 = 16 KB, fits in L1) instead of TILE_M × K
//     (e.g. 256 × 2880 × 2 ≈ 1.4 MB, thrashes caches).
//   - The gather per K-tile is 32 columns at a time — a single cache line
//     read per row — giving much better memory access efficiency.
//   - K_TILE=32 is a power of 2, so row/col from a linear index are cheap
//     bitwise operations (shift + AND) instead of expensive integer division.
//   - B-tile prefetching is preserved across K iterations.
//   - Hidden-state rows partially warm in cache across K iterations since
//     the same set of rows (same M-tile) is accessed with advancing columns.
//
// Parameters:
//   HiddenStates      - [num_tokens, K] original activation matrix (not reordered)
//   sorted_token_ids  - [total_routed_tokens] original token indices, grouped by expert
//   scratch_A         - pre-allocated global scratch, sized total_workgroups * TILE_M * K_TILE
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

  constexpr int32_t K_TILE = 32;

  uint64_t wg_linear_id = uint64_t(BlockIdxX()) +
                          uint64_t(BlockIdxY()) * uint64_t(GridDimX()) +
                          uint64_t(BlockIdxZ()) * uint64_t(GridDimX()) * uint64_t(GridDimY());

  int32_t scratch_stride = tile_m * K_TILE;
  ElementA *my_scratch = scratch_A + wg_linear_id * scratch_stride;

  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto local_id = static_cast<int32_t>(item.get_local_linear_id());
  auto local_size = static_cast<int32_t>(item.get_local_range().size());

  auto sg = item.get_sub_group();
  int32_t sg_lane = static_cast<int32_t>(sg.get_local_linear_id());
  int32_t sg_size = static_cast<int32_t>(sg.get_local_range().size());
  int32_t sg_id = local_id / sg_size;
  int32_t num_sg = local_size / sg_size;

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
      for (int i = prev_group; i < curr_group; i++) {
        cumulative_M += M_per_group[i];
      }
      prev_group = curr_group;
      did_group_change = false;
    }

    int32_t m_start = m_coord * tile_m;
    int32_t tile_rows = M - m_start;
    if (tile_rows > tile_m) tile_rows = tile_m;
    int32_t k_tile_count = (K + K_TILE - 1) / K_TILE;

    auto A_tensor = make_moe_tensor<ElementA, LayoutKindA>(
        my_scratch, tile_rows, K_TILE);

    ElementB *ptr_B_curr =
        const_cast<ElementB *>(Weights) + curr_group * K * N;
    auto B_tensor = make_moe_tensor<ElementB, actual_layout_of_B>(
        ptr_B_curr, N, K);

    ElementD *ptr_D_tile = Outputs + (cumulative_M + m_start) * N;
    auto D_tensor = make_moe_tensor<ElementD, LayoutKindD>(
        ptr_D_tile, tile_rows, N);

    Tensor cA = make_identity_tensor(A_tensor.shape());
    Tensor cB = make_identity_tensor(B_tensor.shape());
    Tensor cD = make_identity_tensor(D_tensor.shape());

    auto wg_tile = mma.tile_mnk(); // (256, 128, 32)
    Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(0, _));
    Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(n_coord, _));
    auto wg_coord = make_coord(0, n_coord, 0);
    Tensor gD = local_tile(cD, wg_tile, wg_coord, Step<_1, _1, X>{});

    auto thr_mma = mma.get_slice(local_id);

    auto tiled_copy_a = get_block_2d_copy_A<GmemTiledCopyA>(mma, A_tensor);
    auto tiled_copy_b = get_block_2d_copy_B<GmemTiledCopyB>(mma, B_tensor);
    auto tiled_copy_d = get_block_2d_copy_D<GmemTiledCopyD>(mma, D_tensor);

    auto thr_copy_a = tiled_copy_a.get_slice(local_id);
    auto thr_copy_b = tiled_copy_b.get_slice(local_id);
    auto thr_copy_d = tiled_copy_d.get_slice(local_id);

    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
    auto tCrD = thr_mma.partition_sg_fragment_C(gD);
    auto tCrD_final = thr_copy_d.partition_sg_fragment_S(gD);

    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tBgB = thr_copy_b.partition_S(gB);
    auto tCgD = thr_copy_d.partition_D(gD);

    auto prefetch_b = make_block_2d_prefetch(tiled_copy_b);
    auto thr_prefetch_B = prefetch_b.get_slice(local_id);
    auto pBgB = thr_prefetch_B.partition_S(gB);

    constexpr int barrier_scope = 2;
    const int prefetch_dist = 3;

    int prefetch_k = 0;
    CUTE_UNROLL
    for (; prefetch_k < prefetch_dist && prefetch_k < k_tile_count;
         prefetch_k++) {
      prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
    }

    for (int k = 0; k < k_tile_count; k++, prefetch_k++) {
      int32_t k_offset = k * K_TILE;
      int32_t k_cols = K - k_offset;
      if (k_cols > K_TILE) k_cols = K_TILE;

      // Row-wise cooperative gather:
      // one subgroup works on one row at a time so lanes read contiguous cols.
      for (int32_t row = sg_id; row < tile_rows; row += num_sg) {
        int32_t orig_token = sorted_token_ids[cumulative_M + m_start + row];
        const ElementA *src = HiddenStates + orig_token * K + k_offset;
        ElementA *dst = my_scratch + row * K_TILE;

        for (int32_t col = sg_lane; col < k_cols; col += sg_size) {
          dst[col] = src[col];
        }

        // Zero-fill tail for partial K tile.
        for (int32_t col = k_cols + sg_lane; col < K_TILE; col += sg_size) {
          dst[col] = ElementA(0);
        }
      }

      item.barrier(sycl::access::fence_space::global_and_local);

      barrier_arrive(barrier_scope);

      copy(tiled_copy_a, tAgA(_, _, _, 0), tArA);
      copy(tiled_copy_b, tBgB(_, _, _, k), tBrB);

      if (prefetch_k < k_tile_count) {
        prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
      }

      reorder(tArA, tCrA);
      reorder(tBrB, tCrB);

      cute::gemm(mma, tCrA, tCrB, tCrD);

      barrier_wait(barrier_scope);
    }

    reorder(tCrD, tCrD_final);
    copy(tiled_copy_d, tCrD_final, tCgD);

    work_tile_info = scheduler.fetch_next_work(work_tile_info);
    did_group_change = work_tile_info.is_valid() &&
                       (curr_group != work_tile_info.L_idx);
  }
}

} // namespace MoE

