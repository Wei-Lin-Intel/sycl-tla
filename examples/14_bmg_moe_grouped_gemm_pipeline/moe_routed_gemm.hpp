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
// Design — K-sliced gather with SLM staging:
//   Both the A-matrix scratch buffer and the sorted_token_ids index array
//   are kept in Shared Local Memory (SLM) instead of global memory:
//
//   Per M-tile:
//     Load sorted_token_ids for this tile → SLM (1 KB, once per M-tile).
//
//   for each K-tile (stride K_TILE = 32):
//     1. Cooperative gather: each work-item loads its share of the
//        TILE_M × K_TILE slice from scattered rows of HiddenStates into
//        the SLM scratch buffer using SLM-cached token IDs.
//     2. Workgroup barrier (SLM fence).
//     3. Each subgroup loads its portion of the A-tile directly from SLM
//        into MMA register fragments using the identity-tensor coordinate
//        mapping (bypasses 2D block load and reorder). B is loaded from
//        global memory via the standard 2D block load path.
//     4. DPAS accumulate.
//     5. Workgroup barrier protects SLM before the next iteration.
//
//   After all K-tiles, the accumulated result is stored to the output.
//
// SLM budget:
//   - A scratch:        TILE_M × K_TILE × sizeof(ElementA) = 256 × 32 × 2 = 16 KB
//   - Token IDs:        TILE_M × sizeof(int32_t)           = 256 × 4     =  1 KB
//   - Total per WG:     ~17 KB  (well within 64 KB SLM limit)
//
// Benefits:
//   - No global scratch allocation or parameter — SLM is per-workgroup.
//   - Gather target is SLM (guaranteed low latency, no cache pollution).
//   - Token IDs loaded once per M-tile, reused across all K iterations
//     (saves k_tile_count × tile_rows global reads ≈ 23 K reads/WG/tile).
//   - A loaded from SLM directly into MMA registers — no 2D block load,
//     no reorder step.
//   - B-tile prefetching from global memory is preserved.
//
// Parameters:
//   HiddenStates      - [num_tokens, K] original activation matrix (not reordered)
//   sorted_token_ids  - [total_routed_tokens] original token indices, grouped by expert
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
              const ElementB *Weights,
              const ElementS *Scales,
              ElementD *Outputs,
              TiledMMA const &mma,
              const int32_t *M_per_group,
              const int32_t num_experts,
              const int32_t N, const int32_t K,
              const int32_t tile_m,
              PersistentTileSchedulerSm90GroupParams<ProblemShape> scheduler_params) {

  // Compile-time tile dimensions (must match WG tile)
  constexpr int32_t TILE_M = 256;
  constexpr int32_t K_TILE = 32;

  // ---- SLM allocations ----
  // A scratch: TILE_M × K_TILE BF16 = 16 KB
  auto *slm_A = compat::local_mem<ElementA[TILE_M * K_TILE]>();
  // Token ID cache: TILE_M int32 = 1 KB
  auto *slm_ids = compat::local_mem<int32_t[TILE_M]>();

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

  // ---- Pre-compute coordinate mapping for A ----
  // The identity tensor + MMA partitioning tells us which (m, k) coordinate
  // each register position in the MMA A-fragment corresponds to. This mapping
  // is compile-time constant (depends only on the WG tile and MMA layout) and
  // is reused across all M-tiles and K iterations.
  auto wg_tile = mma.tile_mnk(); // (256, 128, 32)
  Tensor cA_id = make_identity_tensor(
      make_shape(Int<TILE_M>{}, Int<K_TILE>{}));
  Tensor gA_id = local_tile(cA_id, select<0, 2>(wg_tile), make_coord(0, _));
  auto thr_mma = mma.get_slice(local_id);
  auto coord_frag_A = thr_mma.partition_sg_fragment_A(gA_id(_, _, 0));

  // Create a properly-typed A data fragment (subgroup tensor, ElementA).
  // We use an SLM-backed tensor for type deduction only — partition_sg_fragment_A
  // allocates new register storage; it does not read from SLM here.
  auto sA_tensor = make_tensor(
      make_smem_ptr(slm_A),
      make_layout(make_shape(Int<TILE_M>{}, Int<K_TILE>{}),
                  make_stride(Int<K_TILE>{}, _1{})));
  Tensor gA_typed = local_tile(sA_tensor, select<0, 2>(wg_tile),
                               make_coord(0, _));

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

    // ---- Load sorted_token_ids for this M-tile into SLM ----
    // Loaded once per M-tile, reused across all K iterations.
    for (int32_t i = local_id; i < TILE_M; i += local_size) {
      if (i < tile_rows) {
        slm_ids[i] = sorted_token_ids[cumulative_M + m_start + i];
      } else {
        slm_ids[i] = 0; // out-of-range rows; gather will produce 0
      }
    }
    item.barrier(sycl::access::fence_space::global_and_local);

    // ---- B and D tensors (unchanged — global memory) ----
    ElementB *ptr_B_curr =
        const_cast<ElementB *>(Weights) + curr_group * K * N;
    auto B_tensor = make_moe_tensor<ElementB, actual_layout_of_B>(
        ptr_B_curr, N, K);

    ElementD *ptr_D_tile = Outputs + (cumulative_M + m_start) * N;
    auto D_tensor = make_moe_tensor<ElementD, LayoutKindD>(
        ptr_D_tile, tile_rows, N);

    Tensor cB = make_identity_tensor(B_tensor.shape());
    Tensor cD = make_identity_tensor(D_tensor.shape());

    Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(n_coord, _));
    auto wg_coord = make_coord(0, n_coord, 0);
    Tensor gD = local_tile(cD, wg_tile, wg_coord, Step<_1, _1, X>{});

    auto tiled_copy_b = get_block_2d_copy_B<GmemTiledCopyB>(mma, B_tensor);
    auto tiled_copy_d = get_block_2d_copy_D<GmemTiledCopyD>(mma, D_tensor);

    auto thr_copy_b = tiled_copy_b.get_slice(local_id);
    auto thr_copy_d = tiled_copy_d.get_slice(local_id);

    // MMA register fragments
    auto tCrA = thr_mma.partition_sg_fragment_A(gA_typed(_, _, 0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
    auto tCrD = thr_mma.partition_sg_fragment_C(gD);
    auto tCrD_final = thr_copy_d.partition_sg_fragment_S(gD);

    // B copy fragments (A loaded from SLM, no copy fragment needed)
    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

    Tensor tBgB = thr_copy_b.partition_S(gB);
    auto tCgD = thr_copy_d.partition_D(gD);

    // B prefetch (A is in SLM — no prefetch needed)
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

    // ---- K-loop with SLM-staged A gather ----
    for (int k = 0; k < k_tile_count; k++, prefetch_k++) {
      int32_t k_offset = k * K_TILE;
      int32_t k_cols = K - k_offset;
      if (k_cols > K_TILE) k_cols = K_TILE;

      // Cooperative gather into SLM using SLM-cached token IDs.
      // One subgroup works on one row at a time; lanes read contiguous cols.
      for (int32_t row = sg_id; row < tile_rows; row += num_sg) {
        int32_t orig_token = slm_ids[row];
        const ElementA *src = HiddenStates + orig_token * K + k_offset;
        ElementA *dst = slm_A + row * K_TILE;

        for (int32_t col = sg_lane; col < k_cols; col += sg_size) {
          dst[col] = src[col];
        }
        for (int32_t col = k_cols + sg_lane; col < K_TILE; col += sg_size) {
          dst[col] = ElementA(0);
        }
      }
      // Zero-fill rows beyond tile_rows (so out-of-range MMA reads get 0).
      for (int32_t row = tile_rows + sg_id; row < TILE_M; row += num_sg) {
        ElementA *dst = slm_A + row * K_TILE;
        for (int32_t col = sg_lane; col < K_TILE; col += sg_size) {
          dst[col] = ElementA(0);
        }
      }

      item.barrier(sycl::access::fence_space::global_and_local);

      barrier_arrive(barrier_scope);

      // ---- Load A from SLM into MMA registers ----
      // Each register position in coord_frag_A tells us the (m, k) coordinate
      // it maps to.  Read that element from the SLM scratch buffer.
      CUTE_UNROLL
      for (int i = 0; i < size(tCrA); i++) {
        int m_idx = static_cast<int>(get<0>(coord_frag_A(i)));
        int k_idx = static_cast<int>(get<1>(coord_frag_A(i)));
        tCrA(i) = slm_A[m_idx * K_TILE + k_idx];
      }

      // Load B via 2D block load (unchanged)
      copy(tiled_copy_b, tBgB(_, _, _, k), tBrB);

      if (prefetch_k < k_tile_count) {
        prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
      }

      // No reorder for A — already in MMA register layout.
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

