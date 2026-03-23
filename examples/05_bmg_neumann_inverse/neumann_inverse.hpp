/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
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

/*!
 * \file neumann_inverse.hpp
 * \brief SYCL kernel for iterative Neumann-style lower-triangular matrix inverse (64×64, BF16).
 *
 * Implements the iterative Neumann inverse for unit lower-triangular 64×64 BF16 matrices,
 * ported from the Triton reference kernel `tril_inverse_iterative_64_bf16_kernel` in
 * `python/opt_tril_inverse_triton.py`.
 *
 * Algorithm (STEPS=14 iterations):
 *   inv = I
 *   for _ in range(STEPS):
 *       err = lower_strict(L * inv)   // strictly lower-triangular part of L*inv
 *       inv = inv - inv * err
 *
 * Kernel design (redesigned for reduced barrier pressure):
 *   - One work-group per 64×64 matrix (one batch element).
 *   - Work-group: 16 sub-groups × 16 threads = 256 threads total.
 *   - Sub-group (sg_row, sg_col) is responsible for one 16×16 tile of the 4×4 tile grid.
 *   - Shared local memory (SLM) holds L, two inv buffers (double-buffered), and err
 *     (4 × 64×64 BF16 = 32 KB).  L is loaded once; inv is double-buffered to eliminate
 *     the read-protect barrier between Phase 2 reads and Phase 2 writes, reducing
 *     barriers from 3 per step (original) to 2 per step.
 *   - DPAS matrix-multiply-accumulate (via CuTe TiledMMA / cute::gemm) is used for both
 *     the L×inv and inv×err products.
 *
 * Execution model (per Neumann step):
 *   Phase 1 – Compute err tile and write to slm_err:
 *     - Upper-tri tiles  (sg_row < sg_col):  entirely skipped (err = 0 by structure).
 *     - Diagonal tiles   (sg_row == sg_col): one k_block = sg_col; apply strict-lower mask.
 *     - Lower-tri tiles  (sg_row > sg_col):  k_block ∈ [sg_col, sg_row]; no mask needed.
 *   BARRIER 1 – ensures all slm_err writes are visible before Phase 2 reads them.
 *   Phase 2 – Compute update and write inv_next = inv_cur - update to the inactive buffer:
 *     - Upper-tri tiles  (sg_row < sg_col):  skipped; inv_next stays at its pre-init value (0).
 *     - Diagonal/lower   (sg_row >= sg_col): k_block ∈ [sg_col, sg_row].
 *   BARRIER 2 – ensures all inv_next writes are visible before the next iteration reads them.
 *   Swap cur ↔ 1-cur.
 *
 * k_block range justification:
 *   - L is lower-triangular: L_tile[sg_row][k] = 0 for k > sg_row.
 *   - inv is lower-triangular: inv_tile[k][sg_col] = 0 for k < sg_col.
 *   - For Phase 1 (err = L×inv), only k ∈ [sg_col, sg_row] can contribute.
 *   - err is strictly lower-triangular at the tile level for k > sg_col; the diagonal
 *     err tile (k = sg_col) is mixed (zero above/on the within-tile diagonal, nonzero below).
 *   - For Phase 2 (update = inv×err), inv_tile[sg_row][k] = 0 for k > sg_row and
 *     err_tile[k][sg_col] is zero for k < sg_col; hence the same range [sg_col, sg_row].
 *
 * DPAS register layout notes (XE_8x16x16_F32BF16BF16F32_TT, 16-thread sub-group):
 *   ALayout = Layout<Shape<16, 8>, Stride<8, 1>>:
 *     Thread t holds A[t//2][(t%2)*8 .. (t%2)*8 + 7]  (8 values, one half-row of the atom).
 *   BLayout = Layout<Shape<16, 16>, Stride<1, 16>>:
 *     Thread t holds B[k=0..15][t]  (16 values, full K-column for output column t).
 *   CLayout = Layout<Shape<16, 8>, Stride<8, 1>>:
 *     Thread t holds C[t//2][(t%2)*8 .. (t%2)*8 + 7]  (same pattern as A).
 *   With WG tile 16×16×16 (2 DPAS atoms stacked in M):
 *     tCrA[0..7]  = atom 0 (M rows 0..7)
 *     tCrA[8..15] = atom 1 (M rows 8..15)
 *     (same for tCrC)
 *
 * Precision note:
 *   BF16 inputs and outputs.  Accumulators are float (F32) within each DPAS call; results are
 *   converted back to BF16 for SLM storage, matching the Triton reference semantics.
 */

#pragma once

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>

#include <cute/tensor.hpp>

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace neumann_inverse {

using namespace cute;

// ---------------------------------------------------------------------------
// Fixed constants
// ---------------------------------------------------------------------------
static constexpr int kN = 64;   // Matrix dimension (fixed by algorithm / SLM budget)

// ---------------------------------------------------------------------------
// Default tuning knobs (compile-time; override via template arguments below).
//
//   kBlock  – DPAS output-tile side length.  Must equal the N and K dimensions
//             of the chosen MMA atom.  For XE_8x16x16_F32BF16BF16F32_TT the
//             atom is 8×16×16, so kBlock = 16 is the only valid choice.
//   kTiles  – Number of kBlock-sized tiles per matrix dimension (kN / kBlock).
//             With kN=64 and kBlock=16 this is 4.  Changing kBlock requires a
//             matching change here.
//   kSGSize – Threads per sub-group.  The XE_8x16x16 DPAS instruction operates
//             on SIMD-16 sub-groups, so kSGSize = 16 is the only valid value
//             for this atom.
//
// Derived constants:
//   kSGs    = kTiles × kTiles      (sub-groups per work-group)
//   kWGSize = kSGs   × kSGSize     (threads per work-group)
//   kSteps  – Neumann iteration count.  Exposed as a runtime argument to
//             neumann_inverse_kernel so it can be tuned without recompilation.
//             14 is the default; fewer iterations run faster but reduce accuracy.
// ---------------------------------------------------------------------------
static constexpr int kBlock  = 16;
static constexpr int kTiles  = 4;
static constexpr int kSGSize = 16;
static constexpr int kSGs    = kTiles * kTiles;
static constexpr int kWGSize = kSGs * kSGSize;
static constexpr int kSteps  = 14;

// ---------------------------------------------------------------------------
// neumann_inverse_kernel
//
// Each work-group inverts one 64×64 unit lower-triangular BF16 matrix.
// L_base : input   – batch of B matrices, shape [B, 64, 64] row-major
// Out_base: output – batch of B inverse matrices, shape [B, 64, 64] row-major
// steps  : number of Neumann iterations (default kSteps = 14)
//
// Template parameters:
//   TBlock  – DPAS tile side length (must match MMA atom; default 16).
//   TTiles  – Tiles per matrix dimension, i.e. kN/TBlock (default 4).
//   TSGSize – Threads per sub-group (must match MMA atom; default 16).
// ---------------------------------------------------------------------------
template <typename T,
          int TBlock  = kBlock,
          int TTiles  = kTiles,
          int TSGSize = kSGSize>
CUTE_DEVICE void
neumann_inverse_kernel(const T* L_base, T* Out_base, int batch_size, int steps = kSteps)
{
  // Derived compile-time constants from template parameters.
  constexpr int TSGs    = TTiles * TTiles;   // sub-groups per work-group
  constexpr int TWGSize = TSGs * TSGSize;    // threads per work-group

  // TiledMMA for one TBlock×TBlock DPAS tile.
  //   MMA atom : XE_8x16x16_F32BF16BF16F32_TT (8M×16N×16K DPAS, 16 threads)
  //   WG tile  : TBlock×TBlock×TBlock → 2 DPAS atoms stacked in M (for TBlock=16)
  //   Sub-group layout: (1,1,1) → exactly 1 sub-group (TSGSize threads) per tile
  using TileMMA = typename TiledMMAHelper<
      MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
      Layout<Shape<Int<TBlock>, Int<TBlock>, Int<TBlock>>>,
      Layout<Shape<_1, _1, _1>, Stride<_1, _1, _0>>>::TiledMMA;
  auto item     = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int  batch_id = item.get_group(2);      // one work-group per matrix in batch
  if (batch_id >= batch_size) return;

  auto wg       = item.get_group();
  auto sg        = item.get_sub_group();
  int  sg_id     = static_cast<int>(sg.get_group_linear_id()); // 0..TSGs-1
  int  sg_row    = sg_id / TTiles;         // tile row   (0..TTiles-1)
  int  sg_col    = sg_id % TTiles;         // tile column (0..TTiles-1)
  int  local_id  = static_cast<int>(sg.get_local_linear_id()); // 0..TSGSize-1 within sub-group
  int  global_tid = sg_id * TSGSize + local_id; // 0..TWGSize-1 within work-group

  // ---------------------------------------------------------------------------
  // SLM layout (row-major BF16, stride = kN = 64):
  //   slm_L      [64][64] – copy of L, loaded once                    (8 KB)
  //   slm_inv[0] [64][64] – double-buffered inverse (buffer A)        (8 KB)
  //   slm_inv[1] [64][64] – double-buffered inverse (buffer B)        (8 KB)
  //   slm_err    [64][64] – error term (computed each iteration)      (8 KB)
  //   Total: 4 × 64 × 64 × 2 bytes = 32 768 bytes ≈ 32 KB
  //
  // Double-buffering inv eliminates the read-protect barrier between Phase 2's
  // reads of the current inv and its writes of the next inv.  The active ("cur")
  // buffer is read-only; the inactive ("1-cur") buffer is write-only.
  // ---------------------------------------------------------------------------
  auto slm_raw    = compat::local_mem<T[4 * kN * kN]>();
  T* slm_L        = slm_raw;
  T* slm_inv_bufs[2] = { slm_raw + kN * kN, slm_raw + 2 * kN * kN };
  T* slm_err      = slm_raw + 3 * kN * kN;

  const T* L_ptr   = L_base   + static_cast<int64_t>(batch_id) * kN * kN;
  T*       Out_ptr = Out_base  + static_cast<int64_t>(batch_id) * kN * kN;

  // Load L and initialize BOTH inv buffers to the identity matrix.
  // Upper-tri positions stay 0 and diagonal tiles stay at I_16 throughout
  // (Phase 2 only writes lower-tri and diagonal tiles each step).
  for (int i = global_tid; i < kN * kN; i += TWGSize) {
    slm_L[i] = L_ptr[i];
    int row   = i / kN;
    int col   = i % kN;
    T val     = (row == col) ? T(1.0f) : T(0.0f);
    slm_inv_bufs[0][i] = val;
    slm_inv_bufs[1][i] = val;
  }
  sycl::group_barrier(wg);

  // ---------------------------------------------------------------------------
  // Set up TiledMMA and thread-level coordinate fragments.
  // Each sub-group handles one TBlock×TBlock output tile using TSGSize threads.
  // ---------------------------------------------------------------------------
  TileMMA mma{};
  auto wg_tile  = mma.tile_mnk();            // (TBlock, TBlock, TBlock)
  auto thr_mma  = mma.get_slice(local_id);

  // Coordinate/identity tensors (shapes only; used to allocate register fragments).
  auto A_shape  = make_shape(Int<TBlock>{}, Int<TBlock>{});
  Tensor cA     = make_identity_tensor(A_shape);
  Tensor cB     = make_identity_tensor(A_shape);
  Tensor cC     = make_identity_tensor(A_shape);
  Tensor gA     = local_tile(cA, select<0,2>(wg_tile), make_coord(0, _));
  Tensor gB     = local_tile(cB, select<1,2>(wg_tile), make_coord(0, _));
  Tensor gC     = local_tile(cC, wg_tile, make_coord(0, 0, 0), Step<_1, _1, X>{});

  // Register fragments (allocated in hardware registers).
  auto tCrA  = thr_mma.partition_sg_fragment_A(gA(_, _, 0));  // 16 BF16 per thread
  auto tCrB  = thr_mma.partition_sg_fragment_B(gB(_, _, 0));  // 16 BF16 per thread
  auto tCrC  = thr_mma.partition_sg_fragment_C(gC);            // 16 float per thread

  // Per-thread DPAS element mapping (from ALayout / CLayout analysis):
  //   m_dpas  : M row within atom (0..7); same thread → same M row across K
  //   k_half  : starting K index for this thread's 8-element A chunk (0 or 8)
  //   n_half  : starting N index for this thread's 8-element C chunk (0 or 8)
  int m_dpas = local_id / 2;         // 0..7
  int k_half = (local_id % 2) * 8;   // 0 or 8
  int n_half = (local_id % 2) * 8;   // 0 or 8  (same partitioning for C)

  // Double-buffer index.  cur is the read buffer; 1-cur is the write buffer.
  int cur = 0;

  // ---------------------------------------------------------------------------
  // Neumann iteration loop  (runtime `steps` iterations, default kSteps = 14)
  // ---------------------------------------------------------------------------
  for (int step = 0; step < steps; ++step) {

    T* inv_cur  = slm_inv_bufs[cur];
    T* inv_next = slm_inv_bufs[1 - cur];

    // =========================================================================
    // Phase 1 – Compute err tile for this sub-group and write to slm_err.
    //
    // Structural zeros let us skip k_block iterations and entire tiles:
    //   • L is lower-triangular:  L_tile[sg_row][k] = 0 for k > sg_row.
    //   • inv is lower-triangular: inv_tile[k][sg_col] = 0 for k < sg_col.
    //   ⟹ Only k_block ∈ [sg_col, sg_row] can contribute.
    //   • Upper-tri tiles (sg_row < sg_col): empty range → err = 0, skip.
    // =========================================================================
    if (sg_row >= sg_col) {
      clear(tCrC);

      // Accumulate over contributing k_blocks only.
      CUTE_UNROLL
      for (int k_block = 0; k_block < TTiles; ++k_block) {
        if (k_block < sg_col || k_block > sg_row) continue; // structural zero

        // Load A tile: L[sg_row*TBlock..+TBlock][k_block*TBlock..+TBlock].
        CUTE_UNROLL
        for (int v = 0; v < 8; ++v) {
          tCrA(v)   = slm_L[(sg_row * TBlock + m_dpas)     * kN + k_block * TBlock + k_half + v];
          tCrA(8+v) = slm_L[(sg_row * TBlock + 8 + m_dpas) * kN + k_block * TBlock + k_half + v];
        }

        // Load B tile: inv_cur[k_block*TBlock..+TBlock][sg_col*TBlock..+TBlock].
        CUTE_UNROLL
        for (int v = 0; v < 16; ++v) {
          tCrB(v) = inv_cur[(k_block * TBlock + v) * kN + sg_col * TBlock + local_id];
        }

        cute::gemm(mma, tCrA, tCrB, tCrC);
      }

      // Apply strictly-lower-triangular mask.
      // Diagonal tile (sg_row == sg_col): zero elements where row_local ≤ col_local.
      // Lower-tri tile (sg_row > sg_col): entire tile is valid err, no masking needed.
      if (sg_row == sg_col) {
        CUTE_UNROLL
        for (int v = 0; v < 8; ++v) {
          if (m_dpas     <= n_half + v) tCrC(v)   = 0.0f;
          if (8 + m_dpas <= n_half + v) tCrC(8+v) = 0.0f;
        }
      }

      // Write err tile to SLM (float → BF16).
      CUTE_UNROLL
      for (int v = 0; v < 8; ++v) {
        slm_err[(sg_row * TBlock + m_dpas)     * kN + sg_col * TBlock + n_half + v] = T(tCrC(v));
        slm_err[(sg_row * TBlock + 8 + m_dpas) * kN + sg_col * TBlock + n_half + v] = T(tCrC(8+v));
      }
    }
    // Upper-tri tiles (sg_row < sg_col): err = 0.
    // Phase 2 only reads err[k][sg_col] for k ∈ [sg_col, sg_row]; those positions
    // are only written by lower-tri/diagonal SGs, so stale upper-tri values are
    // never read.

    // Barrier 1: all slm_err writes must complete before Phase 2 reads them.
    sycl::group_barrier(wg);

    // =========================================================================
    // Phase 2 – Compute update = inv_cur × err and write inv_next = inv_cur − update.
    //
    // Same k_block range [sg_col, sg_row] as Phase 1 (symmetric structural zeros):
    //   • inv_cur is lower-triangular: inv_tile[sg_row][k] = 0 for k > sg_row.
    //   • err_tile[k][sg_col] is in the strictly-lower block region for k > sg_col;
    //     the diagonal err tile (k = sg_col) is mixed but has the same zero pattern
    //     as the strictly-lower within-block mask → reading it is correct.
    //   ⟹ Only k_block ∈ [sg_col, sg_row] contributes.
    //   • Upper-tri tiles (sg_row < sg_col): update = 0, inv_next = inv_cur = 0.
    //     These positions were pre-initialised to 0 in both inv buffers and are
    //     never written, so inv_next already holds the correct value; skip.
    //
    // No read-protect barrier is needed before the writes here because inv_next
    // is a DIFFERENT SLM region from inv_cur (double-buffering).
    // =========================================================================
    if (sg_row >= sg_col) {
      clear(tCrC);

      CUTE_UNROLL
      for (int k_block = 0; k_block < TTiles; ++k_block) {
        if (k_block < sg_col || k_block > sg_row) continue; // structural zero

        // Load A tile: inv_cur[sg_row*TBlock..+TBlock][k_block*TBlock..+TBlock].
        CUTE_UNROLL
        for (int v = 0; v < 8; ++v) {
          tCrA(v)   = inv_cur[(sg_row * TBlock + m_dpas)     * kN + k_block * TBlock + k_half + v];
          tCrA(8+v) = inv_cur[(sg_row * TBlock + 8 + m_dpas) * kN + k_block * TBlock + k_half + v];
        }

        // Load B tile: err[k_block*TBlock..+TBlock][sg_col*TBlock..+TBlock].
        // The diagonal err tile (k_block == sg_col) has zeros above/on the
        // within-tile diagonal; DPAS naturally handles them as zero accumulation.
        CUTE_UNROLL
        for (int v = 0; v < 16; ++v) {
          tCrB(v) = slm_err[(k_block * TBlock + v) * kN + sg_col * TBlock + local_id];
        }

        cute::gemm(mma, tCrA, tCrB, tCrC);
      }

      // Write inv_next = inv_cur − update to the inactive buffer.
      // Different sub-groups write to non-overlapping tiles → no write conflict.
      CUTE_UNROLL
      for (int v = 0; v < 8; ++v) {
        int idx0 = (sg_row * TBlock + m_dpas)     * kN + sg_col * TBlock + n_half + v;
        int idx1 = (sg_row * TBlock + 8 + m_dpas) * kN + sg_col * TBlock + n_half + v;
        inv_next[idx0] = T(float(inv_cur[idx0]) - tCrC(v));
        inv_next[idx1] = T(float(inv_cur[idx1]) - tCrC(8+v));
      }
    }
    // Upper-tri tiles: inv_next[upper-tri] stays 0 (pre-initialized, never touched).

    // Barrier 2: all inv_next writes must complete before the next iteration
    // reads inv_next as the new inv_cur.
    sycl::group_barrier(wg);

    // Swap double buffers.
    cur ^= 1;
  } // end Neumann loop

  // ---------------------------------------------------------------------------
  // Write final inverse from the active SLM buffer to global memory.
  // ---------------------------------------------------------------------------
  T* final_inv = slm_inv_bufs[cur];
  for (int i = global_tid; i < kN * kN; i += TWGSize) {
    Out_ptr[i] = final_inv[i];
  }
}

} // namespace neumann_inverse
