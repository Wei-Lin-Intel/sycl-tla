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
 * Kernel design:
 *   - One work-group per 64×64 matrix (one batch element).
 *   - Work-group: 16 sub-groups × 16 threads = 256 threads total.
 *   - Sub-group (sg_row, sg_col) is responsible for one 16×16 tile of the 4×4 tile grid.
 *   - Shared local memory (SLM) holds L, inv, and err (3 × 64×64 BF16 = 24 KB).
 *     L is loaded once from global memory; inv and err are kept in SLM across iterations,
 *     avoiding global-memory round-trips for intermediate state.
 *   - DPAS matrix-multiply-accumulate (via CuTe TiledMMA / cute::gemm) is used for both
 *     the L×inv and inv×err products.  Naive scalar triple loops are not used for these
 *     core products.
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
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

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
  //   slm_L  [64][64] – copy of L, loaded once
  //   slm_inv[64][64] – current inverse estimate (updated each iteration)
  //   slm_err[64][64] – error term (computed each iteration)
  //   Total: 3 × 64 × 64 × 2 bytes = 24 576 bytes ≈ 24 KB
  // ---------------------------------------------------------------------------
  auto slm_raw = compat::local_mem<T[3 * kN * kN]>();
  T* slm_L   = slm_raw;
  T* slm_inv = slm_raw + kN * kN;
  T* slm_err = slm_raw + 2 * kN * kN;

  const T* L_ptr   = L_base   + static_cast<int64_t>(batch_id) * kN * kN;
  T*       Out_ptr = Out_base  + static_cast<int64_t>(batch_id) * kN * kN;

  for (int i = global_tid; i < kN * kN; i += TWGSize) {
    slm_L[i] = L_ptr[i];
    int row   = i / kN;
    int col   = i % kN;
    slm_inv[i] = (row == col) ? T(1.0f) : T(0.0f);
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

  // ---------------------------------------------------------------------------
  // Neumann iteration loop  (runtime `steps` iterations, default kSteps = 14)
  //
  // Structural optimizations vs. the naive port:
  //
  //   (A) Skip Phase 1 GEMM for upper-triangle SGs (sg_row < sg_col).
  //       L and inv are both lower-triangular, so (L×inv)[upper] = 0 exactly.
  //       The GEMM result would be zeroed anyway; skip the DPAS calls entirely.
  //       These tiles are also never written to slm_err.
  //
  //   (B) Skip Phase 2 entirely for upper-triangle SGs (sg_row < sg_col).
  //       Since inv is lower-triangular and err is strictly lower-triangular,
  //       (inv×err)[upper] = 0 exactly.  inv[upper] = 0 is invariant and never
  //       needs updating.
  //
  //   (C) Start Phase 2 k_block at sg_col, not 0.
  //       err[k_block][sg_col] = 0 for k_block < sg_col (upper-triangle tiles
  //       of err, which are never written by Phase 1 with optimization A).
  //       Skipping those k_block iterations avoids 0×anything DPAS calls.
  //
  // Together (A)+(B)+(C) reduce GEMM calls per step from 128 to 70 (~45%).
  //   Phase 1: 10 lower/diag SGs × TTiles k_blocks = 40  (was 64).
  //   Phase 2: Σ_{sg_col=0}^{TTiles-1} SGs_in_col × (TTiles−sg_col) = 30  (was 64).
  //
  // The three group_barrier calls per step remain mandatory:
  //   Barrier A – slm_err written before Phase 2 reads it.
  //   Barrier B – slm_inv read by Phase 2 before any SG writes the update
  //               (SG(r,c1) must not write slm_inv[r][c1] while SG(r,c2) still
  //               reads slm_inv[r][c1] in its k_block=c1 iteration).
  //   Barrier C – slm_inv update complete before next step's Phase 1 reads it.
  // ---------------------------------------------------------------------------
  for (int step = 0; step < steps; ++step) {

    // =========================================================================
    // Phase 1 – Compute err = L × inv  (64×64 blocked GEMM in SLM)
    //           then apply strictly-lower-triangular mask.
    //
    // Optimization (A): upper-triangle SGs (sg_row < sg_col) skip the GEMM.
    // They write nothing to slm_err; Phase 2's k_block loop (optimization C)
    // starts at sg_col and never reads those upper-triangle err tiles.
    // =========================================================================
    if (sg_row >= sg_col) {
      clear(tCrC);

      CUTE_UNROLL
      for (int k_block = 0; k_block < TTiles; ++k_block) {
        // Load A tile from slm_L: L[sg_row*TBlock..+TBlock][k_block*TBlock..+TBlock]  (row-major BF16)
        // ALayout: thread t holds A[m_dpas][(t%2)*8..(t%2)*8+7] per DPAS atom.
        CUTE_UNROLL
        for (int v = 0; v < 8; ++v) {
          // Atom 0 (M rows 0..7 of the TBlock-row tile):
          tCrA(v) = slm_L[
              (sg_row * TBlock + m_dpas) * kN +
              k_block * TBlock + k_half + v];
          // Atom 1 (M rows 8..15 of the TBlock-row tile):
          tCrA(8+v) = slm_L[
              (sg_row * TBlock + 8 + m_dpas) * kN +
              k_block * TBlock + k_half + v];
        }

        // Load B tile from slm_inv: inv[k_block*TBlock..+TBlock][sg_col*TBlock..+TBlock]
        // Column-major (B) interpretation: thread t holds B[k=0..TBlock-1][n=t].
        // B[k][t] = inv[k_block*TBlock + k][sg_col*TBlock + t] → row-major SLM index.
        CUTE_UNROLL
        for (int v = 0; v < 16; ++v) {
          tCrB(v) = slm_inv[
              (k_block * TBlock + v) * kN +
              sg_col * TBlock + local_id];
        }

        // DPAS: tCrC += A × B
        cute::gemm(mma, tCrA, tCrB, tCrC);
      }

      // Apply strictly-lower-triangular mask to tCrC (err = lower_strict(L*inv)).
      // sg_row == sg_col → diagonal block: zero elements where row_local ≤ col_local.
      // sg_row  > sg_col → entire block is strictly lower-triangular: keep all.
      if (sg_row == sg_col) {
        CUTE_UNROLL
        for (int v = 0; v < 8; ++v) {
          // Atom 0: local row = m_dpas, local col = n_half + v
          if (m_dpas <= n_half + v) tCrC(v)   = 0.0f;
          // Atom 1: local row = 8 + m_dpas, local col = n_half + v
          if (8 + m_dpas <= n_half + v) tCrC(8+v) = 0.0f;
        }
      }
      // sg_row > sg_col: keep all (entire block is strictly lower-triangular).

      // Write masked err to SLM (float → BF16).
      CUTE_UNROLL
      for (int v = 0; v < 8; ++v) {
        // Atom 0
        slm_err[(sg_row * TBlock + m_dpas) * kN +
                 sg_col * TBlock + n_half + v] = T(tCrC(v));
        // Atom 1
        slm_err[(sg_row * TBlock + 8 + m_dpas) * kN +
                 sg_col * TBlock + n_half + v] = T(tCrC(8+v));
      }
    }
    // sg_row < sg_col: slm_err[sg_row][sg_col] tiles are not written.
    // Phase 2's k_block loop (optimization C) starts at sg_col, so these
    // upper-triangle err tiles are never read and their content does not matter.

    // Barrier A: all sub-groups must see the complete slm_err before Phase 2.
    sycl::group_barrier(wg);

    // =========================================================================
    // Phase 2 – Compute update = inv × err  (64×64 blocked GEMM in SLM)
    //           then update inv = inv − update.
    //
    // Optimization (B): upper-triangle SGs (sg_row < sg_col) skip Phase 2.
    //   Since inv is lower-triangular and err is strictly lower-triangular,
    //   (inv×err)[upper] = 0 exactly, so slm_inv[upper] needs no update.
    //   The invariant inv[upper] = 0 is maintained throughout all steps.
    //
    // Optimization (C): k_block loop starts at sg_col.
    //   slm_err[k_block < sg_col][sg_col] = 0 (upper-triangle err tiles, not
    //   written by Phase 1).  Those k_block iterations contribute nothing to
    //   tCrC and are skipped.
    // =========================================================================
    if (sg_row >= sg_col) {
      clear(tCrC);

      // k_block starts at sg_col (not 0): err[k_block < sg_col][sg_col] = 0.
      // Note: CUTE_UNROLL is not used here because the loop start (sg_col) is
      // a runtime value; the inner-v loops below are still CUTE_UNROLL-ed.
      for (int k_block = sg_col; k_block < TTiles; ++k_block) {
        // Load A tile from slm_inv: inv[sg_row*TBlock..+TBlock][k_block*TBlock..+TBlock].
        CUTE_UNROLL
        for (int v = 0; v < 8; ++v) {
          tCrA(v) = slm_inv[
              (sg_row * TBlock + m_dpas) * kN +
              k_block * TBlock + k_half + v];
          tCrA(8+v) = slm_inv[
              (sg_row * TBlock + 8 + m_dpas) * kN +
              k_block * TBlock + k_half + v];
        }

        // Load B tile from slm_err: err[k_block*TBlock..+TBlock][sg_col*TBlock..+TBlock].
        CUTE_UNROLL
        for (int v = 0; v < 16; ++v) {
          tCrB(v) = slm_err[
              (k_block * TBlock + v) * kN +
              sg_col * TBlock + local_id];
        }

        // DPAS: tCrC += inv × err
        cute::gemm(mma, tCrA, tCrB, tCrC);
      }
    }

    // Barrier B: every sub-group must finish reading slm_inv before any
    // sub-group writes back its update.  Without this, SG(r, c0) may
    // write slm_inv[r][c0] while SG(r, c1) is still reading slm_inv[r][c0]
    // in its k_block=c0 iteration, causing wrong results.
    sycl::group_barrier(wg);

    // Apply inv = inv − update in-place (lower-triangle and diagonal SGs only).
    // Different sub-groups write to non-overlapping tiles → no conflict.
    if (sg_row >= sg_col) {
      CUTE_UNROLL
      for (int v = 0; v < 8; ++v) {
        // Atom 0: row = sg_row*TBlock + m_dpas, col = sg_col*TBlock + n_half + v
        int idx0 = (sg_row * TBlock + m_dpas) * kN +
                    sg_col * TBlock + n_half + v;
        slm_inv[idx0] = T(float(slm_inv[idx0]) - tCrC(v));

        // Atom 1: row = sg_row*TBlock + 8 + m_dpas, col = sg_col*TBlock + n_half + v
        int idx1 = (sg_row * TBlock + 8 + m_dpas) * kN +
                    sg_col * TBlock + n_half + v;
        slm_inv[idx1] = T(float(slm_inv[idx1]) - tCrC(8+v));
      }
    }

    // Barrier C: all sub-groups must complete slm_inv writes before the next
    // iteration reads slm_inv in Phase 1 (or Phase 2 of the next step).
    sycl::group_barrier(wg);
  } // end Neumann loop

  // ---------------------------------------------------------------------------
  // Write final inverse from SLM to global memory (output).
  // ---------------------------------------------------------------------------
  for (int i = global_tid; i < kN * kN; i += TWGSize) {
    Out_ptr[i] = slm_inv[i];
  }
}

} // namespace neumann_inverse
