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
 * Algorithm (STEPS=12 iterations):
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
// Constants
// ---------------------------------------------------------------------------
static constexpr int kN      = 64;   // Fixed matrix dimension
static constexpr int kSteps  = 12;   // Neumann iteration count (fixed)
static constexpr int kBlock  = 16;   // DPAS tile size
static constexpr int kTiles  = 4;    // kN / kBlock = 4 tiles per dimension
static constexpr int kSGs    = 16;   // kTiles × kTiles sub-groups per work-group
static constexpr int kSGSize = 16;   // Threads per sub-group (DPAS SIMD-16)
static constexpr int kWGSize = kSGs * kSGSize;  // 256 threads per work-group

// ---------------------------------------------------------------------------
// TiledMMA for one 16×16 DPAS tile.
//   MMA atom : XE_8x16x16_F32BF16BF16F32_TT (8M×16N×16K DPAS, 16 threads / sub-group)
//   WG tile  : 16×16×16  →  2 DPAS atoms stacked in M
//   Sub-group layout: (1,1,1) →  exactly 1 sub-group (16 threads) per tile
// ---------------------------------------------------------------------------
using TileMMA = typename TiledMMAHelper<
    MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
    Layout<Shape<_16, _16, _16>>,
    Layout<Shape<_1, _1, _1>, Stride<_1, _1, _0>>>::TiledMMA;

// ---------------------------------------------------------------------------
// neumann_inverse_kernel
//
// Each work-group inverts one 64×64 unit lower-triangular BF16 matrix.
// L_base : input   – batch of B matrices, shape [B, 64, 64] row-major
// Out_base: output – batch of B inverse matrices, shape [B, 64, 64] row-major
// ---------------------------------------------------------------------------
template <typename T>
CUTE_DEVICE void
neumann_inverse_kernel(const T* L_base, T* Out_base, int batch_size)
{
  auto item     = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int  batch_id = item.get_group(2);      // one work-group per matrix in batch
  if (batch_id >= batch_size) return;

  auto wg       = item.get_group();
  auto sg        = item.get_sub_group();
  int  sg_id     = static_cast<int>(sg.get_group_linear_id()); // 0..15
  int  sg_row    = sg_id / kTiles;         // tile row   (0..3)
  int  sg_col    = sg_id % kTiles;         // tile column (0..3)
  int  local_id  = static_cast<int>(sg.get_local_linear_id()); // 0..15 within sub-group
  int  global_tid = sg_id * kSGSize + local_id; // 0..255 within work-group

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

  // ---------------------------------------------------------------------------
  // Cooperative initialization: all 256 threads load L and initialize inv = I.
  // ---------------------------------------------------------------------------
  for (int i = global_tid; i < kN * kN; i += kWGSize) {
    slm_L[i] = L_ptr[i];
    int row   = i / kN;
    int col   = i % kN;
    slm_inv[i] = (row == col) ? T(1.0f) : T(0.0f);
  }
  sycl::group_barrier(wg);

  // ---------------------------------------------------------------------------
  // Set up TiledMMA and thread-level coordinate fragments.
  // Each sub-group handles one kBlock×kBlock (16×16) output tile using 16 threads.
  // ---------------------------------------------------------------------------
  TileMMA mma{};
  auto wg_tile  = mma.tile_mnk();            // (16, 16, 16)
  auto thr_mma  = mma.get_slice(local_id);

  // Coordinate/identity tensors (shapes only; used to allocate register fragments).
  auto A_shape  = make_shape(Int<kBlock>{}, Int<kBlock>{});
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
  // Neumann iteration loop  (kSteps = 12, fixed)
  // ---------------------------------------------------------------------------
  for (int step = 0; step < kSteps; ++step) {

    // =========================================================================
    // Phase 1 – Compute err = L × inv  (64×64 blocked GEMM in SLM)
    //           then apply strictly-lower-triangular mask.
    // =========================================================================
    clear(tCrC);

    CUTE_UNROLL
    for (int k_block = 0; k_block < kTiles; ++k_block) {
      // Load A tile from slm_L: L[sg_row*16..+16][k_block*16..+16]  (row-major BF16)
      // ALayout: thread t holds A[m_dpas][(t%2)*8..(t%2)*8+7] per DPAS atom.
      CUTE_UNROLL
      for (int v = 0; v < 8; ++v) {
        // Atom 0 (M rows 0..7 of the 16-row tile):
        tCrA(v) = slm_L[
            (sg_row * kBlock + m_dpas) * kN +
            k_block * kBlock + k_half + v];
        // Atom 1 (M rows 8..15 of the 16-row tile):
        tCrA(8+v) = slm_L[
            (sg_row * kBlock + 8 + m_dpas) * kN +
            k_block * kBlock + k_half + v];
      }

      // Load B tile from slm_inv: inv[k_block*16..+16][sg_col*16..+16]
      // Column-major (B) interpretation: thread t holds B[k=0..15][n=t].
      // B[k][t] = inv[k_block*16 + k][sg_col*16 + t] → row-major SLM index.
      CUTE_UNROLL
      for (int v = 0; v < 16; ++v) {
        tCrB(v) = slm_inv[
            (k_block * kBlock + v) * kN +
            sg_col * kBlock + local_id];
      }

      // DPAS: tCrC += A × B
      cute::gemm(mma, tCrA, tCrB, tCrC);
    }

    // Apply strictly-lower-triangular mask to tCrC (err = lower_strict(L*inv)).
    // For the (sg_row, sg_col) output block:
    //   sg_row < sg_col → entire block is upper triangular → zero all.
    //   sg_row = sg_col → diagonal block → zero row ≤ col elements.
    //   sg_row > sg_col → entire block is strictly lower → keep all.
    if (sg_row < sg_col) {
      // Zero the entire upper-triangular / on-diagonal block.
      CUTE_UNROLL
      for (int i = 0; i < static_cast<int>(tCrC.size()); ++i) {
        tCrC(i) = 0.0f;
      }
    } else if (sg_row == sg_col) {
      // Zero elements where (row_local ≤ col_local) within the 16×16 block.
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
      slm_err[(sg_row * kBlock + m_dpas) * kN +
               sg_col * kBlock + n_half + v] = T(tCrC(v));
      // Atom 1
      slm_err[(sg_row * kBlock + 8 + m_dpas) * kN +
               sg_col * kBlock + n_half + v] = T(tCrC(8+v));
    }

    // Synchronize: all sub-groups must see the complete slm_err before Phase 2.
    sycl::group_barrier(wg);

    // =========================================================================
    // Phase 2 – Compute update = inv × err  (64×64 blocked GEMM in SLM)
    //           then update inv = inv − update.
    // =========================================================================
    clear(tCrC);

    CUTE_UNROLL
    for (int k_block = 0; k_block < kTiles; ++k_block) {
      // Load A tile from slm_inv: inv[sg_row*16..+16][k_block*16..+16].
      CUTE_UNROLL
      for (int v = 0; v < 8; ++v) {
        tCrA(v) = slm_inv[
            (sg_row * kBlock + m_dpas) * kN +
            k_block * kBlock + k_half + v];
        tCrA(8+v) = slm_inv[
            (sg_row * kBlock + 8 + m_dpas) * kN +
            k_block * kBlock + k_half + v];
      }

      // Load B tile from slm_err: err[k_block*16..+16][sg_col*16..+16].
      // err is strictly lower-triangular; zero elements produce zero accumulation
      // automatically, so we load unconditionally (simpler, correct result).
      CUTE_UNROLL
      for (int v = 0; v < 16; ++v) {
        tCrB(v) = slm_err[
            (k_block * kBlock + v) * kN +
            sg_col * kBlock + local_id];
      }

      // DPAS: tCrC += inv × err
      cute::gemm(mma, tCrA, tCrB, tCrC);
    }

    // Barrier: ensure every sub-group has finished reading slm_inv before any
    // sub-group writes back its update.  Without this, sub-group (r, c0) may
    // write slm_inv[r_tile][c0_tile] while sub-group (r, c1) is still reading
    // slm_inv[r_tile][c0_tile] in its k_block=c0 iteration, causing wrong results.
    sycl::group_barrier(wg);

    // Update inv = inv − update (in-place subtraction in SLM).
    // Different sub-groups write to non-overlapping tiles → no conflict.
    CUTE_UNROLL
    for (int v = 0; v < 8; ++v) {
      // Atom 0: row = sg_row*16 + m_dpas, col = sg_col*16 + n_half + v
      int idx0 = (sg_row * kBlock + m_dpas) * kN +
                  sg_col * kBlock + n_half + v;
      slm_inv[idx0] = T(float(slm_inv[idx0]) - tCrC(v));

      // Atom 1: row = sg_row*16 + 8 + m_dpas, col = sg_col*16 + n_half + v
      int idx1 = (sg_row * kBlock + 8 + m_dpas) * kN +
                  sg_col * kBlock + n_half + v;
      slm_inv[idx1] = T(float(slm_inv[idx1]) - tCrC(8+v));
    }

    // Synchronize: all sub-groups must complete slm_inv writes before the next
    // iteration reads slm_inv in Phase 1 (or Phase 2 of the next step).
    sycl::group_barrier(wg);
  } // end Neumann loop

  // ---------------------------------------------------------------------------
  // Write final inverse from SLM to global memory (output).
  // ---------------------------------------------------------------------------
  for (int i = global_tid; i < kN * kN; i += kWGSize) {
    Out_ptr[i] = slm_inv[i];
  }
}

} // namespace neumann_inverse
