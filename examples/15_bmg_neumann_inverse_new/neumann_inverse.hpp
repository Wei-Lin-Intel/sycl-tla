/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
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

/*!
 * \file neumann_inverse.hpp
 * \brief SYCL kernel for Neumann-series lower-triangular matrix inverse (64×64, BF16).
 *
 * Implements the iterative Neumann inverse for batched 64×64 unit lower-triangular
 * matrices in BF16.  The algorithm is ported from `tril_inverse_iterative_64_bf16_kernel`
 * in `python/opt_tril_inverse_triton.py`.
 *
 * Algorithm (per batch element):
 *   inv = I
 *   for _ in range(STEPS):
 *       err = strict_lower_tri(L @ inv)   // elementwise mask: row > col
 *       inv = inv - inv @ err
 *
 * Implementation notes:
 *
 *   The 64×64 matrix is decomposed into a 4×4 grid of 16×16 blocks (labelled
 *   with block indices I, J ∈ {0,1,2,3}).  Each block-level GEMM is performed
 *   with the TiledMMA (XE_8x16x16_F32BF16BF16F32_TT, 16 threads per subgroup).
 *
 *   Memory layout:
 *     - L (input):   global memory, row-major, read every iteration
 *     - inv (output): global memory, updated in-place each iteration (column-major
 *                     update order ensures correctness without a double buffer)
 *     - err (temp):  separate global memory buffer; holds the full 64×64 error
 *                    matrix computed in Phase 1 and consumed in Phase 2
 *
 *   Phase 1 – compute err blocks:
 *     For off-diagonal blocks (I > J): err[I][J] = Σ_{K=J}^{I} L[I][K] · inv[K][J]
 *     For diagonal blocks (I = J):
 *       err[I][I] = L[I][I] · inv[I][I] – I_{16}
 *       (The product of two unit lower-triangular matrices is unit lower-tri, so
 *        subtracting the identity yields the strictly lower-triangular part without
 *        requiring an explicit upper-triangle zero-out.)
 *
 *   Diagonal subtraction of identity uses the MMA C-fragment thread/element layout.
 *   For XE_8x16x16 with TiledMMA (16×16×16, 16 threads):
 *     CLayout = Layout<Shape<_16,_8>, Stride<_8,_1>> (atom level)
 *     With 2 M-atoms per subgroup:
 *       - Thread t (even), element e = t/2 → C[t/2][t/2]   (atom 0 diagonal)
 *       - Thread t (odd),  element e = 8 + t/2 → C[8+t/2][8+t/2] (atom 1 diagonal)
 *
 *   Phase 2 – update inv blocks (column-first order, J = 0,1,2,3):
 *     new_inv[I][J] = inv[I][J] – Σ_{K=J}^{I} inv[I][K] · err[K][J]
 *     The column-first update order guarantees that when computing new_inv[I][J],
 *     all inv[I][K] values for K ≥ J have not yet been overwritten (they are in
 *     the same row I but in columns K ≥ J, which are processed later).
 *
 *   Register residency:
 *     L and inv are read fresh from global memory for each block GEMM; err is
 *     written to a temporary global buffer between the two phases.  Keeping all
 *     16 blocks resident in registers simultaneously would require ~2560 bytes per
 *     thread (10 blocks × 16 float32 per thread), which exceeds the practical
 *     register budget.  The present design amortizes global-memory traffic across
 *     STEPS iterations by keeping the kernel structure simple and letting the GPU's
 *     L1/L2 cache absorb repeated reads.
 */

#pragma once

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_xe_2d.hpp>
#include <cute/util/xe_split_barrier.hpp>

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace neumann_inverse {

using namespace cute;

// ---------------------------------------------------------------------------
// gemm_TTS : tCrC += A(row-major M×K) · B(col-major N×K)
//
// The col-major (stride=(1, N_stride)) layout of B causes the 2D block copy
// to load B so that DPAS receives the correct input, yielding the standard
// matrix product  C += A · B  (same semantics as chunk_inverse::gemm_TTS).
// ---------------------------------------------------------------------------
template <class ATensor, class BTensor, class SGCTensor, class TiledMMA>
CUTE_DEVICE void
gemm_TTS(ATensor const& A,    // (M, K) row-major
         BTensor const& B,    // (N, K) col-major  (= B^T format for DPAS)
         SGCTensor&     tCrC,
         int wg_m, int wg_n,
         TiledMMA const& mma)
{
  auto item     = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int  local_id = item.get_local_linear_id();

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());

  auto wg_tile = mma.tile_mnk();

  Tensor gA = local_tile(cA, select<0,2>(wg_tile), make_coord(wg_m, _));
  Tensor gB = local_tile(cB, select<1,2>(wg_tile), make_coord(wg_n, _));

  auto copy_a = get_block_2d_copy_A<void>(mma, A);
  auto copy_b = get_block_2d_copy_B<void>(mma, B);

  auto thr_mma    = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  constexpr int prefetch_dist = 3;
  constexpr int barrier_scope = 2;

  int k_tile_count    = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);
    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);
    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }
    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);
    cute::gemm(mma, tCrA, tCrB, tCrC);
    barrier_wait(barrier_scope);
  }
}

// ---------------------------------------------------------------------------
// gemm_STS : tCrC += tCrA_reg(register fragment) · B(col-major N×K)
// ---------------------------------------------------------------------------
template <class ASGCTensor, class BTensor, class CSGCTensor, class TiledMMA>
CUTE_DEVICE void
gemm_STS(ASGCTensor const& tCrA,
         BTensor    const& B,
         CSGCTensor&       tCrC,
         int wg_m, int wg_n,
         TiledMMA const& mma)
{
  auto item     = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int  local_id = item.get_local_linear_id();

  Tensor cB = make_identity_tensor(B.shape());

  auto wg_tile = mma.tile_mnk();

  Tensor gB = local_tile(cB, select<1,2>(wg_tile), make_coord(wg_n, _));

  auto copy_b = get_block_2d_copy_B<void>(mma, B);

  auto thr_mma    = mma.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);

  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tBgB = thr_copy_b.partition_S(gB);

  auto prefetch_b    = make_block_2d_prefetch(copy_b);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);
  auto pBgB          = thr_prefetch_B.partition_S(gB);

  constexpr int prefetch_dist = 3;
  constexpr int barrier_scope = 2;

  int k_tile_count    = ceil_div(shape<1>(B), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);
    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }
    reorder(tBrB, tCrB);
    cute::gemm(mma, tCrA, tCrB, tCrC);
    barrier_wait(barrier_scope);
  }
}

// ---------------------------------------------------------------------------
// neumann_inverse_kernel
//
// Each work-group (one sub-group, 16 threads) computes the Neumann-series
// inverse of one 64×64 unit lower-triangular BF16 matrix.
//
// Parameters:
//   L_base   – batch of input matrices (row-major, 64×64 each)
//   inv_base – output inverse matrices; must be pre-initialised to I_{64}
//              by the host; updated in-place across STEPS iterations
//   err_base – scratch buffer of the same size as inv_base
//   batch_size – number of matrices in the batch
//
// Template parameter STEPS controls the number of Neumann iterations.
// ---------------------------------------------------------------------------
template <typename T, class TiledMMA, int STEPS>
CUTE_DEVICE void
neumann_inverse_kernel(const T* __restrict__ L_base,
                       T*                    inv_base,
                       T*                    err_base,
                       int                   batch_size)
{
  static constexpr int N    = 64;   // matrix dimension
  static constexpr int TILE = 16;   // block tile size

  auto item     = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int  local_id = item.get_local_linear_id();
  int  batch_id = item.get_group(2);   // one work-group per matrix

  auto sg          = item.get_sub_group();
  int  sg_local_id = sg.get_local_linear_id();

  if (batch_id >= batch_size) return;

  TiledMMA mma{};
  auto wg_tile = mma.tile_mnk();
  auto thr_mma = mma.get_slice(local_id);

  // Pointers to this batch element's matrices.
  const T* L_ptr   = L_base   + static_cast<int64_t>(batch_id) * N * N;
  T*       inv_ptr = inv_base + static_cast<int64_t>(batch_id) * N * N;
  T*       err_ptr = err_base + static_cast<int64_t>(batch_id) * N * N;

  auto block_shape = make_shape(Int<TILE>{}, Int<TILE>{});

  // Helper: row-major (TILE×TILE) read-only tensor view of ptr[row_off*N + col_off].
  // Used as the A argument of gemm_TTS (first matrix in the product).
  auto make_row = [&](const T* ptr, int row_off, int col_off) {
    return make_tensor(
        make_gmem_ptr(ptr + row_off * N + col_off),
        make_layout(block_shape, make_stride(N, _1{})));
  };

  // Helper: col-major (TILE×TILE) read-only tensor view — the "transposed"
  // layout consumed by gemm_TTS as its B argument.
  auto make_col = [&](const T* ptr, int row_off, int col_off) {
    return make_tensor(
        make_gmem_ptr(ptr + row_off * N + col_off),
        make_layout(block_shape, make_stride(_1{}, N)));
  };

  // Helper: row-major (TILE×TILE) writable tensor view.
  // Used as the destination in write_block (get_block_2d_copy_D requires
  // a non-const tensor pointer).
  auto make_row_w = [&](T* ptr, int row_off, int col_off) {
    return make_tensor(
        make_gmem_ptr(ptr + row_off * N + col_off),
        make_layout(block_shape, make_stride(N, _1{})));
  };

  // Coordinate / identity tensors for partitioning.
  Tensor cA = make_identity_tensor(block_shape);
  Tensor cB = make_identity_tensor(block_shape);
  Tensor cC = make_identity_tensor(block_shape);
  Tensor gA = local_tile(cA, select<0,2>(wg_tile), make_coord(0, _));
  Tensor gB = local_tile(cB, select<1,2>(wg_tile), make_coord(0, _));
  Tensor gC = local_tile(cC, wg_tile, make_coord(0, 0, 0), Step<_1, _1, X>{});

  auto tCrC = thr_mma.partition_sg_fragment_C(gC);

  // Helper: write float32 fragment tCrC to a BF16 global memory block.
  // ptr must be non-const since get_block_2d_copy_D writes to it.
  auto write_block = [&](T* ptr, int row_off, int col_off) {
    auto dst          = make_row_w(ptr, row_off, col_off);
    auto copy_D       = get_block_2d_copy_D<void>(mma, dst);
    auto thr_copy_D   = copy_D.get_slice(local_id);
    auto tCrD         = thr_copy_D.partition_sg_fragment_S(gC);
    auto tCgD         = thr_copy_D.partition_D(gC);
    reorder(tCrC, tCrD);
    copy(copy_D, tCrD, tCgD);
  };

  // Helper: element-wise load of a 16×16 BF16 block into the float32
  // accumulator tCrC, ADDING to its current value.
  //
  // Thread-to-element mapping for the TiledMMA C fragment
  // (XE_8x16x16_F32BF16BF16F32_TT, 2 M-atoms, 16 threads):
  //   CLayout = Layout<Shape<_16,_8>, Stride<_8,_1>>
  //   For thread t, element e (0..7) in atom 0:
  //       global_pos = t*8 + e  →  row = pos/16, col = pos%16   (rows 0..7)
  //   For element e (0..7) in atom 1 (stored at fragment index e+8):
  //       same formula but row += 8
  auto add_block_to_fragment = [&](const T* ptr, int row_off, int col_off) {
    const T* base = ptr + row_off * N + col_off;
    CUTE_UNROLL
    for (int e = 0; e < 8; ++e) {
      int pos  = sg_local_id * 8 + e;
      int row0 = pos / TILE;
      int col0 = pos % TILE;
      // atom 0: rows 0..7
      tCrC(e)     += static_cast<float>(base[row0 * N + col0]);
      // atom 1: rows 8..15
      tCrC(e + 8) += static_cast<float>(base[(row0 + 8) * N + col0]);
    }
  };

  // Helper: subtract the 16×16 identity from the diagonal elements of tCrC.
  //
  // Diagonal element C[i][i] resides in:
  //   Even thread t=2k: fragment index k  (atom 0, diagonal of row k)
  //   Odd  thread t=2k+1: fragment index 8+k  (atom 1, diagonal of row 8+k)
  // See the header comment for derivation.
  auto subtract_identity_from_fragment = [&]() {
    if (sg_local_id % 2 == 0) {
      tCrC(sg_local_id / 2) -= 1.0f;       // atom 0 diagonal
    } else {
      tCrC(8 + sg_local_id / 2) -= 1.0f;   // atom 1 diagonal
    }
  };

  // =========================================================================
  // Neumann iteration: inv = I,  repeat STEPS times:
  //   Phase 1: err  = strict_lower_tri(L @ inv)
  //   Phase 2: inv  = inv − inv @ err         (column-first in-place update)
  // =========================================================================
  CUTE_UNROLL
  for (int step = 0; step < STEPS; ++step) {

    // -----------------------------------------------------------------------
    // Phase 1 – Compute err = strict_lower_tri(L @ inv)
    //
    // Off-diagonal blocks (I > J): err[I][J] = Σ_{K=J}^{I} L[I][K] · inv[K][J]
    // Diagonal blocks (I = J):     err[I][I] = L[I][I] · inv[I][I] − I_{16}
    // -----------------------------------------------------------------------

    // --- err[0][0] = L[0][0] · inv[0][0] − I ---
    clear(tCrC);
    gemm_TTS(make_row(L_ptr,   0,  0), make_col(inv_ptr,  0,  0), tCrC, 0, 0, mma);
    subtract_identity_from_fragment();
    write_block(err_ptr, 0, 0);

    // --- err[1][0] = L[1][0]·inv[0][0] + L[1][1]·inv[1][0] ---
    clear(tCrC);
    gemm_TTS(make_row(L_ptr,  16,  0), make_col(inv_ptr,  0,  0), tCrC, 0, 0, mma);
    gemm_TTS(make_row(L_ptr,  16, 16), make_col(inv_ptr, 16,  0), tCrC, 0, 0, mma);
    write_block(err_ptr, 16, 0);

    // --- err[1][1] = L[1][1] · inv[1][1] − I ---
    clear(tCrC);
    gemm_TTS(make_row(L_ptr,  16, 16), make_col(inv_ptr, 16, 16), tCrC, 0, 0, mma);
    subtract_identity_from_fragment();
    write_block(err_ptr, 16, 16);

    // --- err[2][0] = Σ_{K=0}^{2} L[2][K]·inv[K][0] ---
    clear(tCrC);
    gemm_TTS(make_row(L_ptr,  32,  0), make_col(inv_ptr,  0,  0), tCrC, 0, 0, mma);
    gemm_TTS(make_row(L_ptr,  32, 16), make_col(inv_ptr, 16,  0), tCrC, 0, 0, mma);
    gemm_TTS(make_row(L_ptr,  32, 32), make_col(inv_ptr, 32,  0), tCrC, 0, 0, mma);
    write_block(err_ptr, 32, 0);

    // --- err[2][1] = L[2][1]·inv[1][1] + L[2][2]·inv[2][1] ---
    clear(tCrC);
    gemm_TTS(make_row(L_ptr,  32, 16), make_col(inv_ptr, 16, 16), tCrC, 0, 0, mma);
    gemm_TTS(make_row(L_ptr,  32, 32), make_col(inv_ptr, 32, 16), tCrC, 0, 0, mma);
    write_block(err_ptr, 32, 16);

    // --- err[2][2] = L[2][2] · inv[2][2] − I ---
    clear(tCrC);
    gemm_TTS(make_row(L_ptr,  32, 32), make_col(inv_ptr, 32, 32), tCrC, 0, 0, mma);
    subtract_identity_from_fragment();
    write_block(err_ptr, 32, 32);

    // --- err[3][0] = Σ_{K=0}^{3} L[3][K]·inv[K][0] ---
    clear(tCrC);
    gemm_TTS(make_row(L_ptr,  48,  0), make_col(inv_ptr,  0,  0), tCrC, 0, 0, mma);
    gemm_TTS(make_row(L_ptr,  48, 16), make_col(inv_ptr, 16,  0), tCrC, 0, 0, mma);
    gemm_TTS(make_row(L_ptr,  48, 32), make_col(inv_ptr, 32,  0), tCrC, 0, 0, mma);
    gemm_TTS(make_row(L_ptr,  48, 48), make_col(inv_ptr, 48,  0), tCrC, 0, 0, mma);
    write_block(err_ptr, 48, 0);

    // --- err[3][1] = Σ_{K=1}^{3} L[3][K]·inv[K][1] ---
    clear(tCrC);
    gemm_TTS(make_row(L_ptr,  48, 16), make_col(inv_ptr, 16, 16), tCrC, 0, 0, mma);
    gemm_TTS(make_row(L_ptr,  48, 32), make_col(inv_ptr, 32, 16), tCrC, 0, 0, mma);
    gemm_TTS(make_row(L_ptr,  48, 48), make_col(inv_ptr, 48, 16), tCrC, 0, 0, mma);
    write_block(err_ptr, 48, 16);

    // --- err[3][2] = L[3][2]·inv[2][2] + L[3][3]·inv[3][2] ---
    clear(tCrC);
    gemm_TTS(make_row(L_ptr,  48, 32), make_col(inv_ptr, 32, 32), tCrC, 0, 0, mma);
    gemm_TTS(make_row(L_ptr,  48, 48), make_col(inv_ptr, 48, 32), tCrC, 0, 0, mma);
    write_block(err_ptr, 48, 32);

    // --- err[3][3] = L[3][3] · inv[3][3] − I ---
    clear(tCrC);
    gemm_TTS(make_row(L_ptr,  48, 48), make_col(inv_ptr, 48, 48), tCrC, 0, 0, mma);
    subtract_identity_from_fragment();
    write_block(err_ptr, 48, 48);

    // -----------------------------------------------------------------------
    // Phase 2 – Update inv = inv − inv @ err  (column-first, in-place)
    //
    // new_inv[I][J] = inv[I][J] − Σ_{K=J}^{I} inv[I][K] · err[K][J]
    //
    // Column-first order (J = 0, 1, 2, 3) ensures that when reading
    // inv[I][K] for K ≥ J, those values have not yet been overwritten.
    // Within a column, rows can be processed in any order since new_inv[I][J]
    // only reads from row I (different rows are independent).
    // -----------------------------------------------------------------------

    // ---- Column J = 0 ----

    // new_inv[0][0] = inv[0][0] − inv[0][0]·err[0][0]
    clear(tCrC);
    gemm_TTS(make_row(inv_ptr,  0,  0), make_col(err_ptr,  0,  0), tCrC, 0, 0, mma);
    CUTE_UNROLL
    for (int i = 0; i < tCrC.size(); ++i) tCrC(i) *= -1.0f;
    add_block_to_fragment(inv_ptr,  0,  0);
    write_block(inv_ptr,  0,  0);

    // new_inv[1][0] = inv[1][0] − inv[1][0]·err[0][0] − inv[1][1]·err[1][0]
    clear(tCrC);
    gemm_TTS(make_row(inv_ptr, 16,  0), make_col(err_ptr,  0,  0), tCrC, 0, 0, mma);
    gemm_TTS(make_row(inv_ptr, 16, 16), make_col(err_ptr, 16,  0), tCrC, 0, 0, mma);
    CUTE_UNROLL
    for (int i = 0; i < tCrC.size(); ++i) tCrC(i) *= -1.0f;
    add_block_to_fragment(inv_ptr, 16,  0);
    write_block(inv_ptr, 16,  0);

    // new_inv[2][0]
    clear(tCrC);
    gemm_TTS(make_row(inv_ptr, 32,  0), make_col(err_ptr,  0,  0), tCrC, 0, 0, mma);
    gemm_TTS(make_row(inv_ptr, 32, 16), make_col(err_ptr, 16,  0), tCrC, 0, 0, mma);
    gemm_TTS(make_row(inv_ptr, 32, 32), make_col(err_ptr, 32,  0), tCrC, 0, 0, mma);
    CUTE_UNROLL
    for (int i = 0; i < tCrC.size(); ++i) tCrC(i) *= -1.0f;
    add_block_to_fragment(inv_ptr, 32,  0);
    write_block(inv_ptr, 32,  0);

    // new_inv[3][0]
    clear(tCrC);
    gemm_TTS(make_row(inv_ptr, 48,  0), make_col(err_ptr,  0,  0), tCrC, 0, 0, mma);
    gemm_TTS(make_row(inv_ptr, 48, 16), make_col(err_ptr, 16,  0), tCrC, 0, 0, mma);
    gemm_TTS(make_row(inv_ptr, 48, 32), make_col(err_ptr, 32,  0), tCrC, 0, 0, mma);
    gemm_TTS(make_row(inv_ptr, 48, 48), make_col(err_ptr, 48,  0), tCrC, 0, 0, mma);
    CUTE_UNROLL
    for (int i = 0; i < tCrC.size(); ++i) tCrC(i) *= -1.0f;
    add_block_to_fragment(inv_ptr, 48,  0);
    write_block(inv_ptr, 48,  0);

    // ---- Column J = 1 ----

    // new_inv[1][1] = inv[1][1] − inv[1][1]·err[1][1]
    clear(tCrC);
    gemm_TTS(make_row(inv_ptr, 16, 16), make_col(err_ptr, 16, 16), tCrC, 0, 0, mma);
    CUTE_UNROLL
    for (int i = 0; i < tCrC.size(); ++i) tCrC(i) *= -1.0f;
    add_block_to_fragment(inv_ptr, 16, 16);
    write_block(inv_ptr, 16, 16);

    // new_inv[2][1]
    clear(tCrC);
    gemm_TTS(make_row(inv_ptr, 32, 16), make_col(err_ptr, 16, 16), tCrC, 0, 0, mma);
    gemm_TTS(make_row(inv_ptr, 32, 32), make_col(err_ptr, 32, 16), tCrC, 0, 0, mma);
    CUTE_UNROLL
    for (int i = 0; i < tCrC.size(); ++i) tCrC(i) *= -1.0f;
    add_block_to_fragment(inv_ptr, 32, 16);
    write_block(inv_ptr, 32, 16);

    // new_inv[3][1]
    clear(tCrC);
    gemm_TTS(make_row(inv_ptr, 48, 16), make_col(err_ptr, 16, 16), tCrC, 0, 0, mma);
    gemm_TTS(make_row(inv_ptr, 48, 32), make_col(err_ptr, 32, 16), tCrC, 0, 0, mma);
    gemm_TTS(make_row(inv_ptr, 48, 48), make_col(err_ptr, 48, 16), tCrC, 0, 0, mma);
    CUTE_UNROLL
    for (int i = 0; i < tCrC.size(); ++i) tCrC(i) *= -1.0f;
    add_block_to_fragment(inv_ptr, 48, 16);
    write_block(inv_ptr, 48, 16);

    // ---- Column J = 2 ----

    // new_inv[2][2] = inv[2][2] − inv[2][2]·err[2][2]
    clear(tCrC);
    gemm_TTS(make_row(inv_ptr, 32, 32), make_col(err_ptr, 32, 32), tCrC, 0, 0, mma);
    CUTE_UNROLL
    for (int i = 0; i < tCrC.size(); ++i) tCrC(i) *= -1.0f;
    add_block_to_fragment(inv_ptr, 32, 32);
    write_block(inv_ptr, 32, 32);

    // new_inv[3][2]
    clear(tCrC);
    gemm_TTS(make_row(inv_ptr, 48, 32), make_col(err_ptr, 32, 32), tCrC, 0, 0, mma);
    gemm_TTS(make_row(inv_ptr, 48, 48), make_col(err_ptr, 48, 32), tCrC, 0, 0, mma);
    CUTE_UNROLL
    for (int i = 0; i < tCrC.size(); ++i) tCrC(i) *= -1.0f;
    add_block_to_fragment(inv_ptr, 48, 32);
    write_block(inv_ptr, 48, 32);

    // ---- Column J = 3 ----

    // new_inv[3][3] = inv[3][3] − inv[3][3]·err[3][3]
    clear(tCrC);
    gemm_TTS(make_row(inv_ptr, 48, 48), make_col(err_ptr, 48, 48), tCrC, 0, 0, mma);
    CUTE_UNROLL
    for (int i = 0; i < tCrC.size(); ++i) tCrC(i) *= -1.0f;
    add_block_to_fragment(inv_ptr, 48, 48);
    write_block(inv_ptr, 48, 48);

  } // end STEPS loop
}

} // namespace neumann_inverse
