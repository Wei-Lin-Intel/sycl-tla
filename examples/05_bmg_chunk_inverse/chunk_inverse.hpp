/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
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
 * \file chunk_inverse.hpp
 * \brief SYCL kernel for block-wise lower-triangular matrix inverse (64x64, BF16).
 *
 * Implements the `chunk_inverse_opt_kernel` approach from vllm-xpu-kernels, adapted
 * for the SYCL*TLA framework.  The 64x64 matrix is split into a 4×4 grid of 16×16
 * blocks.  Diagonal blocks are inverted in-place using subgroup operations; off-
 * diagonal blocks are updated with gemm_TTS / gemm_STS helpers that use Intel 2-D
 * block-copy operations.
 *
 * The computation order is critical for correctness:
 *   1. Invert diagonal blocks A11, A22, A33, A44 in-place.
 *   2. A21_inv = −A22_inv · A21 · A11_inv          (one-shot via gemm_TTS+gemm_STS)
 *   3. A31_inv in two stages:
 *        stage-1 temp  = A31·A11_inv + A32·A21_inv  (written to A31 memory)
 *        stage-2 final = −A33_inv · temp            (written to A31 memory)
 *   4. A41_inv in two stages:
 *        stage-1 temp  = A41·A11_inv + A42·A21_inv + A43·A31_inv
 *                                                    (written to A41 memory)
 *        stage-2 final = −A44_inv · temp            (written to A41 memory)
 *   5. A32_inv = −A33_inv · A32 · A22_inv           (one-shot via gemm_TTS+gemm_STS)
 *   6. A42_inv in two stages:
 *        stage-1 temp  = A42·A22_inv + A43·A32_inv  (written to A42 memory)
 *        stage-2 final = −A44_inv · temp            (written to A42 memory)
 *   7. A43_inv = −A44_inv · A43 · A33_inv           (one-shot via gemm_TTS+gemm_STS)
 *
 * Step 3 MUST complete fully before step 4 reads A31 (which by then holds A31_inv).
 * Step 5 MUST complete fully before step 6 reads A32 (which by then holds A32_inv).
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

namespace chunk_inverse {

using namespace cute;

// ---------------------------------------------------------------------------
// gemm_TTS : C += A(row-major, M×K) · B(col-major, N×K)
//
// Name convention (from vllm-xpu-kernels gemm.hpp):
//   T = first operand is a global Tensor (row-major layout)
//   T = second operand is a global Tensor (col-major / "transposed" layout)
//   S = accumulator is a register fragment (Sub-group fragment)
//
// Both A and B are global tensors; C is a register fragment.
// The col-major (stride=(1, row_stride)) layout of B makes
// get_block_2d_copy_B load it so that DPAS sees the correct B input,
// yielding the standard matrix product  C += A · B.
// ---------------------------------------------------------------------------
template <class ATensor, class BTensor, class SGCTensor, class TiledMMA>
CUTE_DEVICE void
gemm_TTS(ATensor const& A,   // (M, K) row-major  stride=(K_stride, 1)
         BTensor const& B,   // (N, K) col-major  stride=(1, N_stride) = B^T layout
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
// gemm_STS : C += A_reg(M×K, register fragment) · B(col-major, N×K)
//
// Name convention (from vllm-xpu-kernels gemm.hpp):
//   S = first operand is a register fragment (already loaded from a prior gemm)
//   T = second operand is a global Tensor (col-major layout)
//   S = accumulator is a register fragment
// ---------------------------------------------------------------------------
template <class ASGCTensor, class BTensor, class CSGCTensor, class TiledMMA>
CUTE_DEVICE void
gemm_STS(ASGCTensor const& tCrA,   // register fragment, already loaded
         BTensor    const& B,       // (N, K) col-major
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
// chunk_inverse_opt_kernel
//
// Each work-group (one sub-group, 16 threads) inverts one 64×64 unit lower-
// triangular BF16 matrix in-place.
//
// Layout convention:
//   A[b]  is stored at  A_base + b * 64 * 64  in row-major order
//   (row stride = 64, col stride = 1)
// ---------------------------------------------------------------------------
template <typename T, class TiledMMA>
CUTE_DEVICE void
chunk_inverse_opt_kernel(T* A_base, int batch_size)
{
  auto item     = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int  local_id = item.get_local_linear_id();
  int  batch_id = item.get_group(2);   // one work-group per matrix in the batch

  auto sg          = item.get_sub_group();
  int  sg_local_id = sg.get_local_linear_id();

  if (batch_id >= batch_size) return;

  static constexpr int chunk_size = 64;   // matrix dimension

  TiledMMA mma{};
  auto wg_tile = mma.tile_mnk();
  auto thr_mma = mma.get_slice(local_id);

  // Pointer to this batch's 64×64 matrix (row-major).
  T* A_ptr = A_base + static_cast<int64_t>(batch_id) * chunk_size * chunk_size;

  // -----------------------------------------------------------------------
  // Step 1: Invert the 4 diagonal 16×16 blocks in-place.
  //
  // The matrix is unit lower-triangular (1 on diagonal).  The sub-group
  // (16 threads) computes the inverse of each diagonal block sequentially
  // using a column-by-column broadcast scheme.
  // -----------------------------------------------------------------------
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i) {
    int  offset   = i * 16;
    T*   A_ptr_xx = A_ptr + offset * chunk_size + offset;

    // A_local[e] holds: inverse column sg_local_id, row e  (built incrementally)
    //                   Initialized to 0 for rows 0..sg_local_id (diagonal = 1
    //                   is kept implicitly; saving loop skips the diagonal).
    float A_local[16];
    float A_other[16];  // broadcast values received from other sub-group lanes

    CUTE_UNROLL
    for (int e = 0; e < sg_local_id + 1; ++e) {
      A_local[e] = 0.0f;
    }

    // Load the sub-diagonal elements of this thread's row from the 16×16 block.
    T A_load[16];
    CUTE_UNROLL
    for (int e = 0; e < sg_local_id; ++e) {
      A_load[e] = A_ptr_xx[sg_local_id * chunk_size + e];
    }

    // Distribute: after this, thread nn_idx holds A_local[mm_idx] = A[mm_idx][nn_idx]
    // for mm_idx > nn_idx (the original off-diagonal elements).
    CUTE_UNROLL
    for (int mm_idx = 1; mm_idx < 16; ++mm_idx) {
      CUTE_UNROLL
      for (int nn_idx = 0; nn_idx < mm_idx; ++nn_idx) {
        float send_val    = static_cast<float>(A_load[nn_idx]);
        float recv_val    = sycl::group_broadcast(sg, send_val, mm_idx);
        if (sg_local_id == nn_idx) {
          A_local[mm_idx] = recv_val;
        }
      }
    }

    // Compute inverse column-by-column using the recurrence
    //   L_inv[mm][nn] = −( L[mm][nn] + Σ_{k=nn+1}^{mm-1} L_inv[k][nn]·L[mm][k] )
    // Here A_local[mm] for thread nn = the inverse element at (mm, nn).
    // A_other[e] = broadcast of A_local[mm_idx] from thread e = A[mm_idx][e].
    CUTE_UNROLL
    for (int mm_idx = 1; mm_idx < 16; ++mm_idx) {
      float A_sum = 0.0f;
      CUTE_UNROLL
      for (int e = 1; e < mm_idx + 1; ++e) {
        A_other[e] = sycl::group_broadcast(sg, A_local[mm_idx], e);
      }
      CUTE_UNROLL
      for (int e = 1; e < mm_idx + 1; ++e) {
        A_sum += A_local[e] * A_other[e];
      }
      A_local[mm_idx] = -A_local[mm_idx] - A_sum;
    }

    // Write back sub-diagonal inverse elements (diagonal stays 1).
    CUTE_UNROLL
    for (int e = sg_local_id + 1; e < 16; ++e) {
      A_ptr_xx[e * chunk_size + sg_local_id] = static_cast<T>(A_local[e]);
    }
  }

  // -----------------------------------------------------------------------
  // Tensor helpers.
  //
  // Row-major tensor (M×K): make_stride(chunk_size, _1{})
  // Col-major tensor (N×K): make_stride(_1{}, chunk_size)
  //   – this is the "transposed" layout consumed by gemm_TTS as its B argument.
  //     gemm_TTS(A_row, B_col, C) computes  C += A · B  (standard product).
  // -----------------------------------------------------------------------
  auto A_XX_shape = make_shape(Int<16>{}, Int<16>{});

  // Helper: row-major (M×K) view starting at ptr
  auto make_row_tensor = [&](int row_off, int col_off) {
    return make_tensor(
        make_gmem_ptr(A_ptr + row_off * chunk_size + col_off),
        make_layout(A_XX_shape, make_stride(chunk_size, _1{})));
  };

  // Helper: col-major (N×K) view of the same memory = "transposed" layout
  auto make_col_tensor = [&](int row_off, int col_off) {
    return make_tensor(
        make_gmem_ptr(A_ptr + row_off * chunk_size + col_off),
        make_layout(A_XX_shape, make_stride(_1{}, chunk_size)));
  };

  // Row-major tensors (used as the A argument of gemm_TTS)
  auto A_11_tensor   = make_row_tensor( 0,  0);  // [L^{-1}]_11  (already inverted)
  auto A_21_tensor   = make_row_tensor(16,  0);  // L_21 originally; [L^{-1}]_21 after step 2
  auto A_22_tensor   = make_row_tensor(16, 16);  // [L^{-1}]_22  (already inverted)
  auto A_31_tensor   = make_row_tensor(32,  0);
  auto A_32_tensor   = make_row_tensor(32, 16);
  auto A_33_tensor   = make_row_tensor(32, 32);  // [L^{-1}]_33  (already inverted)
  auto A_41_tensor   = make_row_tensor(48,  0);
  auto A_42_tensor   = make_row_tensor(48, 16);
  auto A_43_tensor   = make_row_tensor(48, 32);
  auto A_44_tensor   = make_row_tensor(48, 48);  // [L^{-1}]_44  (already inverted)

  // Col-major (transposed) tensors (used as the B argument of gemm_TTS)
  auto A_11_tensor_T = make_col_tensor( 0,  0);
  auto A_21_tensor_T = make_col_tensor(16,  0);
  auto A_22_tensor_T = make_col_tensor(16, 16);
  auto A_31_tensor_T = make_col_tensor(32,  0);
  auto A_32_tensor_T = make_col_tensor(32, 16);
  auto A_33_tensor_T = make_col_tensor(32, 32);
  auto A_41_tensor_T = make_col_tensor(48,  0);
  auto A_42_tensor_T = make_col_tensor(48, 16);
  auto A_43_tensor_T = make_col_tensor(48, 32);

  // Coordinate / identity tensors for partitioning.
  Tensor cA = make_identity_tensor(A_XX_shape);
  Tensor cB = make_identity_tensor(A_XX_shape);
  Tensor cC = make_identity_tensor(A_XX_shape);
  Tensor gA = local_tile(cA, select<0,2>(wg_tile), make_coord(0, _));
  Tensor gB = local_tile(cB, select<1,2>(wg_tile), make_coord(0, _));
  Tensor gC = local_tile(cC, wg_tile, make_coord(0, 0, 0), Step<_1, _1, X>{});

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
  auto tCrC = thr_mma.partition_sg_fragment_C(gC);

  // -----------------------------------------------------------------------
  // Step 2: A21_inv = −A22_inv · A21 · A11_inv
  //   1. temp  = A22_inv · A21          → gemm_TTS(A22, A21_T, tCrC)
  //   2. result= temp    · A11_inv      → gemm_STS(tCrC, A11_T, tCrC2)
  //   3. negate and write to A21.
  // -----------------------------------------------------------------------
  auto copy_D_21    = get_block_2d_copy_D<void>(mma, A_21_tensor);
  auto thr_copy_D_21 = copy_D_21.get_slice(local_id);
  auto tCrD_21       = thr_copy_D_21.partition_sg_fragment_S(gC);
  auto tCgD_21       = thr_copy_D_21.partition_D(gC);

  clear(tCrC);
  gemm_TTS(A_22_tensor, A_21_tensor_T, tCrC, 0, 0, mma);
  reorder(tCrC, tCrA);     // save intermediate in tCrA
  clear(tCrC);
  gemm_STS(tCrA, A_11_tensor_T, tCrC, 0, 0, mma);
  CUTE_UNROLL
  for (int i = 0; i < tCrC.size(); ++i) tCrC(i) *= -1.0f;
  reorder(tCrC, tCrD_21);
  copy(copy_D_21, tCrD_21, tCgD_21);
  // A21 memory now holds [L^{-1}]_21.

  // -----------------------------------------------------------------------
  // Step 3a: A31 stage-1 – compute temp31 and write to A31 memory.
  //   temp31 = A31·A11_inv + A32·A21_inv
  // -----------------------------------------------------------------------
  auto copy_D_31    = get_block_2d_copy_D<void>(mma, A_31_tensor);
  auto thr_copy_D_31 = copy_D_31.get_slice(local_id);
  auto tCrD_31       = thr_copy_D_31.partition_sg_fragment_S(gC);
  auto tCgD_31       = thr_copy_D_31.partition_D(gC);

  clear(tCrC);
  gemm_TTS(A_31_tensor, A_11_tensor_T, tCrC, 0, 0, mma);
  gemm_TTS(A_32_tensor, A_21_tensor_T, tCrC, 0, 0, mma);
  reorder(tCrC, tCrD_31);
  copy(copy_D_31, tCrD_31, tCgD_31);
  // A31 memory now holds temp31.

  // Step 3b: A31 stage-2 – A31_inv = −A33_inv · temp31.
  //   A31_tensor_T now reads temp31 in col-major layout.
  clear(tCrC);
  gemm_TTS(A_33_tensor, A_31_tensor_T, tCrC, 0, 0, mma);
  CUTE_UNROLL
  for (int i = 0; i < tCrC.size(); ++i) tCrC(i) *= -1.0f;
  reorder(tCrC, tCrD_31);
  copy(copy_D_31, tCrD_31, tCgD_31);
  // A31 memory now holds [L^{-1}]_31.

  // -----------------------------------------------------------------------
  // Step 4a: A41 stage-1 – compute temp41 and write to A41 memory.
  //   temp41 = A41·A11_inv + A42·A21_inv + A43·A31_inv
  //   A31_tensor_T now reads [L^{-1}]_31 (written in step 3b).
  // -----------------------------------------------------------------------
  auto copy_D_41    = get_block_2d_copy_D<void>(mma, A_41_tensor);
  auto thr_copy_D_41 = copy_D_41.get_slice(local_id);
  auto tCrD_41       = thr_copy_D_41.partition_sg_fragment_S(gC);
  auto tCgD_41       = thr_copy_D_41.partition_D(gC);

  clear(tCrC);
  gemm_TTS(A_41_tensor, A_11_tensor_T, tCrC, 0, 0, mma);
  gemm_TTS(A_42_tensor, A_21_tensor_T, tCrC, 0, 0, mma);
  gemm_TTS(A_43_tensor, A_31_tensor_T, tCrC, 0, 0, mma);
  reorder(tCrC, tCrD_41);
  copy(copy_D_41, tCrD_41, tCgD_41);
  // A41 memory now holds temp41.

  // Step 4b: A41 stage-2 – A41_inv = −A44_inv · temp41.
  //   A41_tensor_T now reads temp41 in col-major layout.
  clear(tCrC);
  gemm_TTS(A_44_tensor, A_41_tensor_T, tCrC, 0, 0, mma);
  CUTE_UNROLL
  for (int i = 0; i < tCrC.size(); ++i) tCrC(i) *= -1.0f;
  reorder(tCrC, tCrD_41);
  copy(copy_D_41, tCrD_41, tCgD_41);
  // A41 memory now holds [L^{-1}]_41.

  // -----------------------------------------------------------------------
  // Step 5: A32_inv = −A33_inv · A32 · A22_inv
  // -----------------------------------------------------------------------
  auto copy_D_32    = get_block_2d_copy_D<void>(mma, A_32_tensor);
  auto thr_copy_D_32 = copy_D_32.get_slice(local_id);
  auto tCrD_32       = thr_copy_D_32.partition_sg_fragment_S(gC);
  auto tCgD_32       = thr_copy_D_32.partition_D(gC);

  clear(tCrC);
  gemm_TTS(A_33_tensor, A_32_tensor_T, tCrC, 0, 0, mma);
  reorder(tCrC, tCrA);
  clear(tCrC);
  gemm_STS(tCrA, A_22_tensor_T, tCrC, 0, 0, mma);
  CUTE_UNROLL
  for (int i = 0; i < tCrC.size(); ++i) tCrC(i) *= -1.0f;
  reorder(tCrC, tCrD_32);
  copy(copy_D_32, tCrD_32, tCgD_32);
  // A32 memory now holds [L^{-1}]_32.

  // -----------------------------------------------------------------------
  // Step 6a: A42 stage-1 – compute temp42 and write to A42 memory.
  //   temp42 = A42·A22_inv + A43·A32_inv
  //   A32_tensor_T now reads [L^{-1}]_32 (written in step 5).
  // -----------------------------------------------------------------------
  auto copy_D_42    = get_block_2d_copy_D<void>(mma, A_42_tensor);
  auto thr_copy_D_42 = copy_D_42.get_slice(local_id);
  auto tCrD_42       = thr_copy_D_42.partition_sg_fragment_S(gC);
  auto tCgD_42       = thr_copy_D_42.partition_D(gC);

  clear(tCrC);
  gemm_TTS(A_42_tensor, A_22_tensor_T, tCrC, 0, 0, mma);
  gemm_TTS(A_43_tensor, A_32_tensor_T, tCrC, 0, 0, mma);
  reorder(tCrC, tCrD_42);
  copy(copy_D_42, tCrD_42, tCgD_42);
  // A42 memory now holds temp42.

  // Step 6b: A42 stage-2 – A42_inv = −A44_inv · temp42.
  //   A42_tensor_T now reads temp42 in col-major layout.
  clear(tCrC);
  gemm_TTS(A_44_tensor, A_42_tensor_T, tCrC, 0, 0, mma);
  CUTE_UNROLL
  for (int i = 0; i < tCrC.size(); ++i) tCrC(i) *= -1.0f;
  reorder(tCrC, tCrD_42);
  copy(copy_D_42, tCrD_42, tCgD_42);
  // A42 memory now holds [L^{-1}]_42.

  // -----------------------------------------------------------------------
  // Step 7: A43_inv = −A44_inv · A43 · A33_inv
  // -----------------------------------------------------------------------
  auto copy_D_43    = get_block_2d_copy_D<void>(mma, A_43_tensor);
  auto thr_copy_D_43 = copy_D_43.get_slice(local_id);
  auto tCrD_43       = thr_copy_D_43.partition_sg_fragment_S(gC);
  auto tCgD_43       = thr_copy_D_43.partition_D(gC);

  clear(tCrC);
  gemm_TTS(A_44_tensor, A_43_tensor_T, tCrC, 0, 0, mma);
  reorder(tCrC, tCrA);
  clear(tCrC);
  gemm_STS(tCrA, A_33_tensor_T, tCrC, 0, 0, mma);
  CUTE_UNROLL
  for (int i = 0; i < tCrC.size(); ++i) tCrC(i) *= -1.0f;
  reorder(tCrC, tCrD_43);
  copy(copy_D_43, tCrD_43, tCgD_43);
  // A43 memory now holds [L^{-1}]_43.
}

} // namespace chunk_inverse
