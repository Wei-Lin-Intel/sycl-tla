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
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * This file is derived from vllm-project/vllm-xpu-kernels (gemm.hpp),
 * adapted for use in the sycl-tla example suite.
 *
 **************************************************************************************************/
#pragma once

#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>
#include <cute/util/xe_split_barrier.hpp>
#include <cute/atom/copy_traits_xe_2d.hpp>

namespace chunk_inverse {

using namespace cute;

/// gemm_TTS: GEMM where A and B are Tensors in global memory (T = Tensor),
/// and C is accumulated in a register fragment (S = register/Subgroup-fragment).
///
/// Computes: tCrC += A * B (with B loaded via the DPAS B-path, which handles
/// the required VNNI transpose during loading).
///
/// @param A       Global memory tensor, shape (M, K)
/// @param B       Global memory tensor, shape (N, K) (loaded via copy_B path)
/// @param tCrC    In/out: accumulator fragment in subgroup register space
/// @param wg_m    Tile-coordinate in M dimension (which M-tile this WG covers)
/// @param wg_n    Tile-coordinate in N dimension (which N-tile this WG covers)
/// @param mma     The TiledMMA object describing the DPAS configuration
template <class ATensor, class BTensor, class SGCTensor, class TiledMMA>
CUTE_DEVICE void gemm_TTS(
    ATensor const& A,   // (M,K) global tensor
    BTensor const& B,   // (N,K) global tensor (loaded via B path)
    SGCTensor&     tCrC,// subgroup accumulator fragment
    int wg_m,
    int wg_n,
    TiledMMA const& mma)
{
  auto item     = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int  local_id = item.get_local_linear_id();

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());

  auto wg_tile = mma.tile_mnk();

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));

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

  constexpr int prefetch_dist  = 3;
  constexpr int barrier_scope  = 2;

  int k_tile_count    = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; ++k_tile_prefetch) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }

  for (int k_tile = 0; k_tile < k_tile_count; ++k_tile, ++k_tile_prefetch) {
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

/// gemm_STS: GEMM where A is already in a register fragment (S = subgroup
/// register) and B is a Tensor in global memory.
///
/// Computes: tCrC += tCrA * B.
///
/// @param tCrA    Input: A already in subgroup register fragment
/// @param B       Global memory tensor, shape (N, K) (loaded via B path)
/// @param tCrC    In/out: accumulator fragment
/// @param wg_m    Tile-coordinate in M dimension
/// @param wg_n    Tile-coordinate in N dimension
/// @param mma     The TiledMMA object
template <class ASGCTensor, class BTensor, class CSGCTensor, class TiledMMA>
CUTE_DEVICE void gemm_STS(
    ASGCTensor const& tCrA,  // A already in register fragment (M,K)
    BTensor    const& B,     // (N,K) global tensor
    CSGCTensor&       tCrC,  // subgroup accumulator fragment
    int wg_m,
    int wg_n,
    TiledMMA const& mma)
{
  auto item     = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int  local_id = item.get_local_linear_id();

  Tensor cB = make_identity_tensor(B.shape());

  auto wg_tile = mma.tile_mnk();

  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));

  auto copy_b     = get_block_2d_copy_B<void>(mma, B);
  auto thr_mma    = mma.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);

  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tBgB = thr_copy_b.partition_S(gB);

  auto prefetch_b     = make_block_2d_prefetch(copy_b);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);
  auto pBgB           = thr_prefetch_B.partition_S(gB);

  constexpr int prefetch_dist  = 3;
  constexpr int barrier_scope  = 2;

  int k_tile_count    = ceil_div(shape<1>(B), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; ++k_tile_prefetch) {
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }

  for (int k_tile = 0; k_tile < k_tile_count; ++k_tile, ++k_tile_prefetch) {
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

} // namespace chunk_inverse
