/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * This example is inspired by chunk_inverse_opt_kernel from:
 *   vllm-project/vllm-xpu-kernels
 *   csrc/xpu/gdn_attn/xe_2/chunk_gated_delta_rule_kernels_xe2.hpp
 *
 **************************************************************************************************/
/*! \file
    \brief Benchmark for batched 64x64 lower-triangular matrix inversion on Intel BMG GPU.

    This example benchmarks the inversion of a batch of lower-triangular matrices,
    following the block inverse algorithm from chunk_inverse_opt_kernel in
    vllm-project/vllm-xpu-kernels.

    The 64x64 matrix is decomposed into 4x4 blocks of 16x16:
      | A_11  0    0    0  |
      | A_21  A_22 0    0  |
      | A_31  A_32 A_33 0  |
      | A_41  A_42 A_43 A_44|

    Each diagonal block is first inverted using subgroup-parallel operations.
    Off-diagonal blocks of the inverse are then computed using gemm_TTS, following
    the block lower-triangular inversion formula:
      (L^{-1})_ij = -L_ii^{-1} * sum_{k=j}^{i-1} L_ik * (L^{-1})_kj   for i > j

    The kernel uses one SYCL subgroup (16 threads) per matrix.

    To build & run:
      $ ninja 14_bmg_chunk_inverse
      $ ./examples/sycl/14_bmg_chunk_inverse/14_bmg_chunk_inverse --batch=32 --iterations=100

    Call with --help for usage.
*/

#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/sycl_event_manager.hpp"

#include "helper.h"
#include "sycl_common.hpp"

#include "chunk_inverse_gemm.hpp"

#include <cfloat>
#include <cmath>
#include <iostream>
#include <vector>

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

using namespace cute;

// ─── Element types ───────────────────────────────────────────────────────────
using ElementA   = bfloat16_t; // matrix element type (BF16)
using ElementAcc = float;      // DPAS accumulator (float, standard for BF16 DPAS)

// ─── MMA policy mirroring chunk_gemm_policy_16x16x16 from vllm ──────────────
// WGTile  = 16 x 16 x 16  (M x N x K per work-group)
// SGLayout = 1 x 1 x 1    (single subgroup per work-group)
// Atom: XE_DPAS_TT<8,...> produces 8 rows per DPAS op; TiledMMAHelper tiles
// this 2x in M to cover the 16-row WGTile with one subgroup.
using WGTile16   = Shape<_16, _16, _16>;
using SGLayout16 = Layout<Shape<_1, _1, _1>, Stride<_1, _1, _0>>;

using TiledMma16 = typename TiledMMAHelper<
    MMA_Atom<XE_DPAS_TT<8, ElementAcc, ElementA>>,
    Layout<WGTile16>,
    SGLayout16>::TiledMMA;

// ─── Fixed matrix dimension ───────────────────────────────────────────────────
static constexpr int CHUNK_SIZE = 64; // 64 x 64 lower-triangular matrices

// ─── Command-line options ─────────────────────────────────────────────────────
struct Options {
  bool help  = false;
  bool error = false;

  int batch;       // number of matrices in the batch
  int iterations;  // benchmark loop count

  Options() : help(false), error(false), batch(32), iterations(100) {}

  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("batch",      batch,      32);
    cmd.get_cmd_line_argument("iterations", iterations, 100);

    if (batch <= 0) {
      std::cerr << "Error: --batch must be positive\n";
      error = true;
    }
    if (iterations <= 0) {
      std::cerr << "Error: --iterations must be positive\n";
      error = true;
    }
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "BMG Chunk Inverse Benchmark\n\n"
        << "Benchmarks batched 64x64 lower-triangular matrix inversion using\n"
        << "the block-inverse algorithm from chunk_inverse_opt_kernel, with\n"
        << "gemm_TTS for off-diagonal block updates.\n\n"
        << "Matrix dtype: BF16  |  Accumulator: FP32\n\n"
        << "Options:\n\n"
        << "  --help                      Display this usage statement\n\n"
        << "  --batch=<int>               Number of 64x64 matrices in the batch (default: 32)\n"
        << "  --iterations=<int>          Number of benchmark iterations (default: 100)\n\n"
        << "Examples:\n\n"
        << "  $ 14_bmg_chunk_inverse --batch=64 --iterations=200\n\n";
    return out;
  }
};

// ─── Device kernel ────────────────────────────────────────────────────────────
//
// Adapted from chunk_inverse_opt_kernel in:
//   vllm-project/vllm-xpu-kernels
//   csrc/xpu/gdn_attn/xe_2/chunk_gated_delta_rule_kernels_xe2.hpp
//
// Each work-group (one subgroup, 16 threads) inverts one 64x64 lower-triangular
// matrix in-place.
//
// The 64x64 matrix is partitioned into 4x4 blocks of 16x16:
//   block (i,j) starts at row i*16, col j*16.
//
// Step 1: Invert the four 16x16 diagonal blocks (A_11, A_22, A_33, A_44) using
//         explicit subgroup-parallel forward-substitution.
//
// Step 2: Compute all off-diagonal blocks of L^{-1} using gemm_TTS, following
//         the block lower-triangular inversion recurrence.
//
template <typename T, class TiledMMA>
CUTE_DEVICE void chunk_inverse_opt_kernel_device(T* A, int batch_idx) {
  auto item     = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int  local_id = item.get_local_linear_id();

  auto sg          = item.get_sub_group();
  int  sg_local_id = sg.get_local_linear_id();

  // Base pointer for this matrix in the batch
  T* A_ptr = A + static_cast<int64_t>(batch_idx) * CHUNK_SIZE * CHUNK_SIZE;

  // ── Step 1: invert each 16x16 diagonal block in-place ────────────────────
  //
  // Uses the same sequential forward-substitution as the original kernel.
  // Each lane (sg_local_id) owns one column of the 16x16 block.
  // The loop over 4 diagonal blocks is fully unrolled.
  CUTE_UNROLL
  for (int blk = 0; blk < 4; ++blk) {
    int offset      = blk * 16;
    T*  A_ptr_diag  = A_ptr + offset * CHUNK_SIZE + offset;

    // Local storage for this lane's column of the block
    float A_local[16] = {};

    // Load the strictly lower-triangular part of this 16x16 diagonal block
    // into registers. A_local[mm_idx] will hold L[mm_idx, sg_local_id].
    T A_load[16] = {};
    CUTE_UNROLL
    for (int e = 0; e < sg_local_id; ++e) {
      A_load[e] = A_ptr_diag[sg_local_id * CHUNK_SIZE + e];
    }

    // Broadcast the loaded values so each lane has the full row it needs
    CUTE_UNROLL
    for (int mm_idx = 1; mm_idx < 16; ++mm_idx) {
      CUTE_UNROLL
      for (int nn_idx = 0; nn_idx < mm_idx; ++nn_idx) {
        float send_value    = static_cast<float>(A_load[nn_idx]);
        float receive_value = sycl::group_broadcast(sg, send_value, mm_idx);
        if (sg_local_id == nn_idx) {
          A_local[mm_idx] = receive_value;
        }
      }
    }

    // Compute the inverse column-by-column using forward substitution.
    // A_local[mm_idx] ends up holding (L^{-1})[mm_idx, sg_local_id].
    float A_other[16];
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

    // Write back the inverse entries (strictly lower-triangular part)
    CUTE_UNROLL
    for (int e = sg_local_id + 1; e < 16; ++e) {
      A_ptr_diag[e * CHUNK_SIZE + sg_local_id] = static_cast<T>(A_local[e]);
    }
  }

  // ── Step 2: off-diagonal blocks via gemm_TTS ─────────────────────────────
  //
  // After Step 1, A_ii blocks hold L_ii^{-1} (with diagonal 1 by construction).
  // We now compute the off-diagonal blocks of L^{-1}.
  //
  // Notation: A_ij_tensor    = L_ij stored row-major  (stride = (CHUNK_SIZE, 1))
  //           A_ij_tensor_T  = L_ij stored col-major  (stride = (1, CHUNK_SIZE))
  //
  // All 16x16 sub-blocks share the same shape.
  auto A_XX_shape = make_shape(_16{}, _16{});

  // Row-major 16x16 sub-block at (row_off, col_off) within A_ptr
  auto make_row_tensor = [&](int row_off, int col_off) {
    return make_tensor(
        make_gmem_ptr(A_ptr + row_off * CHUNK_SIZE + col_off),
        make_layout(A_XX_shape, make_stride(Int<CHUNK_SIZE>{}, _1{})));
  };
  // Col-major (transposed) 16x16 sub-block at (row_off, col_off)
  auto make_col_tensor = [&](int row_off, int col_off) {
    return make_tensor(
        make_gmem_ptr(A_ptr + row_off * CHUNK_SIZE + col_off),
        make_layout(A_XX_shape, make_stride(_1{}, Int<CHUNK_SIZE>{})));
  };

  // Diagonal inverse blocks (col-major view, used as "B" in gemm_TTS / gemm_STS)
  auto A_11_tensor_T = make_col_tensor( 0,  0);
  auto A_22_tensor_T = make_col_tensor(16, 16);
  auto A_33_tensor_T = make_col_tensor(32, 32);

  // Off-diagonal original blocks (row-major, used as "A" in gemm_TTS)
  auto A_21_tensor   = make_row_tensor(16,  0);
  auto A_21_tensor_T = make_col_tensor(16,  0);
  auto A_22_tensor   = make_row_tensor(16, 16);
  auto A_31_tensor   = make_row_tensor(32,  0);
  auto A_31_tensor_T = make_col_tensor(32,  0);
  auto A_32_tensor   = make_row_tensor(32, 16);
  auto A_32_tensor_T = make_col_tensor(32, 16);
  auto A_33_tensor   = make_row_tensor(32, 32);
  auto A_41_tensor   = make_row_tensor(48,  0);
  auto A_42_tensor   = make_row_tensor(48, 16);
  auto A_43_tensor   = make_row_tensor(48, 32);
  auto A_44_tensor   = make_row_tensor(48, 48);
  auto A_43_tensor_T = make_col_tensor(48, 32);

  // Output (row-major) tensors for the off-diagonal inverse blocks
  auto A_21_out = make_row_tensor(16,  0);
  auto A_31_out = make_row_tensor(32,  0);
  auto A_32_out = make_row_tensor(32, 16);
  auto A_41_out = make_row_tensor(48,  0);
  auto A_42_out = make_row_tensor(48, 16);
  auto A_43_out = make_row_tensor(48, 32);

  TiledMMA mma{};
  auto wg_tile = mma.tile_mnk();
  auto thr_mma = mma.get_slice(local_id);

  // Create identity tensors for partitioning the C tiles
  Tensor cA = make_identity_tensor(A_XX_shape);
  Tensor cB = make_identity_tensor(A_XX_shape);
  Tensor cC = make_identity_tensor(A_XX_shape);

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(0, _));
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(0, _));
  Tensor gC = local_tile(cC, wg_tile, make_coord(0, 0, 0), Step<_1, _1, X>{});

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrC = thr_mma.partition_sg_fragment_C(gC);

  // Helper: build copy-D operation and its partitioned fragments for storing result
  auto make_copy_d = [&](auto& out_tensor) {
    return get_block_2d_copy_D<void>(mma, out_tensor);
  };

  // ── (2,1): A_21_inv = -(A_22_inv * A_21 * A_11_inv) ─────────────────────
  {
    auto copy_D  = make_copy_d(A_21_out);
    auto thr_copy_D = copy_D.get_slice(local_id);
    auto tCrD    = thr_copy_D.partition_sg_fragment_S(gC);
    auto tCgD    = thr_copy_D.partition_D(gC);

    // temp = A_22_inv * A_21
    clear(tCrC);
    chunk_inverse::gemm_TTS(A_22_tensor, A_21_tensor_T, tCrC, 0, 0, mma);
    reorder(tCrC, tCrA);

    // result = temp * A_11_inv
    clear(tCrC);
    chunk_inverse::gemm_STS(tCrA, A_11_tensor_T, tCrC, 0, 0, mma);

    CUTE_UNROLL
    for (int i = 0; i < tCrC.size(); ++i) tCrC(i) *= -1.0f;

    reorder(tCrC, tCrD);
    copy(copy_D, tCrD, tCgD);
  }

  // ── (3,1): A_31_inv ───────────────────────────────────────────────────────
  {
    auto copy_D     = make_copy_d(A_31_out);
    auto thr_copy_D = copy_D.get_slice(local_id);
    auto tCrD       = thr_copy_D.partition_sg_fragment_S(gC);
    auto tCgD       = thr_copy_D.partition_D(gC);

    // temp = A_33_inv * A_31 + A_33_inv * A_32 * A_21_inv
    //      = A_33_inv * (A_31 + A_32 * A_21_inv)
    // We accumulate: first A_31_tensor then A_32 * A_21_inv (i.e. A_32 * A_21_out)
    clear(tCrC);
    chunk_inverse::gemm_TTS(A_31_tensor, A_11_tensor_T, tCrC, 0, 0, mma);
    chunk_inverse::gemm_TTS(A_32_tensor, A_21_tensor_T, tCrC, 0, 0, mma);
    reorder(tCrC, tCrD);
    copy(copy_D, tCrD, tCgD);

    // Now compute -(A_33_inv * temp)
    clear(tCrC);
    chunk_inverse::gemm_TTS(A_33_tensor, A_31_tensor_T, tCrC, 0, 0, mma);
    CUTE_UNROLL
    for (int i = 0; i < tCrC.size(); ++i) tCrC(i) *= -1.0f;
    reorder(tCrC, tCrD);
    copy(copy_D, tCrD, tCgD);
  }

  // ── (4,1): A_41_inv ───────────────────────────────────────────────────────
  {
    auto copy_D     = make_copy_d(A_41_out);
    auto thr_copy_D = copy_D.get_slice(local_id);
    auto tCrD       = thr_copy_D.partition_sg_fragment_S(gC);
    auto tCgD       = thr_copy_D.partition_D(gC);

    clear(tCrC);
    chunk_inverse::gemm_TTS(A_41_tensor, A_11_tensor_T, tCrC, 0, 0, mma);
    chunk_inverse::gemm_TTS(A_42_tensor, A_21_tensor_T, tCrC, 0, 0, mma);
    chunk_inverse::gemm_TTS(A_43_tensor, A_31_tensor_T, tCrC, 0, 0, mma);
    reorder(tCrC, tCrD);
    copy(copy_D, tCrD, tCgD);

    clear(tCrC);
    chunk_inverse::gemm_TTS(A_44_tensor, A_41_tensor_T, tCrC, 0, 0, mma);
    CUTE_UNROLL
    for (int i = 0; i < tCrC.size(); ++i) tCrC(i) *= -1.0f;
    reorder(tCrC, tCrD);
    copy(copy_D, tCrD, tCgD);
  }

  // ── (3,2): A_32_inv ───────────────────────────────────────────────────────
  {
    auto copy_D     = make_copy_d(A_32_out);
    auto thr_copy_D = copy_D.get_slice(local_id);
    auto tCrD       = thr_copy_D.partition_sg_fragment_S(gC);
    auto tCgD       = thr_copy_D.partition_D(gC);

    // temp = A_33_inv * A_32
    clear(tCrC);
    chunk_inverse::gemm_TTS(A_33_tensor, A_32_tensor_T, tCrC, 0, 0, mma);
    reorder(tCrC, tCrA);

    // result = -(temp * A_22_inv)
    clear(tCrC);
    chunk_inverse::gemm_STS(tCrA, A_22_tensor_T, tCrC, 0, 0, mma);
    CUTE_UNROLL
    for (int i = 0; i < tCrC.size(); ++i) tCrC(i) *= -1.0f;
    reorder(tCrC, tCrD);
    copy(copy_D, tCrD, tCgD);
  }

  // ── (4,2): A_42_inv ───────────────────────────────────────────────────────
  {
    auto copy_D     = make_copy_d(A_42_out);
    auto thr_copy_D = copy_D.get_slice(local_id);
    auto tCrD       = thr_copy_D.partition_sg_fragment_S(gC);
    auto tCgD       = thr_copy_D.partition_D(gC);

    clear(tCrC);
    chunk_inverse::gemm_TTS(A_42_tensor, A_22_tensor_T, tCrC, 0, 0, mma);
    chunk_inverse::gemm_TTS(A_43_tensor, A_32_tensor_T, tCrC, 0, 0, mma);
    reorder(tCrC, tCrD);
    copy(copy_D, tCrD, tCgD);

    clear(tCrC);
    chunk_inverse::gemm_TTS(A_44_tensor, A_42_tensor_T, tCrC, 0, 0, mma);
    CUTE_UNROLL
    for (int i = 0; i < tCrC.size(); ++i) tCrC(i) *= -1.0f;
    reorder(tCrC, tCrD);
    copy(copy_D, tCrD, tCgD);
  }

  // ── (4,3): A_43_inv ───────────────────────────────────────────────────────
  {
    auto copy_D     = make_copy_d(A_43_out);
    auto thr_copy_D = copy_D.get_slice(local_id);
    auto tCrD       = thr_copy_D.partition_sg_fragment_S(gC);
    auto tCgD       = thr_copy_D.partition_D(gC);

    // temp = A_44_inv * A_43
    clear(tCrC);
    chunk_inverse::gemm_TTS(A_44_tensor, A_43_tensor_T, tCrC, 0, 0, mma);
    reorder(tCrC, tCrA);

    // result = -(temp * A_33_inv)
    clear(tCrC);
    chunk_inverse::gemm_STS(tCrA, A_33_tensor_T, tCrC, 0, 0, mma);
    CUTE_UNROLL
    for (int i = 0; i < tCrC.size(); ++i) tCrC(i) *= -1.0f;
    reorder(tCrC, tCrD);
    copy(copy_D, tCrD, tCgD);
  }
}

// ─── Kernel name tag (needed for SYCL parallel_for) ─────────────────────────
struct ChunkInverseKernel {};

// ─── Kernel launcher ─────────────────────────────────────────────────────────
void run_chunk_inverse(ElementA* d_A, int batch_size) {
  // One subgroup (16 threads) per matrix
  constexpr int wg_size = 16;

  // 3D range: [1, batch_size, wg_size]  →  groups: [1, batch_size, 1]
  sycl::range<3> local_range{1, 1, wg_size};
  sycl::range<3> global_range{1, static_cast<size_t>(batch_size), wg_size};

  namespace syclex  = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{
      syclex::sub_group_size<16>,
      intelex::grf_size<256>};

  sycl::queue Q = compat::get_default_queue();

  auto event = Q.parallel_for<ChunkInverseKernel>(
      sycl::nd_range<3>(global_range, local_range),
      kernel_props,
      [=](auto /*item*/) {
        auto nd_item  = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int  batch_id = static_cast<int>(nd_item.get_group(1));
        chunk_inverse_opt_kernel_device<ElementA, TiledMma16>(d_A, batch_id);
      });

  EventManager::getInstance().addEvent(event);
}

// ─── Host-side reference: lower-triangular inversion (for validation) ────────
// Inverts an n×n lower-triangular matrix L in-place (L has unit diagonal entries
// that are set to 1 during initialization).
static void ref_lower_tri_inverse(std::vector<float>& L, int n) {
  // Use standard blocked forward substitution: B = -L^{-1}_{jj} * L_{ij}
  for (int j = 0; j < n; ++j) {
    // L[j][j] == 1.0 (unit diagonal)
    for (int i = j + 1; i < n; ++i) {
      float sum = -L[i * n + j];
      for (int k = j + 1; k < i; ++k) {
        sum -= L[i * n + k] * L[k * n + j];
      }
      L[i * n + j] = sum; // (L^{-1})[i][j]
    }
  }
}

// ─── ExampleRunner ───────────────────────────────────────────────────────────
struct ExampleRunner {
  int batch_size;
  static constexpr int N = CHUNK_SIZE;

  // Device allocation: batch_size × N × N elements
  cutlass::DeviceAllocation<ElementA> d_A;
  cutlass::DeviceAllocation<ElementA> d_A_orig; // saved copy for validation

  // Host buffers
  std::vector<float> h_L;      // original lower-tri matrices (float)
  std::vector<float> h_L_inv;  // computed inverse (read back)

  explicit ExampleRunner(int batch) : batch_size(batch) {}

  // Initialize: create B invertible lower-triangular matrices.
  // We use a deterministic scheme:
  //   - Diagonal entries = 1.0 (unit diagonal → easy inversion)
  //   - Sub-diagonal entries: small random-ish values derived from position
  void initialize() {
    const int64_t total = static_cast<int64_t>(batch_size) * N * N;
    d_A.reset(total);
    d_A_orig.reset(total);

    h_L.assign(total, 0.0f);

    for (int b = 0; b < batch_size; ++b) {
      float* L = h_L.data() + static_cast<int64_t>(b) * N * N;
      for (int i = 0; i < N; ++i) {
        L[i * N + i] = 1.0f; // unit diagonal
        for (int j = 0; j < i; ++j) {
          // Deterministic sub-diagonal: small values in (-0.5, 0.5)
          float val = 0.3f * std::sin(static_cast<float>((b * N + i) * N + j));
          L[i * N + j] = val;
        }
      }
    }

    // Convert float host data to BF16 and upload
    std::vector<ElementA> h_bf16(total);
    for (int64_t i = 0; i < total; ++i) {
      h_bf16[i] = static_cast<ElementA>(h_L[i]);
    }

    d_A.copy_from_host(h_bf16.data());
    d_A_orig.copy_from_host(h_bf16.data());
  }

  // Restore d_A from d_A_orig (needed before each benchmark run)
  void restore() {
    const int64_t total = static_cast<int64_t>(batch_size) * N * N;
    compat::get_default_queue()
        .memcpy(d_A.get(), d_A_orig.get(), total * sizeof(ElementA))
        .wait();
  }

  // Validate: for the first matrix, check that L_orig * L_inv ≈ I
  bool verify() {
    const int64_t total = static_cast<int64_t>(batch_size) * N * N;

    // Read back computed inverse
    std::vector<ElementA> h_inv_bf16(total);
    d_A.copy_to_host(h_inv_bf16.data());
    compat::wait();

    // Compute reference inverse for first matrix
    std::vector<float> L_ref(h_L.begin(), h_L.begin() + N * N);
    ref_lower_tri_inverse(L_ref, N);

    // Compute L_orig[0] * L_inv_computed[0] and check ≈ I
    const float* L_orig = h_L.data(); // first matrix
    constexpr float atol = 0.05f;     // BF16 has limited precision

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j <= i; ++j) {
        float acc = 0.0f;
        for (int k = j; k <= i; ++k) {
          acc += L_orig[i * N + k] * static_cast<float>(h_inv_bf16[k * N + j]);
        }
        float expected = (i == j) ? 1.0f : 0.0f;
        if (std::abs(acc - expected) > atol) {
          std::cerr << "Validation failed at (" << i << "," << j << "): "
                    << "got " << acc << ", expected " << expected << "\n";
          return false;
        }
      }
    }
    return true;
  }

  // Run warmup + benchmark
  cutlass::Status run(const Options& options) {
    initialize();

    constexpr int warmup = 3;

    // ── Warmup ──────────────────────────────────────────────────────────────
    for (int iter = 0; iter < warmup; ++iter) {
      restore();
      run_chunk_inverse(d_A.get(), batch_size);
      compat::wait();
    }

    // ── Validation (on the result of the last warmup run) ────────────────────
    bool passed = verify();
    std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << "\n";
    if (!passed) return cutlass::Status::kErrorInternal;

    // ── Benchmark: time the kernel across many iterations ────────────────────
    // Restore once so the first iteration operates on valid input data.
    // Subsequent iterations run on already-inverted data (effectively double-
    // inversion), but the GPU code path and computational load are identical,
    // giving an accurate throughput measurement.
    restore();
    GPU_Clock timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      run_chunk_inverse(d_A.get(), batch_size);
    }
    compat::wait();

    float total_ms      = timer.seconds() * 1000.0f;
    double avg_ms       = static_cast<double>(total_ms) / options.iterations;

    // Throughput: matrices per second
    double matrices_per_s = static_cast<double>(batch_size) / (avg_ms * 1e-3);

    // Approximate FLOP count: roughly O(n^3/3) FLOPs per lower-tri inverse
    // We use 2/3 * n^3 (standard estimate for triangular system solves)
    double flops_per_matrix = (2.0 / 3.0) * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
    double total_gflops =
        flops_per_matrix * batch_size / (avg_ms * 1e-3) / 1e9;

    std::cout << "\n";
    std::cout << "  Batch size        : " << batch_size << "\n";
    std::cout << "  Matrix size       : " << N << " x " << N << " (BF16)\n";
    std::cout << "  Warmup iterations : " << warmup << "\n";
    std::cout << "  Benchmark iters   : " << options.iterations << "\n";
    std::cout << "  Avg runtime       : " << avg_ms << " ms\n";
    std::cout << "  Matrices/s        : " << matrices_per_s << "\n";
    std::cout << "  GFLOPS (approx)   : " << total_gflops << "\n";

    return cutlass::Status::kSuccess;
  }
};

// ─── main ────────────────────────────────────────────────────────────────────
int main(int argc, const char** argv) {
  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution.\n";
    return -1;
  }

  ExampleRunner runner(options.batch);
  CUTLASS_CHECK(runner.run(options));

  return 0;
}
