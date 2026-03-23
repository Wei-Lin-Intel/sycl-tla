/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
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
 **************************************************************************************************/
/*! \file
    \brief CUTLASS Intel BMG chunk-inverse benchmark.

    This example benchmarks in-place lower-triangular matrix inversion for
    B matrices of size 64×64 in BF16 precision.  The implementation follows
    the block-inverse strategy from `chunk_inverse_opt_kernel` in
    vllm-xpu-kernels, using `gemm_TTS`-based updates with Intel 2-D block-copy
    operations and DPAS instructions.

    Matrix layout:
      - Each of the B matrices is 64×64, stored row-major.
      - All matrices are unit lower-triangular (1 on diagonal).
      - Random BF16 values are placed in the strictly lower-triangular part.

    The kernel dispatches one work-group (16 threads / 1 sub-group) per matrix.

    To build & run (from your build directory):

      $ ninja 05_bmg_chunk_inverse
      $ ./examples/sycl/05_bmg_chunk_inverse/05_bmg_chunk_inverse \
            --batch=16 --iterations=100

    Call with `--help` for information about available options.
*/

#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"

#include "sycl_common.hpp"
#include "helper.h"

#include "chunk_inverse.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

using namespace cute;
using namespace chunk_inverse;

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------
using ElementA = bfloat16_t;

// TiledMMA for the 16×16×16 inverse kernel (1 sub-group, 16 threads).
// chunk_gemm_policy_16x16x16: WGTile=(16,16,16), SGLayout=(1,1,1).
using InverseMMA = typename TiledMMAHelper<
    MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
    Layout<Shape<_16, _16, _16>>,
    Layout<Shape<_1, _1, _1>, Stride<_1, _1, _0>>>::TiledMMA;

static constexpr int kMatrixDim = 64;   // fixed 64×64

// ---------------------------------------------------------------------------
// Command-line options
// ---------------------------------------------------------------------------
struct Options {
  bool help  = false;
  bool error = false;

  int batch      = 16;
  int iterations = 100;

  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("batch",      batch,      16);
    cmd.get_cmd_line_argument("iterations", iterations, 100);

    if (batch <= 0 || iterations <= 0) {
      std::cerr << "Error: --batch and --iterations must be positive.\n";
      error = true;
    }
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "BMG Chunk Inverse Benchmark (64×64 BF16 lower-triangular inverse)\n\n"
        << "Options:\n\n"
        << "  --help                   Display this usage statement\n"
        << "  --batch=<int>            Number of 64×64 matrices (default: 16)\n"
        << "  --iterations=<int>       Profiling iterations (default: 100)\n\n"
        << "Example:\n"
        << "  $ 05_bmg_chunk_inverse --batch=64 --iterations=200\n\n";
    return out;
  }

  // Estimated FLOPs for B inversions of an N×N lower-triangular matrix.
  // We use ~N^3/3 as a rough lower bound (similar to triangular solve).
  double gflops(double runtime_s) const {
    double N    = double(kMatrixDim);
    double flop = double(batch) * N * N * N / 3.0;
    return flop / 1.0e9 / runtime_s;
  }
};

// ---------------------------------------------------------------------------
// Kernel launcher kernel name tag
// ---------------------------------------------------------------------------
struct ChunkInverseKernel;

// ---------------------------------------------------------------------------
// Validation
//
// Compute L · L_inv on the CPU (float arithmetic) and verify that the result
// is close to the identity matrix.  Both L (the original matrix) and L_inv
// (the computed inverse) are BF16; we upcast to float for the check.
// ---------------------------------------------------------------------------
bool verify(const std::vector<ElementA>& L_host,
            const std::vector<ElementA>& L_inv_host,
            int batch)
{
  static constexpr int N = kMatrixDim;
  bool passed = true;

  for (int b = 0; b < batch && passed; ++b) {
    const ElementA* L     = L_host.data()     + b * N * N;
    const ElementA* L_inv = L_inv_host.data() + b * N * N;

    for (int row = 0; row < N && passed; ++row) {
      for (int col = 0; col < N && passed; ++col) {
        // Compute (L · L_inv)[row][col] in float.
        float val = 0.0f;
        for (int k = 0; k < N; ++k) {
          val += float(L[row * N + k]) * float(L_inv[k * N + col]);
        }
        float expected = (row == col) ? 1.0f : 0.0f;
        float diff     = std::abs(val - expected);
        // BF16 has ~7 decimal digits; allow generous tolerance.
        if (diff > 0.05f) {
          std::cout << "Validation failed at batch=" << b
                    << " (" << row << "," << col << ")"
                    << ": got " << val
                    << ", expected " << expected << "\n";
          passed = false;
        }
      }
    }
  }
  return passed;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char const** argv)
{
  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout);
    return 0;
  }
  if (options.error) {
    options.print_usage(std::cerr);
    return 1;
  }

  int batch      = options.batch;
  int iterations = options.iterations;
  int warmup     = 3;

  static constexpr int N = kMatrixDim;

  std::cout << "BMG Chunk Inverse Benchmark\n"
            << "  batch      : " << batch      << "\n"
            << "  matrix dim : " << N << "x" << N << "\n"
            << "  dtype      : BF16\n"
            << "  warmup     : " << warmup     << "\n"
            << "  iterations : " << iterations << "\n\n";

  // -------------------------------------------------------------------------
  // Generate random unit lower-triangular BF16 matrices on host.
  // -------------------------------------------------------------------------
  std::srand(42);
  std::vector<ElementA> h_A(batch * N * N, ElementA(0));

  for (int b = 0; b < batch; ++b) {
    ElementA* mat = h_A.data() + b * N * N;
    for (int r = 0; r < N; ++r) {
      mat[r * N + r] = ElementA(1.0f);            // unit diagonal
      for (int c = 0; c < r; ++c) {
        // Small random values to keep the matrix well-conditioned.
        float v    = (float(std::rand()) / float(RAND_MAX)) * 0.4f - 0.2f;
        mat[r * N + c] = ElementA(v);
      }
    }
  }

  // Save a copy of the original matrices for validation.
  std::vector<ElementA> h_L_orig = h_A;

  // -------------------------------------------------------------------------
  // Allocate device memory.
  // -------------------------------------------------------------------------
  cutlass::DeviceAllocation<ElementA> d_A(batch * N * N);

  // Upload.
  compat::get_default_queue()
      .memcpy(d_A.get(), h_A.data(), batch * N * N * sizeof(ElementA))
      .wait();

  // -------------------------------------------------------------------------
  // Kernel launch configuration.
  //   - 16 threads per work-group (1 sub-group)
  //   - batch work-groups (one per matrix)
  // -------------------------------------------------------------------------
  InverseMMA mma{};
  int local_threads = size(mma);   // should be 16

  sycl::range<3> local{1, 1, static_cast<size_t>(local_threads)};
  sycl::range<3> global{1, 1, static_cast<size_t>(local_threads * batch)};

  namespace syclex  = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties kernel_props{syclex::sub_group_size<16>,
                                  intelex::grf_size<256>};

  ElementA* A_ptr   = d_A.get();
  int       batch_n = batch;

  auto launch_kernel = [&]() {
    return compat::get_default_queue().parallel_for<ChunkInverseKernel>(
        sycl::nd_range<3>(global, local), kernel_props,
        [=](sycl::nd_item<3>) {
          chunk_inverse_opt_kernel<ElementA, InverseMMA>(A_ptr, batch_n);
        });
  };

  // -------------------------------------------------------------------------
  // Warmup runs.
  // -------------------------------------------------------------------------
  for (int i = 0; i < warmup; ++i) {
    // Re-upload the original data before each run (inverse is in-place).
    compat::get_default_queue()
        .memcpy(d_A.get(), h_L_orig.data(), batch * N * N * sizeof(ElementA))
        .wait();
    launch_kernel().wait();
  }

  // -------------------------------------------------------------------------
  // Validation: run once on a fresh upload and download the result.
  // -------------------------------------------------------------------------
  compat::get_default_queue()
      .memcpy(d_A.get(), h_L_orig.data(), batch * N * N * sizeof(ElementA))
      .wait();
  launch_kernel().wait();

  std::vector<ElementA> h_A_inv(batch * N * N);
  compat::get_default_queue()
      .memcpy(h_A_inv.data(), d_A.get(), batch * N * N * sizeof(ElementA))
      .wait();

  bool passed = verify(h_L_orig, h_A_inv, batch);
  if (!passed) {
    std::cout << "\nAccuracy check FAILED.\n";
    return 1;
  }
  std::cout << "Accuracy check passed.\n\n";

  // -------------------------------------------------------------------------
  // Timed runs (measure iteration-average latency).
  // -------------------------------------------------------------------------
  // Upload fresh data for the first timed run.
  compat::get_default_queue()
      .memcpy(d_A.get(), h_L_orig.data(), batch * N * N * sizeof(ElementA))
      .wait();

  GPU_Clock timer;
  timer.start();

  for (int i = 0; i < iterations; ++i) {
    // The kernel overwrites the input; re-upload each iteration so we measure
    // kernel time only (upload is not timed).
    if (i > 0) {
      compat::get_default_queue()
          .memcpy(d_A.get(), h_L_orig.data(),
                  batch * N * N * sizeof(ElementA))
          .wait();
    }
    launch_kernel().wait();
  }

  double total_ms = timer.seconds() * 1000.0;
  double avg_ms   = total_ms / double(iterations);
  double gflops   = options.gflops(avg_ms / 1000.0);

  std::cout << "Performance\n"
            << "  Avg latency : " << avg_ms  << " ms\n"
            << "  GFLOPs      : " << gflops  << " (estimated)\n";

  return 0;
}
