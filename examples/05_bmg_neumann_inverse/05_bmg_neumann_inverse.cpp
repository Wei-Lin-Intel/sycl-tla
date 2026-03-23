/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
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
    \brief CUTLASS Intel BMG Neumann iterative lower-triangular matrix inverse benchmark.

    This example benchmarks the inverse of B unit lower-triangular 64×64 BF16 matrices using
    the iterative Neumann algorithm ported from
    `python/opt_tril_inverse_triton.py` (`tril_inverse_iterative_64_bf16_kernel`).

    Algorithm (STEPS=14, fixed):
        inv = I
        for _ in range(STEPS):
            err = lower_strict(L * inv)   // strictly-lower-triangular part
            inv = inv - inv * err

    Kernel design:
      - One work-group (16 sub-groups × 16 threads = 256 threads) per matrix.
      - L, inv, and err are kept in shared local memory (SLM) across all 12 iterations,
        avoiding global-memory traffic for intermediate state.
      - The two GEMM products (L*inv and inv*err) are computed via Intel DPAS instructions
        through CuTe's TiledMMA / cute::gemm building blocks.

    To build & run (from your build directory):

      $ ninja 05_bmg_neumann_inverse
      $ ./examples/sycl/05_bmg_neumann_inverse/05_bmg_neumann_inverse \
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

#include "neumann_inverse.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

using namespace cute;
using namespace neumann_inverse;

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------
using ElementA = bfloat16_t;

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
    out << "BMG Neumann Inverse Benchmark (64×64 BF16 lower-triangular inverse)\n\n"
        << "Uses the iterative Neumann algorithm (STEPS=" << kSteps << "):\n"
        << "  inv = I\n"
        << "  for _ in range(STEPS):\n"
        << "      err = lower_strict(L * inv)\n"
        << "      inv = inv - inv * err\n\n"
        << "Options:\n\n"
        << "  --help                   Display this usage statement\n"
        << "  --batch=<int>            Number of 64×64 matrices (default: 16)\n"
        << "  --iterations=<int>       Profiling iterations (default: 100)\n\n"
        << "Example:\n"
        << "  $ 05_bmg_neumann_inverse --batch=64 --iterations=200\n\n";
    return out;
  }

  // Estimated GFLOPs for B inversions of an N×N unit lower-triangular matrix.
  // The Neumann algorithm performs 2 × STEPS GEMMs of shape (N×N)×(N×N) → 2*STEPS*2*N^3 FLOPs.
  double gflops(double runtime_s) const {
    double N    = double(kN);
    double flop = double(batch) * 2.0 * double(kSteps) * 2.0 * N * N * N;
    return flop / 1.0e9 / runtime_s;
  }
};

// ---------------------------------------------------------------------------
// Kernel launcher tag
// ---------------------------------------------------------------------------
struct NeumannInverseKernel;

// ---------------------------------------------------------------------------
// Validation
//
// Compute L · L_inv on the CPU (float arithmetic) and verify that the result
// is close to identity.  Tolerances are generous given BF16 precision and the
// iterative nature of the algorithm with only STEPS=12 iterations.
// ---------------------------------------------------------------------------
bool verify(const std::vector<ElementA>& L_host,
            const std::vector<ElementA>& L_inv_host,
            int batch)
{
  static constexpr int N = kN;
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
        // Tolerances: BF16 has ~2 decimal digits; the iterative algorithm
        // with only 12 steps is approximate, so use generous thresholds.
        if (diff > 0.1f) {
          std::cout << "Validation failed at batch=" << b
                    << " (" << row << "," << col << ")"
                    << ": got " << val
                    << ", expected " << expected
                    << " (diff=" << diff << ")\n";
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

  static constexpr int N = kN;

  std::cout << "BMG Neumann Inverse Benchmark\n"
            << "  batch      : " << batch      << "\n"
            << "  matrix dim : " << N << "x" << N << "\n"
            << "  dtype      : BF16\n"
            << "  steps      : " << kSteps     << " (Neumann iterations, fixed)\n"
            << "  warmup     : " << warmup     << "\n"
            << "  iterations : " << iterations << "\n\n";

  // -------------------------------------------------------------------------
  // Generate random unit lower-triangular BF16 matrices on host.
  // Random strictly-lower values are kept small (scale 0.2) so the Neumann
  // series converges within STEPS=12 iterations (matches Python reference).
  // -------------------------------------------------------------------------
  std::srand(42);
  std::vector<ElementA> h_L(batch * N * N, ElementA(0));

  for (int b = 0; b < batch; ++b) {
    ElementA* mat = h_L.data() + b * N * N;
    for (int r = 0; r < N; ++r) {
      mat[r * N + r] = ElementA(1.0f);  // unit diagonal
      for (int c = 0; c < r; ++c) {
        // Small random values to keep the matrix well-conditioned.
        float v = (float(std::rand()) / float(RAND_MAX)) * 0.4f - 0.2f;
        mat[r * N + c] = ElementA(v);
      }
    }
  }

  std::vector<ElementA> h_L_orig = h_L;   // save for validation / repeated launches

  // -------------------------------------------------------------------------
  // Allocate device memory for input (L) and output (inverse).
  // -------------------------------------------------------------------------
  cutlass::DeviceAllocation<ElementA> d_L(batch * N * N);
  cutlass::DeviceAllocation<ElementA> d_Out(batch * N * N);

  // Upload L.
  compat::get_default_queue()
      .memcpy(d_L.get(), h_L.data(), batch * N * N * sizeof(ElementA))
      .wait();

  // -------------------------------------------------------------------------
  // Kernel launch configuration.
  //   - 256 threads per work-group (16 sub-groups × 16 threads)
  //   - batch work-groups (one per matrix)
  // -------------------------------------------------------------------------
  sycl::range<3> local{1, 1, static_cast<size_t>(kWGSize)};
  sycl::range<3> global{1, 1, static_cast<size_t>(kWGSize * batch)};

  namespace syclex  = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties kernel_props{syclex::sub_group_size<kSGSize>,
                                  intelex::grf_size<256>};

  ElementA* L_ptr   = d_L.get();
  ElementA* Out_ptr = d_Out.get();
  int       batch_n = batch;

  auto launch_kernel = [&]() {
    return compat::get_default_queue().parallel_for<NeumannInverseKernel>(
        sycl::nd_range<3>(global, local), kernel_props,
        [=](sycl::nd_item<3>) {
          neumann_inverse_kernel<ElementA>(L_ptr, Out_ptr, batch_n);
        });
  };

  // -------------------------------------------------------------------------
  // Warmup runs.
  // -------------------------------------------------------------------------
  for (int i = 0; i < warmup; ++i) {
    launch_kernel().wait();
  }

  // -------------------------------------------------------------------------
  // Validation: run once and download the result.
  // -------------------------------------------------------------------------
  launch_kernel().wait();

  std::vector<ElementA> h_Out(batch * N * N);
  compat::get_default_queue()
      .memcpy(h_Out.data(), d_Out.get(), batch * N * N * sizeof(ElementA))
      .wait();

  bool passed = verify(h_L_orig, h_Out, batch);
  if (!passed) {
    std::cout << "\nAccuracy check FAILED.\n";
    return 1;
  }
  std::cout << "Accuracy check passed.\n\n";

  // -------------------------------------------------------------------------
  // Timed runs (kernel-only latency).
  // -------------------------------------------------------------------------
  GPU_Clock timer;
  timer.start();

  for (int i = 0; i < iterations; ++i) {
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
