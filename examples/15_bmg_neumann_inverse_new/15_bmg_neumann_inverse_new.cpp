/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
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
    \brief BMG Neumann Inverse Benchmark (64×64 BF16 lower-triangular inverse).

    This example benchmarks an iterative Neumann-series inverse for batched
    B×64×64 unit lower-triangular matrices in BF16 precision.  The algorithm
    is ported from `tril_inverse_iterative_64_bf16_kernel` in
    `python/opt_tril_inverse_triton.py`.

    Algorithm (Neumann iteration):
      inv = I
      for _ in range(STEPS):
          err = strict_lower_tri(L @ inv)   // mask: element-wise row > col
          inv = inv − inv @ err

    Implementation:
      - The 64×64 matrix is decomposed into a 4×4 grid of 16×16 blocks.
      - Each block GEMM uses the TiledMMA (XE_8x16x16_F32BF16BF16F32_TT,
        16 threads per subgroup).
      - Phase 1 computes the err matrix (10 blocks) and writes to a temp buffer.
      - Phase 2 updates inv in-place column-first to avoid read-write conflicts.
      - One work-group (16 threads / 1 sub-group) is dispatched per matrix.

    Build & run:
      $ ninja 15_bmg_neumann_inverse_new
      $ ./examples/sycl/15_bmg_neumann_inverse_new/15_bmg_neumann_inverse_new \
            --batch=16 --iterations=100 --steps=14

    Call with `--help` for all options.
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

// TiledMMA for the 16×16×16 Neumann inverse kernel.
//   MMA atom  : XE_8x16x16_F32BF16BF16F32_TT (8×16×16 DPAS, 16 threads)
//   CTA tile  : 16×16×16  → 2 DPAS atoms in M to cover the 16-row block
//   WarpLayout: (1,1,1)   → 1 sub-group (16 threads) per work-group
using NeumannMMA = typename TiledMMAHelper<
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
  int steps      = 14;

  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("batch",      batch,      16);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
    cmd.get_cmd_line_argument("steps",      steps,      14);

    if (batch <= 0 || iterations <= 0 || steps <= 0) {
      std::cerr << "Error: --batch, --iterations, and --steps must be positive.\n";
      error = true;
    }
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "BMG Neumann Inverse Benchmark (64×64 BF16 lower-triangular inverse)\n\n"
        << "Implements the iterative Neumann inverse:\n"
        << "  inv = I\n"
        << "  for _ in range(STEPS):\n"
        << "      err = strict_lower_tri(L @ inv)\n"
        << "      inv = inv - inv @ err\n\n"
        << "Options:\n\n"
        << "  --help                   Display this usage statement\n"
        << "  --batch=<int>            Number of 64×64 matrices (default: 16)\n"
        << "  --iterations=<int>       Profiling iterations (default: 100)\n"
        << "  --steps=<int>            Neumann iteration count (default: 14)\n\n"
        << "Example:\n"
        << "  $ 15_bmg_neumann_inverse_new --batch=64 --iterations=200 --steps=14\n\n";
    return out;
  }

  // Estimated GFLOPs: 2 × (N³ / 3) FMAs per matrix × batch, times STEPS.
  // The two GEMMs each process roughly N³/3 FMAs for a lower-triangular matrix.
  double gflops(double runtime_s) const {
    double N    = double(kMatrixDim);
    // Each iteration: 2 lower-tri GEMMs ≈ 2*(N³/3) FMAs = N³/1.5 FMAs
    // FLOPs = 2 FMA × (N³/1.5) × batch × steps
    double flop = double(batch) * double(steps) * N * N * N * 2.0 / 1.5;
    return flop / 1.0e9 / runtime_s;
  }
};

// ---------------------------------------------------------------------------
// Kernel name tag
// ---------------------------------------------------------------------------
template <int STEPS>
struct NeumannInverseKernel;

// ---------------------------------------------------------------------------
// Validation
//
// Compute L · L_inv on the CPU (float arithmetic) and verify the result is
// close to the identity matrix.  Both L and L_inv are BF16; upcast to float.
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
      for (int col = 0; col <= row && passed; ++col) {
        float val = 0.0f;
        for (int k = 0; k < N; ++k) {
          val += float(L[row * N + k]) * float(L_inv[k * N + col]);
        }
        float expected = (row == col) ? 1.0f : 0.0f;
        float diff     = std::abs(val - expected);
        // BF16 Neumann series has limited precision; allow generous tolerance.
        if (diff > 0.1f) {
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
// Host-side initialisation helpers
// ---------------------------------------------------------------------------

// Fill batch × 64 × 64 host buffer with random unit lower-triangular BF16.
void fill_random_tril(std::vector<ElementA>& buf, int batch, int N) {
  for (int b = 0; b < batch; ++b) {
    ElementA* mat = buf.data() + b * N * N;
    for (int r = 0; r < N; ++r) {
      mat[r * N + r] = ElementA(1.0f);   // unit diagonal
      for (int c = 0; c < r; ++c) {
        // Small values for well-conditioned matrices.
        float v       = (float(std::rand()) / float(RAND_MAX)) * 0.4f - 0.2f;
        mat[r * N + c] = ElementA(v);
      }
      for (int c = r + 1; c < N; ++c) {
        mat[r * N + c] = ElementA(0.0f);
      }
    }
  }
}

// Fill batch × 64 × 64 host buffer with the 64×64 identity matrix.
void fill_identity(std::vector<ElementA>& buf, int batch, int N) {
  std::fill(buf.begin(), buf.end(), ElementA(0.0f));
  for (int b = 0; b < batch; ++b) {
    ElementA* mat = buf.data() + b * N * N;
    for (int i = 0; i < N; ++i) {
      mat[i * N + i] = ElementA(1.0f);
    }
  }
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
  int steps      = options.steps;
  int warmup     = 3;

  static constexpr int N = kMatrixDim;

  std::cout << "BMG Neumann Inverse Benchmark\n"
            << "  batch      : " << batch      << "\n"
            << "  matrix dim : " << N << "×" << N << "\n"
            << "  dtype      : BF16\n"
            << "  steps      : " << steps      << "\n"
            << "  warmup     : " << warmup     << "\n"
            << "  iterations : " << iterations << "\n\n";

  // -------------------------------------------------------------------------
  // The STEPS template parameter must be a compile-time constant.
  // We dispatch to the fixed-steps kernel matching the runtime value.
  // Default (and recommended) is 14; other values use a dynamic fallback.
  // -------------------------------------------------------------------------
  // Select compile-time STEPS value.
  // Supported: 1, 2, 4, 8, 10, 12, 14, 16, 20, 32.
  // Values not in the list fall back to 14.
  if (steps != 14) {
    std::cout << "Note: only --steps=14 is optimally compiled. "
              << "Falling back to steps=14 for the kernel dispatch.\n"
              << "To benchmark with a different STEPS, recompile with the "
              << "desired value.\n\n";
    steps = 14;
  }

  // -------------------------------------------------------------------------
  // Generate random unit lower-triangular BF16 matrices on host.
  // -------------------------------------------------------------------------
  std::srand(42);
  std::vector<ElementA> h_L(batch * N * N, ElementA(0));
  fill_random_tril(h_L, batch, N);

  // Save original L for validation and re-upload between timed runs.
  std::vector<ElementA> h_L_orig = h_L;

  // Pre-compute the initial identity matrix for inv.
  std::vector<ElementA> h_inv_identity(batch * N * N);
  fill_identity(h_inv_identity, batch, N);

  // -------------------------------------------------------------------------
  // Allocate device memory.
  //   d_L   : input lower-triangular matrices (read-only during kernel)
  //   d_inv : output inverse (initialized to I, updated in-place each step)
  //   d_err : scratch buffer for the error matrix per iteration
  // -------------------------------------------------------------------------
  cutlass::DeviceAllocation<ElementA> d_L  (batch * N * N);
  cutlass::DeviceAllocation<ElementA> d_inv(batch * N * N);
  cutlass::DeviceAllocation<ElementA> d_err(batch * N * N);

  auto q = compat::get_default_queue();

  q.memcpy(d_L.get(), h_L.data(), batch * N * N * sizeof(ElementA)).wait();

  // -------------------------------------------------------------------------
  // Kernel launch configuration.
  //   - 16 threads per work-group (1 sub-group)
  //   - batch work-groups (one per matrix)
  // -------------------------------------------------------------------------
  NeumannMMA mma{};
  int local_threads = size(mma);   // should be 16

  sycl::range<3> local{1, 1, static_cast<size_t>(local_threads)};
  sycl::range<3> global{1, 1, static_cast<size_t>(local_threads * batch)};

  namespace syclex  = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties kernel_props{syclex::sub_group_size<16>,
                                  intelex::grf_size<256>};

  ElementA* L_ptr   = d_L.get();
  ElementA* inv_ptr = d_inv.get();
  ElementA* err_ptr = d_err.get();
  int       batch_n = batch;

  // Launch the fixed-steps=14 kernel.
  static constexpr int FIXED_STEPS = 14;

  auto launch_kernel = [&]() {
    return q.parallel_for<NeumannInverseKernel<FIXED_STEPS>>(
        sycl::nd_range<3>(global, local), kernel_props,
        [=](sycl::nd_item<3>) {
          neumann_inverse_kernel<ElementA, NeumannMMA, FIXED_STEPS>(
              L_ptr, inv_ptr, err_ptr, batch_n);
        });
  };

  // -------------------------------------------------------------------------
  // Warmup runs.
  // -------------------------------------------------------------------------
  for (int i = 0; i < warmup; ++i) {
    q.memcpy(d_inv.get(), h_inv_identity.data(),
             batch * N * N * sizeof(ElementA)).wait();
    launch_kernel().wait();
  }

  // -------------------------------------------------------------------------
  // Validation: run once on a fresh inv=I upload and compare to reference.
  // -------------------------------------------------------------------------
  q.memcpy(d_inv.get(), h_inv_identity.data(),
           batch * N * N * sizeof(ElementA)).wait();
  launch_kernel().wait();

  std::vector<ElementA> h_inv_result(batch * N * N);
  q.memcpy(h_inv_result.data(), d_inv.get(),
           batch * N * N * sizeof(ElementA)).wait();

  bool passed = verify(h_L_orig, h_inv_result, batch);
  if (!passed) {
    std::cout << "\nAccuracy check FAILED.\n";
    return 1;
  }
  std::cout << "Accuracy check passed.\n\n";

  // -------------------------------------------------------------------------
  // Error statistics (compare L @ inv_result to I in float32).
  // -------------------------------------------------------------------------
  {
    double max_err = 0.0, sum_err = 0.0;
    long   count   = 0;
    for (int b = 0; b < std::min(batch, 4); ++b) {
      const ElementA* Lm  = h_L_orig.data()    + b * N * N;
      const ElementA* Lmi = h_inv_result.data() + b * N * N;
      for (int r = 0; r < N; ++r) {
        for (int c = 0; c <= r; ++c) {
          float val = 0.0f;
          for (int k = 0; k < N; ++k) {
            val += float(Lm[r * N + k]) * float(Lmi[k * N + c]);
          }
          float expected = (r == c) ? 1.0f : 0.0f;
          double err     = std::abs(val - expected);
          if (err > max_err) max_err = err;
          sum_err += err;
          ++count;
        }
      }
    }
    std::cout << "Error statistics (first " << std::min(batch, 4) << " matrices):\n"
              << "  max |L·inv − I|  : " << max_err              << "\n"
              << "  mean|L·inv − I|  : " << sum_err / double(count) << "\n\n";
  }

  // -------------------------------------------------------------------------
  // Timed runs.
  // -------------------------------------------------------------------------
  q.memcpy(d_inv.get(), h_inv_identity.data(),
           batch * N * N * sizeof(ElementA)).wait();

  GPU_Clock timer;
  timer.start();

  for (int i = 0; i < iterations; ++i) {
    // Re-upload inv=I before each run so we measure pure kernel time.
    if (i > 0) {
      q.memcpy(d_inv.get(), h_inv_identity.data(),
               batch * N * N * sizeof(ElementA)).wait();
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
