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

    Algorithm (STEPS tunable at runtime, default 14):
        inv = I
        for _ in range(steps):
            err = lower_strict(L * inv)   // strictly-lower-triangular part
            inv = inv - inv * err

    Kernel design:
      - One work-group (kSGs sub-groups × kSGSize threads) per matrix.
      - L, inv, and err are kept in shared local memory (SLM) across all iterations,
        avoiding global-memory traffic for intermediate state.
      - The two GEMM products (L*inv and inv*err) are computed via Intel DPAS instructions
        through CuTe's TiledMMA / cute::gemm building blocks.

    Tuning knobs (see --help):
      --steps   : Neumann iteration count (runtime, default 14).  Fewer = faster, less accurate.
      --block   : DPAS tile size (compile-time, fixed at 16 for XE_8x16x16 atom).
      --tiles   : Tiles per dimension = kN/kBlock (compile-time, fixed at 4 for kN=64).
      --sgs     : Sub-groups per work-group = kTiles² (compile-time, fixed at 16).
      --sgsize  : Threads per sub-group (compile-time, fixed at 16 for DPAS SIMD-16).

    To build & run (from your build directory):

      $ ninja 05_bmg_neumann_inverse
      $ ./examples/sycl/05_bmg_neumann_inverse/05_bmg_neumann_inverse \
            --batch=16 --iterations=100 --steps=14

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

  // Tuning knobs exposed on the command line.
  //
  // --steps    : Neumann iteration count.  Fewer → faster but less accurate.
  //              Default 14 is the minimum that passes the BF16 tolerance check.
  //
  // --block    : DPAS tile size (compile-time constant, informational only).
  //              Must equal the N/K dimension of the MMA atom; 16 is the only
  //              valid value for XE_8x16x16_F32BF16BF16F32_TT.
  //
  // --tiles    : kN / kBlock (informational only).  Fixed at 4 for kN=64, kBlock=16.
  //
  // --sgs      : kTiles² sub-groups per work-group (informational only).
  //
  // --sgsize   : Threads per sub-group (compile-time constant, informational only).
  //              Must be 16 for DPAS SIMD-16.
  int steps  = kSteps;
  int block  = kBlock;
  int tiles  = kTiles;
  int sgs    = kSGs;
  int sgsize = kSGSize;

  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("batch",      batch,      16);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
    cmd.get_cmd_line_argument("steps",      steps,      kSteps);
    cmd.get_cmd_line_argument("block",      block,      kBlock);
    cmd.get_cmd_line_argument("tiles",      tiles,      kTiles);
    cmd.get_cmd_line_argument("sgs",        sgs,        kSGs);
    cmd.get_cmd_line_argument("sgsize",     sgsize,     kSGSize);

    if (batch <= 0 || iterations <= 0 || steps <= 0) {
      std::cerr << "Error: --batch, --iterations, and --steps must be positive.\n";
      error = true;
      return;
    }

    // Validate compile-time-constrained parameters.
    bool bad_config = false;
    if (block != kBlock) {
      std::cerr << "Error: --block=" << block
                << " is not supported. Only --block=" << kBlock
                << " is valid for the XE_8x16x16 DPAS atom.\n";
      bad_config = true;
    }
    if (tiles != kTiles) {
      std::cerr << "Error: --tiles=" << tiles
                << " is not supported. Only --tiles=" << kTiles
                << " is valid for kN=" << kN << ", kBlock=" << kBlock << ".\n";
      bad_config = true;
    }
    if (sgs != kSGs) {
      std::cerr << "Error: --sgs=" << sgs
                << " is not supported. Only --sgs=" << kSGs
                << " is valid (kTiles²=" << kTiles << "²).\n";
      bad_config = true;
    }
    if (sgsize != kSGSize) {
      std::cerr << "Error: --sgsize=" << sgsize
                << " is not supported. Only --sgsize=" << kSGSize
                << " is valid for DPAS SIMD-16.\n";
      bad_config = true;
    }
    if (bad_config) {
      error = true;
    }
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "BMG Neumann Inverse Benchmark (64×64 BF16 lower-triangular inverse)\n\n"
        << "Uses the iterative Neumann algorithm:\n"
        << "  inv = I\n"
        << "  for _ in range(steps):\n"
        << "      err = lower_strict(L * inv)\n"
        << "      inv = inv - inv * err\n\n"
        << "Options:\n\n"
        << "  --help                   Display this usage statement\n"
        << "  --batch=<int>            Number of 64×64 matrices (default: 16)\n"
        << "  --iterations=<int>       Profiling iterations (default: 100)\n"
        << "  --steps=<int>            Neumann iterations (default: " << kSteps << "; fewer=faster, less accurate)\n"
        << "\n"
        << "Informational (compile-time constants, cannot be changed at runtime):\n"
        << "  --block=<int>            DPAS tile size (fixed: " << kBlock << " for XE_8x16x16 atom)\n"
        << "  --tiles=<int>            Tiles per dimension kN/kBlock (fixed: " << kTiles << " for kN=64)\n"
        << "  --sgs=<int>              Sub-groups per work-group kTiles² (fixed: " << kSGs << ")\n"
        << "  --sgsize=<int>           Threads per sub-group (fixed: " << kSGSize << " for DPAS SIMD-16)\n"
        << "\n"
        << "Example:\n"
        << "  $ 05_bmg_neumann_inverse --batch=64 --iterations=200 --steps=10\n\n";
    return out;
  }

  // Estimated GFLOPs for B inversions of an N×N unit lower-triangular matrix.
  // Each Neumann step runs 2 GEMMs (L×inv and inv×err), each of shape N×N×N.
  // One N×N×N GEMM costs 2*N^3 FLOPs (multiply-add pairs), so:
  //   total FLOPs = batch × steps × 2 GEMMs × 2*N^3 = 4 * batch * steps * N^3
  double gflops(double runtime_s) const {
    double N    = double(kN);
    double flop = double(batch) * double(steps) * 2.0 /* GEMMs/step */ * 2.0 * N * N * N;
    return flop / 1.0e9 / runtime_s;
  }
};

// ---------------------------------------------------------------------------
// Validation (forward-declared before launch_config which calls it)
//
// Compute L · L_inv on the CPU (float arithmetic) and verify that the result
// is close to identity.  Tolerances are generous given BF16 precision and the
// iterative nature of the algorithm.
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
        // with fewer steps is approximate, so use generous thresholds.
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
// Kernel launcher tag (templated so different compile-time configs each get
// a distinct SYCL kernel name, which is required by the SYCL spec).
// ---------------------------------------------------------------------------
template <int TBlock, int TTiles, int TSGSize>
struct NeumannInverseKernel {};

// ---------------------------------------------------------------------------
// launch_config<TBlock, TTiles, TSGSize>
//
// Runs the full benchmark (warmup → validation → timed iterations) for the
// given compile-time configuration.  `steps` is the runtime Neumann iteration
// count captured from the command line.
// ---------------------------------------------------------------------------
template <int TBlock, int TTiles, int TSGSize>
int launch_config(const Options& options,
                  ElementA* L_ptr,
                  ElementA* Out_ptr,
                  const std::vector<ElementA>& h_L_orig,
                  int batch,
                  int iterations,
                  int steps)
{
  constexpr int TSGs    = TTiles * TTiles;
  constexpr int TWGSize = TSGs * TSGSize;

  sycl::range<3> local{1, 1, static_cast<size_t>(TWGSize)};
  sycl::range<3> global{1, 1, static_cast<size_t>(TWGSize * batch)};

  namespace syclex  = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties kernel_props{syclex::sub_group_size<TSGSize>,
                                  intelex::grf_size<256>};

  int batch_n = batch;
  int warmup  = 3;

  auto launch_kernel = [&]() {
    return compat::get_default_queue().parallel_for<NeumannInverseKernel<TBlock, TTiles, TSGSize>>(
        sycl::nd_range<3>(global, local), kernel_props,
        [=](sycl::nd_item<3>) {
          neumann_inverse_kernel<ElementA, TBlock, TTiles, TSGSize>(
              L_ptr, Out_ptr, batch_n, steps);
        });
  };

  // Warmup.
  for (int i = 0; i < warmup; ++i) {
    launch_kernel().wait();
  }

  // Validation: run once and download the result.
  launch_kernel().wait();

  static constexpr int N = kN;
  std::vector<ElementA> h_Out(batch * N * N);
  compat::get_default_queue()
      .memcpy(h_Out.data(), Out_ptr, batch * N * N * sizeof(ElementA))
      .wait();

  bool passed = verify(h_L_orig, h_Out, batch);
  if (!passed) {
    std::cout << "\nAccuracy check FAILED.\n";
    return 1;
  }
  std::cout << "Accuracy check passed.\n\n";

  // Timed runs.
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

  static constexpr int N = kN;

  std::cout << "BMG Neumann Inverse Benchmark\n"
            << "  batch      : " << batch      << "\n"
            << "  matrix dim : " << N << "x" << N << "\n"
            << "  dtype      : BF16\n"
            << "  steps      : " << steps      << " (Neumann iterations, runtime-tunable)\n"
            << "  block      : " << options.block  << " (DPAS tile size, compile-time)\n"
            << "  tiles      : " << options.tiles  << " (tiles per dim, compile-time)\n"
            << "  sgs        : " << options.sgs    << " (sub-groups per WG, compile-time)\n"
            << "  sgsize     : " << options.sgsize << " (threads per SG, compile-time)\n"
            << "  iterations : " << iterations << "\n\n";

  // -------------------------------------------------------------------------
  // Generate random unit lower-triangular BF16 matrices on host.
  // Random strictly-lower values are kept small (scale 0.2) so the Neumann
  // series converges within a reasonable number of steps.
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

  std::vector<ElementA> h_L_orig = h_L;   // save for validation

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
  // Dispatch to the (single currently valid) compile-time configuration.
  // -------------------------------------------------------------------------
  return launch_config<kBlock, kTiles, kSGSize>(
      options,
      d_L.get(), d_Out.get(),
      h_L_orig,
      batch, iterations, steps);
}
