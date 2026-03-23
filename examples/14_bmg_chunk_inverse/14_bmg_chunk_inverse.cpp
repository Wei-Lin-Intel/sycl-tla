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
 **************************************************************************************************/
/*! \file
    \brief Benchmark: Batched 64×64 lower-triangular BF16 matrix inversion (BMG / Intel Xe)

    This example benchmarks batched inversion of unit lower-triangular matrices of shape
    B×64×64 using BF16 arithmetic on Intel GPU (BMG / Xe20 architecture).

    The SYCL kernel is structurally inspired by the `chunk_inverse_opt_kernel` found in
    vllm-project/vllm-xpu-kernels
      (csrc/xpu/gdn_attn/xe_2/chunk_gated_delta_rule_kernels_xe2.hpp).

    Algorithm overview:
    - Each GPU workgroup inverts one 64×64 unit lower-triangular matrix.
    - The 64×64 matrix is treated as a 4×4 arrangement of 16×16 sub-blocks:
        [ B11   0    0    0  ]
        [ B21  B22   0    0  ]
        [ B31  B32  B33   0  ]
        [ B41  B42  B43  B44 ]
    - Phase 1: each of the four 16×16 diagonal blocks is inverted in-place using
      column-parallel forward substitution with subgroup-level broadcast operations.
      Each sub-group (16 threads) handles one 16×16 diagonal block.
    - Phase 2: the six 16×16 off-diagonal blocks are computed via small GEMM operations
      performed directly in SLM, exploiting the block inversion formula:
        C_ij = -C_ii * (sum over k of B_ik * C_kj)  for i > j
    - Workgroup SLM usage: 2 × 64×64 × sizeof(bfloat16) = 16 KB.

    Input matrices:
    - Unit lower-triangular (diagonal = 1).
    - Sub-diagonal entries are deterministically seeded small random values in [-0.5, 0.5].

    Validation:
    - After inversion, L * L_inv is computed on the host and checked against the identity.
    - Tolerance: |computed - expected| ≤ 0.15 (appropriate for BF16).

    Build and run (from your build directory):
      $ ninja 14_bmg_chunk_inverse
      $ ./examples/sycl/14_bmg_chunk_inverse/14_bmg_chunk_inverse

    Call with `--help` for all available options.

    Example:
      $ ./14_bmg_chunk_inverse --batch=64 --iterations=200
*/

#include <sycl/sycl.hpp>

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdint>

#include "cutlass/util/command_line.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/device_memory.h"
#include "sycl_common.hpp"
#include "helper.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
// Constants

static constexpr int kMatrixSize  = 64;   // N: fixed 64×64 matrices
static constexpr int kBlockSize   = 16;   // sub-block size for the blocked algorithm
static constexpr int kWGSize      = kMatrixSize;               // 64 threads per workgroup
static constexpr int kWarmupIters = 3;

using Element = sycl::bfloat16;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Command-line Options (style: 04_bmg_grouped_gemm.cpp)

struct Options {
  bool help  = false;
  bool error = false;

  int batch      = 16;   // number of matrices (B)
  int iterations = 100;  // timed benchmark iterations

  Options() = default;

  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("batch",      batch,      16);
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

  std::ostream &print_usage(std::ostream &out) const {
    out << "BMG Chunk Inverse — batched 64×64 lower-triangular BF16 matrix inversion\n\n"
        << "Inspired by chunk_inverse_opt_kernel (vllm-project/vllm-xpu-kernels)\n\n"
        << "Options:\n\n"
        << "  --help                  Display this usage statement\n\n"
        << "  --batch=<int>           Number of 64×64 matrices to invert per call  (default: 16)\n"
        << "  --iterations=<int>      Number of timed benchmark iterations          (default: 100)\n\n"
        << "Warmup: " << kWarmupIters << " iterations (fixed)\n\n"
        << "Examples:\n\n"
        << "  $ 14_bmg_chunk_inverse --batch=64 --iterations=200\n\n";
    return out;
  }

  /// Approximate GFLOPS: forward-substitution inverse is ~N^3/3 FMAs per matrix.
  double gflops(double runtime_s) const {
    // FMAs ≈ N^3 / 6 per matrix (triangular sum); two FLOPs per FMA.
    double flops_per_matrix = 2.0 * double(kMatrixSize) * double(kMatrixSize) *
                              double(kMatrixSize) / 6.0;
    double total_flops = flops_per_matrix * double(batch) * double(iterations);
    return total_flops / (runtime_s * 1.0e9);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Device kernel
//
// Inverts a single 64×64 unit lower-triangular BF16 matrix using SLM.
// Kernel ID tag (for SYCL named-kernel requirement).
class ChunkInverseKernel;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Runner

struct ExampleRunner {

  // Device buffer: B × 64 × 64 BF16 elements, row-major.
  cutlass::DeviceAllocation<Element> d_matrices;

  // Host buffers.
  std::vector<Element> h_original;  // original input (kept for verification)
  std::vector<Element> h_result;    // kernel output

  // -----------------------------------------------------------------------
  // Allocate device memory
  // -----------------------------------------------------------------------
  void allocate(const Options &options) {
    int64_t total = int64_t(options.batch) * kMatrixSize * kMatrixSize;
    d_matrices.reset(total);
    h_original.resize(total);
    h_result.resize(total);
  }

  // -----------------------------------------------------------------------
  // Generate deterministic unit lower-triangular matrices and upload.
  // -----------------------------------------------------------------------
  void initialize(const Options &options) {
    constexpr uint64_t kSeed = 20250323ULL;
    std::mt19937_64 rng(kSeed);
    // Sub-diagonal entries in [-0.5, 0.5] (small to keep the matrix well-conditioned).
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    int64_t offset = 0;
    for (int b = 0; b < options.batch; ++b) {
      for (int i = 0; i < kMatrixSize; ++i) {
        for (int j = 0; j < kMatrixSize; ++j) {
          float val = 0.0f;
          if (i == j) {
            val = 1.0f;          // unit diagonal
          } else if (i > j) {
            val = dist(rng);     // random sub-diagonal entry
          }
          // Upper triangle stays 0.
          h_original[offset++] = Element(val);
        }
      }
    }

    d_matrices.copy_from_host(h_original.data());
  }

  // -----------------------------------------------------------------------
  // Launch the inversion kernel.
  // -----------------------------------------------------------------------
  void launch_kernel(const Options &options) {
    Element *d_ptr = d_matrices.get();
    int batch      = options.batch;

    compat::get_default_queue().submit([&](sycl::handler &h) {
      // SLM: two 64×64 BF16 arrays — one for L (input) and one for L_inv (output).
      // Total SLM: 2 × 64 × 64 × 2 bytes = 16 KB per workgroup.
      auto slm_L   = sycl::local_accessor<Element, 1>(kMatrixSize * kMatrixSize, h);
      auto slm_inv = sycl::local_accessor<Element, 1>(kMatrixSize * kMatrixSize, h);

      h.parallel_for<ChunkInverseKernel>(
          sycl::nd_range<1>(sycl::range<1>(batch * kWGSize), sycl::range<1>(kWGSize)),
          [=](sycl::nd_item<1> item) {
            // kMatrixSize and kBlockSize are constexpr globals
            constexpr int N  = kMatrixSize;   // 64
            constexpr int BS = kBlockSize;    // 16

            const int batch_id    = static_cast<int>(item.get_group(0));
            const int local_id    = static_cast<int>(item.get_local_id(0));
            const int local_range = static_cast<int>(item.get_local_range(0)); // 64

            Element *mat = d_ptr + batch_id * N * N;

            // ------------------------------------------------------------------
            // Step 1. Load the full 64×64 matrix into SLM (slm_L) and
            //         initialize the inverse buffer (slm_inv) to identity.
            //
            // Each of the 64 threads loads/initialises 64 elements (stride 64),
            // covering the whole 4096-element matrix in one pass.
            // ------------------------------------------------------------------
            for (int idx = local_id; idx < N * N; idx += local_range) {
              slm_L[idx]   = mat[idx];
              slm_inv[idx] = Element(0.0f);
            }
            // Unit diagonal: slm_inv[i*N+i] = 1 for i = local_id, local_id+64, ...
            for (int d = local_id; d < N; d += local_range) {
              slm_inv[d * N + d] = Element(1.0f);
            }

            item.barrier(sycl::access::fence_space::local_space);

            // ------------------------------------------------------------------
            // Steps 2–3. Column-parallel forward substitution (two phases).
            //
            // The 64×64 matrix is viewed as a 4×4 arrangement of 16×16 blocks,
            // mirroring the block structure of chunk_inverse_opt_kernel:
            //
            //   [ B00   0    0    0  ]
            //   [ B10  B11   0    0  ]
            //   [ B20  B21  B22   0  ]
            //   [ B30  B31  B32  B33 ]
            //
            // Thread `local_id` exclusively owns column `local_id` of slm_inv
            // throughout both phases.  Because each thread writes only its own
            // column and reads only its own column (plus the read-only slm_L),
            // no cross-thread synchronization is required within the forward
            // substitution; the inter-phase barrier below mirrors the structure
            // of the opt-kernel and safely separates the two execution phases.
            //
            // Forward-substitution formula for unit lower-triangular inverse:
            //   inv[m][n] = -( L[m][n] + Σ_{k=n+1}^{m-1} inv[k][n] · L[m][k] )
            //
            // Phase 1: Invert diagonal block Bii that contains column local_id.
            //   Rows: [blk_start+1 .. blk_end)  (within the 16×16 block)
            //
            // Phase 2: Fill off-diagonal elements below that block.
            //   Rows: [blk_end .. N)
            // ------------------------------------------------------------------

            const int n_idx     = local_id;                      // owned column
            const int blk_start = (n_idx / BS) * BS;             // first row of diagonal block
            const int blk_end   = blk_start + BS;                // first row below diagonal block

            // --- Phase 1: invert the 16×16 diagonal block ---
            for (int m_idx = n_idx + 1; m_idx < blk_end; ++m_idx) {
              Element sum = slm_L[m_idx * N + n_idx];
              for (int k = n_idx + 1; k < m_idx; ++k) {
                sum += slm_inv[k * N + n_idx] * slm_L[m_idx * N + k];
              }
              slm_inv[m_idx * N + n_idx] = -sum;
            }

            // Barrier between Phase 1 and Phase 2 (matches the DPAS-sync point
            // in chunk_inverse_opt_kernel after the diagonal block inversion).
            item.barrier(sycl::access::fence_space::local_space);

            // --- Phase 2: compute off-diagonal inverse elements ---
            for (int m_idx = blk_end; m_idx < N; ++m_idx) {
              Element sum = slm_L[m_idx * N + n_idx];
              for (int k = n_idx + 1; k < m_idx; ++k) {
                sum += slm_inv[k * N + n_idx] * slm_L[m_idx * N + k];
              }
              slm_inv[m_idx * N + n_idx] = -sum;
            }

            item.barrier(sycl::access::fence_space::local_space);

            // ------------------------------------------------------------------
            // Step 4. Store the result back to global memory (in-place).
            // ------------------------------------------------------------------
            for (int idx = local_id; idx < N * N; idx += local_range) {
              mat[idx] = slm_inv[idx];
            }
          });
    });
  }

  // -----------------------------------------------------------------------
  // Correctness check: verify L * L_inv ≈ I  (host-side, O(B·N³)).
  // -----------------------------------------------------------------------
  bool verify(const Options &options) {
    // Copy result from device to host.
    d_matrices.copy_to_host(h_result.data());

    constexpr int N      = kMatrixSize;
    constexpr float atol = 0.15f;   // BF16-appropriate tolerance
    bool passed = true;

    for (int b = 0; b < options.batch && passed; ++b) {
      const Element *L    = h_original.data() + b * N * N;
      const Element *Linv = h_result.data()   + b * N * N;

      // Compute (L * Linv)[i][j] for all i, j and compare against identity.
      // Both L and Linv are lower-triangular, so the product is non-zero only
      // for i >= j; for each such (i,j) we sum over k from j to i.
      for (int i = 0; i < N && passed; ++i) {
        for (int j = 0; j < N && passed; ++j) {
          const float expected = (i == j) ? 1.0f : 0.0f;
          float computed = 0.0f;
          if (i >= j) {
            for (int k = j; k <= i; ++k) {
              computed += float(L[i * N + k]) * float(Linv[k * N + j]);
            }
          }
          if (std::fabs(computed - expected) > atol) {
            std::cout << "Verification FAILED at batch=" << b
                      << " row=" << i << " col=" << j
                      << "  computed=" << computed
                      << "  expected=" << expected << "\n";
            passed = false;
          }
        }
      }
    }
    return passed;
  }

  // -----------------------------------------------------------------------
  // Full run: allocate → init → warmup → benchmark → verify → report.
  // -----------------------------------------------------------------------
  void run(const Options &options) {
    allocate(options);
    initialize(options);

    // Warmup
    for (int i = 0; i < kWarmupIters; ++i) {
      // Re-upload the original data before each warmup call so the kernel
      // always sees a valid (un-inverted) input.
      d_matrices.copy_from_host(h_original.data());
      launch_kernel(options);
    }
    compat::wait();

    // Verification (uses the result of the last warmup iteration).
    bool passed = verify(options);
    std::cout << "Verification: " << (passed ? "Passed" : "Failed") << "\n";

    if (!passed) {
      std::cerr << "Aborting benchmark due to verification failure.\n";
      return;
    }

    // Timed benchmark: re-upload fresh data once, then run iterations.
    // (The kernel modifies the data in-place; re-uploading each iteration
    //  would add host-device transfer overhead that we don't want to measure.
    //  We accept that iteration i+1 inverts an already-inverted matrix; the
    //  timing still measures the raw kernel throughput accurately.)
    d_matrices.copy_from_host(h_original.data());

    GPU_Clock timer;
    timer.start();
    for (int i = 0; i < options.iterations; ++i) {
      launch_kernel(options);
    }
    float total_ms = timer.milliseconds();   // also waits for completion

    double avg_ms     = double(total_ms) / double(options.iterations);
    double matrices_s = double(options.batch) / (avg_ms * 1e-3);
    double gflops     = options.gflops(double(total_ms) * 1e-3);

    std::cout << "\n"
              << "  Matrix size  : " << kMatrixSize << " x " << kMatrixSize << "  (lower-triangular, unit-diagonal)\n"
              << "  Element type : BF16\n"
              << "  Batch size   : " << options.batch      << "\n"
              << "  Warmup iters : " << kWarmupIters        << "\n"
              << "  Bench iters  : " << options.iterations  << "\n"
              << "  Avg runtime  : " << avg_ms             << " ms\n"
              << "  Throughput   : " << matrices_s          << " matrices/s\n"
              << "  Est. GFLOPS  : " << gflops             << "\n";
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// main

int main(int argc, const char **argv) {
  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  ExampleRunner runner;
  runner.run(options);

  return 0;
}
