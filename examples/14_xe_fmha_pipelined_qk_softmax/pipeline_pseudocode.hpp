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

/**
 * @file pipeline_pseudocode.hpp
 * @brief Annotated pseudocode for the 2-slot QK–softmax pipeline in the Xe
 *        forward FMHA mainloop.
 *
 * This file is a documentation/validation aid and is NOT compiled into
 * the kernel.  It captures the scheduling intent of the refactored
 * `xe_fmha_fwd_mainloop.hpp` in C++-style pseudocode with detailed comments
 * so that the overlap strategy can be reviewed, tested in isolation, or ported
 * to future pipeline variants.
 *
 * See README.md in this directory for architecture-level context.
 */

#pragma once

// ============================================================================
// Section 1: Pipeline scheduling pseudocode
//
// The pipeline is a 2-slot (double-buffer) scheme over K-blocks.
// Slot index alternates 0/1 each iteration via `(k - k_start) & 1`.
//
// Notation:
//   QK(k)  = Q * K_k   GEMM  (XMX / DPAS path)
//   SM(k)  = softmax(S_k)    (XVX / vector-scalar path)
//   PV(k)  = P_k * V_k GEMM  (XMX / DPAS path)
//   S[s]   = tSrS_pipe[s]    QK accumulator slot s
//   P[s]   = tArP_pipe[s]    reordered-P slot s (SM output)
//
// Timeline (one iteration of the steady-state loop):
//
//   Cycle -->
//   XMX: [  QK(k+1) → S[next]  |  PV(k) using P[cur]  ]
//   XVX: [             SM(k) on S[cur] → P[cur]         ]
//
// The compiler/hardware must issue QK(k+1) before SM(k) for the overlap to be
// visible to the instruction scheduler.  The two stages touch disjoint register
// sets (S[next] vs S[cur]), so there are no RAW hazards between them.
//
// PV(k) runs *after* SM(k) completes (RAW dependency on P[cur]) and therefore
// cannot overlap with the QK(k+1) XMX work in this conservative design.
// ============================================================================

namespace fmha_pipeline_pseudocode {

// ---------------------------------------------------------------------------
// Data types (illustrative; real types come from TiledMMA helpers)
// ---------------------------------------------------------------------------

using ElementS   = float;      // Q*K accumulator element (typically fp32)
using FragS      = float[64];  // QK fragment  (size depends on tile/atom)
using FragAforP  = float[64];  // PV-A fragment (reordered P, same element type)
using FragSRow   = float[4];   // Per-row softmax state (one entry per row-group;
                               // holds max or sum depending on context)

// ---------------------------------------------------------------------------
// BlockDesc: unifies logical/physical K indexing across cache/paged paths
// ---------------------------------------------------------------------------
struct BlockDesc {
  int  logical_k;   // logical K-block index in [k_start, k_end)
  int  physical_k;  // physical K-block index into the tensor (paged KV aware)
  bool is_cache;    // true  → access KV-cache tensors
                    // false → access main KV tensors
};

// ---------------------------------------------------------------------------
// Annotated pipeline body (pseudocode)
// ---------------------------------------------------------------------------

/**
 * 2-slot block pipeline for QK–softmax overlap.
 *
 * @param k_start  first logical K-block index for this workgroup
 * @param k_end    one-past-last logical K-block index (exclusive)
 */
inline void pipeline_body_pseudocode(int k_start, int k_end) {

  // -- Fragment storage (in real code: declared in register file) ----------
  constexpr int Stages = 2;  // template parameter; number of prefetch stages
  FragS    S[2] = {};      // 2-slot QK accumulator
  FragAforP P[2] = {};     // 2-slot reordered-P
  FragSRow max_state = {}; // running row-wise maximum
  FragSRow sum_state = {}; // running row-wise sum
  FragS    A_out = {};     // output accumulator (O)

  int n_blocks = k_end - k_start;
  if (n_blocks == 0) return;

  // ========================================================================
  // Prologue: compute QK for the first block into slot 0.
  // No overlap possible here (no previous block's softmax to overlap with).
  //
  // Also issue the K prefetch that the original serial loop would have
  // performed at block k_start: K(k_start + Stages).  This ensures the
  // steady-state's first few iterations do not miss the prefetch window.
  // ========================================================================
  {
    // barrier_arrive(WG)              -- ensure prefetched K data is visible
    BlockDesc d0 = make_block_desc(k_start);
    compute_qk(d0, S[0]);            // XMX: QK(k_start) → S[0]
    prefetch_k(k_start + Stages);     // prefetch K for future blocks
    // barrier_wait(WG)
  }

  // ========================================================================
  // Steady-state: overlap QK(next) [XMX] with softmax(cur) [XVX].
  //
  // The key scheduling choice: emit QK(next) BEFORE softmax(cur).
  // This gives the hardware scheduler the earliest possible window to
  // dispatch the XMX DPAS instructions while the XVX units handle softmax.
  // ========================================================================
  for (int k = k_start; k < k_end - 1; ++k) {
    int cur  = (k - k_start) & 1;
    int next = cur ^ 1;

    BlockDesc cur_desc  = make_block_desc(k);
    BlockDesc next_desc = make_block_desc(k + 1);

    // barrier_arrive(WG)

    // [XMX] Issue QK for the NEXT block.
    // Hardware can overlap this with the XVX softmax below because:
    //   • QK(next) writes to S[next] (no conflict with S[cur])
    //   • softmax(cur) reads/writes S[cur] (no conflict with S[next])
    compute_qk(next_desc, S[next]);   // XMX: Q * K_{k+1} → S[next]

    // [scalar/XVX] Apply mask and softmax on the CURRENT block.
    // These operations target the vector/scalar ALU (XVX) and are the
    // primary overlap target with the XMX work above.
    apply_mask(cur_desc, S[cur]);     // scalar: -INF for masked positions
    FragSRow rescale = online_softmax(k == k_start, S[cur],
                                      max_state, sum_state);
    reorder_s_to_p(S[cur], P[cur]);  // layout reorder: S → P layout

    // [XMX] Accumulate P*V for the current block.
    // This follows softmax to avoid competing for XMX with QK(next) above.
    accumulate_pv(cur_desc, P[cur], rescale, k == k_start, A_out);

    // Prefetch K for the block consumed Stages iterations from now.
    // Iteration k computes QK(k+1), so iteration k+Stages will compute
    // QK(k+Stages+1).  Combined with the prologue's K(k_start+Stages)
    // prefetch, every block gets exactly Stages iterations of lead time.
    prefetch_k(k + Stages + 1);

    // barrier_wait(WG)
  }

  // ========================================================================
  // Epilogue: drain the final block.
  // No next-QK to issue, so this is mask + softmax + PV only.
  // ========================================================================
  {
    int last_k    = k_end - 1;
    int last_slot = (last_k - k_start) & 1;
    BlockDesc last_desc = make_block_desc(last_k);

    apply_mask(last_desc, S[last_slot]);
    FragSRow rescale = online_softmax(last_k == k_start, S[last_slot],
                                      max_state, sum_state);
    reorder_s_to_p(S[last_slot], P[last_slot]);
    accumulate_pv(last_desc, P[last_slot], rescale, last_k == k_start, A_out);
  }
}

// ---------------------------------------------------------------------------
// Stubs referenced above (implementations are in xe_fmha_fwd_mainloop.hpp)
// ---------------------------------------------------------------------------

inline BlockDesc make_block_desc(int K);                              // see mainloop
inline void compute_qk(BlockDesc const&, FragS&);                    // Stage A
inline void apply_mask(BlockDesc const&, FragS&);                    // Stage B
inline FragSRow online_softmax(bool, FragS&, FragSRow&, FragSRow&);  // Stage C
inline void reorder_s_to_p(FragS const&, FragAforP&);                // Stage C (reorder)
inline void accumulate_pv(BlockDesc const&, FragAforP const&,
                          FragSRow const&, bool, FragS&);            // Stage D
inline void prefetch_k(int K_next);                                   // hint

} // namespace fmha_pipeline_pseudocode
