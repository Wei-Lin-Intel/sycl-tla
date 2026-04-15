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

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"
#include "fmha_fusion.hpp"

namespace cutlass::fmha {

template <int Stages> class XeDefault {};   // Default FMHA mainloop, P in registers.

};

namespace cutlass::fmha::collective {

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class DispatchPolicy_,
          bool CausalMask_,
          bool CachedKV_,
          bool PagedKV_,
          class TiledMMAQK_,          // Tiling for Q*K GEMM
          class TiledMMAPV_,          // Tiling for P*V GEMM
          int VTiles_,                // # of tiles in V dimension
          class TensorQ_,             // Global Q/K/V tensors
          class TensorK_,
          class TensorV_,
          class TensorK_cache_,
          class TensorV_cache_,
          class TiledCopyQ_ = void,   // Optional TiledCopy for loading Q
          class TiledCopyK_ = void,   // Optional TiledCopy for loading K
          class TiledCopyV_ = void,   // Optional TiledCopy for loading V
          class TiledCopyK_cache_ = void,
          class TiledCopyV_cache_ = void>   // Optional TiledCopy for loading V_cache
struct FMHAFwdMainloop {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy_>, "Could not find a mainloop specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <int Stages,
          bool CausalMask_, bool CachedKV_, bool PagedKV_,
          class TiledMMAQK_, class TiledMMAPV_, int VTiles_,
          class TensorQ_, class TensorK_, class TensorV_,
          class TensorK_cache_, class TensorV_cache_,
          class TiledCopyQ_, class TiledCopyK_, class TiledCopyV_,
          class TiledCopyK_cache_, class TiledCopyV_cache_>
struct FMHAFwdMainloop<XeDefault<Stages>, CausalMask_, CachedKV_, PagedKV_,
                       TiledMMAQK_, TiledMMAPV_, VTiles_,
                       TensorQ_, TensorK_, TensorV_,
                       TensorK_cache_, TensorV_cache_,
                       TiledCopyQ_, TiledCopyK_, TiledCopyV_,
                       TiledCopyK_cache_, TiledCopyV_cache_> {
  //
  // Type Aliases
  //
  using TiledMMAQK = TiledMMAQK_;
  using TiledMMAPV = TiledMMAPV_;
  using TileShapeQK = decltype(TiledMMAQK{}.tile_mnk());
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  static constexpr int VTiles = VTiles_;
  using SubgroupLayoutQK = decltype(TiledMMAQK{}.get_atom_layout_mnk());
  using SGPerWG = decltype(product(take<1,4>(shape(typename TiledMMAQK::ThrLayoutVMNK{}))));

  using TensorQ = TensorQ_;
  using TensorK = TensorK_;
  using TensorV = TensorV_;

  using TensorQ2D = decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_,_),0)));
  using TensorK2D = decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_,_),0)));
  using TensorV2D = decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_,_),0)));

  using TiledCopyQ = conditional_t<is_void_v<TiledCopyQ_>, decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{})), TiledCopyQ_>;
  using TiledCopyK = conditional_t<is_void_v<TiledCopyK_>, decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK2D{})), TiledCopyK_>;
  using TiledCopyV = conditional_t<is_void_v<TiledCopyV_>, decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV2D{})), TiledCopyV_>;
  using TensorK_cache = TensorK_cache_;
  using TensorV_cache = TensorV_cache_;
  using TensorK_cache2D = decltype(TensorK_cache_{}(append<rank_v<TensorK_cache_>>(make_coord(_,_),0)));
  using TensorV_cache2D = decltype(TensorV_cache_{}(append<rank_v<TensorV_cache_>>(make_coord(_,_),0)));
  using TiledCopyK_cache = conditional_t<is_void_v<TiledCopyK_cache_>, decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK_cache2D{})), TiledCopyK_cache_>;
  using TiledCopyV_cache = conditional_t<is_void_v<TiledCopyV_cache_>, decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV_cache2D{})), TiledCopyV_cache_>;

  // TODO: static_asserts on TiledMMAPV here...

  //
  // Accumulator types
  //
  // FragS:    accumulator for Q*K MMA
  // FragO:    accumulator for P*V MMAs.
  //           Note: v mode may be split into multiple pieces
  //             to reduce register pressure.
  // Frag*Row types are reductions of the corresponding Frag* types
  //   over rows.
  //
  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
                           make_identity_tensor(select<0,1>(TiledMMA{}.tile_mnk()))));

  using FragS = FragC<TiledMMAQK>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using ElementS = typename TiledMMAQK::ValTypeD;

  using SingleFragA = FragC<TiledMMAPV>;                          // (atom val,q',v')
  using FragA = expand_sg_fragment_t<SingleFragA, 1, VTiles>;     // (atom val,q',v',VV)
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
  using ElementA = typename TiledMMAPV::ValTypeD;

  static constexpr bool CausalMask = CausalMask_;
  static constexpr bool CachedKV = CachedKV_;
  static constexpr bool PagedKV = PagedKV_;

  // User-facing arguments
  struct Arguments {
    ElementS const scale;
    int const* ptr_page_table = nullptr;
    int page_size = 0;
    int const* num_pages_per_seq = nullptr;
  };

  // Kernel-facing parameters
  using Params = Arguments;

  // SLM data
  struct SharedStorage {};

  Params params;

  //
  // Methods
  //

  FMHAFwdMainloop(Params const& params_, SharedStorage&) : params(params_) {}

  static constexpr
  Params to_underlying_arguments(Arguments const &args, void * /* workspace */) {
    constexpr double kLog2e = 1.4426950408889634074;            // log_2(e)
    ElementS val = args.scale * static_cast<ElementS>(kLog2e);
    return Params{val, args.ptr_page_table, args.page_size, args.num_pages_per_seq};
  }

  CUTLASS_HOST_DEVICE static
  bool can_implement(Arguments const&) {
    return true;
  }

  CUTLASS_DEVICE
  int get_physical_k_tile(int K, int l_coord, int seq_len_kv_cache) {
    int next_page_logical_idx = K * get<1>(TileShapeQK{}) / params.page_size;
    // get<1>(TileShapeQK{}) usually smaller than page_size.
    // assuming page_size is multiple of get<1>(TileShapeQK{})
    int tiles_per_page = params.page_size / get<1>(TileShapeQK{});
    int batch_offset = params.num_pages_per_seq ? params.num_pages_per_seq[l_coord] : l_coord * (seq_len_kv_cache / params.page_size);

    return params.ptr_page_table[
          batch_offset +                  
          next_page_logical_idx] * tiles_per_page +            
          K % tiles_per_page; 
  }

  template <typename QVCoord>
  CUTLASS_DEVICE
  void
  operator()(TensorQ2D const& Q_2D,     // (q,d)
             TensorK2D const& K_2D,     // (k,d)
             TensorV2D const& V_2D,     // (d,k)
             FragA          & tArA,     // Output accumulator (q,v)
             FragARow       & tA_max,   // Softmax row-wise max accumulator
             FragARow       & tA_sum,   // Softmax row-wise sum accumulator
             QVCoord          blk_qv,   // WG tile indices: (Q,V)
             int              blk_k0,   // K block range: [K0,K1)
             int              blk_k1,
             int              total_blk, // Total # of K blocks
             int              thr_id,
             int              seq_len,
             int              seq_len_kv_cache,
             int              l_coord,
             int              full_tile_offset,
             int              discard_seq_coord,
            TensorK_cache2D const& K_cache_2D = TensorK_cache2D{},
            TensorV_cache2D const& V_cache_2D = TensorV_cache2D{}) {
    using namespace sycl::ext::oneapi::this_work_item;

    // Short dimension names:
    //    q = sequence len dimension for Q
    //    k = sequence len dimension for K
    //    d = head size dimension for K/Q
    //    v = head size dimension for V
    //   VV = MMA tile indices for V
    // Capital letters (Q, K, ...) refer to WG block indices.
    // Primed letters (q', k', ...) refer to atom block indices.

    auto tile_shape_v = make_shape(get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));

    /* Create proxy coordinate tensors for Q/K/P/V */
    Tensor cQ = make_identity_tensor(Q_2D.shape());             // (q,d)
    Tensor cK = make_identity_tensor(K_2D.shape());             // (k,d)
    Tensor cV = make_identity_tensor(V_2D.shape());             // (v,k)
    Tensor cK_cache = make_identity_tensor(K_cache_2D.shape()); // (k,d)
    Tensor cV_cache = make_identity_tensor(V_cache_2D.shape()); // (v,k)
    Tensor cP = make_identity_tensor(take<0,2>(TileShapeQK{})); // (q,k)

    /* Partition global tensors into workgroup tiles */
    Tensor gQ       = local_tile(cQ, TileShapeQK{}, append(blk_qv,_),             Step<_1,X,_1>{});   // (q,d,D)
    Tensor gK       = local_tile(cK, TileShapeQK{}, make_coord(_,_,_),            Step<X,_1,_1>{});   // (k,d,K,D)
    Tensor gV       = local_tile(cV, tile_shape_v,  make_coord(get<1>(blk_qv),_));                    // (v,k,K)
    Tensor gV_split = local_tile(gV, TileShapePV{}, make_coord(_,_,0),            Step<X,_1,_1>{});   // (v,k,VV,K)

    Tensor gK_cache       = local_tile(cK_cache, TileShapeQK{}, make_coord(_,_,_),            Step<X,_1,_1>{});   // (k,d,K,D)
    Tensor gV_cache       = local_tile(cV_cache, tile_shape_v,  make_coord(get<1>(blk_qv),_));                    // (v,k,K)
    Tensor gV_cache_split = local_tile(gV_cache, TileShapePV{}, make_coord(_,_,0),            Step<X,_1,_1>{});   // (v,k,VV,K)

    /* Create global -> register copies */
    TiledCopyQ copy_q{Q_2D};
    TiledCopyK copy_k{K_2D};
    TiledCopyV copy_v{V_2D};
    TiledCopyK_cache copy_k_cache{K_cache_2D};
    TiledCopyV_cache copy_v_cache{V_cache_2D};

    /* Create MMAs */
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};

    /* Slice TiledCopy/TiledMMA operations down to to work-item level */
    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_copy_k = copy_k.get_slice(thr_id);
    auto thr_copy_v = copy_v.get_slice(thr_id);
    auto thr_copy_k_cache = copy_k_cache.get_slice(thr_id);
    auto thr_copy_v_cache = copy_v_cache.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);

    /* Partition coordinate tensors for copy */
    auto tQgQ = thr_copy_q.partition_S(gQ);                // (atom_val,q',d',D)
    auto tKgK = thr_copy_k.partition_S(gK);                // (atom_val,k',d',K,D)
    auto tVgV = thr_copy_v.partition_S(gV_split);          // (atom_val,v',k',VV,K)
    auto tKgK_cache = thr_copy_k_cache.partition_S(gK_cache);
    auto tVgV_cache = thr_copy_v_cache.partition_S(gV_cache_split);

    /* Create register fragments for MMA and copies */
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_,_,0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_,_,0));

    auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_,_,0,0));
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_,_,0,0));

    // FragAforP: named type for the P matrix fragment used in P*V GEMM.
    // Named here so we can declare a 2-element array for the pipeline.
    using FragAforP = decltype(thr_mma_pv.partition_sg_fragment_A(cP));

    // 2-slot double-buffer pipeline fragments.
    //
    // Pipeline scheduling overview:
    //   Prologue:     QK(blk_k0)           -> tSrS_pipe[0]          [XMX]
    //   Steady-state: QK(k+1)              -> tSrS_pipe[next_slot]  [XMX]
    //                 mask(k) + softmax(k) on tSrS_pipe[cur_slot]   [XVX - overlaps QK above]
    //                 PV(k)  using tArP_pipe[cur_slot]              [XMX - after softmax]
    //   Epilogue:     mask + softmax + PV on tSrS_pipe[last_slot]   (no next-QK to overlap)
    //
    // Placing QK(next) BEFORE softmax(cur) in the instruction stream gives the
    // hardware scheduler the best opportunity to overlap XMX (matrix engine)
    // and XVX (vector/scalar) work concurrently.
    FragS     tSrS_pipe[2];   // Q*K accumulator, double-buffered over K-blocks
    FragAforP tArP_pipe[2];   // reordered P (softmax output) for P*V input, double-buffered

    auto tVrV = thr_copy_v.partition_sg_fragment_D(gV_split(_,_,0,0));
    auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_,_,0,0));

    /* Create TiledCopy objects for prefetches */
    auto prefetch_q = make_block_2d_prefetch(copy_q);
    auto prefetch_k = make_block_2d_prefetch(copy_k);
    auto prefetch_v = make_block_2d_prefetch(copy_v);
    auto prefetch_k_cache = make_block_2d_prefetch(copy_k_cache);
    auto prefetch_v_cache = make_block_2d_prefetch(copy_v_cache);

    /* Partition global tensors for prefetch */
    auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
    auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK);
    auto pVgV = prefetch_v.get_slice(thr_id).partition_S(gV_split);
    auto pKgK_cache = prefetch_k_cache.get_slice(thr_id).partition_S(gK_cache);
    auto pVgV_cache = prefetch_v_cache.get_slice(thr_id).partition_S(gV_cache_split);

    // ------
    // Kernel
    // ------

    /* Initialization steps for first block: Q/K prefetch, O init */
    /* TODO: limit D prefetch for large head size, and reorder K prefetches */
    int kblocks_cache = ceil_div(seq_len_kv_cache, get<1>(TileShapeQK{}));
    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_,_,_,D));
    }
    for (int D = 0; D < size<4>(pKgK); D++) {
      CUTLASS_PRAGMA_UNROLL
      for (int K = 0; K < Stages; K++) {
        if (K < kblocks_cache) {
          if constexpr (PagedKV) {
            int physical_K_tile = get_physical_k_tile(K, l_coord, seq_len_kv_cache);
            prefetch(prefetch_k_cache, pKgK_cache(_,_,_,physical_K_tile,D));
          } else {
            prefetch(prefetch_k_cache, pKgK_cache(_,_,_,K,D));
          }
        } else {
          prefetch(prefetch_k, pKgK(_,_,_,K - kblocks_cache,D));
        }
      }
    }
    if (blk_k0 == 0) {
      clear(tArA);
      fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
      clear(tA_sum);
    }

    /* Check if the last K tile is a partial (remainder) tile. */
    bool check_remainder_k = (seq_len % get<1>(TileShapeQK{}) != 0);

    // =====================================================================
    // Pipeline helper lambdas
    // =====================================================================

    // Build a per-block descriptor: maps logical K index to its physical index
    // and whether it comes from the KV cache or the main KV tensors.
    // This unifies causal/paged/cached dispatch so a single pipeline loop
    // can drive all K-block variants.
    struct BlockDesc {
      int  logical_k;   // logical block index (used for mask boundary checks)
      int  physical_k;  // physical block index (0-based in the tensor it indexes)
      bool is_cache;    // true => use KV-cache tensors; false => main KV tensors
    };

    auto make_block_desc = [&](int K) -> BlockDesc {
      bool is_cache = CachedKV && (K < kblocks_cache);
      int physical_k;
      if (is_cache) {
        physical_k = K;
        if constexpr (PagedKV) {
          physical_k = get_physical_k_tile(K, l_coord, seq_len_kv_cache);
        }
      } else {
        physical_k = K - kblocks_cache;
      }
      return {K, physical_k, is_cache};
    };

    // Stage A: Compute Q*K for block desc and accumulate into tSrS_dst.
    // Also issues V prefetch for the subsequent P*V GEMM.
    //
    // Overlap target: the XMX DPAS instructions emitted here can run
    // concurrently with the XVX (scalar/vector) softmax work issued on a
    // *different* pipeline slot immediately afterwards.
    auto compute_qk_block = [&](BlockDesc const& desc, FragS& tSrS_dst) {
      clear(tSrS_dst);
      if (desc.is_cache) {
        CUTLASS_PRAGMA_UNROLL
        for (int D = 0; D < size<4>(tKgK); D++) {
          copy(copy_q,       tQgQ(_,_,_,D),                  tQrQ);
          copy(copy_k_cache, tKgK_cache(_,_,_,desc.physical_k,D), tKrK);
          reorder(tQrQ, tSrQ);
          reorder(tKrK, tSrK);
          cute::gemm(mma_qk, tSrQ, tSrK, tSrS_dst);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int VV = 0; VV < VTiles; VV++) {
          prefetch(prefetch_v_cache, pVgV_cache(_,_,_,VV,desc.physical_k));
        }
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int D = 0; D < size<4>(tKgK); D++) {
          copy(copy_q, tQgQ(_,_,_,D),              tQrQ);
          copy(copy_k, tKgK(_,_,_,desc.physical_k,D), tKrK);
          reorder(tQrQ, tSrQ);
          reorder(tKrK, tSrK);
          cute::gemm(mma_qk, tSrQ, tSrK, tSrS_dst);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int VV = 0; VV < VTiles; VV++) {
          prefetch(prefetch_v, pVgV(_,_,_,VV,desc.physical_k));
        }
      }
    };

    // Stage B: Apply causal mask and K-remainder mask to a pipeline S slot.
    // Only active for non-cache blocks that sit at a sequence boundary.
    // These are scalar operations and are part of the XVX overlap window.
    auto apply_mask_block = [&](BlockDesc const& desc, FragS& tSrS_slot) {
      if (!desc.is_cache) {
        if constexpr (CausalMask) {
          if (desc.logical_k == total_blk - 1) {
            Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
            Tensor gP   = local_tile(cPgP, take<0,2>(TileShapeQK{}),
                                     make_coord(get<0>(blk_qv), desc.logical_k));
            auto cS_thread = thr_mma_qk.partition_C(gP);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tSrS_slot.size(); ++i) {
              int row_idx = get<0>(cS_thread(i));
              int col_idx = get<1>(cS_thread(i));
              if (col_idx - seq_len_kv_cache - full_tile_offset >
                  row_idx - discard_seq_coord) {
                tSrS_slot(i) = ElementS(-INFINITY);
              }
            }
          }
        }
        if (check_remainder_k && desc.logical_k == total_blk - 1) {
          FragSRow k_rem_mask;
          int k_val = get<0>(tKgK(0,0,0,desc.physical_k,0))
                      + kblocks_cache * get<1>(TileShapeQK{});
          int k = k_val + get_sub_group().get_local_id()[0];
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
            k_rem_mask(i) = (k < seq_len) ? ElementS(sycl::nan(0u))
                                           : ElementS(-INFINITY);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS_slot.size(); i++) {
            tSrS_slot(i) = sycl::fmin(tSrS_slot(i),
                                      broadcast<1>(k_rem_mask, tSrS_slot, i));
          }
        }
      }
    };

    // Stage C: Online softmax update + reorder S into P layout for P*V GEMM.
    // These are XVX (vector/scalar ALU) operations. Placing them after
    // compute_qk_block (Stage A) for the NEXT slot gives the hardware
    // the opportunity to overlap XMX (Stage A) with XVX (this stage).
    // Returns the per-row rescale factor for updating the output accumulator.
    auto apply_softmax_and_prepare_p = [&](bool first_block,
                                           FragS     & tSrS_slot,
                                           FragAforP & tArP_slot) -> FragSRow {
      auto rescale = softmax(first_block, tSrS_slot, tA_max, tA_sum);
      reorder(tSrS_slot, tArP_slot);
      return rescale;
    };

    // Stage D: Accumulate P*V into the output accumulator tArA.
    // Applies the softmax rescale to the existing accumulator before
    // adding the new P*V contribution.  Kept as a separate phase from
    // compute_qk_block to avoid two XMX streams competing simultaneously.
    auto accumulate_pv_block = [&](BlockDesc   const& desc,
                                   FragAforP   const& tArP_slot,
                                   FragSRow    const& rescale,
                                   bool               first_block) {
      if (desc.is_cache) {
        CUTLASS_PRAGMA_UNROLL
        for (int VV = 0; VV < VTiles; VV++) {
          copy(copy_v_cache, tVgV_cache(_,_,_,VV,desc.physical_k), tVrV);
          reorder(tVrV, tArV);
          if (!first_block) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tArA.size() / VTiles; i++)
              tArA(_,_,_,VV)(i) *= broadcast<0>(rescale, tArA, i);
          }
          cute::gemm(mma_pv, tArP_slot, tArV, tArA(_,_,_,VV));
        }
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int VV = 0; VV < VTiles; VV++) {
          copy(copy_v, tVgV(_,_,_,VV,desc.physical_k), tVrV);
          reorder(tVrV, tArV);
          if (!first_block) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tArA.size() / VTiles; i++)
              tArA(_,_,_,VV)(i) *= broadcast<0>(rescale, tArA, i);
          }
          cute::gemm(mma_pv, tArP_slot, tArV, tArA(_,_,_,VV));
        }
      }
    };

    // Prefetch K tiles for a future block (hint; silently skips out-of-range
    // indices as prefetch is a non-faulting hint on Intel Xe).
    auto prefetch_k_block = [&](int K_next) {
      bool is_cache_next = CachedKV && (K_next < kblocks_cache);
      for (int D = 0; D < size<4>(pKgK); D++) {
        if (is_cache_next) {
          int phys = K_next;
          if constexpr (PagedKV) {
            phys = get_physical_k_tile(K_next, l_coord, seq_len_kv_cache);
          }
          prefetch(prefetch_k_cache, pKgK_cache(_,_,_,phys,D));
        } else {
          prefetch(prefetch_k, pKgK(_,_,_,K_next - kblocks_cache,D));
        }
      }
    };

    // =====================================================================
    // 2-slot block pipeline
    //
    // Unified K range [k_start, k_end).  For CachedKV kernels,
    // K in [k_start, kblocks_cache) are cache blocks and
    // K in [kblocks_cache, k_end) are main KV blocks; make_block_desc
    // handles the dispatch transparently.
    // =====================================================================
    const int k_start  = blk_k0;
    const int k_end    = blk_k1;
    const int n_blocks = k_end - k_start;

    if (n_blocks == 0) return;

    // ------------------------------------------------------------------
    // Prologue: compute QK for the first block into pipeline slot 0.
    // ------------------------------------------------------------------
    {
      barrier_arrive(ScopeWorkgroup);
      compute_qk_block(make_block_desc(k_start), tSrS_pipe[0]);
      barrier_wait(ScopeWorkgroup);
    }

    // ------------------------------------------------------------------
    // Steady-state: for k = k_start .. k_end-2 overlap
    //   [XMX]        QK(k+1)     -> tSrS_pipe[next_slot]
    //   [XVX/scalar] softmax(k)  on tSrS_pipe[cur_slot]   (targets XVX; can overlap XMX above)
    //   [XMX]        PV(k)       using tArP_pipe[cur_slot] (after softmax; separate XMX phase)
    // ------------------------------------------------------------------
    for (int k = k_start; k < k_end - 1; k++) {
      const int cur_slot  = (k - k_start) & 1;
      const int next_slot = cur_slot ^ 1;

      auto cur_desc  = make_block_desc(k);
      auto next_desc = make_block_desc(k + 1);

      barrier_arrive(ScopeWorkgroup);

      // Issue QK for the NEXT block first (XMX).
      // The subsequent softmax work targets XVX and can overlap with this.
      compute_qk_block(next_desc, tSrS_pipe[next_slot]);

      // Apply mask and run softmax on the CURRENT block's S slot (XVX path).
      // These scalar/vector operations are the primary overlap target with
      // the XMX work issued by compute_qk_block above.
      apply_mask_block(cur_desc, tSrS_pipe[cur_slot]);
      auto rescale = apply_softmax_and_prepare_p(k == k_start,
                                                 tSrS_pipe[cur_slot],
                                                 tArP_pipe[cur_slot]);

      // Accumulate P*V for the current block (XMX, runs after softmax).
      // Kept after the softmax window to avoid two concurrent XMX streams.
      accumulate_pv_block(cur_desc, tArP_pipe[cur_slot], rescale, k == k_start);

      // Prefetch K for Stages iterations ahead of the next block
      prefetch_k_block(k + Stages + 1);

      barrier_wait(ScopeWorkgroup);
    }

    // ------------------------------------------------------------------
    // Epilogue: drain the final block.
    // The last block's QK result already sits in tSrS_pipe[last_slot];
    // just apply mask + softmax + PV with no next-QK to overlap.
    // ------------------------------------------------------------------
    {
      const int last_k    = k_end - 1;
      const int last_slot = (last_k - k_start) & 1;
      auto last_desc = make_block_desc(last_k);

      apply_mask_block(last_desc, tSrS_pipe[last_slot]);
      auto rescale = apply_softmax_and_prepare_p(last_k == k_start,
                                                 tSrS_pipe[last_slot],
                                                 tArP_pipe[last_slot]);
      accumulate_pv_block(last_desc, tArP_pipe[last_slot], rescale, last_k == k_start);
    }
  }


  // Single step of blocked softmax.
  CUTLASS_DEVICE
  FragSRow
  softmax(bool       first_block, // First softmax block?
          FragS    & tS,          // Softmax src/dst block
          FragSRow & tS_max,      // Softmax row-wise max accumulator
          FragSRow & tS_sum) {    // Softmax row-wise sum accumulator
    /* Compute row-wise maxima for this block */
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    FragSRow rescale;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      ElementS new_max = sycl::max(tS_max(i), params.scale * tS_bmax(i));
      rescale(i) = sycl::native::exp2(tS_max(i) - new_max);
      tS_max(i) = new_max;
    }

    /* Scale S and subtract maxima, then exponentiate */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = sycl::native::exp2(params.scale * tS(i) - broadcast<0>(tS_max, tS, i));

    /* Rescale existing S sums */
    if (!first_block) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_sum.size(); i++) {
        tS_sum(i) *= rescale(i);
      }
    }

    /* Update sums */
    auto tS_bsum = reduce<1>(tS, sycl::plus<void>{});
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_sum.size(); i++)
      tS_sum(i) += tS_bsum(i);

    return rescale;
  }
};


template <typename SGLayoutQK>
CUTLASS_HOST_DEVICE
constexpr auto
get_sg_layout_pv(SGLayoutQK const&)
{
  return make_layout(
    get<0>(SGLayoutQK{}),
    Layout<_1, _0>{},
    get<1>(SGLayoutQK{})
  );
}

// Get a P*V TiledMMA given K*Q tile size and SG configuration, for mainloops
//   not supporting S data interchange among subgroups (e.g. XeDefault).
template <typename MMAOp,
          typename WGTileQK,
          typename SGLayoutQK,
          typename TileV>
CUTLASS_HOST_DEVICE
constexpr auto
get_tiled_mma_pv(MMAOp const&, WGTileQK const& wg_tile_qk, SGLayoutQK const& sg_layout_qk, TileV const&) {
  using TileQ = decltype(get<0>(wg_tile_qk));
  using TileK = decltype(get<1>(wg_tile_qk));

  using WGTilePV = Shape<TileQ, TileV, TileK>;
  using SGLayoutPV = decltype(get_sg_layout_pv(sg_layout_qk));

  static_assert(size(SGLayoutPV{}) == size(SGLayoutQK{}),
                "Q*K cannot be parallelized in the head size dimension");

  return TiledMMAHelper<MMAOp, WGTilePV, SGLayoutPV>{};
}

} // namespace cutlass::fmha::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
