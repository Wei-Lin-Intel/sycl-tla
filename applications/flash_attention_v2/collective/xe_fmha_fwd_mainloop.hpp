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

  // ---- Ring-Attention P2P (packed K/V) ----
  // The ring recv buffer holds PACKED K/V (identical layout to the global
  // K/V tile this kernel already consumes), NOT MMA-B fragments. On the
  // consume round we simply point the normal block-2D load+reorder path at
  // the recv buffer instead of global memory. This keeps the peer buffer at
  // packed size (s_local*h_kv*d) and reuses the proven reorder path, so the
  // result is bit-identical to a plain load.
  using RingValType = typename TiledMMAQK::ValTypeB;
  static_assert(cute::is_same_v<RingValType, typename TiledMMAPV::ValTypeB>,
                "Ring P2P requires identical K/V MMA-B element types");

  static constexpr bool CausalMask = CausalMask_;
  static constexpr bool CachedKV = CachedKV_;
  static constexpr bool PagedKV = PagedKV_;

  // User-facing arguments
  struct Arguments {
    ElementS const scale;
    int const* ptr_page_table = nullptr;
    int page_size = 0;
    int const* num_pages_per_seq = nullptr;
    // ---- Ring-Attention P2P (packed K/V, batch == 1, non-causal) ----
    // Consume side (rounds 1..N-1): when ring_consume is set, K/V for THIS
    // round are read from this rank's own recv buffer, which holds PACKED
    // K/V written by the previous round's peer push (a plain packed memcpy on
    // the host side). The kernel loads + reorders it exactly like global K/V.
    // Push is done host-side (packed IPC memcpy), so no send-side kernel work
    // is needed here anymore.
    bool ring_consume = false;
    void const* ring_recv_k = nullptr; // packed [s_local, h_kv, d_qk] bf16
    void const* ring_recv_v = nullptr; // packed [s_local, h_kv, d_vo] bf16
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
    return Params{val, args.ptr_page_table, args.page_size, args.num_pages_per_seq,
                  args.ring_consume, args.ring_recv_k, args.ring_recv_v};
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
	     int              ring_head_idx = 0,
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

    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);

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

    /* ------ Ring-Attention P2P: PACKED recv buffers ------
       On the consume round, K/V come from params.ring_recv_k / ring_recv_v,
       which are packed [k, d] (K) and [d, k] (V) tiles for THIS (batch,head),
       laid out identically to the global K_2D / V_2D this kernel normally
       loads. We build 2D tensors over the recv buffers and reuse the same
       TiledCopyK / TiledCopyV + reorder path -> bit-identical to a plain load.

       Row strides:
         K recv: shape (k, d), stride (d_qk, 1)
         V recv: shape (d, k), stride (1, d_vo)   (V is stored [d, k] = "vnni-
                 friendly" packed, matching V_2D's (v,k) coordinate tensor).
       The recv buffer for one (batch,head) is contiguous with row stride
       ring_recv_{k,v}_row_stride (== d_qk / d_vo for the packed per-head tile). */
    using RingElemK = typename TensorK2D::element_type;
    using RingElemV = typename TensorV2D::element_type;

    // The recv buffer is packed [s_local, h_kv, d]; the K/V 2D tile passed in
    // is already sliced to THIS head, so the recv view must skip to this head's
    // sub-block. ring_head_idx = head index for this WG (batch==1 ring scope).
    // K packed row stride = h_kv * d_qk, so head h starts at h * d_qk within
    // each row; V packed row stride = h_kv * d_vo -> h * d_vo.
    auto make_recvK_2D = [&]() {
      auto base = reinterpret_cast<RingElemK const*>(params.ring_recv_k);
      int d = get<1>(K_2D.shape());
      auto ptr = make_gmem_ptr(base + static_cast<int64_t>(ring_head_idx) * d);
      return make_tensor(ptr, make_layout(K_2D.shape(), K_2D.stride()));
    };
    auto make_recvV_2D = [&]() {
      auto base = reinterpret_cast<RingElemV const*>(params.ring_recv_v);
      int v = get<0>(V_2D.shape());
      auto ptr = make_gmem_ptr(base + static_cast<int64_t>(ring_head_idx) * v);
      return make_tensor(ptr, make_layout(V_2D.shape(), V_2D.stride()));
    };

    /* ------ Ring-Attention P2P: hoist recv tensor/copy/partition setup ------
       These constructions (recv 2D view, TiledCopy, thread slice, partition_S)
       depend only on loop-invariant quantities (params, ring_head_idx, K/V 2D
       shape+stride), NOT on the K / D / VV loop indices. Building them here once
       instead of inside mainloop_body avoids re-constructing them every K-block
       (and every D / VV iteration). They are cheap layout+pointer computations
       with no memory access, so building them unconditionally is harmless even
       when params.ring_consume is false; the actual loads below are still
       guarded by the ring_consume branch. The per-K index (k_idx) is applied at
       the copy() call sites, exactly as for the global path. */
    auto recvK_2D = make_recvK_2D();
    TiledCopyK copy_k_recv{recvK_2D};
    auto thr_copy_k_recv = copy_k_recv.get_slice(thr_id);
    auto gK_recv   = local_tile(make_identity_tensor(recvK_2D.shape()),
                                TileShapeQK{}, make_coord(_,_,_), Step<X,_1,_1>{});
    auto tKgK_recv = thr_copy_k_recv.partition_S(gK_recv);
    auto prefetch_k_recv = make_block_2d_prefetch(copy_k_recv);
    auto pKgK_recv = prefetch_k_recv.get_slice(thr_id).partition_S(gK_recv);

    auto recvV_2D = make_recvV_2D();
    TiledCopyV copy_v_recv{recvV_2D};
    auto thr_copy_v_recv = copy_v_recv.get_slice(thr_id);
    auto gV_recv       = local_tile(make_identity_tensor(recvV_2D.shape()),
                                    tile_shape_v, make_coord(get<1>(blk_qv),_));
    auto gV_recv_split = local_tile(gV_recv, TileShapePV{}, make_coord(_,_,0), Step<X,_1,_1>{});
    auto tVgV_recv     = thr_copy_v_recv.partition_S(gV_recv_split);
    auto prefetch_v_recv = make_block_2d_prefetch(copy_v_recv);
    auto pVgV_recv = prefetch_v_recv.get_slice(thr_id).partition_S(gV_recv_split);

    // ------
    // Kernel
    // ------

    /* Initialization steps for first block: Q/K prefetch, O init */
    /* TODO: limit D prefetch for large head size, and reorder K prefetches */
    int kblocks_cache = ceil_div(seq_len_kv_cache, get<1>(TileShapeQK{}));

    /* Ring-Attention P2P fragment-slot descriptors. Kept identical on the
       push (send) and pull (receive) sides so the flat layout never drifts.
         K slot = ((head * kTilesRing + k_idx) * nD_qk + D) * NumThreadsQK + thr
         V slot = ((head * kTilesRing + k_idx) * VTiles + VV) * NumThreadsQK + thr
       nD_qk : # of head-dim (D) sub-tiles of the Q*K MMA-B fragment.
       ring_head : head index for this WG (ring scope is batch==1, so l_coord
                   enumerates heads only).
       kTilesRing : # of non-cache (ring) K tiles. */
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
	  // Non-cache (ring) K: when consuming this round, the K tile is read
          // from the peer recv buffer, so prime the prefetch from pKgK_recv;
          // otherwise prefetch global K. Mirrors the in-loop K prefetch below.
          if (params.ring_consume) {
            prefetch(prefetch_k_recv, pKgK_recv(_,_,_,K - kblocks_cache,D));
          } else {
            prefetch(prefetch_k, pKgK(_,_,_,K - kblocks_cache,D));
          }
        }
      }
    }
    if (blk_k0 == 0) {
      clear(tArA);
      fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
      clear(tA_sum);
    }

    /* Check if */
    bool check_remainder_k = (seq_len % get<1>(TileShapeQK{}) != 0);

    /* Main loop body */
    auto mainloop_body = [&](auto cached_k, int K,
                            auto& copy_k_cur, auto& copy_v_cur,
                            auto& prefetch_v_cur, auto& tKgK_cur,
                            auto& tVgV_cur, auto& pVgV_cur) {
      /* Split barrier to keep threads together */
      barrier_arrive(ScopeWorkgroup);
      constexpr bool is_cache = decltype(cached_k)::value;

      int k_idx;
      if constexpr (is_cache) {
        k_idx = K;
        if constexpr (PagedKV) {
          k_idx = get_physical_k_tile(K, l_coord, seq_len_kv_cache);
        }
      } else {
        k_idx = K - kblocks_cache;
      }

      /* GEMM 1: S = K * Q */
      clear(tSrS);
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < size<4>(tKgK); D++) {
        copy(copy_q, tQgQ(_,_,_,D), tQrQ);
        reorder(tQrQ, tSrQ);

        if constexpr (!is_cache) {
          if (params.ring_consume) {
	    // Round 1..N-1: load K from the PACKED recv buffer, then reorder
            // exactly like the global path. Bit-identical to a plain load.
            // (recv tensor/copy/partition are hoisted above the loop.)
            copy(copy_k_recv, tKgK_recv(_,_,_,k_idx,D), tKrK);
          } else {
            copy(copy_k_cur, tKgK_cur(_,_,_,k_idx,D), tKrK);
          }
        } else {
          copy(copy_k_cur, tKgK_cur(_,_,_,k_idx,D), tKrK);
        }
	reorder(tKrK, tSrK);

        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }

      /* V prefetch for GEMM 2 */
      // V for this round comes from the peer recv buffer on the non-cache
      // ring-consume path, from global V otherwise. Both are global memory
      // and must be prefetched.
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        if constexpr (!is_cache) {
          if (params.ring_consume) {
            prefetch(prefetch_v_recv, pVgV_recv(_,_,_,VV,k_idx));
          } else {
            prefetch(prefetch_v_cur, pVgV_cur(_,_,_,VV,k_idx));
          }
        } else {
          prefetch(prefetch_v_cur, pVgV_cur(_,_,_,VV,k_idx));
        }
      }

      /* Causal masking - only in non-cache mode */
      if constexpr (!is_cache && CausalMask) {
        if (K == total_blk - 1) {
          // Need to get global col and row indices to mask the elements
          Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
          Tensor gP = local_tile(cPgP, take<0,2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
          auto cS_thread = thr_mma_qk.partition_C(gP);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            int row_idx = get<0>(cS_thread(i));
            int col_idx = get<1>(cS_thread(i));
            if (col_idx - seq_len_kv_cache - full_tile_offset > row_idx - discard_seq_coord) {
              tSrS(i) = ElementS(-INFINITY);
            }
          }
        }
      }
      /* k masking for remainder tiles */
      if constexpr (!is_cache) {
        if (check_remainder_k && K == total_blk - 1) {
          FragSRow k_rem_mask;
          int k_val = get<0>(tKgK_cur(0,0,0,k_idx,0)) + kblocks_cache * get<1>(TileShapeQK{});
          int k = k_val + get_sub_group().get_local_id()[0];
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
            k_rem_mask(i) = (k < seq_len) ? ElementS(sycl::nan(0u)) : ElementS(-INFINITY);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); i++) {
            tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(k_rem_mask, tSrS, i));
          }
        }
      }

      /* Apply softmax and scaling (tA rescaling fused into GEMM2 VTile loop) */
      auto rescale = softmax(K == blk_k0, tSrS, tA_max, tA_sum);
      reorder(tSrS, tArP);

      /* GEMM 2: A += P * V, split in v dimension.
        tArA rescaling is fused to per-VTile */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        if constexpr (!is_cache) {
          if (params.ring_consume) {
	    // Round 1..N-1: load V from the PACKED recv buffer + reorder.
            // (recv tensor/copy/partition are hoisted above the loop.)
            copy(copy_v_recv, tVgV_recv(_,_,_,VV,k_idx), tVrV);
          } else {
            copy(copy_v_cur, tVgV_cur(_,_,_,VV,k_idx), tVrV);
          }
        } else {
          copy(copy_v_cur, tVgV_cur(_,_,_,VV,k_idx), tVrV);
        }
	reorder(tVrV, tArV);

        if (K != blk_k0) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tArA.size() / VTiles; i++)
            tArA(_,_,_,VV)(i) *= broadcast<0>(rescale, tArA, i);
        }

        cute::gemm(mma_pv, tArP, tArV, tArA(_,_,_,VV));
      }

      /* K prefetch */
      int K_next = K + Stages;
      for (int D = 0; D < size<4>(pKgK); D++) {
        if constexpr (is_cache) {
          bool is_cache_next = K_next < kblocks_cache;
          int physical_K_next = K_next;
          if constexpr (PagedKV) {
            if (is_cache_next) {
              physical_K_next = get_physical_k_tile(K_next, l_coord, seq_len_kv_cache);
            }
          }
          if (is_cache_next) {
            prefetch(prefetch_k_cache, pKgK_cache(_,_,_,physical_K_next,D));
          } else {
            // Next tile is non-cache (ring): prefetch recv buffer when
            // consuming, otherwise global K.
            if (params.ring_consume) {
              prefetch(prefetch_k_recv, pKgK_recv(_,_,_,K_next-kblocks_cache,D));
            } else {
              prefetch(prefetch_k, pKgK(_,_,_,K_next-kblocks_cache,D));
            }
          }
        } else {
          if (params.ring_consume) {
            prefetch(prefetch_k_recv, pKgK_recv(_,_,_,K_next-kblocks_cache,D));
          } else {
            prefetch(prefetch_k, pKgK(_,_,_,K_next-kblocks_cache,D));
          }
        }
      }
      barrier_wait(ScopeWorkgroup);
    };

    /* Main loop, blocked in k. */
    if constexpr (CachedKV) {
      for (int K = blk_k0; K < kblocks_cache; K++) {
        mainloop_body(std::bool_constant<true>{}, K,
                      copy_k_cache, copy_v_cache,
                      prefetch_v_cache, tKgK_cache,
                      tVgV_cache, pVgV_cache);
      }
    }

    for (int K = (blk_k0 > kblocks_cache ? blk_k0 : kblocks_cache); K < blk_k1; K++) {
      mainloop_body(std::bool_constant<false>{}, K,
                    copy_k, copy_v,
                    prefetch_v, tKgK,
                    tVgV, pVgV);
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
