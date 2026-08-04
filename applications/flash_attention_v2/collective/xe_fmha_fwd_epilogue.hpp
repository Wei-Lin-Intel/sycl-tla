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

#include <sycl/sycl.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/detail/layout.hpp"

#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/algorithm/tensor_algorithms.hpp"

#include "copy_block_slm.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

template <class CollectiveMainloop, // Attention mainloop
          class TileShapeO_,        // Shape of output tile, may be larger than P*V GEMM
          class TensorO_,           // 2D slice of global output tensor
	  class TiledCopyO_ = void, // Optional TiledCopy for loading O
          bool EnableLSE_ = false>
class FMHAFwdEpilogue {

public:
  static constexpr bool EnableLSE = EnableLSE_;

  //
  // Type Aliases
  //
  using TiledMMAPV = typename CollectiveMainloop::TiledMMAPV;
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  using TileShapeO = TileShapeO_;
  using SGPerWG = decltype(product(take<1,4>(shape(typename TiledMMAPV::ThrLayoutVMNK{}))));

  using TensorO = TensorO_;
  using TensorO2D = decltype(TensorO_{}(append<rank_v<TensorO_>>(make_coord(_,_),0)));
  using ElementO = typename TensorO_::value_type;

  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;
  using ElementA = typename FragA::value_type;

  // Split k-reduced tiles between participating subgroups.
  // Assumption: the A tile is contiguous.
  using ReduceK = decltype(size<3>(typename TiledMMAPV::ThrLayoutVMNK{}));

  static auto reduce_sg_v_helper() {
    constexpr auto v_total_sg = get<1>(SGTileShapeA{}) / intel::_SGSize{};
    constexpr auto v_avail_sg = ReduceK{} / ReduceSGQ{};
    return Int<(v_total_sg > v_avail_sg) ? cute::gcd(v_total_sg, v_avail_sg) : v_total_sg>{};
  }

  using SGTileShapeA = decltype(atuple_coshape(FragA{}.tv_layout()));
  using ReduceSGQ = decltype(cute::gcd(get<0>(SGTileShapeA{}), ReduceK{}));
  using ReduceSGV = decltype(reduce_sg_v_helper());
  using ReduceSGLayout = decltype(make_identity_layout(Shape<ReduceSGQ, ReduceSGV>{}));

  using SGTileShapeO = decltype(shape_div(take<0,2>(SGTileShapeA{}), shape(ReduceSGLayout{})));

  using ReduceFragA = decltype(make_subgroup_tensor<ElementA>(
    make_layout(select<1,0>(SGTileShapeO{}),
                Stride<E<1>, E<0>>{})
  ));
  using ReduceFragARow = decltype(reduce<1>(ReduceFragA{}, sycl::plus<void>{}));

  static auto default_tiled_copy_O_helper() {
    if constexpr (ReduceK{} == _1{})
      return make_block_2d_copy_D(TiledMMAPV{}, TensorO2D{});
    else
      return make_block_2d_copy_D_subtiled(TiledMMAPV{}, ReduceFragA{}.tv_layout(), ReduceSGLayout{}, TensorO2D{});
  }

  using DefaultTiledCopyO = decltype(default_tiled_copy_O_helper());
  using TiledCopyO = conditional_t<is_void_v<TiledCopyO_>, DefaultTiledCopyO, TiledCopyO_>;

  static auto default_tiled_load_O_helper() {
    if constexpr (ReduceK{} == _1{})
      return make_block_2d_copy_C(TiledMMAPV{}, TensorO2D{});
    else
      return make_block_2d_copy_C_subtiled(TiledMMAPV{}, ReduceFragA{}.tv_layout(), ReduceSGLayout{}, TensorO2D{});
  }

  using TiledLoadO = decltype(default_tiled_load_O_helper());

  struct Arguments {
    ElementA* lse = nullptr;
    int stride_lse_q = 0;
    int stride_lse_h = 0;
    int stride_lse_b = 0;
    bool accumulate = false;
  };
  using Params = Arguments;

  // Shared memory storage
  // Note sum/max tiles are padded to 16 elements, due to limitations in CuTe block load infrastructure.
  using AlignedSGTileA_Q = C<((size<0>(SGTileShapeA{}) + intel::sg_size - 1) / intel::sg_size) * intel::sg_size>;

  struct SharedStorageNone {};
  struct SharedStorageReduceK {
    cute::array<ElementA, size(SGTileShapeA{}) * SGPerWG{}> a_data;
    cute::array<ElementA,   AlignedSGTileA_Q{} * SGPerWG{}> a_sum_data, a_max_data;
  };

  using SharedStorage = conditional_t<(ReduceK{} > _1{}), SharedStorageReduceK, SharedStorageNone>;

private:
  SharedStorage &shared;

public:
  static constexpr
  Params to_underlying_arguments(Arguments const &args, void * /* workspace */) {
    return args;
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const& args) {
    if constexpr (EnableLSE) {
      return !args.accumulate || args.lse;
    } else {
      return !args.accumulate && args.lse == nullptr;
    }
  }

  CUTLASS_HOST_DEVICE
  FMHAFwdEpilogue(Params const& params_, SharedStorage& shared_)
      : shared(shared_), params(params_) {}

  template <typename QVCoord>
  CUTLASS_DEVICE
  void
  operator()(TensorO2D const& O,        // Global O tensor: (q,v)
             FragA          & tArA,     // O accumulator:   (q,v)
             FragARow       & tA_max,   // Softmax row-wise max accumulator
             FragARow       & tA_sum,   // Softmax row-wise sum accumulator
             QVCoord          blk_qv,   // WG tile indices: (q,v)
             int              thr_id,   // Work-item ID
             int              head_q,
             int              idx_b) {

    using namespace cute;
    using ElementA = typename FragA::element_type;

    // Reduce k-blocks of A and A_sum across WG, if needed.
    auto [rA, rA_sum, rA_max, active] =
        reduce_A(tArA, tA_max, tA_sum, thr_id);

    /* Some subgroups may not have any work to do; if so, quit early. */
    if (!active) return;

    if constexpr (!EnableLSE) {
      // Fast path used by ordinary prefill. This specialization intentionally
      // contains no LSE fragment, LSE broadcast/reorder, output accumulation,
      // or LSE global-memory access.
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < rA_sum.size(); i++) {
        rA_sum(i) = ElementA(1) / rA_sum(i);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < rA.size(); i++) {
        rA(i) *= broadcast<0>(rA_sum, rA, i);
      }

      Tensor cO = make_identity_tensor(O.shape());
      Tensor gO = local_tile(cO, TileShapeO{}, blk_qv);

      TiledCopyO copy_o{O};
      auto thr_copy_o = copy_o.get_slice(thr_id);
      auto tOrO = thr_copy_o.partition_sg_fragment_S(gO);
      auto tOgO = thr_copy_o.partition_D(gO);

      reorder(rA, tOrO);
      copy(copy_o, tOrO, tOgO);
    } else {
      // The row-coordinate mapping below uses the original P*V accumulator
      // subgroup layout directly. The Python prefill configurations currently
      // instantiate ReduceK == 1.
      static_assert(ReduceK{} == _1{},
                    "row-level LSE epilogue currently requires ReduceK == 1");

      auto rA_lse = rA_sum;
      // Keep LSE in the base-2 (log2) domain end-to-end. This drops the *kLn2
      // conversion here and lets the accumulate-merge use exp2/log2 directly,
      // which lower to fewer instructions and temporaries than native exp/log
      // (helps register pressure). rA_lse now stores log2-domain LSE.
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < rA_sum.size(); i++) {
        rA_lse(i) = rA_max(i) + sycl::native::log2(rA_sum(i));   // log2-domain
        rA_sum(i) = ElementA(1) / rA_sum(i);
      }

      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      int lane_id = static_cast<int>(sg.get_local_id()[0]);
      auto thr_mnk =
          group<1,3>(TiledMMAPV{}.get_thr_layout_vmnk())
              .get_flat_coord(assert_uniform(thr_id));
      // ThrLayoutVMNK mode order is (V,M,N,K); group<1,3> collapses M,N into a
      // single mode, giving (V,MN,K). get<0> is therefore the *lane* inside the
      // subgroup, not the subgroup's q-tile index -- using it here made every
      // lane of a subgroup claim a different row block, so LSE rows were both
      // double-written and left untouched (torch::empty garbage), and the
      // accumulate path read back old_lse from the wrong row.
      // get<1> is the MN mode, matching reduce_A()'s a_tile.
      int q_sg = get<1>(thr_mnk);

      // A single helper to reproduce the per-row query index + validity.
      // Cheap integer math, recomputed on demand -> avoids keeping per-row
      // index/validity arrays alive in registers.
      auto row_q = [&](int i) {
        int q_in_sg   = i * cute::intel::sg_size + lane_id;
        int q_in_tile = q_sg * size<0>(SGTileShapeA{}) + q_in_sg;
        return get<0>(blk_qv) * size<0>(TileShapeO{}) + q_in_tile;
      };

      // ---- old-LSE merge: compute alpha (stashed in rA_max) and new log2-LSE.
      if (params.accumulate) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < rA_lse.size(); ++i) {
          int q = row_q(i);
          if (q < size<0>(O)) {
            int lse_idx = q * params.stride_lse_q +
                          head_q * params.stride_lse_h +
                          idx_b * params.stride_lse_b;
            // params.lse is stored in log2 domain (see store below).
            ElementA old_lse     = params.lse[lse_idx];
            ElementA partial_lse = rA_lse(i);
            ElementA merged_max  = sycl::max(old_lse, partial_lse);
            ElementA old_weight  = sycl::native::exp2(old_lse - merged_max);
            ElementA partial_w   = sycl::native::exp2(partial_lse - merged_max);
            ElementA inv_sum     = ElementA(1) / (old_weight + partial_w);
            // Reuse rA_sum(i) as the alpha carrier: it already held 1/sum which
            // has been consumed into rA below only later, so instead we apply
            // the softmax normalization to rA BEFORE overwriting, see ordering.
            rA_lse(i) = merged_max - sycl::native::log2(inv_sum);   // log2-domain
            // stash alpha in rA_max(i) (dead after this point).
            rA_max(i) = old_weight * inv_sum;
          } else {
            rA_max(i) = ElementA(0);
          }
        }
      }

      // Apply softmax normalization to the current-tile output.
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < rA.size(); i++)
        rA(i) *= broadcast<0>(rA_sum, rA, i);

      /* Tile output */
      Tensor cO = make_identity_tensor(O.shape());          // (q,v)
      Tensor gO = local_tile(cO, TileShapeO{}, blk_qv);     // (q,v)

      TiledCopyO copy_o{O};
      auto thr_copy_o = copy_o.get_slice(thr_id);
      auto tOrO = thr_copy_o.partition_sg_fragment_S(gO);
      auto tOgO = thr_copy_o.partition_D(gO);

      if (params.accumulate) {
        // Load old O in the MMA accumulator layout, merge, reorder once.
	// O lives in global memory as ElementO (BF16). We load it in its
        // native BF16 fragment, promote each element to ElementA (FP32) for
        // the LSE-weighted merge, and let the subsequent copy() downconvert
        // the FP32 accumulator back to BF16 on store. This halves the O
        // read-back bandwidth versus an FP32 O tensor.
        TiledLoadO load_o{O};
        auto thr_load_o = load_o.get_slice(thr_id);
        auto tOgOldO = thr_load_o.partition_S(gO);
        auto tOrOldO = thr_load_o.partition_sg_fragment_D(gO);
        copy(load_o, tOgOldO, tOrOldO);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < rA.size(); ++i) {
          ElementA alpha = broadcast<0>(rA_max, rA, i);
	  // Promote the BF16 old-O value to FP32 before merging so the
          // accumulation math stays in full precision; rA remains FP32.
          ElementA old_o = static_cast<ElementA>(tOrOldO(i));
          rA(i) = sycl::fma(alpha, old_o,
                            (ElementA(1) - alpha) * rA(i));
        }
      }
      reorder(rA, tOrO);
      // Store the normalized or accumulated output fragment.
      copy(copy_o, tOrO, tOgO);

      // Store LSE directly from its compact row fragment. The subgroup TV layout
      // assigns each logical row to exactly one physical (lane, local-value)
      // owner, so no v == 0 selection or output-layout reorder is needed.
      if (params.lse) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < rA_lse.size(); ++i) {
          int q = row_q(i);
          if (q < size<0>(O)) {
            int lse_idx = q * params.stride_lse_q +
                          head_q * params.stride_lse_h +
                          idx_b * params.stride_lse_b;
            // Store in log2 domain (consistent with the merge above).
            params.lse[lse_idx] = rA_lse(i);
          }
        }
      }
    }
  }

  // Reduce k-blocks of A and A_sum across WG, if needed.
  // Note that each k block has its own scale factor based on A_max,
  //   so A/A_sum contributions need to be rescaled to match.
  template <typename FragA, typename FragARow>
  CUTLASS_DEVICE
  decltype(auto)
  reduce_A(FragA        & tArA,     // O accumulator:   (q,v)
           FragARow     & tA_max,   // Softmax row-wise max accumulator
           FragARow     & tA_sum,   // Softmax row-wise sum accumulator
           int            thr_id) { // Work-item ID

    using namespace sycl::ext::oneapi::this_work_item;

    if constexpr (ReduceK{} == _1{}) {
      return std::make_tuple(tArA, tA_sum, tA_max, true);
    } else {
      /* Identify A tile ID and k block for this subgroup. */
      auto thr_vak = group<1,3>(TiledMMAPV{}.get_thr_layout_vmnk()).get_flat_coord(assert_uniform(thr_id));
      auto a_tile = get<1>(thr_vak);
      auto k_blk = get<2>(thr_vak);

      /* Set up SLM tensors and partition A tiles among participating subgroups */
      auto shape_A     = append(append(SGTileShapeA{}, ReduceK{}), SGPerWG{}/ReduceK{});
      auto shape_A_row = make_shape(get<0>(SGTileShapeO{}), shape(ReduceSGLayout{}), ReduceK{}, SGPerWG{}/ReduceK{});

      /* Physical layouts, with subtile modes broken out */
      auto sA_layout = group<2,4>(flat_divide(make_ordered_layout(shape_A, Step<_1,_0,_2,_3>{}), SGTileShapeO{}));
      auto sA_row_stride = make_stride(_1{}, make_stride(get<0>(shape_A_row), _0{}),
                                       AlignedSGTileA_Q{}, AlignedSGTileA_Q{} * ReduceK{});
      auto sA_row_layout = make_layout(shape_A_row, sA_row_stride);

      /* Coordinate layouts, with subtile modes broken out */
      auto basis2 = make_basis_like(SGTileShapeO{});
      auto sA_coords = make_layout(append(SGTileShapeO{}, shape(ReduceSGLayout{})),
                                   append(basis2, product_each(zip(SGTileShapeO{}, basis2))));

      auto sA     = make_tensor(make_smem_ptr<ElementA>(&shared.a_data),     sA_layout);      // (q,v,rblk_dst,rblk_src,a_tile)
      auto sA_max = make_tensor(make_smem_ptr<ElementA>(&shared.a_max_data), sA_row_layout);  // (q,rblk_dst,rblk_src,a_tile)
      auto sA_sum = make_tensor(make_smem_ptr<ElementA>(&shared.a_sum_data), sA_row_layout);  // (q,rblk_dst,rblk_src,a_tile)

      /* Write my contributions to SLM. */
      copy_block_r2s(tA_max, sA_max(_,_,k_blk,a_tile));
      barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);
      copy_block_r2s(tA_sum, sA_sum(_,_,k_blk,a_tile));
      copy_block_r2s(tArA, sA(_,_,_,k_blk,a_tile), sA_coords);

      bool active = (k_blk      < size(ReduceSGLayout{}))
                 || (ReduceK{} == size(ReduceSGLayout{}));    // help compiler out

      /* Wait for maxima to be available, signal other data available */
      barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);
      barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);

      ReduceFragA rA;
      ReduceFragARow rA_sum, rA_max, rA_kmax[ReduceK{}];

      if (active) {
        /* Read A_max back from SLM and reduce. */
        CUTLASS_PRAGMA_UNROLL
        for (int kr = 0; kr < ReduceK{}; kr++) {
          copy_block_s2r(sA_max(_,k_blk,kr,a_tile), rA_kmax[kr]);
        }

        rA_max = rA_kmax[0];
        for (int kr = 1; kr < ReduceK{}; kr++)
          cute::transform(rA_max, rA_kmax[kr], rA_max, cute::max_fn{});

        /* Calculate scale factors for aligning per-block maxima. */
        for (int kr = 0; kr < ReduceK{}; kr++) {
          cute::transform(rA_max, rA_kmax[kr], rA_kmax[kr], [](auto gmax, auto kmax) {
            return sycl::native::exp2(kmax - gmax);
          });
        }
      }

      /* Wait for A/A_sum data to be available */
      barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);

      if (active) {
        /* Read A/A_sum back from SLM, align scaling to new maxima, and reduce. */
        clear(rA_sum);

        CUTLASS_PRAGMA_UNROLL
        for (int kr = 0; kr < ReduceK{}; kr++) {
          ReduceFragARow rA_sum_read;
          copy_block_s2r(sA_sum(_,k_blk,kr,a_tile), rA_sum_read);

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < rA_sum_read.size(); i++) {
            rA_sum(i) += rA_sum_read(i) * rA_kmax[kr](i);
          }
        }

        clear(rA);

        CUTLASS_PRAGMA_UNROLL
        for (int kr = 0; kr < ReduceK{}; kr++) {
          ReduceFragA rA_read;
          copy_block_s2r(sA(_,_,k_blk,kr,a_tile), sA_coords(_,_,0), rA_read);

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < rA_read.size(); i++) {
            rA(i) += rA_read(i) * broadcast<0>(rA_kmax[kr], rA, i);
          }
        }
      }
      return std::make_tuple(rA, rA_sum, rA_max, active);
    }
  }

private:
  Params params;
};


} // namespace cutlass::fmha::collective
