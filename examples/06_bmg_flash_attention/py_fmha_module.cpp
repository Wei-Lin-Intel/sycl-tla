#include <torch/extension.h>
#include <pybind11/stl.h>

#include "xe_fmha_fwd_runner.hpp"
#include "flash_attention_v2/comm/ring_symm.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

class CompatDeviceGuard {
public:
      explicit CompatDeviceGuard(unsigned int device_id)
                  : previous_device_id_(compat::get_current_device_id()) {
            if (previous_device_id_ != device_id) {
                  compat::select_device(device_id);
                  changed_ = true;
            }
      }

      ~CompatDeviceGuard() {
            if (changed_) {
                  compat::select_device(previous_device_id_);
            }
      }

private:
      unsigned int previous_device_id_;
      bool changed_ = false;
};

template <bool Causal, typename ShapeQK, typename ShapePV, typename ShapeOut,
	  typename SubgroupLayoutQK, bool EnableLSE>
int runPrefill(const Options &options) {
  constexpr int PipelineStages = 2;
  // The attention output tensor is always consumed as BF16 in practice.
  // Instantiating the kernel with ElementO = bfloat16_t halves O global
  // traffic: the non-LSE store path writes BF16, and the LSE accumulate
  // path reads back BF16 old-O (promoted to FP32 only inside the merge).
  using Config =
      FMHAConfig<Causal, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK,
		 void, PipelineStages, false,
                 bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t>;

  using Scheduler =
      cutlass::fmha::kernel::XeFHMAIndividualTileScheduler;

  // The Python tensor entry point currently supports fixed-length,
  // non-cached, non-paged prefill. Select the BSHD scheduler directly
  // instead of going through FMHAConfig::run(), which selects the default
  // individual scheduler.
  return Config::template run<
      false, false, false, Scheduler, EnableLSE>(options);
}

template <bool Causal, typename ShapeQK, typename ShapePV, typename ShapeOut,
          typename SubgroupLayoutQK>
int runPrefillDispatch(const Options &options) {
  if (options.external_lse) {
    return runPrefill<Causal, ShapeQK, ShapePV, ShapeOut,
                      SubgroupLayoutQK, true>(options);
  }

  if (options.accumulate_output) {
    return -1;
  }

  return runPrefill<Causal, ShapeQK, ShapePV, ShapeOut,
                    SubgroupLayoutQK, false>(options);
}

int prefillBf16Impl(int batch, int numHeadsQ, int numHeadsKV, int seqLenQO,
                    int seqLenKV, int headSizeQK, int headSizeVO, bool isCausal,
                    int iterations, int warmup, int verify,
                    const void *externalQ = nullptr,
                    const void *externalK = nullptr,
                    const void *externalV = nullptr,
                    void *externalO = nullptr,
                    const int64_t *qStrides = nullptr,
                    const int64_t *kStrides = nullptr,
                    const int64_t *vStrides = nullptr,
                    const int64_t *oStrides = nullptr,
                    float *externalLSE = nullptr,
                    bool accumulateOutput = false,
		    const int64_t *lseStrides = nullptr,
                    bool ringEnabled = false,
                    void *ringPeerK = nullptr,
                    void *ringPeerV = nullptr,
                    bool ringConsume = false,
                    const void *ringRecvK = nullptr,
                    const void *ringRecvV = nullptr) {
  if (headSizeVO != 64 && headSizeVO != 96 && headSizeVO != 128 &&
      headSizeVO != 192) {
    return -1;
  }

  // The current Q*K mainloop has no remainder masking in the head dimension.
  // All Python configurations below use a 32-element Q*K K tile.
  // This permits Q/K head_dim=192 independently of V/O head_dim=128.
  if (headSizeQK <= 0 || headSizeQK % 32 != 0) {
    return -1;
  }

  Options options;
  options.help = false;
  options.error = false;
  options.is_causal = isCausal;
  options.varlen = false;
  options.use_paged_kv = false;
  options.scheduler = "Individual";
  options.batch = batch;
  options.num_heads_q = numHeadsQ;
  options.num_heads_kv = numHeadsKV;
  options.seq_len_qo = seqLenQO;
  options.seq_len_kv = seqLenKV;
  options.seq_len_kv_cache = 0;
  options.page_size = 128;
  options.head_size_qk = headSizeQK;
  options.head_size_vo = headSizeVO;
  options.iterations = iterations;
  options.warmup = warmup;
  options.verify = verify;
  options.print_performance = externalQ == nullptr &&
      externalK == nullptr && externalV == nullptr && externalO == nullptr;
  options.softmax_scale = 1.0f / std::sqrt(static_cast<float>(headSizeQK));
  options.external_q = externalQ;
  options.external_k = externalK;
  options.external_v = externalV;
  options.external_o = externalO;
  options.external_lse = externalLSE;
  options.accumulate_output = accumulateOutput;
  options.ring_enabled   = ringEnabled;
  options.ring_peer_k    = ringPeerK;
  options.ring_peer_v    = ringPeerV;
  options.ring_consume   = ringConsume;
  options.ring_recv_k    = ringRecvK;
  options.ring_recv_v    = ringRecvV;

  if (externalLSE && lseStrides) {
    options.stride_lse_q = static_cast<int>(lseStrides[0]);
    options.stride_lse_h = static_cast<int>(lseStrides[1]);
    options.stride_lse_b = static_cast<int>(lseStrides[2]);
  }

      if (qStrides && kStrides && vStrides && oStrides) {
            options.use_external_strides = true;
            options.stride_q_s = static_cast<int>(qStrides[0]);
            options.stride_q_h = static_cast<int>(qStrides[1]);
            options.stride_q_b = static_cast<int>(qStrides[2]);
            options.stride_k_s = static_cast<int>(kStrides[0]);
            options.stride_k_h = static_cast<int>(kStrides[1]);
            options.stride_k_b = static_cast<int>(kStrides[2]);
            options.stride_v_s = static_cast<int>(vStrides[0]);
            options.stride_v_h = static_cast<int>(vStrides[1]);
            options.stride_v_b = static_cast<int>(vStrides[2]);
            options.stride_o_s = static_cast<int>(oStrides[0]);
            options.stride_o_h = static_cast<int>(oStrides[1]);
            options.stride_o_b = static_cast<int>(oStrides[2]);
      }

  if (headSizeVO == 64) {
    using ShapeQK = Shape<_256, _64, _32>;
    using ShapePV = Shape<_256, _32, _64>;
    using ShapeOut = Shape<_256, _64>;
    using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
    return isCausal
	       ? runPrefillDispatch<true, ShapeQK, ShapePV, ShapeOut,
                                    SubgroupLayoutQK>(options)
               : runPrefillDispatch<false, ShapeQK, ShapePV, ShapeOut,
                                    SubgroupLayoutQK>(options);
  }

  if (headSizeVO == 96) {
    using ShapeQK = Shape<_256, _64, _32>;
    using ShapePV = Shape<_256, _32, _64>;
    using ShapeOut = Shape<_256, _96>;
    using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
    return isCausal
	       ? runPrefillDispatch<true, ShapeQK, ShapePV, ShapeOut,
                                    SubgroupLayoutQK>(options)
               : runPrefillDispatch<false, ShapeQK, ShapePV, ShapeOut,
                                    SubgroupLayoutQK>(options);
  }

  if (headSizeVO == 128) {
    using ShapeQK = Shape<_256, _32, _32>;
    using ShapePV = Shape<_256, _32, _32>;
    using ShapeOut = Shape<_256, _128>;
    using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
    return isCausal
	       ? runPrefillDispatch<true, ShapeQK, ShapePV, ShapeOut,
                                    SubgroupLayoutQK>(options)
               : runPrefillDispatch<false, ShapeQK, ShapePV, ShapeOut,
                                    SubgroupLayoutQK>(options);
  }

  using ShapeQK = Shape<_256, _64, _32>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOut = Shape<_256, _192>;
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;
  return isCausal
	     ? runPrefillDispatch<true, ShapeQK, ShapePV, ShapeOut,
                                  SubgroupLayoutQK>(options)
             : runPrefillDispatch<false, ShapeQK, ShapePV, ShapeOut,
                                  SubgroupLayoutQK>(options);
}

inline bool stride_fits_int64_to_int(int64_t value) {
      return value >= std::numeric_limits<int>::min() &&
                         value <= std::numeric_limits<int>::max();
}

inline bool has_supported_base_stride(const at::Tensor &t) {
      if (t.dim() != 4) {
            return false;
      }
      if (t.stride(3) != 1) {
            return false;
      }
      for (int i = 0; i < 4; ++i) {
            if (t.stride(i) < 0 || !stride_fits_int64_to_int(t.stride(i))) {
                  return false;
            }
      }

      return true;
}

inline bool is_packed_bhsd(const at::Tensor &t) {
      if (!has_supported_base_stride(t)) {
            return false;
      }

      const int64_t d = t.size(3);
      const int64_t s = t.size(2);
      const int64_t h = t.size(1);

      return t.stride(2) == d &&
             t.stride(1) == s * d &&
             t.stride(0) == h * s * d;
}

// t has logical shape [B,H,S,D], but its backing storage is contiguous
// [B,S,H,D].
inline bool is_packed_bshd_view(const at::Tensor &t) {
      if (!has_supported_base_stride(t)) {
            return false;
      }

      const int64_t d = t.size(3);
      const int64_t s = t.size(2);
      const int64_t h = t.size(1);

      return t.stride(1) == d &&
             t.stride(2) == h * d &&
             t.stride(0) == s * h * d;
}

// Logical tensor shape expected by the Python API is [B, H, S, D].
//
// Packed BHSD:
//   stride = [H*S*D, S*D, D, 1]
//
// BSHD-backed BHSD view:
//   physical storage: [B, S, H, D]
//   logical view:     [B, H, S, D]
//   stride = [S*H*D, D, H*D, 1]
//
// The latter is produced, for example, by:
//
//   q_bshd = torch.empty([B, S, H, D], ...)
//   q_view = q_bshd.permute(0, 2, 1, 3)
//
// No contiguous conversion is needed for q_view.
inline bool can_use_direct_stride_path(const at::Tensor &t) {
      return is_packed_bhsd(t) || is_packed_bshd_view(t);
}

at::Tensor prefillBf16Tensor(const at::Tensor &q, const at::Tensor &k,
                                                                                     const at::Tensor &v, bool isCausal = false,
                                                                                     int iterations = 1, int warmup = 0,
                                                                                     int verify = 0) {
      TORCH_CHECK(q.device().type() == c10::DeviceType::XPU,
                                  "q must be an XPU tensor");
      TORCH_CHECK(k.device() == q.device() && v.device() == q.device(),
                                  "k and v must be on the same XPU device as q");
      TORCH_CHECK(q.scalar_type() == at::kBFloat16,
                                  "q must have bfloat16 dtype");
      TORCH_CHECK(k.scalar_type() == at::kBFloat16,
                                  "k must have bfloat16 dtype");
      TORCH_CHECK(v.scalar_type() == at::kBFloat16,
                                  "v must have bfloat16 dtype");
      TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                                          "q, k, v must be rank-4 tensors");
      TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(2) > 0 &&
                        q.size(3) > 0,
                  "q dimensions must be non-zero");

      const bool useBshdOutput =
            is_packed_bshd_view(q) &&
            is_packed_bshd_view(k) &&
            is_packed_bshd_view(v);

      at::Tensor out;
      if (useBshdOutput) {
            out = torch::empty(
                  {q.size(0), q.size(2), q.size(1), v.size(3)},
                  q.options().dtype(at::kBFloat16)).permute({0, 2, 1, 3});
      } else {
            out = torch::empty(
                  {q.size(0), q.size(1), q.size(2), v.size(3)},
                  q.options().dtype(at::kBFloat16));
      }

      TORCH_CHECK(out.device() == q.device(),
                                  "out must be on the same XPU device as q");
      TORCH_CHECK(out.scalar_type() == at::kBFloat16,
                                  "out must have bfloat16 dtype");
      TORCH_CHECK(out.dim() == 4, "out must be a rank-4 tensor");
      TORCH_CHECK(q.device().has_index(), "q must have a concrete XPU device index");
      const auto tensor_device_index = q.device().index();
      TORCH_CHECK(tensor_device_index >= 0, "q must have a non-negative XPU device index");
      CompatDeviceGuard device_guard(static_cast<unsigned int>(tensor_device_index));

      bool useDirectStrides =
            verify == 0 &&
	    can_use_direct_stride_path(q) &&
            can_use_direct_stride_path(k) &&
            can_use_direct_stride_path(v) &&
            can_use_direct_stride_path(out);

      at::Tensor qTensor = useDirectStrides ? q : (q.is_contiguous() ? q : q.contiguous());
      at::Tensor kTensor = useDirectStrides ? k : (k.is_contiguous() ? k : k.contiguous());
      at::Tensor vTensor = useDirectStrides ? v : (v.is_contiguous() ? v : v.contiguous());
      TORCH_CHECK(useDirectStrides || out.is_contiguous(),
                  "fallback output must be contiguous");

      const int64_t batch = qTensor.size(0);
      const int64_t numHeadsQ = qTensor.size(1);
      const int64_t seqLenQO = qTensor.size(2);
      const int64_t headSizeQK = qTensor.size(3);

      const int64_t numHeadsKV = kTensor.size(1);
      const int64_t seqLenKV = kTensor.size(2);

      TORCH_CHECK(kTensor.size(0) == batch && vTensor.size(0) == batch,
              "k/v batch dimension must match q");
      TORCH_CHECK(kTensor.size(3) == headSizeQK,
              "k last dimension must equal q last dimension (head_size_qk)");
      TORCH_CHECK(headSizeQK % 32 == 0,
              "q/k head dimension must be a positive multiple of 32, but got ",
              headSizeQK);
      TORCH_CHECK(vTensor.size(1) == numHeadsKV && vTensor.size(2) == seqLenKV,
              "v must match k in num_heads_kv and seq_len_kv");
      TORCH_CHECK(numHeadsKV > 0 && numHeadsQ % numHeadsKV == 0,
              "num_heads_q must be divisible by num_heads_kv");

      const int64_t headSizeVO = vTensor.size(3);
      TORCH_CHECK(headSizeVO == 64 || headSizeVO == 96 ||
                        headSizeVO == 128 || headSizeVO == 192,
              "v head dimension must be one of 64, 96, 128, or 192, but got ",
              headSizeVO);
      TORCH_CHECK(out.size(0) == batch && out.size(1) == numHeadsQ &&
                    out.size(2) == seqLenQO && out.size(3) == headSizeVO,
		    "out logical shape must be "
                    "[batch, num_heads_q, seq_len_qo, head_size_vo]");

      std::array<int64_t, 3> qStrides{
            qTensor.stride(2), qTensor.stride(1), qTensor.stride(0)};
      std::array<int64_t, 3> kStrides{
            kTensor.stride(2), kTensor.stride(1), kTensor.stride(0)};
      std::array<int64_t, 3> vStrides{
            vTensor.stride(2), vTensor.stride(1), vTensor.stride(0)};
      std::array<int64_t, 3> oStrides{
            out.stride(2), out.stride(1), out.stride(0)};

      const int ret = prefillBf16Impl(
            static_cast<int>(batch), static_cast<int>(numHeadsQ),
            static_cast<int>(numHeadsKV), static_cast<int>(seqLenQO),
            static_cast<int>(seqLenKV), static_cast<int>(headSizeQK),
            static_cast<int>(headSizeVO), isCausal, iterations, warmup, verify,
            qTensor.data_ptr(), kTensor.data_ptr(), vTensor.data_ptr(),
            out.data_ptr(),
            useDirectStrides ? qStrides.data() : nullptr,
            useDirectStrides ? kStrides.data() : nullptr,
            useDirectStrides ? vStrides.data() : nullptr,
            useDirectStrides ? oStrides.data() : nullptr);
      TORCH_CHECK(ret == 0, "prefill_bf16_tensor failed in kernel run");
  return out;
}

// Native BSHD entry point.
//
// Inputs:
//   q: [B,Sq,Hq,Dqk]
//   k: [B,Sk,Hkv,Dqk]
//   v: [B,Sk,Hkv,Dvo]
//
// Output:
//   o: [B,Sq,Hq,Dvo], physically contiguous BSHD.
//
// permute() only creates metadata views. No Q/K/V/O data conversion or
// contiguous copy is performed on the supported direct-stride path.
at::Tensor prefillBf16TensorBSHD(
      const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
      bool isCausal = false, int iterations = 1, int warmup = 0,
      int verify = 0) {
      TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                  "q, k, v must be rank-4 BSHD tensors");
      TORCH_CHECK(q.is_contiguous(),
                  "q must be contiguous in [B,S,H,D] layout");
      TORCH_CHECK(k.is_contiguous(),
                  "k must be contiguous in [B,S,H,D] layout");
      TORCH_CHECK(v.is_contiguous(),
                  "v must be contiguous in [B,S,H,D] layout");

      auto outBhsdView = prefillBf16Tensor(
            q.permute({0, 2, 1, 3}),
            k.permute({0, 2, 1, 3}),
            v.permute({0, 2, 1, 3}),
            isCausal, iterations, warmup, verify);

      auto outBshd = outBhsdView.permute({0, 2, 1, 3});
      TORCH_CHECK(outBshd.is_contiguous(),
                  "internal BSHD output must be contiguous");
      return outBshd;
}

py::object prefillBf16TensorBSHDKVList(
      const at::Tensor &q, const std::vector<at::Tensor> &kList,
      const std::vector<at::Tensor> &vList, bool isCausal = false,
      bool returnLSE = false) {
      TORCH_CHECK(!isCausal,
                  "prefill_bf16_bshd_kv_list currently supports only "
                  "non-causal attention");
      TORCH_CHECK(q.device().type() == c10::DeviceType::XPU,
                  "q must be an XPU tensor");
      TORCH_CHECK(q.scalar_type() == at::kBFloat16,
                  "q must have bfloat16 dtype");
      TORCH_CHECK(q.dim() == 4, "q must be a rank-4 BSHD tensor");
      TORCH_CHECK(q.is_contiguous(),
                  "q must be contiguous in [B,S,H,D] layout");
      TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(2) > 0 &&
                        q.size(3) > 0,
                  "q dimensions must be non-zero");
      TORCH_CHECK(!kList.empty(), "k_list and v_list must be non-empty");
      TORCH_CHECK(kList.size() == vList.size(),
                  "k_list and v_list must have equal lengths");

      auto checkInput = [&](const at::Tensor &tensor, const char *name) {
            TORCH_CHECK(tensor.device() == q.device(), name,
                        " tensors must be on the same XPU device as q");
            TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name,
                        " tensors must have bfloat16 dtype");
            TORCH_CHECK(tensor.dim() == 4, name,
                        " tensors must be rank-4 BSHD tensors");
            TORCH_CHECK(tensor.is_contiguous(), name,
                        " tensors must be contiguous in [B,S,H,D] layout");
            TORCH_CHECK(tensor.size(0) > 0 && tensor.size(1) > 0 &&
                              tensor.size(2) > 0 && tensor.size(3) > 0,
                        name, " tensor dimensions must be non-zero");
      };

      const auto &firstK = kList.front();
      const auto &firstV = vList.front();
      for (std::size_t i = 0; i < kList.size(); ++i) {
            checkInput(kList[i], "k_list");
            checkInput(vList[i], "v_list");
            TORCH_CHECK(kList[i].sizes() == firstK.sizes(),
                        "all k_list tensors must have the same shape");
            TORCH_CHECK(vList[i].sizes() == firstV.sizes(),
                        "all v_list tensors must have the same shape");
      }

      TORCH_CHECK(firstK.size(0) == q.size(0) &&
                        firstV.size(0) == q.size(0),
                  "k/v batch dimension must match q");
      TORCH_CHECK(firstK.size(1) == firstV.size(1) &&
                        firstK.size(2) == firstV.size(2),
                  "v must match k in seq_len_kv and num_heads_kv");
      TORCH_CHECK(firstK.size(3) == q.size(3),
                  "k last dimension must equal q last dimension");
      TORCH_CHECK(q.size(3) % 32 == 0,
                  "q/k head dimension must be a positive multiple of 32");
      TORCH_CHECK(q.size(2) % firstK.size(2) == 0,
                  "num_heads_q must be divisible by num_heads_kv");

      const int64_t headSizeVO = firstV.size(3);
      TORCH_CHECK(headSizeVO == 64 || headSizeVO == 96 ||
                        headSizeVO == 128 || headSizeVO == 192,
                  "v head dimension must be one of 64, 96, 128, or 192");
      for (auto dim : {q.size(0), q.size(1), q.size(2), q.size(3),
                       firstK.size(1), firstK.size(2), headSizeVO}) {
            TORCH_CHECK(dim <= std::numeric_limits<int>::max(),
                        "tensor dimensions must fit in int32");
      }

      // O is produced and consumed as BF16. LSE stays FP32 for numerical
      // stability of the cross-chunk log-sum-exp merge.
      auto out = torch::empty(
            {q.size(0), q.size(1), q.size(2), headSizeVO},
            q.options().dtype(at::kBFloat16));

      // When there is a single K/V chunk AND the caller does not request LSE,
      // the computation is mathematically identical to a plain prefill: there
      // is no cross-chunk accumulation (i == 0 only) and no LSE needs to be
      // returned. In that case we skip the LSE tensor entirely so the kernel
      // dispatch selects the faster non-LSE epilogue path (~3.5% higher
      // TFLOPs on B70), instead of paying for LSE compute + global stores.
      const bool needLSE = returnLSE || (kList.size() > 1);
      at::Tensor lse;
      if (needLSE) {
            lse = torch::empty(
                  {q.size(0), q.size(1), q.size(2)},
                  q.options().dtype(at::kFloat));
      }

      TORCH_CHECK(q.device().has_index(),
                  "q must have a concrete XPU device index");
      const auto deviceIndex = q.device().index();
      TORCH_CHECK(deviceIndex >= 0,
                  "q must have a non-negative XPU device index");
      CompatDeviceGuard deviceGuard(static_cast<unsigned int>(deviceIndex));

      auto qView = q.permute({0, 2, 1, 3});
      auto outView = out.permute({0, 2, 1, 3});
      std::array<int64_t, 3> qStrides{
            qView.stride(2), qView.stride(1), qView.stride(0)};
      std::array<int64_t, 3> oStrides{
            outView.stride(2), outView.stride(1), outView.stride(0)};
      std::array<int64_t, 3> lseStrides{};
      if (needLSE) {
            lseStrides = {lse.stride(1), lse.stride(2), lse.stride(0)};
      }

      for (std::size_t i = 0; i < kList.size(); ++i) {
            auto kView = kList[i].permute({0, 2, 1, 3});
            auto vView = vList[i].permute({0, 2, 1, 3});
            std::array<int64_t, 3> kStrides{
                  kView.stride(2), kView.stride(1), kView.stride(0)};
            std::array<int64_t, 3> vStrides{
                  vView.stride(2), vView.stride(1), vView.stride(0)};

            const int ret = prefillBf16Impl(
                  static_cast<int>(q.size(0)), static_cast<int>(q.size(2)),
                  static_cast<int>(firstK.size(2)),
                  static_cast<int>(q.size(1)),
                  static_cast<int>(firstK.size(1)),
                  static_cast<int>(q.size(3)),
                  static_cast<int>(headSizeVO), false, 1, 0, 0,
                  qView.data_ptr(), kView.data_ptr(), vView.data_ptr(),
                  outView.data_ptr(), qStrides.data(), kStrides.data(),
		  vStrides.data(), oStrides.data(),
                  needLSE ? lse.data_ptr<float>() : nullptr,
                  i != 0, needLSE ? lseStrides.data() : nullptr);
            TORCH_CHECK(ret == 0,
                        "prefill_bf16_bshd_kv_list failed in kernel run");
      }

      if (returnLSE) {
            return py::make_tuple(out, lse);
      }
      return py::cast(out);
}

int prefillBf16Benchmark(int batch = 32, int numHeadsQ = 16, int numHeadsKV = 16,
                         int seqLenQO = 512, int seqLenKV = 512,
                         int headSizeQK = 128, int headSizeVO = 128,
                         bool isCausal = false, int iterations = 100,
                         int warmup = 100, int verify = 1) {
  return prefillBf16Impl(batch, numHeadsQ, numHeadsKV, seqLenQO, seqLenKV,
                         headSizeQK, headSizeVO, isCausal, iterations, warmup,
                         verify);
}

// One ring-attention round (batch == 1, non-causal).
//
// Attends the local Q shard against the current-buffer K/V (packed BSHD),
// accumulates into out/lse via online softmax (round_idx > 0 => accumulate),
// and, when ringEnabled, simultaneously pushes the post-reorder MMA-B
// fragments (tSrK / tArV) to the next rank's receive buffer via a raw
// UniversalCopy (fragment layout, not K/V layout -- see mainloop comments).
//
// When ringConsume is set (round_idx > 0, i.e. this rank's own receive
// buffer was populated by the previous round's peer push), K/V for this
// round are read directly from recvK_ptr/recvV_ptr in fragment layout,
// skipping the global load + reorder entirely. ringRecvK/V must then be
// non-null and k_ptr/v_ptr are ignored for the non-cache path (they may
// still be passed for shape/stride bookkeeping but are not read).
//
// All *_ptr are raw device addresses (int64). Strides are [s, h, b] for
// q/k/v/o and [q, h, b] for lse, matching prefillBf16Tensor's BHSD-view
// convention.
void prefillBf16RingRound(
    int64_t q_ptr, int64_t k_ptr, int64_t v_ptr, int64_t o_ptr,
    int64_t lse_ptr,
    int seqLenQO, int seqLenKV,
    int numHeadsQ, int numHeadsKV,
    int headSizeQK, int headSizeVO,
    int roundIdx, bool ringEnabled,
    int64_t peerK_ptr, int64_t peerV_ptr,
    bool ringConsume, int64_t recvK_ptr, int64_t recvV_ptr,
    std::array<int64_t, 3> qS, std::array<int64_t, 3> kS,
    std::array<int64_t, 3> vS, std::array<int64_t, 3> oS,
    std::array<int64_t, 3> lseS) {
  const int ret = prefillBf16Impl(
      /*batch=*/1, numHeadsQ, numHeadsKV, seqLenQO, seqLenKV,
      headSizeQK, headSizeVO, /*isCausal=*/false,
      /*iterations=*/1, /*warmup=*/0, /*verify=*/0,
      reinterpret_cast<const void *>(q_ptr),
      reinterpret_cast<const void *>(k_ptr),
      reinterpret_cast<const void *>(v_ptr),
      reinterpret_cast<void *>(o_ptr),
      qS.data(), kS.data(), vS.data(), oS.data(),
      lse_ptr ? reinterpret_cast<float *>(lse_ptr) : nullptr,
      /*accumulateOutput=*/roundIdx != 0,
      lse_ptr ? lseS.data() : nullptr,
      /*ringEnabled=*/ringEnabled,
      reinterpret_cast<void *>(peerK_ptr),
      reinterpret_cast<void *>(peerV_ptr),
      /*ringConsume=*/ringConsume,
      recvK_ptr ? reinterpret_cast<const void *>(recvK_ptr) : nullptr,
      recvV_ptr ? reinterpret_cast<const void *>(recvV_ptr) : nullptr);
  TORCH_CHECK(ret == 0, "prefillBf16RingRound failed in kernel run");
}

} // namespace

PYBIND11_MODULE(sycl_tla_fmha, m) {
  m.def("prefill_bf16", &prefillBf16Benchmark,
        "Run BMG flash-attention prefill benchmark path (random inputs)",
        py::arg("batch") = 32,
        py::arg("num_heads_q") = 16,
        py::arg("num_heads_kv") = 16,
        py::arg("seq_len_qo") = 512,
        py::arg("seq_len_kv") = 512,
        py::arg("head_size_qk") = 128,
        py::arg("head_size_vo") = 128,
        py::arg("is_causal") = false,
        py::arg("iterations") = 100,
        py::arg("warmup") = 100,
        py::arg("verify") = 1);

  m.def("prefill_bf16_tensor", &prefillBf16Tensor,
        "Run BMG flash-attention with torch.bfloat16 XPU tensors",
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("is_causal") = false,
        py::arg("iterations") = 1,
        py::arg("warmup") = 0,
        py::arg("verify") = 0);
  m.def("prefill_bf16_bshd", &prefillBf16TensorBSHD,
        "Run BMG flash-attention with contiguous [B,S,H,D] "
        "torch.bfloat16 XPU tensors and return contiguous [B,S,H,D] output",
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("is_causal") = false,
        py::arg("iterations") = 1,
        py::arg("warmup") = 0,
        py::arg("verify") = 0);
  m.def("prefill_bf16_bshd_kv_list", &prefillBf16TensorBSHDKVList,
        "Run fused non-causal BMG flash-attention over lists of contiguous "
        "[B,S,H,D] K/V tensors",
        py::arg("q"), py::arg("k_list"), py::arg("v_list"),
        py::arg("is_causal") = false,
        py::arg("return_lse") = false);

  py::class_<RingSymmMemory>(m, "RingSymmMemory")
      .def(py::init([](int seqKvLocal, int hKv, int dQk, int dVo,
                       int rank, int worldSize) {
             // NOTE: MPI must already be initialized by the caller (mpi4py).
	     // Ring P2P fragment-layout constants for the head_dim=128 case:
             //   ShapeQK=<256,32,32>, SubgroupLayoutQK=<16,1,1>
             //   NumThreadsQK = size(TiledMMAQK) = 16 * 16 = 256
             //   TileK = 32  ->  nd_qk = 128 / 32 = 4
             //   VTiles = 128 / 32 = 4
             //   frag  = RingFragElems = 64
             // These MUST match the mainloop's slot formula and MMA config.
             constexpr int kTileK   = 32;
             constexpr int kNdQk    = 4;
             constexpr int kVTiles  = 4;
             constexpr int kThreads = 256;
             constexpr int kFrag    = 64;
             return std::make_unique<RingSymmMemory>(
                 /*batch=*/1, seqKvLocal, hKv, dQk, dVo,
		 rank, worldSize, compat::get_default_queue(),
                 kTileK, kNdQk, kVTiles, kThreads, kFrag);
           }),
           py::arg("seq_kv_local"), py::arg("h_kv"),
           py::arg("d_qk"), py::arg("d_vo"),
           py::arg("rank"), py::arg("world_size"))
      .def("load_local_kv",
           [](RingSymmMemory &s, int64_t src_k, int64_t src_v) {
             s.load_local_kv(reinterpret_cast<const void *>(src_k),
                             reinterpret_cast<const void *>(src_v));
           })
      .def("local_k", [](RingSymmMemory &s, int b) {
        return reinterpret_cast<int64_t>(s.local_k(b));
      })
      .def("local_v", [](RingSymmMemory &s, int b) {
        return reinterpret_cast<int64_t>(s.local_v(b));
      })
      .def("remote_k", [](RingSymmMemory &s, int peer, int b) {
        return reinterpret_cast<int64_t>(s.remote_k(peer, b));
      })
      .def("remote_v", [](RingSymmMemory &s, int peer, int b) {
        return reinterpret_cast<int64_t>(s.remote_v(peer, b));
      })
      .def("barrier", [](RingSymmMemory &s, int ch) { s.barrier(ch); });

  m.def("prefill_bf16_ring_round", &prefillBf16RingRound,
        "One ring-attention round: accumulate Q@current-KV into out/lse and "
        "P2P-push the KV to the next rank.",
        py::arg("q_ptr"), py::arg("k_ptr"), py::arg("v_ptr"), py::arg("o_ptr"),
        py::arg("lse_ptr"), py::arg("seq_len_qo"), py::arg("seq_len_kv"),
        py::arg("num_heads_q"), py::arg("num_heads_kv"),
        py::arg("head_size_qk"), py::arg("head_size_vo"),
        py::arg("round_idx"), py::arg("ring_enabled"),
        py::arg("peer_k_ptr"), py::arg("peer_v_ptr"),
	py::arg("ring_consume") = false,
        py::arg("recv_k_ptr") = 0,
        py::arg("recv_v_ptr") = 0,
        py::arg("q_strides"), py::arg("k_strides"), py::arg("v_strides"),
        py::arg("o_strides"), py::arg("lse_strides"));
}
