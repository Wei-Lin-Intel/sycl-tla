#include <torch/extension.h>

#include "xe_fmha_fwd_runner.hpp"
#include "streaming_ring_attention.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

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
          typename SubgroupLayoutQK>
int runPrefill(const Options &options) {
  constexpr int PipelineStages = 2;
  return FMHAConfig<Causal, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void,
                    PipelineStages, false, bfloat16_t, bfloat16_t,
                    bfloat16_t>::run(options);
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
                    const int64_t *oStrides = nullptr) {
  if (headSizeVO != 64 && headSizeVO != 96 && headSizeVO != 128 &&
      headSizeVO != 192) {
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
               ? runPrefill<true, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK>(
                     options)
               : runPrefill<false, ShapeQK, ShapePV, ShapeOut,
                            SubgroupLayoutQK>(options);
  }

  if (headSizeVO == 96) {
    using ShapeQK = Shape<_256, _64, _32>;
    using ShapePV = Shape<_256, _32, _64>;
    using ShapeOut = Shape<_256, _96>;
    using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
    return isCausal
               ? runPrefill<true, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK>(
                     options)
               : runPrefill<false, ShapeQK, ShapePV, ShapeOut,
                            SubgroupLayoutQK>(options);
  }

  if (headSizeVO == 128) {
    using ShapeQK = Shape<_256, _32, _32>;
    using ShapePV = Shape<_256, _32, _32>;
    using ShapeOut = Shape<_256, _128>;
    using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
    return isCausal
               ? runPrefill<true, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK>(
                     options)
               : runPrefill<false, ShapeQK, ShapePV, ShapeOut,
                            SubgroupLayoutQK>(options);
  }

  using ShapeQK = Shape<_256, _64, _32>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOut = Shape<_256, _192>;
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;
  return isCausal
             ? runPrefill<true, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK>(
                   options)
             : runPrefill<false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK>(
                   options);
}

inline bool stride_fits_int64_to_int(int64_t value) {
      return value >= std::numeric_limits<int>::min() &&
                         value <= std::numeric_limits<int>::max();
}

inline bool can_use_direct_stride_path(const at::Tensor &t) {
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

      // The FMHA kernels are tuned for packed BHSD memory access. Allowing arbitrary
      // strided layouts can remove explicit copies but often hurts kernel throughput.
      // Enable direct stride path only when layout is packed-compatible.
      const int64_t d = t.size(3);
      const int64_t s = t.size(2);
      const int64_t h = t.size(1);
      if (t.stride(2) != d) {
            return false;
      }
      if (t.stride(1) != s * d) {
            return false;
      }
      if (t.stride(0) != h * s * d) {
            return false;
      }

      return true;
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

      auto out = torch::empty({q.size(0), q.size(1), q.size(2), v.size(3)},
                                                                              q.options().dtype(at::kFloat));
      TORCH_CHECK(out.device() == q.device(),
                                  "out must be on the same XPU device as q");
      TORCH_CHECK(out.scalar_type() == at::kFloat,
                                  "out must have float32 dtype");
      TORCH_CHECK(out.dim() == 4, "out must be a rank-4 tensor");
      TORCH_CHECK(q.device().has_index(), "q must have a concrete XPU device index");
      const auto tensor_device_index = q.device().index();
      TORCH_CHECK(tensor_device_index >= 0, "q must have a non-negative XPU device index");
      CompatDeviceGuard device_guard(static_cast<unsigned int>(tensor_device_index));

      bool useDirectStrides =
            verify == 0 && can_use_direct_stride_path(q) &&
            can_use_direct_stride_path(k) && can_use_direct_stride_path(v) &&
            can_use_direct_stride_path(out);

      at::Tensor qTensor = useDirectStrides ? q : (q.is_contiguous() ? q : q.contiguous());
      at::Tensor kTensor = useDirectStrides ? k : (k.is_contiguous() ? k : k.contiguous());
      at::Tensor vTensor = useDirectStrides ? v : (v.is_contiguous() ? v : v.contiguous());
      TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

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
      TORCH_CHECK(vTensor.size(1) == numHeadsKV && vTensor.size(2) == seqLenKV,
              "v must match k in num_heads_kv and seq_len_kv");

      const int64_t headSizeVO = vTensor.size(3);
      TORCH_CHECK(out.size(0) == batch && out.size(1) == numHeadsQ &&
                    out.size(2) == seqLenQO && out.size(3) == headSizeVO,
                    "out shape must be [batch, num_heads_q, seq_len_qo, head_size_vo]");

      std::array<int64_t, 3> qStrides{qTensor.stride(2), qTensor.stride(1), qTensor.stride(0)};
      std::array<int64_t, 3> kStrides{kTensor.stride(2), kTensor.stride(1), kTensor.stride(0)};
      std::array<int64_t, 3> vStrides{vTensor.stride(2), vTensor.stride(1), vTensor.stride(0)};
      std::array<int64_t, 3> oStrides{out.stride(2), out.stride(1), out.stride(0)};

      const int ret = prefillBf16Impl(
            static_cast<int>(batch), static_cast<int>(numHeadsQ),
            static_cast<int>(numHeadsKV), static_cast<int>(seqLenQO),
            static_cast<int>(seqLenKV), static_cast<int>(headSizeQK),
            static_cast<int>(headSizeVO), isCausal, iterations, warmup, verify,
            qTensor.data_ptr(), kTensor.data_ptr(), vTensor.data_ptr(),
            out.data_ptr<float>(),
            useDirectStrides ? qStrides.data() : nullptr,
            useDirectStrides ? kStrides.data() : nullptr,
            useDirectStrides ? vStrides.data() : nullptr,
            useDirectStrides ? oStrides.data() : nullptr);
      TORCH_CHECK(ret == 0, "prefill_bf16_tensor failed in kernel run");
  return out;
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

  m.def("streaming_ring_bf16", &streaming_ring_bf16,
        "Run non-causal BF16 streaming ring attention on XPU",
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("k_workspace"), py::arg("v_workspace"),
        py::arg("signal_pad"), py::arg("peer_k_workspace_ptrs"),
        py::arg("peer_v_workspace_ptrs"), py::arg("peer_signal_ptrs"),
        py::arg("rank"), py::arg("world_size"), py::arg("iteration"),
        py::arg("is_causal") = false, py::arg("work_groups") = 8);

  m.def("streaming_ring_bf16_lse", &streaming_ring_bf16_lse,
        "Run streaming ring attention and return (output, natural-log LSE)",
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("k_workspace"), py::arg("v_workspace"),
        py::arg("signal_pad"), py::arg("peer_k_workspace_ptrs"),
        py::arg("peer_v_workspace_ptrs"), py::arg("peer_signal_ptrs"),
        py::arg("rank"), py::arg("world_size"), py::arg("iteration"),
        py::arg("is_causal") = false, py::arg("work_groups") = 8);
}
