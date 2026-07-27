#!/usr/bin/env python3

import argparse
import time

import torch
import torch.nn.functional as F
from sycl_tla_fmha import prefill_bf16_tensor
from sycl_tla_fmha import prefill_bf16_bshd
#torch.set_printoptions(threshold=float("inf"))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark sycl_tla_fmha BF16 fused attention on XPU."
    )
    parser.add_argument("--bs", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=8192,
        help="Default sequence length used for both Q and K/V when the "
             "dedicated --q-seq-len / --kv-seq-len are not provided.",
    )
    parser.add_argument(
        "--q-seq-len",
        type=int,
        default=None,
        help="Query sequence length (Sq). Defaults to --seq-len.",
    )
    parser.add_argument(
        "--kv-seq-len",
        type=int,
        default=None,
        help="Key/Value sequence length (Sk). Defaults to --seq-len.",
    )
    parser.add_argument(
        "--q-nhead",
        type=int,
        default=40,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--kv-nhead",
        type=int,
        default=40,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--qk-hdim",
        type=int,
        default=128,
        help="Dimension of each attention head",
    )
    parser.add_argument(
        "--v-hdim",
        type=int,
        default=128,
        help="Dimension of each attention head",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup loops",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=10,
        help="Number of benchmark loops",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Benchmark causal attention instead of full attention",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed",
    )
    return parser.parse_args()


def attention_flops(bs, q_seq_len, kv_seq_len, num_head, qk_hdim, v_hdim,
                    causal=False):
    """
    Attention FLOPs, excluding softmax and other elementwise operations.

    Q @ K^T:
        2 * B * H * Sq * Sk * Dqk

    softmax(QK^T) @ V:
        2 * B * H * Sq * Sk * Dvo

    Non-causal total:
        2 * B * H * Sq * Sk * (Dqk + Dvo)

    Causal attention only computes the lower-triangular part of the Sq x Sk
    score matrix. When Sq == Sk this is ~half the work. When Sq != Sk the
    valid fraction is the number of (i, j) pairs with j <= i + (Sk - Sq),
    normalized by Sq * Sk. We compute that fraction exactly.
    """
    full_pairs = q_seq_len * kv_seq_len
    if causal:
        # Number of unmasked (query i, key j) pairs for a causal mask aligned
        # so that query i can attend to keys j <= i + (Sk - Sq).
        offset = kv_seq_len - q_seq_len
        valid_pairs = 0
        # Closed-form count without a Python loop:
        #   for each query i in [0, Sq): visible keys = clamp(i + offset + 1, 0, Sk)
        # Sum over i. Use integer arithmetic via a small vectorized tensor.
        i = torch.arange(q_seq_len, dtype=torch.float64)
        visible = torch.clamp(i + offset + 1, min=0, max=kv_seq_len)
        valid_pairs = float(visible.sum().item())
        frac = valid_pairs / full_pairs if full_pairs > 0 else 0.0
    else:
        frac = 1.0

    flops = 2 * bs * num_head * full_pairs * (qk_hdim + v_hdim)
    return flops * frac


def main():
    args = parse_args()

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("No available XPU device was detected")

    # Resolve Q / KV sequence lengths, falling back to --seq-len.
    q_seq_len = args.q_seq_len if args.q_seq_len is not None else args.seq_len
    kv_seq_len = args.kv_seq_len if args.kv_seq_len is not None else args.seq_len

    if args.bs <= 0:
        raise ValueError("--bs must be greater than 0")
    if q_seq_len <= 0:
        raise ValueError("query sequence length must be greater than 0")
    if kv_seq_len <= 0:
        raise ValueError("key/value sequence length must be greater than 0")
    if args.q_nhead <= 0 or args.kv_nhead <= 0:
        raise ValueError("--num-head must be greater than 0")
    if args.q_nhead % args.kv_nhead != 0:
        raise ValueError("--q-nhead must be divisible by --kv-nhead")
    if args.qk_hdim <= 0 or args.v_hdim <= 0:
        raise ValueError("--head-dim must be greater than 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.loops <= 0:
        raise ValueError("--loops must be greater than 0")

    torch.manual_seed(args.seed)

    device = torch.device("xpu")
    dtype = torch.bfloat16

    # User-facing layout: [B, S, H, D]. Q and K/V may have different S.
    qshape = (
        args.bs,
        q_seq_len,
        args.q_nhead,
        args.qk_hdim,
    )
    kshape = (
        args.bs,
        kv_seq_len,
        args.kv_nhead,
        args.qk_hdim,
    )
    vshape = (
        args.bs,
        kv_seq_len,
        args.kv_nhead,
        args.v_hdim,
    )

    q = torch.randn(qshape, device=device, dtype=dtype)
    k = torch.randn(kshape, device=device, dtype=dtype)
    v = torch.randn(vshape, device=device, dtype=dtype)

    def run_attention():
        return prefill_bf16_bshd(
            q=q,
            k=k,
            v=v,
            is_causal=args.causal,
            iterations=1,
            warmup=0,
            verify=0,
        )

    print("XPU BF16 fused attention benchmark")
    print(f"  Input layout       : [B, S, H, D]")
    print(f"  Q Input shape        : {list(qshape)}")
    print(f"  K Input shape        : {list(kshape)}")
    print(f"  V Input shape        : {list(vshape)}")
    print(f"  Q seq len          : {q_seq_len}")
    print(f"  K/V seq len        : {kv_seq_len}")
    print(f"  Data type          : {dtype}")
    print(f"  Causal             : {args.causal}")
    print(f"  Warmup loops       : {args.warmup}")
    print(f"  Benchmark loops    : {args.loops}")

    # Warmup: trigger initialization/JIT compilation and stabilize the device.
    output = None
    for _ in range(args.warmup):
        output = run_attention()

    torch.xpu.synchronize()

    # Benchmark all loops in one timed region to reduce synchronization and
    # Python timing overhead.
    start = time.perf_counter()

    for _ in range(args.loops):
        output = run_attention()

    torch.xpu.synchronize()
    elapsed_seconds = time.perf_counter() - start

    flops_per_loop = attention_flops(
        bs=args.bs,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        num_head=args.q_nhead,
        qk_hdim=args.qk_hdim,
        v_hdim=args.v_hdim,
        causal=args.causal,
    )

    total_flops = flops_per_loop * args.loops
    average_seconds = elapsed_seconds / args.loops
    average_ms = average_seconds * 1e3
    tflops = flops_per_loop / average_seconds / 1e12

    element_size = torch.tensor([], dtype=dtype).element_size()
    q_bytes = args.bs * q_seq_len * args.q_nhead * args.qk_hdim * element_size
    kv_bytes = (
        args.bs
        * kv_seq_len
        * args.kv_nhead
        * (args.qk_hdim + args.v_hdim)
        * element_size
    )
    qkv_bytes = q_bytes + kv_bytes

    print("\nResults")
    print(f"  Output shape       : {list(output.shape)} [B, S, H, D]")
    print(f"  Q memory           : {q_bytes / 1024**3:.3f} GiB")
    print(f"  K/V memory         : {kv_bytes / 1024**3:.3f} GiB")
    print(f"  Q/K/V memory       : {qkv_bytes / 1024**3:.3f} GiB")
    print(f"  FLOPs per loop     : {flops_per_loop / 1e12:.6f} TFLOP")
    print(f"  Total time         : {elapsed_seconds:.6f} s")
    print(f"  Average latency    : {average_ms:.3f} ms")
    print(f"  Throughput         : {tflops:.3f} TFLOPs")

    out = prefill_bf16_bshd(
            q=q,
            k=k,
            v=v,
            is_causal=args.causal,
            iterations=1,
            warmup=0,
            verify=0).bfloat16()

    ref = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=args.causal).transpose(1, 2)

    torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2)
    print(f"Passed: q={tuple(q.shape)}, k={tuple(k.shape)}, "
          f"v={tuple(v.shape)}, out={tuple(out.shape)}")


if __name__ == "__main__":
    main()
