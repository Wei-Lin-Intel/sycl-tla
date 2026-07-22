#!/usr/bin/env python3

import argparse
import time

import torch
from sycl_tla_fmha import prefill_bf16_tensor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark sycl_tla_fmha BF16 fused attention on XPU."
    )
    parser.add_argument("--bs", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=9450,
        help="Sequence length",
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


def attention_flops(bs, seq_len, num_head, qk_hdim, v_hdim, causal=False):
    """
    Attention FLOPs, excluding softmax and other elementwise operations.

    Q @ K^T:
        2 * B * H * S * S * D

    softmax(QK^T) @ V:
        2 * B * H * S * S * D

    Non-causal total:
        4 * B * H * S^2 * D

    Causal attention only computes approximately half of the attention matrix.
    """
    flops = 2 * bs * num_head * seq_len * seq_len * (qk_hdim + v_hdim)
    if causal:
        flops *= 0.5
    return flops


def main():
    args = parse_args()

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("No available XPU device was detected")

    if args.bs <= 0:
        raise ValueError("--bs must be greater than 0")
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be greater than 0")
    if args.q_nhead <= 0 or args.kv_nhead <= 0:
        raise ValueError("--num-head must be greater than 0")
    if args.qk_hdim <= 0 or args.v_hdim <= 0:
        raise ValueError("--head-dim must be greater than 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.loops <= 0:
        raise ValueError("--loops must be greater than 0")

    torch.manual_seed(args.seed)

    device = torch.device("xpu")
    dtype = torch.bfloat16

    # User-facing layout: [B, S, H, D].
    qshape = (
        args.bs,
        args.seq_len,
        args.q_nhead,
        args.qk_hdim,
    )
    kshape = (
        args.bs,
        args.seq_len,
        args.kv_nhead,
        args.qk_hdim,
    )
    vshape = (
        args.bs,
        args.seq_len,
        args.kv_nhead,
        args.v_hdim,
    )

    q = torch.randn(qshape, device=device, dtype=dtype)
    k = torch.randn(kshape, device=device, dtype=dtype)
    v = torch.randn(vshape, device=device, dtype=dtype)

    # sycl_tla_fmha expects [B, H, S, D].
    q_bhsd = q.transpose(1, 2)#.contiguous()
    k_bhsd = k.transpose(1, 2).contiguous()
    v_bhsd = v.transpose(1, 2).contiguous()

    def run_attention():
        return prefill_bf16_tensor(
            q=q_bhsd,
            k=k_bhsd,
            v=v_bhsd,
            is_causal=args.causal,
            iterations=1,
            warmup=0,
            verify=0,
        )

    print("XPU BF16 fused attention benchmark")
    print(f"  Input layout       : [B, S, H, D]")
    print(f"  Q Input shape        : {list(qshape)}")
    print(f"  Q Kernel layout      : {list(q_bhsd.shape)} [B, H, S, D]")
    print(f"  K Input shape        : {list(kshape)}")
    print(f"  K Kernel layout      : {list(k_bhsd.shape)} [B, H, S, D]")
    print(f"  V Input shape        : {list(vshape)}")
    print(f"  V Kernel layout      : {list(v_bhsd.shape)} [B, H, S, D]")
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
        seq_len=args.seq_len,
        num_head=args.q_nhead,
        qk_hdim=args.qk_hdim,
        v_hdim=args.v_hdim,
        causal=args.causal,
    )

    total_flops = flops_per_loop * args.loops
    average_seconds = elapsed_seconds / args.loops
    average_ms = average_seconds * 1e3
    tflops = flops_per_loop / average_seconds / 1e12

    qkv_bytes = (
        args.bs
        * args.seq_len
        * args.q_nhead
        * (args.qk_hdim * 2 + args.v_hdim)
        * torch.tensor([], dtype=dtype).element_size()
    )

    print("\nResults")
    print(f"  Output shape       : {list(output.shape)} [B, H, S, D]")
    print(f"  Q/K/V memory       : {qkv_bytes / 1024**3:.3f} GiB")
    print(f"  FLOPs per loop     : {flops_per_loop / 1e12:.6f} TFLOP")
    print(f"  Total time         : {elapsed_seconds:.6f} s")
    print(f"  Average latency    : {average_ms:.3f} ms")
    print(f"  Throughput         : {tflops:.3f} TFLOPs")


if __name__ == "__main__":
    main()
