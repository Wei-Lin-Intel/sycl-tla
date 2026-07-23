#!/usr/bin/env python3

import argparse
import time

import torch
from sycl_tla_fmha import prefill_bf16_bshd_kv_list


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark sycl_tla_fmha BF16 fused attention over a list of "
            "K/V tensors on XPU."
        )
    )
    parser.add_argument("--bs", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--q-seq-len",
        type=int,
        default=8192,
        help="Query sequence length",
    )
    parser.add_argument(
        "--kv-seq-len",
        type=int,
        default=8192,
        help="Sequence length of each K/V tensor",
    )
    parser.add_argument(
        "--kv-list-size",
        type=int,
        default=2,
        help="Number of K/V tensor pairs",
    )
    parser.add_argument(
        "--q-nhead",
        type=int,
        default=40,
        help="Number of query attention heads",
    )
    parser.add_argument(
        "--kv-nhead",
        type=int,
        default=40,
        help="Number of K/V attention heads",
    )
    parser.add_argument(
        "--qk-hdim",
        type=int,
        default=128,
        help="Q/K head dimension; must be a multiple of 32",
    )
    parser.add_argument(
        "--v-hdim",
        type=int,
        default=128,
        choices=(64, 96, 128, 192),
        help="V/output head dimension",
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
        "--return-lse",
        action="store_true",
        help="Request both output and log-sum-exp tensors",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare against concatenated K/V attention in float32",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed",
    )
    return parser.parse_args()


def attention_flops(
    bs,
    q_seq_len,
    total_kv_seq_len,
    q_nhead,
    qk_hdim,
    v_hdim,
):
    """
    Attention FLOPs, excluding softmax and other elementwise operations.

    Q @ K^T:
        2 * B * Hq * Sq * Sk * Dqk

    softmax(QK^T) @ V:
        2 * B * Hq * Sq * Sk * Dvo

    The K/V-list API is equivalent to attending over the concatenation of all
    K/V chunks, so Sk is kv_list_size * kv_seq_len.
    """
    return (
        2
        * bs
        * q_nhead
        * q_seq_len
        * total_kv_seq_len
        * (qk_hdim + v_hdim)
    )


def validate_args(args):
    if args.bs <= 0:
        raise ValueError("--bs must be greater than 0")
    if args.q_seq_len <= 0:
        raise ValueError("--q-seq-len must be greater than 0")
    if args.kv_seq_len <= 0:
        raise ValueError("--kv-seq-len must be greater than 0")
    if args.kv_list_size <= 0:
        raise ValueError("--kv-list-size must be greater than 0")
    if args.q_nhead <= 0:
        raise ValueError("--q-nhead must be greater than 0")
    if args.kv_nhead <= 0:
        raise ValueError("--kv-nhead must be greater than 0")
    if args.q_nhead % args.kv_nhead != 0:
        raise ValueError("--q-nhead must be divisible by --kv-nhead")
    if args.qk_hdim <= 0 or args.qk_hdim % 32 != 0:
        raise ValueError("--qk-hdim must be a positive multiple of 32")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.loops <= 0:
        raise ValueError("--loops must be greater than 0")
    score_elements = (
        args.bs
        * args.q_nhead
        * args.q_seq_len
        * args.kv_seq_len
        * args.kv_list_size
    )
    if args.verify and score_elements > 64 * 1024 * 1024:
        raise ValueError(
            "--verify materializes the reference score matrix; use smaller "
            "sequence lengths or fewer heads"
        )


def verify_result(q, k_list, v_list, output, lse):
    head_group = q.size(2) // k_list[0].size(2)
    k = torch.cat(k_list, dim=1).repeat_interleave(head_group, dim=2)
    v = torch.cat(v_list, dim=1).repeat_interleave(head_group, dim=2)
    q_ref = q.float().permute(0, 2, 1, 3)
    k_ref = k.float().permute(0, 2, 1, 3)
    v_ref = v.float().permute(0, 2, 1, 3)
    scores = torch.matmul(q_ref, k_ref.transpose(-2, -1))
    scores *= q.size(3) ** -0.5
    lse_ref = torch.logsumexp(scores, dim=-1).permute(0, 2, 1)
    output_ref = torch.matmul(torch.softmax(scores, dim=-1), v_ref)
    output_ref = output_ref.permute(0, 2, 1, 3)

    torch.testing.assert_close(output, output_ref, rtol=5e-2, atol=5e-2)
    if lse is not None:
        torch.testing.assert_close(lse, lse_ref, rtol=5e-2, atol=5e-2)


def main():
    args = parse_args()
    validate_args(args)

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("No available XPU device was detected")

    torch.manual_seed(args.seed)

    device = torch.device("xpu")
    dtype = torch.bfloat16

    # Native API layout: contiguous [B, S, H, D].
    qshape = (
        args.bs,
        args.q_seq_len,
        args.q_nhead,
        args.qk_hdim,
    )
    kshape = (
        args.bs,
        args.kv_seq_len,
        args.kv_nhead,
        args.qk_hdim,
    )
    vshape = (
        args.bs,
        args.kv_seq_len,
        args.kv_nhead,
        args.v_hdim,
    )

    q = torch.randn(qshape, device=device, dtype=dtype)
    k_list = [
        torch.randn(kshape, device=device, dtype=dtype)
        for _ in range(args.kv_list_size)
    ]
    v_list = [
        torch.randn(vshape, device=device, dtype=dtype)
        for _ in range(args.kv_list_size)
    ]

    def run_attention():
        return prefill_bf16_bshd_kv_list(
            q=q,
            k_list=k_list,
            v_list=v_list,
            is_causal=False,
            return_lse=args.return_lse,
        )

    total_kv_seq_len = args.kv_seq_len * args.kv_list_size
    element_size = torch.tensor([], dtype=dtype).element_size()

    q_bytes = (
        args.bs
        * args.q_seq_len
        * args.q_nhead
        * args.qk_hdim
        * element_size
    )
    kv_bytes = (
        args.kv_list_size
        * args.bs
        * args.kv_seq_len
        * args.kv_nhead
        * (args.qk_hdim + args.v_hdim)
        * element_size
    )

    print("XPU BF16 fused K/V-list attention benchmark")
    print("  Input layout       : [B, S, H, D]")
    print(f"  Q shape            : {list(qshape)}")
    print(f"  K tensor shape     : {list(kshape)}")
    print(f"  V tensor shape     : {list(vshape)}")
    print(f"  K/V list size      : {args.kv_list_size}")
    print(f"  Total K/V length   : {total_kv_seq_len}")
    print(f"  Data type          : {dtype}")
    print("  Causal             : False")
    print(f"  Return LSE         : {args.return_lse}")
    print(f"  Warmup loops       : {args.warmup}")
    print(f"  Benchmark loops    : {args.loops}")

    result = None
    for _ in range(args.warmup):
        result = run_attention()

    torch.xpu.synchronize()

    start = time.perf_counter()
    for _ in range(args.loops):
        result = run_attention()
    torch.xpu.synchronize()
    elapsed_seconds = time.perf_counter() - start

    if args.return_lse:
        output, lse = result
    else:
        output = result
        lse = None

    flops_per_loop = attention_flops(
        bs=args.bs,
        q_seq_len=args.q_seq_len,
        total_kv_seq_len=total_kv_seq_len,
        q_nhead=args.q_nhead,
        qk_hdim=args.qk_hdim,
        v_hdim=args.v_hdim,
    )

    average_seconds = elapsed_seconds / args.loops
    average_ms = average_seconds * 1e3
    tflops = flops_per_loop / average_seconds / 1e12

    print("\nResults")
    print(f"  Output shape       : {list(output.shape)} [B, S, H, D]")
    print(f"  Output dtype       : {output.dtype}")
    if lse is not None:
        print(f"  LSE shape          : {list(lse.shape)} [B, S, H]")
        print(f"  LSE dtype          : {lse.dtype}")
    if args.verify:
        verify_result(q, k_list, v_list, output, lse)
        print("  Correctness        : passed")
    print(f"  Q memory           : {q_bytes / 1024**3:.3f} GiB")
    print(f"  K/V-list memory    : {kv_bytes / 1024**3:.3f} GiB")
    print(f"  Total input memory : {(q_bytes + kv_bytes) / 1024**3:.3f} GiB")
    print(f"  FLOPs per loop     : {flops_per_loop / 1e12:.6f} TFLOP")
    print(f"  Total time         : {elapsed_seconds:.6f} s")
    print(f"  Average latency    : {average_ms:.3f} ms")
    print(f"  Throughput         : {tflops:.3f} TFLOPs")


if __name__ == "__main__":
    main()
