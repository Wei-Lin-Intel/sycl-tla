#!/usr/bin/env python3
"""
Ring attention with non-blocking P2P overlap over prefill_bf16_bshd_kv_round.

Each rank owns a Q shard and a local K/V shard; the local K/V block is rotated
around the sequence-parallel ring. The next block's rotation is posted (non-
blocking) BEFORE the current-block attention kernel, so the transfer overlaps
compute. Accumulation uses the online-softmax LSE-merge epilogue: round 0
initializes out/lse, subsequent rounds LSE-merge.

Reports overlapped TFLOPs, optionally validates against torch SDPA over the
globally concatenated K/V, and optionally dumps a PyTorch profiler trace.

Launch (world >= 2):
    torchrun --nproc-per-node 2 xpu_ring_fa.py --q-seq-len 8192 --loops 10
    torchrun --nproc-per-node 2 xpu_ring_fa.py --q-seq-len 8192 --profile
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, schedule

import sycl_tla_fmha as fa


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Benchmark and validate sycl_tla_fmha ring attention with P2P "
            "overlap on XPU."
        )
    )
    p.add_argument("--q-seq-len", type=int, default=8192,
                   help="Global query/KV sequence length (split across ranks)")
    p.add_argument("--q-nhead", type=int, default=40,
                   help="Number of query attention heads")
    p.add_argument("--kv-nhead", type=int, default=40,
                   help="Number of K/V attention heads")
    p.add_argument("--qk-hdim", type=int, default=128,
                   help="Q/K head dimension; must be a multiple of 32")
    p.add_argument("--v-hdim", type=int, default=128, choices=(64, 96, 128, 192),
                   help="V/output head dimension")
    p.add_argument("--warmup", type=int, default=2,
                   help="Number of warmup loops")
    p.add_argument("--loops", type=int, default=10,
                   help="Number of benchmark loops")
    p.add_argument("--skip-verify", action="store_true",
                   help="Skip validation against scaled_dot_product_attention")
    p.add_argument("--backend", type=str, default="xccl",
                   help="torch.distributed backend (e.g. xccl / ccl / gloo)")
    p.add_argument("--profile", action="store_true",
                   help="Dump a PyTorch profiler trace over the timed loops")
    p.add_argument("--profile-dir", type=str, default="./profiler_out",
                   help="Directory to write profiler traces into")
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def init_distributed(backend):
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world)
    torch.xpu.set_device(local_rank)
    return rank, world, local_rank


def attention_flops(q_seq_len, total_kv_seq_len, q_nhead, qk_hdim, v_hdim):
    """
    Attention FLOPs (Q@K^T + softmax(QK^T)@V), excluding softmax elementwise.
    Per rank: Sq = local shard length, Sk_total = global KV length.
    """
    return 2 * q_nhead * q_seq_len * total_kv_seq_len * (qk_hdim + v_hdim)


def ring_attention_overlap(q, cur_k, cur_v, recv_k, recv_v, out, lse,
                           rank, world):
    """
    One full ring pass with next-block P2P overlapping current-block compute.
    Buffers are provided by the caller so they can be reused across loops.
    cur_k/cur_v must hold this rank's own K/V block on entry.
    """
    next_rank = (rank + 1) % world
    prev_rank = (rank - 1) % world

    for step in range(world):
        reqs = []
        if step < world - 1:
            p2p_ops = [
                dist.P2POp(dist.isend, cur_k, next_rank, tag=0),
                dist.P2POp(dist.irecv, recv_k, prev_rank, tag=0),
                dist.P2POp(dist.isend, cur_v, next_rank, tag=1),
                dist.P2POp(dist.irecv, recv_v, prev_rank, tag=1),
            ]
            reqs = dist.batch_isend_irecv(p2p_ops)

        fa.prefill_bf16_bshd_kv_round(
            q=q, k=cur_k, v=cur_v, out=out, lse=lse, round_idx=step,
        )

        if reqs:
            for req in reqs:
                req.wait()
            cur_k, recv_k = recv_k, cur_k
            cur_v, recv_v = recv_v, cur_v

    return cur_k, cur_v, recv_k, recv_v


def main():
    args = parse_args()

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("No available XPU device was detected")

    rank, world, local_rank = init_distributed(args.backend)
    if world < 2:
        if rank == 0:
            print(f"[SKIP] need world >= 2, got {world}")
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    dev = torch.device("xpu", local_rank)
    dtype = torch.bfloat16

    S_global = args.q_seq_len
    assert S_global % world == 0, "q-seq-len must be divisible by world size"
    s_local = S_global // world
    Hq, Hkv, Dqk, Dvo = args.q_nhead, args.kv_nhead, args.qk_hdim, args.v_hdim
    assert Hq % Hkv == 0 and Dqk % 32 == 0

    # Same seed on all ranks => identical full tensors; each rank slices its
    # own shard.
    torch.manual_seed(args.seed)
    q_full = torch.randn(1, S_global, Hq, Dqk, device=dev, dtype=dtype)
    k_full = torch.randn(1, S_global, Hkv, Dqk, device=dev, dtype=dtype)
    v_full = torch.randn(1, S_global, Hkv, Dvo, device=dev, dtype=dtype)

    q = q_full[:, rank * s_local:(rank + 1) * s_local].contiguous()
    k_local = k_full[:, rank * s_local:(rank + 1) * s_local].contiguous()
    v_local = v_full[:, rank * s_local:(rank + 1) * s_local].contiguous()

    # Persistent buffers reused across all loops.
    out = torch.empty((1, s_local, Hq, Dvo), device=dev, dtype=torch.bfloat16)
    # lse must be [B, Sq, Hq] to match the LSE-merge epilogue's stride layout.
    lse = torch.empty((1, s_local, Hq), device=dev, dtype=torch.float32)
    recv_k = torch.empty_like(k_local)
    recv_v = torch.empty_like(v_local)

    def run_ring():
        cur_k = k_local.contiguous()
        cur_v = v_local.contiguous()
        ring_attention_overlap(
            q, cur_k, cur_v, recv_k, recv_v, out, lse, rank, world
        )

    element_size = torch.tensor([], dtype=dtype).element_size()
    q_bytes = 1 * s_local * Hq * Dqk * element_size
    kv_bytes = 1 * s_local * Hkv * (Dqk + Dvo) * element_size

    if rank == 0:
        print("XPU BF16 ring attention (P2P overlap) benchmark")
        print("  Input layout       : [B, S, H, D]")
        print(f"  World size         : {world}")
        print(f"  Global seq len     : {S_global}")
        print(f"  Local shard len    : {s_local}")
        print(f"  Q heads / KV heads : {Hq} / {Hkv}")
        print(f"  QK hdim / V hdim   : {Dqk} / {Dvo}")
        print(f"  Data type          : {dtype}")
        print(f"  Backend            : {args.backend}")
        print(f"  Verify             : {not args.skip_verify}")
        print(f"  Profile            : {args.profile}")
        print(f"  Warmup loops       : {args.warmup}")
        print(f"  Benchmark loops    : {args.loops}")

    # Warmup.
    for _ in range(args.warmup):
        run_ring()
    torch.xpu.synchronize()
    dist.barrier()

    prof = None
    if args.profile:
        os.makedirs(args.profile_dir, exist_ok=True)
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        prof.__enter__()

    # Timed loops.
    start = time.perf_counter()
    for _ in range(args.loops):
        run_ring()
        if prof is not None:
            prof.step()
    torch.xpu.synchronize()
    elapsed_seconds = time.perf_counter() - start

    if prof is not None:
        prof.__exit__(None, None, None)
        trace_path = os.path.join(
            args.profile_dir, f"ring_fa_trace_rank{rank}.json"
        )
        prof.export_chrome_trace(trace_path)
        # Per-rank key-averages summary sorted by self XPU time.
        try:
            table = prof.key_averages().table(
                sort_by="self_xpu_time_total", row_limit=20
            )
        except Exception:
            # Fallback if this PyTorch build keys the sort differently.
            table = prof.key_averages().table(
                sort_by="self_cpu_time_total", row_limit=20
            )
        print(f"\n[rank {rank}] Profiler key averages:\n{table}", flush=True)
        print(f"[rank {rank}] Chrome trace written to {trace_path}", flush=True)

    # Throughput is bounded by the slowest rank.
    elapsed_tensor = torch.tensor([elapsed_seconds], device=dev)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    elapsed_max = elapsed_tensor.item()

    flops_per_loop = attention_flops(
        q_seq_len=s_local,
        total_kv_seq_len=S_global,
        q_nhead=Hq,
        qk_hdim=Dqk,
        v_hdim=Dvo,
    )
    total_flops_per_loop = flops_per_loop * world

    average_seconds = elapsed_max / args.loops
    average_ms = average_seconds * 1e3
    tflops = total_flops_per_loop / average_seconds / 1e12

    if rank == 0:
        print("\nResults")
        print(f"  Output shape       : {list(out.shape)} [B, S, H, D]")
        print(f"  Output dtype       : {out.dtype}")
        print(f"  Per-rank Q memory  : {q_bytes / 1024**3:.3f} GiB")
        print(f"  Per-rank KV memory : {kv_bytes / 1024**3:.3f} GiB")
        print(f"  FLOPs per loop     : {total_flops_per_loop / 1e12:.6f} TFLOP")
        print(f"  Total time (max)   : {elapsed_max:.6f} s")
        print(f"  Average latency    : {average_ms:.3f} ms")
        print(f"  Throughput         : {tflops:.3f} TFLOPs")

    if not args.skip_verify:
        run_ring()
        torch.xpu.synchronize()

        with torch.no_grad():
            qb = q.transpose(1, 2)
            kb = k_full.transpose(1, 2)
            vb = v_full.transpose(1, 2)
            if qb.size(1) != kb.size(1):
                g = qb.size(1) // kb.size(1)
                kb = kb.repeat_interleave(g, dim=1)
                vb = vb.repeat_interleave(g, dim=1)
            ref = (
                F.scaled_dot_product_attention(
                    query=qb, key=kb, value=vb,
                    attn_mask=None, dropout_p=0.0, is_causal=False,
                )
                .transpose(1, 2)
                .contiguous()
            )
        torch.xpu.synchronize()

        torch.testing.assert_close(
            out.float(), ref.float(), atol=5e-2, rtol=5e-2
        )
        max_abs_diff = (out.float() - ref.float()).abs().max().item()
        if rank == 0:
            print("\nValidation")
            print("  Reference          : F.scaled_dot_product_attention")
            print("  Status             : PASSED")
            print("  Tolerance          : atol=5e-2, rtol=5e-2")
        print(f"  [rank {rank}] max abs diff : {max_abs_diff:.6e}", flush=True)

    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
