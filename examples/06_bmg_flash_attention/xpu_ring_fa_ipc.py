#!/usr/bin/env python3
"""
Ring attention over prefill_bf16_bshd_kv_round with Level-Zero IPC P2P overlap.

Same math as xpu_ring_fa.py, but K/V rotation goes through sycl_tla_ipc_p2p
(zeCommandListAppendMemoryCopy on a dedicated copy engine) instead of
torch.distributed batch_isend_irecv. The copy engine is fully decoupled from
the SYCL compute queue, so the transfer overlaps the attention kernel without
the collective's host-side serialization.

Pull model: rank r opens prev_rank's K/V ring buffers once, then each round
issues  prev_rank.kbuf[read] -> local.kbuf[write]  (and the same for V) while
launching the attention kernel on the block currently in kbuf[read].

Accumulation uses the online-softmax LSE-merge epilogue: round 0 initializes
out/lse, subsequent rounds merge. out/lse therefore persist across the whole
ring pass and are NOT reset between rounds.

Launch (world >= 2):
    torchrun --nproc-per-node 4 xpu_ring_fa_ipc.py --check-transfer
    torchrun --nproc-per-node 4 xpu_ring_fa_ipc.py --q-seq-len 8192 --loops 10
    torchrun --nproc-per-node 4 xpu_ring_fa_ipc.py --q-seq-len 8192 --profile
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

import sycl_tla_fmha as fa
import sycl_tla_ipc_p2p as ipc

# buf[0] permanently holds this rank's own K/V block and is never written.
# The remaining NUM_BUFFERS-1 slots are copy destinations cycled round-robin,
# giving NUM_BUFFERS-2 rounds of slack between a slot being read by the
# (asynchronous) attention kernel and being recycled as a write target.
NUM_BUFFERS = 4


def read_slot_of(step):
    """Slot holding the K/V block valid at round `step`."""
    return 0 if step == 0 else 1 + ((step - 1) % (NUM_BUFFERS - 1))


def write_slot_of(step):
    """Slot the copy issued at round `step` lands in."""
    return 1 + (step % (NUM_BUFFERS - 1))


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Benchmark and validate sycl_tla_fmha ring attention with "
            "Level-Zero IPC P2P overlap on XPU."
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
    p.add_argument("--check-transfer", action="store_true",
                   help="Only validate the IPC K/V rotation (no attention "
                        "kernel). Run this first when debugging.")
    p.add_argument("--backend", type=str, default="xccl",
                   help="bootstrap process group backend; used only for "
                        "handle exchange, barriers and reductions")
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


def current_queue_ptr(device):
    """Raw address of the sycl::queue backing the current XPU stream."""
    stream = torch.xpu.current_stream(device)
    for attr in ("sycl_queue", "_as_parameter_", "queue"):
        q = getattr(stream, attr, None)
        if q is None:
            continue
        if isinstance(q, int):
            return q
        try:
            return int(q)
        except (TypeError, ValueError):
            pass
    raise RuntimeError(
        "cannot obtain sycl_queue pointer from torch.xpu stream; "
        "this PyTorch build does not expose Stream.sycl_queue"
    )


class IpcKVRing:
    """
    Owns NUM_BUFFERS ring buffers for K and V plus the opened peer pointers of
    the previous rank. Buffers are allocated once and never replaced: their
    device addresses are baked into the exported IPC handles.
    """

    def __init__(self, k_local, v_local, rank, world, queue_ptr):
        self.rank = rank
        self.world = world
        self.queue_ptr = queue_ptr
        self.prev_rank = (rank - 1) % world

        self.kbuf = [k_local.contiguous()]
        self.vbuf = [v_local.contiguous()]
        for _ in range(NUM_BUFFERS - 1):
            self.kbuf.append(torch.empty_like(k_local))
            self.vbuf.append(torch.empty_like(v_local))

        self.k_nbytes = self.kbuf[0].numel() * self.kbuf[0].element_size()
        self.v_nbytes = self.vbuf[0].numel() * self.vbuf[0].element_size()

        # ipc_get_handle returns (handle_bytes, offset_within_allocation).
        # K and V buffers frequently share one caching-allocator slab, so the
        # offset is what keeps their peer pointers distinct.
        local = {
            "k": [ipc.ipc_get_handle(b, queue_ptr) for b in self.kbuf],
            "v": [ipc.ipc_get_handle(b, queue_ptr) for b in self.vbuf],
        }
        gathered = [None] * world
        dist.all_gather_object(gathered, local)
        peer = gathered[self.prev_rank]

        self.k_peer_off = [int(o) for _, o in peer["k"]]
        self.v_peer_off = [int(o) for _, o in peer["v"]]
        self.k_peer_ptr = [
            ipc.ipc_open_handle(self.kbuf[0], h, int(o), queue_ptr)
            for h, o in peer["k"]
        ]
        self.v_peer_ptr = [
            ipc.ipc_open_handle(self.vbuf[0], h, int(o), queue_ptr)
            for h, o in peer["v"]
        ]

        # Every rank must have opened its peer before any copy is issued.
        dist.barrier()
        torch.xpu.synchronize()

    def close(self):
        for p, o in zip(self.k_peer_ptr, self.k_peer_off):
            try:
                ipc.ipc_close_handle(self.kbuf[0], p, o, self.queue_ptr)
            except Exception:
                pass
        for p, o in zip(self.v_peer_ptr, self.v_peer_off):
            try:
                ipc.ipc_close_handle(self.vbuf[0], p, o, self.queue_ptr)
            except Exception:
                pass
        self.k_peer_ptr = []
        self.v_peer_ptr = []


def ring_attention_ipc(ring, consume):
    """
    One full ring pass. Round s presents the K/V block owned by rank
    (rank - s) % world and calls consume(k_block, v_block, s).

    All ranks follow the identical slot schedule, so rank r pulls from
    prev_rank's buf[read_slot_of(s)] -- the slot the peer reads this round --
    into its own buf[write_slot_of(s)].

    No per-round dist.barrier(): the attention kernel dominates the transfer,
    so ranks cannot drift beyond the NUM_BUFFERS-2 rounds of slack that the
    write-slot rotation provides. buf[0] is never a write target, so the next
    pass starts from this rank's own block with no restore copy.
    """
    for step in range(ring.world):
        pending = []
        if step < ring.world - 1:
            rs, ws = read_slot_of(step), write_slot_of(step)
            pending.append(ipc.ipc_copy_from_peer_async(
                ring.kbuf[ws], ring.k_peer_ptr[rs],
                ring.k_nbytes, ring.queue_ptr))
            pending.append(ipc.ipc_copy_from_peer_async(
                ring.vbuf[ws], ring.v_peer_ptr[rs],
                ring.v_nbytes, ring.queue_ptr))

        rs = read_slot_of(step)
        # Enqueued asynchronously; overlaps the copy engine transfers above.
        consume(ring.kbuf[rs], ring.vbuf[rs], step)

        for h in pending:
            ipc.ipc_wait(h)

    # One fence per pass keeps ranks from accumulating drift across loops.
    dist.barrier()


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

    # Persistent output buffers reused across all loops.
    out = torch.empty((1, s_local, Hq, Dvo), device=dev, dtype=torch.bfloat16)
    # lse must be [B, Sq, Hq] to match the LSE-merge epilogue's stride layout.
    lse = torch.empty((1, s_local, Hq), device=dev, dtype=torch.float32)

    queue_ptr = current_queue_ptr(dev)
    ring = IpcKVRing(k_local, v_local, rank, world, queue_ptr)

    # ---------------------------------------------------------------- #
    # Transfer self-check: does round s deliver rank (rank-s)'s K/V shard?
    # ---------------------------------------------------------------- #
    if args.check_transfer:
        failures = []

        def check(kb, vb, step):
            src = (rank - step) % world
            k_ref = k_full[:, src * s_local:(src + 1) * s_local]
            v_ref = v_full[:, src * s_local:(src + 1) * s_local]
            torch.xpu.synchronize()
            if not torch.equal(kb, k_ref):
                failures.append((step, "K", (kb != k_ref).sum().item()))
            if not torch.equal(vb, v_ref):
                failures.append((step, "V", (vb != v_ref).sum().item()))

        ring_attention_ipc(ring, check)
        torch.xpu.synchronize()
        if failures:
            for step, which, bad in failures:
                print(f"[rank {rank}] round {step} {which}: FAILED, "
                      f"{bad} elements differ", flush=True)
        else:
            print(f"[rank {rank}] IPC K/V rotation: PASSED "
                  f"({world} rounds, bit-exact)", flush=True)
        dist.barrier()
        ring.close()
        dist.barrier()
        dist.destroy_process_group()
        return

    def run_ring():
        # round_idx drives the epilogue: 0 initializes out/lse, >0 LSE-merges.
        # out/lse accumulate across the whole pass and must not be reset.
        ring_attention_ipc(
            ring,
            lambda kb, vb, step: fa.prefill_bf16_bshd_kv_round(
                q=q, k=kb, v=vb, out=out, lse=lse, round_idx=step,
            ),
        )

    element_size = torch.tensor([], dtype=dtype).element_size()
    q_bytes = 1 * s_local * Hq * Dqk * element_size
    kv_bytes = ring.k_nbytes + ring.v_nbytes

    if rank == 0:
        print("XPU BF16 ring attention (Level-Zero IPC P2P overlap)")
        print("  Input layout       : [B, S, H, D]")
        print(f"  World size         : {world}")
        print(f"  Global seq len     : {S_global}")
        print(f"  Local shard len    : {s_local}")
        print(f"  Q heads / KV heads : {Hq} / {Hkv}")
        print(f"  QK hdim / V hdim   : {Dqk} / {Dvo}")
        print(f"  Data type          : {dtype}")
        print(f"  Ring buffers       : {NUM_BUFFERS} x2 "
              f"(1 pinned + {NUM_BUFFERS - 1} rotating, K and V)")
        print(f"  Bootstrap backend  : {args.backend}")
        print("  Transfer path      : zeCommandListAppendMemoryCopy "
              "(dedicated copy engine)")
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
            args.profile_dir, f"ring_fa_ipc_trace_rank{rank}.json"
        )
        prof.export_chrome_trace(trace_path)
        try:
            table = prof.key_averages().table(
                sort_by="self_xpu_time_total", row_limit=20
            )
        except Exception:
            table = prof.key_averages().table(
                sort_by="self_cpu_time_total", row_limit=20
            )
        print(f"\n[rank {rank}] Profiler key averages:\n{table}", flush=True)
        print(f"[rank {rank}] Chrome trace written to {trace_path}", flush=True)

    # Throughput is bounded by the slowest rank.
    elapsed_tensor = torch.tensor([elapsed_seconds], device=dev)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    elapsed_max = elapsed_tensor.item()

    per_rank_flops = attention_flops(
        q_seq_len=s_local,
        total_kv_seq_len=S_global,
        q_nhead=Hq,
        qk_hdim=Dqk,
        v_hdim=Dvo,
    )
    aggregate_flops = per_rank_flops * world

    average_seconds = elapsed_max / args.loops
    moved_bytes = kv_bytes * (world - 1)

    if rank == 0:
        print("\nResults")
        print(f"  Output shape       : {list(out.shape)} [B, S, H, D]")
        print(f"  Output dtype       : {out.dtype}")
        print(f"  Per-rank Q memory  : {q_bytes / 1024**3:.3f} GiB")
        print(f"  Per-rank KV memory : {kv_bytes / 1024**3:.3f} GiB")
        print(f"  Moved / loop / rank: {moved_bytes / 1024**2:.1f} MiB")
        print(f"  Total time (max)   : {elapsed_max:.6f} s")
        print(f"  Average latency    : {average_seconds * 1e3:.3f} ms")
        print(f"  FLOPs/loop (rank)  : {per_rank_flops / 1e12:.6f} TFLOP")
        print(f"  FLOPs/loop (total) : {aggregate_flops / 1e12:.6f} TFLOP")
        print(f"  Per-rank throughput: "
              f"{per_rank_flops / average_seconds / 1e12:.3f} TFLOPs")
        print(f"  Aggregate through. : "
              f"{aggregate_flops / average_seconds / 1e12:.3f} TFLOPs")
        print(f"  Effective P2P BW   : "
              f"{moved_bytes / average_seconds / 1e9:.2f} GB/s "
              f"(per rank, overlapped)")

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
    ring.close()
    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
