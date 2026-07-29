#!/usr/bin/env python3
"""
Ring GEMM with Level-Zero IPC P2P overlap (sycl_tla_ipc_p2p).

Each rank owns A (local, never moves) and a B shard rotated around the ring.
Rotation uses IPC handles + the XPU copy engine instead of torch.distributed,
so the transfer overlaps torch.matmul.

Pull model: rank r opens prev_rank's ring buffers once, then each round issues
    prev_rank.buf[read_slot] -> local.buf[write_slot]
while computing acc += A @ buf[read_slot].

Triple buffering keeps the copy destination two slots away from the buffer the
(asynchronous) matmul is still reading, so no per-round device sync is needed.

Run the transfer self-check first:
    torchrun --nproc-per-node 4 xpu_ring_gemm_ipc.py --check-transfer
Then the benchmark:
    torchrun --nproc-per-node 4 xpu_ring_gemm_ipc.py --dtype float32
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity

import sycl_tla_ipc_p2p as ipc

# buf[0] permanently holds this rank's own B and is never written.
# The remaining NUM_BUFFERS-1 slots are write targets cycled round-robin,
# giving NUM_BUFFERS-2 rounds of slack between a slot being read by the
# (asynchronous) matmul and being recycled as a copy destination.
NUM_BUFFERS = 4


def read_slot_of(step):
    """Slot holding the block valid at round `step`."""
    return 0 if step == 0 else 1 + ((step - 1) % (NUM_BUFFERS - 1))


def write_slot_of(step):
    """Slot the copy issued at round `step` lands in."""
    return 1 + (step % (NUM_BUFFERS - 1))

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark/validate ring GEMM with XPU IPC P2P overlap."
    )
    p.add_argument("--m", type=int, default=4096, help="rows of local A")
    p.add_argument("--k", type=int, default=4096, help="contraction dim")
    p.add_argument("--n", type=int, default=4096, help="cols of B shard")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--loops", type=int, default=10)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=("bfloat16", "float16", "float32"))
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--check-transfer", action="store_true",
                   help="Only validate the IPC rotation itself (no GEMM). "
                        "Run this first when debugging the IPC path.")
    p.add_argument("--backend", type=str, default="xccl",
                   help="bootstrap process group backend (handle exchange, "
                        "barriers, reductions only)")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile-dir", type=str, default="./profiler_out")
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


def gemm_flops(m, k, n):
    return 2 * m * k * n


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


class IpcRing:
    """
    Owns NUM_BUFFERS ring buffers for B and the opened peer pointers of the
    previous rank. Buffers are allocated once and never replaced, because their
    device addresses are baked into the exported IPC handles.
    """

    def __init__(self, b_local, rank, world, queue_ptr):
        self.rank = rank
        self.world = world
        self.queue_ptr = queue_ptr
        self.prev_rank = (rank - 1) % world

        self.buf = [b_local.contiguous()]
        for _ in range(NUM_BUFFERS - 1):
            self.buf.append(torch.empty_like(b_local))
        self.nbytes = self.buf[0].numel() * self.buf[0].element_size()

        # Export every buffer: the pull side cycles through all of them.
        # ipc_get_handle returns (handle_bytes, offset_within_allocation).
        local = [ipc.ipc_get_handle(b, queue_ptr) for b in self.buf]

        gathered = [None] * world
        dist.all_gather_object(gathered, local)

        peer = gathered[self.prev_rank]
        self.peer_offset = [int(off) for _, off in peer]
        self.peer_ptr = [
            ipc.ipc_open_handle(self.buf[0], h, int(off), queue_ptr)
            for h, off in peer
        ]
        # Every rank must have opened its peer before any copy is issued.
        dist.barrier()
        torch.xpu.synchronize()

    def close(self):
        for p, off in zip(self.peer_ptr, self.peer_offset):
            try:
                ipc.ipc_close_handle(self.buf[0], p, off, self.queue_ptr)
            except Exception:
                pass
        self.peer_ptr = []


def ring_rotate(ring, consume):
    """
    One full ring pass. Round s presents the block owned by rank
    (rank - s) % world in ring.buf[read_slot_of(s)] and calls consume(block, s).

    All ranks follow the identical slot schedule, so rank r pulls from
    prev_rank's buf[read_slot_of(s)] -- the slot the peer is reading this
    round -- into its own buf[write_slot_of(s)].

    No per-round dist.barrier(): compute is guaranteed to dominate transfer,
    so ranks cannot drift by more than the NUM_BUFFERS-2 rounds of slack the
    write-slot rotation provides. buf[0] is never a write target, so the next
    pass starts from this rank's own block with no restore copy.
    """
    for step in range(ring.world):
        pending = None
        if step < ring.world - 1:
            pending = ipc.ipc_copy_from_peer_async(
                ring.buf[write_slot_of(step)],
                ring.peer_ptr[read_slot_of(step)],
                ring.nbytes,
                ring.queue_ptr,
            )

        # Enqueued asynchronously; overlaps the copy engine transfer above.
        consume(ring.buf[read_slot_of(step)], step)

        if pending is not None:
            ipc.ipc_wait(pending)

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
    dtype = getattr(torch, args.dtype)
    M, K, N = args.m, args.k, args.n

    # Same seed on all ranks => identical full tensors; each rank slices its
    # own shard so the reference can be recomputed locally.
    torch.manual_seed(args.seed)
    a_all = torch.randn(world, M, K, device=dev, dtype=dtype)
    b_all = torch.randn(world, K, N, device=dev, dtype=dtype)

    a = a_all[rank].contiguous()
    b_local = b_all[rank].contiguous()
    acc = torch.zeros((M, N), device=dev, dtype=torch.float32)

    queue_ptr = current_queue_ptr(dev)
    ring = IpcRing(b_local, rank, world, queue_ptr)

    # ---------------------------------------------------------------- #
    # Transfer self-check: does round s really deliver rank (rank-s)'s B?
    # ---------------------------------------------------------------- #
    if args.check_transfer:
        failures = []

        def check(block, step):
            expect = b_all[(rank - step) % world]
            torch.xpu.synchronize()
            if not torch.equal(block, expect):
                bad = (block != expect).sum().item()
                failures.append((step, bad, block.numel()))

        ring_rotate(ring, check)
        torch.xpu.synchronize()
        if failures:
            for step, bad, total in failures:
                print(f"[rank {rank}] round {step}: FAILED, "
                      f"{bad}/{total} bytes differ", flush=True)
        else:
            print(f"[rank {rank}] IPC rotation: PASSED "
                  f"({world} rounds, bit-exact)", flush=True)
        dist.barrier()
        ring.close()
        dist.barrier()
        dist.destroy_process_group()
        return

    if rank == 0:
        print("XPU ring GEMM with Level-Zero IPC P2P overlap")
        print(f"  World size         : {world}")
        print(f"  A shape (per rank) : [{M}, {K}]")
        print(f"  B shape (per rank) : [{K}, {N}]")
        print(f"  Ring buffers       : {NUM_BUFFERS} "
              f"(1 pinned + {NUM_BUFFERS - 1} rotating)")
        print(f"  Data type          : {dtype}")
        print(f"  Accumulator dtype  : {acc.dtype}")
        print(f"  Bootstrap backend  : {args.backend}")
        print("  Transfer path      : zeCommandListAppendMemoryCopy "
              "(dedicated copy engine)")
        print(f"  Verify             : {not args.skip_verify}")
        print(f"  Warmup / loops     : {args.warmup} / {args.loops}")

    def run_ring():
        acc.zero_()
        ring_rotate(ring, lambda block, step: acc.add_(torch.matmul(a, block)))

    for _ in range(args.warmup):
        run_ring()
    torch.xpu.synchronize()
    dist.barrier()

    prof = None
    if args.profile:
        os.makedirs(args.profile_dir, exist_ok=True)
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
            record_shapes=True, profile_memory=True, with_stack=False,
        )
        prof.__enter__()

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
            args.profile_dir, f"ring_gemm_ipc_trace_rank{rank}.json")
        prof.export_chrome_trace(trace_path)
        try:
            table = prof.key_averages().table(
                sort_by="self_xpu_time_total", row_limit=20)
        except Exception:
            table = prof.key_averages().table(
                sort_by="self_cpu_time_total", row_limit=20)
        print(f"\n[rank {rank}] Profiler key averages:\n{table}", flush=True)
        print(f"[rank {rank}] Chrome trace -> {trace_path}", flush=True)

    elapsed_tensor = torch.tensor([elapsed_seconds], device=dev)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    elapsed_max = elapsed_tensor.item()

    per_rank_flops = gemm_flops(M, K, N) * world      # world GEMMs per rank
    aggregate_flops = per_rank_flops * world
    average_seconds = elapsed_max / args.loops

    if rank == 0:
        print("\nResults")
        print(f"  Output shape       : {list(acc.shape)}")
        print(f"  Per-rank B bytes   : {ring.nbytes / 1024**2:.1f} MiB")
        print(f"  Moved / loop / rank: "
              f"{ring.nbytes * (world - 1) / 1024**2:.1f} MiB")
        print(f"  Average latency    : {average_seconds * 1e3:.3f} ms")
        print(f"  FLOPs/loop (rank)  : {per_rank_flops / 1e12:.6f} TFLOP")
        print(f"  FLOPs/loop (total) : {aggregate_flops / 1e12:.6f} TFLOP")
        print(f"  Per-rank throughput: "
              f"{per_rank_flops / average_seconds / 1e12:.3f} TFLOPs")
        print(f"  Aggregate through. : "
              f"{aggregate_flops / average_seconds / 1e12:.3f} TFLOPs")
        print(f"  Effective P2P BW   : "
              f"{ring.nbytes * (world - 1) / average_seconds / 1e9:.2f} GB/s "
              f"(per rank, overlapped)")

    if not args.skip_verify:
        run_ring()
        torch.xpu.synchronize()

        with torch.no_grad():
            ref = torch.zeros((M, N), device=dev, dtype=torch.float32)
            for r in range(world):
                ref.add_(torch.matmul(a, b_all[r]).float())
        torch.xpu.synchronize()

        # Normalize: bf16 accumulation over `world` rounds makes raw atol
        # meaningless near zero.
        scale = ref.abs().max().clamp_min(1e-6)
        atol, rtol = (1e-3, 1e-3) if dtype == torch.float32 else (2e-2, 2e-2)
        torch.testing.assert_close(acc / scale, ref / scale,
                                   atol=atol, rtol=rtol)
        max_abs_diff = (acc - ref).abs().max().item()
        if rank == 0:
            print("\nValidation")
            print("  Reference          : sum_r torch.matmul(A_local, B_r)")
            print("  Status             : PASSED")
            print(f"  Tolerance          : atol={atol}, rtol={rtol} "
                  f"(normalized by {scale.item():.3f})")
        print(f"  [rank {rank}] max abs diff : {max_abs_diff:.6e}", flush=True)

    dist.barrier()
    ring.close()
    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
