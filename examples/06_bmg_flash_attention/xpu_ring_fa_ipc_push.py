#!/usr/bin/env python3
"""
Ring attention over prefill_bf16_bshd_kv_round with Level-Zero IPC P2P overlap.

Same math as xpu_ring_fa.py, but K/V rotation goes through sycl_tla_ipc_p2p
(zeCommandListAppendMemoryCopy on a dedicated copy engine) instead of
torch.distributed batch_isend_irecv. The copy engine is fully decoupled from
the SYCL compute queue, so the transfer overlaps the attention kernel without
the collective's host-side serialization.

Push model: rank r opens next_rank's K/V ring buffers once, then each round
issues  local.kbuf[read] -> next_rank.kbuf[write]  (and the same for V) while
launching the attention kernel on the block currently in kbuf[read].

The transfer is a CE *write* into the peer, never a CE read from it. On XPU a
cross-UPI CE read costs >2x a same-socket one; posted CE writes do not.

Completion is symmetric to the pull version: each rank waits on the event of
the copy *it issued*, then a barrier establishes that every rank has finished
sending -- which, since all ranks run the identical schedule, is exactly the
condition that every rank has finished receiving. No flags, no credits, no
extra buffers: the memory layout is unchanged from the pull version.

Accumulation uses the online-softmax LSE-merge epilogue: round 0 initializes
out/lse, subsequent rounds merge. out/lse therefore persist across the whole
ring pass and are NOT reset between rounds.

Launch (world >= 2):
    torchrun --nproc-per-node 4 xpu_ring_fa_ipc.py --check-transfer
    torchrun --nproc-per-node 4 xpu_ring_fa_ipc.py --q-seq-len 8192 --loops 10
    torchrun --nproc-per-node 4 xpu_ring_fa_ipc.py --q-seq-len 8192 --profile
"""

import argparse
import atexit
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

import sycl_tla_fmha as fa
import sycl_tla_ipc_p2p as ipc

# buf[0] permanently holds this rank's own K/V block and is never written.
# (asynchronous) attention kernel and being recycled as a write target.
# The remaining NUM_BUFFERS-1 slots are copy destinations cycled round-robin.
# Slot numbering is identical on every rank, so write_slot_of(step) names the
# same slot in the peer's arena as it does locally.
NUM_BUFFERS = 3

SLOT_ALIGN = 4096

def _align_up(v, a=SLOT_ALIGN):
    return (v + a - 1) // a * a

def read_slot_of(step):
    return (step - 1) % NUM_BUFFERS


def write_slot_of(step):
    return step % NUM_BUFFERS


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
    所有 ring buffer 都住在一块 IpcArena 里：进程生命周期内只导出 1 个 IPC handle、
    只 open 1 次对端。这样上层每层新建的 k/v tensor 只需要 stage() 进 slot 0，
    handle 永远不会因为 caching allocator 回收/复用 slab 而变野指针。
    """

    def __init__(self, k_local, v_local, rank, world, queue_ptr):
        self.rank = rank
        self.world = world
        self.queue_ptr = queue_ptr
        self.dev_index = k_local.device.index
        # Push: we write into next_rank and are written into by prev_rank.
        self.next_rank = (rank + 1) % world

        self.k_shape = list(k_local.shape)
        self.v_shape = list(v_local.shape)
        self.dtype = k_local.dtype
        self.k_nbytes = k_local.numel() * k_local.element_size()
        self.v_nbytes = v_local.numel() * v_local.element_size()

        # arena 布局: slot0[K][V] slot1[K][V] ... 每段 4KB 对齐。
        self.k_off_in_slot = 0
        self.v_off_in_slot = _align_up(self.k_nbytes)
        self.slot_stride = _align_up(self.v_off_in_slot + self.v_nbytes)
        total = self.slot_stride * NUM_BUFFERS

        self.arena = ipc.make_arena(self.dev_index, total, queue_ptr)

        self.k_off = [i * self.slot_stride + self.k_off_in_slot
                      for i in range(NUM_BUFFERS)]
        self.v_off = [i * self.slot_stride + self.v_off_in_slot
                      for i in range(NUM_BUFFERS)]
        self.kbuf = [self.arena.view(o, self.k_shape, self.dtype)
                     for o in self.k_off]
        self.vbuf = [self.arena.view(o, self.v_shape, self.dtype)
                     for o in self.v_off]


        # 只交换一次 arena base handle；各 slot 的 offset 所有 rank 一致。
        self.local_handle = self.arena.export_handle()
        gathered = [None] * world
        dist.all_gather_object(gathered, self.local_handle)
        self.peer_handle = gathered[self.next_rank]

        peer_base = ipc.open_peer(self.dev_index, self.peer_handle, queue_ptr)
        self.k_peer_ptr = [peer_base + o for o in self.k_off]
        self.v_peer_ptr = [peer_base + o for o in self.v_off]

        self._closed = False
        atexit.register(self.close)

        # 任何 copy 发出前，所有 rank 必须已经 open 完对端。
        dist.barrier()
        torch.xpu.synchronize()

    def fence(self, device):
        """
        等产出 K/V 的 kernel 退休。

        push 模型下不再需要把 K/V 搬进 arena：CE 的 source 只是一个本地指针，
        不导出 IPC handle，因此 caching-allocator slab 被回收/复用的野指针问题
        根本不存在 —— 那个坑只在 destination 侧，而 destination 永远是对端的槽。

        但同步点省不掉：上层的 k/v 由 compute queue 上的 kernel 产出，而 CE 是
        另一条引擎，看不到这个依赖。不等它就 push 会读到半成品。
        """
        torch.xpu.current_stream(device).synchronize()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            torch.xpu.synchronize()
            ipc.close_peer(self.dev_index, self.peer_handle, self.queue_ptr)
        except Exception:
            pass
        self.k_peer_ptr = []
        self.v_peer_ptr = []
        # arena view 必须先失效，arena 本身由 ipc.shutdown() 释放。
        self.kbuf = []
        self.vbuf = []


def ring_attention_ipc(ring, consume, kv_new=None):
    """
    One full ring pass. Round s presents the K/V block owned by rank
    (rank - s) % world and calls consume(k_block, v_block, s).

    kv_new: optional (k, v) freshly produced by the caller (a transformer layer
    in the real workload). Staged into slot 0 before the ring starts, which is
    what upstream code will actually do every layer. Pass None to reuse whatever
    slot 0 already holds.
    All ranks follow the identical slot schedule, so rank r pushes its own
    buf[read_slot_of(s)] -- the block it is reading this round -- into
    next_rank's buf[write_slot_of(s)].

    Each rank waits on the events of the copies *it* issued, then hits a
    barrier. Because the schedule is identical everywhere, "everyone finished
    sending" is equivalent to "everyone finished receiving", so no flag or
    credit protocol is needed. The barrier runs after the attention kernel has
    been enqueued, so it serializes the host, not the GPU: the kernel and the
    copy engine keep overlapping across it.

    buf[0] is never a write target, so the next pass starts from this rank's
    own block with no restore copy.
    """

    if kv_new is not None:
        # 安全性依赖 pass 尾部的 dist.barrier(): 它保证上一轮所有 rank 都已读完
        # 本 rank 的 slot 0，这里覆写才不会打断对端还在进行的拉取。
        # 只 fence，不再 D2D。round 0 直接以调用方的 tensor 为 CE source。
        ring.fence(kv_new[0].device)
        dist.barrier()
    k_src0, v_src0 = kv_new

    for step in range(ring.world):
        pending = []
        if step < ring.world - 1:
            rs, ws = read_slot_of(step), write_slot_of(step)
            base = ring.arena.base_ptr()
            # round 0 的 source 是调用方自己的 tensor（不在 arena 里）；
            # 之后转发的是上一轮收到的那个 arena 槽。
            if step == 0:
                src_k, src_v = k_src0.data_ptr(), v_src0.data_ptr()
            else:
                src_k, src_v = base + ring.k_off[rs], base + ring.v_off[rs]
            pending.append(ipc.copy_async(
                ring.dev_index, ring.k_peer_ptr[ws], src_k,
                ring.k_nbytes, ring.queue_ptr))
            pending.append(ipc.copy_async(
                ring.dev_index, ring.v_peer_ptr[ws], src_v,
                ring.v_nbytes, ring.queue_ptr))

        rs = read_slot_of(step)
        # Enqueued asynchronously; overlaps the copy engine transfers above.
        # step 0 读调用方的 tensor，之后读 arena 槽。
        if step == 0:
            consume(k_src0, v_src0, step)
        else:
            consume(ring.kbuf[rs], ring.vbuf[rs], step)

        for h in pending:
            h.wait()

        # Every rank has now finished sending, hence every rank has finished
        # receiving. This is what replaces the pull version's implicit
        # completion guarantee -- and it is the only synchronization the push
        # model adds.
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

        ring_attention_ipc(ring, check, kv_new=(k_local, v_local))
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
        ipc.shutdown()
        dist.destroy_process_group()
        return

    def run_ring():
        # round_idx drives the epilogue: 0 initializes out/lse, >0 LSE-merges.
        # out/lse accumulate across the whole pass and must not be reset.
        # kv_new 模拟真实场景：每层 transformer 产出新的 k/v tensor，先 stage 进
        # arena 再进 ring。这次 D2D copy 是实际部署时无法回避的开销。
        ring_attention_ipc(
            ring,
            lambda kb, vb, step: fa.prefill_bf16_bshd_kv_round(
                q=q, k=kb, v=vb, out=out, lse=lse, round_idx=step,
            ),
            kv_new=(k_local, v_local),
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
        print("  Transfer path      : CE write to peer "
              "(zeCommandListAppendMemoryCopy, dedicated copy engine)")
        print("  Per-loop staging   : enabled (new K/V copied into arena, "
              "matches per-layer model behavior)")
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
    # stage() 每 pass 一次本地 D2D（写 + 读），计入有效带宽会更贴近真实。
    staged_bytes = kv_bytes

    if rank == 0:
        print("\nResults")
        print(f"  Output shape       : {list(out.shape)} [B, S, H, D]")
        print(f"  Output dtype       : {out.dtype}")
        print(f"  Per-rank Q memory  : {q_bytes / 1024**3:.3f} GiB")
        print(f"  Per-rank KV memory : {kv_bytes / 1024**3:.3f} GiB")
        print(f"  Moved / loop / rank: {moved_bytes / 1024**2:.1f} MiB")
        print(f"  Staged / loop /rank: {staged_bytes / 1024**2:.1f} MiB "
              f"(local D2D into arena slot 0)")
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
    # shutdown 必须早于 destroy_process_group，且早于解释器卸载 extension。
    ipc.shutdown()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
