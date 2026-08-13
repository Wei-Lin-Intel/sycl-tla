#!/usr/bin/env python3
"""
Schedule-driven ring attention with Level-Zero IPC P2P, PUSH model.

Same forward logic as xpu_ring_fa_ipc_push.py -- no staging D2D, round 0
consumes and sends the caller's own K/V tensors, every round waits on the events
of the copies it issued and then barriers -- but the ring topology is no longer
structural (rank+1 / rank+GROUP_SIZE). It is driven by an explicit HOLD table.

Why a table
-----------
On a two-switch box (ranks 0-3 behind one PCIe switch, 4-7 behind the other) a
cross-switch hop costs >2x an intra-switch one. A naive flat ring pays 2 cross
hops on EVERY round (3->4 and 7->0). The classic two-level ring pays only
n_groups-1 cross transfers per rank but serializes a whole block onto the
copy engine at each outer-round boundary.

This version instead takes an arbitrary hand-designed permutation schedule and
executes it verbatim. Each round is still a single permutation (exactly one
send and one receive per rank), so the push model's "everyone finished sending
== everyone finished receiving" invariant, and therefore the single per-round
barrier, are unchanged.

The schedule below is HOLD[t][r] = the GLOBAL block index that rank r holds at
round t. Round 0 is the identity (every rank holds its own block). The
transfer edges are derived from the table at runtime: at round t, rank r pushes
the block it currently holds to the unique rank that must hold that block at
round t+1.

Cross-group load is NOT uniform under this schedule: participation counts are
rank0:2 rank1:4 rank2:3 rank3:5 rank4:3 rank5:4 rank6:2 rank7:5 (28 total =
14 cross edges x 2 endpoints). Ranks 3 and 7 are the cross-UPI hot spots. The
counts are printed at startup so the imbalance is visible.

Push model
----------
Every transfer is a copy-engine *write* into the peer, never a read from it.
On XPU a cross-UPI CE read costs >2x a same-socket one; posted CE writes do not.

Because the CE *source* is never exported as an IPC handle, it may be an
ordinary caching-allocator tensor. That removes the pull version's mandatory
stage() D2D: round 0 pushes the layer's freshly produced K/V straight out of the
tensor the caller handed us, and the local kernel of round 0 reads that same
tensor. Only destinations must live in the arena, and they always do.

fence() is still required: the caller's K/V is produced by a kernel on the
compute queue, and the CE is a different engine that cannot see that dependency.

Peers
-----
Under a structural ring each rank has one (or two) fixed downstream peers. Under
a table-driven schedule the destination changes every round, so every rank opens
EVERY other rank's arena once at construction time. That is world-1 = 7 opens on
an 8-rank box -- still one-time, still one IPC handle exported per process.

Slot layout
-----------
Slot numbering is identical on every rank, so the destination slot chosen by the
sender names the same slot in the peer's arena as it does locally. Because each
rank receives exactly ONE block per round, the destination slot can be a pure
function of the round number -- no handshake, no per-peer bookkeeping:

    the block delivered at round t lives in slot  t % NUM_BUFFERS
    the sender at round t therefore writes slot   (t + 1) % NUM_BUFFERS

Round 0 is the exception on the read side: the block is the caller's own tensor,
never a slot. With NUM_BUFFERS = 4 a slot is not recycled until 4 rounds later,
which is 3 barriers of slack against an asynchronous kernel still reading it.

Accumulation uses the online-softmax LSE-merge epilogue: round 0 initializes
out/lse, subsequent rounds merge.

Launch:
    torchrun --nproc-per-node 8 xpu_ring_fa_ipc_push_group.py --check-transfer
    torchrun --nproc-per-node 8 xpu_ring_fa_ipc_push_group.py --q-seq-len 65536 --loops 10
    torchrun --nproc-per-node 4 xpu_ring_fa_ipc_push_group.py --check-transfer   # flat fallback
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

# Ranks per PCIe switch. Groups are contiguous: 0-3, 4-7. Used only to classify
# an edge as intra- or cross-switch for reporting; the schedule itself is
# explicit and does not derive anything from this.
GROUP_SIZE = 4

# Slots are recycled round-robin by round index. 4 gives 3 barriers of slack
# before a slot is overwritten.
NUM_BUFFERS = 4

SLOT_ALIGN = 4096

# ---------------------------------------------------------------------- #
# HOLD[t][r] = global block index held by rank r at round t.
# Row 0 must be the identity: every rank starts with its own block.
# ---------------------------------------------------------------------- #
HOLD_W8 = (
    (0, 1, 2, 3, 4, 5, 6, 7),
    (7, 0, 1, 2, 3, 4, 5, 6),
    (2, 7, 4, 1, 6, 3, 0, 5),
    (1, 4, 6, 7, 5, 0, 2, 3),
    (4, 5, 7, 6, 0, 1, 3, 2),
    (6, 3, 5, 4, 2, 7, 1, 0),
    (3, 6, 0, 5, 1, 2, 7, 4),
    (5, 2, 3, 0, 7, 6, 4, 1),
)


def _align_up(v, a=SLOT_ALIGN):
    return (v + a - 1) // a * a


def flat_hold(world):
    """HOLD table for the plain rank+1 ring: rank r holds (r - t) % world."""
    return tuple(
        tuple((r - t) % world for r in range(world)) for t in range(world)
    )


def build_hold(world):
    """Pick the schedule for this world size."""
    if world == 8:
        return HOLD_W8
    return flat_hold(world)


def assert_valid_schedule(hold, world):
    """
    A schedule is usable iff:
      * it has exactly `world` rounds and `world` columns;
      * round 0 is the identity (every rank starts on its own block);
      * every round is a permutation of 0..world-1 (one block per rank);
      * every rank sees every block exactly once over the whole pass;
      * consecutive rounds differ by a permutation, i.e. the induced edge set
        has exactly one outgoing and one incoming edge per rank.
    The last property is what makes the single per-round barrier sufficient.
    """
    assert len(hold) == world, f"schedule has {len(hold)} rounds, need {world}"
    for t, row in enumerate(hold):
        assert len(row) == world, f"round {t} has {len(row)} entries"
        assert sorted(row) == list(range(world)), (
            f"round {t} is not a permutation: {row}"
        )
    assert tuple(hold[0]) == tuple(range(world)), "round 0 must be the identity"
    for r in range(world):
        col = [hold[t][r] for t in range(world)]
        assert sorted(col) == list(range(world)), (
            f"rank {r} does not see every block exactly once: {col}"
        )
    for t in range(world - 1):
        pos = {b: r for r, b in enumerate(hold[t])}
        dsts = [pos[hold[t + 1][r]] for r in range(world)]
        # dsts[r] is the SOURCE of rank r at round t+1; it must be a bijection.
        assert sorted(dsts) == list(range(world)), (
            f"round {t}->{t + 1} is not a permutation"
        )


def build_edges(hold, world):
    """
    edges[t][src] = dst, for t in 0..world-2.

    At round t rank `src` holds block hold[t][src]; the unique rank that must
    hold that same block at round t+1 is its push destination.
    """
    edges = []
    for t in range(world - 1):
        nxt = {b: r for r, b in enumerate(hold[t + 1])}
        edges.append(tuple(nxt[hold[t][src]] for src in range(world)))
    return tuple(edges)


def read_slot_of(step):
    """Slot holding the block delivered for round `step` (step >= 1)."""
    return step % NUM_BUFFERS


def write_slot_of(step):
    """Slot in the PEER's arena that the push issued at round `step` lands in."""
    return (step + 1) % NUM_BUFFERS


def same_group(a, b, group_size=GROUP_SIZE):
    return a // group_size == b // group_size


def cross_counts(edges, world, group_size=GROUP_SIZE):
    """(send, recv) cross-group counts per rank, for reporting."""
    send = [0] * world
    recv = [0] * world
    for row in edges:
        for src, dst in enumerate(row):
            if not same_group(src, dst, group_size):
                send[src] += 1
                recv[dst] += 1
    return send, recv


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Benchmark and validate sycl_tla_fmha ring attention with "
            "Level-Zero IPC P2P push overlap on XPU, driven by an explicit "
            "permutation schedule."
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
    p.add_argument("--group-size", type=int, default=GROUP_SIZE,
                   help="Ranks per PCIe switch; only affects cross-group "
                        "accounting and reporting")
    p.add_argument("--print-schedule", action="store_true",
                   help="Dump the HOLD table and the derived edge set")
    p.add_argument("--backend", type=str, default="xccl",
                   help="bootstrap process group backend; used only for "
                        "handle exchange, per-round barriers and reductions")
    p.add_argument("--profile", action="store_true",
                   help="Dump a PyTorch profiler trace over the timed loops")
    p.add_argument("--profile-dir", type=str, default="./profiler_out_push_group",
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
    所有 ring buffer 都住在一块 IpcArena 里：进程生命周期内只导出 1 个 IPC handle。

    与结构化 ring 的区别：目标 rank 每轮都不同，所以构造时把 **所有** 对端的
    arena 都 open 一遍（world-1 个），之后每轮按调度表挑指针即可，运行期不再有
    任何 open/close。

    push 模型下 source 永远是本地指针（可以是 caching allocator 的 tensor），
    destination 永远是对端 arena 里的槽 —— slab 被回收/复用变野指针的坑只在
    destination 侧，所以这里天然不存在，也就不需要任何 staging D2D。

    Arena 布局（每段 4KB 对齐）:

        [slot 0][slot 1][slot 2][slot 3]

    每个 slot 内部都是 [K][pad][V]，布局对所有 rank 完全一致，因此
    write_slot_of() 在对端 arena 里指的就是同一个槽。
    """

    def __init__(self, k_local, v_local, rank, world, queue_ptr,
                 group_size=GROUP_SIZE):
        self.rank = rank
        self.world = world
        self.queue_ptr = queue_ptr
        self.dev_index = k_local.device.index
        self.group_size = group_size

        self.hold = build_hold(world)
        assert_valid_schedule(self.hold, world)
        self.edges = build_edges(self.hold, world)
        self.table_driven = (world == 8)

        # 本 rank 每轮的 push 目标。
        self.dst_of_round = [self.edges[t][rank] for t in range(world - 1)]

        self.k_shape = list(k_local.shape)
        self.v_shape = list(v_local.shape)
        self.dtype = k_local.dtype
        self.k_nbytes = k_local.numel() * k_local.element_size()
        self.v_nbytes = v_local.numel() * v_local.element_size()

        # slot 内部布局
        self.k_off_in_slot = 0
        self.v_off_in_slot = _align_up(self.k_nbytes)
        self.slot_stride = _align_up(self.v_off_in_slot + self.v_nbytes)

        self.n_slots = NUM_BUFFERS
        total = self.slot_stride * self.n_slots

        self.arena = ipc.make_arena(self.dev_index, total, queue_ptr)
        self.base = self.arena.base_ptr()

        self.slot_off = [i * self.slot_stride for i in range(self.n_slots)]
        self.k_off = [o + self.k_off_in_slot for o in self.slot_off]
        self.v_off = [o + self.v_off_in_slot for o in self.slot_off]

        self.kbuf = [self.arena.view(self.k_off[i], self.k_shape, self.dtype)
                     for i in range(self.n_slots)]
        self.vbuf = [self.arena.view(self.v_off[i], self.v_shape, self.dtype)
                     for i in range(self.n_slots)]

        # 只交换一次 arena base handle；各 slot 的 offset 所有 rank 一致。
        self.local_handle = self.arena.export_handle()
        gathered = [None] * world
        dist.all_gather_object(gathered, self.local_handle)

        # 目标每轮都变 -> 把所有会用到的对端一次性打开。
        self.peer_handles = []
        self.k_peer_ptr = {}   # peer rank -> [slot -> device ptr]
        self.v_peer_ptr = {}
        for peer in sorted(set(self.dst_of_round)):
            if peer == rank:
                continue
            h = gathered[peer]
            pbase = ipc.open_peer(self.dev_index, h, queue_ptr)
            self.peer_handles.append(h)
            self.k_peer_ptr[peer] = [pbase + self.k_off[i]
                                     for i in range(self.n_slots)]
            self.v_peer_ptr[peer] = [pbase + self.v_off[i]
                                     for i in range(self.n_slots)]

        self._closed = False
        atexit.register(self.close)

        # 任何 copy 发出前，所有 rank 必须已经 open 完对端。
        dist.barrier()
        torch.xpu.synchronize()

    # ------------------------------------------------------------------ #
    # push helper: local pointer -> peer slot
    # ------------------------------------------------------------------ #
    def push(self, src_k, src_v, dst_rank, write_slot):
        """本地 (src_k, src_v) -> dst_rank 的 write_slot，整块一次发完。"""
        return [
            ipc.copy_async(self.dev_index, self.k_peer_ptr[dst_rank][write_slot],
                           src_k, self.k_nbytes, self.queue_ptr),
            ipc.copy_async(self.dev_index, self.v_peer_ptr[dst_rank][write_slot],
                           src_v, self.v_nbytes, self.queue_ptr),
        ]

    def slot_ptrs(self, slot):
        return self.base + self.k_off[slot], self.base + self.v_off[slot]

    def expected_block(self, step):
        """本 rank 在第 step 轮应当持有的 global block 编号。"""
        return self.hold[step][self.rank]

    # ------------------------------------------------------------------ #
    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            torch.xpu.synchronize()
            for h in self.peer_handles:
                ipc.close_peer(self.dev_index, h, self.queue_ptr)
        except Exception:
            pass
        self.peer_handles = []
        self.k_peer_ptr = {}
        self.v_peer_ptr = {}
        # arena view 必须先失效，arena 本身由 ipc.shutdown() 释放。
        self.kbuf = []
        self.vbuf = []


def ring_attention_ipc(ring, consume, kv_new):
    """
    One full pass over all `world` K/V blocks, following ring.edges.

    Forward logic is identical to the structural push version, just with the
    destination read out of the schedule instead of computed as rank+1:

      * kv_new is the K/V this layer produced. It is NOT staged anywhere -- we
        only fence() it (the CE cannot see the compute queue's dependency) and
        then use it directly as the CE source and as the round-0 kernel input.
        The barrier right after the fence is what guarantees that no peer is
        still writing into the slots we are about to consume.

      * Per round t:
            t <  world-1 : push the block we are reading this round into
                           dst_of_round[t]'s slot write_slot_of(t)
            t == world-1 : no copy (the pass is over)
        then enqueue the attention kernel (asynchronous, overlaps the copy),
        then wait on the events of the copies WE issued, then barrier.

      * Every round is a permutation, so "everyone finished sending" ==
        "everyone finished receiving". That per-round barrier is the only
        synchronization the push model needs.

    consume(k_block, v_block, round_idx) is called once per block with
    round_idx increasing 0..world-1, so the LSE-merge epilogue initializes
    exactly once.
    """
    dist.barrier()
    k_src0, v_src0 = kv_new

    for t in range(ring.world):
        pending = []

        # 本轮读的块（也就是本轮要转发出去的块）。
        # t == 0 是调用方自己的 tensor，不在 arena 里。
        if t == 0:
            cur_k_ptr, cur_v_ptr = k_src0.data_ptr(), v_src0.data_ptr()
            cur_k_buf, cur_v_buf = k_src0, v_src0
        else:
            rs = read_slot_of(t)
            cur_k_ptr, cur_v_ptr = ring.slot_ptrs(rs)
            cur_k_buf, cur_v_buf = ring.kbuf[rs], ring.vbuf[rs]

        # 1) 按调度表 push（最后一轮没有：pass 已经结束）。
        if t < ring.world - 1:
            pending += ring.push(cur_k_ptr, cur_v_ptr,
                                 ring.dst_of_round[t], write_slot_of(t))

        # 2) attention kernel，异步下发，盖住上面的传输。
        consume(cur_k_buf, cur_v_buf, t)

        for h in pending:
            h.wait()

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

    group_size = args.group_size

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
    ring = IpcKVRing(k_local, v_local, rank, world, queue_ptr,
                     group_size=group_size)

    if args.print_schedule and rank == 0:
        print("HOLD table (rows = round, cols = rank, value = block):")
        for t, row in enumerate(ring.hold):
            print(f"  round {t}: {' '.join(f'{b}' for b in row)}")
        print("Derived edges (src->dst, '*' = cross-group):")
        for t, row in enumerate(ring.edges):
            desc = " ".join(
                f"{s}->{d}{'*' if not same_group(s, d, group_size) else ''}"
                for s, d in enumerate(row)
            )
            print(f"  round {t}: {desc}")

    # ---------------------------------------------------------------- #
    # Transfer self-check: does round t deliver the block that HOLD says?
    # ---------------------------------------------------------------- #
    if args.check_transfer:
        failures = []

        def check(kb, vb, round_idx):
            src = ring.expected_block(round_idx)
            k_ref = k_full[:, src * s_local:(src + 1) * s_local]
            v_ref = v_full[:, src * s_local:(src + 1) * s_local]
            torch.xpu.synchronize()
            if not torch.equal(kb, k_ref):
                failures.append((round_idx, src, "K",
                                 (kb != k_ref).sum().item()))
            if not torch.equal(vb, v_ref):
                failures.append((round_idx, src, "V",
                                 (vb != v_ref).sum().item()))

        mode = "table-driven" if ring.table_driven else "flat ring"
        # 跑三个 pass：槽位是按轮次轮换的，多跑几遍能暴露 pass 之间的复用问题。
        for it in range(3):
            failures.clear()
            ring_attention_ipc(ring, check, kv_new=(k_local, v_local))
            torch.xpu.synchronize()
            if failures:
                for round_idx, src, which, bad in failures:
                    print(f"[rank {rank}] pass {it} round {round_idx} "
                          f"(expect block {src}) {which}: FAILED, "
                          f"{bad} elements differ", flush=True)
            else:
                print(f"[rank {rank}] pass {it} IPC K/V rotation ({mode}, "
                      f"push): PASSED ({world} rounds, bit-exact)", flush=True)
            dist.barrier()

        ring.close()
        dist.barrier()
        ipc.shutdown()
        dist.destroy_process_group()
        return

    def run_ring():
        # round_idx drives the epilogue: 0 initializes out/lse, >0 LSE-merges.
        # out/lse accumulate across the whole pass and must not be reset.
        # kv_new 就是这一层产出的 K/V：push 模型下直接拿它当 CE source，
        # 不需要任何 staging copy。
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

    xsend, xrecv = cross_counts(ring.edges, world, group_size)
    n_cross_edges = sum(xsend)

    if rank == 0:
        print("XPU BF16 ring attention (Level-Zero IPC P2P, push, "
              "schedule-driven)")
        print("  Input layout       : [B, S, H, D]")
        print(f"  World size         : {world}")
        print(f"  Schedule           : "
              f"{'explicit HOLD table' if ring.table_driven else 'flat ring (rank+1)'}")
        print(f"  Group size         : {group_size} "
              f"(ranks 0-{group_size - 1} / {group_size}-{world - 1})")
        print(f"  Rounds             : {world} ({world - 1} transfers)")
        print(f"  Cross-group edges  : {n_cross_edges} total "
              f"({n_cross_edges / max(world - 1, 1):.1f} per round)")
        print("  Cross-group per rank (send/recv):")
        print("      " + "  ".join(
            f"r{i}:{xsend[i]}/{xrecv[i]}" for i in range(world)))
        print(f"  Global seq len     : {S_global}")
        print(f"  Local shard len    : {s_local}")
        print(f"  Q heads / KV heads : {Hq} / {Hkv}")
        print(f"  QK hdim / V hdim   : {Dqk} / {Dvo}")
        print(f"  Data type          : {dtype}")
        print(f"  Arena slots        : {ring.n_slots} "
              f"(round-indexed rotation)")
        print(f"  Arena size         : "
              f"{ring.slot_stride * ring.n_slots / 1024**3:.3f} GiB")
        print(f"  Peers opened       : {len(ring.peer_handles)}")
        print(f"  Bootstrap backend  : {args.backend}")
        print("  Transfer path      : CE write to peer "
              "(zeCommandListAppendMemoryCopy, single dedicated copy engine)")
        print("  Per-loop staging   : none (push sources may be plain "
              "allocator tensors; only a stream fence is needed)")
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
            args.profile_dir, f"ring_fa_ipc_push_group_trace_rank{rank}.json"
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
    # 每个 rank 每个 pass 发 world-1 块，其中 xsend[rank] 块走跨组链路。
    cross_moved = kv_bytes * xsend[rank]
    intra_moved = kv_bytes * (world - 1) - cross_moved
    moved_bytes = kv_bytes * (world - 1)

    if rank == 0:
        print("\nResults")
        print(f"  Output shape       : {list(out.shape)} [B, S, H, D]")
        print(f"  Output dtype       : {out.dtype}")
        print(f"  Per-rank Q memory  : {q_bytes / 1024**3:.3f} GiB")
        print(f"  Per-rank KV memory : {kv_bytes / 1024**3:.3f} GiB")
        print(f"  Intra-switch moved : {intra_moved / 1024**2:.1f} MiB "
              f"/ loop (rank 0)")
        print(f"  Cross-switch moved : {cross_moved / 1024**2:.1f} MiB "
              f"/ loop (rank 0)")
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
            out.float(), ref.float(), atol=5e-3, rtol=5e-3
        )
        max_abs_diff = (out.float() - ref.float()).abs().max().item()
        if rank == 0:
            print("\nValidation")
            print("  Reference          : F.scaled_dot_product_attention")
            print("  Status             : PASSED")
            print("  Tolerance          : atol=5e-3, rtol=5e-3")
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
