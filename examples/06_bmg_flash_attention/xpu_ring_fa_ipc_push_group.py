#!/usr/bin/env python3
"""
Two-level (grouped) ring attention with Level-Zero IPC P2P, PUSH model.

Exactly the forward logic of xpu_ring_fa_ipc_push.py -- no staging D2D, round 0
consumes and sends the caller's own K/V tensors, every round waits on the events
of the copies it issued and then barriers -- extended to a two-level ring.

Push model
----------
Every transfer is a copy-engine *write* into the peer, never a read from it.
On XPU a cross-UPI CE read costs >2x a same-socket one; posted CE writes do not,
which is exactly why the cross-switch hop is pushed rather than pulled.

Because the CE *source* is never exported as an IPC handle, it may be an
ordinary caching-allocator tensor. That is what removes the pull version's
mandatory stage() D2D: round 0 pushes the layer's freshly produced K/V straight
out of the tensor the caller handed us, and the local kernel of round 0 reads
that same tensor. Only destinations must live in the arena, and they always do.

fence() is still required: the caller's K/V is produced by a kernel on the
compute queue, and the CE is a different engine that cannot see that dependency.

Topologies
----------
  world <= GROUP_SIZE   -- flat ring (n_groups == 1, no cross-group traffic).
  world in {8,16,32}    -- groups of GROUP_SIZE=4, one PCIe switch each
                           (0-3 / 4-7 / ...). Intra-group: push to rank+1 within
                           the group. Cross-group: push to (rank+GROUP_SIZE)
                           % world, the positionally matching rank of the next
                           group.

Cross-switch scheduling (no chunking)
-------------------------------------
Within one outer round of inner = GROUP_SIZE rounds only rounds 0..inner-2 issue
an intra-group ring push; round inner-1 issues none, because it consumes the
block it already holds. The single dedicated copy engine is therefore completely
idle for that whole round, so the cross-UPI K/V block is pushed there as ONE
unchunked transfer:

    ring0 | ring1 | ring2 | [ round 3: whole cross-UPI K/V push, unopposed ]

No weights, no chunk schedule, no head-of-line blocking (nothing intra-group is
ever queued behind it -- that round has no intra-group copy at all), and its
completion wait folds into that round's existing barrier, so an outer-round
boundary adds no synchronization of its own.

Slot layout
-----------
Slot numbering is identical on every rank, so write_slot_of(s) names the same
slot in the peer's arena as it does locally.

    slots 1..NUM_BUFFERS-1  -- intra-group rotation (push destinations)
    BASE_SLOTS = (0, 4, 5)  -- "the block an outer round starts from", rotating
                               3-deep so that the slot we fill this outer round
                               is distinct from the one being read this round
                               and from the one we fill next round

base_slot_of(0) == 0, but at g == 0 slot 0 is never a source: round 0 uses the
caller's tensor. Slot 0 is only ever a cross-group destination (base_slot_of(3),
reached when n_groups > 3), so nothing is written into it before the pass has
consumed round 0.

Accumulation uses the online-softmax LSE-merge epilogue: round 0 initializes
out/lse, subsequent rounds merge. round_idx is the GLOBAL round counter
(g * inner + s), so only the very first round initializes.

Launch:
    torchrun --nproc-per-node 4  xpu_ring_fa_ipc_push_group.py --check-transfer
    torchrun --nproc-per-node 8  xpu_ring_fa_ipc_push_group.py --check-transfer
    torchrun --nproc-per-node 16 xpu_ring_fa_ipc_push_group.py --q-seq-len 65536 --loops 10
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

# Ranks per PCIe switch. Groups are contiguous: 0-3, 4-7, 8-11, ...
GROUP_SIZE = 4

# Slots 1..NUM_BUFFERS-1 are the intra-group rotation (push destinations),
# cycled round-robin so a slot is not recycled while an asynchronous kernel may
# still be reading it.
NUM_BUFFERS = 4

# Two extra slots so the base rotation is three-deep.
NUM_XGROUP_BUFFERS = 2

# The slots that rotate as "the block this rank starts an outer round with".
# All are outside the 1..NUM_BUFFERS-1 intra-group rotation range.
BASE_SLOTS = (0, NUM_BUFFERS, NUM_BUFFERS + 1)

SLOT_ALIGN = 4096


def _align_up(v, a=SLOT_ALIGN):
    return (v + a - 1) // a * a


def base_slot_of(g):
    """
    Slot holding the block this rank starts outer round g with.

    Deterministic and identical on every rank, so the pusher knows which slot of
    the peer to write without any handshake. g == 0 is special: the block is the
    caller's own tensor, not slot 0 -- base_slot_of(0) is only meaningful as a
    *destination* for whoever wraps around to it.
    """
    return BASE_SLOTS[g % len(BASE_SLOTS)]


def read_slot_of(step, g):
    """
    Slot holding the K/V block valid at inner round `step` of outer round g.

    step == 0 at g == 0 is handled by the caller (own tensor), so this is only
    consulted for g > 0 there.
    """
    if step == 0:
        return base_slot_of(g)
    return 1 + ((step - 1) % (NUM_BUFFERS - 1))


def write_slot_of(step):
    """
    Slot in the PEER's arena that the intra-group push issued at inner round
    `step` lands in. Only ever 1..NUM_BUFFERS-1, so it can never collide with
    any base slot.
    """
    return 1 + (step % (NUM_BUFFERS - 1))


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Benchmark and validate sycl_tla_fmha ring attention with "
            "Level-Zero IPC P2P push overlap on XPU. Flat ring for "
            "world <= group size, two-level (grouped) ring beyond that."
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
                   help="Ranks per PCIe switch (two-level ring group size)")
    p.add_argument("--backend", type=str, default="xccl",
                   help="bootstrap process group backend; used only for "
                        "handle exchange, per-round barriers and reductions")
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


def expected_block_of(rank, world, group_size, g, s):
    """
    Which global block rank `rank` sees at outer round g, inner round s.

    Pushing forward is receiving from behind, so the rotation is identical to
    the pull version:

        b0    = (rank - group_size * g) % world        # block at inner step 0
        block = group_base(b0) + ((local(rank) - s) % group_size)
    """
    local = rank % group_size
    b0 = (rank - group_size * g) % world
    return (b0 - local) + ((local - s) % group_size)


class IpcKVRing:
    """
    所有 ring buffer 都住在一块 IpcArena 里：进程生命周期内只导出 1 个 IPC handle、
    每个对端只 open 1 次。

    push 模型下 source 永远是本地指针（可以是 caching allocator 的 tensor），
    destination 永远是对端 arena 里的槽 —— slab 被回收/复用变野指针的坑只在
    destination 侧，所以这里天然不存在，也就不需要任何 staging D2D。

    Arena 布局（每段 4KB 对齐）:

        [slot 0][slot 1][slot 2][slot 3][slot 4][slot 5]
         ^base A  ^---- 组内轮换 ----^   ^base B  ^base C   (B/C 仅 two-level)

    每个 slot 内部都是 [K][pad][V]，布局对所有 rank 完全一致，因此
    write_slot_of() 在对端 arena 里指的就是同一个槽。

    构造函数只分配 arena、交换 handle、打开两个下游对端；不做任何 staging。
    """

    def __init__(self, k_local, v_local, rank, world, queue_ptr,
                 group_size=GROUP_SIZE):
        self.rank = rank
        self.world = world
        self.queue_ptr = queue_ptr
        self.dev_index = k_local.device.index

        self.group_size = group_size
        self.two_level = world > group_size
        if self.two_level:
            assert world % group_size == 0, (
                f"world ({world}) must be a multiple of group size "
                f"({group_size}) for the two-level ring"
            )
        self.n_groups = world // group_size if self.two_level else 1
        self.group_id = rank // group_size if self.two_level else 0
        self.local_id = rank % group_size if self.two_level else rank
        self.inner_world = group_size if self.two_level else world

        group_base = self.group_id * group_size if self.two_level else 0
        # Push destinations: intra-group ring downstream ...
        self.next_rank = group_base + (self.local_id + 1) % self.inner_world
        # ... and the cross-group (cross-switch / cross-UPI) downstream.
        self.next_group_rank = (rank + group_size) % world

        self.k_shape = list(k_local.shape)
        self.v_shape = list(v_local.shape)
        self.dtype = k_local.dtype
        self.k_nbytes = k_local.numel() * k_local.element_size()
        self.v_nbytes = v_local.numel() * v_local.element_size()

        # slot 内部布局
        self.k_off_in_slot = 0
        self.v_off_in_slot = _align_up(self.k_nbytes)
        self.slot_stride = _align_up(self.v_off_in_slot + self.v_nbytes)

        n_slots = NUM_BUFFERS + (NUM_XGROUP_BUFFERS if self.two_level else 0)
        self.n_slots = n_slots
        total = self.slot_stride * n_slots

        self.arena = ipc.make_arena(self.dev_index, total, queue_ptr)
        self.base = self.arena.base_ptr()

        self.slot_off = [i * self.slot_stride for i in range(n_slots)]
        self.k_off = [o + self.k_off_in_slot for o in self.slot_off]
        self.v_off = [o + self.v_off_in_slot for o in self.slot_off]

        self.kbuf = [self.arena.view(self.k_off[i], self.k_shape, self.dtype)
                     for i in range(n_slots)]
        self.vbuf = [self.arena.view(self.v_off[i], self.v_shape, self.dtype)
                     for i in range(n_slots)]

        # 只交换一次 arena base handle；各 slot 的 offset 所有 rank 一致。
        self.local_handle = self.arena.export_handle()
        gathered = [None] * world
        dist.all_gather_object(gathered, self.local_handle)

        self.peer_handles = []

        # 组内下游（push destination）
        self.ring_peer_handle = gathered[self.next_rank]
        ring_base = ipc.open_peer(self.dev_index, self.ring_peer_handle,
                                  queue_ptr)
        self.peer_handles.append(self.ring_peer_handle)
        self.k_peer_ptr = [ring_base + self.k_off[i] for i in range(n_slots)]
        self.v_peer_ptr = [ring_base + self.v_off[i] for i in range(n_slots)]

        # 跨组下游（push destination）
        if self.two_level:
            self.xg_peer_handle = gathered[self.next_group_rank]
            xg_base = ipc.open_peer(self.dev_index, self.xg_peer_handle,
                                    queue_ptr)
            self.peer_handles.append(self.xg_peer_handle)
            self.xg_k_peer_ptr = [xg_base + self.k_off[i]
                                  for i in range(n_slots)]
            self.xg_v_peer_ptr = [xg_base + self.v_off[i]
                                  for i in range(n_slots)]
        else:
            self.xg_peer_handle = None
            self.xg_k_peer_ptr = []
            self.xg_v_peer_ptr = []

        self._closed = False
        atexit.register(self.close)

        # 任何 copy 发出前，所有 rank 必须已经 open 完对端。
        dist.barrier()
        torch.xpu.synchronize()

    # ------------------------------------------------------------------ #
    # push helpers: local pointer -> peer slot
    # ------------------------------------------------------------------ #
    def push_ring(self, src_k, src_v, write_slot):
        """组内 ring：本地 (src_k, src_v) -> next_rank 的 write_slot。"""
        return [
            ipc.copy_async(self.dev_index, self.k_peer_ptr[write_slot],
                           src_k, self.k_nbytes, self.queue_ptr),
            ipc.copy_async(self.dev_index, self.v_peer_ptr[write_slot],
                           src_v, self.v_nbytes, self.queue_ptr),
        ]

    def push_xgroup(self, src_k, src_v, dst_slot):
        """
        跨 switch / 跨 UPI：本地 (src_k, src_v) -> next_group_rank 的 dst_slot，
        整块一次发完，不分块。只在组内最后一轮调用 —— 那一轮没有任何 ring copy，
        copy engine 全空，大块可以独占 BCS 跑满整个 kernel 的时间。
        """
        return [
            ipc.copy_async(self.dev_index, self.xg_k_peer_ptr[dst_slot],
                           src_k, self.k_nbytes, self.queue_ptr),
            ipc.copy_async(self.dev_index, self.xg_v_peer_ptr[dst_slot],
                           src_v, self.v_nbytes, self.queue_ptr),
        ]

    def slot_ptrs(self, slot):
        return self.base + self.k_off[slot], self.base + self.v_off[slot]

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
        self.k_peer_ptr = []
        self.v_peer_ptr = []
        self.xg_k_peer_ptr = []
        self.xg_v_peer_ptr = []
        # arena view 必须先失效，arena 本身由 ipc.shutdown() 释放。
        self.kbuf = []
        self.vbuf = []


def ring_attention_ipc(ring, consume, kv_new):
    """
    One full ring pass over all `world` K/V blocks.

    Forward logic is identical to the flat push version, just nested:

      * kv_new is the K/V this layer produced. It is NOT staged anywhere -- we
        only fence() it (the CE cannot see the compute queue's dependency) and
        then use it directly as the CE source and as the round-0 kernel input.
        The barrier right after the fence is what guarantees that no peer is
        still writing into the slots we are about to consume.

      * Per inner round s of outer round g:
            s <  inner-1 : push the block we are reading this round into
                           next_rank's slot write_slot_of(s)
            s == inner-1 : no ring copy (we already hold what we need), so the
                           whole cross-UPI block -- the block this outer round
                           STARTED from -- is pushed in one shot into
                           next_group_rank's slot base_slot_of(g+1)
        then enqueue the attention kernel (asynchronous, overlaps the copies),
        then wait on the events of the copies WE issued, then barrier.

      * Because every rank runs the identical schedule, "everyone finished
        sending" == "everyone finished receiving". That per-round barrier is
        the only synchronization the push model needs, and the cross-group wait
        folds into it, so an outer-round boundary costs nothing extra.

    consume(k_block, v_block, global_round) is called once per block with
    global_round = g * inner + s increasing 0..world-1, so the LSE-merge
    epilogue initializes exactly once.

    Flat mode is simply n_groups == 1: no cross-group traffic is ever issued.
    """
    dist.barrier()
    k_src0, v_src0 = kv_new

    inner = ring.inner_world
    for g in range(ring.n_groups):
        prefetch = ring.two_level and g < ring.n_groups - 1
        xg_dst = base_slot_of(g + 1)   # 下游下一个 outer round 的起始槽

        # 本 outer round 的起始块：g == 0 是调用方的 tensor（不在 arena 里），
        # 之后是上一个 outer round 跨组 push 落进来的槽。
        if g == 0:
            base_k_ptr, base_v_ptr = k_src0.data_ptr(), v_src0.data_ptr()
            base_k_buf, base_v_buf = k_src0, v_src0
        else:
            bs = base_slot_of(g)
            base_k_ptr, base_v_ptr = ring.slot_ptrs(bs)
            base_k_buf, base_v_buf = ring.kbuf[bs], ring.vbuf[bs]

        for s in range(inner):
            pending = []

            # 本轮读的块（也就是本轮组内要转发出去的块）。
            if s == 0:
                cur_k_ptr, cur_v_ptr = base_k_ptr, base_v_ptr
                cur_k_buf, cur_v_buf = base_k_buf, base_v_buf
            else:
                rs = read_slot_of(s, g)
                cur_k_ptr, cur_v_ptr = ring.slot_ptrs(rs)
                cur_k_buf, cur_v_buf = ring.kbuf[rs], ring.vbuf[rs]

            # 1) 组内 ring push（最后一轮没有：要的 KV 已经在手上）。
            if s < inner - 1:
                pending += ring.push_ring(cur_k_ptr, cur_v_ptr,
                                          write_slot_of(s))
            elif prefetch:
                # 2) copy engine 本轮全空，整块跨 UPI push 独占 BCS。
                #    源是本 outer round 的起始块 —— 正是下游下一轮要的那块。
                pending += ring.push_xgroup(base_k_ptr, base_v_ptr, xg_dst)

            # 3) attention kernel，异步下发，盖住上面的传输。
            consume(cur_k_buf, cur_v_buf, g * inner + s)

            for h in pending:
                h.wait()


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
    two_level = world > group_size
    if two_level and world % group_size != 0:
        raise RuntimeError(
            f"world size {world} is not a multiple of group size {group_size}; "
            f"supported world sizes are 1/2/4/8/16/32"
        )

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

    # ---------------------------------------------------------------- #
    # Transfer self-check: does global round (g, s) deliver the block of
    # rank expected_block_of(rank, world, group_size, g, s)?
    # ---------------------------------------------------------------- #
    if args.check_transfer:
        failures = []
        inner = ring.inner_world

        def check(kb, vb, round_idx):
            g, s = divmod(round_idx, inner)
            src = expected_block_of(rank, world, group_size, g, s)
            k_ref = k_full[:, src * s_local:(src + 1) * s_local]
            v_ref = v_full[:, src * s_local:(src + 1) * s_local]
            torch.xpu.synchronize()
            if not torch.equal(kb, k_ref):
                failures.append((round_idx, src, "K",
                                 (kb != k_ref).sum().item()))
            if not torch.equal(vb, v_ref):
                failures.append((round_idx, src, "V",
                                 (vb != v_ref).sum().item()))

        # 跑三个 pass：base slot 是三深轮换，n_groups=4 时一个 pass 用掉
        # slot -,4,5,0，三个 pass 才能把所有起始相位都走一遍。
        mode = "two-level" if two_level else "flat"
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

    if rank == 0:
        print("XPU BF16 ring attention (Level-Zero IPC P2P, push)")
        print("  Input layout       : [B, S, H, D]")
        print(f"  World size         : {world}")
        print(f"  Topology           : "
              f"{'two-level (grouped)' if two_level else 'flat ring'}")
        if two_level:
            print(f"  Group size         : {group_size} "
                  f"({ring.n_groups} groups, one per PCIe switch)")
            print("  Intra-group target : rank+1 within group (push)")
            print(f"  Cross-group target : rank+{group_size} (positional, push)")
            print(f"  X-group schedule   : whole K/V block, unchunked, in "
                  f"inner round {ring.inner_world - 1} (no ring copy there)")
            print(f"  Base slots         : {BASE_SLOTS} (3-deep rotation)")
            print("  Outer sync         : none beyond the per-round barrier")
        print(f"  Global seq len     : {S_global}")
        print(f"  Local shard len    : {s_local}")
        print(f"  Q heads / KV heads : {Hq} / {Hkv}")
        print(f"  QK hdim / V hdim   : {Dqk} / {Dvo}")
        print(f"  Data type          : {dtype}")
        print(f"  Arena slots        : {ring.n_slots} "
              f"({NUM_BUFFERS - 1} rotating + "
              f"{len(BASE_SLOTS) if two_level else 1} base)")
        print(f"  Arena size         : "
              f"{ring.slot_stride * ring.n_slots / 1024**3:.3f} GiB")
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
    # 组内 ring：每个 outer round (inner_world - 1) 次。
    intra_moved = kv_bytes * (ring.inner_world - 1) * ring.n_groups
    # 跨组：n_groups - 1 次整块。
    inter_moved = (kv_bytes * (ring.n_groups - 1)) if two_level else 0
    moved_bytes = intra_moved + inter_moved

    if rank == 0:
        print("\nResults")
        print(f"  Output shape       : {list(out.shape)} [B, S, H, D]")
        print(f"  Output dtype       : {out.dtype}")
        print(f"  Per-rank Q memory  : {q_bytes / 1024**3:.3f} GiB")
        print(f"  Per-rank KV memory : {kv_bytes / 1024**3:.3f} GiB")
        print(f"  Intra-switch moved : {intra_moved / 1024**2:.1f} MiB "
              f"/ loop / rank")
        if two_level:
            print(f"  Cross-switch moved : {inter_moved / 1024**2:.1f} MiB "
                  f"/ loop / rank (one whole block per outer round)")
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
