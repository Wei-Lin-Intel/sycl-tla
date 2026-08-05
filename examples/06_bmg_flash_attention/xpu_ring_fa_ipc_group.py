#!/usr/bin/env python3
"""
Ring attention over prefill_bf16_bshd_kv_round with Level-Zero IPC P2P overlap.

Same math as xpu_ring_fa.py, but K/V rotation goes through sycl_tla_ipc_p2p
(zeCommandListAppendMemoryCopy on a dedicated copy engine) instead of
torch.distributed batch_isend_irecv. The copy engine is fully decoupled from
the SYCL compute queue, so the transfer overlaps the attention kernel without
the collective's host-side serialization.

Two topologies, selected by world size:

  world <= GROUP_SIZE -- flat ring.
      Every rank pulls from prev_rank each round. All links stay inside one
      PCIe switch, so P2P is fast and a single-level ring is optimal. This is
      just the degenerate n_groups == 1 case of the loop below.

  world in {8, 16, 32} -- two-level (grouped) ring.
      Ranks are partitioned into groups of GROUP_SIZE=4 (one PCIe switch per
      group: 0-3 / 4-7 / 8-11 / ...). Inside a group we run the flat ring.
      Between groups we run a second, outer ring where rank r pulls from rank
      (r - GROUP_SIZE) mod world -- the positionally matching rank in the
      previous group.

Cross-switch scheduling
-----------------------
The cross-switch hop is an order of magnitude slower than the intra-switch hop
(a 64k KV block takes ~11 ms on B70, comparable to the FA kernel itself). It is
split into chunks appended to the same in-order copy engine queue as the
intra-group ring copies, one chunk after each round's ring copy.

The chunk sizes are deliberately NOT uniform. Within one outer round of
inner=GROUP_SIZE=4 rounds, only rounds 0..2 issue an intra-group ring copy --
round 3 needs no transfer because it consumes the block it already holds. That
leaves the copy engine completely idle for the whole of round 3.

Default weights (2, 2, 6, 0), i.e. tenths of the block:

    ring0 | xg 2/10 | ring1 | xg 2/10 | ring2 | xg 6/10 | (round 3: no ring copy)
                                                ^ starts the moment ring2
                                                  retires, then runs unopposed
                                                  through round 3's kernel

  * Zero head-of-line blocking. The bulk chunk is queued after the LAST
    intra-group copy of the outer round, so no ring copy can ever be stuck
    behind it -- which was the entire reason for chunking in the first place.

  * Maximum contention-free window. Queueing the bulk at round inner-2 (rather
    than at inner-1) lets it begin during the tail of that round's kernel and
    continue through the final round's kernel, with no intra-group traffic
    competing for the single BCS.

  * The 2/10 head chunks are large enough to keep the link busy during the two
    fully-loaded rounds, yet small enough that a ring copy queued behind one
    waits only ~1/5 of a cross-switch block.

Tune with --xgroup-weights (e.g. "1,1,8,0" or "3,3,4,0") after looking at a
trace; the schedule is data-driven and needs no code change.

This device exposes exactly one dedicated copy engine (one copy-only queue group
with numQueues == 1), so both traffic classes necessarily share a single BCS.
Weighted chunk interleaving is what makes them share it well.

Three-slot base rotation (why there is no outer-round collective)
-----------------------------------------------------------------
Outer round g starts from the block in base_slot_of(g); the cross-group chunks
issued during round g fill base_slot_of(g+1).

With only two base slots those roles would alternate, so base_slot_of(g+2) ==
base_slot_of(g): a rank one outer round ahead of its cross-group downstream
would start overwriting the very block that downstream was still pulling. Zero
slack -- which would force a collective at every outer boundary, landing exactly
where the bulk chunk is supposed to stay hidden and serializing every rank on
the transfer we were trying to overlap.

With THREE base slots rotating (g % 3), round g reads slot[g%3] and writes
slot[(g+1)%3]; round g+1 writes slot[(g+2)%3], still distinct from slot[g%3].
A rank may therefore run a full outer round (GROUP_SIZE attention kernels)
ahead of its downstream before any conflict is possible -- the same order of
slack the intra-group ring already relies on. There is consequently no
synchronization of any kind at an outer-round boundary.

Per outer round the slot roles are:

    slot base_slot_of(g)    -- read only (local kernel at s==0, intra-group peer
                               at s==0, cross-group peer for the whole round)
    slot base_slot_of(g+1)  -- written only, by cross-group chunks
    slot base_slot_of(g+2)  -- idle this round; the downstream's grace period
    slots 1..NUM_BUFFERS-1  -- intra-group rotation, untouched by cross traffic

Read set and write set are physically disjoint, so intra-group and cross-group
copies need no Level-Zero event dependency between them -- that is exactly what
lets them interleave freely on the single BCS, in any proportion.

The one wait that remains at the outer boundary is draining our OWN cross-group
chunks: round g+1's very first kernel reads base_slot_of(g+1), so those bytes
must have landed. That is a local event wait, not a collective, and it is
precisely the window the bulk chunk was sized to fit into.

Per-layer staging
-----------------
In the real workload every transformer layer produces brand-new K/V tensors out
of the caching allocator, which cannot be exported as IPC handles. They must be
copied into the arena first. Every pass therefore begins with stage(), which is
the one unavoidable local D2D. Nothing is staged at construction time.

Accumulation uses the online-softmax LSE-merge epilogue: round 0 initializes
out/lse, subsequent rounds merge. round_idx is the GLOBAL round counter
(g * inner + s) so that only the very first round initializes.

Launch:
    torchrun --nproc-per-node 4  xpu_ring_fa_ipc.py --check-transfer
    torchrun --nproc-per-node 8  xpu_ring_fa_ipc.py --check-transfer
    torchrun --nproc-per-node 16 xpu_ring_fa_ipc.py --q-seq-len 65536 --loops 10
    torchrun --nproc-per-node 16 xpu_ring_fa_ipc.py --q-seq-len 65536 --profile
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

# Slot 0 holds the block this rank starts an outer round with; slots
# 1..NUM_BUFFERS-1 are intra-group copy destinations cycled round-robin, giving
# NUM_BUFFERS-2 rounds of slack between a slot being read by the (asynchronous)
# attention kernel and being recycled as a write target.
NUM_BUFFERS = 4

# Two extra slots so the base rotation is three-deep. Three is the minimum that
# keeps "the slot I write this round" distinct from "the slot my downstream
# reads this round" AND from "the slot I write next round" -- which is what buys
# a full outer round of slack and removes the need for any outer-round
# synchronization.
NUM_XGROUP_BUFFERS = 2

# The slots that rotate as "the block this rank starts an outer round with".
# All are outside the 1..NUM_BUFFERS-1 intra-group rotation range.
BASE_SLOTS = (0, NUM_BUFFERS, NUM_BUFFERS + 1)

SLOT_ALIGN = 4096


def _align_up(v, a=SLOT_ALIGN):
    return (v + a - 1) // a * a


def _align_down(v, a=SLOT_ALIGN):
    return v // a * a


def base_slot_of(g):
    """
    Slot holding the block this rank starts outer round g with.

    Deterministic and identical on every rank, so a peer can compute which of
    our slots to pull from without any handshake. base_slot_of(0) == 0 always,
    so stage() writing slot 0 is correct at the start of every pass regardless
    of where the previous pass ended.
    """
    return BASE_SLOTS[g % len(BASE_SLOTS)]


def read_slot_of(step, g):
    """Slot holding the K/V block valid at inner round `step` of outer round g."""
    if step == 0:
        return base_slot_of(g)
    return 1 + ((step - 1) % (NUM_BUFFERS - 1))


def write_slot_of(step):
    """
    Slot the intra-group copy issued at inner round `step` lands in.

    Only ever produces 1..NUM_BUFFERS-1, so it can never collide with any base
    slot.
    """
    return 1 + (step % (NUM_BUFFERS - 1))


def default_xgroup_weights(inner):
    """
    Relative share of the cross-switch block carried by each inner round,
    expressed in tenths for the standard inner == 4 case: (2, 2, 6, 0).

    Rounds 0..inner-2 issue an intra-group ring copy; round inner-1 does not, so
    the copy engine is idle for its entire duration. Give the loaded rounds a
    moderate share each and put the bulk on round inner-2, the last round that
    has any intra-group traffic: the bulk then starts as soon as that traffic
    retires and runs unopposed through the final round's kernel.

    Weight 0 on the final round is intentional -- queueing there would start the
    same bytes strictly later for no benefit.
    """
    if inner <= 1:
        return [10]
    if inner == 2:
        return [10, 0]
    head = inner - 2                      # rounds that still carry a ring copy
    bulk = 10 - 2 * head                  # inner == 4 -> 6
    if bulk < 1:
        # Very large groups: fall back to an even split over the loaded rounds.
        return [1] * head + [1] + [0]
    return [2] * head + [bulk] + [0]


def build_chunk_schedule(span, weights):
    """
    Turn per-round weights into a per-round (offset, nbytes) schedule.

    Returns a list of len(weights) entries; entry i is None if round i carries
    no cross-group traffic. Boundaries are SLOT_ALIGN-aligned via align_down and
    the last non-zero round absorbs the remainder, so the pieces always tile
    [0, span) exactly with no gap and no overlap.
    """
    sched = [None] * len(weights)
    total_w = sum(weights)
    if span <= 0 or total_w <= 0:
        return sched

    active = [i for i, w in enumerate(weights) if w > 0]
    off = 0
    for j, i in enumerate(active):
        if j == len(active) - 1:
            n = span - off                     # tail absorbs the rounding slack
        else:
            n = _align_down(span * weights[i] // total_w)
            n = min(n, span - off)
        if n <= 0:
            continue
        sched[i] = (off, n)
        off += n
    return sched


def parse_weights(text, inner):
    ws = [int(x) for x in text.split(",") if x.strip() != ""]
    if len(ws) != inner:
        raise RuntimeError(
            f"--xgroup-weights needs exactly {inner} comma-separated values "
            f"(one per inner round), got {len(ws)}"
        )
    if any(w < 0 for w in ws):
        raise RuntimeError("--xgroup-weights must be non-negative")
    if sum(ws) <= 0:
        raise RuntimeError("--xgroup-weights must not be all zero")
    return ws


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Benchmark and validate sycl_tla_fmha ring attention with "
            "Level-Zero IPC P2P overlap on XPU. Uses a flat ring for "
            "world <= group size and a two-level (grouped) ring beyond that."
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
    p.add_argument("--xgroup-weights", type=str, default="",
                   help="Comma-separated per-inner-round share of the "
                        "cross-switch transfer, e.g. '2,2,6,0' (the default "
                        "for group size 4, i.e. tenths). Round inner-1 has no "
                        "intra-group copy, so the bulk belongs on round "
                        "inner-2.")
    p.add_argument("--backend", type=str, default="xccl",
                   help="bootstrap process group backend; used only for "
                        "handle exchange, pass-boundary barriers and "
                        "reductions")
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

    The outer ring shifts by a whole group each outer round; the inner ring then
    rotates within that group. Both the group offset and the rank-local index
    are preserved by the outer shift, so:

        b0    = (rank - group_size * g) % world        # block at inner step 0
        block = group_base(b0) + ((local(rank) - s) % group_size)
    """
    local = rank % group_size
    b0 = (rank - group_size * g) % world
    return (b0 - local) + ((local - s) % group_size)


class IpcKVRing:
    """
    所有 ring buffer 都住在一块 IpcArena 里：进程生命周期内只导出 1 个 IPC handle、
    每个对端只 open 1 次。上层每层新建的 k/v tensor 只需要 stage() 进 arena，
    handle 永远不会因为 caching allocator 回收/复用 slab 而变野指针。

    Arena 布局（每段 4KB 对齐）:

        [slot 0][slot 1][slot 2][slot 3][slot 4][slot 5]
         ^base A  ^---- 组内轮换 ----^   ^base B  ^base C   (B/C 仅 two-level)

    每个 slot 内部都是 [K][pad][V]，布局对所有 rank 完全一致，因此跨组分块传输
    可以把整个 slot 当成一段连续内存来切（padding 里的垃圾一起搬走，无害）。

    三个 base slot 轮换让 outer 边界完全不需要同步：见模块 docstring。

    构造函数只分配 arena、交换 handle、打开对端；不做任何 staging，也不预取。
    每个 pass 都必须从 stage() 新的 k/v 开始，跟真实每层 transformer 一致。
    """

    def __init__(self, k_local, v_local, rank, world, queue_ptr,
                 group_size=GROUP_SIZE, xgroup_weights=None):
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
        # Intra-group ring upstream.
        self.prev_rank = group_base + (self.local_id - 1) % self.inner_world
        # Cross-group (cross-switch) upstream: same position, previous group.
        self.prev_group_rank = (rank - group_size) % world

        self.k_shape = list(k_local.shape)
        self.v_shape = list(v_local.shape)
        self.dtype = k_local.dtype
        self.k_nbytes = k_local.numel() * k_local.element_size()
        self.v_nbytes = v_local.numel() * v_local.element_size()

        # slot 内部布局
        self.k_off_in_slot = 0
        self.v_off_in_slot = _align_up(self.k_nbytes)
        self.slot_span = self.v_off_in_slot + self.v_nbytes   # 有效字节
        self.slot_stride = _align_up(self.slot_span)

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

        # 跨组分块计划：按 per-round 权重切分（默认十分之 2/2/6/0），把大头放在
        # 最后一个带组内传输的 round 之后，让它在没有 ring copy 的最后一轮独占
        # copy engine。
        if self.two_level:
            self.xg_weights = list(xgroup_weights) if xgroup_weights \
                else default_xgroup_weights(self.inner_world)
            self.xg_sched = build_chunk_schedule(self.slot_span,
                                                 self.xg_weights)
            covered = sum(n for e in self.xg_sched if e is not None
                          for _, n in [e])
            assert covered == self.slot_span, (
                f"cross-group chunk schedule covers {covered} of "
                f"{self.slot_span} bytes"
            )
        else:
            self.xg_weights = []
            self.xg_sched = []

        # 只交换一次 arena base handle；各 slot 的 offset 所有 rank 一致。
        self.local_handle = self.arena.export_handle()
        gathered = [None] * world
        dist.all_gather_object(gathered, self.local_handle)

        self.peer_handles = []

        self.ring_peer_handle = gathered[self.prev_rank]
        ring_base = ipc.open_peer(self.dev_index, self.ring_peer_handle,
                                  queue_ptr)
        self.peer_handles.append(self.ring_peer_handle)
        self.k_peer_ptr = [ring_base + self.k_off[i] for i in range(n_slots)]
        self.v_peer_ptr = [ring_base + self.v_off[i] for i in range(n_slots)]

        if self.two_level:
            self.xg_peer_handle = gathered[self.prev_group_rank]
            xg_base = ipc.open_peer(self.dev_index, self.xg_peer_handle,
                                    queue_ptr)
            self.peer_handles.append(self.xg_peer_handle)
            self.xg_peer_slot_ptr = [xg_base + self.slot_off[i]
                                     for i in range(n_slots)]
        else:
            self.xg_peer_handle = None
            self.xg_peer_slot_ptr = []

        self._closed = False
        atexit.register(self.close)

        # 任何 copy 发出前，所有 rank 必须已经 open 完对端。
        dist.barrier()
        torch.xpu.synchronize()

    # ------------------------------------------------------------------ #
    # staging
    # ------------------------------------------------------------------ #
    def stage(self, k_new, v_new):
        """
        把这一层新产出的 K/V 写进 slot 0 == base_slot_of(0)。

        slot 0 同时是三个读者的数据源：本 rank inner round 0 的 kernel、组内下游
        rank 的 ring 拉取、跨组下游 rank 的 chunk 拉取。三者共用一份，不需要额外
        的副本。这两次 copy_ 是实际部署时唯一无法回避的本地 D2D。

        这里必须同步：copy_ 排在 compute queue 上，而 copy engine 看不到它。
        同步点必须在“对端可能开始拉取本 rank slot 0”之前。
        """
        self.kbuf[0].copy_(k_new)
        self.vbuf[0].copy_(v_new)
        # 只等当前 stream，不等整个设备；语义足够且比 torch.xpu.synchronize() 轻。
        torch.xpu.current_stream(self.vbuf[0].device).synchronize()

    # ------------------------------------------------------------------ #
    # copy issue helpers
    # ------------------------------------------------------------------ #
    def issue_ring_copy(self, read_slot, write_slot):
        """组内 ring：从 prev_rank 的 read_slot 拉到本地 write_slot。"""
        return [
            ipc.copy_async(self.dev_index,
                           self.base + self.k_off[write_slot],
                           self.k_peer_ptr[read_slot],
                           self.k_nbytes, self.queue_ptr),
            ipc.copy_async(self.dev_index,
                           self.base + self.v_off[write_slot],
                           self.v_peer_ptr[read_slot],
                           self.v_nbytes, self.queue_ptr),
        ]

    def issue_xgroup_chunk(self, off, nbytes, src_slot, dst_slot):
        """
        跨组预取的一块：prev_group_rank 的 src_slot -> 本地 dst_slot，
        [off, off+nbytes)。整个 slot 当成一段连续内存切，K/V 一次搞定。
        """
        if nbytes <= 0:
            return []
        return [
            ipc.copy_async(self.dev_index,
                           self.base + self.slot_off[dst_slot] + off,
                           self.xg_peer_slot_ptr[src_slot] + off,
                           nbytes, self.queue_ptr),
        ]

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
        self.xg_peer_slot_ptr = []
        # arena view 必须先失效，arena 本身由 ipc.shutdown() 释放。
        self.kbuf = []
        self.vbuf = []


def ring_attention_ipc(ring, consume, kv_new):
    """
    One full ring pass over all `world` K/V blocks, starting from the K/V this
    layer produced.

    consume(k_block, v_block, global_round) is called once per block, with
    global_round increasing monotonically 0..world-1 so the LSE-merge epilogue
    initializes exactly once.

    kv_new: (k, v) for this layer. Always staged into the arena before the pass
    starts -- caching-allocator tensors cannot be exported as IPC handles, so
    this local D2D is what upstream code must do every layer.

    The loop below covers both topologies: flat mode is simply n_groups == 1,
    where the outer body degenerates to the original single-level ring
    (base_slot_of(0) == 0, no cross-group traffic).

    Ordering within an inner round: the intra-group ring copy is appended BEFORE
    that round's cross-group chunk. Combined with the default weights, which put
    zero on the last round and the bulk on the last round that HAS a ring copy,
    no ring copy is ever queued behind the large cross-switch piece.

    Outer boundary: with three base slots there is a full outer round of slack
    against the cross-group downstream, so no collective is required anywhere in
    this loop. The only wait is draining our own chunks -- round g+1's first
    kernel reads the slot they filled.
    """
    ring.stage(kv_new[0], kv_new[1])
    dist.barrier()

    inner = ring.inner_world
    for g in range(ring.n_groups):
        prefetch = ring.two_level and g < ring.n_groups - 1
        xg_src = base_slot_of(g)        # 对端本轮的起始块
        xg_dst = base_slot_of(g + 1)    # 本 rank 下一轮的起始块
        xg_pending = []

        for s in range(inner):
            pending = []
            rs = read_slot_of(s, g)

            # 1) 组内 ring copy 先入队 —— in-order list 下这就是优先级。
            #    s == inner-1 没有 ring copy：它要的 KV 已经在手上，于是这一轮
            #    copy engine 全空，正好留给上一轮排下的大块跨组传输。
            if s < inner - 1:
                pending += ring.issue_ring_copy(rs, write_slot_of(s))

            # 2) 再排本轮份额的跨 switch chunk（为 None 即本轮不承担）。
            if prefetch:
                ent = ring.xg_sched[s]
                if ent is not None:
                    xg_pending += ring.issue_xgroup_chunk(ent[0], ent[1],
                                                          xg_src, xg_dst)

            # 3) attention kernel，异步下发，盖住上面两类传输。
            consume(ring.kbuf[rs], ring.vbuf[rs], g * inner + s)

            # 只等组内 ring copy；跨组 chunk 故意跨轮存活，攒到 outer 边界再 drain。
            for h in pending:
                h.wait()

        if prefetch:
            # 唯一的 outer 等待：下一轮 s=0 的 kernel 要读 xg_dst，字节必须到齐。
            # 纯本地 event 等待，不是集合通信 —— 这正是大块被设计去填满的窗口。
            for h in xg_pending:
                h.wait()

    # Only for benchmark
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
    two_level = world > group_size
    if two_level and world % group_size != 0:
        raise RuntimeError(
            f"world size {world} is not a multiple of group size {group_size}; "
            f"supported world sizes are 1/2/4/8/16/32"
        )

    inner_world = group_size if two_level else world
    xgroup_weights = parse_weights(args.xgroup_weights, inner_world) \
        if args.xgroup_weights else None

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
                     group_size=group_size, xgroup_weights=xgroup_weights)

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
        # slot 0,4,5,0，三个 pass 才能把所有起始相位都走一遍。任何 base-slot
        # 记账错误都会在其中一轮暴露。
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
                print(f"[rank {rank}] pass {it} IPC K/V rotation ({mode}): "
                      f"PASSED ({world} rounds, bit-exact)", flush=True)
            dist.barrier()

        ring.close()
        dist.barrier()
        ipc.shutdown()
        dist.destroy_process_group()
        return

    def run_ring():
        # round_idx drives the epilogue: 0 initializes out/lse, >0 LSE-merges.
        # out/lse accumulate across the whole pass and must not be reset.
        # kv_new 直接传本 rank 的 K/V；stage() 里的两次 D2D 就是实际部署时每层
        # 必须付出的开销。
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
        print(f"  Topology           : "
              f"{'two-level (grouped)' if two_level else 'flat ring'}")
        if two_level:
            print(f"  Group size         : {group_size} "
                  f"({ring.n_groups} groups, one per PCIe switch)")
            print("  Intra-group upstr. : rank-1 within group")
            print(f"  Cross-group upstr. : rank-{group_size} (positional)")
            total_w = sum(ring.xg_weights)
            print(f"  X-group weights    : {ring.xg_weights} "
                  f"(share per inner round, /{total_w})")
            for s, ent in enumerate(ring.xg_sched):
                if ent is None:
                    tag = "no ring copy" if s == ring.inner_world - 1 \
                        else "ring copy only"
                    print(f"    round {s}: {tag}, no cross chunk")
                else:
                    off, n = ent
                    print(f"    round {s}: cross chunk "
                          f"{n / 1024**2:8.2f} MiB @ +{off / 1024**2:.2f} MiB "
                          f"({100.0 * n / ring.slot_span:.1f}%)")
            print(f"  Base slots         : {BASE_SLOTS} "
                  f"(3-deep rotation, one outer round of slack)")
            print("  Outer sync         : none (local event drain only)")
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
        print("  Transfer path      : zeCommandListAppendMemoryCopy "
              "(single dedicated copy engine, in-order)")
        print("  Per-loop staging   : 2 copies (K, V) into slot 0 -- the only "
              "unavoidable local D2D")
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
    # 组内 ring：每组 (inner_world - 1) 次；总共 n_groups 个 outer round。
    intra_moved = kv_bytes * (ring.inner_world - 1) * ring.n_groups
    # 跨组：n_groups - 1 次整块（切成 chunk，但字节数相同）。
    inter_moved = (ring.slot_span * (ring.n_groups - 1)) if two_level else 0
    moved_bytes = intra_moved + inter_moved
    # stage() 每 pass 一次本地 D2D（写 + 读），计入有效带宽会更贴近真实。
    staged_bytes = kv_bytes

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
                  f"/ loop / rank (weighted chunk interleave)")
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
