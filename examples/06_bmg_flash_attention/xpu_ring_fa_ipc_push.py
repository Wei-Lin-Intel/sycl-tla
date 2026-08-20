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

The receive arena uses at most three slots. Transfers are assigned monotonically
increasing tickets and slots are selected by ticket modulo the buffer count.
The sender publishes ready after both K/V CE writes complete. Before reusing a
slot, the receiver waits for the attention kernel that consumed the old value,
publishes free, and the sender waits for that free ticket before overwriting the
peer slot.

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
import hashlib
import os
import time

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

import sycl_tla_fmha as fa
import sycl_tla_ipc_p2p as ipc

SLOT_ALIGN = 4096
MAX_CONTROL_TICKET = (1 << 32) - 1


def _align_up(v, a=SLOT_ALIGN):
    return (v + a - 1) // a * a


def slot_of_ticket(ticket, num_buffers):
    if ticket <= 0:
        raise ValueError(f"ticket must be positive, got {ticket}")
    return (ticket - 1) % num_buffers


def ring_control_name(world):
    """
    Build a node-local launch-specific POSIX shm name.

    torchrun normally provides TORCHELASTIC_RUN_ID and MASTER_PORT. Hashing the
    complete token also removes characters that are illegal in shm_open names.
    """
    token = "|".join([
        os.environ.get("TORCHELASTIC_RUN_ID", ""),
        os.environ.get("MASTER_ADDR", "localhost"),
        os.environ.get("MASTER_PORT", "29500"),
        str(world),
        os.environ.get("USER", ""),
    ])
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]
    return f"/sycl_tla_ring_{digest}"


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
    p.add_argument("--num-buffers", type=int, default=3, choices=(2, 3),
                   help="Maximum number of IPC receive buffers. Three is the "
                        "recommended default; world=2 automatically uses one.")
    p.add_argument("--control-timeout-ms", type=int, default=30000,
                   help="Timeout for shared-memory ready/free/teardown waits; "
                        "a timeout raises instead of hanging indefinitely")
    p.add_argument("--profile", action="store_true",
                   help="Dump a PyTorch profiler trace over the timed loops")
    p.add_argument("--profile-dir", type=str, default="./profiler_out",
                   help="Directory to write profiler traces into")
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def init_process():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
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
    All received K/V blocks live in one persistent IpcArena.

    Initial local K/V stays in the caller-owned tensors. Received blocks rotate
    through at most three arena slots. ready/free ticket counters protect each
    slot from being read before arrival or overwritten before compute finishes.
    """

    def __init__(
        self,
        k_local,
        v_local,
        rank,
        world,
        queue_ptr,
        control_timeout_ms,
        num_buffers,
    ):
        self.rank = rank
        self.world = world
        if world == 2:
            self.num_buffers = 1
        else:
            if num_buffers not in (2, 3):
                raise ValueError(
                    "num_buffers must be 2 or 3 when world > 2"
                )
            self.num_buffers = min(num_buffers, world - 1)

        self.control_timeout_ms = control_timeout_ms
        self.queue_ptr = queue_ptr
        self.dev_index = k_local.device.index
        self.next_rank = (rank + 1) % world
        self.prev_rank = (rank - 1) % world

        self.k_shape = list(k_local.shape)
        self.v_shape = list(v_local.shape)
        self.dtype = k_local.dtype
        self.k_nbytes = k_local.numel() * k_local.element_size()
        self.v_nbytes = v_local.numel() * v_local.element_size()

        # Arena layout: slot0[K][V], slot1[K][V], ...
        self.k_off_in_slot = 0
        self.v_off_in_slot = _align_up(self.k_nbytes)
        self.slot_stride = _align_up(self.v_off_in_slot + self.v_nbytes)
        total = self.slot_stride * self.num_buffers

        self.arena = ipc.make_arena(self.dev_index, total, queue_ptr)

        self.k_off = [i * self.slot_stride + self.k_off_in_slot
                      for i in range(self.num_buffers)]
        self.v_off = [i * self.slot_stride + self.v_off_in_slot
                      for i in range(self.num_buffers)]
        self.kbuf = [self.arena.view(o, self.k_shape, self.dtype)
                     for o in self.k_off]
        self.vbuf = [self.arena.view(o, self.v_shape, self.dtype)
                     for o in self.v_off]

        # For every local receive slot, track the ticket currently being
        # consumed and an event recorded after its attention launch.
        self.slot_compute_done = [None] * self.num_buffers

        self.control = ipc.RingControl(
            name=ring_control_name(world),
            rank=rank,
            world=world,
            slots=self.num_buffers,
            handle_bytes=64,
            timeout_ms=control_timeout_ms,
        )

        # Publish first, then wait for next_rank. Publishing first prevents a
        # circular bootstrap dependency.
        self.local_handle = self.arena.export_handle()
        self.control.publish_handle(self.local_handle)
        self.peer_handle = self.control.wait_handle(self.next_rank)

        peer_base = ipc.open_peer(self.dev_index, self.peer_handle, queue_ptr)
        self.k_peer_ptr = [peer_base + o for o in self.k_off]
        self.v_peer_ptr = [peer_base + o for o in self.v_off]

        self._closed = False
        atexit.register(self.close)

        # Bootstrap-only rendezvous: all peer mappings must be open before any
        # process can issue its first CE write.
        self.control.barrier(1)

    def record_local_consumer(self, slot, ticket):
        state = self.slot_compute_done[slot]
        if state is not None:
            old_ticket, _ = state
            raise RuntimeError(
                f"rank {self.rank}: slot {slot} still tracks ticket "
                f"{old_ticket}; cannot record ticket {ticket}"
            )

        done = torch.xpu.Event()
        done.record(torch.xpu.current_stream())
        self.slot_compute_done[slot] = (ticket, done)

    def release_local_slot(self, slot, expected_ticket):
        """
        Wait until local compute has stopped reading a slot, then publish the
        free ticket so prev_rank may overwrite it.

        The forwarding CE read of this slot is already complete: every ring
        round waits for its outgoing K/V PendingCopy objects before advancing.
        """
        state = self.slot_compute_done[slot]
        if state is None:
            raise RuntimeError(
                f"rank {self.rank}: slot {slot} has no compute event; "
                f"expected ticket {expected_ticket}"
            )

        ticket, done = state
        if ticket != expected_ticket:
            raise RuntimeError(
                f"rank {self.rank}: slot {slot} tracks ticket {ticket}, "
                f"expected ticket {expected_ticket}"
            )

        deadline = (
            time.monotonic() + self.control_timeout_ms / 1000.0
        )
        while not done.query():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"rank {self.rank}: timed out waiting for compute "
                    f"to release slot {slot}, ticket {ticket}, "
                    f"timeout_ms={self.control_timeout_ms}"
                )

            # Yield the host thread without adding a millisecond-scale delay.
            time.sleep(0)

        self.slot_compute_done[slot] = None
        self.control.publish_free(self.rank, slot, ticket)

    def close(self):
        if self._closed:
            return
        self._closed = True
        ipc.close_peer(self.dev_index, self.peer_handle, self.queue_ptr)
        self.k_peer_ptr = []
        self.v_peer_ptr = []
        # Views must be invalidated before ipc.shutdown() frees the arena.
        self.kbuf = []
        self.vbuf = []


def ring_attention_ipc(ring, consume, kv_new, epoch):
    """
    One ring pass using a globally continuous transfer-ticket sequence.

    For pass epoch E and world W:

      pass_base = (E - 1) * (W - 1)

    Outgoing step s has ticket pass_base+s+1. The data consumed by attention
    step s>0 arrived with ticket pass_base+s.

    A slot is ticket-modulo-num_buffers. Before ticket T overwrites its slot,
    every rank first releases its own old ticket T-num_buffers, then waits for
    next_rank to release the corresponding destination slot. Publishing local
    free before waiting for peer free prevents a circular host-side wait.
    """

    if epoch <= 0:
        raise ValueError(f"epoch must be positive, got {epoch}")

    transfers_per_pass = ring.world - 1
    pass_base = (epoch - 1) * transfers_per_pass
    last_ticket = pass_base + transfers_per_pass
    if last_ticket > MAX_CONTROL_TICKET:
        raise OverflowError(
            "ring transfer ticket exceeds the uint32 shared-control range: "
            f"last_ticket={last_ticket}"
        )

    k_src0, v_src0 = kv_new

    # The copy-engine queue is independent of the current PyTorch/SYCL stream.
    # Keep this synchronization until producer->CE ordering is represented by
    # an explicit cross-queue event/dependency.
    torch.xpu.current_stream(v_src0.device).synchronize()

    for step in range(ring.world):
        if step == 0:
            source_ticket = None
            read_slot = None
            src_k_tensor = k_src0
            src_v_tensor = v_src0
            src_k_ptr = k_src0.data_ptr()
            src_v_ptr = v_src0.data_ptr()
        else:
            # Incoming transfer step-1 produced the data for attention step.
            source_ticket = pass_base + step
            read_slot = slot_of_ticket(
                source_ticket, ring.num_buffers
            )

            # prev_rank publishes only after both K and V CE writes complete.
            ring.control.wait_ready(
                ring.rank, read_slot, source_ticket
            )

            src_k_tensor = ring.kbuf[read_slot]
            src_v_tensor = ring.vbuf[read_slot]
            base = ring.arena.base_ptr()
            src_k_ptr = base + ring.k_off[read_slot]
            src_v_ptr = base + ring.v_off[read_slot]

        pending = []
        if step < ring.world - 1:
            ticket = pass_base + step + 1
            write_slot = slot_of_ticket(
                ticket, ring.num_buffers
            )
            previous_ticket = ticket - ring.num_buffers

            if previous_ticket > 0:
                # Critical deadlock-avoidance ordering:
                #
                #   1. release our local slot to prev_rank;
                #   2. wait until next_rank releases its slot to us.
                #
                # If every rank waited first, the rank ring could deadlock.
                ring.release_local_slot(
                    write_slot,
                    expected_ticket=previous_ticket,
                )
                ring.control.wait_free(
                    ring.next_rank,
                    write_slot,
                    previous_ticket,
                )

            pending.append(ipc.copy_async(
                ring.dev_index, ring.k_peer_ptr[write_slot], src_k_ptr,
                ring.k_nbytes, ring.queue_ptr))
            pending.append(ipc.copy_async(
                ring.dev_index, ring.v_peer_ptr[write_slot], src_v_ptr,
                ring.v_nbytes, ring.queue_ptr))

        # Local attention and outgoing CE transfer are both readers of the same
        # K/V block, so they may safely overlap.
        consume(src_k_tensor, src_v_tensor, step)

        if source_ticket is not None:
            # Record immediately after enqueueing attention. On an in-order
            # compute stream, event completion means the kernel no longer reads
            # this local receive slot.
            ring.record_local_consumer(
                read_slot, source_ticket
            )

        for h in pending:
            h.wait()

        if pending:
            # Publish only after both remote K and V writes have completed.
            ring.control.publish_ready(
                ring.next_rank, write_slot, ticket
            )


def shutdown_ring(ring):
    """
    Tear down IPC mappings without torch.distributed synchronization.

    Phase 1: all ranks stop compute/copy work.
    Phase 2: each rank closes its mapping to next_rank.
    Phase 3: each arena owner waits until prev_rank closed the mapping to it.
    """
    torch.xpu.synchronize()

    ring.control.publish_work_done(ring.rank)
    ring.control.wait_work_done(ring.next_rank)

    ring.close()

    # mapping_closed[r] means rank r closed its outgoing mapping to r+1.
    ring.control.publish_mapping_closed(ring.rank)
    ring.control.wait_mapping_closed(ring.prev_rank)

    ipc.shutdown()
    ring.control.close(unlink_name=(ring.rank == 0))


def main():
    args = parse_args()

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("No available XPU device was detected")

    rank, world, local_rank = init_process()
    if world < 2:
        if rank == 0:
            print(f"[SKIP] need world >= 2, got {world}")
        return

    if args.control_timeout_ms <= 0:
        raise ValueError("--control-timeout-ms must be positive")

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
    ring = IpcKVRing(
        k_local, v_local, rank, world, queue_ptr,
        args.control_timeout_ms,
        args.num_buffers,
    )
    next_epoch = 1

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

        ring_attention_ipc(
            ring, check, kv_new=(k_local, v_local), epoch=next_epoch
        )
        torch.xpu.synchronize()
        if failures:
            for step, which, bad in failures:
                print(f"[rank {rank}] round {step} {which}: FAILED, "
                      f"{bad} elements differ", flush=True)
        else:
            print(f"[rank {rank}] IPC K/V rotation: PASSED "
                  f"({world} rounds, bit-exact)", flush=True)

        shutdown_ring(ring)
        return

    def run_ring():
        nonlocal next_epoch
        # round_idx drives the epilogue: 0 initializes out/lse, >0 LSE-merges.
        # out/lse accumulate across the whole pass and must not be reset.
        ring_attention_ipc(
            ring,
            lambda kb, vb, step: fa.prefill_bf16_bshd_kv_round(
                q=q, k=kb, v=vb, out=out, lse=lse, round_idx=step,
            ),
            kv_new=(k_local, v_local),
            epoch=next_epoch,
        )
        next_epoch += 1

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
        print(f"  Ring buffers       : {ring.num_buffers} x2 "
              f"(cyclic destination slots, K and V)")
        print(f"  Buffer limit       : {args.num_buffers} "
              f"(effective {ring.num_buffers})")
        print("  Control plane      : POSIX shm + futex "
              "(no torch.distributed scheduling)")
        print(f"  Control timeout    : {args.control_timeout_ms} ms")
        print("  Transfer path      : CE write to peer "
              "(zeCommandListAppendMemoryCopy, dedicated copy engine)")
        print("  Initial local K/V  : caller tensors outside the IPC arena")
        print(f"  Verify             : {not args.skip_verify}")
        print(f"  Profile            : {args.profile}")
        print(f"  Warmup loops       : {args.warmup}")
        print(f"  Benchmark loops    : {args.loops}")

    # Warmup.
    for _ in range(args.warmup):
        run_ring()
    torch.xpu.synchronize()
    ring.control.barrier(2)

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
    elapsed_ns = int(elapsed_seconds * 1_000_000_000)

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

    # Throughput is bounded by the slowest rank. This replaces dist.all_reduce
    # with one shared-memory publication per rank, outside the timed interval.
    elapsed_result_epoch = 1
    ring.control.publish_elapsed_ns(
        rank, elapsed_ns, elapsed_result_epoch
    )
    elapsed_max_ns = ring.control.wait_max_elapsed_ns(elapsed_result_epoch)
    elapsed_max = elapsed_max_ns / 1_000_000_000.0

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
            out.float(), ref.float(), atol=5e-3, rtol=5e-3
        )
        max_abs_diff = (out.float() - ref.float()).abs().max().item()
        if rank == 0:
            print("\nValidation")
            print("  Reference          : F.scaled_dot_product_attention")
            print("  Status             : PASSED")
            print("  Tolerance          : atol=5e-3, rtol=5e-3")
        print(f"  [rank {rank}] max abs diff : {max_abs_diff:.6e}", flush=True)

    shutdown_ring(ring)


if __name__ == "__main__":
    main()
