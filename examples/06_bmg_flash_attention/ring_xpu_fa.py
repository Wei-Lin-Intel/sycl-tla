#!/usr/bin/env python3
#
# Ring Attention (sequence-parallel, non-causal) on XPU.
#   mpirun -n <P> python ring_xpu_fa.py --q-seq-len 32768 [...]
#
# Cross-round accumulation reuses the LSE-update (online softmax) path already
# validated by xpu_kv_fa.py (accumulate_output + shared external LSE).

import argparse
import time

import torch
import torch.nn.functional as F
from mpi4py import MPI

import sycl_tla_fmha as fa


def parse_args():
    p = argparse.ArgumentParser(description="Ring Attention on XPU")
    p.add_argument("--q-seq-len", type=int, default=32768,
                   help="GLOBAL sequence length (must be divisible by world_size)")
    p.add_argument("--q-nhead", type=int, default=40)
    p.add_argument("--kv-nhead", type=int, default=40)
    p.add_argument("--qk-hdim", type=int, default=128)
    p.add_argument("--v-hdim", type=int, default=128, choices=(64, 96, 128, 192))
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--loops", type=int, default=10)
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def bshd_strides(t):
    # t: contiguous [1, S, H, D] -> BHSD-view strides (s, h, b)
    v = t.permute(0, 2, 1, 3)
    return [v.stride(2), v.stride(1), v.stride(0)]


def lse_strides(lse):
    # lse: [1, H, S] -> (q, h, b)
    return [lse.stride(2), lse.stride(1), lse.stride(0)]


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()

    assert torch.xpu.is_available(), "no XPU"
    torch.xpu.set_device(rank)          # one rank per device
    device = torch.device("xpu", rank)
    dtype = torch.bfloat16

    assert args.q_seq_len % world == 0, "--q-seq-len must be divisible by world_size"
    s_local = args.q_seq_len // world
    Hq, Hkv = args.q_nhead, args.kv_nhead
    Dqk, Dvo = args.qk_hdim, args.v_hdim
    assert Hq % Hkv == 0 and Dqk % 32 == 0

    torch.manual_seed(args.seed)
    q_full = torch.randn(1, args.q_seq_len, Hq, Dqk, device=device, dtype=dtype)
    k_full = torch.randn(1, args.q_seq_len, Hkv, Dqk, device=device, dtype=dtype)
    v_full = torch.randn(1, args.q_seq_len, Hkv, Dvo, device=device, dtype=dtype)

    def shard(t):
        return t[:, rank * s_local:(rank + 1) * s_local].contiguous()

    q = shard(q_full)          # local Q shard (never rotates)
    k_local = shard(k_full)    # this rank's own K block (#rank)
    v_local = shard(v_full)

    ring = fa.RingSymmMemory(seq_kv_local=s_local, h_kv=Hkv,
                             d_qk=Dqk, d_vo=Dvo, rank=rank, world_size=world)
    dst = (rank + 1) % world

    out = torch.empty(1, s_local, Hq, Dvo, device=device, dtype=dtype)
    lse = torch.empty(1, Hq, s_local, device=device, dtype=torch.float32)

    qS, oS, lseS = bshd_strides(q), bshd_strides(out), lse_strides(lse)
    kS, vS = bshd_strides(k_local), bshd_strides(v_local)

    def run_ring_once():
        #out = torch.empty(1, s_local, Hq, Dvo, device=device, dtype=dtype)
        ring.load_local_kv(k_local.data_ptr(), v_local.data_ptr())
        comm.Barrier()
        for t in range(world):
            cur, nxt = t % 2, (t + 1) % 2
            enabled = (t + 1 < world)
            # Round 0 loads real K/V from global memory (via k_ptr/v_ptr) and,
            # when enabled, pushes the post-reorder MMA-B fragments to the
            # next rank's buf[nxt] (see mainloop's ring_enabled path).
            # Round 1..N-1 must NOT re-load+reorder from global memory:
            # this rank's own buf[cur] was populated by the *previous*
            # round's peer push, already in MMA-B fragment layout. Setting
            # ring_consume=True makes the mainloop pull tSrK/tArV straight
            # from recv_k_ptr/recv_v_ptr instead (see mainloop ring_consume
            # path). k_ptr/v_ptr are still passed for shape/stride purposes
            # but their contents are not read on the consume path.
            consume = t > 0
            fa.prefill_bf16_ring_round(
                q_ptr=q.data_ptr(),
#                k_ptr=ring.local_k(cur), v_ptr=ring.local_v(cur),
                # Round 0 reads the real packed KV (this rank's own shard).
                # Rounds >=1 read from buf[cur] in fragment layout via
                # ring_consume; k_ptr/v_ptr are then ignored by the kernel.
                k_ptr=(k_local.data_ptr() if t == 0 else ring.local_k(cur)),
                v_ptr=(v_local.data_ptr() if t == 0 else ring.local_v(cur)),
                o_ptr=out.data_ptr(), lse_ptr=lse.data_ptr(),
                seq_len_qo=s_local, seq_len_kv=s_local,
                num_heads_q=Hq, num_heads_kv=Hkv,
                head_size_qk=Dqk, head_size_vo=Dvo,
                round_idx=t, ring_enabled=enabled,
                peer_k_ptr=ring.remote_k(dst, nxt) if enabled else 0,
                peer_v_ptr=ring.remote_v(dst, nxt) if enabled else 0,
                ring_consume=consume,
                recv_k_ptr=ring.local_k(cur) if consume else 0,
                recv_v_ptr=ring.local_v(cur) if consume else 0,
                q_strides=qS, k_strides=kS, v_strides=vS,
                o_strides=oS, lse_strides=lseS)
            torch.xpu.synchronize()   # compute (>= comm) done => P2P done
            ring.barrier(0)           # confirm peers finished writing
            comm.Barrier()            # round boundary (the only host lock)

    for _ in range(args.warmup):
        run_ring_once()
    torch.xpu.synchronize()
    comm.Barrier()

    start = time.perf_counter()
    for _ in range(args.loops):
        run_ring_once()
    torch.xpu.synchronize()
    comm.Barrier()
    elapsed = time.perf_counter() - start
    avg_ms = elapsed / args.loops * 1e3
    flops = 2 * 1 * Hq * s_local * args.q_seq_len * (Dqk + Dvo)
    tflops = flops / (elapsed / args.loops) / 1e12
    if rank == 0:
        print(f"[ring] world={world} S_global={args.q_seq_len} S_local={s_local}")
        print(f"[ring] avg {avg_ms:.3f} ms  {tflops:.2f} TFLOPs/rank")

    if not args.skip_verify:
        with torch.no_grad():
            qb = q.transpose(1, 2)
            kb, vb = k_full.transpose(1, 2), v_full.transpose(1, 2)
            if qb.size(1) != kb.size(1):
                g = qb.size(1) // kb.size(1)
                kb = kb.repeat_interleave(g, dim=1)
                vb = vb.repeat_interleave(g, dim=1)
            ref = F.scaled_dot_product_attention(qb, kb, vb, is_causal=False)
            ref = ref.transpose(1, 2).contiguous()
        torch.xpu.synchronize()
        torch.testing.assert_close(out.float(), ref.float(), atol=5e-2, rtol=5e-2)
        md = (out.float() - ref.float()).abs().max().item()
        print(f"[ring][rank {rank}] PASSED  max_abs_diff={md:.3e}")


if __name__ == "__main__":
    main()
