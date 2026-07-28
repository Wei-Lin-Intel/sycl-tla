#!/usr/bin/env python3
# world>=2 真实 ring attention,用 RingSymmMemory(内部 IPC 走 base+offset +
# socket 传 fd,已验证正确)。buffer 大小由 RingSymmMemory 按 frag=128 分配。
#
# 用法:
#   mpirun -n 2 python3 test_ring_multi_rank_symm.py --q-seq-len 128 --q-nhead 1 --kv-nhead 1
#   mpirun -n 4 python3 test_ring_multi_rank_symm.py --q-seq-len 2048 --q-nhead 16

import argparse
import torch
import torch.nn.functional as F
from mpi4py import MPI
import sycl_tla_fmha as fa


def bshd_strides(t):
    v = t.permute(0, 2, 1, 3)
    return [v.stride(2), v.stride(1), v.stride(0)]

def lse_strides(l):
    return [l.stride(2), l.stride(1), l.stride(0)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--q-seq-len", type=int, default=128)
    p.add_argument("--q-nhead",  type=int, default=1)
    p.add_argument("--kv-nhead", type=int, default=1)
    p.add_argument("--qk-hdim",  type=int, default=128)
    p.add_argument("--v-hdim",   type=int, default=128, choices=(64, 96, 128, 192))
    p.add_argument("--seed",     type=int, default=2026)
    return p.parse_args()


def main():
    a = parse_args()
    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    if world < 2:
        if rank == 0:
            print(f"[SKIP] 需要 world>=2,当前 {world}")
        return

    torch.xpu.set_device(rank)
    dev = torch.device("xpu", rank)
    dt = torch.bfloat16

    S_global = a.q_seq_len
    assert S_global % world == 0
    s_local = S_global // world
    Hq, Hkv, Dqk, Dvo = a.q_nhead, a.kv_nhead, a.qk_hdim, a.v_hdim
    assert Hq % Hkv == 0 and Dqk % 32 == 0
    assert s_local % 32 == 0, "s_local 必须是 TileK(32) 的倍数"

    torch.manual_seed(a.seed)   # 所有 rank 同 seed => 同一份全局张量
    q_full = torch.randn(1, S_global, Hq,  Dqk, device=dev, dtype=dt)
    k_full = torch.randn(1, S_global, Hkv, Dqk, device=dev, dtype=dt)
    v_full = torch.randn(1, S_global, Hkv, Dvo, device=dev, dtype=dt)

    def shard(t):
        return t[:, rank * s_local:(rank + 1) * s_local].contiguous()

    q       = shard(q_full)
    k_local = shard(k_full)
    v_local = shard(v_full)

    qS = bshd_strides(q)
    kS = bshd_strides(k_local)
    vS = bshd_strides(v_local)

    out = torch.empty(1, s_local, Hq, Dvo, device=dev, dtype=dt)
    lse = torch.empty(1, Hq, s_local, device=dev, dtype=torch.float32)
    oS, lseS = bshd_strides(out), lse_strides(lse)

    # RingSymmMemory 内部按 frag=128 分配 buffer(见 py_fmha_module 的 kFrag)。
    ring = fa.RingSymmMemory(seq_kv_local=s_local, h_kv=Hkv,
                             d_qk=Dqk, d_vo=Dvo, rank=rank, world_size=world)
    ring.load_local_kv(k_local.data_ptr(), v_local.data_ptr())
    comm.Barrier()
    dst = (rank + 1) % world

    for t in range(world):
        cur, nxt = t % 2, (t + 1) % 2
        enabled = (t + 1 < world)
        consume = (t > 0)
        fa.prefill_bf16_ring_round(
            q_ptr=q.data_ptr(),
            k_ptr=(k_local.data_ptr() if t == 0 else ring.local_k(cur)),
            v_ptr=(v_local.data_ptr() if t == 0 else ring.local_v(cur)),
            o_ptr=out.data_ptr(), lse_ptr=lse.data_ptr(),
            seq_len_qo=s_local, seq_len_kv=s_local,
            num_heads_q=Hq, num_heads_kv=Hkv,
            head_size_qk=Dqk, head_size_vo=Dvo,
            round_idx=t, ring_enabled=enabled,
            peer_k_ptr=(ring.remote_k(dst, nxt) if enabled else 0),
            peer_v_ptr=(ring.remote_v(dst, nxt) if enabled else 0),
            ring_consume=consume,
            recv_k_ptr=(ring.local_k(cur) if consume else 0),
            recv_v_ptr=(ring.local_v(cur) if consume else 0),
            q_strides=qS, k_strides=kS, v_strides=vS,
            o_strides=oS, lse_strides=lseS)
        torch.xpu.synchronize()
        ring.barrier(0)
        comm.Barrier()

    # 参考:本地 Q 对 GLOBAL K/V 全量 SDPA
    with torch.no_grad():
        qb = q.transpose(1, 2)
        kb = k_full.transpose(1, 2)
        vb = v_full.transpose(1, 2)
        if qb.size(1) != kb.size(1):
            g = qb.size(1) // kb.size(1)
            kb, vb = kb.repeat_interleave(g, 1), vb.repeat_interleave(g, 1)
        ref = F.scaled_dot_product_attention(qb, kb, vb, is_causal=False).transpose(1, 2).contiguous()
    torch.xpu.synchronize()

    diff = (out.float() - ref.float()).abs()
    ok = torch.allclose(out.float(), ref.float(), atol=5e-2, rtol=5e-2)
    print(out)
    print()
    print(ref)
    print(f"[ring rank {rank}/{world}] S_global={S_global} s_local={s_local} "
          f"ok={ok} max|Δ|={diff.max().item():.3e}")

    all_ok = comm.allreduce(1 if ok else 0, op=MPI.MIN)
    if rank == 0:
        print(f"[ring] OVERALL: {'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
