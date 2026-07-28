import argparse
import torch
import torch.nn.functional as F
import numpy as np
from mpi4py import MPI
import sycl_tla_fmha as fa


def bshd_strides(t):
    v = t.permute(0, 2, 1, 3)
    return [v.stride(2), v.stride(1), v.stride(0)]

def lse_strides(l):
    # 必须和 kv_list 路径一致:{stride(1), stride(2), stride(0)}
    return [l.stride(1), l.stride(2), l.stride(0)]


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

    # 所有 rank 用同一份全局数据(CPU 生成 + 广播,保证 bit 一致)
    torch.manual_seed(a.seed)
    if rank == 0:
        q_full = torch.randn(1, S_global, Hq,  Dqk, dtype=torch.float32)
        k_full = torch.randn(1, S_global, Hkv, Dqk, dtype=torch.float32)
        v_full = torch.randn(1, S_global, Hkv, Dvo, dtype=torch.float32)
    else:
        q_full = torch.empty(1, S_global, Hq,  Dqk, dtype=torch.float32)
        k_full = torch.empty(1, S_global, Hkv, Dqk, dtype=torch.float32)
        v_full = torch.empty(1, S_global, Hkv, Dvo, dtype=torch.float32)

    def bcast(t):
        arr = np.ascontiguousarray(t.numpy())
        comm.Bcast(arr, root=0)
        return torch.from_numpy(arr).to(dt).to(dev)

    q_full = bcast(q_full)
    k_full = bcast(k_full)
    v_full = bcast(v_full)

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

    print(f"[rank {rank}] before RingSymmMemory", flush=True); comm.Barrier()
    ring = fa.RingSymmMemory(seq_kv_local=s_local, h_kv=Hkv,
                             d_qk=Dqk, d_vo=Dvo, rank=rank, world_size=world)
    print(f"[rank {rank}] after ctor", flush=True); comm.Barrier()

    # 把本 rank 的 packed K/V 放进 compute buffer 0(round 0 也可直接用原 tensor)
    ring.load_local_kv(k_local.data_ptr(), v_local.data_ptr())
    comm.Barrier()

    dst = (rank + 1) % world

    for t in range(world):
        cur, nxt = t % 2, (t + 1) % 2
        consume = (t > 0)

        # 计算这一轮:Q @ (round0=原始 K/V, round>0=recv buffer[cur] 的 packed K/V)
        fa.prefill_bf16_ring_round(
            q_ptr=q.data_ptr(),
            k_ptr=(k_local.data_ptr() if t == 0 else ring.local_k(cur)),
            v_ptr=(v_local.data_ptr() if t == 0 else ring.local_v(cur)),
            o_ptr=out.data_ptr(), lse_ptr=lse.data_ptr(),
            seq_len_qo=s_local, seq_len_kv=s_local,
            num_heads_q=Hq, num_heads_kv=Hkv,
            head_size_qk=Dqk, head_size_vo=Dvo,
            round_idx=t,
            # push 已改到 host(ring.push_packed),kernel 内不再 push:占位
            ring_enabled=False, peer_k_ptr=0, peer_v_ptr=0,
            ring_consume=consume,
            recv_k_ptr=(ring.local_k(cur) if consume else 0),
            recv_v_ptr=(ring.local_v(cur) if consume else 0),
            q_strides=qS, k_strides=kS, v_strides=vS,
            o_strides=oS, lse_strides=lseS)
        torch.xpu.synchronize()

        # push:把本 rank 当前 buffer[cur] 的 packed K/V 拷到 peer 的 buffer[nxt]
        if t + 1 < world:
            ring.push_packed(dst, cur, nxt)
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

    print(out)
    print()
    print(ref)
    diff = (out.float() - ref.float()).abs()
    over = (diff > (5e-2 + 5e-2 * ref.float().abs()))
    over_ratio = over.float().mean().item()
    ok = over_ratio < 1e-3
    print(f"[ring rank {rank}/{world}] S_global={S_global} s_local={s_local} "
          f"ok={ok} max|Δ|={diff.max().item():.3e} "
          f"mean|Δ|={diff.mean().item():.5f} >tol比例={over_ratio:.5f}", flush=True)

    all_ok = comm.allreduce(1 if ok else 0, op=MPI.MIN)
    if rank == 0:
        print(f"[ring] OVERALL: {'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
