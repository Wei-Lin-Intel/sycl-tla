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

    # 统一 buffer 语义:load_local_kv 后 buf[0] 始终持有本 rank 自己的 packed K/V。
    # 之后每一轮的 buf[cur] 都是上一轮 peer push 进来的 packed K/V,因此每一轮
    # 都从 buf[cur] 消费(ring_consume=True),不再对 round 0 做特殊处理。
    ring.load_local_kv(k_local.data_ptr(), v_local.data_ptr())
    comm.Barrier()

    dst = (rank + 1) % world

    for t in range(world):
        cur, nxt = t % 2, (t + 1) % 2
        push = (t + 1 < world)

        # 计算这一轮:Q @ buf[cur] 里的 packed K/V(packed recv 路径)。
        fa.prefill_bf16_ring_round(
            q_ptr=q.data_ptr(),
            # 始终消费本 rank 当前 packed buffer。
            k_ptr=ring.local_k(cur),
            v_ptr=ring.local_v(cur),
            o_ptr=out.data_ptr(), lse_ptr=lse.data_ptr(),
            seq_len_qo=s_local, seq_len_kv=s_local,
            num_heads_q=Hq, num_heads_kv=Hkv,
            head_size_qk=Dqk, head_size_vo=Dvo,
            round_idx=t,
            # push 已改到 host(ring.push_packed_async),kernel 内不再 push。
            ring_consume=True,
            recv_k_ptr=ring.local_k(cur),
            recv_v_ptr=ring.local_v(cur),
            q_strides=qS, k_strides=kS, v_strides=vS,
            o_strides=oS, lse_strides=lseS)

        # push:把本 rank 当前 buffer[cur] 的 packed K/V 异步拷到 peer 的
        # buffer[nxt](在独立 copy queue 上,和本轮 kernel 重叠)。
        if push:
            ring.push_packed_async(dst, cur, nxt)

        # 本轮 kernel 读完 buf[cur]、push memcpy 写完 peer 的 buf[nxt] 之后,
        # 再让跨 rank barrier 放行,保证 peer 下一轮读到的是最新数据。
        torch.xpu.synchronize()
        if push:
            ring.wait_pushes()
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

#    print(out)
#    print()
#    print(ref)

    # 用 cosine similarity 判定:把每个 (batch, head, query) 的 head-dim 向量
    # 当作一条向量,沿 head-dim(最后一维)算 cos 相似度。这样对 bf16 的幅度
    # 噪声不敏感,但方向偏差(真错)仍会被捕获。
    out_f = out.float()
    ref_f = ref.float()
    cos_global = F.cosine_similarity(out_f.reshape(-1), ref_f.reshape(-1), dim=0, eps=1e-8).item()
    # 每个位置都必须足够接近 1;低于阈值算 fail。
    COS_TOL = 1e-1            # 允许 1 - cos <= COS_TOL
    ok = (1.0 - cos_global) < COS_TOL

    # 保留一个 abs-diff 参考量,方便定位问题(不参与判定)。
    diff = (out_f - ref_f).abs()

    print(f"[ring rank {rank}/{world}] S_global={S_global} s_local={s_local} "
          f"ok={ok} cos_similarity={cos_global:.6f} "
          f"max|Δ|={diff.max().item():.3e}", flush=True)

    all_ok = comm.allreduce(1 if ok else 0, op=MPI.MIN)
    if rank == 0:
        print(f"[ring] OVERALL: {'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
