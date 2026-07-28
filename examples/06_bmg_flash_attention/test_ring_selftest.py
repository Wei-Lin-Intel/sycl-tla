#!/usr/bin/env python3
"""
world_size==1 self-loopback 精度测试:不使用 RingSymmMemory(避开 IPC-to-self),
用普通 torch device buffer 做本地 push -> pull -> gemm。

  baseline : ring_enabled=False, ring_consume=False
  selftest : round A 把 K/V fragment push 到本地 buf,
             round B 用 ring_consume 从同一 buf pull 回来做 gemm

push/pull 正确 => selftest 与 baseline bit-identical。

运行: mpirun -n 1 python test_ring_selftest_rank1.py
"""

import argparse
import torch
import torch.nn.functional as F
from mpi4py import MPI
import sycl_tla_fmha as fa


def bshd_strides(t):
    v = t.permute(0, 2, 1, 3)
    return [v.stride(2), v.stride(1), v.stride(0)]

def lse_strides(lse):
    return [lse.stride(2), lse.stride(1), lse.stride(0)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--q-seq-len", type=int, default=1024)
    p.add_argument("--q-nhead", type=int, default=16)
    p.add_argument("--kv-nhead", type=int, default=16)
    p.add_argument("--qk-hdim", type=int, default=128)
    p.add_argument("--v-hdim", type=int, default=128, choices=(64, 96, 128, 192))
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def one_round(q, k_ptr, v_ptr, out, lse, S, Hq, Hkv, Dqk, Dvo,
              qS, kS, vS, oS, lseS,
              ring_enabled=False, peer_k=0, peer_v=0,
              ring_consume=False, recv_k=0, recv_v=0):
    fa.prefill_bf16_ring_round(
        q_ptr=q.data_ptr(),
        k_ptr=k_ptr, v_ptr=v_ptr,
        o_ptr=out.data_ptr(), lse_ptr=lse.data_ptr(),
        seq_len_qo=S, seq_len_kv=S,
        num_heads_q=Hq, num_heads_kv=Hkv,
        head_size_qk=Dqk, head_size_vo=Dvo,
        round_idx=0, ring_enabled=ring_enabled,
        peer_k_ptr=peer_k, peer_v_ptr=peer_v,
        ring_consume=ring_consume,
        recv_k_ptr=recv_k, recv_v_ptr=recv_v,
        q_strides=qS, k_strides=kS, v_strides=vS,
        o_strides=oS, lse_strides=lseS)
    torch.xpu.synchronize()


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    if world != 1:
        if rank == 0:
            print(f"[SKIP] 需要 world_size==1,当前 {world}")
        return

    assert torch.xpu.is_available(), "no XPU"
    torch.xpu.set_device(rank)
    device = torch.device("xpu", rank)
    dtype = torch.bfloat16

    S = args.q_seq_len
    Hq, Hkv = args.q_nhead, args.kv_nhead
    Dqk, Dvo = args.qk_hdim, args.v_hdim
    assert Hq % Hkv == 0 and Dqk % 32 == 0

    torch.manual_seed(args.seed)
    q = torch.randn(1, S, Hq, Dqk, device=device, dtype=dtype)
    k = torch.randn(1, S, Hkv, Dqk, device=device, dtype=dtype)
    v = torch.randn(1, S, Hkv, Dvo, device=device, dtype=dtype)

    qS, kS, vS = bshd_strides(q), bshd_strides(k), bshd_strides(v)

    out_base = torch.empty(1, S, Hq, Dvo, device=device, dtype=dtype)
    out_self = torch.empty(1, S, Hq, Dvo, device=device, dtype=dtype)
    tmp_out  = torch.empty_like(out_self)
    lse = torch.empty(1, Hq, S, device=device, dtype=torch.float32)
    oS, lseS = bshd_strides(out_base), lse_strides(lse)

    # ---- 本地 push/pull scratch(不走 IPC / RingSymmMemory)----
    # 关键:每 work-item fragment 是 128 个 bf16(编译器确认 ArrayEngine<bf16,128>)。
    # slot 公式:K = ((head*kTiles + k_idx)*nD_qk + D)*threads + thr,每 slot 占 N 个元素。
    # 这里按“全量 + 余量”上界分配,保证不越界。
    # 每 slot N 个元素;总 slot 数 <= heads*kTiles*nD_qk*threads(K),难以在 host 精确算,
    # 直接给一个足够大的上界:K/V 全量元素数 * 一个安全倍数。
    k_bytes = k.numel()
    v_bytes = v.numel()
    SAFETY = 4  # fragment 重排后可能比原始 packed 大,给 4x 冗余上界防越界
    self_k = torch.zeros(k_bytes * SAFETY, device=device, dtype=torch.uint16)
    self_v = torch.zeros(v_bytes * SAFETY, device=device, dtype=torch.uint16)

    # ---- 1) baseline ----
    one_round(q, k.data_ptr(), v.data_ptr(), out_base, lse,
              S, Hq, Hkv, Dqk, Dvo, qS, kS, vS, oS, lseS,
              ring_enabled=False, ring_consume=False)

    # ---- 2) self-loopback ----
    # round A: push post-reorder fragments 到本地 self_k/self_v
    one_round(q, k.data_ptr(), v.data_ptr(), tmp_out, lse,
              S, Hq, Hkv, Dqk, Dvo, qS, kS, vS, oS, lseS,
              ring_enabled=True,
              peer_k=self_k.data_ptr(), peer_v=self_v.data_ptr(),
              ring_consume=False)
    torch.xpu.synchronize()

    # round B: consume 从本地 self_k/self_v pull 回来
    one_round(q, k.data_ptr(), v.data_ptr(), out_self, lse,
              S, Hq, Hkv, Dqk, Dvo, qS, kS, vS, oS, lseS,
              ring_enabled=False, ring_consume=True,
              recv_k=self_k.data_ptr(), recv_v=self_v.data_ptr())

    identical = torch.equal(out_base, out_self)
    max_abs = (out_base.float() - out_self.float()).abs().max().item()
    print(f"[rank {rank}] selfloop_bit_identical={identical} "
          f"(max|Δ vs baseline|={max_abs:.3e})")

    if not identical:
        diff = (out_base.float() - out_self.float()).abs()
        bad = diff > 0.0
        bad_d = bad.any(dim=(0, 1, 2)).nonzero().flatten()
        print(f"[rank {rank}] 出错的 head-dim 位置: {bad_d.tolist()}")
        print(f"[rank {rank}] 每个 D 的出错计数: {bad.sum(dim=(0,1,2)).tolist()}")
        print(f"[rank {rank}] -> push/pull round-trip 有 bug(纯序列化问题,与网络无关)")

    with torch.no_grad():
        qb, kb, vb = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        if qb.size(1) != kb.size(1):
            g = qb.size(1)//kb.size(1)
            kb, vb = kb.repeat_interleave(g,1), vb.repeat_interleave(g,1)
        ref = F.scaled_dot_product_attention(qb, kb, vb, is_causal=False).transpose(1,2).contiguous()
    torch.xpu.synchronize()
    ref_ok = torch.allclose(out_self.float(), ref.float(), atol=5e-2, rtol=5e-2)
    ref_max = (out_self.float()-ref.float()).abs().max().item()
    print(f"[rank {rank}] selftest_vs_sdpa_ok={ref_ok} (max|Δ|={ref_max:.3e})")
    print(f"[rank {rank}] OVERALL: {'PASS' if (identical and ref_ok) else 'FAIL'}")


if __name__ == "__main__":
    main()
