#!/usr/bin/env python3
# world=1 two-call 路径最小化测试:
#   round A: ring_enabled 把 K/V fragment push 到本地 buffer
#   round B: ring_consume 从同一 buffer pull 回来做 gemm
#   比 round B 输出 vs baseline(纯 prefill)
#
# 默认 S=32, H=1 —— 单 head、单 K-tile,把 tile-scheduler 的 WG/head 映射
# 变量降到最小。如果这样 two-call 就 bit_identical,而大配置全错,
# 说明 bug 是“两次独立 kernel launch 之间 fragment->slot 映射不一致”。
#
# 前提:kFrag 已改为 128,self buffer 按 128 精确分配。
#
# 用法:
#   mpirun -n 1 python3 test_ring_two_call_min.py                 # S=32 H=1
#   mpirun -n 1 python3 test_ring_two_call_min.py --q-seq-len 512 --q-nhead 16

import argparse
import torch
import torch.nn.functional as F
from mpi4py import MPI
import sycl_tla_fmha as fa


def bshd_strides(t):
    v = t.permute(0, 2, 1, 3)
    return [v.stride(2), v.stride(1), v.stride(0)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--q-seq-len", type=int, default=32)   # 一个 K-tile(TileK=32)
    p.add_argument("--q-nhead",   type=int, default=1)    # 单 head
    p.add_argument("--kv-nhead",  type=int, default=1)
    p.add_argument("--qk-hdim",   type=int, default=128)
    p.add_argument("--v-hdim",    type=int, default=128, choices=(64, 96, 128, 192))
    p.add_argument("--seed",      type=int, default=2026)
    return p.parse_args()


def main():
    a = parse_args()
    MPI.COMM_WORLD  # 保证 mpi4py 初始化 MPI

    torch.xpu.set_device(0)
    dev = torch.device("xpu", 0)
    dt = torch.bfloat16

    S, Hq, Hkv, Dqk, Dvo = a.q_seq_len, a.q_nhead, a.kv_nhead, a.qk_hdim, a.v_hdim

    # head_dim=128 的 ring fragment 常量(与 mainloop / py_fmha_module 保持一致)
    #   ShapeQK=<256,32,32>, SubgroupLayoutQK=<16,1,1>
    #   NumThreadsQK = size(TiledMMAQK) = 256
    #   TileK = 32 -> nD_qk = Dqk/32 = 4
    #   VTiles = Dvo/32 = 4
    #   frag = 128  (编译器确认 ArrayEngine<bf16,128>)
    TileK   = 32
    nD_qk   = Dqk // 32
    VTiles  = Dvo // 32
    threads = 256
    frag    = 128

    torch.manual_seed(a.seed)
    q = torch.randn(1, S, Hq,  Dqk, device=dev, dtype=dt)
    k = torch.randn(1, S, Hkv, Dqk, device=dev, dtype=dt)
    v = torch.randn(1, S, Hkv, Dvo, device=dev, dtype=dt)
    qS, kS, vS = bshd_strides(q), bshd_strides(k), bshd_strides(v)

    out_base = torch.empty(1, S, Hq, Dvo, device=dev, dtype=dt)
    out_self = torch.empty(1, S, Hq, Dvo, device=dev, dtype=dt)
    tmp      = torch.empty_like(out_self)
    lse = torch.empty(1, Hq, S, device=dev, dtype=torch.float32)
    oS = bshd_strides(out_base)
    lseS = [lse.stride(1), lse.stride(2), lse.stride(0)]
#    lseS = [lse.stride(2), lse.stride(1), lse.stride(0)]

    # self buffer 按 128 精确上界分配(slot = ((head*kTiles + k_idx)*nD_qk + D)*threads + thr)
    kTiles = (S + TileK - 1) // TileK
    k_elems = Hkv * kTiles * nD_qk  * threads * frag
    v_elems = Hkv * kTiles * VTiles * threads * frag
    self_k = torch.zeros(k_elems, device=dev, dtype=torch.uint16)
    self_v = torch.zeros(v_elems, device=dev, dtype=torch.uint16)

    def call(out, **kw):
        fa.prefill_bf16_ring_round(
            q_ptr=q.data_ptr(), k_ptr=k.data_ptr(), v_ptr=v.data_ptr(),
            o_ptr=out.data_ptr(), lse_ptr=lse.data_ptr(),
            seq_len_qo=S, seq_len_kv=S,
            num_heads_q=Hq, num_heads_kv=Hkv,
            head_size_qk=Dqk, head_size_vo=Dvo,
            round_idx=0,
            q_strides=qS, k_strides=kS, v_strides=vS,
            o_strides=oS, lse_strides=lseS, **kw)
        torch.xpu.synchronize()

    # baseline:无 ring
    call(out_base, ring_enabled=False, peer_k_ptr=0, peer_v_ptr=0,
         ring_consume=False, recv_k_ptr=0, recv_v_ptr=0)

    # round A:push 到本地 self buffer
    call(tmp, ring_enabled=True,
         peer_k_ptr=self_k.data_ptr(), peer_v_ptr=self_v.data_ptr(),
         ring_consume=False, recv_k_ptr=0, recv_v_ptr=0)

    # round B:从本地 self buffer consume
    call(out_self, ring_enabled=False, peer_k_ptr=0, peer_v_ptr=0,
         ring_consume=True,
         recv_k_ptr=self_k.data_ptr(), recv_v_ptr=self_v.data_ptr())

    identical = torch.equal(out_base, out_self)
    md = (out_base.float() - out_self.float()).abs().max().item()
    print(f"[two-call S={S} H={Hq}] bit_identical={identical} max|Δ|={md:.3e}")

    if not identical:
        diff = (out_base.float() - out_self.float()).abs()
        bad = diff > 0.0
        print(f"  bad D count: {bad.sum(dim=(0,1,2)).tolist()}")
        # 看是不是整段为 0(说明 consume 读到未写入的 slot)
        z = (out_self.float().abs() < 1e-9).float().mean().item()
        print(f"  out_self 中 ~0 的比例: {z:.3f}  (接近 1 => consume 读到空 slot)")

    # 跟 SDPA 对一下,确认 baseline 本身是对的
    with torch.no_grad():
        qb, kb, vb = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if qb.size(1) != kb.size(1):
            g = qb.size(1) // kb.size(1)
            kb, vb = kb.repeat_interleave(g, 1), vb.repeat_interleave(g, 1)
        ref = F.scaled_dot_product_attention(qb, kb, vb, is_causal=False).transpose(1, 2).contiguous()
    torch.xpu.synchronize()
    base_ok = torch.allclose(out_base.float(), ref.float(), atol=5e-2, rtol=5e-2)
    print(f"[baseline vs sdpa] ok={base_ok} "
          f"max|Δ|={(out_base.float()-ref.float()).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
