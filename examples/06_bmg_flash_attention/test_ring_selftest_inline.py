#!/usr/bin/env python3
# world=1 内联 self-loopback:单次 kernel,push 完立刻 pull。
import torch, torch.nn.functional as F
from mpi4py import MPI
import sycl_tla_fmha as fa

def bshd_strides(t):
    v = t.permute(0,2,1,3); return [v.stride(2), v.stride(1), v.stride(0)]

def main():
    MPI.COMM_WORLD  # 保证 mpi4py 初始化
    torch.xpu.set_device(0); dev = torch.device("xpu", 0); dt = torch.bfloat16
    S, Hq, Hkv, Dqk, Dvo = 512, 16, 16, 128, 128
    TileK, nD_qk, VTiles, threads, frag = 32, 4, 4, 256, 128

    torch.manual_seed(2026)
    q = torch.randn(1,S,Hq,Dqk,device=dev,dtype=dt)
    k = torch.randn(1,S,Hkv,Dqk,device=dev,dtype=dt)
    v = torch.randn(1,S,Hkv,Dvo,device=dev,dtype=dt)
    qS,kS,vS = bshd_strides(q), bshd_strides(k), bshd_strides(v)

    out_base = torch.empty(1,S,Hq,Dvo,device=dev,dtype=dt)
    out_self = torch.empty(1,S,Hq,Dvo,device=dev,dtype=dt)
    oS = bshd_strides(out_base)

    kTiles = (S + TileK - 1)//TileK
    k_elems = Hkv * kTiles * nD_qk  * threads * frag
    v_elems = Hkv * kTiles * VTiles * threads * frag
    self_k = torch.zeros(k_elems, device=dev, dtype=torch.uint16)
    self_v = torch.zeros(v_elems, device=dev, dtype=torch.uint16)

    # baseline
    fa.prefill_bf16_ring_round(
        q_ptr=q.data_ptr(), k_ptr=k.data_ptr(), v_ptr=v.data_ptr(),
        o_ptr=out_base.data_ptr(), lse_ptr=0,
        seq_len_qo=S, seq_len_kv=S, num_heads_q=Hq, num_heads_kv=Hkv,
        head_size_qk=Dqk, head_size_vo=Dvo, round_idx=0,
        ring_enabled=False, peer_k_ptr=0, peer_v_ptr=0,
        ring_consume=False, recv_k_ptr=0, recv_v_ptr=0,
        q_strides=qS, k_strides=kS, v_strides=vS,
        o_strides=oS, lse_strides=[0,0,0])
    torch.xpu.synchronize()

    # 内联 self-loopback
    fa.prefill_bf16_ring_selftest(
        q_ptr=q.data_ptr(), k_ptr=k.data_ptr(), v_ptr=v.data_ptr(),
        o_ptr=out_self.data_ptr(),
        seq_len_qo=S, seq_len_kv=S, num_heads_q=Hq, num_heads_kv=Hkv,
        head_size_qk=Dqk, head_size_vo=Dvo,
        self_k_ptr=self_k.data_ptr(), self_v_ptr=self_v.data_ptr(),
        q_strides=qS, k_strides=kS, v_strides=vS, o_strides=oS)
    torch.xpu.synchronize()

    identical = torch.equal(out_base, out_self)
    md = (out_base.float()-out_self.float()).abs().max().item()
    print(f"[inline selftest] bit_identical={identical} max|Δ|={md:.3e}")

if __name__ == "__main__":
    main()
