import torch, torch.nn.functional as F
from mpi4py import MPI
import sycl_tla_fmha as fa
dev='xpu'; dt=torch.bfloat16
S=2048; H=8; D=128           # 合法 shape
torch.manual_seed(0)
q=torch.randn(1,S,H,D,device=dev,dtype=dt)
k=torch.randn(1,S,H,D,device=dev,dtype=dt); v=torch.randn(1,S,H,D,device=dev,dtype=dt)

ring = fa.RingSymmMemory(seq_kv_local=S, h_kv=H, d_qk=D, d_vo=D, rank=0, world_size=1)
out=torch.empty(1,S,H,D,device=dev,dtype=dt); lse=torch.empty(1,H,S,device=dev,dtype=torch.float32)
def bs(t): v=t.permute(0,2,1,3); return [v.stride(2),v.stride(1),v.stride(0)]
def ls(l): return [l.stride(2),l.stride(1),l.stride(0)]

# round0: global load K, push fragments to OUR OWN buf[1] (peer=self)
fa.prefill_bf16_ring_round(
    q_ptr=q.data_ptr(), k_ptr=k.data_ptr(), v_ptr=v.data_ptr(),
    o_ptr=out.data_ptr(), lse_ptr=lse.data_ptr(),
    seq_len_qo=S, seq_len_kv=S, num_heads_q=H, num_heads_kv=H,
    head_size_qk=D, head_size_vo=D, round_idx=0,
    ring_enabled=True, peer_k_ptr=ring.local_k(1), peer_v_ptr=ring.local_v(1),
    ring_consume=False, recv_k_ptr=0, recv_v_ptr=0,
    q_strides=bs(q), k_strides=bs(k), v_strides=bs(v),
    o_strides=bs(out), lse_strides=ls(lse))
torch.xpu.synchronize()

# round1: CONSUME from our buf[1] (should be identical K we just pushed).
# Compare against SDPA over [k;k] (i.e. round0 K counted twice).
out2=torch.empty(1,S,H,D,device=dev,dtype=dt); lse2=torch.empty(1,H,S,device=dev,dtype=torch.float32)
fa.prefill_bf16_ring_round(
    q_ptr=q.data_ptr(), k_ptr=k.data_ptr(), v_ptr=v.data_ptr(),
    o_ptr=out2.data_ptr(), lse_ptr=lse2.data_ptr(),
    seq_len_qo=S, seq_len_kv=S, num_heads_q=H, num_heads_kv=H,
    head_size_qk=D, head_size_vo=D, round_idx=0,   # round0 semantics, but consume
    ring_enabled=False, peer_k_ptr=0, peer_v_ptr=0,
    ring_consume=True, recv_k_ptr=ring.local_k(1), recv_v_ptr=ring.local_v(1),
    q_strides=bs(q), k_strides=bs(k), v_strides=bs(v),
    o_strides=bs(out2), lse_strides=ls(lse2))
torch.xpu.synchronize()

# consume(K) must equal global-load(K): same K, same result
qb=q.transpose(1,2); kb=k.transpose(1,2); vb=v.transpose(1,2)
ref=F.scaled_dot_product_attention(qb,kb,vb,is_causal=False).transpose(1,2)
d1=(out2.float()-ref.float()).abs()
print(f"CONSUME vs global-load: max {d1.max():.4f} mean {d1.mean():.6f}")
torch.testing.assert_close(out2.float(), ref.float(), atol=5e-2, rtol=5e-2)
print("CONSUME path PASSED")
