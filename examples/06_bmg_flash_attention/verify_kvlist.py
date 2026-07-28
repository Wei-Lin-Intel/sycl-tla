import torch, torch.nn.functional as F
import sycl_tla_fmha as fa
dev='xpu'; dt=torch.bfloat16
S=2048; H=8; D=128           # S 是 256 的倍数,合法
torch.manual_seed(0)
q =torch.randn(1,S,H,D,device=dev,dtype=dt)
k0=torch.randn(1,S,H,D,device=dev,dtype=dt); v0=torch.randn(1,S,H,D,device=dev,dtype=dt)
k1=torch.randn(1,S,H,D,device=dev,dtype=dt); v1=torch.randn(1,S,H,D,device=dev,dtype=dt)
out = fa.prefill_bf16_bshd_kv_list(q=q, k_list=[k0,k1], v_list=[v0,v1], is_causal=False)
qb=q.transpose(1,2); kb=torch.cat([k0,k1],1).transpose(1,2); vb=torch.cat([v0,v1],1).transpose(1,2)
ref=F.scaled_dot_product_attention(qb,kb,vb,is_causal=False).transpose(1,2)
torch.testing.assert_close(out.float(), ref.float(), atol=5e-2, rtol=5e-2)
print("KV_LIST 2-chunk (valid shape) PASSED")
