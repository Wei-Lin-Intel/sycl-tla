import torch, torch.nn.functional as F
import sycl_tla_fmha as fa
dev='xpu'; dt=torch.bfloat16
S=32; H=1; D=128
torch.manual_seed(0)
q =torch.randn(1,S,H,D,device=dev,dtype=dt)
k =torch.randn(1,S,H,D,device=dev,dtype=dt); v=torch.randn(1,S,H,D,device=dev,dtype=dt)

# single chunk: NO accumulate, NO lse merge — pure prefill
out = fa.prefill_bf16_bshd_kv_list(q=q, k_list=[k], v_list=[v], is_causal=False)
qb=q.transpose(1,2); kb=k.transpose(1,2); vb=v.transpose(1,2)
ref=F.scaled_dot_product_attention(qb,kb,vb,is_causal=False).transpose(1,2)
torch.testing.assert_close(out.float(), ref.float(), atol=5e-2, rtol=5e-2)
print("KV_LIST single-chunk PASSED")
