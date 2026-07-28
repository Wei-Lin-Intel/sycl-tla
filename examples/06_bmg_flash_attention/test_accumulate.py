import torch, torch.nn.functional as F
import sycl_tla_fmha as fa
torch.xpu.set_device(0); dev=torch.device("xpu",0); dt=torch.bfloat16
s,H,D = 256,1,128
torch.manual_seed(2026)
q  = torch.randn(1,s,H,D,device=dev,dtype=dt)
k0 = torch.randn(1,s,H,D,device=dev,dtype=dt); v0 = torch.randn(1,s,H,D,device=dev,dtype=dt)
k1 = torch.randn(1,s,H,D,device=dev,dtype=dt); v1 = torch.randn(1,s,H,D,device=dev,dtype=dt)

def stats(out, ref, name):
    d = (out.float()-ref.float()).abs()
    over = (d > 5e-2).float().mean().item()
    print(f"{name}: max={d.max().item():.4f} mean={d.mean().item():.5f} "
          f">5e-2 比例={over:.4f}")

# 单块:q attend k0
out1 = fa.prefill_bf16_bshd_kv_list(q, [k0], [v0], is_causal=False)
with torch.no_grad():
    ref1 = F.scaled_dot_product_attention(q.transpose(1,2),k0.transpose(1,2),v0.transpose(1,2),is_causal=False).transpose(1,2).contiguous()
torch.xpu.synchronize()
stats(out1, ref1, "1-block H=1")

# 两块
out2 = fa.prefill_bf16_bshd_kv_list(q, [k0,k1], [v0,v1], is_causal=False)
with torch.no_grad():
    kb=torch.cat([k0,k1],1).transpose(1,2); vb=torch.cat([v0,v1],1).transpose(1,2)
    ref2 = F.scaled_dot_product_attention(q.transpose(1,2),kb,vb,is_causal=False).transpose(1,2).contiguous()
torch.xpu.synchronize()
stats(out2, ref2, "2-block H=1")

# 对照:H=40 两块(和 xpu_kv_fa 一致的规模)
H2=40
q40=torch.randn(1,s,H2,D,device=dev,dtype=dt)
k0b=torch.randn(1,s,H2,D,device=dev,dtype=dt); v0b=torch.randn(1,s,H2,D,device=dev,dtype=dt)
k1b=torch.randn(1,s,H2,D,device=dev,dtype=dt); v1b=torch.randn(1,s,H2,D,device=dev,dtype=dt)
out40 = fa.prefill_bf16_bshd_kv_list(q40, [k0b,k1b], [v0b,v1b], is_causal=False)
with torch.no_grad():
    kb=torch.cat([k0b,k1b],1).transpose(1,2); vb=torch.cat([v0b,v1b],1).transpose(1,2)
    ref40 = F.scaled_dot_product_attention(q40.transpose(1,2),kb,vb,is_causal=False).transpose(1,2).contiguous()
torch.xpu.synchronize()
stats(out40, ref40, "2-block H=40")
