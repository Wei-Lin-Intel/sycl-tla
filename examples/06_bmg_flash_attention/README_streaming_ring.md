# XPU streaming ring attention

`sycl_tla_fmha.streaming_ring_bf16` computes non-causal attention for a local
BHSD Q shard while equal-sized K/V shards circulate clockwise through the
ranks. Transport runs on a bounded communication queue while FA2 and FP32
online merging run on the calling PyTorch XPU stream.

## Build

```bash
source /opt/intel/oneapi/setvars.sh
export CXX=icpx CC=icx
rm -rf build && cmake -S . -B build -G Ninja \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g21 \
  -DCUTLASS_SYCL_RUNNING_CI=ON \
  -DSYCL_TLA_BUILD_PYTHON_FMHA=ON
ninja -C build sycl_tla_fmha
export PYTHONPATH="$PWD/build/examples/06_bmg_flash_attention:$PYTHONPATH"
```

## API and memory contract

```python
output = sycl_tla_fmha.streaming_ring_bf16(
    q, k, v, k_workspace, v_workspace, signal_pad,
    peer_k_workspace_ptrs, peer_v_workspace_ptrs, peer_signal_ptrs,
    rank, world_size, iteration, is_causal=False, work_groups=8)
```

- `q`, `k`, and `v` are contiguous BF16 XPU tensors on one concrete device,
  with shapes `[B,H,S,Dq]`, `[B,H,S,Dq]`, and `[B,H,S,Dv]`.
- `Dq` and `Dv` are each one of 64, 96, 128, or 192. Batch, head, and local
  sequence dimensions are equal; GQA/MQA, unequal shards, and causal attention
  are not supported.
- `k_workspace` and `v_workspace` are peer-accessible symmetric allocations
  with shapes `[world_size-1,B,H,S,Dq]` and
  `[world_size-1,B,H,S,Dv]`. A slot is a packed BHSD shard.
- `signal_pad` is a peer-accessible contiguous `torch.int32` allocation with
  shape `[world_size-1,N]`, where `N >= work_groups`.
- Each peer pointer argument is a contiguous XPU `torch.int64` tensor of
  length `world_size`. Entry `r` is the address, mapped into the current
  process, of rank `r`'s corresponding workspace or signal allocation.
- The caller owns all allocations and mappings. They must remain alive through
  output consumption. Every rank must use the same shapes and `work_groups`.
- `iteration` is a positive, strictly increasing tag shared by all ranks.
  Pads need only be zeroed before the first invocation. All ranks must finish
  workspace registration and execute a process-group barrier before invocation.
- The output is FP32 `[B,H,S,Dv]`. `streaming_ring_bf16_lse` has the same
  arguments and returns `(output, lse)`, where LSE is natural-log FP32
  `[B,H,S]`.

Phase 0 pushes local K/V to the right peer's slot 0. Phase `t` forwards local
slot `t-1` to the right peer's slot `t`. Each slot has one release/acquire
system-scope signal per communication work-group. `work_groups` is intentionally
bounded to 32 so signal polling cannot occupy the device.

## Symmetric-memory setup

The extension accepts raw mapped pointers because this branch has no
higher-level XPU symmetric-memory allocator. The following uses PyTorch's
experimental symmetric-memory API when its XPU backend is available:

```python
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import sycl_tla_fmha

rank, world = dist.get_rank(), dist.get_world_size()
device = torch.device("xpu", rank)
torch.xpu.set_device(device)
B, H, S, D = 2, 8, 257, 128
groups = 8

def symmetric_tensor(shape, dtype):
    flat = symm_mem.empty(
        max(1, torch.Size(shape).numel()), dtype=dtype, device=device)
    handle = symm_mem.rendezvous(flat, dist.group.WORLD)
    return flat[:torch.Size(shape).numel()].view(shape), handle

k_workspace, kh = symmetric_tensor((world - 1, B, H, S, D), torch.bfloat16)
v_workspace, vh = symmetric_tensor((world - 1, B, H, S, D), torch.bfloat16)
signal_pad, sh = symmetric_tensor((world - 1, groups), torch.int32)
signal_pad.zero_()

peer_k_ptrs = torch.tensor(kh.buffer_ptrs, dtype=torch.int64, device=device)
peer_v_ptrs = torch.tensor(vh.buffer_ptrs, dtype=torch.int64, device=device)
peer_signal_ptrs = torch.tensor(
    sh.buffer_ptrs, dtype=torch.int64, device=device)
dist.barrier()

q = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
out = sycl_tla_fmha.streaming_ring_bf16(
    q, k, v, k_workspace, v_workspace, signal_pad,
    peer_k_ptrs, peer_v_ptrs, peer_signal_ptrs,
    rank, world, iteration=1, work_groups=groups)
assert out.shape == (B, H, S, D)
```

An alternative Level Zero IPC/symmetric allocator may be used if it produces
the same locally mapped peer-pointer arrays. Ordinary unrelated XPU allocations
from other processes are not sufficient.

## Multi-rank validation

```bash
torchrun --standalone --nproc-per-node=2 \
  examples/06_bmg_flash_attention/test_streaming_ring.py
torchrun --standalone --nproc-per-node=4 \
  examples/06_bmg_flash_attention/test_streaming_ring.py
```
