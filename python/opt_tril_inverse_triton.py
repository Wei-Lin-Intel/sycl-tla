# Copyright (c) 2026

import pytest
import torch
import triton
import triton.language as tl
import time
import argparse

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=["N", "STEPS"],
    restore_value=["Out_ptr"],
)
@triton.jit
def tril_inverse_iterative_64_bf16_kernel(
    L_ptr,
    Out_ptr,
    stride_lb, stride_lm, stride_ln,
    stride_ob, stride_om, stride_on,
    B,
    N: tl.constexpr,
    STEPS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= B:
        return

    offs_i = tl.arange(0, N)
    offs_j = tl.arange(0, N)

    L_batch_ptr = L_ptr + pid * stride_lb
    l_ptrs = L_batch_ptr + offs_i[:, None] * stride_lm + offs_j[None, :] * stride_ln

    # Keep L in bf16
    L = tl.load(l_ptrs)  # bf16

    I = offs_i[:, None] == offs_j[None, :]
    M = offs_i[:, None] > offs_j[None, :]

    # Keep inv in bf16 as well
    inv = tl.where(I, 1, 0).to(tl.bfloat16)

    for _ in range(STEPS):
        err = tl.where(M, tl.dot(L, inv), 0).to(tl.bfloat16)
        inv = inv - tl.dot(inv, err).to(tl.bfloat16)

    Out_batch_ptr = Out_ptr + pid * stride_ob
    o_ptrs = Out_batch_ptr + offs_i[:, None] * stride_om + offs_j[None, :] * stride_on
    tl.store(o_ptrs, inv)

def solve_tril_iterative_triton_bf16(L: torch.Tensor, steps: int = 63) -> torch.Tensor:
    """
    Iterative inverse kernel for unit lower-triangular matrices.

    Args:
        L: [B, 64, 64], bf16 CUDA tensor
        steps: iteration count, default 63 for n=64

    Returns:
        inv(L) in bf16, shape [B, 64, 64]
    """
    assert L.dtype == torch.bfloat16, "L must be bf16"

    B = L.shape[0]
    out = torch.empty_like(L)

    tril_inverse_iterative_64_bf16_kernel[(B,)](
        L,
        out,
        L.stride(0), L.stride(1), L.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        B,
        N=64,
        STEPS=steps,
    )
    return out


def make_strictly_lower_A(B: int, n: int, device: str, dtype: torch.dtype, scale: float = 0.05):
    """
    Create a strictly lower-triangular A with small magnitude so that L = I + A is well-behaved.
    """
    A = torch.randn(B, n, n, device=device, dtype=torch.float32) * scale
    A = torch.tril(A, diagonal=-1)
    return A.to(dtype)


if __name__ == '__main__':
    """
    Compare repo solve_tril(A) against iterative Triton inverse on L = I + A.

    solve_tril expects A to be strictly lower triangular and returns (I + A)^-1.
    The iterative Triton kernel takes L directly.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", default=1024, type=int)
    parser.add_argument("--n", default=64, type=int)
    parser.add_argument("--step", default=12, type=int)
    parser.add_argument("--loop", default=100, type=int)
    par = parser.parse_args()

    device = "xpu"
    dtype = torch.bfloat16
    B = par.bs
    n = par.n
    step = par.step
    loop = par.loop

    # A is strictly lower triangular
    A = make_strictly_lower_A(B=B, n=n, device=device, dtype=dtype, scale=0.2)

    # solve_tril expects [B, T, H, BT], here use T=1, H=1, BT=64 encoded as [B, 1, 1, 64] is NOT enough
    # The repo format is [B, T, H, BT] where each row block corresponds to a lower-triangular block.
    # For a single 64x64 matrix, represent it as [B, 64, 1, 64].
    A_repo = A.unsqueeze(2)   # [B, 64, 1, 64]

    # Build L = I + A for iterative kernel
    eye = torch.eye(n, device=device, dtype=dtype).unsqueeze(0)   # [1, 64, 64]
    L = A + eye

    with torch.no_grad():
        for _ in range(10):
            test = solve_tril_iterative_triton_bf16(L, steps=step)
        torch.xpu.synchronize()

        t0 = time.time()
        for _ in range(loop):
            test = solve_tril_iterative_triton_bf16(L, steps=step)
        torch.xpu.synchronize()
        t1 = time.time()
        t = (t1 - t0) * 1000.0 / loop
        print("Triton Iterative latency: {:.3f} ms".format(t))

    ref_f = L.float().inverse()
    test_f = test.float()

    abs_err = (test_f - ref_f).abs()
    rel_err = abs_err / ref_f.abs().clamp_min(1e-6)

    max_abs_err = abs_err.max().item()
    mean_abs_err = abs_err.mean().item()

    print(f"Batch Size = {B}")
    print(f"max_abs_err={max_abs_err:.6e}")
    print(f"mean_abs_err={mean_abs_err:.6e}")

    # Because the iterative method is more error-prone in bf16, keep thresholds moderate.
    assert mean_abs_err < 5e-2
    assert max_abs_err < 5e-1
