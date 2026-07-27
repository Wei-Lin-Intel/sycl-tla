#!/usr/bin/env python3

import argparse
import contextlib
import os
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist

try:
    from sycl_tla_fmha import prefill_bf16_tensor
except ImportError as exc:
    raise RuntimeError(
        "Cannot import sycl_tla_fmha.prefill_bf16_tensor. "
        "Please install sycl_tla_fmha or add it to PYTHONPATH."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark XPU Ulysses SP: "
            "QKV All-to-All + BF16 fused attention + output All-to-All"
        )
    )

    # Attention shape.
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=4096,
        help="Global sequence length before SP sharding",
    )
    parser.add_argument(
        "--num-head",
        type=int,
        default=16,
        help="Global attention head count before Ulysses SP",
    )
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--causal", action="store_true")

    # Benchmark.
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--loops", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)

    # Distributed.
    parser.add_argument(
        "--backend",
        type=str,
        default="xccl",
        help="torch.distributed backend",
    )

    # PyTorch profiler.
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Generate one PyTorch profiler Chrome trace per rank",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="./profiles",
        help="Directory for profiler traces",
    )
    parser.add_argument(
        "--profile-wait",
        type=int,
        default=1,
        help="Number of profiler wait iterations",
    )
    parser.add_argument(
        "--profile-warmup",
        type=int,
        default=2,
        help="Number of profiler warmup iterations",
    )
    parser.add_argument(
        "--profile-active",
        type=int,
        default=3,
        help="Number of profiler active iterations",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Record memory allocations in the profiler",
    )
    parser.add_argument(
        "--profile-stack",
        action="store_true",
        help="Record Python stacks; significantly increases profiling overhead",
    )
    parser.add_argument(
        "--profile-sync-each-step",
        action="store_true",
        help="Synchronize XPU after every profiler iteration",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace, world_size: int) -> None:
    if args.bs <= 0:
        raise ValueError("--bs must be greater than zero")
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be greater than zero")
    if args.num_head <= 0:
        raise ValueError("--num-head must be greater than zero")
    if args.head_dim <= 0:
        raise ValueError("--head-dim must be greater than zero")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.loops <= 0:
        raise ValueError("--loops must be greater than zero")

    if args.profile_wait < 0:
        raise ValueError("--profile-wait must be non-negative")
    if args.profile_warmup < 0:
        raise ValueError("--profile-warmup must be non-negative")
    if args.profile_active <= 0:
        raise ValueError("--profile-active must be greater than zero")

    if args.seq_len % world_size != 0:
        raise ValueError(
            f"seq_len={args.seq_len} must be divisible by "
            f"SP world_size={world_size}"
        )

    if args.num_head % world_size != 0:
        raise ValueError(
            f"num_head={args.num_head} must be divisible by "
            f"SP world_size={world_size}"
        )


def init_distributed(backend: str) -> tuple[int, int, int]:
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("No available Intel XPU device was detected")

    if backend == "ccl":
        try:
            import oneccl_bindings_for_pytorch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Failed to import oneccl_bindings_for_pytorch. "
                "Install a torch-ccl build matching the current PyTorch, "
                "Python, and XPU runtime versions."
            ) from exc

        if not dist.is_backend_available("ccl"):
            raise RuntimeError(
                "oneccl_bindings_for_pytorch was imported, but the CCL "
                "distributed backend was not registered. This usually means "
                "torch-ccl and torch have incompatible versions."
            )

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    if local_rank >= torch.xpu.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank}, but only "
            f"{torch.xpu.device_count()} XPU devices are visible"
        )

    torch.xpu.set_device(local_rank)

    if not dist.is_initialized():
        # device:backend format tells PyTorch that CCL handles XPU tensors.
        process_group_backend = (
            f"xpu:{backend}" if backend == "ccl" else backend
        )

        dist.init_process_group(
            backend=process_group_backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )

    return rank, local_rank, world_size


def xpu_sync() -> None:
    torch.xpu.synchronize()


def distributed_barrier() -> None:
    xpu_sync()
    dist.barrier()
    xpu_sync()


def record_range(name: str, enabled: bool):
    if enabled:
        return torch.profiler.record_function(name)
    return contextlib.nullcontext()


def reduce_max_seconds(
    local_seconds: float,
    device: torch.device,
) -> float:
    """
    Distributed latency is determined by the slowest rank.
    """
    elapsed = torch.tensor(
        local_seconds,
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return elapsed.item()


@torch.no_grad()
def ulysses_seq_to_head_flat(
    tensor: torch.Tensor,
    num_heads: int,
    head_dim: int,
    group: dist.ProcessGroup | None = None,
    annotate: bool = False,
    range_name: str = "ulysses_seq_to_head",
) -> torch.Tensor:
    """
    Forward Ulysses All-to-All.

    Input:
        [B, S_local, H_global * D]

    Output:
        [B, S_global, H_local * D]

    The sequence is gathered and heads are scattered.
    """
    with record_range(range_name, annotate):
        if tensor.ndim != 3:
            raise ValueError(
                "Expected flattened [B, S, H*D], "
                f"got shape={tuple(tensor.shape)}"
            )

        batch, local_seq_len, hidden_size = tensor.shape
        world_size = dist.get_world_size(group)

        expected_hidden_size = num_heads * head_dim
        if hidden_size != expected_hidden_size:
            raise ValueError(
                f"hidden_size={hidden_size}, expected "
                f"num_heads * head_dim={expected_hidden_size}"
            )

        if num_heads % world_size != 0:
            raise ValueError(
                f"num_heads={num_heads} must be divisible by "
                f"SP world_size={world_size}"
            )

        if world_size == 1:
            return tensor

        local_heads = num_heads // world_size
        local_hidden_size = local_heads * head_dim

        # [B, S_local, H_global*D]
        # -> [B, S_local, SP, H_local*D]
        # -> [destination_rank, B, S_local, H_local*D]
        #
        # This contiguous() is required to pack destination-rank send chunks.
        with record_range(f"{range_name}/pack", annotate):
            send = (
                tensor.view(
                    batch,
                    local_seq_len,
                    world_size,
                    local_hidden_size,
                )
                .permute(2, 0, 1, 3)
                .contiguous()
            )

        recv = torch.empty_like(send)

        with record_range(f"{range_name}/all_to_all", annotate):
            dist.all_to_all_single(
                recv,
                send,
                group=group,
            )

        # [source_rank, B, S_local, H_local*D]
        # -> [B, S_global, H_local*D]
        with record_range(f"{range_name}/unpack", annotate):
            output = recv.permute(1, 0, 2, 3).reshape(
                batch,
                local_seq_len * world_size,
                local_hidden_size,
            )

        return output


@torch.no_grad()
def ulysses_head_to_seq_flat(
    tensor: torch.Tensor,
    num_heads: int,
    head_dim: int,
    group: dist.ProcessGroup | None = None,
    annotate: bool = False,
) -> torch.Tensor:
    """
    Reverse Ulysses All-to-All.

    Input:
        [B, S_global, H_local * D]

    Output:
        [B, S_local, H_global * D]

    The sequence is scattered and heads are gathered.
    """
    with record_range("ulysses_output_reverse_a2a", annotate):
        if tensor.ndim != 3:
            raise ValueError(
                "Expected flattened [B, S, H*D], "
                f"got shape={tuple(tensor.shape)}"
            )

        batch, global_seq_len, local_hidden_size = tensor.shape
        world_size = dist.get_world_size(group)

        if num_heads % world_size != 0:
            raise ValueError(
                f"num_heads={num_heads} must be divisible by "
                f"SP world_size={world_size}"
            )

        local_heads = num_heads // world_size
        expected_local_hidden_size = local_heads * head_dim

        if local_hidden_size != expected_local_hidden_size:
            raise ValueError(
                f"local_hidden_size={local_hidden_size}, expected "
                f"{expected_local_hidden_size}"
            )

        if global_seq_len % world_size != 0:
            raise ValueError(
                f"global_seq_len={global_seq_len} must be divisible by "
                f"SP world_size={world_size}"
            )

        if world_size == 1:
            return tensor

        local_seq_len = global_seq_len // world_size

        # [B, S_global, H_local*D]
        # -> [B, SP, S_local, H_local*D]
        # -> [destination_rank, B, S_local, H_local*D]
        with record_range("ulysses_output_reverse_a2a/pack", annotate):
            send = (
                tensor.view(
                    batch,
                    world_size,
                    local_seq_len,
                    local_hidden_size,
                )
                .permute(1, 0, 2, 3)
                .contiguous()
            )

        recv = torch.empty_like(send)

        with record_range(
            "ulysses_output_reverse_a2a/all_to_all",
            annotate,
        ):
            dist.all_to_all_single(
                recv,
                send,
                group=group,
            )

        # [source_rank, B, S_local, H_local*D]
        # -> [B, S_local, H_global*D]
        with record_range("ulysses_output_reverse_a2a/unpack", annotate):
            output = recv.permute(1, 2, 0, 3).reshape(
                batch,
                local_seq_len,
                num_heads * head_dim,
            )

        return output


@torch.no_grad()
def fused_attention_flat(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    head_dim: int,
    causal: bool,
    annotate: bool = False,
) -> torch.Tensor:
    """
    Execute BF16 fused attention.

    Persistent input/output layout:
        [B, S, H * D]

    Temporary prefill_bf16_tensor input layout:
        [B, H, S, D]

    view() and transpose() do not explicitly materialize copies.
    """
    with record_range("prefill_bf16_tensor_stage", annotate):
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError(
                "Q/K/V shapes must match: "
                f"q={tuple(q.shape)}, "
                f"k={tuple(k.shape)}, "
                f"v={tuple(v.shape)}"
            )

        if q.ndim != 3:
            raise ValueError(
                f"Expected flattened [B, S, H*D], got {tuple(q.shape)}"
            )

        batch, seq_len, hidden_size = q.shape

        expected_hidden_size = num_heads * head_dim
        if hidden_size != expected_hidden_size:
            raise ValueError(
                f"hidden_size={hidden_size}, expected "
                f"{expected_hidden_size}"
            )

        with record_range("prefill_bf16_tensor_stage/layout_input", annotate):
            # Zero-copy metadata operations:
            # [B, S, H*D] -> [B, S, H, D] -> [B, H, S, D]
            q_bhsd = q.view(
                batch,
                seq_len,
                num_heads,
                head_dim,
            ).transpose(1, 2)

            k_bhsd = k.view(
                batch,
                seq_len,
                num_heads,
                head_dim,
            ).transpose(1, 2)

            v_bhsd = v.view(
                batch,
                seq_len,
                num_heads,
                head_dim,
            ).transpose(1, 2)

        with record_range(
            "prefill_bf16_tensor_stage/kernel",
            annotate,
        ):
            output_bhsd = prefill_bf16_tensor(
                q=q_bhsd,
                k=k_bhsd,
                v=v_bhsd,
                is_causal=causal,
                iterations=1,
                warmup=0,
                verify=0,
            )

        with record_range("prefill_bf16_tensor_stage/layout_output", annotate):
            if output_bhsd.dtype != torch.bfloat16:
                output_bhsd = output_bhsd.to(torch.bfloat16)

            # [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
            #
            # reshape may materialize the required BS(H*D) result. There is no
            # redundant explicit contiguous() here.
            output = output_bhsd.transpose(1, 2).reshape(
                batch,
                seq_len,
                hidden_size,
            )

        return output


@torch.no_grad()
def ulysses_attention_flat(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    num_heads: int,
    head_dim: int,
    causal: bool,
    group: dist.ProcessGroup | None = None,
    annotate: bool = False,
) -> torch.Tensor:
    """
    Complete Ulysses attention:

        Q forward All-to-All
        K forward All-to-All
        V forward All-to-All
        prefill_bf16_tensor
        output reverse All-to-All
    """
    world_size = dist.get_world_size(group)
    local_heads = num_heads // world_size

    with record_range("ulysses_attention_e2e", annotate):
        q_sp = ulysses_seq_to_head_flat(
            q_local,
            num_heads,
            head_dim,
            group,
            annotate=annotate,
            range_name="ulysses_q_forward_a2a",
        )

        k_sp = ulysses_seq_to_head_flat(
            k_local,
            num_heads,
            head_dim,
            group,
            annotate=annotate,
            range_name="ulysses_k_forward_a2a",
        )

        v_sp = ulysses_seq_to_head_flat(
            v_local,
            num_heads,
            head_dim,
            group,
            annotate=annotate,
            range_name="ulysses_v_forward_a2a",
        )

        output_sp = fused_attention_flat(
            q_sp,
            k_sp,
            v_sp,
            local_heads,
            head_dim,
            causal,
            annotate=annotate,
        )

        output_local = ulysses_head_to_seq_flat(
            output_sp,
            num_heads,
            head_dim,
            group,
            annotate=annotate,
        )

    return output_local


def attention_flops(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    causal: bool,
) -> float:
    """
    QK^T:
        2 * B * H * S^2 * D

    Attention @ V:
        2 * B * H * S^2 * D
    """
    flops = (
        4.0
        * batch_size
        * num_heads
        * seq_len
        * seq_len
        * head_dim
    )

    if causal:
        flops *= 0.5

    return flops


def benchmark(
    fn: Callable[[], Any],
    warmup: int,
    loops: int,
    device: torch.device,
) -> tuple[float, Any]:
    result = None

    for _ in range(warmup):
        result = fn()

    distributed_barrier()

    start = time.perf_counter()

    for _ in range(loops):
        result = fn()

    xpu_sync()
    local_elapsed = time.perf_counter() - start

    # Ensure all ranks leave the measured section before the next stage.
    dist.barrier()

    max_elapsed = reduce_max_seconds(local_elapsed, device)

    return max_elapsed / loops, result


def get_profiler_activities() -> list[torch.profiler.ProfilerActivity]:
    activities = [torch.profiler.ProfilerActivity.CPU]

    if not hasattr(torch.profiler.ProfilerActivity, "XPU"):
        raise RuntimeError(
            "The current PyTorch build does not expose "
            "torch.profiler.ProfilerActivity.XPU. Use an XPU PyTorch build "
            "with Kineto/XPU profiler support."
        )

    activities.append(torch.profiler.ProfilerActivity.XPU)
    return activities


@torch.no_grad()
def profile_end_to_end(
    fn: Callable[[], torch.Tensor],
    args: argparse.Namespace,
    rank: int,
    local_rank: int,
) -> None:
    """
    Generate a separate Chrome trace for every distributed rank.
    """
    os.makedirs(args.profile_dir, exist_ok=True)

    activities = get_profiler_activities()

    total_steps = (
        args.profile_wait
        + args.profile_warmup
        + args.profile_active
    )

    trace_path = os.path.join(
        args.profile_dir,
        f"ulysses_rank{rank}_xpu{local_rank}.json",
    )

    if rank == 0:
        print("\nPyTorch profiler configuration")
        print(f"  Wait steps          : {args.profile_wait}")
        print(f"  Warmup steps        : {args.profile_warmup}")
        print(f"  Active steps        : {args.profile_active}")
        print(f"  Total steps         : {total_steps}")
        print(f"  Profile memory      : {args.profile_memory}")
        print(f"  Record stack        : {args.profile_stack}")
        print(f"  Trace directory     : {args.profile_dir}")

    distributed_barrier()

    trace_exported = False

    def export_trace(prof: torch.profiler.profile) -> None:
        nonlocal trace_exported

        prof.export_chrome_trace(trace_path)
        trace_exported = True

    schedule = torch.profiler.schedule(
        wait=args.profile_wait,
        warmup=args.profile_warmup,
        active=args.profile_active,
        repeat=1,
    )

    output = None

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=export_trace,
        record_shapes=True,
        profile_memory=args.profile_memory,
        with_stack=args.profile_stack,
        with_flops=False,
    ) as prof:
        for step in range(total_steps):
            with torch.profiler.record_function(
                f"profile_iteration_{step}"
            ):
                output = fn()

            if args.profile_sync_each_step:
                xpu_sync()

            prof.step()

        # Ensure all active XPU work has completed before closing profiler.
        xpu_sync()

    distributed_barrier()

    if not trace_exported or not os.path.isfile(trace_path):
        raise RuntimeError(
            f"Profiler did not generate the expected trace: {trace_path}"
        )

    trace_size_mb = os.path.getsize(trace_path) / 1024**2

    print(
        f"[rank {rank}] profiler trace: {trace_path}, "
        f"size={trace_size_mb:.2f} MiB, "
        f"output_shape={tuple(output.shape)}"
    )


def main() -> None:
    args = parse_args()

    rank, local_rank, world_size = init_distributed(args.backend)
    device = torch.device(f"xpu:{local_rank}")

    validate_args(args, world_size)

    local_seq_len = args.seq_len // world_size
    local_num_heads = args.num_head // world_size

    global_hidden_size = args.num_head * args.head_dim
    local_hidden_size = local_num_heads * args.head_dim

    # Different sequence shards receive different random values.
    torch.manual_seed(args.seed + rank)

    # Persistent Q/K/V layout:
    # [B, S_local, H_global * D]
    local_shape = (
        args.bs,
        local_seq_len,
        global_hidden_size,
    )

    q_local = torch.randn(
        local_shape,
        device=device,
        dtype=torch.bfloat16,
    )
    k_local = torch.randn(
        local_shape,
        device=device,
        dtype=torch.bfloat16,
    )
    v_local = torch.randn(
        local_shape,
        device=device,
        dtype=torch.bfloat16,
    )

    distributed_barrier()

    if rank == 0:
        print("XPU BF16 Ulysses-SP fused-attention benchmark")
        print(f"  Backend             : {args.backend}")
        print(f"  SP world size       : {world_size}")
        print(f"  Global batch        : {args.bs}")
        print(f"  Global sequence     : {args.seq_len}")
        print(f"  Global heads        : {args.num_head}")
        print(f"  Local heads         : {local_num_heads}")
        print(f"  Head dimension      : {args.head_dim}")
        print(f"  Local QKV shape     : {list(local_shape)}")
        print(
            "  Post-A2A flat shape : "
            f"[{args.bs}, {args.seq_len}, {local_hidden_size}]"
        )
        print(
            "  Actual kernel shape : "
            f"[{args.bs}, {local_num_heads}, "
            f"{args.seq_len}, {args.head_dim}]"
        )
        print("  Persistent layout   : [B, S, H * D]")
        print("  Kernel layout       : [B, H_local, S_global, D]")
        print(f"  Causal              : {args.causal}")
        print(f"  Warmup loops        : {args.warmup}")
        print(f"  Benchmark loops     : {args.loops}")

    # ------------------------------------------------------------------
    # Benchmark Q/K/V forward All-to-All.
    # ------------------------------------------------------------------
    def run_qkv_all_to_all():
        q_sp = ulysses_seq_to_head_flat(
            q_local,
            args.num_head,
            args.head_dim,
            range_name="ulysses_q_forward_a2a",
        )
        k_sp = ulysses_seq_to_head_flat(
            k_local,
            args.num_head,
            args.head_dim,
            range_name="ulysses_k_forward_a2a",
        )
        v_sp = ulysses_seq_to_head_flat(
            v_local,
            args.num_head,
            args.head_dim,
            range_name="ulysses_v_forward_a2a",
        )
        return q_sp, k_sp, v_sp

    qkv_a2a_seconds, qkv_sp = benchmark(
        run_qkv_all_to_all,
        args.warmup,
        args.loops,
        device,
    )

    q_sp, k_sp, v_sp = qkv_sp

    expected_post_a2a_shape = (
        args.bs,
        args.seq_len,
        local_hidden_size,
    )

    for name, tensor in (
        ("q_sp", q_sp),
        ("k_sp", k_sp),
        ("v_sp", v_sp),
    ):
        if tuple(tensor.shape) != expected_post_a2a_shape:
            raise AssertionError(
                f"{name}.shape={tuple(tensor.shape)}, "
                f"expected={expected_post_a2a_shape}"
            )

    # ------------------------------------------------------------------
    # Benchmark prefill_bf16_tensor only.
    # ------------------------------------------------------------------
    def run_attention():
        return fused_attention_flat(
            q_sp,
            k_sp,
            v_sp,
            local_num_heads,
            args.head_dim,
            args.causal,
        )

    attention_seconds, output_sp = benchmark(
        run_attention,
        args.warmup,
        args.loops,
        device,
    )

    # ------------------------------------------------------------------
    # Benchmark reverse output All-to-All.
    # ------------------------------------------------------------------
    def run_output_all_to_all():
        return ulysses_head_to_seq_flat(
            output_sp,
            args.num_head,
            args.head_dim,
        )

    output_a2a_seconds, local_output = benchmark(
        run_output_all_to_all,
        args.warmup,
        args.loops,
        device,
    )

    # ------------------------------------------------------------------
    # Benchmark complete E2E path.
    # ------------------------------------------------------------------
    def run_end_to_end():
        return ulysses_attention_flat(
            q_local,
            k_local,
            v_local,
            args.num_head,
            args.head_dim,
            args.causal,
            annotate=False,
        )

    e2e_seconds, e2e_output = benchmark(
        run_end_to_end,
        args.warmup,
        args.loops,
        device,
    )

    expected_output_shape = (
        args.bs,
        local_seq_len,
        global_hidden_size,
    )

    if tuple(local_output.shape) != expected_output_shape:
        raise AssertionError(
            f"Reverse A2A output shape={tuple(local_output.shape)}, "
            f"expected={expected_output_shape}"
        )

    if tuple(e2e_output.shape) != expected_output_shape:
        raise AssertionError(
            f"E2E output shape={tuple(e2e_output.shape)}, "
            f"expected={expected_output_shape}"
        )

    global_flops = attention_flops(
        args.bs,
        args.seq_len,
        args.num_head,
        args.head_dim,
        args.causal,
    )

    kernel_tflops = global_flops / attention_seconds / 1e12
    e2e_tflops = global_flops / e2e_seconds / 1e12

    isolated_sum_seconds = (
        qkv_a2a_seconds
        + attention_seconds
        + output_a2a_seconds
    )

    if rank == 0:
        print("\nResults — isolated stages")
        print(f"  QKV forward A2A     : {qkv_a2a_seconds * 1e3:.3f} ms")
        print(f"  Attention kernel    : {attention_seconds * 1e3:.3f} ms")
        print(f"  Output reverse A2A  : {output_a2a_seconds * 1e3:.3f} ms")
        print(f"  Isolated-stage sum  : {isolated_sum_seconds * 1e3:.3f} ms")

        print("\nResults — end to end")
        print(f"  E2E latency         : {e2e_seconds * 1e3:.3f} ms")
        print(f"  Global FLOPs/call   : {global_flops / 1e12:.6f} TFLOP")
        print(f"  Kernel aggregate    : {kernel_tflops:.3f} TFLOPs")
        print(
            f"  Kernel per XPU      : "
            f"{kernel_tflops / world_size:.3f} TFLOPs/XPU"
        )
        print(f"  E2E effective       : {e2e_tflops:.3f} TFLOPs")
        print(
            f"  E2E per XPU         : "
            f"{e2e_tflops / world_size:.3f} TFLOPs/XPU"
        )

        communication_ratio = (
            1.0 - attention_seconds / e2e_seconds
            if e2e_seconds > 0
            else 0.0
        )

        print(
            f"  Non-kernel fraction : "
            f"{communication_ratio * 100.0:.2f}%"
        )

        print("\nOutput")
        print(f"  Local output shape  : {list(e2e_output.shape)}")
        print(
            f"  Output finite       : "
            f"{bool(torch.isfinite(e2e_output).all().item())}"
        )

    # ------------------------------------------------------------------
    # Optional profiler pass.
    #
    # This pass is separate from the performance benchmark because profiler
    # instrumentation changes latency.
    # ------------------------------------------------------------------
    if args.profile:
        def run_profile_end_to_end():
            return ulysses_attention_flat(
                q_local,
                k_local,
                v_local,
                args.num_head,
                args.head_dim,
                args.causal,
                annotate=True,
            )

        profile_end_to_end(
            run_profile_end_to_end,
            args,
            rank,
            local_rank,
        )

    distributed_barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
