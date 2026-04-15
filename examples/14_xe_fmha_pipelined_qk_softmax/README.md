# Example 14: Xe FMHA Pipelined QK–Softmax Overlap

## Goal

Prototype a 2-slot block pipeline in the Xe forward FMHA mainloop that overlaps
`softmax(cur_block)` [XVX] with `QK(next_block)` [XMX], modelled on the
Flash Attention V3 cross-block pipeline idea but restricted to the conservative
scope that avoids two concurrent XMX streams.

## Background

The serial per-block schedule in the baseline FA v2 mainloop looks like:

```
for k in [k_start, k_end):
    QK(k)        # XMX
    mask(k)      # scalar
    softmax(k)   # XVX
    PV(k)        # XMX
```

`softmax(k)` only depends on `S_k = Q * K_k` and the running `(max, sum)` state.
`QK(k+1)` only depends on `Q` and `K_{k+1}`.
→ These two are **data-independent** and can overlap if XMX and XVX engines
  can dispatch concurrently.

## Refactored schedule (implemented in `xe_fmha_fwd_mainloop.hpp`)

```
Prologue:
    QK(k_start) → S_pipe[0]              [XMX]

Steady-state  k = k_start .. k_end-2:
    cur_slot  = (k - k_start) & 1
    next_slot = cur_slot ^ 1

    QK(k+1)  → S_pipe[next_slot]         [XMX]   ← issued first
    mask(k)    on S_pipe[cur_slot]        [scalar] ← targets XVX; overlaps XMX above
    softmax(k) on S_pipe[cur_slot]        [XVX]   ←      "
    PV(k)  using P_pipe[cur_slot]         [XMX]   ← after softmax; avoids dual-XMX

Epilogue:
    mask(last) + softmax(last) + PV(last) [no next-QK]
```

The critical reordering is: issue **QK(next)** first, then run **softmax(cur)**.
The compiler/hardware has a window to overlap XMX (DPAS) and XVX (vector ALU)
when the two instruction streams access disjoint registers (`S_pipe[next_slot]`
vs `S_pipe[cur_slot]`).

## Key data structures

| Symbol | Description |
|--------|-------------|
| `FragS tSrS_pipe[2]` | QK output accumulator, double-buffered |
| `FragAforP tArP_pipe[2]` | Reordered P (softmax output) for P×V GEMM input |
| `BlockDesc` | Per-block metadata: `{logical_k, physical_k, is_cache}` |

The `BlockDesc` struct unifies logical/physical K indexing and the
cache/non-cache/paged dispatch into a single loop, replacing the original
two separate `for` loops.

## Helper stage lambdas

```
compute_qk_block(desc, tSrS_pipe[slot])
    Stage A: Q*K GEMM + V prefetch → tSrS_pipe[slot]   [XMX]

apply_mask_block(desc, tSrS_pipe[slot])
    Stage B: causal mask / remainder mask                [scalar]

apply_softmax_and_prepare_p(first, tSrS_pipe[slot], tArP_pipe[slot])
    Stage C: online softmax + row-wise reorder           [XVX]

accumulate_pv_block(desc, tArP_pipe[slot], rescale, first)
    Stage D: P*V GEMM + accumulator rescale              [XMX]

prefetch_k_block(K_next)
    Prefetch hint for K tile K_next                      [prefetch]
```

## What is and is NOT overlapped in this version

| Overlap | Status |
|---------|--------|
| `softmax(cur)` ‖ `QK(next)` | ✅ Target of this PR |
| `PV(cur)` ‖ `QK(next)` | ❌ Deferred (would create two XMX streams) |
| Full FA3-style role assignment | ❌ Deferred (needs multi-group scheduling) |

## Prerequisites for hardware overlap

1. Intel Xe XMX and XVX engines can dispatch concurrently on the target SKU.
2. The compiler emits a mixed DPAS + vector instruction stream (verify with
   `ocloc disasm` or Intel VTune's GPU ISA view).
3. Register pressure stays below the GRF budget (256 GRF / thread on PVC,
   128 on BMG).  The extra `FragS + FragAforP` slot doubles the S/P buffer
   size; profile occupancy if throughput drops.

## How to validate

### 1. Correctness (numerical)

Run the existing `06_bmg_flash_attention` example against its reference:

```bash
source /opt/intel/oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
cd build
ninja 06_xe_fmha_fwd
./examples/sycl/06_bmg_flash_attention/06_xe_fmha_fwd \
    --seq_len_qo=1024 --seq_len_kv=1024 \
    --head_size_vo=128 --head_size_qk=128 \
    --verify=true
```

The pipelined mainloop produces bit-identical results to the serial one for
all non-masked tiles (online softmax accumulation is unchanged).

### 2. Performance (XMX/XVX overlap)

Profile with Intel VTune Profiler:

```bash
vtune -collect gpu-hotspots -knob sampling-interval=1ms \
      -result-dir vtune_out -- ./06_xe_fmha_fwd --bench_iters=100
vtune -report hw-events -r vtune_out | grep -E "XMX|XVX|Util"
```

Look for reduced `XMX Stall` or improved `XVX Active` overlap with `XMX Active`.

### 3. ISA inspection

```bash
ocloc compile -file kernel.spv -device <pvc|bmg> -options "-ze-opt-large-register-file"
ocloc disasm -file kernel_pvc.bin -device pvc | grep -A2 -B2 "dpas\|mad\|mov"
```

Verify that `dpas` (XMX) and `mad`/`mov` (XVX) instructions are interleaved
rather than fully serialized.

## Known limitations and tradeoffs

- **Register pressure**: two `FragS` + two `FragAforP` buffers.  If occupancy
  drops, try reducing `VTiles` or `Stages`.
- **Compiler freedom**: the C++ source ordering is a hint, not a guarantee.
  Actual overlap depends on the IGC (Intel Graphics Compiler) scheduler.
- **Single-block case**: when `n_blocks == 1`, the steady-state loop is skipped
  and execution is prologue → epilogue (no overlap opportunity).
- **Cache/non-cache boundary**: the pipeline slot may hold a cache block in
  one slot and a non-cache block in the other; `BlockDesc::is_cache` handles
  this transparently via runtime dispatch.
- **No SLM ping-pong**: this design keeps P entirely in registers (matching the
  `XeDefault` policy); an SLM-based exchange for sub-group cooperation is
  future work.

## Future extensions

- **PV ‖ QK overlap**: once profiling shows the XVX window is fully utilized,
  consider a 3-stage pipeline where PV(cur) and QK(next+1) overlap.  This
  requires ensuring PV does not compete with QK for XMX bandwidth.
- **Sub-group role specialization**: assign dedicated sub-groups to QK vs.
  softmax/PV, similar to FA3's warp-group model.  This requires changing the
  sub-group layout and is a more invasive refactor.
- **Larger pipeline depth**: increase `Stages` beyond 2 to hide longer L3
  latency on sequence-length-limited shapes.
