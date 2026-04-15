# Tier 2 — Multi-row Q4_K kernels (attempt)

> **Date**: 2026-04-13
> **Goal**: match llama.cpp's bandwidth utilization (~55-80%) vs our ~22%.
> **Approach**: port multi-row-per-warp pattern from llama.cpp `mmvq_q4_K`.
> **Result**: ❌ both `mr4` and `mr2` variants show no speedup (mr4 is regression).

## What was implemented

Two opt-in Q4_K matmul kernel variants:

- `matmul_q4_k_mr4.cu` — 4 rows per warp, blockDim=128 (4 warps × 4 rows = 16 rows/block)
- `matmul_q4_k_mr2.cu` — 2 rows per warp, blockDim=256 (8 warps × 2 rows = 16 rows/block)

Both share the same idea: amortize FP32 input reads across multiple output rows, increase per-lane work to hide memory-load latency via instruction-level parallelism.

Wired through `Q4_KCudaTensor` with flags `cuda.q4k.mr4=true` / `cuda.q4k.mr2=true`. Default OFF (preserves baseline).

## Measured results (Llama-3.2-1B Q4_K_M, RTX 4050)

Five-run alternating A/B (eliminates JIT warmup bias):

| Run | baseline | mr4 | Δ |
|---|---:|---:|---:|
| 1 | 31.9 | 27.1 | -15% |
| 2 | 29.7 | 21.9 | -26% |
| 3 | 25.0 | 19.2 | -23% |
| 4 | 18.7 | 17.6 |  -6% |
| 5 | 16.1 | 10.4 | -35% |
| **avg** | 24.3 | 19.2 | **-21%** |

Three-run mr2 vs baseline:

| Run | baseline | mr2 |
|---|---:|---:|
| 1 | 38.0 | 24.6 |
| 2 | 28.3 | 35.6 |
| 3 | 30.6 | 30.7 |

PPL preserved (1.00 in all runs — output mathematically identical to baseline).

Variance is high (~30% between runs) due to thermal throttling on laptop GPU.
On best-of-3: baseline 38.0 vs mr2 35.6 = **-6%**.

## Why it didn't work

**Register pressure** is the most likely cause:
- Baseline: 1 partial sum + temporary scales = ~10 registers per lane
- mr4: 4 partial sums + 4 scale sets + 8 float4 input vectors = ~50+ registers per lane
- mr2: 2 partial sums + 2 scale sets + 8 float4 inputs = ~30 registers per lane

When register count exceeds the per-thread limit, the compiler spills to local memory (HBM access per spill) — which destroys throughput. NVCC profiling tools (`ncu --metrics smsp__sass_thread_inst_executed_op_global_st_pred_on.sum`) would confirm this, but WSL blocks NVIDIA performance counters.

Other contributing factors:
- Lower SM occupancy from larger per-warp footprint
- NVRTC (JIT) compiler is less aggressive than NVCC (offline) at register allocation

## What WOULD work to match llama.cpp

The structural gap (22% → 55-80% bandwidth) requires deeper changes:

1. **Switch from NVRTC to pre-compiled PTX**: ship pre-built PTX binaries instead of source-level JIT. NVCC offline optimization (especially register allocation, instruction scheduling) is much better.

2. **WMMA tensor cores**: write `wmma::fragment<>`-based kernels with FP16 accumulator. Requires:
   - Pre-dequantize Q4_K → FP16 in shared memory
   - 16×16 tile-based matmul
   - Significant rewrite of matmul kernels
   - Estimated effort: 2-3 weeks for Q4_K, 1-2 months for all quant types

3. **`cp.async` for memory/compute overlap** (Ampere+): pipeline HBM loads with compute via doubled shared memory buffers. Requires kernel architecture change.

4. **MMQ-style kernels**: llama.cpp's MMQ kernels for batched workloads are completely different from mmvq (matrix-vector). For batch=1 inference, mmvq is what they use — and our kernel is structurally similar. The gap is in compiler optimization quality.

## Honest verdict

**We are NOT going to easily match llama.cpp on RTX 4050.** The remaining 2-3× gap requires either:

- **Engineering investment**: 2-3 weeks minimum for proper WMMA + cp.async kernels, with uncertain payoff (NVRTC may still produce sub-optimal code vs NVCC)
- **Switch to pre-compiled binaries**: abandon NVRTC for NVCC offline compilation. Major architectural change to the build/distribution.

For the project's current trajectory (zero external deps, NVRTC JIT), we're at a **practical plateau**. The current Tier 1 + Tier 2 attempts show that incremental kernel tweaks don't move the needle on this hardware.

## What's preserved

All experimental kernels remain in the codebase as opt-in:
- `cuda.q4k.mr4=true` (4 rows/warp — slower)
- `cuda.q4k.mr2=true` (2 rows/warp — break-even at best)

Default behavior **completely unchanged**. Future work on different GPUs (where register pressure may be less of an issue) might benefit from these.

## What we learned

1. **Profile-mode timing doesn't predict graph-mode wins** (Tier 1 lesson, reconfirmed).
2. **Multi-row-per-warp helps only when register pressure stays low** — our Q4_K kernel is too register-hungry for this trick.
3. **Variance on consumer laptop GPUs is huge** (±30% between runs) due to thermal throttling. Reliable A/B requires 5+ runs.
4. **NVRTC is a real performance ceiling**. The same kernel source compiled by NVCC offline would likely be faster.
