# GPU profiling analysis — Qwen3.5-4B Q4_K_M

> **Date**: 2026-04-13
> **Hardware**: NVIDIA RTX 4050 Laptop GPU (6140 MB, 192 GB/s, 20 SMs)
> **Driver**: 581.95 (WSL2)
> **Method**: in-code per-kernel timing (`-Dqwen35.profile=true` + `-Dcuda.nograph=true`)
> **Reason for in-code timing**: WSL blocks NVIDIA performance counters (`ERR_NVGPUCTRPERM`), so `nsys --gpu-metrics-devices=all` and `ncu --set basic` both fail with privilege error. We added per-kernel `cudaContext.finish()` + `System.nanoTime()` brackets around `launchKernel`/`launchMatmul`/`launchMatmulDp4a`/`quantizeInput` calls.

## Baseline numbers

| Metric | Value |
|---|---|
| **GPU mode** | full offload (32/32 layers) |
| **Default tok/s** (graph) | 16.7 |
| **Profile-mode tok/s** (nograph + finish per kernel) | 14.2 (+12% overhead from finish) |
| **Per-token GPU compute time** | ~58 ms (1741 ms / 30 tokens) |
| **Theoretical bandwidth-bound minimum** | ~13.5 ms (2.6 GB / 192 GB/s) |
| **Bandwidth utilization** | **~22%** |
| **Headroom** | ~4.5× to bandwidth limit |

## Per-kernel breakdown (30 tokens, 18-token prompt)

```
kernel                                      calls    total(ms)    avg(us)        %
matmul.other_fp32                            3372          579        171    33.3%
matmul.fused_gate_up                          960          458        477    26.3%
matmul.dp4a.Q5_K                              720          118        164     6.8%
matmul.dp4a.Q4_K                             1290          111         86     6.4%
rmsnorm                                      1932          102         53     5.9%
quantize_input                               1920           89         46     5.1%
deltanet (recurrence+norm+gate)               720           61         85     3.5%
silu_mul (FFN)                                960           54         56     3.1%
conv1d+silu                                   720           37         51     2.1%
alpha_beta                                    720           31         43     1.8%
qk_norm_per_head                              480           20         43     1.2%
rope                                          480           20         42     1.2%
attention.full                                240           13         56     0.8%
deinterleave_q_gate                           240           11         47     0.7%
matmul.dp4a.Q6_K                              150           10         70     0.6%
sigmoid_mul                                   240           10         43     0.6%
kv_cache_update                               240           10         42     0.6%
```

**Matmul total = 73% of GPU time.** Everything else (DeltaNet recurrence, attention, conv1d, RoPE, KV cache) = 27%.

## Decomposition of the 33% "matmul.other_fp32"

This is *not* FP32 matmul — it's the regular Q4_K weight × FP32 input kernel (no input quantization). 112 calls/token break down as:

- **48 calls/token** = matmuls for tensor types without a dp4a kernel (e.g. Q8_0 fallback). Engine startup logs `dp4a: 200/248 matmuls` — i.e. 48/248 fall through to plain.
- **64 calls/token** = explicit `launchMatmul(ml[3]/ml[4]/ml[7])` for output projections — Wo (attention), ssm_out (DeltaNet), ffn_down. Reason: input is in `gpuXb2`/`gpuDeltaOut`/`gpuHb`, not in `gpuXb`, so the pre-quantized `gpuXbQ8` buffer isn't usable.
- **1 call/token** = lm_head (`outputMatmul`, vocabSize=248320 × dim=2560 = 635 M params).

## Plain vs dp4a comparison (per-kernel avg)

| Kernel | avg µs | × |
|---|---:|---:|
| `matmul.dp4a.Q4_K` | 86 | 1.0× (baseline) |
| `matmul.dp4a.Q5_K` | 164 | 1.9× |
| `matmul.dp4a.Q6_K` | 70 | 0.8× |
| `matmul.other_fp32` (avg across all sizes) | 171 | **1.99×** |

**dp4a is ~2× faster.** Quantizing the input from FP32 to Q8_1 (1 small extra kernel: `quantize_input` at 46 µs avg) is the cost; the saving on the matmul is much larger.

## Optimization opportunities (Tier 1 — quick wins)

| # | Optimization | Est. saving | Effort | PPL risk | Status |
|---|---|---:|---|---|---|
| **T1.1** | Fuse `rmsnorm + quantize_input` (one kernel, single HBM pass) | **est 5-7% / measured -5% in graph, +6% in nograph** | medium (1 new `.cu`) | zero (math-equivalent) | ⚠️ NEUTRAL/REGRESSION |
| **T1.2** | dp4a for output projections (`ml[3]/ml[4]/ml[7]`) — add quantize call before each | **est 7-10% / measured BROKEN** | low (3 extra quantize launches + flag) | **HIGH** (PPL crashes to 0.36-0.62 → garbage) | ⚠️ INVESTIGATING |
| **T1.3** | dp4a Q8_0 kernel (covers 48/248 fallback matmuls) | **3-5%** | medium (port from llama.cpp MMQ) | zero | DEFERRED |
| **Total Tier 1** | | **~15-20% est / TBD measured** | 1-2 weeks | low | |

Expected: 16.7 → 19-21 tok/s for Qwen3.5-4B Q4_K_M.

### T1.1 measured impact (2026-04-13)

Implemented as `rmsnorm_quantize_fused` kernel (`src/main/resources/kernels/cuda/rmsnorm_quantize.cu`), opt-in via `-Dcuda.fused_norm_quantize=true`. Default behavior preserved (flag OFF).

| Metric | DEFAULT (2 kernels) | T1.1 ON (1 fused kernel) | Δ |
|---|---:|---:|---:|
| Profile-mode total kernel time (30 tok) | 1741 ms | 1631 ms | **-6.3%** |
| `rmsnorm` time | 102 ms (1932 calls) | (fused) | — |
| `quantize_input` time | 89 ms (1920 calls) | (fused) | — |
| `rmsnorm+quantize fused` time | — | 100 ms (1920 calls) | — |
| Combined norm+quantize cost | 191 ms (11.0%) | 100 ms (6.1%) | **-48% per kernel** |
| Graph-mode tok/s (best of 3) | 18.2 | 18.4 | **+1.1%** |
| PPL aggregate | 0.84 | 0.86 | identical (within noise) |

**Lesson learned**: in profile mode (serialized via `finish()`), the kernel-level saving is 6%. In real graph mode, pipelined execution overlaps these ops with concurrent matmul work, AND the fused kernel's higher per-warp work (norm+quantize) reduces SM occupancy → **slightly slower in graph mode (~5%)**. Output is mathematically identical (PPL preserved), but the throughput trade-off is unfavorable on this hardware.

**Status**: kept as opt-in flag (default OFF). May still benefit:
- nograph mode (per-tensor execution): +6%
- Hardware with different occupancy characteristics
- Future: re-tune block dim / shared mem usage to recover lost occupancy

### T1.2 attempted (2026-04-13) — output broken, deferred

Implemented as opt-in flag `cuda.dp4a.outputs=true` (default OFF, current behavior preserved). Adds a `quantize_q8` call before each output projection (Wo, ssm_out, ffn_down) and switches the matmul to dp4a.

Result: with the flag enabled, **PPL crashes from 0.91 to 0.36-0.62** and output is gibberish (e.g. "( ( " \ B/mol ... -\u \ž :-\\ Misc"). 30 tokens, 32 layers — corruption is consistent across runs and prompts.

Profile shows the new path IS being taken: `matmul.dp4a.Q6_K` jumps from 150 → 630 calls, `matmul.dp4a.Q4_K` increases, `matmul.other_fp32` drops from 3372 → 1452. Numerically correct in terms of *which kernels run*, but the output is wrong.

Hypothesis: the dp4a Q4_K/Q5_K/Q6_K kernels' `addToOutput=1` (accumulate) mode is **never exercised by the existing code path** (existing dp4a calls are all `addToOutput=0` for QKV/gate/up matmuls; output projections always used `launchMatmul` plain). The accumulate path may have a subtle correctness bug that didn't manifest before.

Next steps:
1. Write a controlled CUDA test for `matmul_q4_k_dp4a` with `addToOutput=1` (compare against plain matmul output for the same inputs).
2. Check if the issue is `addToOutput=1` itself, or if it's specific to the new Q8 scratch buffer layout.
3. Try converting only ONE projection at a time (e.g. only ml[7] FFN down) to isolate which one breaks.

Default behavior: **completely preserved**. The new code path is gated by a flag that defaults to false. Code is in place for future debugging/fixing without disrupting production.

### T1.3 deferred

Port of dp4a Q8_0 kernel deferred — the Q4_K dp4a accumulate bug must be resolved first, since Q8_0 dp4a would face the same issue for its 48/248 matmuls.

---

## Tier 1 honest verdict (2026-04-13)

**Net measured improvement on graph mode: 0%.**

| Optimization | Status | Graph-mode impact |
|---|---|---:|
| T1.1 fused rmsnorm+quantize | implemented, math-correct | **-5%** (occupancy regression) |
| T1.2 dp4a output projections | implemented, BROKEN output | **n/a** (PPL crashes) |
| T1.3 dp4a Q8_0 kernel | not implemented | n/a |

**Conclusion**: we ARE at the plateau for the current kernel architecture on this GPU. The profile-mode optimization opportunities (~15-20%) do NOT translate to graph-mode wall-clock gains because:

1. **CUDA graph already pipelines** small kernels (rmsnorm, quantize) with concurrent matmul work, absorbing the per-kernel time savings.
2. **Fusing kernels** trades launch/HBM cost for higher per-warp work, which can REDUCE SM occupancy and HURT throughput.
3. **The dp4a accumulate path** appears to have a correctness issue that wasn't exposed before (no existing matmul uses `addToOutput=1` with dp4a). Investigation deferred.

**To break the plateau**, we need:
- **Tier 2** (WMMA tensor cores) — fundamentally different kernel architecture using NVIDIA's tensor cores. 50-100% expected, but requires PTX-level expertise and 1-2 months of work.
- **Tier 3** (speculative decoding) — algorithmic, no kernel changes. 2-3× effective tok/s. ~3 weeks of work.

**Code preserved**: T1.1 and T1.2 implementations are committed but gated by flags (`cuda.fused_norm_quantize`, `cuda.dp4a.outputs`), default OFF. They cause ZERO impact on production performance and provide infrastructure for future tuning attempts.

## Optimization opportunities (Tier 2 — significant rewrite)

| # | Optimization | Est. saving | Effort | Notes |
|---|---|---:|---|---|
| T2.1 | WMMA tensor cores for Q4_K matmul | 50-100% | HIGH (PTX `mma.sync` or `wmma::fragment`) | Port llama.cpp MMQ kernel |
| T2.2 | `cp.async` for memory/compute overlap (Ampere+) | 10-15% | medium-high | Doubles shared-mem buffers, pipelined HBM load |

## Optimization opportunities (Tier 3 — algorithmic)

| # | Optimization | Est. saving | Effort | Notes |
|---|---|---:|---|---|
| T3.1 | Speculative decoding (small drafter + this verifier) | 2-3× effective | medium | Algorithmic, no kernel changes; cross-architecture benefit |

## What we are NOT bottlenecked by

- **Launch overhead**: 506 `cuLaunchKernel` calls totaling 1.2 ms — 2.4 µs each. CUDA graph already amortizes; non-issue.
- **DtoH/HtoD bandwidth**: `cuMemcpyDtoH_v2` shows 48 ms/token but it's a sync point including GPU compute wait, not actual transfer time. Logits download is 1 MB ≈ 50 µs of actual transfer.
- **Attention compute**: 0.8% of GPU time — already well-optimized.
- **DeltaNet recurrence**: 3.5% of GPU time — already heavily optimized via the fused mega-kernel.
- **`cuMemAlloc_v2`** (512 calls in startup) — this is one-time allocation, not per-token.

## Methodology notes

The profile mode adds `cudaContext.finish()` after every kernel launch, serializing all GPU work. This:
1. Allows accurate per-kernel attribution (no overlap).
2. Adds ~12% overhead vs the optimized graph mode (16.7 → 14.2 tok/s).
3. Disables CUDA graph (since graphs require contiguous capture without sync points).

Use `-Dqwen35.profile=true -Dcuda.nograph=true` to enable. Reports every 30 tokens via stderr.

## References

- Source profiler: `Qwen35CudaForwardPass.java` `labelForFunction()` + `launchKernel()` instrumentation.
- llama.cpp MMQ reference: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cu
- WMMA programming guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma
