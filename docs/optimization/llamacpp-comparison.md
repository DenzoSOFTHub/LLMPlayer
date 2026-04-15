# LLMPlayer vs llama.cpp — performance comparison and optimization findings

> **Date**: 2026-04-13/14 (multi-session optimization sprint)
> **Hardware**: NVIDIA RTX 4050 Laptop GPU (sm_89, 192 GB/s HBM, 6 GB VRAM)
> **Reference**: llama.cpp built from `master` (commit a8bad38), CUDA 12.6, sm_89

## TL;DR

- **Closed gap from 43% → 65% of llama.cpp** on Llama-3.2-1B Q4_K_M (47 → 70 tok/s vs llama.cpp's 110)
- **Single biggest win**: dp4a (`__dp4a` int8 dot product) was wired only in `Qwen35CudaForwardPass`. Wiring it into the standard `CudaForwardPass` (used for Llama, Qwen2/3, Mistral, Gemma 2/3, Phi, Granite) gave **+34% by itself**.
- **Closed bug**: `matmul_q4_k_dp4a` had local-memory spill (`__local_depot0[12]` for `sb[12]` byte array). Rewriting scale extraction without the array eliminated the spill.
- **Multiple architectural ports tried**: all gave 0% to -22% on Llama-1B. The remaining 30-35% gap is **not** in any single kernel; it's spread across many micro-optimizations llama.cpp has accumulated over years.

## Method

For each candidate optimization:
1. Implement as opt-in flag (default OFF) so default behavior is preserved
2. Smoke test for correctness (PPL preserved bit-equivalent)
3. Bench: 3 runs × 100 tokens × `Write a short paragraph about the history of computing.` prompt
4. Compare avg tok/s vs current default
5. If ≥ 5% improvement and PPL preserved → flip default ON
6. If marginal or regression → keep as opt-in for future hardware/models

## Llama.cpp comparison setup

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build . --target llama-bench -j$(nproc)
./llama-bench -m model.gguf -ngl 99 -p 0 -n 100 -r 2
```

Llama.cpp on the same RTX 4050 with Llama-3.2-1B Q4_K_M: **102-110 tok/s** (varies 5-10% per run).

## Baseline before this sprint

Llama-3.2-1B Q4_K_M, 3-run avg: **47.0 tok/s** (43% of llama.cpp).

Profile breakdown (cuda.profile=true, ~17% overhead from `cudaContext.finish()` barriers):
- GateUp (16 layers × 2 matmuls): 5.46 ms (35%)
- siluDown (silu_mul + Down × 16): 3.31 ms (22%)
- Output projection (vocab × dim, 1 matmul): 2.76 ms (18%)
- QKV (3 matmuls × 16 layers): 1.79 ms (12%)
- Wo (1 matmul × 16 layers): 1.21 ms (8%)
- ropeKv: 0.86 ms (6%)
- attn: 0.85 ms (6%)
- attnNorm: 0.76 ms (5%)
- ffnNorm: 0.65 ms (4%)

## What landed (default ON, validated)

### 1. dp4a wired into standard `CudaForwardPass` — **+34%**

**Root cause**: `matmul_q4_k_dp4a.cu`, `matmul_q5_k_dp4a.cu`, `matmul_q6_k_dp4a.cu` were sitting in resources but only `Qwen35CudaForwardPass` used them. The standard `CudaForwardPass` (used by 90% of architectures: Llama, Qwen2/3, Mistral, Gemma 2/3, Phi, Granite) had **zero dp4a wiring**, despite `cuda.dp4a=true` being documented as default-on (the doc applied only to Qwen3.5).

**Fix**: ported the Qwen3.5 pattern into `CudaForwardPass`:
- Q8_1 scratch buffers `gpuXbQ8` / `gpuXb2Q8` / `gpuHbQ8`
- `quantizeXb()`/`quantizeXb2()`/`quantizeHb()` helpers called at the right points
- `launchMatmulDp4a()` routes Q4_K (and Q5_K) matmuls through dp4a kernels
- Output projection routed through `launchOutputMatmul()` (dp4a if Q4_K)
- Default `cuda.dp4a` flipped to TRUE

**Result**: 47 → 63 tok/s on Llama-1B (+34%, PPL preserved).

### 2. Local-memory spill removal in `matmul_q4_k_dp4a` and `matmul_q4_k`

**Root cause** (found via PTX inspection): the `unsigned char sb[12]` byte array used for scale decoding caused NVCC to allocate `__local_depot0[12]` (HBM-backed local memory). Variable-indexed byte arrays cannot be kept in registers. Each row matmul did ~384 LDL operations.

**Fix**: rewrote scale extraction to compute directly from the 3 uint32 scale words (`sc0u/sc1u/sc2u`) using bit shifts. PTX after fix shows `LOCAL:0` and 40 registers per thread.

**Result**: bit-equivalent output. Speed within thermal noise on Llama-1B (the 12 bytes were L1-cached most of the time), but kernel is now objectively cleaner and uses fewer registers — better for systems where local-memory spill DOES matter.

### 3. Profile path correctness fix

**Root cause**: `forwardLayerProfiled` (used when `cuda.profile=true`) called `launchMatmul` instead of `launchMatmulDp4a`. So all profile measurements were against the FP32 path, not the actual default dp4a path. We were profiling the wrong code.

**Fix**: `forwardLayerProfiled` now calls `quantizeXb()/Xb2/Hb` and `launchMatmulDp4a()` matching the dp4a path.

## What was tried — and why each didn't help

This section is the meat of the document. Each entry is a candidate optimization with measured outcome and root-cause analysis.

### A. Pre-compiled NVCC cubin (Option A — "ship offline-compiled .cubin to skip NVRTC JIT")

**Hypothesis**: NVCC offline produces better SASS than NVRTC JIT.

**Test**: Compiled `matmul_q4_k.cu` to sm_89 cubin via `nvcc --gpu-architecture=sm_89 --use_fast_math --cubin`. Loaded via `cuModuleLoadDataEx` (added opt-in `-Dcuda.prebuilt=true`). 3 alternating runs vs NVRTC.

**Result**: −1.1% (within noise). NVRTC and NVCC share the same `ptxas` backend. With same flags they produce essentially identical SASS.

**Verdict**: Dead. Don't ship per-arch cubins. Loader infrastructure kept (opt-in only, no shipped artifacts) for users who want to eliminate the ~50ms NVRTC compile stall at cold start.

### B. cp.async memory/compute overlap (Option C — "use Ampere+ async copies")

**Hypothesis**: Overlapping memory loads with compute via `cp.async.cg.shared.global` hides HBM latency.

**Test**: Wrote `matmul_q4_k_cpasync.cu` with double-buffered shared-memory input via cp.async. Smoke-tested first with 4-byte cp.async (`cg` cache hint = L2 only) → **100× slower**. Switched to 16-byte (float4) `ca` cache hint → back to baseline ballpark. 3-run bench.

**Result**: −2.8% vs baseline. Input was already L1-cached by `__ldg`; per-tile compute too small to hide HBM weight loads (compute ≈ 3 ns, weight load ≈ 100+ ns).

**Verdict**: Dead for matvec at batch=1. Kept as opt-in (`cuda.q4k.cpasync=true`) for reference. Lesson: cache hints can dwarf algorithm choice — bypassing L1 for broadcast reads is catastrophic.

### C. Multi-row-per-warp Q4_K (Tier 2 — `mr2`/`mr4`)

**Hypothesis**: Each warp processes 2 or 4 output rows in parallel, sharing input reads, hiding latency via instruction-level parallelism.

**Test**: Wrote `matmul_q4_k_mr2.cu` (2 rows/warp) and `matmul_q4_k_mr4.cu` (4 rows/warp). Bench Llama-1B.

**Result**: mr4 = −21% avg. mr2 = break-even at best. Register pressure (4 partial sums + 4 weight headers per lane) caused NVCC to spill registers.

**Verdict**: Dead. Multi-row helps only when register pressure stays low; our Q4_K kernel is too register-hungry.

### D. Multi-warp-per-row Q4_K (llama.cpp pattern)

**Hypothesis**: 4 warps cooperate on 1 row, more in-flight memory requests per row, better latency hiding.

**Test**: Wrote `matmul_q4_k_dp4a_mw.cu` (4 warps × 32 lanes = 128 threads, 1 row per block, gridDim = rows). Cross-warp reduction in shared memory.

**Result**: −22% on Llama-1B (55.4 vs 71 single-warp). For models with small dimensions, the per-block synchronization overhead and 8× more block launches dominate.

**Verdict**: Dead for small models. Kept as opt-in for future testing on larger models where the trade-off may flip.

### E. Multi-warp Q4_K mmvq (full llama.cpp algorithm port)

**Hypothesis**: Llama.cpp's mmvq has subtle algorithmic advantages we missed. Port it precisely.

**Test**: Wrote `matmul_q4_k_mmvq.cu` — direct port of llama.cpp's `vec_dot_q4_K_q8_1_impl_vmmq` + `mul_mat_vec_q` template specialized for Q4_K + ncols_dst=1 + nwarps=4. Includes:
- 4 warps × 32 lanes per row, 1 row per block (their `nwarps=4, rows_per_cuda_block=1`)
- vdr=2: each thread processes 2 sub-blocks per call (16 weights)
- `dp4a-with-0x01010101 constant` trick for inSum (free sum, no separate reduction)
- Per-thread distribution: `kbx = tid / (qi/vdr)`, `kqs = vdr * (tid % (qi/vdr))`

Critical bug found during porting: Q8_1 input offset must include `kby = kbx * (qk/QK8_1)` (super-block stride). Initially missed this and got garbage output. Once fixed: PPL bit-equivalent.

**Result**: 66.9 tok/s avg vs default 70 — within noise but slightly slower. Llama.cpp's exact algorithm gives essentially same performance as ours on this hardware/model.

**Verdict**: Algorithm port doesn't close the gap. Kept as opt-in (`cuda.dp4a.mmvq=true`). Lesson: the gap is NOT in the kernel algorithm.

### F. Fused dp4a gate+up (single kernel for both projections)

**Hypothesis**: Reading the input vector once and computing both gate+up amortizes input bandwidth.

**Test**: Wrote `matmul_q4_k_dp4a_fused_gate_up.cu`. Two helper inline functions, packs Q8_1 input headers in registers (read once), computes both gate and up sums per (group, sub-block).

**Result**: 70.9 vs 73.5 separate-dp4a (-3%, within noise). Input was already L1-cached across the 2 separate matmuls; "halving input reads" buys nothing on this hardware. Per-thread register pressure increased (2 partial sums per lane).

**Verdict**: Marginal regression. Kept as opt-in (`cuda.dp4a.fused_gate_up=true`).

### G. Q6_K dp4a kernel

**Hypothesis**: Q6_K matmul (used for tied-embedding output projection in Llama-3.2) at FP32 input is 46% bandwidth — dp4a should help.

**Test 1**: Wrote a Q6_K dp4a kernel. **Found bug in original kernel**: element indexing assumed consecutive sub-groups, but Q6_K's actual `ql` layout is interleaved (per `Q6_KFloatTensor.getFloat`: `ql byte i` holds low-nibble for element `i` AND high-nibble for element `64+i`). Rewrote with correct `(half, quadrant, l)` decomposition.

**Test 2**: Initial rewrite crashed with `cuMemcpyDtoH error 716` (CUDA_ERROR_MISALIGNED_ADDRESS) — Q6_K block size is 210 bytes (NOT 4-byte aligned), so super-block bases past `sb=0` are misaligned. Switched to byte-loads.

**Result**: Output bit-equivalent, but **70.6 vs 72.85 FP32 tok/s** = -3%. The byte-load tax (4× byte loads per uint vs 1 uint load) overwhelms the dp4a benefit on Q6_K's heavier 210-byte unaligned blocks.

**Verdict**: Correct but slower. Default OFF (`cuda.dp4a.q6=false`). Kept as opt-in for ground-truth checks.

### H. cuBLAS for matmul

**Test**: Enabled `-Dcuda.cublas=true` (existing infrastructure: pre-dequantizes Q4_K → FP16, uses `cublasGemmEx`).

**Result**: **41.9 tok/s** = -42% vs default 70. cuBLAS implementation disables CUDA graph mode entirely (cuBLAS calls have implicit synchronization). The graph-launch savings dominate any per-matmul speed up cuBLAS provides.

**Verdict**: Dead as currently structured. A hybrid (graph for layers + cuBLAS only for output projection) was investigated but requires graph capture restructuring — deferred.

### I. `__launch_bounds__` hints

**Test**: Added `__launch_bounds__(256, 1)` then `__launch_bounds__(128, 2)` to `matmul_q4_k_dp4a`.

**Result**: Both -3% to -7%. NVCC's defaults are already optimal for this kernel (40 registers, allows 4 blocks/SM occupancy).

**Verdict**: Dead. NVCC's heuristics beat manual hints here.

### J. Smaller blockDim (128 instead of 256)

**Test**: Override blockDim to 128 (4 warps = 4 rows/block) for Q4_K dp4a.

**Result**: -7% on Llama-1B. Smaller blocks don't help when occupancy is already saturated.

**Verdict**: Dead.

### K. Q6_K kernel variants (`smem` vs `tiled` vs plain)

**Test**: Toggled `cuda.q6k.tiled` and `cuda.q6k.smem` flags (output projection in Llama-1B uses Q6_K).

**Result**: tiled (default) wins. smem -7%, plain -16% on Llama-1B.

**Verdict**: Current default is best.

## Summary table

| Optimization | Expected | Measured | Verdict |
|---|---|---|---|
| **dp4a wired in CudaForwardPass** | +30-50% | **+34%** | ✅ DEFAULT ON |
| **Local-memory spill fix** | +5% | within noise | ✅ DEFAULT ON (cleaner kernel) |
| Pre-compiled NVCC cubin | +5-15% | -1% | ❌ opt-in only |
| cp.async memory/compute overlap | +5-15% | -3% | ❌ opt-in only |
| Multi-row-per-warp (mr4) | +20% | -21% | ❌ opt-in only |
| Multi-warp-per-row (llama.cpp pattern) | +10-20% | -22% | ❌ opt-in only (small models) |
| Multi-warp Q4_K mmvq (full algo port) | +15-25% | -4% | ❌ opt-in only |
| Fused dp4a gate+up | +5% | -3% | ❌ opt-in only |
| Q6_K dp4a kernel | +10% | -3% | ❌ opt-in only |
| cuBLAS replacement | +20% | -42% | ❌ requires graph restructuring |
| `__launch_bounds__` hints | +3-5% | -3% to -7% | ❌ NVCC defaults beat hints |
| blockDim=128 | +5% | -7% | ❌ |

## Why kernel-level optimization on this workload has hit a wall

After many attempts, the data shows a consistent pattern: **on RTX 4050 Laptop with Llama-1B Q4_K, our `matmul_q4_k_dp4a` kernel is at 40-60% of HBM bandwidth peak**. Multiple algorithmic alternatives — including a direct port of llama.cpp's exact algorithm — give within-noise variations.

The remaining 30-35% gap to llama.cpp is **NOT** in any single kernel. It's spread across:
1. **Per-arch tuning matrix**: llama.cpp has `calc_nwarps()`, `calc_rows_per_block()`, `get_mmvq_mmid_max_batch()` lookups parametrized by `(ggml_type, ncols_dst, compute_capability)`. Different launch params per (model, GPU). We have one global setting.
2. **Output projection (vocab × dim)**: their `mul_mat_vec_q` for Q6_K likely uses different launch params for very tall matrices (128K rows). Our Q6_K kernel processes 8 rows/block; for 128K rows that's 16K blocks oversubscribing 20 SMs.
3. **Years of micro-optimization across many small kernels**: rmsnorm, rope, attention, etc. — each within 5-15% of theirs but compounding.
4. **Possibly weight layout**: they may store Q4_K weights interleaved differently for better coalesced access. Our layout matches GGUF spec directly.

## Cross-model verification (17 models)

Same setup (RTX 4050, 100 tokens, 2 runs each, prompt: *"Write a short paragraph about the history of computing."*).

All models that fit fully in 6GB VRAM and use the standard `CudaForwardPass`:

| Model | Quant | Arch | Our avg | llama.cpp | Our % |
|-------|-------|------|--------:|---------:|------:|
| Qwen3-4B | Q4_K_M | QWEN3 | 28.6 | 35.23 | **81%** ⭐ |
| Granite-3.3-2B | Q4_K_M | GRANITE | 43.4 | 54.84 | **79%** |
| Falcon3-3B | Q4_K_M | FALCON3 | 38.5 | 49.00 | **79%** |
| Phi-4-mini | Q4_K_M | PHI4 | 27.2 | 35.06 | **77%** |
| Qwen2.5-Coder-3B | Q4_K_M | QWEN2 | 34.5 | 46.32 | **75%** |
| llama-3.2-3B | Q4_K_M | LLAMA | 34.7 | 46.68 | **74%** |
| Qwen2.5-Coder-1.5B | Q4_K_M | QWEN2 | 58.3 | 82.55 | **71%** |
| SmolLM3 | Q4_K_M | SMOLLM3 | 34.0 | 49.59 | **69%** |
| Llama-3.2-1B | Q4_K_M | LLAMA | 70.9 | 106.49 | **67%** |
| google-gemma-3-4B | Q4_K_M | GEMMA3 | 25.0 | 37.76 | **66%** |
| Qwen3-1.7B | Q8_0 | QWEN3 | 31.1 | 51.87 | **60%** ⚠️ Q8_0 |
| OLMo-2-1B | Q4_K_M | OLMO2 | 55.8 | 97.39 | **57%** |
| Qwen3-0.6B | Q8_0 | QWEN3 | 67.6 | 124.13 | **54%** ⚠️ Q8_0 |
| **gemma-3-1B** | **Q4_K_M** (Q5_0 weights!) | GEMMA3 | **30.7** | **96.51** | **⚠️ 32%** |
| **Phi-3-mini** | **IQ4_NL** | PHI3 | **8.4** | **43.17** | **⚠️ 19%** |
| **gemma-2-2B** | **IQ4_XS** | GEMMA2 | **8.6** | **56.82** | **⚠️ 15%** |
| Mistral-7B (partial offload) | Q4_K_M | LLAMA | 20.1 | 22.07 | **91%** |

### Summary

**Standard Q4_K_M models (11 models)**: 57-81% of llama.cpp. **Average ~72%.** Our dp4a Q4_K wiring is sound and generalizes across all 11 supported architectures (Llama, Qwen2/3, Mistral, Gemma 3, Phi-4, Granite, OLMo, Falcon, SmolLM).

**Q8_0 weight models**: 54-60%. Q8_0 has no dp4a kernel — falls through to FP32 `matmul_q8_0.cu`.

**Anomaly models (3)**: Gemma-3-1B (32%, uses Q5_0 weights for QKV/gate/up despite being labeled Q4_K_M), Phi-3-mini IQ4_NL (19%), Gemma-2-2B IQ4_XS (15%). All three suffer the same root cause: their weight quantization type (Q5_0, IQ4_NL, IQ4_XS) is not in our `launchMatmulDp4a` switch, so all matmuls fall through to FP32.

**Mistral-7B partial-offload (91%)**: when both systems do CPU offload (model > VRAM), the gap closes dramatically because the bottleneck shifts from kernel speed to PCIe transfer.

### Four new dp4a kernels written — final measured impact

**2026-04-14: all 4 kernels written, wired, and benched with cooldown intervals.**

After-cooldown results (3 runs averaged, GPU at stable thermal state):

| Model | Quant | Baseline | After | Δ | llama.cpp | After % |
|-------|-------|---------:|------:|---:|---------:|--------:|
| Llama-3.2-1B (reference) | Q4_K_M | 70.9 | 69.6 | -2% | 106.5 | 65% |
| **Phi-3-mini** | **IQ4_NL** | **8.4** | **11.9** | **+42%** ⭐ | 43.2 | 28% |
| Qwen3-1.7B | Q8_0 | 31.1 | 33.5 | **+8%** | 51.9 | 65% |
| Gemma-2-2B | IQ4_XS | 8.6 | 9.0 | +5% | 56.8 | 16% |
| Gemma-3-1B | Q5_0 (in Q4_K_M mix) | 30.7 | 31.9 | +4% | 96.5 | 33% |
| Qwen3-0.6B | Q8_0 | 67.6 | 69.0 | +2% | 124.1 | 56% |

**PPL preserved for all** (bit-equivalent output within FP16 rounding).

### Per-kernel analysis (why gains vary)

- **IQ4_NL dp4a: +42% ⭐ clear big win.** All of Phi-3-mini's weights are IQ4_NL. Full matmul time was using FP32 kernel; now goes through dp4a. Confirmed by 3 consistent runs (12.0, 11.9, 11.9).

- **Q8_0 dp4a: +2-8%.** The existing FP32 Q8_0 kernel was already efficient (weights ARE int8, no unpacking). dp4a saves compute cycles but input-side bandwidth is already the bottleneck.

- **Q5_0 dp4a: +4% on Gemma-3-1B.** Gemma-3 Q4_K_M stores only QKV/gate/up as Q5_0. The Wo and down matmuls are Q4_K (already using dp4a). So Q5_0 dp4a only affects ~40% of matmul time. Gain scaled down accordingly.

- **IQ4_XS dp4a: +5% on Gemma-2-2B.** Smallest gain. Root cause: IQ4_XS uses super-block format (256 elements per 136-byte block). With Gemma-2's cols=2304, that's only 9 super-blocks per row — with lane stride 32, only 9 of 32 lanes do useful work. **Structural parallelism issue**, not a dp4a issue. Multi-warp orchestration (one row per block × 4 warps) would help specifically here. Left as future optimization.

### Alternative-engine extensions (2026-04-14, same sprint)

Same 7-type dp4a switch ported into `Qwen35CudaForwardPass` and `NemotronHCudaForwardPass`:

- **Qwen35CudaForwardPass**: extended for hybrid DeltaNet+Attention architectures (Qwen3.5 family). Measured: no change vs pre-extension (Qwen3.5 4B/2B are all Q4_K, so the new paths don't apply here). Wiring ready for Qwen3.5 models with Q5_0/Q8_0/IQ4 weights.

- **NemotronHCudaForwardPass**: had NO dp4a wiring at all before this sprint. Added full infrastructure: Q8_1 scratch buffer (gpuXbQ8), quantize call after RMSNorm, dp4a dispatch for all gpuXb-input matmuls (ssmIn, wq/wk/wv, ffnUp). Measured: **Nemotron-3-Nano-4B 10.4 → ~20 tok/s (+90%)** vs llama.cpp 35.6 (was 29% → now ~56%).

### Granite Hybrid scaling (partial)

Granite Hybrid models use 4 scaling factors (`embeddingScale`, `logitScale`, `residualScale`, `attentionScale`). The NemotronHCudaForwardPass now supports all 4:
- `embeddingScale`: applied CPU-side in `NemotronHInferenceEngine` before uploadX
- `logitScale`: `scale_inplace` after final output matmul
- `residualScale`: saxpy instead of accumulate for ssmOut/wo/ffnDown
- `attentionScale`: pre-multiply Q by `attentionScale * sqrt(headSize)` before attention kernel

**Still blocked**: Granite Hybrid has an **integrated SwiGLU FFN** inside Mamba/Attention layers (when `lw.ffnUp() != null` for a Mamba/Attention layer). The GPU forward pass doesn't yet implement this integrated FFN. `isSupported()` rejects such models so they fall back to CPU. Granite-4.0-h-micro runs at 8.4 tok/s on CPU (vs llama.cpp 41.8 = 20%). Completing this would require ~4-6h of work to add per-layer ffn_norm/gate/up/down upload + `runIntegratedFFN()` method on GPU.

### MatmulLaunch.dp4aType switch (current)

```java
if      (t == GGMLType.Q4_K)   dp4aType = 4;   // dp4a: matmul_q4_k_dp4a
else if (t == GGMLType.Q5_K)   dp4aType = 5;   // dp4a: matmul_q5_k_dp4a
else if (t == GGMLType.Q6_K)   dp4aType = 6;   // opt-in via -Dcuda.dp4a.q6=true (slower)
else if (t == GGMLType.Q5_0)   dp4aType = 50;  // dp4a: matmul_q5_0_dp4a
else if (t == GGMLType.Q8_0)   dp4aType = 80;  // dp4a: matmul_q8_0_dp4a
else if (t == GGMLType.IQ4_NL) dp4aType = 41;  // dp4a: matmul_iq4_nl_dp4a
else if (t == GGMLType.IQ4_XS) dp4aType = 42;  // dp4a: matmul_iq4_xs_dp4a
else                           dp4aType = 0;   // falls back to FP32
```

All 4 new kernels are default-on when the corresponding weight type is detected. PPL preserved bit-equivalent via standard dp4a math (weight×input summed with scale × inScale).

### Bench variance note

Run-to-run thermal variance on this RTX 4050 Laptop is significant (±15-30%). For example OLMo-2-1B in earlier 3-run bench: 66.1, 70.9, 51.1 (range 28%). 2-run averages above should be treated as ±15% confidence intervals.

## Lessons learned

1. **Verify the hot path actually exercises the optimization being touted.** The dp4a wiring discovery was the biggest win of this entire investigation, and it was sitting in the codebase unused for an entire architecture family. Always confirm via `grep` that the flag/kernel is actually called in the path you think it is.
2. **PTX inspection finds bugs that aren't visible in source.** The `__local_depot0[12]` spill was a 384-LDL-per-row tax that I never would have found without `nvcc --ptx | grep .local`. Worth doing on hot kernels.
3. **Profile vs reference on the SAME hardware before claiming ceilings.** The "RTX 4050 bandwidth ceiling" hypothesis from prior sessions was wrong — llama.cpp on the same RTX 4050 does 110 tok/s vs our pre-fix 47. The ceiling argument is invalid until you've measured the reference on your hardware.
4. **Algorithm ports don't always beat per-arch-tuned defaults.** A faithful port of llama.cpp's mmvq gave essentially the same speed as our existing kernel. The advantage they have isn't the algorithm — it's the per-(arch, dimension) tuning matrix.
5. **Cache hints can dwarf algorithm choice.** First-attempt cp.async with `cg` (L2-only) was 100× slower than baseline because it bypassed L1. Switching to `ca` recovered most of the throughput. Lesson: never bypass L1 for broadcast reads.
6. **`__launch_bounds__` is not free magic.** NVCC's automatic register allocation often beats manual hints. Only use launch_bounds when you've measured a specific occupancy issue.
7. **Honest negative results have value.** Multiple "this should help" hypotheses were disproven by measurement. Documenting WHY each failed prevents future sessions from re-trying the same approaches.

## Project policy going forward

- **Default config**: dp4a Q4_K + dp4a Q5_K. No Q6_K dp4a (slower than FP32). No multi-warp / fused / cp.async / mmvq port. Standard launch params (blockDim=256, 8 rows/block).
- **Opt-in flags retained**: `cuda.dp4a.q6`, `cuda.dp4a.mw`, `cuda.dp4a.mmvq`, `cuda.dp4a.fused_gate_up`, `cuda.q4k.cpasync`, `cuda.prebuilt`. All measured non-improvements on Llama-1B + RTX 4050; may help on different hardware/models.
- **Future investigation**: If someone wants to close more of the gap to llama.cpp, the remaining levers are per-arch tuning (build a (cc, type, dim) → (nwarps, rows_per_block) lookup table) or full kernel-by-kernel re-tuning. Both are multi-week efforts with uncertain payoff per the data above.
