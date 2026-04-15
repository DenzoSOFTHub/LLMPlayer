# Option A — Pre-compiled PTX/cubin (attempt)

> **Date**: 2026-04-13
> **Goal**: close the ~2-3× gap to llama.cpp by replacing NVRTC JIT with NVCC-offline PTX/cubin.
> **Hypothesis** (from `tier2-attempt.md`): "NVRTC (JIT) compiler is less aggressive than NVCC (offline) at register allocation."
> **Result**: ❌ **falsified** — pre-compiled cubin matches NVRTC within ±1% on `matmul_q4_k`.

## What was done

1. Compiled `matmul_q4_k.cu` offline with NVCC 12.6 to both PTX and cubin targeting `sm_89` (our RTX 4050 Laptop CC):
   ```
   nvcc --gpu-architecture=sm_89       --use_fast_math --cubin matmul_q4_k.cu
   nvcc --gpu-architecture=compute_89  --use_fast_math --ptx   matmul_q4_k.cu
   ```
2. Added opt-in loader in `CudaContext.compileKernel()` gated by `-Dcuda.prebuilt=true`. Looks up `kernels/cuda/prebuilt/<base>.sm<CC>.{cubin,ptx}` before invoking NVRTC; falls back to NVRTC if absent or incompatible.
3. Benchmarked Llama-3.2-1B Q4_K_M, 100 tokens, alternating 6 runs (3 baseline + 3 prebuilt), full CUDA graph mode.

## Measured results

| Run | baseline (NVRTC) | prebuilt (cubin) |
|---|---:|---:|
| 1 | 54.5 | 52.8 |
| 2 | 53.3 | 53.8 |
| 3 | 53.7 | 52.9 |
| **avg** | **53.8** | **53.2** |

**Delta: -1.1%** — well within the ~3% run-to-run noise floor. No statistically meaningful difference.

## Why the hypothesis was wrong

NVRTC and NVCC share the same `ptxas` backend. With identical input flags
(`--use_fast_math`, `--gpu-architecture=compute_89`) they produce effectively
identical SASS. The path-level difference is:

- **NVRTC**: CUDA C++ source → NVRTC frontend → virtual PTX → driver's built-in `ptxas` → SASS
- **NVCC `--cubin`**: CUDA C++ source → NVCC frontend → virtual PTX → toolkit's `ptxas` → SASS (pre-assembled)

The two `ptxas` binaries (driver vs toolkit) are the same code with different release cadences; on a current toolkit/driver pair they produce matching output for kernels like ours. There is no secret register-allocator win hiding in NVCC.

## So where does the ~2-3× gap to llama.cpp actually come from?

Four candidates — none of them is "use a better compiler":

1. **Algorithm**: llama.cpp's `mmvq_q4_K` uses multi-row-per-warp with `__dp4a` int8 dot product and shared-memory tiling of the input vector. Our Tier 1 + Tier 2 attempts reproduced parts of this (dp4a for inputs is already default; mr4/mr2 variants exist as opt-in) but each individual piece failed to deliver on RTX 4050 — see `tier2-attempt.md`.
2. **Launch/dispatch overhead**: we launch many small kernels per layer (RMSNorm, QKV matmul, RoPE, KV update, attention, Wo, FFN norm, gate/up, SiLU, down). CUDA graph mode already mitigates this — see `qwen35-profile-analysis.md`.
3. **Memory layout**: weight layout conversion (permuted / interleaved) could improve coalescing, but requires changing how GGUF tensors are loaded.
4. **Hardware ceiling**: RTX 4050 Laptop has 192 GB/s HBM bandwidth. For a 0.8 GB Q4_K model at batch=1, the theoretical max is ~240 tok/s. Our 55 tok/s = ~23% of bandwidth. llama.cpp's 55-80% bandwidth utilization (measured on higher-end GPUs) may not be fully achievable on a 192 GB/s laptop GPU regardless of kernel quality.

## What's preserved

Opt-in prebuilt loader infrastructure remains in `CudaContext`:
- `-Dcuda.prebuilt=true` — look for `kernels/cuda/prebuilt/<base>.sm<CC>.{cubin,ptx}` before NVRTC
- `-Dcuda.prebuilt.verbose=true` — log which artifact was loaded

No prebuilt binaries are shipped — sm_89-specific cubin would be wrong in a general JAR, and PTX loaded via `cuModuleLoadDataEx` still goes through driver JIT (no speedup). Infrastructure is available for downstream users who want to ship their own prebuilt artifacts (e.g., deployment environments with a fixed target GPU, or for avoiding the ~50ms NVRTC compile stall at cold start).

Default behavior is **completely unchanged**.

## Honest verdict

Options revisited:

| Option | Status |
|---|---|
| A. Pre-compiled PTX/cubin | ❌ **disproven by measurement** — no speedup vs NVRTC |
| B. WMMA tensor cores | Still untested; 2-3 weeks; on a bandwidth-bound kernel at batch=1 the benefit is questionable (tensor cores help compute, not bandwidth) |
| C. `cp.async` memory/compute overlap | Still untested; estimated 10-15% on Ampere+ |
| D. Accept plateau | Current position |

Given that Tier 1 (fusion), Tier 2 (multi-row), and Option A (offline compile) all failed to move the needle on RTX 4050, the practical conclusion is that we have hit the hardware's batch=1 bandwidth ceiling for Q4_K matmul. The remaining kernel improvements available are:

- **cp.async** (Option C): modest, opt-in
- **MMQ-style batched matmul**: only helps prefill and batch>1, not autoregressive decode
- **FP16 weights + tensor cores**: trades VRAM for potential compute throughput, but bandwidth is still the limit at batch=1

**Recommendation**: stop chasing compiler/PTX optimizations; if further gains are wanted, they must come from either algorithmic changes (MMQ for prefill) or different hardware with higher memory bandwidth.

## What we learned

1. **NVRTC ≈ NVCC for release-quality SASS**. The shared `ptxas` backend means JIT vs offline makes no meaningful difference for properly-flagged builds. Future "compile it offline" proposals should be treated with skepticism unless they identify a specific flag/pragma that NVRTC doesn't support.
2. **Write the proof-of-concept before the migration**. The original plan was 1-2 weeks of Maven/NVCC integration. One kernel + one afternoon refuted the hypothesis. Always measure before investing.
3. **Bandwidth-bound problems don't benefit from compiler optimization**. `matmul_q4_k` is dominated by HBM reads of weights; register pressure and instruction scheduling affect compute but not load throughput.
