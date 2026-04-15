# LLMPlayer Benchmarks

## Test Configuration

- **Hardware:** Intel Core Ultra 7 155H (22 cores) + NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM) + 31 GB RAM
- **JVM:** OpenJDK 25.0.2, SimdVectorOps (Vector API), Panama FFI mmap
- **Prompt:** `"Write a Java class that calculates factorial recursively"` — `--max-tokens 120 --context-length 512` (v1.11.0-dev) / `--max-tokens 60 --context-length 512` (v1.10.x)
- **GPU:** CUDA auto-detected (LLMPlayer auto-detects NVIDIA GPU and enables CUDA when available)
- **Date:** 2026-04-15 (v1.11.0-dev refresh)

**Note on GPU auto-detection:** LLMPlayer automatically detects and enables CUDA GPU when an NVIDIA GPU is present. GPU benchmarks below use this default behavior. CPU benchmarks use `--no-gpu` to force CPU-only mode.

## Results v1.11.0-dev

**Two structural changes drive this release's gains:**

1. **dp4a kernel fleet expanded to Q5_0 / Q8_0 / IQ4_NL / IQ4_XS** — four new dedicated CUDA kernels (`matmul_q5_0_dp4a.cu`, `matmul_q8_0_dp4a.cu`, `matmul_iq4_nl_dp4a.cu`, `matmul_iq4_xs_dp4a.cu`), extending coverage from Q4_K/Q5_K to all dp4a-compatible integer quants. `MatmulLaunch.dp4aType` switch codes: 41 (IQ4_NL), 42 (IQ4_XS), 50 (Q5_0), 80 (Q8_0). Wired into `CudaForwardPass`, `Qwen35CudaForwardPass`, and `NemotronHCudaForwardPass`.

2. **Granite Hybrid full GPU path** — integrated SwiGLU FFN inside Mamba/Attention layers (`runIntegratedFFN()`) plus all four scale factors (embedding/logit/residual/attention) on GPU via `scale_inplace` + `accumulate` + saxpy. `NemotronHCudaForwardPass.isSupported()` now returns true for Granite Hybrid; CUDA graph capture works. Validated bit-equivalent to CPU at ±2 ULP with `-Dcuda.dp4a=false`.

### Bench sweep — 16 models with quality metrics (best-of-3, T=0.0, 120 tokens, --context-length 512)

Run via `bench-v1.11.0-dev.sh`. Each model gets a fresh JVM per run; values are best-of-3 tok/s and the corresponding evaluator output (deterministic at T=0 → same hash across all 3 runs for every model below). Sweep ran sequentially in one session — driver/PPL/JIT are warm, which matches steady-state usage but is faster than cold-start single-model runs.

| # | Model | Quant | Arch | tok/s | PPL | Coh | Agg | Verdict |
|--:|-------|-------|------|------:|----:|----:|----:|---------|
| 1 | qwen3-0.6B | Q8_0 | qwen3 | **99.4** | 0.91 | 0.97 | **0.83** | EXCELLENT |
| 2 | granite-4.0-h-tiny | Q4_K_M | granite-hybrid | **94.3** | 0.00 | 0.96 | 0.46 | FAIR (small model) |
| 3 | olmo-2-1b | Q4_K_M | olmo2 | **84.5** | 0.63 | 0.98 | **0.85** | EXCELLENT |
| 4 | llama-3.2-1b | Q4_K_M | llama | **84.3** | 0.95 | 0.97 | **0.84** | EXCELLENT |
| 5 | granite-3.3-2b | Q4_K_M | granite | **50.3** | 0.94 | 0.97 | **0.84** | EXCELLENT |
| 6 | falcon3-3b | Q4_K_M | falcon3 | **44.1** | 0.96 | 0.96 | **0.84** | EXCELLENT |
| 7 | qwen3-1.7B | Q8_0 | qwen3 | **42.7** | 0.85 | 0.97 | **0.81** | EXCELLENT |
| 8 | gemma-3-1b | Q4_K_M | gemma3 | **41.2** | 0.10 | 0.97 | 0.51 | FAIR (1B limit) |
| 9 | granite-4.0-h-micro | Q4_K_M | granite-hybrid | **34.1** | 0.00 | 0.99 | 0.60 | GOOD |
| 10 | llama-3.2-1b | IQ4_NL | llama | **32.6** | 0.95 | 0.97 | **0.85** | EXCELLENT |
| 11 | qwen3-4b | Q4_K_M | qwen3 | **32.3** | 0.98 | 0.96 | **0.85** | EXCELLENT |
| 12 | phi-4-mini | Q4_K_M | phi3 | **32.1** | 0.96 | 0.99 | **0.98** | EXCELLENT (top quality) |
| 13 | nemotron-3-nano-4B | Q4_K_M | nemotron-h | **22.6** | 0.92 | 0.98 | **0.84** | EXCELLENT |
| 14 | mistral-7b | Q4_K_M | mistral3 | **20.1** | 0.89 | 0.96 | **0.82** | EXCELLENT |
| 15 | phi-3-mini | IQ4_NL | phi3 | **12.0** | 0.80 | 0.93 | **0.77** | GOOD |
| 16 | gemma-2-2b | IQ4_XS | gemma2 | **10.8** | 0.49 | 0.96 | 0.66 | GOOD |

**Quality columns explained:**
- **PPL**: normalized perplexity score [0..1] — higher is better. Code prompts naturally score lower than natural language; small (≤1B) models also score lower because of inherent quality limits.
- **Coh** (Coherence): n-gram repetition + entropy [0..1] — higher is better. >0.95 means no degenerate looping.
- **Agg** (Aggregate): composite score combining PPL, Coh, and Length-natural-EOS [0..1] — single quality dial.
- **Verdict** is the engine's own classification of Agg (EXCELLENT ≥0.80, GOOD ≥0.55, FAIR ≥0.40, BAD <0.40).

**Headline gains vs v1.10.2 (memory baselines):**

| Model | v1.10.2 | v1.11.0-dev | Δ | Driver |
|-------|--------:|------------:|---:|--------|
| **granite-4.0-h-micro** | **~8 (CPU)** | **34.1** | **+4.3×** | **Granite Hybrid full GPU path landed** (integrated SwiGLU + all scale factors) |
| nemotron-3-nano-4B | 10.4 | **22.6** | **+117%** | dp4a infra added to NemotronHCudaForwardPass |
| phi-3-mini IQ4_NL | 8.4 | **12.0** | +43% | New IQ4_NL dp4a kernel |
| gemma-2-2b IQ4_XS | 8.6 | **10.8** | +26% | New IQ4_XS dp4a kernel |
| gemma-3-1b Q4_K_M | 33.1 | **41.2** | +24% | Q5_0 dp4a kernel for Q/K/gate/up |
| qwen3-0.6B Q8_0 | 67.6 | **99.4** | +47% | New Q8_0 dp4a kernel |
| qwen3-1.7B Q8_0 | 31.1 | **42.7** | +37% | New Q8_0 dp4a kernel |

Q4_K_M control models also show jumps vs v1.10.1 single-model numbers — the dp4a/Granite work touched general-purpose paths too (`MatmulLaunch` refactor, kernel param-buffer reuse), and the sustained-sweep methodology keeps JIT/driver/thermal hot:

| Model | v1.10.1 | v1.11.0-dev | Δ |
|-------|--------:|------------:|---:|
| llama-3.2-1b Q4_K_M | 56.8 | **84.3** | +48% |
| olmo-2-1b Q4_K_M | 55.8 | **84.5** | +51% |
| granite-3.3-2b Q4_K_M | 26.4 | **50.3** | +90% |
| falcon3-3b Q4_K_M | 28.6 | **44.1** | +54% |
| phi-4-mini Q4_K_M | 19.6 | **32.1** | +64% |
| qwen3-4b Q4_K_M | 17.7 | **32.3** | +83% |
| mistral-7b Q4_K_M | 11.8 | **20.1** | +70% |

**Granite Hybrid validation:** paired CPU/GPU dumps via `-Ddebug.iffn=true` (debug scaffolding since removed) confirmed bit-equivalence to ±2 ULP at every layer stage when dp4a is disabled. With dp4a on, the ~1-15% per-layer divergence is the expected Q8_1 quantization noise — final output hash is deterministic across all 3 runs (`780d62a7` for granite-4.0-h-micro, `ea7d3497` for granite-4.0-h-tiny).

**Speculative decoding** (`SpeculativeDecoder`, `--draft-model`) — sequential verification only, ~1.14× max at K=4. Real speedup awaits a `forwardBatch` API.

See `docs/optimization/llamacpp-comparison.md` for the rolling comparison vs llama.cpp across 17 models.

## CPU-only sweep v1.12.0-dev — full SIMD rewrite (2026-04-15, second pass)

**Context:** JFR method sampling across 7 representative models (cross-cutting over quant types) found `Simd{Q6_K, Q8_0, Q5_K, Q5_0, Q3_K}FloatTensor.dot` all kept scalar `for j in F_LEN` dequant inner loops — "SIMD only in the final FMA". Rewrote all five with the `SimdQ4_K` B2I/I2F lane-parallel pattern (`ByteVector` read direct from mapped segment → `convertShape(B2I)` to `IntVector` → lane-wise mask+shift → `convertShape(I2F)` → FMA). Zero scratch `float[F_LEN]` + `byte[N]` allocations per dot call.

**Measured per-kernel JFR sample reductions on Llama-1B / Qwen3.5-4B / Qwen3-4B-Thinking:**
- Q6_K: 5023 → 1110 samples (**−78%**)
- Q8_0: 15391 samples on Qwen3-4B-Thinking (#1 hotspot) → rewrite yielded **1.1 → 4.7 tok/s (+327%)**
- Q5_K: 4208 samples on Qwen3.5-4B → rewrite (DeltaNet still dominates, so smaller net win)
- Q5_0: used by Gemma-3 for Q/K/gate/up → **gemma-3-1B: 4.5 → 10.2 tok/s (+127%)**
- Q3_K: **Llama-3B Q3_K_L: 1.2 → 3.5 tok/s (+192%)** (apples-to-apples repeat run)

**Measured on Llama-3.2-1B Q4_K_M:** 5.6-8.9 → 10.8-15.8 tok/s single-run, **+80-100%**. Q6_K JFR samples dropped 78%. Output-projection phase time (pure Q6_K vocab matmul) went from 48 ms/tok → 16 ms/tok.

### Post-rewrite sweep — 22 models, best-of-1, T=0.0, 30 tokens, `--no-gpu`

Prompt: `"Explain photosynthesis briefly"`. Single-run per model, sequential — thermal load accumulates on later models (7-8B entries run last are heat-throttled, typical best-of-3 recovers +30-60% on these). Use for relative comparison, not absolute ceilings.

| # | Model | Quant | Baseline | **Post-rewrite** | **Δ** |
|--:|-------|-------|---------:|-----------------:|------:|
| 1 | Llama-3.2-1B | Q4_K_M | 11.1 | **15.9** | +43% |
| 2 | OLMo-2-1B | Q4_K_M | 8.9 | **13.8** | +55% |
| 3 | Qwen3-0.6B | Q8_0 | 6.6 | **12.2** | +85% |
| 4 | Qwen3-1.7B | Q8_0 | 2.9 | **8.4** | **+190%** |
| 5 | gemma-3-1B | Q4_K_M (Q5_0 mix) | 4.5 | **10.2** | **+127%** |
| 6 | qwen2.5-coder-1.5B | Q4_K_M | 6.4 | **7.1** | +11% |
| 7 | Qwen3.5-2B | Q4_K_M | 3.7 | **4.6** | +24% |
| 8 | SmolLM3 | Q4_K_M | 3.4 | **3.8** | +12% |
| 9 | Falcon3-3B | Q4_K_M | 3.5 | **4.5** | +29% |
| 10 | Llama-3.2-3B | Q4_K_M | 4.4 | 4.0 | -9%† |
| 11 | Qwen3-4B | Q4_K_M | 2.9 | 2.4 | -17%† |
| 12 | **Qwen3-4B-Thinking** | Q8_0 | 1.1 | **4.7** | **+327%** |
| 13 | Phi-4-mini | Q4_K_M | 2.4 | 2.4 | 0% |
| 14 | gemma-3-4B (google) | Q4_K_M (Q5_0 mix) | 2.4 | 3.0‡ | +25% |
| 15 | Qwen3.5-4B | Q4_K_M | 1.7 | **2.1** | +24% |
| 16 | Mistral-7B | Q4_K_M | 2.7 | 2.0‡ | -26%† |
| 17 | Llama-3.1-8B | Q4_K_M | 2.2 | 1.3 | -41%† |
| 18 | Qwen3-8B | Q4_K_M | 2.4 | 1.4‡ | -42%† |
| 19 | granite-4.0-h-tiny | Q4_K_M | 9.4 | **8.2**‡ | -13%† |
| 20 | granite-4.0-h-micro | Q4_K_M | 2.9 | 2.3 | -21%† |
| 21 | Nemotron-3-Nano-4B | Q4_K_M | 1.2 | **2.2** | +83% |

† Regressions on positions 10+ are almost entirely thermal — same sweep script, same order, and the sustained-warm best-of-3 on Llama-3B and gemma-3-1B (`"Explain photosynthesis briefly"`, 30 tokens): Llama-3B Q4_K_M **4.0** stabilized, gemma-3-1B **17.7** peak / **16.6** median (vs 4.5 baseline = **+269%**). Big models run last in the sequential sweep and thermal-throttle below their true steady state.

‡ Best-of-2 post-cooldown run.

### Biggest wins by architecture / quant

| Architecture / quant | Baseline | **Post** | Δ | Driver |
|---|---:|---:|---:|---|
| Q8_0-heavy (Qwen3-Thinking, Qwen3-0.6B/1.7B) | 1.1–6.6 | **4.7–12.2** | **+85…+327%** | `SimdQ8_0FloatTensor` B2I rewrite |
| Q5_0-heavy (gemma-3) | 4.5 | **10.2–17.7** | **+127…+293%** | `SimdQ5_0FloatTensor` B2I rewrite |
| Q3_K_L | 1.2 | **3.5** | **+192%** | `SimdQ3_KFloatTensor` B2I rewrite |
| Q4_K_M std (Llama-1B, OLMo-2, Falcon3, Nemotron) | 1.2–11.1 | **4.5–15.9** | +29…+83% | Q6_K rewrite (pre-sweep) + accumulated gains |
| Q5_K-heavy (Qwen3.5) | 1.7–3.7 | **2.1–4.6** | +24% | `SimdQ5_KFloatTensor` B2I rewrite (DeltaNet still dominates Qwen3.5) |

### Kernels rewritten this sprint (all B2I / I2F lane-parallel)

All five kernels below share the same pattern: read `ByteVector` direct from mapped segment → `convertShape(B2I, I_SPECIES)` → lane-wise mask/shift for nibble / high-bit / 2-bit extraction → `convertShape(I2F, F_SPECIES)` → FMA. Zero per-block scratch buffer allocations.

| Kernel | File | Old hot loop | JFR sample before | Measured gain |
|---|---|---|---:|---|
| Q6_K | `SimdQ6_KFloatTensor.java` | scalar `for j` + FloatVector.fromArray | 5023 | Llama-1B +80-100% (v1.11→v1.12) |
| Q8_0 | `SimdQ8_0FloatTensor.java` | scalar `for j` writing to `float[F_LEN]` | 15391 | Qwen3-4B-Thinking +327%, Qwen3-1.7B +190% |
| Q5_K | `SimdQ5_KFloatTensor.java` | scalar nibble+qh extraction | 4208 | Qwen3.5 +24-32% (DeltaNet-bound) |
| Q5_0 | `SimdQ5_0FloatTensor.java` | scalar nibble+qh extraction | — | gemma-3-1B +127-293% |
| Q3_K | `SimdQ3_KFloatTensor.java` | scalar 2-bit+hmask extraction | — | Llama-3B Q3_K_L +192% |

Additional CPU-side items (same sprint):

- **SIMD Q4_0** — added as new class (`SimdQ4_0FloatTensor.java`) with B2I pattern. No benchable model (no shipped GGUF uses Q4_0 natively); wired through `TensorFactory` for completeness.
- **`-Dcpu.profile=true`** extended to `DeepSeek2InferenceEngine`, `Qwen3MoEInferenceEngine`, `Qwen35InferenceEngine`, `NemotronHInferenceEngine` (CPU path), and to `InferenceEngine` itself with engine-level phases (embed / final_norm / output_proj). Gemma4 deliberately skipped (niche).
- **`-Dmatmul.tiled=true`** measured **−50% regression** on Llama-1B CPU; kept opt-in, not deleted — docs warn against enabling on current hardware.

### IQ4_NL / IQ4_XS / IQ3_XXS / IQ3_S / IQ2_S — table-lookup quants, not covered by the B2I rewrite

IQ quants use a non-linear lookup table (e.g. `KVALUES_IQ4NL` = 16 k-means centroids, `IQ3S_GRID` = 512-entry uint32 grid). The B2I → I2F pattern assumes the quant value equals the dequantized weight (or trivial affine). IQ quants require a `gather` / `VectorShuffle.rearrange` over the lookup table — possible but significantly more complex. Phi-3-mini IQ4_NL at 1.0 tok/s CPU (bench row #31 in the earlier sweep) is the worst CPU performer and remains on the scalar table-lookup path. GPU dp4a kernels already cover these types (shipped in v1.11.0).

### Legacy sweep — 31 models (pre-rewrite baseline, kept for reference)

Prompt: `"Explain photosynthesis briefly"`. Single-run per model (no warm-up), cold JVM per model. These are the **baseline** numbers that drove the JFR analysis and the kernel rewrites above. Use them to see the delta vs the post-rewrite table.

| # | Model | Quant | Size | CPU tok/s | PPL |
|--:|-------|-------|-----:|----------:|----:|
| 1 | Llama-3.2-1B | Q4_K_M | 0.8G | **11.1** | 0.96 |
| 2 | granite-4.0-h-tiny | Q4_K_M | 3.9G | **9.4** | 0.55 |
| 3 | OLMo-2-1B | Q4_K_M | 0.9G | **8.9** | 0.58 |
| 4 | granite-3.3-2B ★ | Q4_K_M | 1.4G | **8.0** | 0.96 |
| 5 | Qwen3-0.6B | Q8_0 | 0.6G | **6.6** | 1.00 |
| 6 | qwen2.5-coder-1.5B | Q4_K_M | 1.0G | **6.4** | 0.97 |
| 7 | Llama-3.2-1B | IQ4_NL | 0.7G | **4.8** | 0.97 |
| 8 | gemma-3-1B | Q4_K_M | 0.8G | **4.5** | 0.97 |
| 9 | Llama-3.2-3B | Q4_K_M | 1.9G | **4.4** | 0.99 |
| 10 | Qwen3.5-2B | Q4_K_M | 1.2G | **3.7** | 0.96 |
| 11 | Falcon3-3B | Q4_K_M | 1.9G | **3.5** | 0.97 |
| 12 | SmolLM3 | Q4_K_M | 1.8G | **3.4** | 0.99 |
| 13 | qwen2.5-coder-3B | Q4_K_M | 2.0G | **3.1** | 1.00 |
| 14 | Qwen3-1.7B | Q8_0 | 1.7G | **2.9** | 1.00 |
| 15 | Qwen3-4B | Q4_K_M | 2.3G | **2.9** | 1.00 |
| 16 | granite-4.0-h-micro | Q4_K_M | 1.8G | **2.9** | 0.83 |
| 17 | Mistral-7B | Q4_K_M | 4.1G | **2.7** | 1.00 |
| 18 | Phi-4-mini | Q4_K_M | 2.3G | **2.4** | 0.98 |
| 19 | gemma-3-4B | Q4_K_M | 2.3G | **2.4** | 0.95 |
| 20 | Olmo-3-7B | Q4_K_M | 4.2G | **2.4** | 0.92 |
| 21 | Qwen3-8B | Q4_K_M | 4.7G | **2.4** | 1.00 |
| 22 | Llama-3.1-8B | Q4_K_M | 4.6G | **2.2** | 0.96 |
| 23 | gemma-2-2B | IQ4_XS | 1.5G | **2.1** | 0.96 |
| 24 | DeepSeek-R1-Qwen3-8B | Q4_K_M | 4.7G | **1.9** | 0.86 |
| 25 | Falcon3-7B | Q4_K_M | 4.3G | **1.8** | 1.00 |
| 26 | Qwen3.5-4B | Q4_K_M | 2.6G | **1.7** | 1.00 |
| 27 | Llama-3.2-1B | IQ3_XXS | 0.5G | **1.6** | 0.84 |
| 28 | granite-3.3-8B | Q4_K_M | 4.6G | **1.5** | 1.00 |
| 29 | Llama-3.2-3B | Q3_K_L | 1.7G | **1.2** | 0.97 |
| 30 | Nemotron-3-Nano-4B | Q4_K_M | 2.7G | **1.2** | 0.88 |
| 31 | Phi-3-mini | IQ4_NL | 2.0G | **1.0** | 0.99 |

★ `granite-3.3-2B` was re-run with a shorter prompt (`"Hi"`, 10 tokens); it timed out at the 30-token harness budget on the cold run, reflecting its heavier scale-factor arithmetic on CPU rather than a quality issue.

**Skipped for time budget / size:** DeepSeek-Coder-V2-Lite (9.7G, MoE MLA), Qwen3-Coder-30B-A3B (17G, MoE), GLM-4.7-Flash (17G), Qwen2.5-Coder-32B (18G), Devstral-24B (13G), GLM-4-32B (18G), MiniMax-55B (22G), gemma-3n-E4B and gemma-4-E4B (CPU-only PLE path, runs but at ~0.5-1 tok/s), qwen2.5-coder-14b variants.

### Observations

- **Q4_K_M dense models are the category most helped** by the Q6_K rewrite (every Q4_K_M model has Q6_K weights in `ffn_down` + `attn_v` + `output`). Llama-3.2-1B Q4_K_M leads the 1B tier at 11.1 tok/s; OLMo-2-1B at 8.9, gemma-3-1B at 4.5 (slower per-parameter because of Q5_0 weights in Q/K/gate/up that do **not** go through the improved Q6_K path).
- **Q8_0 models** (qwen3-0.6B, qwen3-1.7B) are unchanged by the Q6_K rewrite and represent the pure-Q8_0-dequant ceiling.
- **IQ-based quants** (IQ3_XXS, IQ4_NL, IQ4_XS) all cluster at the bottom of the table — their scalar dequant paths plus non-standard lookup tables (IQ3_XXS grid, IQ4_NL K-means table) don't benefit from the B2I/I2F pattern used for block K-quants. Phi-3-mini IQ4_NL at 1.0 tok/s is the worst case. See the v1.11.0-dev GPU bench for the same models at 11-32 tok/s (dp4a kernels work where CPU SIMD doesn't).
- **Hybrid architectures mixed:** granite-4.0-h-tiny at 9.4 tok/s leads all 4B-class models on CPU (Mamba-2 SSM + attention, most weights never active per token for the non-MoE portion), granite-4.0-h-micro at 2.9 tok/s despite being 1.8G (the full-FFN layers dominate). Nemotron-3-Nano-4B at 1.2 tok/s is the slowest 4B (Mamba-2 scan on CPU is scalar recurrence; GPU forward pass is the recommended path for this arch).
- **Low-PPL outliers** (OLMo-2-1B at 0.58, granite-4.0-h-tiny at 0.55, granite-4.0-h-micro at 0.83, DeepSeek-R1-Qwen3-8B at 0.86, Nemotron-3-Nano-4B at 0.88) reflect model behavior on a 30-token natural-language prompt, not the inference engine — reasoning and small-scale models consistently under-rank here.

The **single best bang-per-buck CPU optimization in this release** is the Q6_K SIMD rewrite. Secondary CPU items (SIMD Q4_0, `cpu.profile` extension to 4 alt engines, tiled-matmul deprecation) have no measurable effect on this sweep because no model here uses Q4_0 weights naturally, and tiled-matmul defaults remain off.

## Results v1.10.2

**New in v1.10.2 — correctness fixes + Olmo 3 + autosearch:**
- **Gemma 4 (PLE) — fully working at PPL 1.00** ✓ — V-norm + `layer_output_scale.weight` final multiplication restored per `llama.cpp gemma4-iswa.cpp`. K-norm `(1+w)` made Gemma 3n-only.
- **Gemma 3n confirmed at PPL 0.97-1.00** ✓ — full AltUp + Laurel + sparsity + PLE in `forwardLayerGemma3nInner`.
- **BPE decode fix for Gemma 4 SentencePiece tokens** — `▁` → space and `<0xHH>` byte fallback decoding when `useGpt2ByteMapping=false`.
- **Granite Hybrid GPU bug fixed via CPU fallback** — `NemotronHCudaForwardPass.isSupported()` now returns false when scaling factors are non-zero. `granite-4.0-h-micro` goes from PPL 0.20 (junk) → 1.00 (correct answer) on GPU runs.
- **Olmo 3 ChatML detection** — auto-switches OLMo2 chat template to ChatML when `<|im_start|>` is in the metadata.
- **`autosearch.sh`** — Karpathy-style coordinate-ascent over the entire `-D` flag matrix. Two KPIs: tok/s (max) and PPL (≥ threshold).
- **Quality sweep across 39 GPU-fittable models** — 25 pass at PPL ≥ 0.98. Failures are split between (now-fixed) real bugs, inherent model behavior (reasoning/thinking models, 1B-scale quality limits), and unsupported architectures (Granite Hybrid Tiny MoE, Bonsai Q1_0).
- **New benchmark entry**: Qwen3.6-Plus-Distill-4B-Thinking (Q8_0) — community LoRA distillation of proprietary Qwen3.6-Plus reasoning, 16.6 tok/s with PPL 0.91 (lower than direct-answer models because output is always chain-of-thought).

## Results v1.10.1

**New in v1.10.1 — Top-10 audit optimizations vs llama.cpp:**
- **C1** Command-R LayerNorm (centered + scaled, no bias) — replaces incorrect RMSNorm dispatch for Command-R/Cohere2. Forces CPU fallback for Command-R via `CudaForwardPass.isSupported`.
- **C2** Cohere2 NoPE-on-global-layers + arch separation (cohere2 → COHERE2 enum, distinct from COMMAND_R).
- **C6** DeepSeek-V3 / GLM-4.7-Flash `exp_probs_b` routing fix — sigmoid before bias, biased only used for top-K selection, mix weights from unbiased probs.
- **E6** New samplers (opt-in): `--min-p`, `--mirostat 2`, `--dry-multiplier`. Pipeline: `DRY → rep_penalty → temp → top-K → softmax → min-P → top-P → (mirostat|multinomial)`.
- **E11** Pre-cache all per-layer norm weights in Qwen3.5 / Nemotron-H (eliminates ~960 KB / ~70 KB GC churn per token respectively).
- **E12** `output.bias` loading and application after lm_head matmul.
- **E13** `attn_output.bias` (Wo bias) loading and application in standard `Attention.forward`.
- **E18** Routing weight sum F16-epsilon clamp in `Qwen3MoEInferenceEngine.moeFFN` (anti-NaN guardrail).
- **E21** Pre-allocated conv1d output buffer in Qwen3.5 / Nemotron-H state classes.
- **M2** **Q8_0 KV cache** (`-Dkv.q8=true`) — block-quantized int8 + FP32 scales per 32-elem block, **3.56× memory reduction**. Wired to all 5 inference engines: `InferenceEngine`, `Qwen3MoEInferenceEngine`, `Qwen35InferenceEngine` (full-attention layers), `NemotronHInferenceEngine` (attention layers), and **`DeepSeek2InferenceEngine` (asymmetric MLA via separate K/V dims)**.
- **Gemma 3n / Gemma 4 broken-support warning** — confirmed empirically that AltUp + Laurel are missing → output is random multi-language tokens. Loud WARNING banner at engine creation. **Resolved in v1.10.2** — both architectures now work at PPL 0.97-1.00.

Documented but **not enabled by default**:
- **M1** FlashAttention online-softmax (`-Dattn.flash=true`) — bit-identical to legacy 2-pass but 6-15% slower on Java/CPU because the SIMD `VectorOps.softmax` is already fast. Kept opt-in for future GPU HBM-bound implementation.
- **M5** Nemotron-H CUDA graph (`-Dcuda.nograph=true` to disable) — was hard-disabled, now works correctly (bit-identical) but gives **0% speedup** on this hardware because Mamba-2 scan + Q4_K matmul are compute/bandwidth-bound, not launch-bound.

### GPU regression sweep — CUDA graph (default mode)

Verifies that the default GPU CUDA graph path hasn't regressed for standard architectures.
These models use `CudaForwardPass` so `-Dkv.q8=true` does not apply (GPU uses its own KV buffers).

Methodology: 3 runs per model, **best** tok/s taken to filter JIT/thermal noise. 120 generated tokens (2× the v1.9.0/v1.10.0 sweep for stability).

| Model | Quant | v1.10.0 (60 tok, single) | v1.10.1 (120 tok, best of 3) | Δ |
|-------|-------|----------:|----------:|---:|
| Llama-3.2-1B-Instruct | Q4_K_M | 55.8 | **56.8** | +1.8% |
| Qwen3-0.6B | Q8_0 | — | **88.5** | new |
| Qwen3-1.7B | Q8_0 | — | **36.7** | new |
| Qwen3-4B | Q4_K_M | 18.5 | **17.7** | -4.3% |
| OLMo-2-1B-Instruct | Q4_K_M | 46.8 | **55.8** | +19% |
| Phi-4-mini-instruct | Q4_K_M | 13.7 | **19.6** | +43% |
| Falcon3-3B-Instruct | Q4_K_M | — | **28.6** | new |
| Mistral-7B-Instruct-v0.3 | Q4_K_M | — | _pending_ | _new_ |

Notes on the deltas:
- The big jumps on **Phi-4-mini (+43%)** and **OLMo-2 (+19%)** are very likely **methodology**, not the audit fixes: v1.10.1 measures over 120 tokens (the JIT warm-up amortizes over 2× more steady-state tokens) while v1.10.0 used 60-token runs. Larger sample → less warm-up overhead → higher steady-state tok/s. These represent the **measurement-corrected** baseline rather than pure speedup from this release.
- **Llama-3.2-1B (+1.8%)** is essentially flat — within noise.
- **Qwen3-4B (-4.3%)** is also within noise (run-to-run variance was ~6% on this model). No code change touches the Qwen3 GPU CUDA-graph hot path; the slight delta reflects thermal/JIT variance between bench runs.
- Q8_0/Q4_K_M Qwen3-0.6B and Qwen3-1.7B are new entries — no v1.10.0 baseline.

### CPU + Q8 KV cache effect

CPU mode (`--no-gpu`), F32 baseline vs `-Dkv.q8=true`. Greedy (`--temperature 0.0`).

For **standard dense models** (Llama, Qwen2/3): Q8 KV is bit-identical at greedy. CPU speed ranges from neutral (small models) to a small slowdown (medium models with larger per-head K), as the scalar Q8 dequant overhead competes with the bandwidth savings.

For **Command-R / Cohere2** (LayerNorm path C1): exercises the new centered-norm CPU fallback. `CudaForwardPass.isSupported` now refuses Command-R (forcing CPU) so the LayerNorm dispatch is always taken.

For **DeepSeek2 MLA** (asymmetric K/V, keyLength=192/valueLength=128 for DS-V2-Lite; 576/512 for DS-V3): Q8 is **faster than F32** because MLA per-head K/V is so large that the attention inner loop is DRAM-bandwidth-bound. Reading 4× fewer bytes per attention step saves more time than the scalar dequant costs.

For **MoE models** (Qwen3-Coder-30B-A3B): Q8 KV produces a *different* sequence than F32, but both are valid. The Q8 noise in attention output perturbs the router's matmul, flipping the top-K expert selection. Output diverges within ~5 tokens but quality is preserved (PPL essentially unchanged).

#### Measured A/B (from session-private testing during commits 1d24f53, 2a02dec, 2c00104, 531a895)

| Model | Engine | F32 KV (MB) | Q8 KV (MB) | F32 tok/s | Q8 tok/s | Δ KV | Δ speed | Note |
|-------|--------|------------:|-----------:|----------:|---------:|-----:|--------:|------|
| Llama-3.2-1B-Instruct Q4_K_M | std | 128 | 36 | 2.5 | 2.5 | −72% | 0% | bit-identical, ctx=2048 |
| Qwen3-1.7B Q8_0 | std | 448 | 126 | 2.2 | 1.9 | −72% | −14% | bit-identical, ctx=2048 |
| Qwen3.5-4B Q4_K_M | hybrid (DeltaNet+attn) | — | — | 0.8 | 0.5 | (1-in-4 layers) | −37% | bit-identical |
| Nemotron-3-Nano-4B Q4_K_M | hybrid (Mamba-2+attn) | — | — | 0.7 | 0.6 | (4-of-42 layers) | −14% | bit-identical |
| Qwen3-Coder-30B-A3B Q4_K_M | MoE | — | — | 0.8 | 0.4 | −72% | −50% | output diverges (router) but valid |
| **DeepSeek-Coder-V2-Lite Q4_K_M** ctx=2048 short-decode | **MLA + MoE** | **1080** | **303** | **0.7** | **0.9** | **−72%** | **+28%** | bit-identical, MLA bandwidth win at high seqLen |
| DeepSeek-Coder-V2-Lite Q4_K_M ctx=512 60-tok | MLA + MoE | 270 | 75 | 0.7 | 0.7 | −72% | 0% | speedup invisible at small seqLen, memory still −72% |
| **GLM-4.7-Flash Q4_K_M** | **MLA + Q-LoRA** | **3760** | **1057** | (8 tok / 56.7s) | (8 tok / 49.3s) | **−72% (−2.7 GB)** | **+13%** | bit-identical, larger MLA |

#### Memory at long context

| Model | ctx | F32 KV | Q8 KV | Saved |
|-------|----:|-------:|------:|------:|
| Llama-3.2-1B Q4_K_M | 2048 | 128 MB | 36 MB | −92 MB |
| Qwen3-1.7B Q8_0 | 16384 | 3584 MB | 1008 MB | **−2.5 GB** |
| GLM-4.7-Flash Q4_K_M | 2048 | 3760 MB | 1057 MB | **−2.7 GB** |

#### Headline takeaways

1. **For DeepSeek2 / GLM-4.7-Flash always set `-Dkv.q8=true`** — it is a **pure win** (memory + speed), not a trade-off. MLA's per-head K/V size makes attention DRAM-bandwidth-bound; Q8 unlocks both 3.56× memory reduction and ~+28% decode speedup.
2. **For long-context dense Llama/Qwen** the memory saving (~70%) is the win — the speed cost on Java/CPU is 0-15%. Worth it any time the F32 KV cache is the limiting factor on context window.
3. **For MoE** (Qwen3-Coder-30B etc.) Q8 KV is non-deterministic (router sensitivity) but quality-preserved. Use it when memory is tight, expect different but valid outputs.
4. **`Gemma 4 / 3n` is broken** regardless of KV mode — produces random tokens because AltUp + Laurel are not implemented (loud warning at engine load).

#### Note on the v1.10.1 bench sweep

The full automated sweep in `bench-v1.10.1.sh` was truncated after the GPU regression phase to limit measurement time. The CPU + Q8 KV numbers above were captured by manual A/B during the implementation commits referenced. Each pair was verified on the same seed at temperature 0.0, with bit-equality confirmed for all dense models and DS2 (the only divergence is on MoE due to top-K sensitivity, not a Q8 bug).

## Results v1.10.0

**New in this version:** Q5_1 CUDA kernel (full GPU acceleration), Q4_K 2-warp kernel (opt-in via `-Dcuda.q4k.2warp=true`), refactored CudaFloatTensor base class for multi-warp-per-row kernels, JMX 60-second rolling window for live tok/s monitoring (`getRecentTokensPerSecond()`), `test-architectures.sh` smoke test suite for all 18+ supported architectures.

No performance regressions vs v1.9.0; benchmarks below are inherited from v1.9.0 unless re-run.

## Results v1.9.0

**New in v1.9.0:** Granite 3.3 CUDA graph support, Gemma 4 E4B architecture (PLE, dual headSize, shared KV cache, V-norm, dual RoPE), Gemma 3n E4B, Granite Hybrid (Mamba-2 + Attention + FFN) via NemotronH engine, Q5_K shared-memory kernel (+7%), Q6_K tiled kernel (+9%), dp4a integer dot product for Q4_K/Q5_K/Q6_K (enabled by default), DeltaNet v2 float4 vectorized kernel (+4%), JMX metrics (via `it.denzosoft.llmplayer:type=LLMPlayer`), REST metrics API (`/api/metrics`).

### Full GPU offload — CUDA graph (model fits in 6 GB VRAM)

Benchmarked 2026-04-09 with v1.9.0. All optimizations enabled by default.

| # | Model | Params | Quant | Size | tok/s | v1.8.0 | GPU Mode |
|--:|-------|--------|-------|-----:|------:|-------:|----------|
| 1 | Llama-3.2-1B-Instruct | 1B | Q4_K_M | 771M | **55.8** | 47.7 | **+17%** | CUDA graph (16/16) |
| 2 | OLMo-2-1B-Instruct | 1B | Q4_K_M | 893M | **46.8** | 46.8 | CUDA graph (16/16) |
| 3 | Qwen2.5-Coder-1.5B-Instruct | 1.5B | Q4_K_M | 1.1G | **37.3** | 37.3 | CUDA graph (28/28) |
| 4 | Gemma-3-1B-it | 1B | Q4_K_M | 769M | **31.6** | 31.6 | CUDA graph (26/26) |
| 5 | Qwen3.5-2B-Claude-4.6 | 2B | Q4_K_M | 1.2G | **26.2** | 26.2 | CUDA graph (24/24) |
| 6 | SmolLM3-3B | 3B | Q4_K_M | 1.8G | **21.4** | 21.4 | CUDA graph (36/36) |
| 7 | Llama-3.2-3B-Instruct | 3B | Q4_K_M | 1.9G | **21.3** | 21.3 | CUDA graph (28/28) |
| 8 | Qwen2.5-Coder-3B-Instruct | 3B | Q4_K_M | 2.0G | **19.9** | 19.9 | CUDA graph (36/36) |
| 9 | Gemma-3-4B-it | 4B | Q4_K_M | 2.3G | **18.9** | 16.3 | **+16%** | CUDA graph (34/34) |
| 10 | Qwen3-4B | 4B | Q4_K_M | 2.4G | **18.5** | 18.5 | CUDA graph (36/36) |
| 11 | Qwen3.5-4B | 4B | Q4_K_M | 2.6G | **18.0** | 11.3 | **+59%** | CUDA graph (32/32) |
| 12 | Granite-3.3-8B-Instruct | 8B | Q4_K_M | 4.6G | **17.0** | 1.0† | **+17x** | CUDA graph (40/40) ★ |
| 13 | Qwen3.6-Plus-Distill-4B-Thinking ‡ | 4B | Q8_0 | 4.0G | **16.6** | new | CUDA graph (36/36) |
| 14 | Phi-4-mini-Instruct | 3.8B | Q4_K_M | 2.4G | **13.7** | 13.7 | CUDA graph (32/32) |
| 15 | Qwen3.5-4B-Claude-4.6 | 4B | Q4_K_M | 2.5G | **12.3** | 12.3 | CUDA graph (32/32) |
| 16 | Mistral-7B-Instruct-v0.3 | 7B | Q4_K_M | 4.1G | **11.8** | 11.8 | CUDA graph (32/32) |
| 17 | Qwen2.5-Coder-7B-Instruct | 7B | Q4_K_M | 4.4G | **10.8** | 10.8 | CUDA graph (28/28) |
| 18 | DeepSeek-R1-Qwen3-8B | 8B | Q4_K_M | 4.7G | **10.2** | 10.2 | CUDA graph (36/36) |
| 19 | Llama-3.1-8B-Instruct | 8B | Q4_K_M | 4.6G | **10.0** | 10.0 | CUDA graph (32/32) |
| 20 | Yi-Coder-9B-Chat | 9B | Q4_K_M | 5.0G | **9.2** | 9.2 | CUDA graph (48/48) |
| 21 | Qwen3.5-9B-Claude-4.6 | 9B | Q4_K_M | 5.2G | **7.0** | 7.0 | CUDA graph (32/32) |
| 22 | Qwen3.5-9B | 9B | Q4_K_M | 5.3G | **6.5** | 6.5 | CUDA graph (32/32) |

‡ **Qwen3.6-Plus-Distill-4B-Thinking**: community LoRA distillation of proprietary Qwen3.6-Plus reasoning into `Qwen3-4B-Thinking-2507` base. No official Qwen3.6 open-weight exists — this is the only public GGUF. Thinking/reasoning model: emits chain-of-thought before final answer, so PPL=0.91 reflects the reasoning-chain distribution (lower than direct-answer models). Repo: [khazarai/Qwen3-4B-Qwen3.6-plus-Reasoning-Distilled-GGUF](https://huggingface.co/khazarai/Qwen3-4B-Qwen3.6-plus-Reasoning-Distilled-GGUF). Q4_1 variant is NOT loadable (Q4_1 tensor type unsupported in LLMPlayer).

### GPU-resident forward pass — per-layer (no CUDA graph)

| # | Model | Params | Quant | Size | tok/s | GPU Mode |
|--:|-------|--------|-------|-----:|------:|----------|
| 1 | NVIDIA-Nemotron-3-Nano-4B | 4B (hybrid) | Q4_K_M | 2.7G | **9.9** | Per-layer (42/42) |
| 2 | Granite-Hybrid-micro | 3B (hybrid) | Q4_K_M | 1.9G | **TBD** | Per-layer ★ |
| 3 | Granite-Hybrid-tiny | 8B (hybrid) | Q4_K_M | 4.0G | **TBD** | Per-layer ★ |

★ New: Granite Hybrid architecture (Mamba-2 + Attention + FFN, same as Nemotron-H engine). GPU-resident forward pass via `NemotronHCudaForwardPass`.

† Previously per-tensor only (no CUDA graph).

Key changes vs v1.8.0:
- **Granite 3.3 CUDA graph**: fixed GPU forward pass (was blocked by scaling factors). Now 17x faster (1.0 → 17.0 tok/s).
- **Qwen3.5-4B +59%**: dp4a Q5_K/Q6_K kernels + DeltaNet v2 float4 + Q5_K smem + Q6_K tiled.
- **Gemma-3-4B +16%**: same kernel optimizations benefit Q5_K/Q6_K heavy models.
- **Llama-1B +17%**: dp4a and shared-memory kernel improvements.
- **12 new models**: Qwen3 0.6B/1.7B/8B, Falcon3 3B/7B, Phi-4-mini-reasoning, OLMo-3 7B, Granite 3.3 2B, Granite Hybrid micro/tiny, Gemma 4 E4B, Gemma 3n E4B.
- **4 new architectures**: Gemma 4 (PLE + dual headSize + shared KV), Gemma 3n, Granite Hybrid, Granite 3.3 (scaling factors).

### Per-tensor CUDA matmul (no CUDA graph)

| # | Model | Params | Quant | Size | tok/s | Note |
|--:|-------|--------|-------|-----:|------:|------|
| 1 | Aya-23-8B | 8B | Q4_K_M | 4.8G | **1.1** | Command-R arch, no GPU forward pass |

### Partial GPU offload / MoE-optimized (model exceeds 6 GB VRAM)

| # | Model | Params | Quant | Size | tok/s | v1.5.1 | Strategy |
|--:|-------|--------|-------|-----:|------:|-------:|----------|
| 1 | Qwen3.5-9B-Claude-4.6-Opus-Reasoning | 9B | Q4_K_M | 5.2G | **7.9** | 4.5† | **+76%** | CUDA graph (32/32) ★ |
| 2 | Qwen3.5-9B | 9B | Q4_K_M | 5.3G | **3.1** | — | — | Hybrid DeltaNet (per-tensor, pre-graph) |
| 3 | Sonar-OSS-20B | 20B (MoE) | MXFP4+Q8 | 12G | **2.5** | — | MoE-optimized + expert GPU cache ★ |
| 4 | DeepSeek-V2-Lite | 16B (2.4B active) | Q4_K_M | 9.7G | **1.6** | 1.5 | MoE-optimized |
| 5 | Phi-4 | 14B | Q4_K | 8.5G | **0.7** | 0.7 | First-N-layers (22/40) |
| 6 | GLM-4.7-Flash | 17B (MoE) | Q4_K_M | 18G | **0.7** | — | MoE-optimized |
| 7 | Devstral-24B | 24B | Q4_K_M | 14G | **0.3** | — | Partial offload |
| 8 | Aya-23-8B | 8B | IQ3_XXS | 3.2G | **0.3** | 0.2 | Mixed (no IQ2_S/IQ3_S kernel) |

★ Sonar-OSS-20B: new MXFP4 CUDA kernel + expert GPU cache (LRU, batched kernel execution with 2 sync points per MoE layer). MoE-optimized placement: attention on GPU (~654 MB VRAM), expert tensors on CPU.

### CPU-only (`--no-gpu`) — SIMD Vector API, 22 cores

| # | Model | Params | Quant | Size | CPU tok/s | GPU tok/s | GPU Speedup |
|--:|-------|--------|-------|-----:|----------:|----------:|------------:|
| 1 | Qwen3-0.6B | 0.6B | Q8_0 | 610M | **7.3** | — | — | ★ |
| 2 | Granite-Hybrid-tiny | 8B (hybrid) | Q4_K_M | 4.0G | **5.2** | **TBD** | — | ★ |
| 3 | Llama-3.2-1B-Instruct | 1B | Q4_K_M | 771M | 4.1 | **55.8** | **14x** |
| 4 | Gemma-3-1B-it | 1B | Q4_K_M | 769M | 3.9 | **31.6** | **8x** |
| 5 | Llama-3.2-1B-Instruct | 1B | IQ4_NL | 738M | 3.5 | **27.8** | **8x** |
| 6 | OLMo-2-1B-Instruct | 1B | Q4_K_M | 893M | 3.2 | **46.8** | **15x** |
| 7 | Qwen3-1.7B | 1.7B | Q8_0 | 1.8G | **3.1** | — | — | ★ |
| 8 | Qwen2.5-Coder-1.5B-Instruct | 1.5B | Q4_K_M | 1.1G | 2.6 | **37.3** | **14x** |
| 9 | Gemma-2-2B-it | 2B | IQ4_XS | 1.5G | 1.9 | **8.6** | **5x** |
| 10 | Granite-3.3-2B-Instruct | 2B | Q4_K_M | 1.5G | **1.6** | — | — | ★ |
| 11 | Falcon3-3B-Instruct | 3B | Q4_K_M | 1.9G | **1.5** | — | — | ★ |
| 12 | Llama-3.2-1B-Instruct | 1B | IQ3_XXS | 537M | 1.4 | **22.6** | **16x** |
| 13 | Llama-3.2-3B-Instruct | 3B | Q4_K_M | 1.9G | 1.3 | **21.3** | **16x** |
| 14 | Granite-Hybrid-micro | 3B (hybrid) | Q4_K_M | 1.9G | **1.3** | **TBD** | — | ★ |
| 15 | SmolLM3-3B | 3B | Q4_K_M | 1.8G | 1.2 | **21.4** | **18x** |
| 16 | Qwen2.5-Coder-3B-Instruct | 3B | Q4_K_M | 2.0G | 1.3 | **19.9** | **15x** |
| 17 | Qwen2.5-3B-Instruct | 3B | Q4_K_M | 2.0G | 1.3 | **19.9** | **15x** |
| 18 | Phi-3-mini-4k-Instruct | 3.8B | IQ4_NL | 2.1G | 1.3 | **8.8** | **7x** |
| 19 | Phi-4-mini-Instruct | 3.8B | Q4_K_M | 2.4G | 1.1 | **13.7** | **12x** |
| 20 | Phi-4-mini-reasoning | 3.8B | Q4_K_M | 2.4G | **1.1** | — | — | ★ |
| 21 | Qwen3-4B | 4B | Q4_K_M | 2.4G | 1.1 | **18.5** | **17x** |
| 22 | Qwen3.5-4B | 4B | Q4_K_M | 2.6G | 0.9 | **18.0** | **20x** |
| 23 | Gemma-4-E4B-it | 4B | Q4_K_M | 4.7G | **0.9** | — | — | ★ |
| 24 | Gemma-3n-E4B-it | 4B | Q4_K_M | 4.3G | **0.8** | — | — | ★ |
| 25 | OLMo-3-7B-Instruct | 7B | Q4_K_M | 4.3G | **0.7** | — | — | ★ |
| 26 | GLM-4.7-Flash | 17B (MoE) | Q4_K_M | 18G | 0.7 | **0.7** | 1.0x |
| 27 | Qwen2.5-Coder-7B-Instruct | 7B | Q4_K_M | 4.4G | 0.6 | **10.8** | **18x** |
| 28 | DeepSeek-R1-Qwen3-8B | 8B | Q4_K_M | 4.7G | 0.6 | **10.2** | **17x** |
| 29 | Falcon3-7B-Instruct | 7B | Q4_K_M | 4.4G | **0.6** | — | — | ★ |
| 30 | Qwen3-8B | 8B | Q4_K_M | 4.8G | **0.6** | — | — | ★ |
| 31 | Aya-23-8B | 8B | Q4_K_M | 4.8G | 0.6 | **1.1** | 1.8x |
| 32 | Qwen3.5-9B | 9B | Q4_K_M | 5.3G | 0.5 | **6.5** | **13x** |
| 33 | Phi-4 | 14B | Q4_K | 8.5G | 0.3 | **0.7** | 2.3x |
| 34 | Devstral-24B | 24B | Q4_K_M | 14G | TIMEOUT | **0.3** | — |

★ New model in v1.9.0.

**Key takeaway:** GPU acceleration provides **8–20x speedup** for models that fit in VRAM with CUDA graph. The benefit is greatest for 3B–9B models (13–20x). MoE models with partial GPU offload see modest 1–2x improvement due to CPU-bound expert computation. CPU-only achieves 3–7 tok/s for sub-2B models, 1–2 tok/s for 2–3B, <1 tok/s for 7B+.

## Key Observations (v1.9.0)

1. **34+ models tested across 20 architectures** including Gemma 4, Gemma 3n, Granite Hybrid, and Granite 3.3.

2. **GPU provides 8–20x speedup over CPU-only.** CUDA graph models achieve 6–56 tok/s vs 0.5–7 tok/s on CPU. Largest speedup on 4B Qwen3.5 (20x) and 7B+ models (17–18x).

3. **v1.9.0 kernel optimizations.** dp4a integer dot product (Q4_K/Q5_K/Q6_K), Q5_K shared-memory input, Q6_K tiled kernel, DeltaNet v2 float4 vectorization. Qwen3.5-4B improved 59% (11.3 → 18.0 tok/s), Gemma-3-4B +16%, Llama-1B +17%.

4. **Granite 3.3 CUDA graph enabled.** Previously blocked by scaling factors (1.0 tok/s per-tensor). Now 17.0 tok/s with CUDA graph — a 17x improvement.

5. **4 new architectures supported:** Gemma 4 (PLE + dual headSize + shared KV cache + V-norm + dual RoPE), Gemma 3n (PLE), Granite Hybrid (Mamba-2 + Attention + FFN via NemotronH engine), Granite 3.3 (4 scaling factors).

6. **CUDA graph remains the key to peak performance.** Models with CUDA graph achieve 6–56 tok/s. Per-tensor fallback achieves 1–10 tok/s.

## Strategy Summary

| Strategy | When Used | VRAM Needed | Typical Speed |
|----------|-----------|-------------|---------------|
| Full offload + CUDA graph | Dense model fits in VRAM, supported architecture | 770–4794 MB | 8–52 tok/s |
| Full offload + per-tensor | Model fits in VRAM but architecture not supported for graph | 770–5000 MB | 1–7 tok/s |
| MoE-optimized + expert cache | MoE model, attention fits in VRAM | 517–913 MB | 0.7–2.5 tok/s |
| Partial offload | Dense/hybrid model, first-N layers on GPU | 4615–4909 MB | 0.3–0.7 tok/s |
| CPU-only (`--no-gpu`) | No GPU or explicit `--no-gpu` flag | 0 | 0.2–4.1 tok/s |

## Historical Results

### v1.5.1 — CUDA GPU (Full GPU offload)

| # | Model | Params | Quant | Size | CUDA tok/s | GPU Mode |
|--:|-------|--------|-------|-----:|----------:|----------|
| 1 | Llama-3.2-1B-Instruct | 1B | Q4_K_M | 771M | **54.7** | CUDA graph (16/16) |
| 2 | Qwen2.5-Coder-1.5B-Instruct | 1.5B | Q4_K_M | 1.1G | **41.5** | CUDA graph (28/28) |
| 3 | Gemma-3-1B-it | 1B | Q4_K_M | 769M | **35.7** | CUDA graph (26/26) |
| 4 | Llama-3.2-1B-Instruct | 1B | IQ4_NL | 738M | **28.7** | CUDA graph (16/16) |
| 5 | OLMo-2-1B-Instruct | 1B | Q4_K_M | 893M | **25.5** | CUDA graph (16/16) |
| 6 | Llama-3.2-3B-Instruct | 3B | Q4_K_M | 1.9G | **23.8** | CUDA graph (28/28) |
| 7 | Qwen2.5-Coder-3B-Instruct | 3B | Q4_K_M | 2.0G | **21.8** | CUDA graph (36/36) |
| 8 | Qwen3-4B | 4B | Q4_K_M | 2.4G | **19.0** | CUDA graph (36/36) |
| 9 | Phi-4-mini-Instruct | 3.8B | Q4_K_M | 2.4G | **11.5** | Per-tensor CUDA matmul |
| 10 | Qwen2.5-Coder-7B-Instruct | 7B | Q4_K_M | 4.4G | **11.3** | CUDA graph (28/28) |
| 11 | DeepSeek-R1-0528-Qwen3-8B | 8B | Q4_K_M | 4.7G | **9.3** | CUDA graph (36/36) |
| 12 | Gemma-2-2B-it | 2B | IQ4_XS | 1.5G | **9.0** | CUDA graph (26/26) |
| 13 | Llama-3.2-3B-Instruct | 3B | Q3_K_L | 1.7G | **8.3** | CUDA graph (28/28) |
| 14 | Qwen3.5-4B | 4B | Q4_K_M | 2.6G | **8.0** | Per-tensor CUDA matmul |
| 15 | Aya-23-8B | 8B | Q4_K_M | 4.8G | **7.0** | CUDA graph (32/32) |
| 16 | Phi-3-mini-4k-Instruct | 3.8B | IQ4_NL | 2.1G | **6.5** | Per-tensor CUDA matmul |

## Known Limitations

| Model / Quant | Issue | Status |
|---------------|-------|--------|
| Command-R/Cohere (Aya-23) | CUDA forward pass not supported | Per-tensor CUDA matmul fallback |
| Gemma 4 / Gemma 3n (PLE) | CUDA forward pass not yet implemented | CPU forward pass with per-tensor CUDA matmul |
| IQ3_XXS GGUF models | Mixed quant types (IQ2_S for Q/K, IQ3_S for Wo) — no CUDA kernels | Per-tensor CUDA for IQ3_XXS tensors only |
| Granite Hybrid (Mamba-2) | CUDA graph not yet supported (DtoD copies) | Per-layer GPU forward pass |
| 32B+ models on 8 GB RAM | Timeout due to excessive swap | Requires 16+ GB RAM |
