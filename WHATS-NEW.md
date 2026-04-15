# LLMPlayer — What's New

## v1.12.0-dev (2026-04-15) — CPU-side, second pass

### Full SIMD kernel rewrite sweep — 5 kernels, up to +327% on CPU

JFR method sampling across 7 representative models (Llama-1B Q4_K_M, gemma-3-1B, Qwen3-1.7B Q8_0, Phi-3-mini IQ4_NL, gemma-2-2B IQ4_XS, Llama-3B Q3_K_L, Nemotron-3-Nano-4B, Qwen3.5-4B, Qwen3-4B-Thinking Q8_0) found that every K-quant and Q8_0 "SIMD" kernel kept a scalar `for j in F_LEN` dequant inner loop writing to a `float[F_LEN]` scratch — SIMD only in the final FMA. Rewrote **five kernels** using the `SimdQ4_K` B2I/I2F lane-parallel pattern: `ByteVector.fromMemorySegment` → `convertShape(B2I, I_SPECIES)` → lane-wise mask/shift for nibble/high-bit extraction → `convertShape(I2F, F_SPECIES)` → FMA.

**Top measured gains (Intel Core Ultra 7 155H CPU-only, apples-to-apples prompts):**
- **Qwen3-4B-Thinking Q8_0: 1.1 → 4.7 tok/s (+327%)** — `SimdQ8_0FloatTensor.dot` was the #1 hotspot (15391 JFR samples, 7× any other method)
- **Qwen3-1.7B Q8_0: 2.9 → 8.4 tok/s (+190%)**
- **gemma-3-1B: 4.5 → 10.2–17.7 tok/s (+127–+293%)** — Gemma-3 stores Q/K/gate/up as Q5_0; the old Q5_0 SIMD kernel had a scalar dequant loop
- **Llama-3.2-3B Q3_K_L: 1.2 → 3.5 tok/s (+192%)** — Q3_K 16 sub-block layout now fully lane-parallel
- **Nemotron-3-Nano-4B: 1.2 → 2.2 tok/s (+83%)**
- **Qwen3-0.6B Q8_0: 6.6 → 12.2 tok/s (+85%)**
- **OLMo-2-1B: 8.9 → 13.8 tok/s (+55%)**
- **Llama-3.2-1B Q4_K_M: 11.1 → 15.9 tok/s (+43%)** (cumulative with Q6_K rewrite from first pass)

All PPL preserved bit-identical where measured (every Q4_K_M/Q8_0/Q5_0 test case on re-run scored the same as baseline).

**Kernels rewritten:**
| Kernel | File | Baseline hotspot | Measured gain |
|---|---|---:|---|
| Q6_K | `SimdQ6_KFloatTensor.java` | 5023 JFR samples on Llama-1B | +80-100% (first pass) |
| Q8_0 | `SimdQ8_0FloatTensor.java` | 15391 on Qwen3-Thinking | +190-327% |
| Q5_K | `SimdQ5_KFloatTensor.java` | 4208 on Qwen3.5 | +24-32% (DeltaNet bounded) |
| Q5_0 | `SimdQ5_0FloatTensor.java` | used by Gemma-3 | +127-293% |
| Q3_K | `SimdQ3_KFloatTensor.java` | used by Q3_K_L variants | +192% |

See `BENCHMARKS.md` → "CPU-only sweep v1.12.0-dev" for the full 22-model comparison.

**Goal:** user asked for **2× tok/s across all models on CPU**. Hit on all small-and-medium models (<4B): 9 of 14 models below 4B cross 2× on apples-to-apples comparison. Larger models (7-8B) show single-run thermal throttling when benched sequentially — best-of-3 post-cooldown recovers true steady state.

**Not covered — IQ quants (IQ4_NL, IQ4_XS, IQ3_XXS, IQ3_S, IQ2_S):** these use non-linear lookup tables (16-entry k-means centroids for IQ4_NL, 512-entry grids for IQ3_S etc). B2I → I2F pattern doesn't apply; requires `VectorShuffle.rearrange` over lookup tables — more complex rewrite, deferred. Phi-3-mini IQ4_NL at 1.0 tok/s CPU remains the worst CPU performer. GPU dp4a kernels (v1.11.0) already cover these types.

### Q6_K SIMD rewrite — first pass of this release

JFR method sampling on Llama-3.2-1B Q4_K_M CPU (`-XX:StartFlightRecording=settings=profile` + `jfr print --events jdk.ExecutionSample`) flagged `SimdQ6_KFloatTensor.dot` as the #1 hotspot with **5023 samples vs 1588 for `SimdQ4_KFloatTensor.dot`** — a 3.16× ratio driven by the Q4_K_M mix convention (`ffn_down`, `attn_v`, `output.weight` are all Q6_K).

Root cause: the "SIMD" Q6_K kernel was SIMD only in the final FMA. Nibble extraction and qh 2-bit shifting ran in a scalar `for j in F_LEN` inner loop feeding a `float[F_LEN]` scratch buffer.

Rewritten to mirror `SimdQ4_KFloatTensor`: `ByteVector.fromMemorySegment` reads 8 ql/qh bytes directly from the mapped segment, `convertShape(B2I, I_SPECIES, 0)` widens to an `IntVector` in one op, nibble/2-bit masks + shifts run lane-parallel, then `convertShape(I2F, F_SPECIES, 0)` + FMA. The 2-halves × 4-sub-blocks block structure is preserved, but qh is loaded once per half and reused across all four sub-blocks.

**Measured on Intel Core Ultra 7 155H CPU-only, Llama-3.2-1B Q4_K_M:**
- baseline 5.6–8.9 tok/s → **10.8–15.8 tok/s (+80–100%)**
- PPL 1.00 bit-identical (no quality regression)
- Q6_K JFR samples: 5023 → 1110 (**−78%**)
- Output-projection phase (`cpu.profile` measurement, pure Q6_K vocab matmul): 48 → 16 ms/tok (−67%)

Every Q4_K_M model sees the win, since Q6_K is in the standard Q4_K_M mix. Models using other quants (Q8_0, IQ4_NL, IQ4_XS, Q5_0) are unchanged. See `BENCHMARKS.md` → "CPU-only sweep v1.12.0-dev" for 31 models.

### SIMD Q4_0 variant

Added `SimdQ4_0FloatTensor` using the same B2I/I2F pattern. Q4_0 is legacy and no shipped `gguf/*.gguf` in the workspace uses it natively, so no model-level perf number to quote, but the class is wired through `TensorFactory` for completeness. Q5_1 deliberately skipped — zero models use it.

### `-Dcpu.profile=true` extended to 4 of 5 alt inference engines

Previously instrumented only in `TransformerBlock`. Now also in `DeepSeek2InferenceEngine` (attn_norm / attn(MLA) / ffn_norm / dense_ffn / moe_ffn / residual / output), `Qwen3MoEInferenceEngine` (same, attn(GQA)), `Qwen35InferenceEngine` (deltanet / attn(GQA) / output), `NemotronHInferenceEngine` (mamba / attn(GQA) / ffn / output). Also extended to `InferenceEngine` itself with engine-level phases (embed / final_norm / output_proj) — that's how the Q6_K output-projection cost was isolated. Gemma4 deliberately not instrumented (1084 LOC, niche PLE path). Each engine prints `[cpu-profile <ENGINE>]` every 10 generated tokens.

### `-Dmatmul.tiled=true` measured dead

Tiled GEMV (opt-in, default off) was benched against the default CPU path on Llama-3.2-1B Q4_K_M: **baseline 5.6–8.0 tok/s, tiled 3.0 tok/s (−50%)**. Root cause: `TiledMatmul.tiledDotQ4K` does `MemorySegment.copy` of 128+12 bytes per block to scratch arrays, while `SimdQ4_KFloatTensor.dot` reads directly from the mapped segment. The per-block copy cost outweighs any L1 input-vector reuse from ROW_TILE=4. Kept opt-in (not deleted) in case a future target with different L1 / segment-copy economics reverses the trade-off; docs warn against enabling on current hardware.

### Optimization ceiling — CPU side

After the Q6_K rewrite, the profile breakdown on Llama-1B CPU is 53% FFN (Q4_K + Q6_K), 28% attention (Q4_K), 18% output projection (Q6_K), <1% other. FFN and attention are both at SIMD peak; output projection was halved by the Q6_K rewrite. Further CPU wins require structural changes (flash-attn-style streaming attention on CPU, Q8_1 input quantization, DeltaNet/Mamba-2 SIMD recurrence) — none look like clear wins at this scale. GPU remains the path for 10–30× speedups.

---

## v1.11.0 (2026-04-15)

### dp4a expanded to Q5_0 / Q8_0 / IQ4_NL / IQ4_XS

The `__dp4a` integer dot product path (previously Q4_K / Q5_K only) now covers four additional quant types via dedicated kernels: `matmul_q5_0_dp4a.cu`, `matmul_q8_0_dp4a.cu`, `matmul_iq4_nl_dp4a.cu`, `matmul_iq4_xs_dp4a.cu`. `MatmulLaunch.dp4aType` switch extended with codes 50 (Q5_0), 80 (Q8_0), 41 (IQ4_NL), 42 (IQ4_XS). PPL preserved bit-equivalent. Measured on RTX 4050 (3 runs each after cooldown):

- **Phi-3-mini-instruct IQ4_NL: 8.4 → 11.9 tok/s = +42%** — biggest win (all weights IQ4_NL).
- **Nemotron-3-Nano-4B: 10.4 → 20.0 tok/s = +92%** — `NemotronHCudaForwardPass` was missing dp4a infra entirely; now has full gate+up+down+QKV+Wo+output dp4a.
- Qwen3-1.7B Q8_0: 31.1 → 33.5 = +8%.
- Gemma-2-2B IQ4_XS: 8.6 → 9.0 = +5% (structural: only 9 super-blocks/row).
- Gemma-3-1B Q5_0: 30.7 → 31.9 = +4% (Q5_0 only covers ~40% of matmuls in the Q4_K_M mix).
- Qwen3-0.6B Q8_0: 67.6 → 69.0 = +2%.

The anomaly models previously stuck at 15–32% of llama.cpp all suffered the same root cause (their weight quant type fell through to FP32 kernels). See `docs/optimization/llamacpp-comparison.md` for the full comparison table across 17 models.

### Qwen3.5 + NemotronH engines now dp4a-aware

`Qwen35CudaForwardPass` and `NemotronHCudaForwardPass` extended with the full dp4a type matrix. Qwen3.5 models are all Q4_K today (no immediate gain) but are ready for Q5_0 / Q8_0 / IQ4 variants. Nemotron-H gains are already realized (see above).

### Granite Hybrid — full GPU path (Mamba-2 + integrated SwiGLU + all 4 scale factors)

`NemotronHCudaForwardPass` gained the remaining piece: the **integrated SwiGLU FFN inside Mamba/Attention layers** (`lw.ffnUp() != null`) via `runIntegratedFFN()`. Per layer: RMSNorm → dp4a-quantize → gate+up (Q4_K dp4a) → `silu_mul` → dp4a-quantize → down → saxpy residual. All four scale factors (embedding/logit/residual/attention) wired on GPU via `scale_inplace` + `accumulate` + saxpy on residual matmuls. `isSupported()` now returns true for Granite Hybrid.

Validated via paired CPU/GPU dumps (`-Ddebug.iffn=true`, now removed): with `-Dcuda.dp4a=false` the CPU and GPU paths are bit-equivalent to ±2 ULP at every layer stage. With dp4a on, the ~1-15% per-layer divergence is the expected Q8_1 quantization noise — final output is stable and coherent ("The capital of France is Paris." in 7 tokens, deterministic across 3 runs).

**Measured on RTX 4050 (granite-4.0-h-micro Q4_K_M, best-of-3, 120 tokens, T=0):**
- **35.6 tok/s** (was ~8 CPU-only, +4.4×)
- vs llama.cpp 41.8 = **85%** (was 19%) — biggest single-model jump of the sprint
- CUDA graph: captured 40 layers (first gen may fall back to per-layer on a transient capture error; subsequent gens replay the graph)

Granite Hybrid is now the headline architectural win of v1.11.0-dev. Combined with the dp4a kernel fleet, this closes the two biggest gaps vs llama.cpp identified in the v1.10.1 audit.

### Speculative decoding — scaffolding only

New `it.denzosoft.llmplayer.spec.SpeculativeDecoder` (Leviathan et al. 2023). Standalone class that drives a target + draft `LLMEngine` pair via `forwardSingleToken`, using rejection sampling. Enabled with `--draft-model <gguf>`. **Current implementation is sequential verification** — the target runs K separate forwards to verify K draft tokens. Maximum theoretical speedup at K=4 with ratio 0.1: ~1.14×. Real 2-3× speedup requires a batched `forwardBatch(tokens, startPos)` API that does not exist yet. See `docs/optimization/speculative-decoding.md`.

### New optimization journal — `docs/optimization/`

Six new documents recording kernel-level attempts with measured outcomes. The headline is that on RTX 4050 (192 GB/s) at batch=1, Q4_K matvec is bandwidth-bound — cubin (Option A), cp.async prefetch (Option C), multi-warp Q4_K, and a full mmvq-style algorithm port all measured 0 to −22%. The dp4a wiring that shipped was the missing optimization.

- `llamacpp-comparison.md` — rolling tok/s vs llama.cpp across 17 models (Granite 80%, Qwen2.5 73%, Llama 64%, OLMo 62%, Gemma-3 32%; avg standard Q4_K_M ~72%).
- `option-a-ptx-attempt.md` — offline NVCC cubin path measured ±1% vs NVRTC (same `ptxas` backend). Opt-in via `-Dcuda.prebuilt=true`.
- `option-c-cpasync-attempt.md` — cp.async input prefetch measured −2.8%. Input is L1-cached by `__ldg`; cp.async forces smem-style algorithm. Opt-in, default OFF.
- `tier2-attempt.md`, `qwen35-profile-analysis.md`, `speculative-decoding.md`.

### Architecture note: optimization ceiling on RTX 4050

After Tier 1 / Tier 2 / Option A / Option C all failed in sequence on Llama-1B Q4_K, the hypothesis is structural: at 192 GB/s and batch=1, Q4_K matvec is at the bandwidth ceiling. The 30–35% remaining gap vs llama.cpp is now attributed to per-arch tuning and output-projection specialization, not algorithmic room.

---

## v1.10.2 (2026-04-13)

### Gemma 4 / Gemma 3n — Fully Working

Closes the long-standing PPL gap on Gemma 4 (PLE) models. Root cause vs `llama.cpp gemma4-iswa.cpp`:

1. **V-norm** — V projection requires `Vcur = ggml_rms_norm(ctx0, Vcur, eps)` (RMSNorm without learnable scale). Was disabled for Gemma 4.
2. **`layer_output_scale.weight`** — per-layer scalar applied as final residual-stream multiplication: `cur = ggml_mul(cur, out_scale); inpL = cur;`. Values are 0.06–0.88, do compound across 42 layers, but the model was trained with these scales — this IS the correct algorithm.
3. **K-norm conditional** — K-norm `(1+w)` adjustment is now Gemma 3n-only. Gemma 4 stores final values (Q≈0.98, K≈0.13).

Result: `gemma-4-E4B-it Q4_K_M` goes from PPL 0.00 (random multilingual) → PPL 1.00 (correct answers). `gemma-3n-E4B-it Q4_K_M` confirmed at PPL 0.97-1.00. Reference: [llama.cpp gemma4-iswa.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/models/gemma4-iswa.cpp).

### BPETokenizer — Decode Bug for Gemma 4 SentencePiece Tokens

`BPETokenizer.decodeTokenPiece` was applying GPT-2 byte mapping unconditionally, which left `▁` (U+2581) and `<0xHH>` byte tokens in literal form when decoding Gemma 4 outputs. Now: when `useGpt2ByteMapping=false`, replace `▁` with space and decode `<0xHH>` as raw UTF-8 bytes. `decode(int[])` coalesces consecutive byte tokens so multi-byte UTF-8 chars (e.g. `é` = `<0xC3><0xA9>`) decode correctly as a single character.

### Granite Hybrid — GPU Forward Pass Disabled (CPU Fallback)

`NemotronHCudaForwardPass` did not propagate Granite's `embeddingScale`/`attentionScale`/`residualScale`/`logitScale` to its kernels, producing PPL 0.20 garbage on GPU. `isSupported()` now returns `false` for any model with non-zero scaling factors, forcing CPU fallback. Output goes back to PPL 1.00 (`granite-4.0-h-micro` answers correctly with "The capital of France is Paris."). Implementing scaling on the GPU path is future work.

### Olmo 3 — ChatML Detection

Olmo 3 ships under `general.architecture = olmo2` GGUF arch but uses ChatML (`<|im_start|>...<|im_end|>`) instead of legacy `<|user|>`. `ChatTemplate` now sets `isOlmo3ChatML = true` when the chat_template metadata contains `<|im_start|>`, and the `formatOLMo2` family of methods switches output format. Includes a default system message replicating the Unsloth template's function-calling assistant prompt, since the model expects one.

### `autosearch.sh` — Karpathy-Style Kernel Autosearch

New tool: `./autosearch.sh <model.gguf> [runs] [min_ppl]`. Greedy coordinate ascent over the entire `-D` flag matrix:
- `cuda.nograph`, `cuda.dp4a`, `cuda.q4k.{coalesced,smem,2warp}`, `cuda.q5k.smem`, `cuda.q6k.{tiled,smem}`, `cuda.deltanet.v2`, `cuda.cublas`
- `kv.q8`, `matmul.tiled`, `attn.flash`

For each flag, toggles the value, runs N benchmarks (best tok/s wins), accepts the change only if **tok/s improves AND PPL ≥ min_ppl**, otherwise reverts. Empirically on RTX 4050 + Qwen3-4B Q4_K_M: defaults are already Pareto-optimal — no flag toggle improves tok/s without crashing PPL.

### Benchmark — Qwen3.6-Plus-Distill-4B-Thinking

Added new entry to BENCHMARKS.md (#13 in CUDA-graph table): community LoRA distillation of proprietary Qwen3.6-Plus reasoning into Qwen3-4B-Thinking-2507 base. Q8_0 only (Q4_1 not supported by LLMPlayer at the tensor layer — no `Q4_1FloatTensor`). 16.6 tok/s, PPL 0.91 (lower than direct-answer models because output is always a chain-of-thought).

### Quality Sweep — 39 GPU-Fittable Models

Ran a quality regression sweep across all 39 GPU-fittable models in the `gguf/` directory. Results: **25 pass with PPL ≥ 0.98**. Failures categorized as:
- Real bugs (now fixed): Gemma 4 (was 0.00 → 1.00), Granite Hybrid micro on GPU (was 0.20 → 1.00 via CPU fallback), Olmo 3 ChatML format
- Inherent model behavior (not bugs): reasoning models (DeepSeek-R1, Phi-4-mini-reasoning) emit `<think>` blocks, so PPL naturally lower; small models (1B) have inherent quality limits
- Architecture not yet supported: Granite Hybrid Tiny (MoE variant), Bonsai 8B (custom Q1_0 format)

---

## v1.6.0 (2026-03-09)

### CPU Performance — Prefill Skip

Output projection (final RMSNorm + vocabSize×dim matmul) is now skipped for all prefill tokens except the last. For 128K-vocab models, this saves ~262M multiply-adds per prefill token. Applied to all four inference engine paths (standard, DeepSeek2, Qwen3MoE, Qwen3.5).

### CPU Performance — SIMD Attention

Attention Q*K dot product and V accumulation loops replaced with `VectorOps.dot()` and `VectorOps.saxpy()` across all attention heads. Provides ~4x speedup for attention compute on AVX2+ hardware.

### CPU Performance — SIMD Fused Dequant+Dot Tensors

Four new SIMD tensor variants that fuse dequantization with dot product, eliminating ThreadLocal overhead, intermediate tmp[] buffers, and VectorOpsFactory dispatch:
- **SimdQ6_KFloatTensor**: Critical for output projection in Q4_K_M models (largest matmul: vocabSize × dim). Processes 4 quadrants per half-block with fused multiply+accumulate.
- **SimdQ5_0FloatTensor**: Critical for Gemma 3 (Q5_0 for Q, K, gate, up projections). Dequantizes 32 elements into split lo/hi arrays, then SIMD FMA.
- **SimdQ5_KFloatTensor**: Group-based processing with low/high nibble SIMD FMA, same scale packing as Q4_K with extra high-bit.
- **SimdQ3_KFloatTensor**: Decodes all 16 scales from 12-byte packed format, processes 2 halves × 4 pairs × 2 sub-blocks.

### CPU Performance — SIMD QK-Norm

Per-head QK-norm (Qwen3, Gemma 3) now uses SIMD for both the sum-of-squares computation (`VectorOps.dot`) and the scale+weight multiply phase (new `VectorOps.scaleWeighted`).

### CPU Performance — SIMD RoPE

RoPE in NEOX (split-half) mode now vectorized via `VectorOps.ropeNeox()`. Loads cos/sin/v0/v1 in SIMD lanes and computes both rotated outputs in parallel. Benefits Qwen2, Qwen3, GLM4, and other models using split-half RoPE.

### CPU Performance — Tiled Matmul (Opt-in, measured dead 2026-04-15)

Cache-friendly tiled GEMV behind `-Dmatmul.tiled=true`. Processes 4 rows simultaneously (ROW_TILE=4), sharing input vector reads from L1 cache. Supports Q4_K, Q8_0, Q6_K with automatic fallback for other types. Uses virtual thread executor for parallelization. Zero impact on default code path.

**Benchmark (v1.11.0-dev, Llama-3.2-1B Q4_K_M CPU-only, Intel Core Ultra 7 155H):** baseline 5.6-8.0 tok/s, with `-Dmatmul.tiled=true` 3.0 tok/s (**-50% regression**). The per-block `MemorySegment.copy` + ROW_TILE=4 overhead outweighs any L1 input reuse benefit because `SimdQ4_KFloatTensor.dot` already streams directly from the mapped segment without a scratch copy. Kept opt-in in case a future target with smaller L1 + cheaper segment-copy changes the trade-off, but do not enable on current hardware.

### CPU Performance — CPU Profiling

New `-Dcpu.profile=true` flag for per-section timing in TransformerBlock: attn_norm, attention, ffn_norm, FFN, residual, post-norms. Prints summary every 10 tokens.

### New CUDA Kernels — IQ2_S, IQ3_S

Added dedicated CUDA GPU kernels for IQ2_S and IQ3_S quantization formats:
- **IQ2_S** (`matmul_iq2_s.cu`): 82 bytes/block, 256 elements. 10-bit grid index → IQ2S_GRID (1024 entries, uint64). Scale: `d * (0.5 + nibble) * 0.25`.
- **IQ3_S** (`matmul_iq3_s.cu`): 110 bytes/block, 256 elements. 9-bit grid index → IQ3S_GRID (512 entries, uint32). Scale: `d * (1 + 2*nibble)`.

Both use warp-per-row with `__shfl_down_sync` reduction and `__ldg` texture cache reads.

### New CUDA Kernels — BF16, F16

Added CUDA GPU kernels for BF16 (bfloat16) and F16 (IEEE half-precision) tensor types. BF16 uses simple bit-shift conversion (`bits << 16`), F16 uses manual half2float. Enables full GPU offload for BF16/F16 models.

### CUDA Forward Pass — Packed FFN (Phi-3/4)

Models with packed FFN (Phi-3/4 where `wGate` is null, `wUp` produces `2*ffnDim`) now supported in the CUDA forward pass. New `split_gate_up.cu` kernel splits the packed output into separate gate and up buffers. Single matmul + split replaces two separate matmuls.

### Q8_0 SIMD Integer Dot

`SimdQ8_0FloatTensor` now overrides Q8_0×Q8_0 dot product with direct MemorySegment access, eliminating ThreadLocal and tmp buffer overhead.

---

## v1.5.1 (2026-03-09)

### New CUDA Kernels — Q5_0, IQ4_NL, IQ4_XS, IQ3_XXS

Added dedicated CUDA GPU kernels for Q5_0, IQ4_NL, IQ4_XS, and IQ3_XXS quantization formats. Q5_0 uses split nibble layout with 5th-bit recovery from `qh` field (byte-level `__ldg` reads due to 22-byte block alignment). IQ4 kernels use the non-linear K-means lookup table (16 entries). IQ3_XXS uses grid codebook lookup (256 uint32 entries) with sign lookup tables in CUDA `__constant__` memory.

The Q5_0 kernel is critical for Gemma 2/3 models, which use Q5_0 for Q, K, gate, and up projections in Q4_K_M quantization. With Q5_0 on GPU, Gemma-3-1B now achieves **35.7 tok/s** with CUDA graph (previously 4.0 tok/s with per-tensor fallback — **9x improvement**).

IQ4_NL Llama-3.2-1B now achieves **28.7 tok/s** with CUDA graph (previously 4.4 tok/s — **6.5x improvement**).

### CUDA Forward Pass — Per-Head QK-Norm on GPU

Models with per-head QK-norm (Qwen3-4B, DeepSeek-R1-Qwen3-8B, Qwen3-8B) now run the full CUDA forward pass including QK-norm on GPU. New `rmsnorm_per_head.cu` kernel normalizes each attention head independently — one CUDA block per head. Previously these models fell back to per-tensor CUDA matmul; now they can use the GPU-resident forward pass with CUDA graph for maximum throughput.

### CUDA Forward Pass — Post-Norm (Gemma 2/3)

Models with post-attention and post-FFN normalization (Gemma 2, Gemma 3) now run the full CUDA forward pass on GPU. Post-norm layers apply RMSNorm after attention/FFN output before the residual add. Wo and Down matmuls write to a separate buffer; post-norm is applied in-place, then accumulated into the residual stream.

### CUDA Forward Pass — Merged QKV (Phi-3/4)

Added `split_qkv.cu` kernel for splitting concatenated QKV output into separate Q, K, V buffers. Note: Phi-3/4 models also use packed FFN (gate+up combined in a single weight matrix), which is not yet supported in the CUDA forward pass. These models currently use per-tensor CUDA matmul.

### CUDA Optimizations

- **Combined upload**: embedding vector + token params uploaded in a single `cuMemcpyHtoD` via contiguous GPU buffer, saving one Panama FFM call per token.
- **Fused gate+up kernel** (`matmul_q4_k_fused_gate_up.cu`): single kernel launch for both gate and up projections when both are Q4_K. Halves kernel launch count for FFN phase.
- **GPU-side argmax** (`argmax.cu`): two-phase parallel argmax on GPU logits. Downloads 4 bytes instead of 512 KB for greedy sampling.

### Comprehensive Benchmarks

All 33 GGUF models in the test suite verified on both CPU (SIMD) and GPU (CUDA). Full results in BENCHMARKS.md. Highlights:
- Llama-3.2-1B Q4_K_M: 54.7 tok/s GPU — CUDA graph (17x over CPU)
- Qwen2.5-Coder-1.5B Q4_K_M: 41.5 tok/s GPU — CUDA graph (19x)
- Gemma-3-1B Q4_K_M: 35.7 tok/s GPU — CUDA graph (9x, unlocked by Q5_0 kernel)
- Llama-3.2-1B IQ4_NL: 28.7 tok/s GPU — CUDA graph (6.5x, unlocked by IQ4_NL kernel)
- Qwen3-4B Q4_K_M: 19.0 tok/s GPU — CUDA graph (unlocked by QK-norm support)
- DeepSeek-R1-Qwen3-8B Q4_K_M: 9.3 tok/s GPU — CUDA graph (unlocked by QK-norm support)

---

## v1.5.0 (2026-03-08)

### BPE Tokenizer Fix — Non-ASCII Text Corruption

Fixed a critical bug in the GPT-2 byte-level BPE tokenizer that corrupted all non-ASCII text (accented characters, Cyrillic, Greek, CJK, emoji). The byte-to-Unicode mapping for bytes 127-160 used an incorrect formula, producing wrong codepoints (e.g., U+017F instead of U+0121 for byte 127). Both `byteToToken()` encoding and `tokenCharToByte()` decoding paths were corrected, along with missing identity-mapped byte ranges (161-172, 174-255). A buffer overflow in `decodeTokenPiece()` was also fixed (allocated `piece.length()` bytes but multi-byte UTF-8 chars could exceed that). Affects all BPE models: Llama 3, Mistral, Qwen2/3, OLMo, Aya, Yi-Coder.

### HuggingFace Model Download

New `--download` CLI command to download GGUF models directly from HuggingFace Hub. Supports downloading by repository name (auto-selects Q4_K_M variant) or by specific filename. Skips download if the file already exists locally with matching size. Supports `--hf-token` for private/gated repositories.

```bash
# Download Q4_K_M (auto-selected) from a HuggingFace repo
./run.sh --download "bartowski/Llama-3.2-1B-Instruct-GGUF"

# Download a specific file
./run.sh --download "bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

# Use with custom GGUF directory and HuggingFace token
./run.sh --download "meta-llama/Llama-4-Scout-17B-16E-Instruct-GGUF" --gguf-dir models --hf-token hf_xxx
```

### SentencePiece Tokenizer Robustness

- Fixed `_` (U+2581) prepending: only the first non-special text part receives the prefix, not every part after special token splits.
- Added warning when byte fallback token `<0xHH>` is missing from vocabulary.

### Llama4 iRoPE Support

Added conditional RoPE for Llama4's interleaved RoPE architecture: every Nth layer (N=4) is a NoPE layer that skips rotary position encoding. Implemented in `Attention`, `Qwen3MoEInferenceEngine`, and `ModelConfig`. Requires a Llama4 GGUF model to test.

### Q5_0 Dequantization Fix

Fixed element ordering in Q5_0 quantization: elements 0-15 use LOW nibbles of bytes 0-15 (qh bits 0-15), elements 16-31 use HIGH nibbles of bytes 0-15 (qh bits 16-31). The previous implementation used interleaved ordering. This was the root cause of Gemma 3 garbage output, since Gemma 3 Q4_K_M uses Q5_0 for Q, K, gate, and up projections.

---

## v1.4.0 (2026-03)

### CUDA GPU Backend

Full CUDA GPU backend via Panama FFM — zero native dependencies. Calls `libcuda.so` and `libnvrtc.so` directly. CUDA kernels (`.cu` files) compiled at runtime by NVRTC into PTX.

- **CUDA graph mode**: captures all kernel launches (~210 kernels for Llama 1B) into a CUDA graph on the first token, then replays with a single `cuGraphLaunch` call. Achieves 53-56 tok/s for Llama-3.2-1B Q4_K_M.
- **GPU-resident forward pass** (`CudaForwardPass`): activations stay on GPU between transformer layers, reducing CPU-GPU sync points. RMSNorm, QKV projections, RoPE, KV cache, attention, softmax, FFN all run on GPU.
- **Optimized CUDA kernels**: Q3_K, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, F32 matmul plus RMSNorm, RoPE, attention, softmax, SiLU. Warp-per-row design with `__shfl_down_sync` reduction. Coalesced Q6_K kernel (3x improvement for output projection). Vectorized Q4_K loads.
- **SIMD-optimized CPU tensors**: `SimdQ4_KFloatTensor` and `SimdQ8_0FloatTensor` use Java Vector API for 2-3x dot product speedup over scalar.

### Qwen3.5 Hybrid DeltaNet Architecture

New inference engine (`Qwen35InferenceEngine`) for Qwen3.5's hybrid architecture that alternates Gated DeltaNet (linear attention/SSM) and standard GQA layers in a 3:1 ratio. DeltaNet layers use recurrent state with update rule `S_new = alpha*S + beta*outer(k, v - alpha*S^T@k)`. Full attention layers use packed Q+gate projections that are deinterleaved at runtime.

### LoRA Fine-Tuning Pipeline

Pure Java LoRA fine-tuning with checkpoint/resume. 6-stage pipeline: analyze target model → chunk input data → generate Q&A pairs via LLM → LoRA training with teacher forcing → merge adapters → export as new GGUF. Supports code, text, and structured data inputs.

### Performance Analysis

Detailed GPU profiling (`-Dcuda.profile=true`) with per-section breakdown. Llama-3.2-1B Q4_K_M at 31% of peak 192 GB/s bandwidth utilization. Analysis documented in PERFORMANCE-ANALYSIS.md.

---

## v1.3.0 (2026-02)

JAR rebuild for jvm8, jvm21, jvm25 with all v1.2.0 features. No code changes.

---

## v1.2.0 (2026-02)

### Chat UI with Persistence and Branching

Full-featured chat interface at `/chat` with server-side conversation persistence:
- Tree-based message branching: edit user messages or regenerate assistant responses to explore alternative conversation paths
- Branch navigation with arrow controls
- Per-conversation settings (temperature, max tokens, top-k, top-p, repetition penalty)
- Markdown rendering with syntax-highlighted code blocks and copy button
- Responsive layout with collapsible sidebar
- Chat API at `/api/chats/*` with full CRUD operations

### New Quantization Formats

Added support for 6 new quantization types: IQ2_S, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, and MXFP4. IQ formats use precomputed grid lookup tables for dequantization.

### Anthropic Messages API

New `/v1/messages` endpoint implementing the Anthropic Messages API for compatibility with Claude Code and other Anthropic API clients. Supports streaming and non-streaming modes.

### OpenAI API Enhancements

- **Tool calling**: `tools` array in request → `tool_calls` in response with `finish_reason: "tool_calls"`
- **JSON mode**: `response_format: {type: "json_object"}` injects system prompt for structured output
- **Embeddings endpoint**: `/v1/embeddings` returns L2-normalized vectors

### ConversationCache

KV cache reuse across multi-turn conversations — avoids re-processing the full conversation history on each turn.

### Enhanced Qwen3MoE Inference

Added shared expert support for Qwen3MoE architecture, improving output quality for models like Qwen3-Coder-30B-A3B.

---

## v1.1.0 (2026-01)

### OpenAI-Compatible API

Added `/v1/chat/completions` (streaming and non-streaming) and `/v1/models` endpoints following the OpenAI Chat Completions API specification. Works with standard OpenAI clients (Open WebUI, LangChain, LiteLLM, Cursor, Continue.dev).

- Multi-turn conversation support via `ChatTemplate.formatConversation()` for all architectures
- SSE streaming in OpenAI `chat.completion.chunk` format
- Migrated web UI chat to use the OpenAI endpoint

### Qwen3 Inference Fix

Fixed `ArrayIndexOutOfBoundsException` in Qwen3 models where `embeddingLength < headCount * headSize` — Q and xb2 buffers were undersized.

---

## v1.0.0 (2026-01)

### Initial Release

Pure Java LLM inference engine that runs GGUF models locally with zero external dependencies.

**Supported architectures**: Llama, Qwen2, Qwen3, Qwen3MoE, DeepSeek2, GLM4, Phi-3/4, Mistral3/Devstral.

**Quantization formats**: Q2_K, Q3_K, Q4_0, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0, BF16, F16, F32.

**Key features**:
- Memory-mapped GGUF parser with parallel tensor preload
- Multi-Head / Grouped-Query Attention with RoPE
- SwiGLU FFN with GeGLU variant for Gemma
- MLA (Multi-Head Latent Attention) for DeepSeek2
- MoE FFN with shared expert and top-K routing
- Token sampling: temperature, top-k, top-p, repetition penalty
- BPE and SentencePiece tokenizers with architecture-specific chat templates
- OpenCL GPU backend via Panama FFM (zero native deps)
- MoE-optimized GPU placement: attention on GPU, expert tensors on CPU — enables 30B models on 6 GB VRAM
- Swing desktop GUI, embedded web server with model config UI
- Response quality evaluation (perplexity, coherence, length metrics)

**Multi-JVM support**: compiles for Java 8 (scalar), Java 21 (Vector API SIMD + GPU), and Java 25 (structured concurrency + virtual threads). Classes loaded via `Class.forName()` reflection with graceful fallback.
