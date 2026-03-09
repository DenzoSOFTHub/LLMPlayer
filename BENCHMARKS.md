# LLMPlayer Benchmarks

## Test Configuration

- **Hardware:** Intel Core Ultra 7 155H (22 cores) + NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM) + 31 GB RAM
- **JVM:** OpenJDK 25.0.2, SimdVectorOps (Vector API), Panama FFI mmap
- **Prompt:** `"Write a Java hello world program"`
- **Parameters:** `--max-tokens 100 --context-length 256` (GPU); `--max-tokens 20-50 --context-length 256` (CPU)
- **Date:** 2026-03-09

## Results v1.5.1 — CUDA GPU (updated)

Tested with `--gpu --gpu-backend cuda`. Key improvements over v1.5.0: QK-norm CUDA forward pass (Qwen3), Q5_0 CUDA kernel (unlocks Gemma 2/3 graph), IQ4_NL/IQ4_XS/IQ3_XXS CUDA kernels, post-norm CUDA forward pass (Gemma 2/3).

### Full GPU offload (model fits in 6 GB VRAM)

| # | Model | Params | Quant | Size | CUDA tok/s | v1.5.0 | Change | GPU Mode |
|--:|-------|--------|-------|-----:|----------:|-------:|-------:|----------|
| 1 | Llama-3.2-1B-Instruct | 1B | Q4_K_M | 771M | **54.7** | 51.7 | +6% | CUDA graph (16/16) |
| 2 | Qwen2.5-Coder-1.5B-Instruct | 1.5B | Q4_K_M | 1.1G | **41.5** | 39.9 | +4% | CUDA graph (28/28) |
| 3 | Gemma-3-1B-it | 1B | Q4_K_M | 769M | **35.7** | 4.0 | **9x** | CUDA graph (26/26) ★ |
| 4 | Llama-3.2-1B-Instruct | 1B | IQ4_NL | 738M | **28.7** | 4.4 | **6.5x** | CUDA graph (16/16) ★ |
| 5 | OLMo-2-1B-Instruct | 1B | Q4_K_M | 893M | **25.5** | 24.0 | +6% | CUDA graph (16/16) |
| 6 | Llama-3.2-3B-Instruct | 3B | Q4_K_M | 1.9G | **23.8** | 23.4 | +2% | CUDA graph (28/28) |
| 7 | Qwen2.5-Coder-3B-Instruct | 3B | Q4_K_M | 2.0G | **21.8** | 21.3 | +2% | CUDA graph (36/36) |
| 8 | Qwen2.5-3B-Instruct | 3B | Q4_K_M | 2.0G | **21.7** | — | new | CUDA graph (36/36) |
| 9 | Qwen3-4B | 4B | Q4_K_M | 2.4G | **19.0** | 10.6 | **1.8x** | CUDA graph (36/36) ★ |
| 10 | Phi-4-mini-Instruct | 3.8B | Q4_K_M | 2.4G | **11.5** | 11.1 | +4% | Per-tensor CUDA matmul* |
| 11 | Qwen2.5-Coder-7B-Instruct | 7B | Q4_K_M | 4.4G | **11.3** | 11.4 | = | CUDA graph (28/28) |
| 12 | DeepSeek-R1-0528-Qwen3-8B | 8B | Q4_K_M | 4.7G | **9.3** | 7.3 | **+27%** | CUDA graph (36/36) ★ |
| 13 | Gemma-2-2B-it | 2B | IQ4_XS | 1.5G | **9.0** | 2.4 | **3.8x** | CUDA graph (26/26) ★ |
| 14 | Llama-3.2-3B-Instruct | 3B | Q3_K_L | 1.7G | **8.3** | 8.5 | = | CUDA graph (28/28) |
| 15 | Qwen3.5-4B | 4B | Q4_K_M | 2.6G | **8.0** | 8.0 | = | Per-tensor CUDA matmul† |
| 16 | Aya-23-8B | 8B | Q4_K_M | 4.8G | **7.0** | 7.1 | = | CUDA graph (32/32) |
| 17 | Phi-3-mini-4k-Instruct | 3.8B | IQ4_NL | 2.1G | **6.5** | 1.6 | **4x** | Per-tensor CUDA matmul*‡ |
| 18 | Llama-3.2-1B-Instruct | 1B | IQ3_XXS | 537M | **1.3** | 1.3 | = | Mixed (IQ2_S/IQ3_S no kernel)§ |
| 19 | Aya-23-8B | 8B | IQ3_XXS | 3.2G | **0.2** | — | new | Mixed (IQ2_S/IQ3_S no kernel)§ |

★ Major improvement from new CUDA kernels or CUDA forward pass features in v1.5.1.
\* Phi-3/4 uses packed FFN (wGate=null); CUDA forward pass not yet supported, falls back to per-tensor CUDA matmul.
† Qwen3.5 uses `Qwen35InferenceEngine` (hybrid DeltaNet); CUDA graph not applicable.
‡ IQ4_NL CUDA kernel active for IQ4_NL tensors; model also has packed FFN blocking full CUDA forward pass.
§ IQ3_XXS GGUF files use mixed quant types (IQ2_S for Q/K, IQ3_S for Wo); these types have no CUDA kernels yet.

### Partial GPU offload / MoE-optimized (model exceeds 6 GB VRAM)

| # | Model | Params | Quant | Size | CUDA tok/s | GPU Layers | Strategy |
|--:|-------|--------|-------|-----:|----------:|------------|----------|
| 1 | DeepSeek-Coder-V2-Lite | 16B (2.4B active) | Q4_K_M | 9.7G | **1.5** | 27/27 attn | MoE-optimized |
| 2 | Qwen3-Coder-30B-A3B | 30B (3B active) | Q4_K_M | 18G | **1.1** | 48/48 attn | MoE-optimized |
| 3 | Phi-4 | 14B | Q4_K | 8.5G | **0.7** | 22/40 | First-N-layers |

Note: CPU results are pure SIMD (Vector API) on Intel Core Ultra 7 155H (22 cores) with `--no-gpu`.

## Historical Results (v1.3.0) — Ranked by tok/s

| # | Model | Params | Quant | File Size | HW Configuration | GPU Layers | VRAM Used | tok/s |
|--:|-------|--------|-------|----------:|------------------|------------|----------:|------:|
| 1 | **Llama-3.2-1B-Instruct** | **1B** | **Q4_K_M** | **771 MB** | **GPU full + CUDA graph** | **16/16** | **770 MB** | **53-56** |
| 2 | **Qwen2.5-Coder-3B-Instruct** | **3B** | **Q4_K_M** | **2.0 GB** | **GPU full + CUDA graph** | **36/36** | **2007 MB** | **19.6** |
| 3 | Llama-3.2-1B-Instruct | 1B | Q4_K_M | 771 MB | GPU full offload (no graph) | 16/16 | 770 MB | 13.1 |
| 4 | OLMo-2-1B-Instruct | 1B | Q4_K_M | 893 MB | GPU full offload | 16/16 | 892 MB | 11.9 |
| 5 | Qwen2.5-Coder-1.5B-Instruct | 1.5B | Q4_K_M | 1.1 GB | GPU full offload | 28/28 | 1065 MB | 11.4 |
| 6 | Llama-3.2-3B-Instruct | 3B | Q4_K_M | 1.9 GB | GPU full offload | 28/28 | 1925 MB | 6.9 |
| 7 | **Qwen3.5-4B** | **4B** | **Q4_K_M** | **2.6 GB** | **GPU full offload** | **32/32** | **2613 MB** | **6.6** |
| 8 | Phi-4-mini-Instruct | 3.8B | Q4_K_M | 2.4 GB | GPU full offload | 32/32 | 2376 MB | 6.5 |
| 9 | Qwen3-4B | 4B | Q4_K_M | 2.4 GB | GPU full offload | 36/36 | 2381 MB | 6.4 |
| 10 | Qwen2.5-3B-Instruct | 3B | Q4_K_M | 2.0 GB | GPU full offload | 36/36 | 2007 MB | 5.5 |
| 11 | Qwen2.5-Coder-7B-Instruct | 7B | Q4_K_M | 4.4 GB | GPU full offload | 28/28 | 4466 MB | 5.0 |
| 12 | Llama-3.2-3B-Instruct | 3B | Q3_K_L | 1.7 GB | GPU full offload | 28/28 | 1731 MB | 4.8 |
| 13 | Llama-3.2-1B-Instruct | 1B | IQ4_NL | 738 MB | CPU only (SIMD) | — | — | 4.1 |
| 14 | Yi-Coder-9B-Chat | 9B | Q4_K_M | 5.0 GB | CPU only (SIMD) | — | — | 3.5 |
| 15 | Aya-23-8B | 8B | Q4_K_M | 4.8 GB | GPU full offload | 32/32 | 4822 MB | 3.2 |
| 16 | DeepSeek-R1-0528-Qwen3-8B | 8B | Q4_K_M | 4.7 GB | GPU full offload | 36/36 | 4794 MB | 3.1 |
| 17 | Gemma-2-2B-it | 2B | IQ4_XS | 1.5 GB | CPU only (SIMD) | — | — | 1.8 |
| 18 | **Qwen3.5-9B** | **9B** | **Q4_K_M** | **5.3 GB** | **GPU partial offload** | **29/32** | **4909 MB** | **1.6** |
| 19 | DeepSeek-Coder-V2-Lite | 16B (2.4B active) | Q8_0 | 16 GB | GPU MoE-optimized | 27/27 attn | 913 MB | 1.6 |
| 20 | DeepSeek-Coder-V2-Lite | 16B (2.4B active) | Q4_K_M | 9.7 GB | GPU MoE-optimized | 27/27 attn | 517 MB | 1.5 |
| 21 | Qwen3-Coder-30B-A3B | 30B (3B active) | Q4_K_M | 18 GB | GPU MoE-optimized | 48/48 attn | 540 MB | 1.2 |
| 22 | Phi-3-mini-4k-Instruct | 3.8B | IQ4_NL | 2.1 GB | CPU only (SIMD) | — | — | 1.2 |
| 23 | Sonar-OSS-20B | 20B (MoE) | MXFP4+Q8 | 12 GB | GPU MoE-optimized | 24/24 attn | 654 MB | 0.8 |
| 24 | Phi-4 | 14B | Q4_K | 8.5 GB | CPU only (SIMD) | — | — | 0.4 |
| 25 | Qwen2.5-Coder-14B | 14B | Q6_K | 12 GB | CPU only (SIMD) | — | — | 0.3 |
| 26 | GLM-4-32B-0414 | 32B | Q4_K_M | 19 GB | GPU partial | 15/61 | 4615 MB | 0.2 |
| 27 | Qwen2.5-Coder-32B | 32B | Q4_K_M | 19 GB | GPU partial | 16/64 | 4732 MB | 0.2 |
| 28 | Qwen2.5-Coder-14B FP16 | 14B | F16 | 28 GB | GPU partial | 8/48 | 4696 MB | 0.2 |

## Known Issues (v1.5.1)

No critical issues. All 22 tested models produce correct output on GPU.

### Known Limitations

| Model / Quant | Issue | Status |
|---------------|-------|--------|
| Phi-3/4 (packed FFN) | CUDA forward pass not supported — wGate=null, wUp has 2x output | Per-tensor CUDA matmul fallback |
| IQ3_XXS GGUF models | Mixed quant types (IQ2_S for Q/K, IQ3_S for Wo) — no CUDA kernels | Per-tensor CUDA for IQ3_XXS tensors only |
| Qwen3.5 (DeltaNet) | Hybrid architecture — no CUDA forward pass | Per-tensor CUDA matmul |

### Fixed in v1.5.1 (vs v1.5.0)

| Model | Issue (v1.5.0) | Fix | Speedup |
|-------|----------------|-----|---------|
| Gemma-3-1B | No CUDA graph (Q5_0 tensors on CPU) | Added Q5_0 CUDA kernel → full CUDA graph | 4.0 → **35.7** tok/s (9x) |
| Qwen3-4B | Per-tensor matmul (QK-norm unsupported) | Per-head QK-norm on GPU → CUDA graph | 10.6 → **19.0** tok/s (1.8x) |
| DSR1-Qwen3-8B | Per-tensor matmul (QK-norm unsupported) | Per-head QK-norm on GPU → CUDA graph | 7.3 → **9.3** tok/s (+27%) |
| Llama-1B IQ4_NL | No IQ4_NL kernel (Q8_0 only on GPU) | IQ4_NL CUDA kernel → full CUDA graph | 4.4 → **28.7** tok/s (6.5x) |
| Gemma-2 IQ4_XS | No IQ4_XS kernel (Q8_0 only on GPU) | IQ4_XS CUDA kernel → CUDA graph | 2.4 → **9.0** tok/s (3.8x) |
| Phi-3 IQ4_NL | No IQ4_NL kernel | IQ4_NL CUDA kernel (per-tensor, packed FFN blocks graph) | 1.6 → **6.5** tok/s (4x) |

## Key Observations (v1.5.1)

1. **New CUDA kernels deliver massive improvements.** Q5_0 kernel unlocks Gemma 2/3 CUDA graph (9x). IQ4_NL kernel unlocks Llama IQ4_NL graph (6.5x). QK-norm enables Qwen3 graph (1.8x).

2. **CUDA graph remains the key to peak performance.** Models with CUDA graph achieve 7–55 tok/s. Per-tensor fallback (Phi-4, Qwen3.5) achieves 6-12 tok/s.

3. **Llama-3.2-1B-Q4K peaks at 54.7 tok/s** — near-interactive speed. This is the fastest model in the test suite.

4. **MoE-optimized GPU placement enables 30B on 6 GB VRAM.** DeepSeek-Coder-V2-Lite and Qwen3-Coder-30B run with only ~500-540 MB VRAM.

5. **IQ3_XXS models use mixed quantization** — the GGUF contains IQ2_S (Q/K), IQ3_S (Wo), IQ3_XXS (FFN), Q4_K (V). Only IQ3_XXS and Q4_K have CUDA kernels. Adding IQ2_S and IQ3_S kernels would unlock full GPU acceleration.

6. **Packed FFN (Phi-3/4) blocks CUDA forward pass.** Phi-4-mini's wGate=null packed FFN prevents CUDA graph despite merged QKV support being ready. Adding packed FFN support would unlock CUDA graph for Phi-3/4.

## GPU Strategy Summary

| Strategy | When Used | VRAM Needed | Typical Speed |
|----------|-----------|-------------|---------------|
| Full offload + CUDA graph | Dense model fits in VRAM, all tensors have CUDA kernels | 770–4822 MB | 7–55 tok/s |
| Full offload + per-tensor | Model fits in VRAM but architecture not supported for graph | 770–2600 MB | 6–12 tok/s |
| MoE-optimized | MoE model, attention fits in VRAM | 517–913 MB | 0.8–1.6 tok/s |
| Partial offload | Dense/hybrid model, first-N layers on GPU | 4615–4909 MB | 0.2–1.6 tok/s |
