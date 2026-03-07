# LLMPlayer Benchmarks

## Test Configuration

- **Hardware:** Intel Core Ultra 7 155H (22 cores) + NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM) + 31 GB RAM
- **JVM:** OpenJDK 25.0.2, SimdVectorOps (Vector API), Panama FFI mmap
- **Prompt:** `"Write a Java class that calculates factorial recursively"`
- **Parameters:** `--max-tokens 60 --context-length 512`
- **Date:** 2026-03-07

## Results v1.4.0 — GPU vs CPU Comparison

| # | Model | Params | Quant | GPU tok/s | CPU tok/s | GPU Mode |
|--:|-------|--------|-------|----------:|----------:|----------|
| 1 | Llama-3.2-1B-Instruct | 1B | Q4_K_M | **52.2** | 53.1 | CUDA graph (16/16 layers) |
| 2 | Qwen2.5-Coder-3B-Instruct | 3B | Q4_K_M | **22.9** | 22.7 | CUDA graph (36/36 layers) |
| 3 | Llama-3.2-3B-Instruct | 3B | Q4_K_M | **20.8** | 20.3 | CUDA graph (28/28 layers) |
| 4 | OLMo-2-1B-Instruct | 1B | Q4_K_M | 20.3 | — | Per-tensor CUDA matmul* |
| 5 | Qwen2.5-Coder-1.5B-Instruct | 1.5B | Q4_K_M | **17.4** | — | CUDA graph (28/28 layers) |
| 6 | Qwen2.5-Coder-7B-Instruct | 7B | Q4_K_M | 5.1 | **11.6** | Per-tensor CUDA matmul |
| 7 | DeepSeek-R1-0528-Qwen3-8B | 8B | Q4_K_M | **11.0** | — | CUDA graph (36/36 layers) |
| 8 | Phi-4-mini-Instruct | 3.8B | Q4_K_M | 8.4 | **11.3** | Per-tensor CUDA matmul* |
| 9 | Aya-23-8B | 8B | Q4_K_M | **7.0** | — | GPU full offload |
| 10 | Qwen2.5-3B-Instruct | 3B | Q4_K_M | 6.1 | — | CUDA graph (36/36 layers) |
| 11 | Qwen3.5-4B | 4B | Q4_K_M | 4.8 | **6.2** | Per-tensor CUDA matmul† |
| 12 | Qwen3.5-9B | 9B | Q4_K_M | **2.3** | — | Per-tensor CUDA matmul† |

\* GPU chain not supported (OLMo2/Phi-4 architecture); falls back to individual CUDA matmul per tensor.
† Qwen3.5 uses `Qwen35InferenceEngine` (hybrid DeltaNet); CUDA graph not applicable.

### Key finding: CUDA graph is essential for GPU speedup

Models with CUDA graph (rows 1-5, 7, 10) show competitive or better GPU performance. Models without graph support (rows 6, 8, 11) are **slower on GPU** because per-tensor upload/download overhead exceeds the CUDA compute advantage. For these models, CPU SIMD is faster.

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

## Failed Models

| Model | Quant | Error |
|-------|-------|-------|
| Llama-3.2-1B-Instruct | IQ3_XXS | Segfault — requires IQ2_S support (not implemented) |
| Aya-23-8B | IQ3_XXS | Segfault — requires IQ2_S support (not implemented) |
| GLM-4.7-Flash | Q4_K_M | `Required tensor not found: blk.0.attn_q.weight` — loader bug |
| Devstral-Small-2-24B | Q4_K_M | Segfault on both GPU and CPU-only |
| Yi-Coder-9B-Chat | Q4_K_M | Segfault on GPU (46/48 layers partial offload); works on CPU at 3.5 tok/s |
| Qwen3-4B | Q4_K_M | ArrayIndexOutOfBoundsException in BPETokenizer.decodeTokenPiece (v1.4.0) |

## Key Observations

1. **CUDA graph is the key differentiator.** Models with CUDA graph support (Llama, Qwen2/2.5, DeepSeek-R1-Qwen3) achieve 11–52 tok/s on GPU. Models without graph support (OLMo2, Phi-4, Qwen3.5) can actually be **slower on GPU than CPU** because per-tensor upload/download overhead negates the CUDA compute advantage.

2. **CPU SIMD is surprisingly competitive.** Phi-4-mini achieves 11.3 tok/s on CPU vs 8.4 tok/s on GPU. Qwen2.5-Coder-7B: 11.6 CPU vs 5.1 GPU. The Intel Core Ultra 7 155H's AVX-512 SIMD with the Vector API is very efficient for matrix-vector multiply.

3. **Q4_K_M is the sweet spot.** Best speed/quality ratio. IQ4_NL and IQ4_XS lack GPU kernel support and fall back to CPU, making them significantly slower despite smaller file sizes.

4. **MoE-optimized GPU placement works.** Models from 10–18 GB run with only 500–900 MB VRAM by placing all attention tensors on GPU and keeping expert tensors on CPU. This enables 30B MoE models on a 6 GB GPU.

5. **14B+ without full GPU is impractical.** Phi-4 (14B) on CPU runs at 0.4 tok/s — 30x slower than Llama-3.2-1B on GPU. Models that don't fit in VRAM need partial offload, with diminishing returns.

6. **32B dense models are borderline.** GLM-4-32B and Qwen2.5-Coder-32B only fit 15–16 of 60+ layers in GPU, yielding 0.2 tok/s. Usable for short completions but not interactive chat.

7. **CUDA graph capture/replay** delivers a major speedup. Qwen2.5-Coder-3B with CUDA graph achieves 22.9 tok/s vs 6.1 tok/s for the same-size Qwen2.5-3B — a 3.8× boost. Graph mode eliminates per-kernel launch overhead by replaying the entire forward pass as a single GPU operation. Requires all layers on GPU (full offload) and standard `InferenceEngine` architecture.

8. **Llama-3.2-1B with CUDA graph reaches 52 tok/s.** Combined CPU-side optimizations (reflection Method caching, quickselect sampler, sparse logits history) with CUDA graph mode deliver the fastest inference. Profiling shows 31% of peak 192 GB/s memory bandwidth utilization — the remaining gap is due to Q4_K non-coalesced memory access patterns and Panama FFM overhead (~2 ms/tok). Full analysis in `PERFORMANCE-ANALYSIS.md`.

9. **DeepSeek-R1-Qwen3-8B** achieves 11.0 tok/s with CUDA graph — the fastest 8B model, likely due to optimized Qwen3 architecture with QK-norm enabling efficient GPU execution.

10. **Qwen3.5 hybrid DeltaNet+attention** models work but are slower on GPU. The 4B variant: 4.8 tok/s GPU vs 6.2 tok/s CPU. The 9B variant: 2.3 tok/s GPU. Qwen3.5 uses `Qwen35InferenceEngine` (not standard `InferenceEngine`), so CUDA graph is not used — GPU acceleration comes from per-tensor CUDA matmul only, which introduces more overhead than it saves for models under ~8B.

## GPU Strategy Summary

| Strategy | When Used | VRAM Needed | Typical Speed |
|----------|-----------|-------------|---------------|
| Full offload | Model fits in VRAM (< 6 GB) | 770–4822 MB | 3–56 tok/s |
| MoE-optimized | MoE model, attention fits in VRAM | 517–913 MB | 0.8–1.6 tok/s |
| Partial offload | Dense/hybrid model, most layers on GPU | 4615–4909 MB | 0.2–1.6 tok/s |
| CPU only (SIMD) | No GPU or unsupported quant (IQ*) | 0 | 0.3–4.1 tok/s |
