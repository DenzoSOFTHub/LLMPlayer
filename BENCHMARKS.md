# LLMPlayer Benchmarks

## Test Configuration

- **Hardware:** Intel Core Ultra 7 155H (22 cores) + NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM) + 31 GB RAM
- **JVM:** OpenJDK 25.0.2, SimdVectorOps (Vector API), Panama FFI mmap
- **Prompt:** `"Write a Java class that calculates factorial recursively"`
- **Parameters:** `--max-tokens 60 --context-length 512`
- **Date:** 2026-03-07

## Results v1.4.0 — CUDA GPU vs CPU (SIMD)

Tested with `--gpu --gpu-backend cuda` and `--no-gpu` respectively.

| # | Model | Params | Quant | CUDA tok/s | CPU tok/s | Speedup | GPU Mode |
|--:|-------|--------|-------|----------:|----------:|--------:|----------|
| 1 | Llama-3.2-1B-Instruct | 1B | Q4_K_M | **50.9** | 2.6 | **20x** | CUDA graph (16/16 layers) |
| 2 | Llama-3.2-3B-Instruct | 3B | Q4_K_M | **23.9** | 1.0 | **24x** | CUDA graph (28/28 layers) |
| 3 | Qwen2.5-Coder-3B-Instruct | 3B | Q4_K_M | **22.4** | 1.0 | **22x** | CUDA graph (36/36 layers) |
| 4 | Qwen2.5-Coder-7B-Instruct | 7B | Q4_K_M | **11.6** | — | — | CUDA graph (28/28 layers) |
| 5 | Phi-4-mini-Instruct | 3.8B | Q4_K_M | **11.3** | 0.9 | **13x** | Per-tensor CUDA matmul* |
| 6 | DeepSeek-R1-0528-Qwen3-8B | 8B | Q4_K_M | **11.1** | — | — | CUDA graph (36/36 layers) |
| 7 | Qwen3.5-4B | 4B | Q4_K_M | **7.5** | 0.8 | **9x** | Per-tensor CUDA matmul† |

\* GPU chain not supported (Phi-4 architecture); falls back to individual CUDA matmul per tensor.
† Qwen3.5 uses `Qwen35InferenceEngine` (hybrid DeltaNet); CUDA graph not applicable.

Note: CPU results are pure SIMD (Vector API) on Intel Core Ultra 7 155H (22 cores). OpenCL CPU (PoCL) yields similar speeds (~2.8 tok/s for Llama 1B). The previous benchmark table (v1.3.0) was measured on Windows native; this WSL2 environment shows lower CPU throughput.

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

1. **CUDA delivers 9–24x speedup over CPU.** All models benefit substantially from GPU acceleration. The speedup is highest for models with CUDA graph support (20-24x for Llama 1B/3B) and lower for models without graph (9-13x for Qwen3.5, Phi-4).

2. **CUDA graph is the key to peak performance.** Models with CUDA graph (Llama, Qwen2/2.5, DeepSeek-R1-Qwen3) achieve 11–51 tok/s. Models without graph support (Phi-4, Qwen3.5) still benefit from per-tensor CUDA matmul (7-11 tok/s) but miss the graph's elimination of per-kernel launch overhead.

3. **Q4_K_M is the sweet spot.** Best speed/quality ratio. IQ4_NL and IQ4_XS lack GPU kernel support and fall back to CPU, making them significantly slower despite smaller file sizes.

4. **MoE-optimized GPU placement works.** Models from 10–18 GB run with only 500–900 MB VRAM by placing all attention tensors on GPU and keeping expert tensors on CPU. This enables 30B MoE models on a 6 GB GPU.

5. **14B+ without full GPU is impractical.** Phi-4 (14B) on CPU runs at 0.4 tok/s — 30x slower than Llama-3.2-1B on GPU. Models that don't fit in VRAM need partial offload, with diminishing returns.

6. **32B dense models are borderline.** GLM-4-32B and Qwen2.5-Coder-32B only fit 15–16 of 60+ layers in GPU, yielding 0.2 tok/s. Usable for short completions but not interactive chat.

7. **CUDA graph capture/replay** delivers a major speedup. Qwen2.5-Coder-3B with CUDA graph achieves 22.4 tok/s vs 1.0 tok/s CPU — a 22× boost. Graph mode eliminates per-kernel launch overhead by replaying the entire forward pass as a single GPU operation. Requires all layers on GPU (full offload) and standard `InferenceEngine` architecture.

8. **Llama-3.2-1B with CUDA graph reaches 51 tok/s.** Combined CPU-side optimizations (reflection Method caching, quickselect sampler, sparse logits history) with CUDA graph mode deliver the fastest inference. Profiling shows 31% of peak 192 GB/s memory bandwidth utilization — the remaining gap is due to Q4_K non-coalesced memory access patterns and Panama FFM overhead (~2 ms/tok). Full analysis in `PERFORMANCE-ANALYSIS.md`.

9. **DeepSeek-R1-Qwen3-8B** achieves 11.1 tok/s with CUDA graph — the fastest 8B model, likely due to optimized Qwen3 architecture with QK-norm enabling efficient GPU execution.

10. **Qwen3.5 hybrid DeltaNet+attention** models benefit from GPU (7.5 tok/s GPU vs 0.8 tok/s CPU = 9x speedup for 4B variant). Qwen3.5 uses `Qwen35InferenceEngine` (not standard `InferenceEngine`), so CUDA graph is not used — GPU acceleration comes from per-tensor CUDA matmul only.

## GPU Strategy Summary

| Strategy | When Used | VRAM Needed | Typical Speed |
|----------|-----------|-------------|---------------|
| Full offload | Model fits in VRAM (< 6 GB) | 770–4822 MB | 3–56 tok/s |
| MoE-optimized | MoE model, attention fits in VRAM | 517–913 MB | 0.8–1.6 tok/s |
| Partial offload | Dense/hybrid model, most layers on GPU | 4615–4909 MB | 0.2–1.6 tok/s |
| CPU only (SIMD) | No GPU or unsupported quant (IQ*) | 0 | 0.3–4.1 tok/s |
