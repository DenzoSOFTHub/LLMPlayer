# LLMPlayer Benchmarks

## Test Configuration

- **Hardware:** Intel Core Ultra 7 155H (22 cores) + NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM) + 31 GB RAM
- **JVM:** OpenJDK 25.0.2, SimdVectorOps (Vector API), Panama FFI mmap
- **Prompt:** `"Write a Java class that calculates factorial recursively"` — `--max-tokens 60 --context-length 512`
- **GPU:** CUDA auto-detected (LLMPlayer auto-detects NVIDIA GPU and enables CUDA when available)
- **Date:** 2026-03-21 (updated)

**Note on GPU auto-detection:** LLMPlayer automatically detects and enables CUDA GPU when an NVIDIA GPU is present. GPU benchmarks below use this default behavior. CPU benchmarks use `--no-gpu` to force CPU-only mode.

## Results v1.8.0

**New in this version:** Qwen3.5 CUDA graph forward pass (`Qwen35CudaForwardPass`), fused DeltaNet recurrence kernel (`deltanet_fused.cu`), embedding on CPU for all architectures (frees ~500+ MB VRAM), improved VRAM budget estimation, cuBLAS support (opt-in via `-Dcuda.cublas=true`), dp4a integer dot product (opt-in via `-Dcuda.dp4a=true`).

## Results v1.6.0

**New in v1.6.0:** Sonar-OSS-20B (GPT-OSS) support with MXFP4 quantization, CUDA MXFP4 kernel, expert GPU cache with batched kernel execution, Command-R/Cohere architecture, OLMo2 architecture, fused QKV/gate+up matmul, prefill optimization (skip output projection for non-final tokens).

### Full GPU offload — CUDA graph (model fits in 6 GB VRAM)

Benchmarked 2026-03-21 with v1.8.0 (embedding on CPU, improved VRAM budget, Qwen35CudaForwardPass).

| # | Model | Params | Quant | Size | tok/s | v1.6.0 | GPU Mode |
|--:|-------|--------|-------|-----:|------:|-------:|----------|
| 1 | Llama-3.2-1B-Instruct | 1B | Q4_K_M | 771M | **47.7** | 48.8 | CUDA graph (16/16) |
| 2 | OLMo-2-1B-Instruct | 1B | Q4_K_M | 893M | **46.8** | 52.1 | CUDA graph (16/16) |
| 3 | Qwen2.5-Coder-1.5B-Instruct | 1.5B | Q4_K_M | 1.1G | **37.3** | 40.8 | CUDA graph (28/28) |
| 4 | Gemma-3-1B-it | 1B | Q4_K_M | 769M | **31.6** | 33.1 | CUDA graph (26/26) |
| 5 | Qwen3.5-2B-Claude-4.6 | 2B | Q4_K_M | 1.2G | **26.2** | — | CUDA graph (24/24) ★ |
| 6 | SmolLM3-3B | 3B | Q4_K_M | 1.8G | **21.4** | 22.9 | CUDA graph (36/36) |
| 7 | Llama-3.2-3B-Instruct | 3B | Q4_K_M | 1.9G | **21.3** | 22.6 | CUDA graph (28/28) |
| 8 | Qwen2.5-Coder-3B-Instruct | 3B | Q4_K_M | 2.0G | **19.9** | 21.5 | CUDA graph (36/36) |
| 9 | Gemma-3-4B-it | 4B | Q4_K_M | 2.3G | **16.3** | — | CUDA graph (34/34) ★ |
| 10 | Qwen3-4B | 4B | Q4_K_M | 2.4G | **18.5** | 18.3 | CUDA graph (36/36) |
| 10 | Phi-4-mini-Instruct | 3.8B | Q4_K_M | 2.4G | **13.7** | 14.5 | CUDA graph (32/32) |
| 11 | Qwen3.5-4B-Claude-4.6 | 4B | Q4_K_M | 2.5G | **12.3** | 8.7† | **+41%** | CUDA graph (32/32) ★ |
| 12 | Qwen3.5-4B | 4B | Q4_K_M | 2.6G | **11.3** | 7.4† | **+53%** | CUDA graph (32/32) ★ |
| 13 | Qwen2.5-Coder-7B-Instruct | 7B | Q4_K_M | 4.4G | **10.8** | 11.2 | CUDA graph (28/28) |
| 14 | DeepSeek-R1-Qwen3-8B | 8B | Q4_K_M | 4.7G | **10.2** | 10.7 | CUDA graph (36/36) |
| 15 | Mistral-7B-Instruct-v0.3 | 7B | Q4_K_M | 4.1G | **11.8** | — | CUDA graph (32/32) ★ |
| 16 | Llama-3.1-8B-Instruct | 8B | Q4_K_M | 4.6G | **10.0** | — | CUDA graph (32/32) ★ |
| 17 | Yi-Coder-9B-Chat | 9B | Q4_K_M | 5.0G | **9.2** | 6.0† | **+53%** | CUDA graph (48/48) ★ |
| 18 | Qwen3.5-9B-Claude-4.6 | 9B | Q4_K_M | 5.2G | **7.0** | 4.5† | **+56%** | CUDA graph (32/32) ★ |
| 19 | Qwen3.5-9B | 9B | Q4_K_M | 5.3G | **6.5** | 3.1† | **+110%** | CUDA graph (32/32) ★ |

### GPU-resident forward pass — per-layer (no CUDA graph)

| # | Model | Params | Quant | Size | tok/s | GPU Mode |
|--:|-------|--------|-------|-----:|------:|----------|
| 1 | NVIDIA-Nemotron-3-Nano-4B | 4B (hybrid) | Q4_K_M | 2.7G | **9.9** | Per-layer (42/42) ★ |

★ New architecture: Nemotron-H hybrid Mamba-2 + Attention + squared-ReLU FFN. GPU-resident forward pass via `NemotronHCudaForwardPass` with dedicated Mamba-2 scan kernel. CUDA graph not yet supported (DtoD copies in Mamba layers).

★ New models or major improvements. † Previously per-tensor only (no CUDA graph / partial GPU offload).

Key changes vs v1.6.0:
- **Qwen3.5 CUDA graph**: new `Qwen35CudaForwardPass` with fused DeltaNet recurrence kernel. Qwen3.5-4B: +41-53%, Qwen3.5-9B: +56-110%.
- **Qwen3.5-9B now fits in 6 GB VRAM**: embedding on CPU frees ~500 MB, enabling 32/32 layer offload (was 29/32).
- **Yi-Coder-9B +53%**: same VRAM optimization moved from 29→48/48 layers + graph.
- **Llama-3.1-8B-Instruct**: new model, 10.0 tok/s with full GPU offload.
- Small regressions (1-8%) on some models due to VRAM budget recalculation changing memory allocation patterns.

### Per-tensor CUDA matmul (no CUDA graph)

| # | Model | Params | Quant | Size | tok/s | v1.5.1 | Note |
|--:|-------|--------|-------|-----:|------:|-------:|------|
| 1 | Yi-Coder-9B-Chat | 9B | Q4_K_M | 5.0G | **6.0** | — | New model |
| 2 | Aya-23-8B | 8B | Q4_K_M | 4.8G | **1.1** | 7.0 | Command-R arch, no GPU forward pass† |
| 3 | Granite-3.3-8B-Instruct | 8B | Q4_K_M | 4.6G | **1.0** | — | Granite arch, per-tensor GPU (no chain yet) ★ |

★ Granite 3.3 uses 4 unique scaling factors (embedding_scale=12, attention_scale=1/128, residual_scale=0.22, logit_scale=1/16). Output quality verified. CudaForwardPass GPU chain not yet compatible — needs dedicated GraniteCudaForwardPass with scaling kernels integrated.

† Aya-23-8B: in v1.5.1, detected as LLAMA → CUDA graph. Now correctly detected as COMMAND_R → per-tensor only. Adding COMMAND_R to CudaForwardPass would restore GPU graph speed.

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
| 1 | Llama-3.2-1B-Instruct | 1B | Q4_K_M | 771M | 4.1 | **48.8** | **12x** |
| 2 | Gemma-3-1B-it | 1B | Q4_K_M | 769M | 3.9 | **33.1** | **8x** |
| 3 | Llama-3.2-1B-Instruct | 1B | IQ4_NL | 738M | 3.5 | **27.8** | **8x** |
| 4 | OLMo-2-1B-Instruct | 1B | Q4_K_M | 893M | 3.2 | **52.1** | **16x** |
| 5 | Qwen2.5-Coder-1.5B-Instruct | 1.5B | Q4_K_M | 1.1G | 2.6 | **40.8** | **16x** |
| 6 | Gemma-2-2B-it | 2B | IQ4_XS | 1.5G | 1.9 | **8.6** | **5x** |
| 7 | Llama-3.2-1B-Instruct | 1B | IQ3_XXS | 537M | 1.4 | **22.6** | **16x** |
| 8 | Llama-3.2-3B-Instruct | 3B | Q4_K_M | 1.9G | 1.3 | **22.6** | **17x** |
| 9 | SmolLM3-3B | 3B | Q4_K_M | 1.8G | 1.2 | **22.9** | **19x** |
| 10 | Qwen2.5-Coder-3B-Instruct | 3B | Q4_K_M | 2.0G | 1.3 | **21.5** | **17x** |
| 11 | Qwen2.5-3B-Instruct | 3B | Q4_K_M | 2.0G | 1.3 | **21.4** | **16x** |
| 12 | Phi-3-mini-4k-Instruct | 3.8B | IQ4_NL | 2.1G | 1.3 | **8.8** | **7x** |
| 13 | DeepSeek-V2-Lite | 16B (2.4B active) | Q4_K_M | 9.7G | 1.3 | **1.6** | 1.2x |
| 14 | Sonar-OSS-20B | 20B (MoE) | MXFP4+Q8 | 12G | 1.2 | **2.5** | 2.1x |
| 15 | Llama-3.2-3B-Instruct | 3B | Q3_K_L | 1.7G | 1.1 | **8.2** | **7x** |
| 16 | Qwen3-4B | 4B | Q4_K_M | 2.4G | 1.1 | **18.3** | **17x** |
| 17 | Phi-4-mini-Instruct | 3.8B | Q4_K_M | 2.4G | 1.1 | **14.5** | **13x** |
| 18 | Qwen3.5-4B | 4B | Q4_K_M | 2.6G | 0.9 | **12.8** | **14x** |
| 19 | Qwen3.5-4B-Claude-4.6-Opus-Reasoning | 4B | Q4_K_M | 2.5G | 0.9 | **13.8** | **15x** |
| 20 | GLM-4.7-Flash | 17B (MoE) | Q4_K_M | 18G | 0.7 | **0.7** | 1.0x |
| 21 | Qwen2.5-Coder-7B-Instruct | 7B | Q4_K_M | 4.4G | 0.6 | **11.2** | **19x** |
| 22 | DeepSeek-R1-Qwen3-8B | 8B | Q4_K_M | 4.7G | 0.6 | **10.7** | **18x** |
| 23 | Aya-23-8B | 8B | Q4_K_M | 4.8G | 0.6 | **1.1** | 1.8x |
| 24 | Qwen3.5-9B | 9B | Q4_K_M | 5.3G | 0.5 | **3.1** | **6x** |
| 25 | Qwen3.5-9B-Claude-4.6-Opus-Reasoning | 9B | Q4_K_M | 5.2G | 0.6 | **4.5** | **8x** |
| 26 | Phi-4 | 14B | Q4_K | 8.5G | 0.3 | **0.7** | 2.3x |
| 27 | Devstral-24B | 24B | Q4_K_M | 14G | TIMEOUT | **0.3** | — |
| 28 | Aya-23-8B | 8B | IQ3_XXS | 3.2G | 0.2 | **0.3** | 1.5x |
| 29 | Yi-Coder-9B-Chat | 9B | Q4_K_M | 5.0G | TIMEOUT | **6.0** | — |

**Key takeaway:** GPU acceleration provides **5–19x speedup** for models that fit in VRAM with CUDA graph. The benefit is greatest for 3B–8B models (16–19x). MoE models with partial GPU offload see modest 1–2x improvement due to CPU-bound expert computation. CPU-only achieves 1–4 tok/s for 1B models, <1 tok/s for 7B+.

## Key Observations (v1.6.0)

1. **32+ models tested across 16 architectures** including Nemotron-H (Mamba-2 hybrid).

2. **GPU provides 5–19x speedup over CPU-only.** CUDA graph models achieve 8–52 tok/s vs 0.6–4.1 tok/s on CPU. Largest speedup on 3B–8B models (16–19x). CPU-only is viable only for 1B models (~4 tok/s).

3. **Major improvements for several models.** SmolLM3-3B (new architecture with NoPE support, 22.9 tok/s CUDA graph), OLMo-2 +104% (architecture recognition), Llama-1B IQ3_XXS +17x (complete IQ kernel set), Phi-4-mini +26%, Phi-3-mini +35%, DeepSeek-R1-Qwen3-8B +15%.

4. **Sonar-OSS-20B (GPT-OSS) now supported.** MXFP4 quantization with custom CUDA kernel. Expert GPU cache reduces sync overhead from 12 to 2 per MoE layer via batched kernel execution.

5. **CUDA graph remains the key to peak performance.** Models with CUDA graph achieve 8–52 tok/s. Per-tensor fallback achieves 1–7 tok/s.

6. **Command-R/Cohere architecture properly recognized.** Aya-23-8B now uses COMMAND_R instead of generic LLAMA. This blocks CUDA forward pass but improves correctness. Adding COMMAND_R support to CudaForwardPass would restore GPU graph speed.

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
| Phi-3/4 (packed FFN) | CUDA forward pass partially supported | Graph mode works for attention, per-tensor for FFN |
| IQ3_XXS GGUF models | Mixed quant types (IQ2_S for Q/K, IQ3_S for Wo) — no CUDA kernels | Per-tensor CUDA for IQ3_XXS tensors only |
| Qwen3.5 (DeltaNet) | Hybrid architecture — no CUDA forward pass | Per-tensor CUDA matmul |
| 32B+ models on 8 GB RAM | Timeout due to excessive swap | Requires 16+ GB RAM |
