# LLMPlayer Benchmarks

## Test Configuration

- **Hardware:** Intel Core Ultra 7 155H (22 cores) + NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM) + 31 GB RAM
- **JVM:** OpenJDK 25.0.2, SimdVectorOps (Vector API), Panama FFI mmap
- **Prompt:** `"Write a Java class that calculates factorial recursively"` — `--max-tokens 60 --context-length 512`
- **GPU:** CUDA auto-detected (LLMPlayer auto-detects NVIDIA GPU and enables CUDA when available)
- **Date:** 2026-03-09

**Note on GPU auto-detection:** LLMPlayer automatically detects and enables CUDA GPU when an NVIDIA GPU is present. All benchmarks below use this default behavior. Use `--no-gpu` to force CPU-only mode (significantly slower: ~2-3 tok/s for 1B models vs ~50 tok/s with GPU).

## Results v1.6.0

**New in v1.6.0:** Sonar-OSS-20B (GPT-OSS) support with MXFP4 quantization, CUDA MXFP4 kernel, expert GPU cache with batched kernel execution, Command-R/Cohere architecture, OLMo2 architecture, fused QKV/gate+up matmul, prefill optimization (skip output projection for non-final tokens).

### Full GPU offload — CUDA graph (model fits in 6 GB VRAM)

| # | Model | Params | Quant | Size | tok/s | v1.5.1 | Change | GPU Mode |
|--:|-------|--------|-------|-----:|------:|-------:|-------:|----------|
| 1 | OLMo-2-1B-Instruct | 1B | Q4_K_M | 893M | **52.1** | 25.5 | **+104%** | CUDA graph (16/16) ★ |
| 2 | Llama-3.2-1B-Instruct | 1B | Q4_K_M | 771M | **48.8** | 54.7 | -11% | CUDA graph (16/16) |
| 3 | Qwen2.5-Coder-1.5B-Instruct | 1.5B | Q4_K_M | 1.1G | **40.8** | 41.5 | -2% | CUDA graph (28/28) |
| 4 | Gemma-3-1B-it | 1B | Q4_K_M | 769M | **33.1** | 35.7 | -7% | CUDA graph (26/26) |
| 5 | Llama-3.2-1B-Instruct | 1B | IQ4_NL | 738M | **27.8** | 28.7 | -3% | CUDA graph (16/16) |
| 6 | Llama-3.2-3B-Instruct | 3B | Q4_K_M | 1.9G | **22.6** | 23.8 | -5% | CUDA graph (28/28) |
| 7 | Llama-3.2-1B-Instruct | 1B | IQ3_XXS | 537M | **22.6** | 1.3 | **+17x** | CUDA graph (16/16) ★ |
| 8 | Qwen2.5-Coder-3B-Instruct | 3B | Q4_K_M | 2.0G | **21.5** | 21.8 | -1% | CUDA graph (36/36) |
| 9 | Qwen2.5-3B-Instruct | 3B | Q4_K_M | 2.0G | **21.4** | 21.7 | -1% | CUDA graph (36/36) |
| 10 | Qwen3-4B | 4B | Q4_K_M | 2.4G | **18.3** | 19.0 | -4% | CUDA graph (36/36) |
| 11 | Phi-4-mini-Instruct | 3.8B | Q4_K_M | 2.4G | **14.5** | 11.5 | **+26%** | CUDA graph (32/32) ★ |
| 12 | Qwen2.5-Coder-7B-Instruct | 7B | Q4_K_M | 4.4G | **11.2** | 11.3 | = | CUDA graph (28/28) |
| 13 | DeepSeek-R1-Qwen3-8B | 8B | Q4_K_M | 4.7G | **10.7** | 9.3 | **+15%** | CUDA graph (36/36) ★ |
| 14 | Phi-3-mini-4k-Instruct | 3.8B | IQ4_NL | 2.1G | **8.8** | 6.5 | **+35%** | CUDA graph ★ |
| 15 | Gemma-2-2B-it | 2B | IQ4_XS | 1.5G | **8.6** | 9.0 | -4% | CUDA graph (26/26) |
| 16 | Llama-3.2-3B-Instruct | 3B | Q3_K_L | 1.7G | **8.2** | 8.3 | = | CUDA graph (28/28) |

★ Major improvement: OLMo-2 (+104%, architecture now recognized), Llama-1B IQ3_XXS (+17x, IQ kernels now complete), Phi-4-mini (+26%, CUDA graph now supported), DeepSeek-R1-Qwen3-8B (+15%), Phi-3-mini (+35%).

### Per-tensor CUDA matmul (no CUDA graph)

| # | Model | Params | Quant | Size | tok/s | v1.5.1 | Note |
|--:|-------|--------|-------|-----:|------:|-------:|------|
| 1 | Qwen3.5-4B | 4B | Q4_K_M | 2.6G | **7.4** | 8.0 | Hybrid DeltaNet — no GPU forward pass |
| 2 | Yi-Coder-9B-Chat | 9B | Q4_K_M | 5.0G | **6.0** | — | New model |
| 3 | Qwen3.5-9B | 9B | Q4_K_M | 5.3G | **3.1** | — | Hybrid DeltaNet |
| 4 | Aya-23-8B | 8B | Q4_K_M | 4.8G | **1.1** | 7.0 | Command-R arch, no GPU forward pass† |

† Aya-23-8B: in v1.5.1, detected as LLAMA → CUDA graph. Now correctly detected as COMMAND_R → per-tensor only. Adding COMMAND_R to CudaForwardPass would restore GPU graph speed.

### Partial GPU offload / MoE-optimized (model exceeds 6 GB VRAM)

| # | Model | Params | Quant | Size | tok/s | v1.5.1 | Strategy |
|--:|-------|--------|-------|-----:|------:|-------:|----------|
| 1 | Sonar-OSS-20B | 20B (MoE) | MXFP4+Q8 | 12G | **2.5** | — | MoE-optimized + expert GPU cache ★ |
| 2 | DeepSeek-V2-Lite | 16B (2.4B active) | Q4_K_M | 9.7G | **1.6** | 1.5 | MoE-optimized |
| 3 | Phi-4 | 14B | Q4_K | 8.5G | **0.7** | 0.7 | First-N-layers (22/40) |
| 4 | GLM-4.7-Flash | 17B (MoE) | Q4_K_M | 18G | **0.7** | — | MoE-optimized |
| 5 | Devstral-24B | 24B | Q4_K_M | 14G | **0.3** | — | Partial offload |
| 6 | Aya-23-8B | 8B | IQ3_XXS | 3.2G | **0.3** | 0.2 | Mixed (no IQ2_S/IQ3_S kernel) |

★ Sonar-OSS-20B: new MXFP4 CUDA kernel + expert GPU cache (LRU, batched kernel execution with 2 sync points per MoE layer). MoE-optimized placement: attention on GPU (~654 MB VRAM), expert tensors on CPU.

## Key Observations (v1.6.0)

1. **27 models tested, 25 produce output.** Timeouts for Qwen3-Coder-30B-A3B (18 GB, needs more time).

2. **Major improvements for several models.** OLMo-2 +104% (architecture recognition), Llama-1B IQ3_XXS +17x (complete IQ kernel set), Phi-4-mini +26%, Phi-3-mini +35%, DeepSeek-R1-Qwen3-8B +15%.

3. **Sonar-OSS-20B (GPT-OSS) now supported.** MXFP4 quantization with custom CUDA kernel. Expert GPU cache reduces sync overhead from 12 to 2 per MoE layer via batched kernel execution.

4. **CUDA graph remains the key to peak performance.** Models with CUDA graph achieve 8–52 tok/s. Per-tensor fallback achieves 1–7 tok/s.

5. **Command-R/Cohere architecture properly recognized.** Aya-23-8B now uses COMMAND_R instead of generic LLAMA. This blocks CUDA forward pass but improves correctness. Adding COMMAND_R support to CudaForwardPass would restore GPU graph speed.

## GPU Strategy Summary

| Strategy | When Used | VRAM Needed | Typical Speed |
|----------|-----------|-------------|---------------|
| Full offload + CUDA graph | Dense model fits in VRAM, supported architecture | 770–4794 MB | 8–52 tok/s |
| Full offload + per-tensor | Model fits in VRAM but architecture not supported for graph | 770–5000 MB | 1–7 tok/s |
| MoE-optimized + expert cache | MoE model, attention fits in VRAM | 517–913 MB | 0.7–2.5 tok/s |
| Partial offload | Dense/hybrid model, first-N layers on GPU | 4615–4909 MB | 0.3–0.7 tok/s |

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
