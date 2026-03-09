# Qwen3-4B-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `Qwen3-4B-Q4_K_M.gguf` |
| Size | 2382 MB |
| Parameters | 4B |
| Quantization | Q4_K_M |
| Tokenizer | SentencePiece |
| Chat Template | `<\|im_start\|>user\nPrompt<\|im_end\|>\n<\|im_start\|>assistant\n` |

## Architecture

| Field | Value |
|-------|-------|
| Architecture | QWEN3 |
| Inference Engine | InferenceEngine (standard) |
| Layers | 36 |
| Attention Heads | 32 |
| KV Heads | 8 |
| Embedding Dim | 2560 |
| Head Size | 128 |
| FFN Dim | 9216 |
| GQA Ratio | 32:8 (4:1) |
| RoPE | NEOX, base=1000000 |
| FFN Type | SwiGLU |
| Norm | Pre-norm (RMSNorm) |
| QK-Norm | Yes (per-head RMSNorm on each Q/K head independently) |
| Bias | No |
| Sliding Window | No |

Key architectural difference from Qwen2: per-head QK-norm applies RMSNorm to each Q and K head independently before attention scoring.

## GPU Profile

Hardware: NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM, 192 GB/s peak bandwidth).

| Version | tok/s | Mode | Notes |
|---------|-------|------|-------|
| v1.4.0 | 10.6 | Per-tensor CUDA | No QK-norm CUDA support; per-tensor fallback for norm heads |
| v1.5.1 | 19.0 | CUDA graph (36/36 layers) | QK-norm via `rmsnorm_per_head.cu` kernel |
| v1.6.0 | 16.3-19.1 | CUDA graph + fused gate+up | Fused gate+up kernel enabled |

- Full offload: 2382 MB VRAM (fits comfortably in 6 GB)
- **+80% improvement** from v1.4.0 to v1.5.1, driven by QK-norm CUDA kernel

## CPU Profile

| Version | tok/s | Notes |
|---------|-------|-------|
| v1.6.0 | 0.6-0.8 | SIMD tensors + SIMD QK-norm (scaleWeighted) |

## Known Issues

None.

## Version History

| Version | Change |
|---------|--------|
| v1.4.0 | Initial Qwen3 support; QK-norm ran on CPU even with GPU offload |
| v1.5.1 | Added `rmsnorm_per_head.cu` kernel for GPU QK-norm; CUDA graph support |
| v1.6.0 | Fused gate+up kernel for Q4_K FFN layers |
