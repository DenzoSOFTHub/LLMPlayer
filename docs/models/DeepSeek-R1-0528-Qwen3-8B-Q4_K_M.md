# DeepSeek-R1-0528-Qwen3-8B-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf` |
| Size | 4795 MB |
| Parameters | 8B |
| Quantization | Q4_K_M |
| Tokenizer | SentencePiece |
| Chat Template | `<\|im_start\|>user\nPrompt<\|im_end\|>\n<\|im_start\|>assistant\n` |

Despite the "DeepSeek-R1" name, this model uses the **Qwen3 dense architecture** (not DeepSeek2 MLA/MoE). It is a distilled model fine-tuned from Qwen3-8B.

## Architecture

| Field | Value |
|-------|-------|
| Architecture | QWEN3 |
| Inference Engine | InferenceEngine (standard) |
| Layers | 36 |
| Attention Heads | 32 |
| KV Heads | 8 |
| GQA Ratio | 32:8 (4:1) |
| RoPE | NEOX |
| FFN Type | SwiGLU |
| Norm | Pre-norm (RMSNorm) |
| QK-Norm | Yes (per-head RMSNorm, Qwen3 feature) |
| Bias | No |
| Sliding Window | No |

## GPU Profile

Hardware: NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM).

| Version | tok/s | Mode | Notes |
|---------|-------|------|-------|
| v1.4.0 | 7.3 | Per-tensor CUDA | No QK-norm CUDA support |
| v1.5.1 | 9.3 | CUDA graph (36/36 layers) | QK-norm via `rmsnorm_per_head.cu` kernel |
| v1.6.0 | 10.9 | CUDA graph + fused gate+up | **+17% over v1.5.1** |

- Full offload: 4795 MB VRAM (fits in 6 GB)
- All 36 layers on GPU with CUDA graph capture/replay
- v1.5.1 to v1.6.0 improvement driven by fused gate+up kernel for Q4_K FFN layers

## CPU Profile

No CPU-only benchmark data available yet.

## Known Issues

None.

## Version History

| Version | Change |
|---------|--------|
| v1.4.0 | Initial support via Qwen3 architecture; QK-norm on CPU |
| v1.5.1 | QK-norm CUDA kernel; CUDA graph mode |
| v1.6.0 | Fused gate+up kernel; +17% throughput improvement |
