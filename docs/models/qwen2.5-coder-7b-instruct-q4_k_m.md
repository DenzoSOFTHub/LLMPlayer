# qwen2.5-coder-7b-instruct-q4_k_m

## Model Info

| Field | Value |
|-------|-------|
| File | `qwen2.5-coder-7b-instruct-q4_k_m.gguf` |
| Size | 4467 MB |
| Parameters | 7B |
| Quantization | Q4_K_M |
| Tokenizer | SentencePiece |
| Chat Template | `<\|im_start\|>user\nPrompt<\|im_end\|>\n<\|im_start\|>assistant\n` |

## Architecture

| Field | Value |
|-------|-------|
| Architecture | QWEN2 |
| Inference Engine | InferenceEngine (standard) |
| Layers | 28 |
| Attention Heads | 28 |
| KV Heads | 4 |
| Embedding Dim | 3584 |
| Head Size | 128 |
| FFN Dim | 18944 |
| GQA Ratio | 28:4 (7:1) |
| RoPE | NEOX |
| FFN Type | SwiGLU |
| Norm | Pre-norm (RMSNorm) |
| QK-Norm | No |
| Bias | No |
| Sliding Window | No |

## GPU Profile

Hardware: NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM).

| Version | tok/s | Mode | Notes |
|---------|-------|------|-------|
| v1.5.1 | 11.3 | CUDA graph (28/28 layers) | Fused gate+up enabled |
| v1.6.0 | 11.4 | CUDA graph + fused gate+up | Stable |

- Full offload: 4467 MB VRAM (fits in 6 GB)
- All layers on GPU with CUDA graph capture/replay

## CPU Profile

No CPU-only benchmark data available yet.

## Known Issues

None.

## Version History

| Version | Change |
|---------|--------|
| v1.4.0 | Qwen2 architecture support |
| v1.5.1 | CUDA graph mode, fused gate+up kernel |
| v1.6.0 | Stable performance |
