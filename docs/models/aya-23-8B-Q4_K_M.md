# aya-23-8B-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `aya-23-8B-Q4_K_M.gguf` |
| Size | 4823 MB |
| Parameters | 8B |
| Quantization | Q4_K_M |
| Tokenizer | SentencePiece |
| Chat Template | Llama 3 format (Aya uses standard Llama template) |

## Architecture

| Field | Value |
|-------|-------|
| Architecture | LLAMA (Aya-23 uses the llama architecture) |
| Inference Engine | InferenceEngine (standard) |
| Layers | 32 |
| Attention Heads | 32 |
| KV Heads | 8 |
| Embedding Dim | 4096 |
| Head Size | 128 |
| FFN Dim | 14336 |
| GQA Ratio | 32:8 (4:1) |
| RoPE | NORMAL |
| FFN Type | SwiGLU |
| Norm | Pre-norm (RMSNorm) |
| QK-Norm | No |
| Bias | No |
| Sliding Window | No |

## GPU Profile

Hardware: NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM).

| Version | tok/s | Mode | Notes |
|---------|-------|------|-------|
| v1.5.1 | 7.0 | CUDA graph (32/32 layers) | Full offload |

- Full offload: 4823 MB VRAM (fits in 6 GB)
- All layers on GPU with CUDA graph capture/replay

## CPU Profile

No CPU-only benchmark data available yet.

## Known Issues

None.

## Version History

| Version | Change |
|---------|--------|
| v1.4.0 | Llama architecture support (covers Aya models) |
| v1.5.1 | CUDA graph mode support |
