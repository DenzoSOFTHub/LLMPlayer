# Yi-Coder-9B-Chat-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `Yi-Coder-9B-Chat-Q4_K_M.gguf` |
| Size | 5083 MB |
| Parameters | 9B |
| Quantization | Q4_K_M |
| Tokenizer | BPE |
| Chat Template | Llama format (Yi uses llama architecture) |

## Architecture

| Field | Value |
|-------|-------|
| Architecture | LLAMA (Yi uses the llama architecture) |
| Inference Engine | InferenceEngine (standard) |
| RoPE | NORMAL |
| FFN Type | SwiGLU |
| Norm | Pre-norm (RMSNorm) |
| QK-Norm | No |

## GPU Profile

Hardware: NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM).

| Version | tok/s | Mode | Notes |
|---------|-------|------|-------|
| -- | -- | CUDA graph expected | No benchmark data available yet |

- Full offload: 5083 MB VRAM (fits in 6 GB)
- Expected to support CUDA graph mode (standard dense llama architecture)
- Performance expected in the 7-11 tok/s range based on similar-sized Llama models

## CPU Profile

No benchmark data available yet.

## Known Issues

None expected. Standard Llama architecture with well-tested code paths.

## Version History

| Version | Change |
|---------|--------|
| v1.4.0 | Llama architecture support (covers Yi models) |
| v1.5.1 | CUDA graph mode support for Llama models |
