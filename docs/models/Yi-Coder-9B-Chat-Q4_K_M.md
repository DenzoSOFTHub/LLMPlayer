# Yi-Coder-9B-Chat-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `Yi-Coder-9B-Chat-Q4_K_M.gguf` |
| Size | 5083 MB |
| Parameters | 9B |
| Quantization | Q4_K_M |
| Tokenizer | BPE |
| Chat Template | ChatML (`<|im_start|>user`), detected from the GGUF template since v1.20.0; earlier versions used the Llama 3 headers, which do not match the model |

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

## CPU Profile (2026-10-07)

Shared 8-vCPU VirtualBox VM with other heavy workloads running, so the speed is indicative only.
`--no-gpu`, greedy: a factual question (22 prompt tokens) answered correctly ("The capital of France
is Paris."), PPL 1.01, natural EOS, 0.6 tok/s.

## Known Issues

None expected. Standard Llama architecture with well-tested code paths.

## Version History

| Version | Change |
|---------|--------|
| v1.4.0 | Llama architecture support (covers Yi models) |
| v1.5.1 | CUDA graph mode support for Llama models |
| v1.20.0 | ChatML template detected from the GGUF instead of the Llama 3 headers |
