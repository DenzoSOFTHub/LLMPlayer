# Qwen3.5-9B-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `Qwen3.5-9B-Q4_K_M.gguf` |
| Size | 5418 MB |
| Parameters | 9B |
| Quantization | Q4_K_M |
| Tokenizer | SentencePiece |
| Chat Template | `<\|im_start\|>user\nPrompt<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n\n</think>\n\n` |

## Architecture

Same hybrid DeltaNet + Attention architecture as Qwen3.5-4B. See [Qwen3.5-4B-Q4_K_M.md](Qwen3.5-4B-Q4_K_M.md) for full architectural details.

| Field | Value |
|-------|-------|
| Architecture | QWEN35 (hybrid DeltaNet + Attention) |
| Inference Engine | Qwen35InferenceEngine |
| Full Attention Interval | 4 (every 4th layer) |
| DeltaNet Layers | 75% |
| Full Attention Layers | 25% |
| FFN Type | SwiGLU |
| QK-Norm | Yes (per-head) |

## GPU Profile

Hardware: NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM).

| Version | tok/s | Mode | Notes |
|---------|-------|------|-------|
| -- | -- | Per-tensor CUDA matmul | No benchmark data available yet |

- Full offload: 5418 MB VRAM (fits in 6 GB)
- CudaForwardPass / CUDA graph mode NOT supported (hybrid architecture)
- Per-tensor CUDA matmul only

## CPU Profile

No benchmark data available yet.

## Known Issues

Same as [Qwen3.5-4B-Q4_K_M](Qwen3.5-4B-Q4_K_M.md):
- No CUDA forward pass support (DeltaNet recurrence incompatible)
- llama.cpp eval callback breaks recurrent models
- Packed Q+gate deinterleaving complexity in full attention layers

## Version History

| Version | Change |
|---------|--------|
| v1.5.0 | Initial Qwen3.5 support |
