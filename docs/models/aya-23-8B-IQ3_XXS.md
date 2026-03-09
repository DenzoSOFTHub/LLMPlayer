# aya-23-8B-IQ3_XXS

## Model Info

| Field | Value |
|-------|-------|
| File | `aya-23-8B-IQ3_XXS.gguf` |
| Size | 3254 MB |
| Parameters | 8B |
| Quantization | IQ3_XXS |
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

Same architecture as [aya-23-8B-Q4_K_M](aya-23-8B-Q4_K_M.md) but with IQ3_XXS quantization for smaller file size.

## GPU Profile

Hardware: NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM).

| Version | tok/s | Mode | Notes |
|---------|-------|------|-------|
| v1.5.1 | 0.2 | Per-tensor CUDA | Mixed quant; most layers IQ3_XXS without efficient CUDA kernel |

- Full offload: 3254 MB VRAM
- Extremely slow GPU performance due to lack of optimized IQ3_XXS CUDA kernels
- Falls back to per-tensor matmul without graph capture for mixed quantization layers

## CPU Profile

No benchmark data available yet.

## Known Issues

- **Extremely slow GPU inference.** IQ3_XXS lacks an optimized CUDA matmul kernel. Per-tensor CUDA fallback yields only 0.2 tok/s, which is slower than CPU-only for this model.
- **IQ3_XXS kernel alignment.** Same alignment constraints as observed with other IQ3_XXS models (e.g., Llama 1B IQ3_XXS).

## Version History

| Version | Change |
|---------|--------|
| v1.4.0 | IQ3_XXS quantization support |
| v1.5.1 | Per-tensor CUDA matmul (limited benefit for IQ3_XXS) |
