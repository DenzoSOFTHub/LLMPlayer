# Qwen 2.5 3B Instruct Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `qwen2.5-3b-instruct-q4_k_m.gguf` |
| Size | 2008 MB |
| Parameters | 3B |
| Quantization | Q4_K_M |
| Tokenizer | SentencePiece |
| Chat Template | `<\|im_start\|>user\nPrompt<\|im_end\|>\n<\|im_start\|>assistant\n` |

## Architecture

| Parameter | Value |
|-----------|-------|
| Architecture | QWEN2 |
| Layers | 36 |
| Attention Heads | 16 |
| KV Heads | 2 |
| Embedding Dim | 2048 |
| Head Size | 128 |
| FFN Dim | 11008 |
| GQA Ratio | 8:1 (16 Q heads : 2 KV heads) |
| RoPE | NEOX, base=1000000 |
| Norm | Pre-norm (RMSNorm) |
| FFN | SwiGLU |

**Inference Engine:** `InferenceEngine` (standard). Pre-norm transformer with high GQA ratio (8:1). No QK-norm, no bias (Qwen2 3B does not have bias unlike larger Qwen2 variants). Separate Q, K, V weight matrices.

## GPU Profile (RTX 4050 Laptop, 6140 MB VRAM)

| Version | tok/s | Mode | Notes |
|---------|-------|------|-------|
| v1.5.1 | 21.7 | CUDA graph | 36/36 layers |
| v1.6.0 | 18.3 | CUDA graph | 36/36 layers |

- **VRAM usage:** 2008 MB (full offload, all 36 layers on GPU)
- **Test config:** `--prompt "..." --max-tokens 200 --context-length 256 --gpu --gpu-backend cuda`

## CPU Profile

| Version | tok/s | Notes |
|---------|-------|-------|
| v1.6.0 | 0.7 | SIMD tensors active |

- **Hardware:** Intel Core Ultra 7 155H

## Known Issues

None.

## Version History

| Version | Change | tok/s |
|---------|--------|-------|
| v1.5.1 | CUDA graph mode | 21.7 |
| v1.6.0 | Minor change | 18.3 |
