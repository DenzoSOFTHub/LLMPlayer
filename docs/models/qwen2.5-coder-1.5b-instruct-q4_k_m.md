# Qwen 2.5 Coder 1.5B Instruct Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` |
| Size | 1066 MB |
| Parameters | 1.5B |
| Quantization | Q4_K_M |
| Tokenizer | SentencePiece |
| Chat Template | `<\|im_start\|>user\nPrompt<\|im_end\|>\n<\|im_start\|>assistant\n` |

## Architecture

| Parameter | Value |
|-----------|-------|
| Architecture | QWEN2 |
| Layers | 28 |
| Attention Heads | 12 |
| KV Heads | 2 |
| Embedding Dim | 1536 |
| Head Size | 128 |
| FFN Dim | 8960 |
| GQA Ratio | 6:1 (12 Q heads : 2 KV heads) |
| RoPE | NEOX |
| Norm | Pre-norm (RMSNorm) |
| FFN | SwiGLU |

**Inference Engine:** `InferenceEngine` (standard). Smallest Qwen2 variant tested. High GQA ratio (6:1) reduces KV cache memory. Separate Q, K, V weight matrices. No QK-norm, no bias.

## GPU Profile (RTX 4050 Laptop, 6140 MB VRAM)

| Version | tok/s | Mode | Notes |
|---------|-------|------|-------|
| v1.5.1 | 41.5 | CUDA graph | 28/28 layers, fused gate+up |
| v1.6.0 | 29.5-41.0 | CUDA graph | 28/28 layers, fused gate+up |

- **VRAM usage:** 1066 MB (full offload, all 28 layers on GPU)
- **Fastest model tested** due to small size fitting entirely in GPU memory with headroom
- **Features:** CUDA graph mode, fused gate+up kernel (`matmul_q4_k_fused_gate_up.cu`), GPU-side argmax
- **Test config:** `--prompt "..." --max-tokens 200 --context-length 256 --gpu --gpu-backend cuda`

## CPU Profile

| Version | tok/s | Notes |
|---------|-------|-------|
| v1.6.0 | 1.2-1.8 | SIMD tensors active |

- **Hardware:** Intel Core Ultra 7 155H

## Known Issues

None.

## Version History

| Version | Change | tok/s |
|---------|--------|-------|
| v1.5.1 | CUDA graph + fused gate+up | 41.5 |
| v1.6.0 | Variable performance observed | 29.5-41.0 |
