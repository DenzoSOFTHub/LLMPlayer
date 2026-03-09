# Llama 3.2 3B Instruct Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `llama-3.2-3b-instruct-q4_k_m.gguf` |
| Size | 1926 MB |
| Parameters | 3B |
| Quantization | Q4_K_M (4.5 bpw, 256 elements/block, 144 bytes/block) |
| Tokenizer | BPE |
| Chat Template | Llama 3 format (`<\|start_header_id\|>user<\|end_header_id\|>`) |

## Architecture

| Parameter | Value |
|-----------|-------|
| Architecture | LLAMA |
| Layers | 28 |
| Attention Heads | 24 |
| KV Heads | 8 |
| Embedding Dim | 3072 |
| Head Size | 128 |
| FFN Dim | 8192 |
| GQA Ratio | 3:1 (24 Q heads : 8 KV heads) |
| RoPE | NORMAL, base=500000 |
| Norm | Pre-norm (RMSNorm) |
| FFN | SwiGLU |

**Inference Engine:** `InferenceEngine` (standard). Pre-norm transformer with grouped-query attention and SwiGLU feed-forward network. No QK-norm, no bias.

## GPU Profile (RTX 4050 Laptop, 6140 MB VRAM)

| Version | tok/s | Mode | Notes |
|---------|-------|------|-------|
| v1.5.1 | 23.8 | CUDA graph | 28/28 layers, fused gate+up |
| v1.6.0 | 24.4 | CUDA graph | 28/28 layers, fused gate+up |

- **VRAM usage:** 1926 MB (full offload, all 28 layers on GPU)
- **Features:** CUDA graph mode, fused gate+up kernel (`matmul_q4_k_fused_gate_up.cu`), GPU-side argmax
- **Test config:** `--prompt "..." --max-tokens 200 --context-length 256 --gpu --gpu-backend cuda`

## CPU Profile

| Version | tok/s | Notes |
|---------|-------|-------|
| v1.6.0 | 0.7-0.8 | SIMD Vector API active |

- **SIMD kernels:** Q4_K, Q8_0, Q6_K, Q5_0, Q5_K, Q3_K fused dequant+dot
- **Hardware:** Intel Core Ultra 7 155H

## Known Issues

None.

## Version History

| Version | Change | tok/s |
|---------|--------|-------|
| v1.5.1 | CUDA graph + fused gate+up | 23.8 |
| v1.6.0 | Minor optimizations | 24.4 |
