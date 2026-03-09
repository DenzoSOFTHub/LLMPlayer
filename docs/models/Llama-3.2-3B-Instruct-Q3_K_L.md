# Llama 3.2 3B Instruct Q3_K_L

## Model Info

| Field | Value |
|-------|-------|
| File | `Llama-3.2-3B-Instruct-Q3_K_L.gguf` |
| Size | 1732 MB |
| Parameters | 3B |
| Quantization | Q3_K_L (3.4 bpw, 256 elements/block, 110 bytes/block) |
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

**Inference Engine:** `InferenceEngine` (standard). Same architecture as the Q4_K_M variant; only the quantization differs.

## GPU Profile (RTX 4050 Laptop, 6140 MB VRAM)

| Version | tok/s | Mode | Notes |
|---------|-------|------|-------|
| v1.5.1 | 8.3 | CUDA graph | 28/28 layers |
| v1.6.0 | 7.9 | CUDA graph | 28/28 layers |

- **VRAM usage:** 1732 MB (full offload, all 28 layers on GPU)
- **Performance note:** Significantly slower than Q4_K_M (7.9 vs 24.4 tok/s) because the Q3_K CUDA kernel is less bandwidth-efficient. Q3_K blocks are 110 bytes (not 4-byte aligned), preventing `uint32` vectorized loads. Only byte-level `__ldg` is safe. Additionally, the 3-bit unpacking logic is more complex than Q4_K.
- **Test config:** `--prompt "..." --max-tokens 200 --context-length 256 --gpu --gpu-backend cuda`

## CPU Profile

| Version | tok/s | Notes |
|---------|-------|-------|
| v1.6.0 | 0.6 | SIMD Q3_K fused dequant+dot active |

- **Hardware:** Intel Core Ultra 7 155H

## Known Issues

None.

## Version History

| Version | Change | tok/s |
|---------|--------|-------|
| v1.5.1 | CUDA graph mode | 8.3 |
| v1.6.0 | No significant change | 7.9 |
