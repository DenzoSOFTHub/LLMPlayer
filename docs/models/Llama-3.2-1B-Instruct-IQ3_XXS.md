# Llama-3.2-1B-Instruct IQ3_XXS

## Model Summary

| Field | Value |
|-------|-------|
| File | `Llama-3.2-1B-Instruct.IQ3_XXS.gguf` |
| Size | 537 MB |
| Architecture | LLAMA |
| Parameters | 1B |
| Quantization | IQ3_XXS (3.06 bpw) |
| Layers | 16 |
| Heads | 32 Q / 8 KV (GQA 4:1) |
| Embedding dim | 2048 |
| Head size | 64 |
| FFN dim | 8192 |

## Architecture Description

### Forward Pass

Standard pre-norm transformer (InferenceEngine):

```
RMSNorm -> Attention -> Residual -> RMSNorm -> SwiGLU FFN -> Residual
```

### Attention

- Grouped Query Attention (GQA): 32 query heads, 8 key-value heads (4:1 ratio)
- RoPE positional encoding: type NORMAL (consecutive pairs), base frequency 500000
- No QK-norm, no bias, no sliding window

### FFN

- SwiGLU activation
- FFN intermediate dimension: 8192

### Normalization

- Pre-norm only (RMSNorm)

### Tokenizer

- BPE (GPT-2 byte-level encoding)
- Llama 3 pre-tokenization regex

### Chat Template

```
<|start_header_id|>user<|end_header_id|>

Prompt<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

## Quantization Details

IQ3_XXS (Importance-weighted 3-bit, extra-extra-small):

- 256 elements per block
- 98 bytes per block (NOT 4-byte aligned -- this causes CUDA issues)
- 3.06 bits per weight average
- Uses grid lookup tables and sign indices for dequantization
- Mixed quantization in the model: IQ3_XXS for most tensors, Q6_K for the output projection

## GPU Execution Profile

**Hardware:** NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM)

| Metric | Value |
|--------|-------|
| Mode | CudaForwardPass + CUDA graph |
| VRAM | 537 MB (full offload, 16/16 layers) |
| v1.5.1 | 1.3 tok/s (per-tensor) |
| v1.6.0 | Crashed (CUDA error 716) |
| v1.6.0+ | **20.2 tok/s** (CUDA graph, alignment fix) |

### CUDA Alignment Fix (v1.6.0+)

The IQ3_XXS CUDA kernel (`matmul_iq3_xxs.cu`) previously used `*(unsigned int*)ptr` and `*(unsigned short*)ptr` casts that require 4-byte/2-byte alignment. Since IQ3_XXS blocks are 98 bytes (not 4-byte aligned), block start addresses were frequently misaligned, causing CUDA error 716 (`ILLEGAL_ADDRESS`).

Fixed by replacing all unaligned pointer casts with byte-level read helpers (`read_u16()`, `read_u32()`) that reconstruct values from individual bytes. Grid indices use `__ldg()` for single-byte texture cache reads (already safe).

## CPU Execution Profile

**Hardware:** Intel Core Ultra 7 155H

| Metric | Value |
|--------|-------|
| v1.6.0 | 0.8 tok/s |
| SIMD tensors | Q8_0 layers only |

IQ3_XXS uses scalar dequantization (no fused SIMD variant). Only the Q8_0 intermediate tensors and Q6_K output tensor benefit from SIMD acceleration.

## Implementation Notes

- The smallest quantization available for this model at 537 MB (30% smaller than Q4_K_M)
- Quality degradation is expected at 3.06 bpw compared to 4.5 bpw variants
- Mixed quantization means the output projection uses Q6_K while hidden layers use IQ3_XXS
- The grid lookup dequantization is computationally more expensive than simple linear dequant

## Known Issues

1. **No SIMD fused dequant+dot for IQ3_XXS**: CPU path uses scalar dequantization, limiting throughput.
2. **Low output quality at 3.06 bpw**: Output coherence is poor compared to Q4_K_M (expected at this compression level).

## Version History

| Version | tok/s (GPU) | Notes |
|---------|-------------|-------|
| v1.5.1 | 1.3 | Per-tensor CUDA |
| v1.6.0 | Crash | CUDA error 716 (unaligned access) |
| v1.6.0+ | **20.2** | CUDA graph, byte-level reads fix |
