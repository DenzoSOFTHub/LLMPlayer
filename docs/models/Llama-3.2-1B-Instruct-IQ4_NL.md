# Llama-3.2-1B-Instruct IQ4_NL

## Model Summary

| Field | Value |
|-------|-------|
| File | `Llama-3.2-1B-Instruct.IQ4_NL.gguf` |
| Size | 738 MB |
| Architecture | LLAMA |
| Parameters | 1B |
| Quantization | IQ4_NL (4.5 bpw) |
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

IQ4_NL (Importance-weighted Non-Linear 4-bit):

- 32 elements per block
- 18 bytes per block (NOT 4-byte aligned)
- 4.5 bits per weight average
- Uses a non-linear lookup table for dequantization, providing better quality than uniform Q4 at the same bit rate
- Importance-weighted: quantization error is minimized on high-importance weights

## GPU Execution Profile

**Hardware:** NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM)

| Metric | Value |
|--------|-------|
| Mode | CUDA graph (16/16 layers) |
| VRAM | 738 MB (full offload) |
| v1.5.1 | 28.7 tok/s |
| v1.6.0 | 28.5 tok/s |
| Features | `cuda_graph`, IQ4_NL CUDA kernel |

### CUDA Kernel Notes

The IQ4_NL CUDA kernel was added in v1.5.1. Before that, IQ4_NL tensors fell back to per-tensor CPU computation even when GPU was enabled, resulting in only 4.4 tok/s. The dedicated kernel provided a 6.5x improvement.

The 18 bytes/block size is not 4-byte aligned, so the kernel uses byte-level `__ldg` loads rather than `uint32` vectorized loads.

## CPU Execution Profile

**Hardware:** Intel Core Ultra 7 155H

| Metric | Value |
|--------|-------|
| v1.6.0 | 3.2 tok/s |
| SIMD tensors | Active for supported types |

CPU throughput is higher than Q4_K_M (3.2 vs 2.5 tok/s) because IQ4_NL uses smaller blocks (32 elements vs 256), reducing per-element overhead in the scalar dequantization path.

## Implementation Notes

- Same LLAMA architecture as the Q4_K_M variant; only the quantization differs
- IQ4_NL provides slightly better quality per bit than Q4_K_M due to importance weighting and non-linear quantization levels
- File is 33 MB smaller than the Q4_K_M variant (738 MB vs 771 MB)
- GPU performance is roughly half of Q4_K_M (28.5 vs 52.7 tok/s) due to the non-aligned block structure preventing vectorized memory loads

## Known Issues

None.

## Version History

| Version | tok/s (GPU) | Notes |
|---------|-------------|-------|
| v1.4.0 | 4.4 | No IQ4_NL CUDA kernel; per-tensor CPU fallback |
| v1.5.1 | 28.7 | IQ4_NL CUDA kernel added (6.5x improvement) |
| v1.6.0 | 28.5 | Stable |
