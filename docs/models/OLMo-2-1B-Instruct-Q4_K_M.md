# OLMo-2-1B-Instruct Q4_K_M

## Model Summary

| Field | Value |
|-------|-------|
| File | `OLMo-2-1B-Instruct-Q4_K_M.gguf` |
| Size | 893 MB |
| Architecture | OLMO2 |
| Parameters | 1B |
| Quantization | Q4_K_M (4.5 bpw) |
| Layers | 16 |
| Heads | 16 Q / 16 KV (MHA 1:1) |
| Embedding dim | 2048 |
| Head size | 128 |
| FFN dim | 8192 |

## Architecture Description

### Forward Pass

**Post-norm only** transformer (InferenceEngine). This is unique among LLMPlayer's supported models -- all others use pre-norm or pre+post-norm.

```
Attention -> PostAttnNorm -> Residual -> FFN -> PostFfnNorm -> Residual
```

There is no pre-attention norm and no pre-FFN norm. The RMSNorm is applied after the attention/FFN output, before the residual addition.

### Attention

- Multi-Head Attention (MHA): 16 query heads, 16 key-value heads (1:1 ratio, not grouped)
- RoPE positional encoding: type NEOX (split-half), base frequency 10000
- No QK-norm, no bias, no sliding window
- Head size 128 (larger than Llama's 64)

### FFN

- SwiGLU activation
- FFN intermediate dimension: 8192

### Normalization

- Post-norm only (RMSNorm after attention and after FFN)
- No pre-norms at all

### Tokenizer

- SentencePiece (score-based)

### Chat Template

```
<|user|>
Prompt
<|assistant|>
```

## Quantization Details

Q4_K_M: same structure as Llama Q4_K_M (256 elements/block, 144 bytes/block, 4-byte aligned).

## GPU Execution Profile

**Hardware:** NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM)

| Metric | Value |
|--------|-------|
| Mode | CudaForwardPass + CUDA graph (v1.6.0+), fused gate+up |
| VRAM | 893 MB (full offload, 16/16 layers) |
| v1.5.1 | 25.5 tok/s (per-tensor CUDA) |
| v1.6.0 | 22.4 tok/s (per-tensor, pre-fix) |
| v1.6.0+ | **49.2 tok/s** (CudaForwardPass + CUDA graph) |

### CudaForwardPass Support (v1.6.0+)

Post-norm-only support was added to CudaForwardPass. When `attnNorm`/`ffnNorm` are null (no pre-norm), the forward pass uses `cuMemcpyDtoDAsync` to copy gpuX -> gpuXb instead of applying RMSNorm. Post-attention and post-FFN norms are then applied as usual. This enables full CUDA graph capture with fused gate+up kernel, yielding a 2.2x speedup over per-tensor CUDA.

## CPU Execution Profile

**Hardware:** Intel Core Ultra 7 155H

| Metric | Value |
|--------|-------|
| v1.6.0 | 1.8 tok/s |
| SIMD tensors | Q4_K, Q8_0, Q6_K (fused dequant+dot) |

## Implementation Notes

- OLMo2 is the only post-norm-only architecture currently supported
- MHA (1:1 ratio) means no KV head sharing -- each query head has its own KV head
- Head size is 128 (vs 64 for Llama 1B), so each head processes a larger feature space
- RoPE uses NEOX (split-half) layout rather than NORMAL (consecutive pairs)
- File is 122 MB larger than Llama 1B Q4_K_M (893 MB vs 771 MB) due to MHA having twice as many KV heads

## Known Issues

None. All previous issues (CudaForwardPass rejection, performance variance) have been resolved.

## Version History

| Version | tok/s (GPU) | Notes |
|---------|-------------|-------|
| v1.5.1 | 25.5 | Per-tensor CUDA matmul |
| v1.6.0 | 22.4 | Per-tensor (pre-CudaForwardPass fix) |
| v1.6.0+ | **49.2** | CudaForwardPass + CUDA graph + fused gate+up |
