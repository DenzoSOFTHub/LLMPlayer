# Gemma 2 2B IT IQ4_XS

## Model Summary

| Field | Value |
|-------|-------|
| File | `gemma-2-2b-it-IQ4_XS.gguf` |
| Size | 1500 MB |
| Architecture | GEMMA2 |
| Parameters | 2B |
| Quantization | IQ4_XS (4.25 bpw) |
| Layers | 26 |
| Heads | 8 Q / 4 KV (GQA 2:1) |
| Embedding dim | 2304 |
| Head size | 256 |
| FFN dim | 9216 |

## Architecture Description

### Forward Pass

Pre+Post norm transformer (InferenceEngine). Same norm pattern as Gemma 3.

```
RMSNorm -> Attention -> PostAttnNorm -> Residual -> RMSNorm -> GeGLU FFN -> PostFfnNorm -> Residual
```

### Attention

- Grouped Query Attention (GQA): 8 query heads, 4 key-value heads (2:1 ratio)
- RoPE positional encoding: type NORMAL (consecutive pairs), base frequency 10000
- **Sliding window (ISWA):** alternating layers -- even layers are local (sliding window), odd layers are global (full attention). This differs from Gemma 3's every-6th-layer pattern.
- **Dual RoPE:** global layers use main theta, local layers use theta=10000
- **No QK-norm** (unlike Gemma 3 which has per-head QK-norm)
- **Attention logit soft-capping**
- Head size 256

### FFN

- **GeGLU activation** (GELU instead of SiLU): `GELU(gate * x) * up`
- FFN intermediate dimension: 9216

### Normalization

- Pre+Post norm (RMSNorm before and after attention and FFN)
- Gemma norm weights stored as final values in GGUF (no +1 adjustment)

### Embedding

- Embedding scaling: `sqrt(embeddingLength)` (i.e., `sqrt(2304)`)
- Final logit soft-capping

### Tokenizer

- SentencePiece (score-based)

### Chat Template

```
<start_of_turn>user
Prompt<end_of_turn>
<start_of_turn>model
```

## Quantization Details

IQ4_XS (Importance-weighted 4-bit, extra-small):

- 4.25 bits per weight average
- Importance-weighted quantization: allocates more precision to high-importance weights
- Uses lookup tables for dequantization
- Provides better quality-per-bit than uniform quantization methods

## GPU Execution Profile

**Hardware:** NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM)

| Metric | Value |
|--------|-------|
| Mode | CUDA graph (26/26 layers) |
| VRAM | ~1500 MB (full offload) |
| v1.5.1 | 9.0 tok/s |
| v1.6.0 | 8.3-8.9 tok/s |

### IQ4_XS CUDA Kernel Impact

Before the IQ4_XS CUDA kernel was added (v1.4.0), performance was 2.4 tok/s with CPU fallback for IQ4_XS tensors. The dedicated kernel provided a 3.8x improvement to 9.0 tok/s.

### Performance vs Gemma 3 1B

Despite Gemma 2 2B being a larger model (2304 dim vs 1152, 9216 FFN vs 6144), it achieves similar CUDA graph throughput to early per-tensor Gemma 3 results. The IQ4_XS quantization is more compact per element but requires more compute in the dequantization path.

## CPU Execution Profile

**Hardware:** Intel Core Ultra 7 155H

| Metric | Value |
|--------|-------|
| v1.6.0 | 1.1 tok/s |
| SIMD tensors | Q8_0 (IQ4_XS uses scalar path) |

CPU throughput is limited because IQ4_XS does not have a fused SIMD dequant+dot implementation. Only the Q8_0 intermediate tensors benefit from SIMD acceleration.

## Gemma 2 vs Gemma 3 Differences

| Feature | Gemma 2 | Gemma 3 |
|---------|---------|---------|
| Sliding window pattern | Alternating (even=local, odd=global) | Every 6th layer (layer%6==5 is global) |
| QK-norm | No | Yes (per-head) |
| Embedding dim (1B/2B) | 2304 (2B) | 1152 (1B) |
| FFN dim | 9216 | 6144 |

Both share: pre+post norm, GQA 2:1, dual RoPE, attention logit soft-capping, GeGLU FFN, embedding scaling.

## Implementation Notes

- The alternating sliding window pattern (even=local, odd=global) was a bug fix -- before v1.5.0, Gemma 2 incorrectly used the Gemma 3 pattern (every 6th layer)
- At 1500 MB, this is the largest model in the small-model benchmark suite but still fits entirely in 6 GB VRAM
- CUDA graph mode is supported because all 26 layers use the same forward pass structure

## Known Issues

1. **Gemma 2 sliding window pattern was incorrect before v1.5.0:** Used the Gemma 3 every-6th-layer pattern instead of alternating. This caused incorrect attention masking on half the layers.

2. **No SIMD IQ4_XS implementation:** CPU path uses scalar dequantization for IQ4_XS tensors, limiting CPU throughput to 1.1 tok/s.

## Version History

| Version | tok/s (GPU) | Notes |
|---------|-------------|-------|
| v1.4.0 | 2.4 | No IQ4_XS CUDA kernel; CPU fallback |
| v1.5.0 | -- | Sliding window pattern fix, Q5_0 dequant fix |
| v1.5.1 | 9.0 | IQ4_XS CUDA kernel added (3.8x improvement) |
| v1.6.0 | 8.3-8.9 | Stable, CUDA graph mode |
