# Gemma 3 1B IT Q4_K_M (bartowski)

## Model Summary

| Field | Value |
|-------|-------|
| File | `gemma-3-1b-it-Q4_K_M-bartowski.gguf` |
| Size | 769 MB |
| Architecture | GEMMA3 |
| Parameters | 1B |
| Quantization | Q4_K_M (bartowski quantization) |
| Layers | 26 |
| Heads | 8 Q / 4 KV (GQA 2:1) |
| Embedding dim | 1152 |
| Head size | 256 |
| FFN dim | 6144 |
| Quantizer | bartowski |

## Architecture Description

Identical architecture to the official Gemma 3 1B IT Q4_K_M. See [gemma-3-1b-it-Q4_K_M.md](gemma-3-1b-it-Q4_K_M.md) for full architecture details.

### Forward Pass

```
RMSNorm -> Attention -> PostAttnNorm -> Residual -> RMSNorm -> GeGLU FFN -> PostFfnNorm -> Residual
```

### Key Architectural Features

- Pre+Post norm (RMSNorm)
- GQA with 8 Q heads, 4 KV heads (2:1 ratio)
- RoPE NORMAL, base=10000
- Interleaved Sliding Window Attention (every 6th layer global)
- Dual RoPE (different theta for global vs local layers)
- Per-head QK-norm
- Attention logit soft-capping
- GeGLU FFN (GELU activation)
- Embedding scaling: sqrt(embeddingLength)
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

Same Q4_K_M label as the official quantization, but produced by the bartowski quantizer. The tensor type mix may differ from the official version, which can affect performance characteristics. Both files are 769 MB.

## GPU Execution Profile

**Hardware:** NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM)

| Metric | Value |
|--------|-------|
| Mode | CUDA graph (26/26 layers) |
| VRAM | 769 MB (full offload) |
| v1.5.1 | 4.4 tok/s (per-tensor) |
| v1.6.0 | 23.1 tok/s (CUDA graph) |

### Comparison with Official Q4_K_M

| Variant | v1.5.1 (per-tensor) | v1.6.0 (CUDA graph) |
|---------|----------------------|----------------------|
| Official | 4.2 tok/s | 31.1 tok/s |
| bartowski | 4.4 tok/s | 23.1 tok/s |

The bartowski variant is 26% slower than the official Q4_K_M in CUDA graph mode (23.1 vs 31.1 tok/s). This is likely due to a different tensor type mix -- the bartowski quantizer may assign different quantization types to certain tensors, which can affect kernel selection and throughput in the fused forward pass.

## CPU Execution Profile

**Hardware:** Intel Core Ultra 7 155H

| Metric | Value |
|--------|-------|
| v1.6.0 | 2.0 tok/s |
| SIMD tensors | Q4_K, Q8_0, Q6_K, Q5_0, Q5_K, Q3_K |

CPU performance is identical to the official variant.

## Implementation Notes

- The same CudaForwardPass code path is used for both variants
- Performance difference is entirely attributable to the quantization choices made by the bartowski quantizer
- Both variants benefit equally from the v1.6.0 CudaForwardPass post-norm support

## Known Issues

Same as the official Gemma 3 1B IT Q4_K_M variant:

1. Output quality limitations at 1B scale (model limitation)
2. Historical Q5_0 dequantization bug (fixed in v1.5.0)
3. Norm weight handling (GGUF stores final values, no +1 adjustment)

## Version History

| Version | tok/s (GPU) | Notes |
|---------|-------------|-------|
| v1.5.1 | 4.4 | Per-tensor CUDA |
| v1.6.0 | 23.1 | CUDA graph (26/26 layers) |
