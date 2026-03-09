# qwen2.5-coder-14b-instruct-fp16

## Model Info

| Field | Value |
|-------|-------|
| File | `qwen2.5-coder-14b-instruct-fp16.gguf` |
| Size | 28179 MB |
| Parameters | 14B |
| Architecture | QWEN2 |
| Quantization | F16 (16-bit IEEE half-precision, 2 bytes/element) |
| Tokenizer | BPE |

## Architecture

**Inference Engine:** `InferenceEngine` (standard transformer path)

| Property | Value |
|----------|-------|
| Normalization | Pre-norm (RMSNorm) |
| Attention | GQA with RoPE |
| FFN Activation | SwiGLU |

**Chat Template:**
```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

## GPU Profile

- **VRAM fit:** Minimal offload. At 28 GB, far exceeds 6 GB VRAM. Only a handful of layers can be offloaded.
- **F16 CUDA kernel:** Added in v1.6.0.
- **CudaForwardPass:** Supported in principle (dense pre-norm), but impractical with partial offload at this size.
- **Benchmark data:** None available.

## CPU Profile

No benchmark data available.

## Known Issues

- Very large model footprint (28 GB). On systems with 32 GB RAM, memory pressure may cause swapping and severe performance degradation.
- F16 provides maximum quality (no quantization loss) but at 2x the size of Q8_0 and ~7x the size of Q4_K_M. Consider Q6_K or Q8_0 variants for a better quality/size tradeoff.
- Minimal GPU offload means nearly all compute runs on CPU; SIMD Vector API is the primary acceleration path.

## Version History

| Version | Notes |
|---------|-------|
| v1.6.0 | F16 CUDA kernel support added |
