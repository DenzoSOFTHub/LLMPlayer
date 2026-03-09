# qwen2.5-coder-14b-instruct-q6_k

## Model Info

| Field | Value |
|-------|-------|
| File | `qwen2.5-coder-14b-instruct-q6_k.gguf` |
| Size | 11563 MB |
| Parameters | 14B |
| Architecture | QWEN2 |
| Quantization | Q6_K (6 bpw, 256 elements/block, 210 bytes/block) |
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

- **VRAM fit:** Partial offload only. 11.5 GB exceeds 6 GB VRAM budget.
- **Offload mode:** Per-tensor CUDA for offloaded layers (first-N-layers strategy).
- **CudaForwardPass:** Supported (dense pre-norm with separate Q/K/V).
- **Benchmark data:** None available.

**Q6_K CUDA kernel note:** The coalesced Q6_K kernel introduced in v1.4.0 achieved a 3x improvement for output projection matmul (8.3 ms/tok down to 2.7 ms/tok). This benefits Q6_K-heavy models significantly, as the output projection is typically the largest single matmul per token.

## CPU Profile

No benchmark data available.

## Known Issues

- Q6_K block size (210 bytes) is not divisible by 4, so CUDA kernels use byte-level `__ldg` loads rather than vectorized uint32 loads.
- At 11.5 GB, only a fraction of layers fit in 6 GB VRAM; partial offload may be slower than pure CPU.

## Version History

| Version | Notes |
|---------|-------|
| v1.4.0 | Coalesced Q6_K CUDA kernel (3x improvement for output matmul) |
