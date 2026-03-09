# Qwen2.5-Coder-32B-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `Qwen2.5-Coder-32B-Q4_K_M.gguf` |
| Size | 18932 MB |
| Parameters | 32B |
| Architecture | QWEN2 |
| Quantization | Q4_K_M |
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

- **VRAM fit:** Minimal offload. 19 GB far exceeds 6 GB VRAM budget.
- **Offload mode:** First-N-layers strategy with per-tensor CUDA.
- **CudaForwardPass:** Supported (dense pre-norm with separate Q/K/V).
- **Benchmark data:** None available.

## CPU Profile

No benchmark data available.

## Known Issues

- At 19 GB, very few layers fit in 6 GB VRAM. CPU-only inference with SIMD Vector API is the practical acceleration path.
- This is the 32B variant of the Qwen2.5-Coder family. For systems with limited RAM or VRAM, consider the 14B variants (Q6_K at 11.5 GB or Q4_K_M at ~8 GB).

## Version History

| Version | Notes |
|---------|-------|
| v1.4.0 | QWEN2 architecture support |
