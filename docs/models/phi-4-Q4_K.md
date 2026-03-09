# phi-4-Q4_K

## Model Info

| Field | Value |
|-------|-------|
| File | `phi-4-Q4_K.gguf` |
| Size | 8634 MB |
| Parameters | 14B |
| Architecture | PHI3 |
| Quantization | Q4_K |
| Tokenizer | BPE |

## Architecture

**Inference Engine:** `InferenceEngine` (standard transformer path)

| Property | Value |
|----------|-------|
| Layers | 40 |
| Attention Heads | 40 |
| KV Heads | 10 |
| Embedding Dim | 5120 |
| Head Size | 128 |
| FFN Dim | 17920 |
| GQA Ratio | 4:1 (40 Q heads : 10 KV heads) |
| Normalization | Pre-norm (RMSNorm) |
| RoPE Type | NEOX |
| QKV Layout | Merged QKV projection |
| FFN Layout | Packed FFN (gate+up merged) |
| FFN Activation | SwiGLU |

**Chat Template:**
```
<|user|>
{prompt}<|end|>
<|assistant|>
```

## GPU Profile

- **VRAM fit:** Partial offload only. 8634 MB exceeds 6 GB VRAM budget.
- **Estimated layers on GPU:** ~22/40 with 6 GB VRAM (first-N-layers strategy).
- **Offload mode:** Per-tensor CUDA for offloaded layers.
- **CudaForwardPass:** Supported (dense pre-norm with separate Q/K/V after unmerge), but partial offload limits effectiveness.
- **Benchmark data:** None available.

## CPU Profile

No benchmark data available.

## Known Issues

- Partial GPU offload is often slower than CPU-only inference due to CPU-GPU synchronization overhead at layer boundaries. For models that only partially fit in VRAM, CPU-only with SIMD Vector API may be faster.
- Merged QKV projection requires splitting into separate Q, K, V tensors during forward pass.

## Version History

| Version | Notes |
|---------|-------|
| v1.4.0 | PHI3 architecture support with merged QKV and packed FFN |
