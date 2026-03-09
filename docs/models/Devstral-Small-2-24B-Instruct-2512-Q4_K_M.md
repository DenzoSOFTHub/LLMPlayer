# Devstral-Small-2-24B-Instruct-2512-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf` |
| Size | 13671 MB |
| Parameters | 24B |
| Architecture | MISTRAL3 |
| Quantization | Q4_K_M |
| Tokenizer | BPE |

## Architecture

**Inference Engine:** `InferenceEngine` (standard transformer path)

| Property | Value |
|----------|-------|
| Normalization | Pre-norm (RMSNorm) |
| Attention | GQA with RoPE NORMAL |
| RoPE Base | 1000000 (high base for long context support) |
| Key Length Override | 128 (headSize=160, keyLength=128) |
| FFN Activation | SwiGLU |

**Chat Template:**
```
[INST] {prompt} [/INST]
```

**Architecture notes:**
- Uses RoPE NORMAL variant (not NEOX) with a very high base frequency of 1,000,000, enabling long context windows.
- Key length (128) differs from head size (160). The key/query projections use 128-dimensional RoPE, while value projections use the full 160-dimensional head size.

## GPU Profile

- **VRAM fit:** Partial offload only. 13.7 GB exceeds 6 GB VRAM budget.
- **Offload mode:** First-N-layers strategy with per-tensor CUDA.
- **CudaForwardPass:** Supported (dense pre-norm with separate Q/K/V).
- **Benchmark data:** None available.

## CPU Profile

No benchmark data available.

## Known Issues

- The key length vs head size mismatch (128 vs 160) requires careful handling in the attention computation. RoPE is applied only to the key length dimensions.
- At 13.7 GB, partial GPU offload may not provide meaningful speedup over CPU-only inference.

## Version History

| Version | Notes |
|---------|-------|
| v1.4.0 | Mistral3/Devstral architecture support |
