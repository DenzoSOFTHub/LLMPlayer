# Gemma 3 1B IT Q4_K_M

## Model Summary

| Field | Value |
|-------|-------|
| File | `gemma-3-1b-it-Q4_K_M.gguf` |
| Size | 769 MB |
| Architecture | GEMMA3 |
| Parameters | 1B |
| Quantization | Q4_K_M (uses Q5_0 for Q, K, gate, up projections) |
| Layers | 26 |
| Heads | 8 Q / 4 KV (GQA 2:1) |
| Embedding dim | 1152 |
| Head size | 256 |
| FFN dim | 6144 |

## Architecture Description

### Forward Pass

Pre+Post norm transformer (InferenceEngine). Gemma 3 applies normalization both before and after attention and FFN blocks.

```
RMSNorm -> Attention -> PostAttnNorm -> Residual -> RMSNorm -> GeGLU FFN -> PostFfnNorm -> Residual
```

### Attention

- Grouped Query Attention (GQA): 8 query heads, 4 key-value heads (2:1 ratio)
- RoPE positional encoding: type NORMAL (consecutive pairs), base frequency 10000
- **Interleaved Sliding Window Attention (ISWA):** every 6th layer is global (layer % 6 == 5), all others use sliding window (local attention)
- **Dual RoPE:** global layers use the main RoPE theta, local layers use theta=10000
- **Per-head QK-norm:** RMSNorm applied to each Q and K head independently before attention score computation
- **Attention logit soft-capping:** scores are capped via `tanh(score / cap) * cap` to prevent extreme attention weights
- Head size 256 (largest among supported 1B models)

### FFN

- **GeGLU activation** (GELU instead of SiLU): `GELU(gate * x) * up`
- FFN intermediate dimension: 6144
- This differs from most other supported architectures which use SwiGLU

### Normalization

- Pre+Post norm: RMSNorm before and after both attention and FFN
- **Important:** Gemma norm weights in GGUF are stored as final values. They do NOT need `(1 + w)` adjustment. Adding +1 breaks output for both Gemma 2 and Gemma 3.

### Embedding

- Embedding scaling: input embeddings are multiplied by `sqrt(embeddingLength)` (i.e., `sqrt(1152)`)
- Final logit soft-capping applied

### Tokenizer

- SentencePiece (score-based)

### Chat Template

```
<start_of_turn>user
Prompt<end_of_turn>
<start_of_turn>model
```

## Quantization Details

Although the model is labeled Q4_K_M, it uses mixed quantization types:

- **Q5_0** for Q, K, gate, and up projection weights (higher precision for critical paths)
- **Q4_K** for other weight tensors
- **Q6_K** for the output projection
- **Q8_0** for intermediate activations

Q5_0 specifics:
- 32 elements per block, 18 bytes per block
- Split nibble layout (NOT interleaved): elements 0-15 use low nibbles of bytes 0-15, elements 16-31 use high nibbles of bytes 0-15
- The qh (high bit) field provides the 5th bit for each element

## GPU Execution Profile

**Hardware:** NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM)

| Metric | Value |
|--------|-------|
| Mode | CUDA graph (26/26 layers) |
| VRAM | 769 MB (full offload) |
| v1.5.1 | 4.2 tok/s (per-tensor) |
| v1.6.0 | 31.1 tok/s (CUDA graph) |
| Improvement | **+640% from v1.5.1 to v1.6.0** |

### v1.5.1 vs v1.6.0

In v1.5.1, CudaForwardPass did not support post-norm architectures, so Gemma 3 was limited to per-tensor CUDA matmul (4.2 tok/s). In v1.6.0, CudaForwardPass was extended to handle pre+post norm, enabling full CUDA graph capture across all 26 layers and delivering 31.1 tok/s.

### CUDA Kernel Coverage

- Q4_K: vectorized warp-per-row kernel with `uint32` loads
- Q5_0: dedicated CUDA kernel (added v1.5.1)
- Q6_K: coalesced position-parallel kernel
- Q8_0: CUDA kernel for intermediate tensors
- GeGLU: handled by substituting GELU for SiLU in the fused activation kernel

## CPU Execution Profile

**Hardware:** Intel Core Ultra 7 155H

| Metric | Value |
|--------|-------|
| v1.6.0 | 2.0-2.2 tok/s |
| SIMD tensors | Q4_K, Q8_0, Q6_K, Q5_0, Q5_K, Q3_K (fused dequant+dot) |

## Implementation Notes

- Gemma 3's ISWA pattern (every 6th layer global) differs from Gemma 2's alternating pattern -- the correct pattern was a bug fix in v1.5.0
- The per-head QK-norm requires separate RMSNorm computation for each of the 8 Q heads and 4 K heads per layer
- Dual RoPE means the attention implementation must check each layer's type (local vs global) and apply the corresponding theta
- Attention logit soft-capping adds a `tanh` computation to the attention score path
- Despite being 1B parameters, Gemma 3 has 26 layers (more than Llama 1B's 16), trading width for depth

## Known Issues

1. **Output quality at 1B scale:** Gemma 3 1B IT produces somewhat coherent but imperfect output. This appears to be a model quality limitation at the 1B parameter scale, not a LLMPlayer code bug. Dequantization has been verified bit-exact against llama.cpp reference output.

2. **Q5_0 dequantization history:** Before v1.5.0, Q5_0 used incorrect interleaved nibble ordering (like Q4_0) instead of the correct split layout. This caused garbage output since Q5_0 is used for Q, K, gate, and up projections -- the most critical weight tensors. The fix was verified against llama.cpp reference values.

3. **Norm weight handling:** Early development mistakenly applied `(1 + w)` to Gemma norm weights. GGUF stores Gemma norms as final values, so this adjustment produces incorrect results.

## Version History

| Version | tok/s (GPU) | Notes |
|---------|-------------|-------|
| v1.4.0 | -- | Garbage output (Q5_0 dequantization bug) |
| v1.5.0 | 4.0 | Q5_0 fix applied; per-tensor GPU |
| v1.5.1 | 4.2 | Q5_0 CUDA kernel added |
| v1.6.0 | 31.1 | CudaForwardPass supports Gemma 3 (+640%) |
