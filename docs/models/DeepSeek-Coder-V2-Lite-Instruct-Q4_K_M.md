# DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf` |
| Size | 9885 MB |
| Parameters | 16B total (2.4B active) |
| Architecture | DEEPSEEK2 |
| Quantization | Q4_K_M |
| Tokenizer | SentencePiece |

## Architecture

**Inference Engine:** `DeepSeek2InferenceEngine`

| Property | Value |
|----------|-------|
| Layers | 27 |
| Normalization | Pre-norm (RMSNorm) |
| Attention | Multi-Head Latent Attention (MLA) |
| KV LoRA Rank | 512 |
| RoPE Dims | 64 (shared across all heads) |
| FFN (leading blocks) | Dense SwiGLU |
| FFN (remaining blocks) | MoE: 64 experts, top-6 selection + 2 shared experts |

**Chat Template:** `User: {prompt}\n\nAssistant:`

**Multi-Head Latent Attention (MLA) details:**
- Q path: `wq * x` produces `[headCount * keyLength]`
- KV path: `wkvA * x` produces compressed latent, then `wkvB * norm(latent)` expands to full K and V
- K per head is composed of: `[K_nope, broadcast(RoPE(k_rope))]`
- Shared 64-dim RoPE applied across all heads
- KV compression via low-rank bottleneck reduces KV cache size significantly

**MoE FFN details:**
- Leading blocks use standard dense SwiGLU FFN
- Remaining blocks use Mixture-of-Experts: 64 total experts, top-6 selected per token, plus 2 shared experts that are always active
- Only ~2.4B parameters active per token despite 16B total

## GPU Profile

- **VRAM fit:** MoE-optimized placement supported.
- **GPU strategy:** MoE-optimized -- all 27/27 attention layers on GPU, expert tensors remain on CPU.
- **VRAM used:** ~517 MB (attention tensors only).
- **CudaForwardPass:** NOT supported (MLA architecture requires custom attention path).
- **Benchmark:** v1.5.1: ~2.1 tok/s with MoE-optimized placement.

**Why MoE-optimized works well:**
- Expert tensors are large but only top-6 of 64 are activated per token -- GPU parallelism is wasted on idle experts.
- Attention is compute-bound and benefits from GPU acceleration on every token.
- With standard first-N-layers, only ~3/27 layers would fit in 6 GB VRAM for this model.

## CPU Profile

No benchmark data available.

## Known Issues

- MLA requires a custom attention path in `DeepSeek2InferenceEngine`, distinct from standard GQA. The KV compression and shared RoPE are not compatible with the standard `Attention` class.
- Expert tensors constitute the bulk of the model's weight (~80-90% per MoE layer) but are sparsely activated, making them inefficient for GPU offload.
- CudaForwardPass (GPU-resident forward pass) is not supported for MLA architectures.

## Version History

| Version | Notes |
|---------|-------|
| v1.4.0 | DeepSeek2 MLA + MoE architecture support |
| v1.5.0 | MoE-optimized GPU placement strategy |
