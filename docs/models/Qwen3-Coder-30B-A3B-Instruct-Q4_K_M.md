# Qwen3-Coder-30B-A3B-Instruct-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` |
| Size | 17698 MB |
| Parameters | 30B total (3B active) |
| Architecture | QWEN3MOE |
| Quantization | Q4_K_M |
| Tokenizer | SentencePiece |

## Architecture

**Inference Engine:** `Qwen3MoEInferenceEngine`

| Property | Value |
|----------|-------|
| Layers | 48 |
| Normalization | Pre-norm (RMSNorm) |
| Attention | GQA with per-head QK-norm |
| Total Experts | 128 |
| Active Experts | top-8 per token |
| Shared Experts | 1 (always active) |
| FFN (leading blocks) | Dense SwiGLU |
| FFN (remaining blocks) | MoE: 128 experts, top-8 + 1 shared |

**Chat Template:**
```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

**Architecture notes:**
- Uses per-head QK-norm (RMSNorm applied to Q and K independently per head before attention), which stabilizes training and inference at scale.
- Leading blocks use standard dense SwiGLU FFN; remaining blocks switch to MoE.
- Only 3B of 30B parameters are active per token (top-8 of 128 experts + 1 shared expert).
- Expert tensors (`ffnGateExps`, `ffnUpExps`, `ffnDownExps`) comprise ~90% of model weight but are sparsely activated.

## GPU Profile

- **VRAM fit:** MoE-optimized placement.
- **GPU strategy:** MoE-optimized -- all 48/48 attention layers on GPU, expert tensors on CPU.
- **VRAM used:** ~540 MB (attention + norm + router + shared expert tensors only).
- **CudaForwardPass:** NOT supported (MoE architecture).
- **Benchmark:** v1.4.0: 1.7 tok/s with MoE-optimized placement, Perplexity 0.98, Coherence 0.99.

**MoE-optimized GPU efficiency:**
- With standard first-N-layers, only ~2/48 layers would fit in 6 GB VRAM for this 17.7 GB model.
- MoE-optimized placement fits 100% of attention on GPU using only ~540 MB VRAM because expert tensors (the bulk of each layer) stay on CPU.
- This is inspired by the KTransformers (SOSP'25) strategy: GPU for attention (compute-bound, benefits every token), CPU for experts (large but sparse).

## CPU Profile

No benchmark data available.

## Known Issues

- Expert tensors total ~17 GB on CPU. Only top-8 of 128 experts are activated per token, so most expert weight is idle during any given forward pass.
- At 1.7 tok/s, generation is slow but functional. The bottleneck is CPU-side expert computation (top-8 expert matmuls per MoE layer).
- The 48-layer depth means even attention-only GPU offload requires managing state across many layers.

## Version History

| Version | Notes |
|---------|-------|
| v1.4.0 | QWEN3MOE architecture support, benchmark: 1.7 tok/s MoE-optimized |
| v1.5.0 | MoE-optimized GPU placement auto-detection |
