# Qwen3.5-4B-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `Qwen3.5-4B-Q4_K_M.gguf` |
| Size | 2614 MB |
| Parameters | 4B |
| Quantization | Q4_K_M |
| Tokenizer | SentencePiece |
| Chat Template | `<\|im_start\|>user\nPrompt<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n\n</think>\n\n` |

## Architecture

| Field | Value |
|-------|-------|
| Architecture | QWEN35 (hybrid DeltaNet + Attention) |
| Inference Engine | Qwen35InferenceEngine (dedicated hybrid engine) |
| Layers | 36 |
| Attention Heads | 32 |
| KV Heads | 4 |
| Embedding Dim | 2560 |
| Full Attention Interval | 4 (every 4th layer) |
| FFN Type | SwiGLU (all layers) |
| Norm | Pre-norm (RMSNorm) |
| QK-Norm | Yes (per-head) |

### Hybrid Layer Structure

The model alternates between two layer types in a 3:1 ratio:

**DeltaNet layers (75% of layers):**
- Gated linear attention / SSM (no KV cache, recurrent state only)
- State update: `S_new = alpha*S + beta*outer(k, v - alpha*S^T@k)`
- Output: `o = S^T_new @ q`
- Short conv1d (width 4) on Q and K
- Per-head QK-norm

**Full attention layers (25% of layers):**
- Standard GQA with KV cache
- Packed Q+gate projection: interleaved `[Q_h0, gate_h0, Q_h1, gate_h1, ...]`
- Deinterleaved before attention; gate applied as `sigmoid(gate) * attn_output`
- Short conv1d on Q/K, per-head QK-norm
- RoPE NEOX

## GPU Profile

Hardware: NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM).

| Version | tok/s | Mode | Notes |
|---------|-------|------|-------|
| v1.5.1 | 8.0 | Per-tensor CUDA matmul | No CudaForwardPass (hybrid arch not supported) |
| v1.6.0 | 5.5-7.7 | Per-tensor CUDA | Same limitation |

- Full offload: 2614 MB VRAM
- CudaForwardPass / CUDA graph mode NOT supported: DeltaNet recurrence is incompatible with CUDA graph capture

## CPU Profile

| Version | tok/s | Notes |
|---------|-------|-------|
| v1.6.0 | 0.6 | SIMD tensors + SIMD QK-norm + SIMD RoPE NEOX |

## Known Issues

- **No CUDA forward pass support.** The DeltaNet recurrence requires custom CUDA kernels that have not been implemented. All GPU acceleration is per-tensor matmul only.
- **llama.cpp eval callback incompatibility.** Eval callbacks break recurrent models by forcing tensor materialization. When debugging, dump one tensor per full model load cycle.
- **Packed Q+gate deinterleaving complexity.** Full attention layers use a packed projection that must be deinterleaved in two steps (extract gates first, then compact Q). Errors here produce silent corruption.

## Version History

| Version | Change |
|---------|--------|
| v1.5.0 | Initial Qwen3.5 hybrid DeltaNet+Attention support |
| v1.5.1 | Per-tensor CUDA matmul for hybrid layers |
| v1.6.0 | Stability improvements |
