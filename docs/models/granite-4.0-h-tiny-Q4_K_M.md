# granite-4.0-h-tiny-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `granite-4.0-h-tiny-Q4_K_M.gguf` |
| Size | 4035 MB |
| Parameters | ~8B (hybrid) |
| Quantization | Q4_K_M (mixed Q4_K, Q5_0, Q6_K) |
| Tokenizer | BPE |
| Chat Template | Granite format with `<|start_of_role|>` markers |

## Architecture

| Field | Value |
|-------|-------|
| Architecture | GRANITE_HYBRID |
| Inference Engine | `NemotronHInferenceEngine` (shared engine for Mamba-2 + Attention + FFN hybrids) |
| Layer Mix | Mamba-2 SSM + GQA Attention + integrated SwiGLU FFN per layer |
| RoPE | NEOX |
| Scaling | embedding, attention, residual, logit (Granite 3.3-style 4 scaling factors) |

### Notes

Granite Hybrid alternates Mamba-2 SSM layers with full Attention layers, similar to Nemotron-H, but with the addition of a per-layer SwiGLU FFN block (with its own `ffn_norm`) inside both Mamba and Attention layers. LLMPlayer reuses `NemotronHInferenceEngine` and adds the Granite-style scaling factors and the `runIntegratedFFN()` path.

## GPU Profile

Hardware: NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM).

| Version | tok/s | Mode | Notes |
|---------|-------|------|-------|
| v1.10.0 | -- | Per-layer GPU forward pass | No CUDA graph (DtoD copies in Mamba layers) |

## CPU Profile

| Version | tok/s | Notes |
|---------|-------|-------|
| v1.10.0 | ~5.2 | Mamba-2 scan + GQA attention + integrated SwiGLU on CPU |

## Known Issues

- CUDA graph not yet supported (DtoD copies in Mamba state updates make graph capture difficult).
- Per-layer GPU mode achieves reasonable throughput but is slower than fully captured graphs.

## Version History

| Version | Change |
|---------|--------|
| v1.9.0 | Initial Granite Hybrid support via NemotronH engine + integrated FFN + scaling |
