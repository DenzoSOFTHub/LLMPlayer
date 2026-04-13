# granite-4.0-h-micro-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `granite-4.0-h-micro-Q4_K_M.gguf` |
| Size | 1853 MB |
| Parameters | ~3B (hybrid) |
| Quantization | Q4_K_M |
| Tokenizer | BPE |
| Chat Template | Granite format with `<|start_of_role|>` markers |

## Architecture

| Field | Value |
|-------|-------|
| Architecture | GRANITE_HYBRID |
| Inference Engine | `NemotronHInferenceEngine` |
| Layer Mix | Mamba-2 SSM + Attention + integrated SwiGLU FFN |
| Scaling | Granite 3.3-style (embedding, attention, residual, logit) |

## GPU Profile

Hardware: NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM).

| Version | tok/s | PPL | Mode | Notes |
|---------|-------|-----|------|-------|
| v1.10.0 | ~6.2 | **0.20** ❌ | Per-layer GPU forward pass | Output garbage: ", and— as, as–..." |
| v1.10.2 | ~1.3 | **1.00** ✓ | Forced CPU fallback | Output correct: "The capital of France is Paris." |

**v1.10.2 fix**: `NemotronHCudaForwardPass` was missing the Granite scaling factors (`embeddingScale`, `attentionScale`, `residualScale`, `logitScale`) on the GPU path. `isSupported()` now returns `false` when any of these is non-zero, forcing CPU fallback. Future work: implement scaling on GPU to recover the 6.2 tok/s.

## CPU Profile

| Version | tok/s | PPL | Notes |
|---------|-------|------|-------|
| v1.10.0 | ~1.3 | 1.00 | Mamba-2 + Attention + FFN on CPU |
| v1.10.2 | ~1.3 | 1.00 | Same — was always working on CPU |

## Version History

| Version | Change |
|---------|--------|
| v1.9.0 | Initial Granite Hybrid support |
| v1.10.2 | Force CPU fallback on GPU (scaling factors not in CUDA path) — fixes PPL 0.20 → 1.00 |
