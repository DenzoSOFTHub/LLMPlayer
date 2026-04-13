# gemma-3n-E4B-it-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `gemma-3n-E4B-it-Q4_K_M.gguf` |
| Size | 4329 MB |
| Parameters | 4B (with PLE) |
| Quantization | Q4_K_M (mixed Q4_K, Q5_0, Q6_K) |
| Tokenizer | SentencePiece BPE |
| Chat Template | Gemma format |

## Architecture

| Field | Value |
|-------|-------|
| Architecture | GEMMA3N |
| Inference Engine | `Gemma4InferenceEngine` (shared with Gemma 4 for PLE handling) |
| RoPE | NORMAL |
| FFN Type | GeGLU |
| Norm | Pre+post attention/FFN RMSNorm |

### Notes

Gemma 3n is the Per-Layer Embedding variant that targets on-device deployment. It shares the PLE machinery with Gemma 4 (`per_layer_token_embd`, `per_layer_model_proj`, `per_layer_input_gate`), so LLMPlayer reuses `Gemma4InferenceEngine` whenever the model declares `embeddingLengthPerLayer > 0`.

## GPU Profile

Hardware: NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM).

| Version | tok/s | PPL | Mode | Notes |
|---------|-------|-----|------|-------|
| v1.10.0 | -- | -- | CPU + per-tensor CUDA matmul | No dedicated `Gemma4CudaForwardPass` |
| v1.10.2 | -- | **0.97-1.00** ✓ | CPU + per-tensor CUDA matmul | Output coherent across canonical Q&A |

## CPU Profile

| Version | tok/s | PPL | Notes |
|---------|-------|------|-------|
| v1.10.0 | ~0.8 | (broken) | CPU SIMD via `Gemma4InferenceEngine` |
| v1.10.2 | ~3.7 | **0.97-1.00** ✓ | Full AltUp + Laurel + Gaussian top-k sparsity + PLE |

## Architecture Details

Uses the dedicated `forwardLayerGemma3nInner` + `forwardLayerAltup` paths in `Gemma4InferenceEngine`. Distinct from Gemma 4 (which uses the simpler `forwardLayer` path):
- **AltUp** — 4 parallel activation streams with learned router/predict/correct coefficients
- **Laurel** — low-rank residual branch (`laurel_l → laurel_r → post_norm → +cur`)
- **Gaussian top-k activation sparsity** — first 10 FFN layers apply `mean + std*1.6448 cutoff` ReLU
- **K-norm `(1+w)`** — Gemma 3n stores raw values, runtime adds 1.0
- **V-norm** — `RMSNorm.applyNoScale` on V projections
- **Shared KV cache** — last `sharedKvLayers` layers (15 of 35 for E4B) reuse earlier layers' KV

## Verified outputs (v1.10.2)

- "What is the capital of France?" → "The capital of France is **Paris**." (PPL 1.00)
- "Hello" → "Hi there! 👋"
- "Who wrote Hamlet?" → "**William Shakespeare** is the author of Hamlet."

## Known Issues

- No CUDA graph mode (would require dedicated `Gemma4CudaForwardPass` with PLE + AltUp on GPU).

## Version History

| Version | Change |
|---------|--------|
| v1.9.0 | Initial Gemma 3n support via the shared Gemma4 PLE engine (output broken) |
| v1.10.2 | AltUp + Laurel + sparsity working, PPL 0.97-1.00 |
