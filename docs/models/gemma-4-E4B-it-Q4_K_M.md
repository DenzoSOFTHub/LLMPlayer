# gemma-4-E4B-it-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `gemma-4-E4B-it-Q4_K_M.gguf` |
| Size | 4747 MB |
| Parameters | 4B (with PLE) |
| Quantization | Q4_K_M (mixed Q4_K, Q5_0, Q6_K) |
| Tokenizer | BPE (SentencePiece-style with space prefix `▁`) |
| Chat Template | Gemma format with special turn markers |

## Architecture

| Field | Value |
|-------|-------|
| Architecture | GEMMA4 |
| Inference Engine | `Gemma4InferenceEngine` (PLE variant) |
| RoPE | NORMAL (dual: SWA theta=10K, full theta=1M with proportional factors) |
| FFN Type | GeGLU |
| Norm | Pre+post attention/FFN RMSNorm |
| QK-Norm | Yes (per-head, with `(1+w)` weight adjustment for K-norm) |
| V-Norm | Yes (RMSNorm without learnable scale) |
| Attention Scale | 1.0 (no `1/sqrt(headSize)` scaling) |
| Logit Soft-Cap | Yes (tanh-based) |

### Special features

- **Per-Layer Embeddings (PLE):** layer-specific token embedding tensors loaded into `per_layer_token_embd`, projected via `per_layer_model_proj` and `per_layer_input_gate` per block.
- **Dual headSize:** SWA layers use 256 head dim, full attention layers use 512.
- **Shared KV cache:** layers 24-41 reuse KV from layers 6-23 (mapping: SWA→`firstShared - 2`, full→`firstShared - 1`).
- **Sliding window pattern:** boolean array from GGUF `sliding_window_pattern` metadata.

## GPU Profile

Hardware: NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM).

| Version | tok/s | Mode | Notes |
|---------|-------|------|-------|
| v1.10.0 | -- | CPU + per-tensor CUDA matmul | No dedicated `Gemma4CudaForwardPass` yet |
| v1.10.2 | -- | CPU + per-tensor CUDA matmul | Output now correct (PPL 1.00) — see Resolved Issues |

- The standard `CudaForwardPass` does not support PLE; falls back to per-tensor CUDA matmul.
- Per-tensor mode still uses GPU for the heavy matmul ops, but pays full upload/download per layer.
- Future: dedicated `Gemma4CudaForwardPass` with PLE pre-computation and dual headSize support.

## CPU Profile

| Version | tok/s | PPL | Notes |
|---------|-------|------|-------|
| v1.10.0 | ~0.9 | 0.00 (junk) | CPU SIMD, output broken (multilingual gibberish) |
| v1.10.2 | ~0.9 | **1.00** ✓ | Output coherent: "The capital of France is **Paris**." |

## Resolved Issues (v1.10.2)

Two missing pieces vs `llama.cpp gemma4-iswa.cpp`:

1. **V-norm** on V projections — `Vcur = ggml_rms_norm(ctx0, Vcur, eps)` (RMSNorm without learnable scale). Was disabled for Gemma 4.
2. **`layer_output_scale.weight`** per-layer scalar applied as final multiplication of the residual stream: `cur = ggml_mul(cur, out_scale); inpL = cur;`. Per-layer values 0.06-0.88 do compound across 42 layers — but the model was trained with these scales, so this IS the correct algorithm.
3. **K-norm conditional** — `(1+w)` adjustment is now Gemma 3n-only. Gemma 4 stores final values (Q≈0.98, K≈0.13).

Verified outputs at v1.10.2 (Q4_K_M, RTX 4050):
- "What is the capital of France?" → "The capital of France is **Paris**." (PPL 1.00)
- "Who wrote Hamlet?" → "**William Shakespeare** wrote *Hamlet*." (PPL 0.98)
- "1+1=" → "1+1=**2**" (PPL 0.99)
- "The largest planet is" → "The largest planet in our solar system is **Jupiter**." (PPL 1.00)

## Known Issues

- No CUDA graph mode (would require dedicated `Gemma4CudaForwardPass`).
- BPE decode required a fix in v1.10.2 — `BPETokenizer.decodeTokenPiece` was applying GPT-2 byte mapping to gemma4-mode tokens, leaving `▁` literal in output. Fixed: when `useGpt2ByteMapping=false`, replace `▁` with space and decode `<0xHH>` byte fallback tokens.

## Version History

| Version | Change |
|---------|--------|
| v1.9.0 | Initial Gemma 4 architecture support (PLE, dual headSize, shared KV, V-norm, dual RoPE, K-norm `(1+w)`, logit soft-cap) |
