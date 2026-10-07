# MiniCPM4.1-8B-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `MiniCPM4.1-8B-Q4_K_M.gguf` (openbmb/MiniCPM4.1-8B-GGUF) |
| Size | 4736 MB |
| Parameters | 8.2B |
| Tokenizer | SentencePiece (`llama`), 73,448 tokens |
| Chat Template | ChatML (`<|im_start|>user`); `<think>\n\n</think>\n` unless `--thinking`, which appends nothing |

## Architecture

| Field | Value |
|-------|-------|
| Architecture | MINICPM (`minicpm`) |
| Inference Engine | `InferenceEngine` (batched prefill) |
| RoPE | NORM, θ=10000, LongRoPE short factors (`rope_factors_short.weight`, 64 values) |
| Scaling | embedding × 12, residual × 0.247, logits ÷ 16 |

32 layers, 32 query heads and 2 KV heads of size 128, SwiGLU FFN of 16384, separate output matrix.
Attention is dense, as in llama.cpp.

## CPU Profile (2026-10-07)

Shared 8-vCPU VirtualBox VM (Core Ultra 7 155H, AVX2) with other heavy workloads running, so the
speed is indicative only. `--no-gpu`, greedy.

| Test | Speed | Result |
|------|-------|--------|
| Java prompt (29 prompt tokens, 120 generated) | 0.4 tok/s | PPL 2.26, coherent Javadoc and class skeleton, truncated at the token limit |

The GPU path was not measured: the VM has no GPU. This is the largest MiniCPM text model whose
Q4_K_M file (4.97 GB) is in reach of a 6 GB GPU. Token embeddings stay on the CPU, but the remaining
weights and the KV cache are close to the budget, so expect the auto-detected `--gpu-layers` to keep a
few layers on the CPU.
