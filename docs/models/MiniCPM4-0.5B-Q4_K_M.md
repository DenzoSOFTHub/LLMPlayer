# MiniCPM4-0.5B-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `MiniCPM4-0.5B.Q4_K_M.gguf` (mradermacher/MiniCPM4-0.5B-GGUF) |
| Size | 266 MB |
| Parameters | 0.5B |
| Tokenizer | SentencePiece (`llama`), 73,448 tokens |
| Chat Template | ChatML (`<|im_start|>user`), no thinking toggle |

## Architecture

| Field | Value |
|-------|-------|
| Architecture | MINICPM (`minicpm`) |
| Inference Engine | `InferenceEngine` (batched prefill) |
| RoPE | NORM, θ=10000, LongRoPE short factors (`rope_factors_short.weight`, 32 values) |
| Scaling | embedding × 12, residual × 0.286, logits ÷ 4 |

24 layers, 16 query heads and 2 KV heads of size 64, SwiGLU FFN of 4096, tied output embedding.

## CPU Profile (2026-10-07)

Shared 8-vCPU VirtualBox VM (Core Ultra 7 155H, AVX2) with other heavy workloads running (load
average 8 to 13), so the speeds are indicative only. `--no-gpu`, greedy.

| Test | Speed | Result |
|------|-------|--------|
| Factual question (21 prompt tokens) | 3.3 tok/s | "The capital of France is Paris.", PPL 1.01, natural EOS |
| Retrieval prompt (1691 tokens) | 142 s total | Correct badge number and city |

The GPU path was not measured: the VM has no GPU.
