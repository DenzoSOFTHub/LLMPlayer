# Nanbeige4.2-3B-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `Nanbeige4.2-3B-Q4_K_M.gguf` |
| Size | 2684 MB |
| Parameters | 3B (22 layers run twice) |
| Tokenizer | SentencePiece (BOS `<|im_start|>` not prepended) |
| Chat Template | ChatML with the model's default Chinese system prompt; `<think>` block suppressed unless `--thinking` |

## Architecture

| Field | Value |
|-------|-------|
| Architecture | NANBEIGE (`nanbeige`) |
| Inference Engine | `InferenceEngine` (44 logical layers) |

`num_loops = 2`; the output norm is applied at the loop boundary. Twice the compute of a 22-layer model per token.

## CPU Profile (2026-09-24)

Shared 8-vCPU VirtualBox VM (Core Ultra 7 155H, AVX2) with other heavy workloads running, so the
speeds are indicative only. `--no-gpu`, default sampling unless stated.

| Test | Speed | Result |
|------|-------|--------|
| Short prompt (45 tokens, 100 generated) | 2.0 tok/s | PPL 1.69, correct Java answer |
| Retrieval prompt (1586 tokens) | 1.2 tok/s | Correct badge number and city |
