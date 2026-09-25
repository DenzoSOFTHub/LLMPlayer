# Qwen3-VL-4B-Instruct-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `Qwen3-VL-4B-Instruct-Q4_K_M.gguf` |
| Size | 2497 MB |
| Parameters | 4.4B |
| Tokenizer | BPE (qwen2 pre-tokenizer) |
| Chat Template | ChatML, no think block (the template has no `<think>`) |

## Architecture

| Field | Value |
|-------|-------|
| Architecture | QWEN3 (GGUF `qwen3vl`) |
| Inference Engine | `InferenceEngine` |

Multi-axis RoPE with interleaved sections `[24, 20, 20, 0]`; text tokens use positions `(p, p, p, 0)`. Image input with `mmproj-F16.gguf` from `unsloth/Qwen3-VL-4B-Instruct-GGUF` (qwen3vl_merger, 3 deepstack layers).

## CPU Profile (2026-09-24)

Shared 8-vCPU VirtualBox VM (Core Ultra 7 155H, AVX2) with other heavy workloads running, so the
speeds are indicative only. `--no-gpu`, default sampling unless stated.

| Test | Speed | Result |
|------|-------|--------|
| Short prompt (19 tokens, 80 generated) | 1.8 tok/s | PPL 1.01, correct Java answer |
| Retrieval prompt (1536 tokens) | 2.8 tok/s | Correct badge number and city |
| Image (640 × 480, 300 image tokens) | 1.2 tok/s, 26 s encode | Correct description of the llama.cpp test image |
| Teacher-forced PPL, prose + Java sample | — | 2.92 over 248 tokens |
