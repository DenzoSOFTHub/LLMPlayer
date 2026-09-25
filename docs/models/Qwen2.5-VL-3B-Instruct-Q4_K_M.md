# Qwen2.5-VL-3B-Instruct-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf` |
| Size | 1930 MB |
| Parameters | 3.1B |
| Tokenizer | BPE (qwen2 pre-tokenizer) |
| Chat Template | ChatML |

## Architecture

| Field | Value |
|-------|-------|
| Architecture | QWEN2 (GGUF `qwen2vl`) |
| Inference Engine | `InferenceEngine` |

Multi-axis RoPE with sequential sections `[16, 24, 24, 0]`, which for text equals NEOX. Image input with `mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf` from `ggml-org/Qwen2.5-VL-3B-Instruct-GGUF` (qwen2.5vl_merger, window attention).

## CPU Profile (2026-09-24)

Shared 8-vCPU VirtualBox VM (Core Ultra 7 155H, AVX2) with other heavy workloads running, so the
speeds are indicative only. `--no-gpu`, default sampling unless stated.

| Test | Speed | Result |
|------|-------|--------|
| Short prompt (19 tokens, 80 generated) | 2.4 tok/s | PPL 2.55, correct Java answer |
| Retrieval prompt (1536 tokens) | 2.5 tok/s | Correct badge number and city |
| Image (644 × 476, 391 image tokens) | 3.8 tok/s, 52 s encode | Correct description of the llama.cpp test image, including the date |
