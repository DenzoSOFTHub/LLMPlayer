# Hy-MT2-1.8B-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `Hy-MT2-1.8B-Q4_K_M.gguf` |
| Size | 1133 MB |
| Parameters | 1.8B |
| Tokenizer | BPE with the multi-regex `hunyuan-dense` pre-tokenizer |
| Chat Template | `<｜hy_User｜>…<｜hy_Assistant｜>`, BOS prepended by the engine |

## Architecture

| Field | Value |
|-------|-------|
| Architecture | HUNYUAN_DENSE (`hunyuan-dense`) |
| Inference Engine | `InferenceEngine` |

Q/K RMSNorm applied after RoPE. NTK alpha already baked into `rope.freq_base` (11158840). A translation model: prompt with "Translate the following segment into Italian, without additional explanation."

## CPU Profile (2026-09-24)

Shared 8-vCPU VirtualBox VM (Core Ultra 7 155H, AVX2) with other heavy workloads running, so the
speeds are indicative only. `--no-gpu`, default sampling unless stated.

| Test | Speed | Result |
|------|-------|--------|
| English → Italian translation (44 prompt tokens) | 7.0 tok/s | PPL 1.04, fluent translation, natural EOS |
| Retrieval prompt (1358 tokens) | 3.8 tok/s | Correct badge number and city |
