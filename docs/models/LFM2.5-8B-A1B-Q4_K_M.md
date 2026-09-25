# LFM2.5-8B-A1B-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `LFM2.5-8B-A1B-Q4_K_M.gguf` |
| Size | 5156 MB |
| Parameters | 8.3B total, ~1B active |
| Tokenizer | BPE (lfm2) |
| Chat Template | ChatML; the model always reasons in `<think>` |

## Architecture

| Field | Value |
|-------|-------|
| Architecture | LFM2 (GGUF `lfm2moe`) |
| Inference Engine | `LFM2InferenceEngine` |

32 experts, top-4, sigmoid routing with selection bias, 2 leading dense layers. Experts run through `ExpertViews` with row ranges over all selected experts (3.8 → 8.0 tok/s).

## CPU Profile (2026-09-24)

Shared 8-vCPU VirtualBox VM (Core Ultra 7 155H, AVX2) with other heavy workloads running, so the
speeds are indicative only. `--no-gpu`, default sampling unless stated.

| Test | Speed | Result |
|------|-------|--------|
| Factual question (22 tokens) | 8.0 tok/s | Correct answer (Canberra) after a short reasoning block |
| Retrieval prompt (1350 tokens, 120 generated) | 6.0 tok/s | Correct badge number; the budget ran out before the city |
| Teacher-forced PPL with BOS, prose + Java sample | — | 4.94 over 247 tokens (dense LFM2-1.2B: 3.14) |
