# Ling-3.0-tiny-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `Ling-3.0-tiny-Q4_K_M.gguf` |
| Size | 4917 MB |
| Parameters | 7.9B total, 1.3B active |
| Tokenizer | BPE with the `bailingmoe` pre-tokenizer (single digits) |
| Chat Template | `<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>…<role>ASSISTANT</role>` |

## Architecture

| Field | Value |
|-------|-------|
| Architecture | BAILINGMOE3 (`bailingmoe3`) |
| Inference Engine | `BailingMoE3InferenceEngine` |

18 Kimi Delta Attention layers and 6 gated MLA layers (3, 7, …, 23); 128 experts in 8 groups (4 kept), top-8, scale 2.5, one shared expert. Token-by-token prefill. Greedy decoding of code can fall into repetition inside comments; sampled output is coherent.

## CPU Profile (2026-09-24)

Shared 8-vCPU VirtualBox VM (Core Ultra 7 155H, AVX2) with other heavy workloads running, so the
speeds are indicative only. `--no-gpu`, default sampling unless stated.

| Test | Speed | Result |
|------|-------|--------|
| Factual question (34 tokens) | 4.6 tok/s | PPL 1.01, correct answer, natural EOS |
| Java prompt, sampled (100 generated) | 5.4 tok/s | PPL 2.20, correct method |
| Retrieval prompt (1581 tokens) | 8.0 tok/s overall | Correct badge number and city |
| Teacher-forced PPL, prose + Java sample | — | 2.41 over 265 tokens |
