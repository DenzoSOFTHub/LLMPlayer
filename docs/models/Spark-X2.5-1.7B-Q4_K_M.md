# Spark-X2.5-1.7B-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `Spark-X2.5-1.7B-Q4_K_M.gguf` |
| Size | 1107 MB |
| Parameters | 1.7B |
| Tokenizer | BPE with the multi-regex `spark2_5` pre-tokenizer |
| Chat Template | `<｜start▁of▁sentence｜><|User|>…<｜end▁of▁sentence｜>`, `</think>` after `<|Bot|>` unless `--thinking` |

## Architecture

| Field | Value |
|-------|-------|
| Architecture | SPARK2_5 (`spark2_5`) |
| Inference Engine | `InferenceEngine` (batched prefill) |

3:1 sliding-window (512) to full attention; SWA layers rotate all 256 dims at θ=1e4, full layers 64 dims at θ=5e6; per-head sigmoid output gate; GELU FFN; fused QKV, split after one multi-token matmul in batched prefill.

## CPU Profile (2026-09-24)

Shared 8-vCPU VirtualBox VM (Core Ultra 7 155H, AVX2) with other heavy workloads running, so the
speeds are indicative only. `--no-gpu`, default sampling unless stated.

| Test | Speed | Result |
|------|-------|--------|
| Short prompt (27 tokens, 100 generated) | 7.1 tok/s | PPL 1.28, correct Java answer |
| Retrieval prompt (1592 tokens, beyond the 512 window), token-by-token prefill | 191 s total | Correct badge number and city |
| Same prompt, batched prefill (greedy) | 93 s total | Same answer |
