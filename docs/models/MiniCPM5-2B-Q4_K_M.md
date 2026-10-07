# MiniCPM5-2B-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `MiniCPM5-2B-Q4_K_M.gguf` (openbmb/MiniCPM5-2B-GGUF, released 2026-09-05) |
| Size | 1489 MB |
| Parameters | 2.6B |
| Tokenizer | BPE (`gpt2`) with the `minicpm5` pre-tokenizer, 130,560 tokens |
| Chat Template | ChatML (`<|im_start|>user`); `<think>\n\n</think>\n\n` unless `--thinking`, which appends `<think>\n` |

## Architecture

| Field | Value |
|-------|-------|
| Architecture | LLAMA (`llama`; the ChatML template is detected from the GGUF) |
| Inference Engine | `InferenceEngine` (batched prefill) |
| RoPE | NORM, θ=5,000,000 |

42 layers, 16 query heads and 2 KV heads of size 128, SwiGLU FFN of 6144, separate output matrix.
There is no muP scaling: MiniCPM5 is a plain Llama checkpoint. The Java tokenizer was compared with
the Hugging Face `tokenizers` library on `tokenizer.json` and produced identical IDs on eight test
strings (code, contractions, long numbers, accented Latin, Chinese and Japanese, whitespace runs,
emoji and HTML).

## CPU Profile (2026-10-07)

Shared 8-vCPU VirtualBox VM (Core Ultra 7 155H, AVX2) with other heavy workloads running (load
average 8 to 13), so the speeds are indicative only. `--no-gpu`, greedy.

| Test | Speed | Result |
|------|-------|--------|
| Java prompt (22 prompt tokens, 200 generated) | 0.9 tok/s | PPL 2.20; a reasonable structure, but the greedy draft contains a wrong identifier (`header` for `head`) before it restarts with a corrected method |
| Retrieval prompt (1380 tokens) | 831 s total | Correct badge number and city, natural EOS |

The GPU path was not measured: the VM has no GPU. At 1.5 GB the model fits entirely on a 6 GB GPU.
