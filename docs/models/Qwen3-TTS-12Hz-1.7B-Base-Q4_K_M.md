# Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf` |
| Size | 1036 MB talker + 446 MB mmproj (Q8_0) |
| Parameters | 1.7B talker + 0.3B codec |
| Tokenizer | BPE (qwen2), text + 3072 codec entries |
| Chat Template | Embedding-sum prompt (see `docs/architecture/vision-and-tts.md`) |

## Architecture

| Field | Value |
|-------|-------|
| Architecture | Talker `qwen3tts` (loaded as QWEN3) + mmproj `qwen3tts_gen` |
| Inference Engine | `it.denzosoft.llmplayer.tts.Qwen3Tts` |

From `ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF`. No speaker encoder, so no voice cloning. 24 kHz mono output, 80 ms per frame.

## CPU Profile (2026-09-24)

Shared 8-vCPU VirtualBox VM (Core Ultra 7 155H, AVX2) with other heavy workloads running, so the
speeds are indicative only. `--no-gpu`, default sampling unless stated.

| Test | Speed | Result |
|------|-------|--------|
| Italian sentence (10 words) | 0.17× real time | 3.52 s of audio in 20.3 s (decoder 6.2 s), natural `codec_eos` after 44 frames; waveform checked numerically (voiced 56 %, median F0 146 Hz), not by listening |
