# Vision input and text-to-speech

This document describes the two multimodal pipelines added in the 2026-09-24 work: image input for
the Qwen vision-language models, and speech synthesis with Qwen3-TTS. Both follow the llama.cpp
`tools/mtmd` implementation and read the same GGUF files (a text model plus an `mmproj` file), so the
files published for llama.cpp work unchanged. Both pipelines run on the CPU only. Loading a vision
projector drops the model's GPU-resident forward pass (individual matmuls can still use GPU-backed
tensors), because image tokens need the multi-axis RoPE positions and embedding input that only the
CPU layer path implements. The standard GPU passes also decline Qwen3-VL and the Qwen3-TTS talker
outright, since their text rotation is not plain NEOX; Qwen2.5-VL and Qwen3.5 text keep the GPU pass.
None of this was tested on a GPU: the reference machine has none.

## Vision (`it.denzosoft.llmplayer.vision`)

### Supported models

| Text model | `general.architecture` | Engine | mmproj projector | Deepstack |
|---|---|---|---|---|
| Qwen3-VL 2B / 4B / 8B | `qwen3vl` (loaded as `QWEN3`) | `InferenceEngine` | `qwen3vl_merger` | 3 layers |
| Qwen3.5 0.8B – 9B | `qwen35` | `Qwen35InferenceEngine` | `qwen3vl_merger` | none |
| Qwen2.5-VL 3B / 7B (and fine-tunes such as UI-TARS-1.5) | `qwen2vl` (loaded as `QWEN2`) | `InferenceEngine` | `qwen2.5vl_merger` | none |

### Encoder (`VisionEncoder`)

The encoder mirrors `tools/mtmd/models/qwen3vl.cpp`:

1. **Preprocessing** (`ImagePreprocessor`). The image is decoded with `javax.imageio` and resized
   with its aspect ratio preserved so that both sides are multiples of 32 (patch size 16 × spatial
   merge 2) and the number of merged tokens lies between `-Dvision.min.tokens` (default 8) and
   `-Dvision.max.tokens` (default 576, about 768 × 768 pixels). The resampler is a Pillow-style
   separable bicubic filter with a = -0.5 that widens its support when downsampling, as in
   `mtmd-image.cpp`. Pixels are normalised with the mean and standard deviation from the mmproj.
2. **Patch embedding.** The two temporal Conv3d kernels (`v.patch_embd.weight` and `.weight.1`) are
   applied to the same frame and summed, which is how llama.cpp handles a still image. Patches are
   processed in 2 × 2 merge-block order: block row, block column, then the four patches of the block.
3. **Position embedding.** The learned square grid (48 × 48 for a 768-pixel native size) is
   bilinearly interpolated with aligned corners to the actual patch grid.
4. **ViT blocks.** Pre-LayerNorm blocks with a fused QKV projection, a 2D vision RoPE, full
   (non-causal) attention and a GELU MLP. The vision RoPE rotates the pairs `(i, i + d/2)`; the
   first quarter of the pairs uses the patch row and the second quarter the patch column, each with
   frequencies `10000^(-k/(d/4))` — ggml's `GGML_ROPE_TYPE_VISION` with independent sections.
5. **Merger.** A final LayerNorm, then groups of four consecutive tokens (one merge block) are
   concatenated and projected by a two-layer GELU MLP to the text model's width. Qwen3-VL also runs
   a "deepstack" merger (LayerNorm over the concatenated 4 × dim vector, then the MLP) on the output
   of layers 5, 11 and 17.

The Qwen2.5-VL projector (`tools/mtmd/models/qwen2vl.cpp`) uses 14-pixel patches (so sides are
multiples of 28) and differs in the blocks: RMSNorm without bias, separate Q/K/V projections with
bias, a SiLU-gated MLP, and neither a learned position embedding nor a patch bias. It also uses
**window attention**: the merged tokens are grouped into windows of 112 pixels (4 × 4 merged
tokens), the patches are reordered so that each window is contiguous, every layer except each
`n_wa_pattern`-th (layers 7, 15, 23 and 31 on the 3B) attends only within its window, and the merged
output is put back in grid order before it reaches the text model.

Every output token is `[main, deepstack_0, deepstack_1, deepstack_2]`, each part as wide as the text
model. The mmproj weights are expanded to F32 when the projector is loaded (about 1.7 GB of heap for
the Qwen3-VL-4B projector), because there is no SIMD F16 kernel; projections run as tiled multi-token
matmuls on `MatmulPool`.

### Prompt layout and multi-axis RoPE

`LLMEngine.generate` places one marker, `<|vision_start|><|image_pad|><|vision_end|>`, per image
before the user text (or, in raw mode, wherever the caller put it). Each `<|image_pad|>` is replaced
by the image's tokens, which are prefilled as embeddings; the text runs around them go through the
normal batched prefill.

The Qwen-VL text models use multi-axis RoPE: every rotated pair belongs to one of four sections
(temporal, height, width, extra). Qwen2-VL assigns the sections sequentially (`MROPE`); Qwen3-VL and
Qwen3.5 interleave them (`IMROPE`). `RoPE.mropeSectionMap` reproduces ggml's `ggml_mrope_cache_init`
exactly. Positions follow llama.cpp `mtmd`:

- A text token at logical position `p` rotates with `(t, h, w, e) = (p, p, p, 0)`. The fourth
  position is 0, not `p`: with Qwen3-VL's sections `[24, 20, 20, 0]` two of the 64 pairs fall into
  the extra section and are therefore not rotated for text. For the other layouts every pair is
  covered by t/h/w, and the rotation equals plain NEOX RoPE.
- Image token `i` of an image that starts at logical position `p0` on a merged grid with `nx`
  columns rotates with `(p0, p0 + i / nx, p0 + i % nx, 0)`.
- The image advances the logical position by `max(nx, ny)`, not by its token count.

The KV cache stays sequential; only the rotation angles depend on these positions.
`MRopePositions` computes them from the KV index, so decoding after the prompt needs no special
handling. For Qwen3-VL, the three deepstack vectors of an image token are added to the residual
stream after layers 0, 1 and 2 (`InferenceEngine.prefillEmbeddings`).

### Entry points

- CLI: `--mmproj <file>` and `--image <file>` (repeatable), with `--prompt`.
- Java: `LLMEngine.loadVisionProjector(Path)` and `GenerationRequest.Builder.image(byte[])`.
- Web: `/api/models/load` accepts `"mmproj"`; `/v1/chat/completions` accepts `image_url` parts with
  `data:` URLs, and `/v1/messages` accepts `image` blocks with a `base64` source. Remote URLs are
  rejected, so the server never fetches content on a client's behalf.

### Measurements

On the shared 8-vCPU reference VM, with other workloads running, a 640 × 480 image (300 merged
tokens) took 26 s to encode with the Qwen3-VL-4B projector and 42 s with the Qwen3.5-4B projector;
the same image became 391 tokens and took 52 s with the Qwen2.5-VL-3B projector (32 layers at width
1280). All three models described the llama.cpp test image (the New York Times front page of
21 July 1969) correctly, including the headline "MEN WALK ON MOON" or the date. The OpenAI and
Anthropic endpoints were tested with the same image through `/api/models/load` with `mmproj`.

## Text-to-speech (`it.denzosoft.llmplayer.tts`)

### Files

Qwen3-TTS in the llama.cpp layout, for example `ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF`: the talker
GGUF (`general.architecture = qwen3tts`) and its mmproj (`clip.gen.audio.projector_type =
qwen3tts_gen`). Other Qwen3-TTS GGUF layouts on Hugging Face (`qwen3-tts`, `qwen3tts_tokenizer`, and
so on) belong to other runtimes and are not supported.

### Pipeline (`Qwen3Tts`, `Qwen3TtsCodec`)

The implementation follows `tools/mtmd/mtmd-helper-gen.cpp` and `tools/mtmd/models/qwen3tts-gen.cpp`.

1. **Prompt.** The text is wrapped as `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`
   and tokenized once. Each prompt position is a *sum of two embeddings* from the talker's token
   table, which holds both the text vocabulary and the 3072 codec entries: the role tokens alone,
   then `tts_pad` plus `codec_think`, `codec_think_bos`, the language token and `codec_think_eos`;
   `tts_text_bos + codec_pad`; every text token plus `codec_pad`; `tts_text_eod + codec_pad`; and
   `tts_pad + codec_bos`.
2. **Talker.** A Qwen3-VL-style decoder (IMROPE, text positions only) whose output head has 3072
   rows. Each step samples the codebook-0 code of one 80 ms frame (temperature 0.9, top-k 50,
   repetition penalty 1.05, the special codec ids listed in `tokenizer.ggml.suppress_tokens`
   masked) and stops at `codec_eos_token`.
3. **Code predictor.** From the talker's output-normed hidden state and the codebook-0 code, a
   5-layer Qwen3-style transformer (width 1024, 16/8 heads of 128) samples the 15 acoustic codebooks
   one per position, each with its own head. The sum of the 16 code embeddings plus `tts_pad` is the
   talker's next input.
4. **Code2wav.** The 16 codes of each frame are decoded to 24 kHz PCM: RVQ codebook lookup and
   projection to 512 dimensions, a causal convolution, an 8-layer transformer whose attention is
   limited to a 72-frame window, two ×2 upsamplers (transposed convolution plus a ConvNeXt block),
   and a DAC decoder (Snake activations, transposed convolutions with strides 8, 5, 4 and 3, dilated
   residual units) — 1920 samples per frame. Decoding runs in 72-frame chunks and carries the
   causal state (convolution left context, transposed-convolution tails, transformer K/V and
   position) between chunks, so the result equals decoding the whole sequence at once.

The speaker encoder in the mmproj (an ECAPA-TDNN for voice cloning from a reference recording) is
not implemented. Without a reference, the Base checkpoint speaks with the voice it produces
unconditioned.

### Entry points

- CLI: `--tts out.wav --tts-lang it --mmproj mmproj.gguf --model talker.gguf --prompt "text"`.
  `--max-tokens` limits the number of frames (80 ms each).
- Java: `Qwen3Tts.load(talker, mmproj)`, `synthesize(text, options, listener)` and
  `WavWriter.write(pcm, 24000, path)`.

Languages: `zh`, `en`, `de`, `it`, `pt`, `es`, `ja`, `ko`, `fr`, `ru`.

### Measurements and validation status

On the shared reference VM under heavy load, the 1.7B Base model produced a 3.5-second Italian
sentence in 20 s (0.17× real time), of which 6 s were spent in the decoder. The talker ended the
utterance on its own with `codec_eos` after 44 frames. The waveform was checked numerically only:
silence at both ends, a syllable-rate energy envelope with pauses between words, 56 % voiced frames
with a median pitch of 146 Hz, and no clipping. It has not been verified by listening, so
intelligibility still needs a human check.
