# Inference engine dispatch — full reference

This document holds the per-engine detail for the nine forward-pass implementations in
`it.denzosoft.llmplayer.inference`. `CLAUDE.md` carries only the one-line dispatch summary and
points here. Read the relevant section before modifying an engine, because most of the
architecture-specific behaviour below is not derivable from the code without knowing which
upstream llama.cpp implementation it mirrors.

The engine is selected in `LLMEngine.load()` and `ModelLoader`, based on
`ModelConfig.architecture()` and, for a few architectures, on metadata such as `expertCount()`.

## 1. Standard (`InferenceEngine`)

Handles Llama, Qwen2, Qwen3, SmolLM3, GLM4, Gemma 2, Gemma 3, Phi-3/4, Mistral3, Command-R,
OLMo2, Falcon3, GPT-OSS, Granite 3.3, ERNIE 4.5, Qwen2.5-VL and Qwen3-VL (text backbones),
Hunyuan dense, Nanbeige, and Spark2.5.

The forward pass runs `TransformerBlock` → `Attention` (GQA with optional QK-norm and bias,
sliding window, dual RoPE) followed by `SwiGLUFFN` (GeGLU for Gemma). Gemma 2 and Gemma 3 use
pre- and post-attention/FFN norms plus embedding scaling. Granite 3.3 applies four custom scaling
factors (embedding, attention, residual, logit) read from the GGUF `granite.*` metadata keys.

ERNIE 4.5 (`ernie4_5`) is a plain dense transformer that maps directly onto this path: RMSNorm,
GQA with an explicit `head_dim=128` taken from `attention.key_length` rather than the implied
`embd/heads=64`, RoPE in NORM mode with θ=500000, SwiGLU, and tied embeddings. It uses a custom
`<|begin_of_sentence|>` "User:/Assistant:" chat template and runs the full `CudaForwardPass` on GPU.

**Qwen2.5-VL and Qwen3-VL** (`qwen2vl`, `qwen3vl`) are aliased to `QWEN2` and `QWEN3`: their
decoders are the plain text models. What differs is RoPE. Both declare `rope.dimension_sections`,
which `ModelConfig` exposes as `ropeSections()`, and `Attention` then rotates with multi-axis RoPE
(`RoPE.applyMrope`) instead of 1D NEOX. For text tokens the three axes are equal and the fourth is
0 (llama.cpp `llm_graph_input_pos`); with Qwen3-VL's interleaved sections `[24, 20, 20, 0]` that
leaves two pairs unrotated, so the text path is *not* exactly NEOX. Image tokens and the
multimodal prompt (for all three Qwen vision families) are described in
[`vision-and-tts.md`](vision-and-tts.md). The Qwen3-TTS talker
(`qwen3tts`) is the same decoder and is also loaded as `QWEN3`, but only through
`it.denzosoft.llmplayer.tts.Qwen3Tts`, because its output head covers only the 3072 codec entries.

**Hunyuan dense** (`hunyuan-dense`, e.g. Hy-MT2 1.8B/7B) is Llama-shaped with per-head Q/K RMSNorm,
but applies the norm **after** RoPE (llama.cpp `hunyuan-vl.cpp`), which `ModelConfig.qkNormAfterRope()`
selects in `Attention.normAndRope`. RoPE is NEOX. The NTK-aware "alpha" scaling
(`base · α^(d/(d−2))`) is applied when `rope.scaling.alpha` is present; current converters bake it
into `rope.freq_base` instead. The pre-tokenizer is llama.cpp's multi-regex `hunyuan-dense` set.

**Nanbeige** (`nanbeige`, e.g. Nanbeige4.2-3B) has looped depth: the GGUF stores 22 physical layers
that run twice (`num_loops = 2`) with shared weights. `ModelConfig.blockCount()` is the logical
count (44), so every logical layer gets its own KV slot, and `ModelLoader.loadWeights` points the
second loop's entries at the first loop's `TransformerLayerWeights`. Between loops the residual
stream is replaced by its output norm (`ModelConfig.isLoopBoundary`) unless
`skip_loop_final_norm` is set. RoPE is NORM at θ=7e7. The tokenizer is SentencePiece whose BOS is
`<|im_start|>`; `SpecialTokens` therefore does not prepend it, because the ChatML template opens the
first turn itself.

**Spark2.5** (`spark2_5`, Spark-X2.5 1.7B/4B) interleaves three sliding-window layers (window 512)
with one full-attention layer, from `attention.sliding_window_pattern`. The two layer kinds use
different RoPE: sliding-window layers rotate every dimension at `rope.freq_base_swa` (1e4), full
layers rotate `rope.dimension_count` (a quarter) at the main θ (5e6); the engine builds the second
table as the local RoPE. The attention output is multiplied per head by
`sigmoid(attn_gate · x)` (`TransformerLayerWeights.attnGate`) before the output projection, the FFN
is GELU-gated, Q/K/V is one fused tensor, and there is no QK-norm. Batched prefill covers the fused
QKV (one multi-token matmul, split per token) and applies the gate per token from the chunk's normed
inputs; on a 1592-token prompt that halved the prefill time (191 → 93 s end to end).

The GPU-resident passes (`CudaForwardPass`, `GpuForwardPass`, `BatchedCudaForwardPass`) decline
Hunyuan, Spark2.5 and looped Nanbeige through `ModelConfig.requiresCpuLayerPath()`; their tensors can
still live on the GPU for individual matmuls.

## 2. DeepSeek2 (`DeepSeek2InferenceEngine`)

Handles DeepSeek2 and GLM-4.7-Flash, which also declares the `deepseek2` GGUF architecture. Uses
Multi-Head Latent Attention (MLA) together with a MoE FFN that includes a shared expert. The
leading blocks use a dense SwiGLU FFN instead of the MoE path.

## 3. Qwen3 MoE (`Qwen3MoEInferenceEngine`)

Handles Qwen3-Coder-30B-A3B and similar models, and also the Llama 4 MoE and GLM4 MoE variants
(dispatched in `ModelLoader`). Standard GQA attention with QK-norm plus a MoE FFN with a shared
expert; leading blocks use a dense SwiGLU FFN. Llama 4 uses the Llama chat template.

GLM4 MoE (`glm4moe`: GLM-4.5, GLM-4.5-Air, GLM-4.6, GLM-4.7) is aliased to `GLM4` and takes this
engine through the `GLM4 && expertCount > 0` branch. It follows llama.cpp `glm4-moe.cpp`:

- Q/K/V projections carry a bias, and the QK-norm is optional (present on the 355B model).
- RoPE is NEOX and partial (half of each head), unlike dense `glm4`, which is NORM.
- Routing uses sigmoid gating. The top-K experts are chosen on `sigmoid(logits) + exp_probs_b`,
  but weighted by the unbiased probabilities, sum-normalised when `expert_weights_norm` is set and
  multiplied by `expert_weights_scale`. This is the `sigmoidRouting` branch of `routeExperts`,
  which decode and batched prefill share.
- `block_count` includes the NextN/MTP layers used for speculative decoding. `ModelConfig`
  subtracts `nextn_predict_layers`, so only the trunk runs.
- The GLM-4.5 chat template is a hybrid reasoning template. Without `--thinking`, `ChatTemplate`
  appends `/nothink` to user turns and an empty `<think></think>` to the generation prompt, as
  the template does for `enable_thinking=false`.

The logits of a tiny random `glm4moe` GGUF match an independent re-implementation of the llama.cpp
graph to five decimal places, including a copy with a non-zero `exp_probs_b` that changes the
selected experts. A real GLM-4.5-Air has not been run.

## 4. Qwen3.5 (`Qwen35InferenceEngine`)

Hybrid DeltaNet plus full-attention architecture. It alternates Gated DeltaNet (linear
attention / SSM) layers and standard GQA layers in a 3:1 ratio (`full_attention_interval=4`).

DeltaNet layers maintain a recurrent state `S` with the update rule
`S_new = alpha*S + beta*outer(k, v - alpha*S^T@k)` and produce output `o = S^T_new @ q`.

Full-attention layers use a **packed Q+gate projection**: `wq` outputs interleaved
`[Q_h0, gate_h0, Q_h1, gate_h1, ...]`, which must be deinterleaved into separate Q and gate arrays
before use. The gate is then applied as `sigmoid(gate) * attn_output`.

Both layer types include a short conv1d (width 4) on Q and K, and both use QK-norm. State is
maintained per layer in `Qwen35State`.

## 5. Nemotron-H and Granite Hybrid (`NemotronHInferenceEngine`)

Hybrid Mamba-2 SSM plus GQA attention plus squared-ReLU FFN. Unlike a standard transformer these
are three **distinct layer types** rather than components combined in every layer. The per-layer
arrays in the GGUF metadata (`head_count_kv[]`, `feed_forward_length[]`) determine the type of each
layer: `kvHeads > 0` means attention, `ffnLength > 0` means FFN, and both zero means Mamba-2.

Mamba-2 uses SSD (Structured State Space Duality) with a state shaped
`[nheads][headDim][stateSize]`, a causal conv1d with bias and SiLU, and a grouped RMSNorm with gate
(`norm_before_gate=False`). The GPU-resident forward pass is `NemotronHCudaForwardPass`, with
dedicated kernels `mamba2_scan.cu`, `mamba2_dt_softplus.cu`, `mamba2_gate_norm.cu`, and `sqrelu.cu`.

### Granite Hybrid (dense)

Fully GPU-accelerated as of v1.11.0-dev (2026-04-15). All four scale factors (embedding, logit,
residual, attention) are wired on GPU via `scale_inplace`, `accumulate`, and saxpy, and the
integrated SwiGLU FFN inside the Mamba and attention layers (`lw.ffnUp() != null`) runs on GPU via
`runIntegratedFFN()` with a fused RMSNorm, Q8_1 quantization, gate/up/down dp4a, and `silu_mul`.

This was validated bit-equivalent to CPU within ±2 ULP when dp4a is disabled. With dp4a enabled the
1–15% per-layer divergence is the expected Q8_1 quantization noise, the same as every other
dp4a-accelerated model. CUDA graph capture works on this path.

### Granite Hybrid MoE

Models such as `granite-4.0-h-tiny` (64 experts, top-6, shared expert, softmax routing) run the
dense attention and Mamba parts on the **CPU engine path**, because
`NemotronHCudaForwardPass.isSupported()` returns false when `expertCount > 0`. The expert FFN is
still GPU-accelerated via `GraniteExpertGpu` (2026-06-07): `runIntegratedMoEFFN()` performs the
router softmax, top-K selection, and renormalization on CPU, then `GraniteExpertGpu.computeMoE()`
runs the routed and shared experts on GPU.

Each expert's 2D slice of the 3D `ffn_*_exps` tensor is multiplied through an **offset weight
pointer** (`getGpuWeights() + e·(getWeightsBytes()/expertCount)`), reusing each tensor's own FP32
matmul kernel. No new kernel was required and it works for both Q4_K and Q6_K. The path also uses
`silu_mul`, `saxpy`, `fill_zero`, and `accumulate`. Measured roughly 3–4 → 9 tok/s (about 2.5×) at
PPL 0.98.

Experts are loaded GPU-resident when the GPU is active, otherwise they fall back to CPU `dot()`.
The routed-expert FFN size falls back to `feed_forward_length` when `expert_feed_forward_length` is
absent; the shared-expert size comes from `expert_shared_feed_forward_length`.

This work is deliberately contained — it does **not** touch the validated dense
`NemotronHCudaForwardPass`.

## 6. Gemma 4 and Gemma 3n (`Gemma4InferenceEngine`)

Handles the full Gemma 4 lineage: the PLE (Per-Layer Embeddings) models E2B and E4B, the dense 12B
and 31B, and Gemma 3n. **All** `GEMMA4` and `GEMMA3N` models route here — `LLMEngine.load` no longer
gates on `embedding_length_per_layer_input > 0`, and the dense path simply runs with `hasPle=false`.
The standard `InferenceEngine` cannot handle Gemma 4's dual head size, per-layer KV, V-norm, or
output scale, so dense Gemma 4 must never fall through to it.

Two sub-paths are dispatched by whether `Gemma3nWeights` has AltUp tensors loaded.

**Gemma 3n** (arch=GEMMA3N) runs full AltUp (four parallel activation streams with learned router,
predict, and correct coefficients), Laurel (a low-rank residual branch), Gaussian top-k activation
sparsity on the first 10 FFN layers, and PLE, in `forwardLayerGemma3nInner` and `forwardLayerAltup`.
K-norm uses the `(1+w)` adjustment.

**Gemma 4** (arch=GEMMA4, both PLE and dense) uses a single `forwardLayer` path with no AltUp,
Laurel, or sparsity. Following llama.cpp `src/models/gemma4.cpp` it applies V-norm
(`ggml_rms_norm` on V, no learnable scale), a per-layer `layer_output_scale.weight` scalar applied
as a final `cur *= scale` (loaded unconditionally, since it is present on dense variants where
`pleDim == 0`), K-norm stored as final values with no `(1+w)`, and **alternative attention** on the
global (full-attention) layers. Those global layers ship no `attn_v`, so `V = raw K projection`;
`config.layerKvHeads()` returns the per-layer KV count, for example 1 on the every-sixth global
layer versus 8 on the SWA layers.

Common to all Gemma 4 variants: per-layer token embedding and projection tensors (PLE models only),
a per-layer KV head count (`attention.head_count_kv` may be a per-layer INT32 array on the 12B), a
per-layer FFN width (`feed_forward_length` may be a per-layer INT32 array on E2B/E4B — the
"double-wide MLP", for example 6144 then 12288), dual head size (SWA=256, full=512 per layer), a
shared KV cache where layers at or beyond `blockCount - sharedKvLayers` reuse earlier KV (0 for the
12B), dual RoPE (SWA θ=10K, full θ=1M with proportional frequency factors), attention scale 1.0, and
logit soft-capping. State lives in `Gemma4State` with per-layer-sized KV.

> **Regression-prone.** GGUF encodes some of these hyperparameters as a **UINT32 scalar**
> (E2B/E4B `head_count_kv`, E4B `feed_forward_length`) and others as an **INT32 array**
> (12B `head_count_kv`, E2B/E4B `feed_forward_length`). `GGUFMetadata.getIntArray` returns non-null
> only for INT32 arrays, which is why `ModelConfig` falls back to the scalar when the per-layer
> array is absent. Removing that fallback silently breaks a subset of the family.

## 7. LFM2 (`LFM2InferenceEngine`)

Liquid Foundation Model 2 (`lfm2`). A hybrid of gated **short-convolution** mixers and GQA
**attention** mixers, chosen per layer by `attention.head_count_kv[i]` — 0 means a conv layer,
greater than 0 means an attention layer (10 conv and 6 attention layers on the 1.2B). Each mixer is
followed by a SwiGLU FFN.

Per layer: `prev=x; n=RMSNorm(x, operator_norm); blk=conv-or-attn(n); x=prev+blk;
x+=SwiGLU(RMSNorm(x, ffn_norm))`.

The short-conv block mirrors `build_shortconv_block` in llama.cpp `lfm2.cpp`: `in_proj` then split
into `[b|c|x]`, compute `bx=b*x`, apply a depthwise causal conv1d (width `shortconv.l_cache`=3,
rolling state 2) over `bx`, then `y=c*conv_out` and `out_proj`.

Attention layers use per-head QK-norm before RoPE, with RoPE in NEOX mode at θ=1e6. The final norm
tensor is `token_embd_norm` and the output is tied to `token_embd`. ChatML template, BPE tokenizer.

**LFM2-MoE** (`lfm2moe`, e.g. LFM2.5-8B-A1B) is aliased to `LFM2`. Layers at or after
`leading_dense_block_count` replace the dense SwiGLU with routed experts (llama.cpp `lfm2.cpp`
`build_moe_feed_forward`): sigmoid router scores (`expert_gating_func = 2`), top-k selection on the
scores plus `exp_probs_b`, the selected *unbiased* scores normalised to sum 1, then
`sum_k w_k · down_k(silu(gate_k x) · up_k x)`. The experts run through per-expert `ExpertViews`
(no copy) with the rows of all selected experts split across `MatmulPool`, which doubled decode
speed over a per-expert loop. `LFM2CudaForwardPass` declines MoE models.

GPU support is the dedicated `LFM2CudaForwardPass`, which is per-layer GPU-resident: conv,
attention, RoPE, QK-norm, and SwiGLU all run as kernels, reusing `conv1d_short`, `attention_full`,
`rope_apply`, `rmsnorm_per_head`, `silu_mul`, and `elementwise_mul`. Matmuls are FP32 and there is
no graph capture yet. Measured roughly 33 → 56 tok/s (+70%) with output bit-identical to CPU. Gated
by `isSupported` with per-tensor fallback. Supporting types are `LFM2State`, `LFM2LayerWeights`, and
`LFM2Weights`, loaded by `ModelLoader.loadLFM2Weights`.

## 8. Falcon-H1 (`FalconH1InferenceEngine`)

TII Falcon-H1 (`falcon-h1`). A **parallel** hybrid: every layer runs both a GQA attention path and a
Mamba-2 SSM path over the *same* pre-normed input (a shared `attn_norm`), sums the two outputs, and
then applies a SwiGLU FFN:

```
n = RMSNorm(x, attn_norm)
x += attn(n) + mamba2(n)
x += SwiGLU(RMSNorm(x, ffn_norm))
```

This differs from Nemotron-H, which has *separate* layer types rather than parallel branches.

The Mamba-2 block mirrors the Nemotron-H SSD math: conv1d with bias, then SiLU, then a scan with
`dA=exp(dt·A)` where `A` is stored as `-exp(A_log)`, then `y += D·x`, then the gate `y *= silu(z)`,
and finally a grouped RMSNorm **only when `ssm_norm` is present** — it is absent on the 0.5B and
present on the 1.5B.

Attention has **no** QK-norm and uses RoPE in NEOX mode at θ=1e11. `head_dim` comes from
`attention.key_length` (64 for the 0.5B, 128 for the 1.5B) and the SSM `nheads` from
`ssm.time_step_rank`. The HF channel and attention **multiplier scalars are baked into the GGUF
weights at conversion time**, so the current llama.cpp forward pass — and ours — applies none of
them. `ffn_norm` is stored without a `.weight` suffix. ChatML template, BPE tokenizer, untied output.

GPU support is the dedicated `FalconH1CudaForwardPass`, per-layer GPU-resident: attention and
Mamba-2 both run as kernels over the shared normed input, are summed, and then feed SwiGLU. It
reuses `mamba2_scan`, `conv1d_short`, `mamba2_dt_softplus`, `mamba2_gate_norm`, `attention_full`,
and `rope_apply`, taking a gate-only path when `ssm_norm` is absent (0.5B) and `mamba2_gate_norm`
when present (1.5B). Matmuls are FP32 with no graph capture yet. Measured roughly 12 → 40–42 tok/s
(about 3×) at PPL 0.98–0.99. Gated by `isSupported` with per-tensor fallback. Supporting types are
`FalconH1State`, `FalconH1LayerWeights`, and `FalconH1Weights`, loaded by
`ModelLoader.loadFalconH1Weights`.

## 9. BailingMoE3 (`BailingMoE3InferenceEngine`)

Ling 3.0 (`bailingmoe3`, e.g. Ling-3.0-tiny 7.9B-A1.3B), following llama.cpp `bailingmoe3.cpp`. The
engine loads its own tensors from the GGUF (`ModelLoader` skips the weight step). Two layer types,
chosen by `attention.head_count_kv[i]`:

- **KDA (Kimi Delta Attention)**, kv heads 0. The Q, K and V projections each pass a causal
  depthwise short convolution (width `ssm.conv_kernel`) and SiLU; Q and K are L2-normalised per
  head. A per-channel log-decay `g = kda.gate_lower_bound · sigmoid(A_h · (f_a·x + dt_b))` and a
  per-head `β = sigmoid(w_β·x)` drive the gated delta rule on a `d × d` state per head, with
  `S[i][j]` indexed by key channel `i` and value channel `j` (ggml `gated_delta_net`):
  `S ← diag(exp g) S`, `δ = β (v − Sᵀk)`, `S ← S + k δᵀ`, `o = Sᵀq / √d`. The output is RMS-normed per
  head and multiplied by `sigmoid(g_a·x)`.
- **Gated MLA**, kv heads 1. Q-LoRA (`q_a`, norm, `q_b`); the KV latent (`kv_lora_rank`) plus one
  shared rope key from `kv_a_mqa`, RoPE in NORM mode on the 64 rope dimensions. The query's
  non-rope part is absorbed into the latent space with `k_b`, so the cache stores only the normed
  latent and the rope key per position; the attention output is expanded with `v_b` and multiplied
  per head by `sigmoid(attn_gate·x)`. The score scale is `1/√key_length_mla`.

The first `leading_dense_block_count` layers use a dense SwiGLU FFN; the others a routed MoE plus one
shared expert. Routing uses sigmoid scores with a selection-only bias and is **group-limited**: the
128 experts form 8 groups, each group is scored by the sum of its two best biased scores, the best
4 groups are kept, and the top 8 experts are chosen among them. The selected unbiased scores are
normalised (`expert_weights_norm`) and scaled by `expert_weights_scale` (2.5).

Prefill runs token by token. Teacher-forced perplexity on a mixed prose and Java sample is 2.41,
lower than Qwen3-VL-4B's 2.92 on the same text.

## Tokenizer and chat template dispatch

`TokenizerFactory` reads `tokenizer.ggml.model` from the GGUF metadata. The values `"gpt2"` and
`"bpe"` select `BPETokenizer`, which uses the merge table and Llama 3 style pre-tokenization regex.
The value `"gemma4"` also selects `BPETokenizer` but with `useGpt2ByteMapping=false`, because the
Gemma 4 vocabulary uses SentencePiece-style `▁` (U+2581) for spaces and `<0xHH>` byte-fallback
tokens rather than GPT-2 byte mapping. Anything else selects the score-based
`SentencePieceTokenizer`.

Chat formatting is architecture-specific in `ChatTemplate`, with distinct templates for Llama
(`<|start_header_id|>`), Qwen2/3 and Qwen3MoE (`<|im_start|>`), GLM4 (`[gMASK]<sop>`), DeepSeek2
(`User: ... Assistant:`), Phi-3/4 (`<|user|>`), Mistral3 (`[INST]`), Gemma 4 (`<|turn>...<turn|>`),
and OLMo2 (`<|user|>...<|assistant|>`).

`BPETokenizer.setPreTokenizer` selects a llama.cpp multi-regex pre-tokenizer from
`tokenizer.ggml.pre` for the models whose splitting the default pattern gets wrong: `hunyuan-dense`,
`spark2_5` and `bailingmoe`/`bailingmoe2`. The regexes are applied in sequence, each splitting every
piece produced by the previous one, as llama.cpp's `unicode_regex_split` does.

The templates added with these models: Hunyuan (`<｜hy_User｜>…<｜hy_Assistant｜>`), Spark2.5
(`<｜start▁of▁sentence｜><|User|>…<｜end▁of▁sentence｜>` with `<think>`/`</think>` after `<|Bot|>`),
Ling (`<role>HUMAN</role>…<|role_end|><role>ASSISTANT</role>` with `detailed thinking on|off` in the
system turn), and Nanbeige (ChatML with its default system prompt and `<think>` handling). A Qwen3
checkpoint whose template never mentions `<think>` (Qwen3-VL-Instruct, Qwen3-2507-Instruct) no longer
receives the empty think block.

**Olmo 3 detection**: when the `chat_template` metadata contains `<|im_start|>`,
`ChatTemplate.isOlmo3ChatML` is set to true and the OLMo2 format method switches to ChatML output
(`<|im_start|>user\n...<|im_end|>`).
