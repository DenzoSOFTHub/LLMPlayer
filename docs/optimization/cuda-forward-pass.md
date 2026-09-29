# CUDA GPU-resident forward pass — full reference

This document holds the detail on the GPU-resident forward passes, the dp4a integer path, the CUDA
kernel design conventions, and the cuBLAS alternative. `CLAUDE.md` carries the summary and the
hard invariants; everything below is reference material for when you are actually editing a kernel
or a forward pass.

Related: [`jvm-flags.md`](jvm-flags.md) for the tuning properties that switch these paths on and off,
and [`llamacpp-comparison.md`](llamacpp-comparison.md) for the rolling tok/s gap and a journal of
optimization attempts with their measured outcomes.

## Two execution modes

Both modes keep activations on the GPU between transformer layers, reducing CPU↔GPU sync points.

1. **Per-layer mode** (`CudaForwardPass.forwardLayer()`): each layer runs entirely on GPU —
   RMSNorm, QKV, RoPE, KV cache, attention, Wo, FFN norm, gate/up, SiLU, down. It synchronizes only
   at the `uploadX` and `downloadX` boundaries.
2. **CUDA graph mode** (`CudaForwardPass.forwardGraph()`): captures every kernel launch into a CUDA
   graph on the first token, then replays it with a single `cuGraphLaunch` on subsequent tokens.
   Dynamic values (position, seqLen) are read from the GPU-resident `tokenParams` buffer. Enabled by
   default; disable with `-Dcuda.nograph=true`.

`CudaForwardPass` captures two graphs: all layers plus the final norm and output projection (a
decode token), and all layers alone (`forwardGraphLayers()`). The second serves prefill tokens
whose logits are discarded — they used to replay the full graph, computing the largest weight's
matmul and downloading 0.5 MB of logits per prompt token, which cost about a third of the
per-token prefill — and the GPU prefix of a partial offload, which used to launch layer by layer
through reflection. A failed capture is ended, its partial graph destroyed, and after two failures
graphs are abandoned for the pass instead of being retried on every token. `LFM2CudaForwardPass`,
`FalconH1CudaForwardPass` and `Gemma4CudaForwardPass` gained the same two graphs; their first token
runs per-layer, because weights upload and kernels compile lazily on first use and neither
`cuMemAlloc` nor module loading is allowed inside a capture (it fails with error 900).

The central design constraint is **zero-allocation hot paths**. All kernel parameter buffers
(`ParamBuffer`) and matmul launch descriptors (`MatmulLaunch`) are pre-allocated in the constructor.
`forwardLayer()` only writes parameter values in place and launches kernels — it must never
allocate.

## Internal split (v1.13.0)

`forwardLayer` is a thin wrapper around two private helpers:

- `forwardAttentionPart(layerIdx, position)` — steps 1–6c: attention norm, QKV, QK-norm, Granite
  scale, RoPE, KV cache, attention, Wo, post-attention norm.
- `forwardFFNPart(layerIdx)` — steps 7–10c: FFN norm, gate/up, `silu_mul`, down, post-FFN norm.

A public `forwardAttentionOnly(state, layerWeights, layerIdx, position, attention)` is still exposed,
but the MoE engines do not use it: `CudaForwardPass` needs `ModelWeights` and builds FFN launches
for every layer, which MoE weight classes do not provide. The Qwen3-MoE family uses the dedicated
`MoeAttentionCudaPass` instead (see below).

The three per-layer implementations that had drifted apart (the normal path, a profiled copy that
skipped QK-norm and the Granite scaling, and a graph-capture copy that used FP32 kernels for the
non-Q4_K gate/up) are now one: `forwardAttentionPart` and `forwardFFNPart` with timing marks that
only exist under `-Dcuda.profile=true`, and a position of `-1` while capturing.

## Supported architectures

Running the full `CudaForwardPass`: Llama, Qwen2, Qwen3, Falcon3, OLMo2 (including Olmo 3),
Mistral3, Gemma 2 and Gemma 3 (post-norm), Phi-3/4 (packed FFN via `split_gate_up.cu`), Granite 3.3
(with scaling factors), ERNIE 4.5 (dense, explicit head_dim, RoPE NORM), and Nemotron-H / Granite
Hybrid via `NemotronHCudaForwardPass` (Granite Hybrid uses the full integrated-FFN path with all
scale factors on GPU). Per-head QK-norm (Qwen3) is handled by `rmsnorm_per_head.cu`.

Dedicated per-layer GPU-resident forward passes exist for **LFM2** (`LFM2CudaForwardPass`),
**Falcon-H1** (`FalconH1CudaForwardPass`), and **Gemma 4** (`Gemma4CudaForwardPass`), all added
2026-06-07.

**MoE.** The Qwen3-MoE family (Qwen3-MoE, Llama 4 MoE, GLM4 MoE) runs the attention half of every
layer on the GPU through `MoeAttentionCudaPass`, with the router and the experts on the CPU or in
the GPU expert cache. Granite Hybrid MoE runs its experts through `GraniteExpertGpu` while the dense
parts stay on the CPU engine with per-tensor GPU matmuls. DeepSeek2 (MLA), Ling 3.0, LFM2-MoE and
GPT-OSS attention still use per-tensor matmuls.

**Not supported on GPU**, falling back to per-tensor matmul or CPU: Gemma 3n, whose AltUp path runs
on CPU via `Gemma4InferenceEngine`, and the layer shapes only the CPU `TransformerBlock` implements
(`ModelConfig.requiresCpuLayerPath()`).

## Gemma 4 CUDA forward pass (`Gemma4CudaForwardPass`)

Keeps PLE, dual head size (SWA 256 / full 512), shared KV, GeGLU, and the per-layer output scale all
GPU-resident. It needed exactly one new kernel, `gelu.cu`; V-norm is done via `rmsnorm_per_head`
with a ones-vector, and Gemma's attention scale of 1.0 is achieved by pre-scaling Q by √headSize.
The embedding scale, the PLE precompute, and the logit soft-cap remain on CPU. Measured about 4.5×
over CPU (roughly 4 → 18 tok/s) at PPL 1.00.

**Dense Gemma 4 (12B/31B)** also runs here. The pass consumes the per-layer KV head count and the
per-layer FFN width, handles the global layers' alternative attention (no `attn_v`, so `V = raw K`,
copied via `copyBufferDtoD` before K-norm and RoPE), and supports **first-N-layer partial offload**:
it counts the leading GPU-resident layers via `getGpuLayerCount`, allocates GPU buffers and KV only
for those, and `Gemma4InferenceEngine.forwardGpu` runs layers `0..N-1` on GPU, then `downloadX` and
CPU `forwardLayer` for layers `N..blockCount-1`, then re-uploads for the GPU-resident output
projection.

On the 6 GB RTX 4050 the 12B Q4_K_M (7 GB) places 37 of 48 layers — the KV-aware budget
auto-enables FP16 KV — at PPL 0.91. Throughput stays close to CPU because the 11 CPU layers plus the
per-token sync dominate a bandwidth-bound 12B at batch 1.

## Qwen3.5 CUDA forward pass (`Qwen35CudaForwardPass`)

Dedicated GPU-resident forward pass for the hybrid DeltaNet plus attention architecture. It handles
both the DeltaNet layers (three quarters) and the full GQA attention layers (one quarter), with CUDA
graph support.

DeltaNet-specific kernels:

- `deltanet_fused.cu` — the mega-kernel: recurrence, per-head RMSNorm, `SiLU(gate)`, and the gate
  multiply, using a transposed S matrix `[dV][dQK]` for coalesced access and a parallel L2 norm via
  warp-shuffle reduction.
- `conv1d_silu.cu` — fused causal conv1d plus SiLU.
- `alpha_beta_gates.cu` — computes the alpha (exponential decay) and beta (sigmoid) gates from the
  projections.
- `deinterleave_q_gate.cu` — splits the packed Q+gate projection for the attention layers.
- `sigmoid_elementwise_mul.cu` — attention output gating, `xb2 *= sigmoid(gate)`.

Optimizations on this path: a fused FFN gate+up Q4_K kernel (reusing
`matmul_q4_k_fused_gate_up.cu`); GPU-side argmax (`forwardGraphArgmax()`, `forwardFinalArgmax()`)
which downloads 4 bytes instead of the full logit vector; the embedding kept on CPU, freeing roughly
500 MB of VRAM since it is only a one-element lookup per token; and a corrected VRAM budget that
subtracts the non-layer tensor sizes before the per-layer estimate and targets 90% of VRAM.

## Nemotron-H CUDA forward pass (`NemotronHCudaForwardPass`)

GPU-resident forward pass for the Mamba-2 plus attention plus FFN hybrid, handling all three layer
types on GPU.

Mamba-2 kernels: `mamba2_scan.cu` (SSM state update per head, `[nheads][headDim][stateSize]`),
`mamba2_dt_softplus.cu` (timestep discretization), `mamba2_gate_norm.cu` (fused gate plus grouped
RMSNorm with `norm_before_gate=False`), and `sqrelu.cu` (squared ReLU for the FFN layers). It reuses
`conv1d_short.cu`, `silu.cu`, `rmsnorm.cu`, `rope.cu`, and `attention.cu` for shared operations.

CUDA graph capture works here — `NemotronH CUDA graph: captured 40 layers` was confirmed on
`granite-4.0-h-micro` in v1.11.0-dev. The first generation may fall back to per-layer mode on a
transient `cuMemcpyDtoH` error (906) during capture; subsequent generations replay the graph.
Measured throughput: Nemotron-3-Nano-4B at 20.0 tok/s (llama.cpp 35.6, so 56%), Granite 4.0-h-micro
at 35.6 tok/s (llama.cpp 41.8, so 85%).

## The dp4a integer path

Default-on via `cuda.dp4a`. It inserts `quantize_q8` calls after each FP32 buffer is produced — post
attention norm, post attention, post FFN norm, post `silu_mul`, and post final norm — and routes all
eligible matmuls (QKV, Wo, gate, up, down, output) through int8 dp4a kernels that read Q8_1 input.

As of v1.13.0 the dispatch covers Q3_K, Q4_K, Q5_K, Q5_0, Q8_0, IQ4_NL, and IQ4_XS across all three
forward passes (`CudaForwardPass`, `Qwen35CudaForwardPass`, `NemotronHCudaForwardPass`) and across
both the per-layer matmul dispatch and the final output projection. `launchOutputMatmul` was
previously missing cases 50/80/41/42, so any model whose output weight was Q5_0, Q8_0, IQ4_NL, or
IQ4_XS silently fell back to the FP32 kernel; fixing it gave a small measured gain on Gemma-3 1B
(42.4 → 43.6 tok/s, +2.8%).

Q6_K dp4a is opt-in only — see `cuda.dp4a.q6` — because the byte loads forced by its 210-byte block
outweigh the dp4a benefit. Non-eligible types, or any type when `cuda.dp4a=false`, fall back to the
FP32 kernel.

### Identifying per-type dp4a bottlenecks

The v1.13.0 JFR and GPU-profile session on Gemma-3 1B and Phi-3-mini showed that the remaining gap
to llama.cpp is concentrated in specific dp4a kernels whose underlying block size is not 4-byte
aligned. To check whether a new model hits the same class of problem, run with
`-Dcuda.profile=true -Dcuda.nograph=true` and read the per-section `ms/tok` line. A single stage
(QKV, GateUp, siluDown, output) carrying more than 40% of the total is the symptom, and the culprit
is almost always the Q5_0, IQ4_NL, Q3_K, or Q6_K block size forcing byte `__ldg` instead of uint32
`__ldg`.

The three worst-performing configurations on v1.13.0:

- **Llama-3.2-1B Q4_K_M** (baseline): total around 10.5 ms/tok, balanced across GateUp, siluDown,
  and output at roughly 25–28% each.
- **Gemma-3 1B Q4_K_M** (anomaly): total around 25 ms/tok, with **GateUp alone at 12.7 ms (51%)**
  because the Q4_K_M mix ships Q5_0 for Q, K, gate, and up, and the Q5_0 dp4a kernel must use byte
  `__ldg` since the 22-byte block is not 4-byte aligned. The closing action was
  `matmul_q5_0_dp4a_smem.cu`, which caches the Q8_1 input vector in shared memory — default on via
  `cuda.q5_0.smem`, worth 2–3% on Gemma-3 1B and 4B.
- **Phi-3-mini IQ4_NL** (anomaly): total around 95 ms/tok, with QKV at 21 ms, GateUp at 36 ms, and
  siluDown at 21 ms. All IQ4_NL kernels suffer the same 18-byte non-aligned-block problem plus a
  non-linear codebook lookup per weight. `matmul_iq4_nl_dp4a_smem.cu` was written but measured
  neutral in graph mode, so it is opt-in (`cuda.iq4nl.smem=false` by default).

## Flash-decoding attention

Every GPU-resident pass launches its single-token attention through `FlashAttention`
(`attention_flash.cu`). The previous `attention_full` kernel ran one block per query head, kept
every score in shared memory (`(seqLen + 32) * 4` bytes, so a launch failed past about 12,200
tokens under the 48 KB default and the engine then fell back to the CPU with an empty KV history),
let one thread walk each K row serially (a warp's loads were `kvDim`-strided and uncoalesced), kept
only `headSize` of its 256 threads busy in the weighted V sum, and re-read every K/V row once per
query head of a GQA group.

In the new kernel one block handles the `G` query heads that share a KV head over one slice of the
sequence. Each warp takes one timestep at a time: the 32 lanes read the K row and the V row
coalesced, once for all `G` heads, each head's score is reduced with warp shuffles, and an online
softmax (running max, running sum, rescaled accumulator) avoids storing any score. The last block of
a head group to finish (an atomic ticket per group) merges the slices in the same launch. Slices are
at least 64 timesteps long, computed on the device from the current sequence length, so at short
contexts the idle slices exit at once while the launch configuration stays fixed for graph capture.
The kernel is instantiated for `G` in {1, 2, 3, 4, 6, 7, 8} and `NJ = ceil(headSize / 32)` with
`G * NJ <= 32`, so the query and the accumulators stay in registers; each instantiation is compiled
as its own module on first use (the `FA_G` / `FA_NJ` defines), not all 54 at once. `gridDim.z`
tokens can run in one launch, which the batched prefill uses.

Measured on Llama-3.2-1B Q4_K_M at 4158 tokens of context: decode 42 → 75 tok/s, prefill 88 → 129
tok/s (per-token path). At short context it is neutral. The kernel takes the score scale and an
attention-logit soft-cap (Gemma 2) as parameters; the legacy kernel hard-coded `1/sqrt(headSize)`,
which is why Granite and Gemma 4 pre-scaled Q, and ignored the soft-cap. `-Dcuda.attn.flash=false`
restores `attention_full`, with its shared memory raised above 48 KB through `cuFuncSetAttribute`
when the device allows it.

## Sliding-window attention on GPU

For Gemma 2, Gemma 3, and GPT-OSS, `CudaForwardPass` builds a per-layer `slidingWindowPerLayer[]`
table (0 meaning global/full attention) that mirrors `Attention.isGlobalLayer`, and passes the
active window size to the attention kernel. There is no dedicated SWA kernel — the attention kernel
applies the window mask when that parameter is nonzero. The table follows the per-architecture
pattern: Gemma 2 alternating, Gemma 3 every-sixth-global, GPT-OSS alternating. Architectures with a
single uniform window mark every layer as SWA.

## Batched prefill

At batch 1 every weight is read once per prompt token, so prefill ran at decode speed.
`CudaForwardPass.prefillBatch` takes a chunk of up to 256 tokens through each layer together
(`InferenceEngine.forwardPrefill`, full offload, prompts of 32 tokens or more):

- every projection is one GEMM: the weight is dequantized to FP16 tile by tile (32 MB scratch,
  `dequant_f16.cu`, one kernel per type) and multiplied with cuBLAS `cublasGemmEx` (FP16 inputs,
  FP32 accumulation and output, tensor cores). A weight is read once per chunk instead of once per
  token. Merged Q/K/V and packed gate/up are row ranges of one GEMM, and residual additions are the
  GEMM's `beta = 1`.
- norms, RoPE, KV writes, biases and the elementwise ops are one launch for the whole chunk
  (`batch_ops.cu`);
- attention is one causal launch of the flash kernel over the chunk (each token reads its own
  position; all of the chunk's K/V is written first).

The layer math is the same as `forwardAttentionPart` / `forwardFFNPart`, including biases, QK-norm,
post-norms, Granite scaling, NoPE layers, sliding windows and the soft-cap. The activations entering
a GEMM are rounded to FP16 and saturated at ±65504, as llama.cpp does for its cuBLAS prefill, so the
generated text can differ slightly from the per-token path. The last prompt token still runs the
single-token path, which produces the logits. The chunk buffers and the cuBLAS handle are set up
when the pass is created; the chunk size halves until they fit in free VRAM. A failure falls back to
the per-token path, which rewrites the same KV slots.

Measured on prompts of about 1000 tokens (per-token GPU prefill → batched): Qwen3-0.6B Q8_0
142 → 1402 tok/s, Llama-3.2-1B 135 → 764, Gemma-3-1B 104 → 1570, OLMo-2-1B 106 → 1011,
Granite-3.3-2B 54 → 372, SmolLM3 48 → 275, Qwen2.5-3B 41 → 296, Llama-3.2-3B Q3_K_L 40 → 435,
Gemma-2-2B IQ4_XS 18 → 375, Phi-3-mini IQ4_NL 11 → 274.

The same GEMM, extracted as `GemmF16`, batches the prefill of three more passes:

- **`MoeAttentionCudaPass` and `MlaAttentionCudaPass`** (`GpuAttentionPass.attentionLayerBatch`):
  the attention half of each MoE layer over the chunk, the routed experts batched on the CPU, and
  the experts resident in the GPU expert cache computed on the GPU over all the chunk's tokens
  routed to them. The MLA pass covers the latent layout (GLM-4.7-Flash) and the expanded one
  (DeepSeek-V2-Lite's combined `wkv_b`, or `-Dmla.latent=false`); in the expanded one the per-head
  K/V assembly runs per token, because the shared k_rope differs per token. DeepSeek-Coder-V2-Lite,
  120-token prompt: 156 → 78 ms per prompt token, logits within the run-to-run noise.
- **`Qwen35CudaForwardPass.prefillBatch`** (`LayerGpuForwardPass`, full offload): the DeltaNet and
  attention projections and the FFN as GEMMs, the DeltaNet conv and recurrence token by token on the
  chunk's rows (the decode kernels with their parameter blocks pointed at the rows and restored
  afterwards), attention with the batch kernels and one flash launch, the Qwen3.5-MoE experts
  through a batched engine callback. Qwen3.5-0.8B, 452-token prompt: 9.8 → 1.5 ms per prompt token,
  with the prompt's final logits within 0.0125 of the CPU. A pass with recurrent state must zero it
  when a sequence starts at position 0, in the batched path as in `uploadXAndUpdateParams`.

`-Dprefill.gpu.batched=false` keeps the per-token path in these passes, and `-Dprefill.gpu.gemm=false`
disables the cuBLAS GEMM.

## Threads, streams and failures

- **Current context.** A CUDA context is current per thread, and `cuCtxCreate` makes it current only
  on the creating thread. `CudaContext.ensureCurrent()` (a thread-local check, then
  `cuCtxSetCurrent`) runs at every entry point, so a model loaded on one HTTP worker and used on
  another no longer fails every call with `CUDA_ERROR_INVALID_CONTEXT`.
- **One non-blocking stream.** The work stream is created with `CU_STREAM_NON_BLOCKING`, every copy
  and memset is issued on it (`readBuffer` is an async copy followed by a stream sync), and graph
  capture uses `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`. The global capture mode on a blocking stream is
  what produced the intermittent `cuMemcpyDtoH` error 906 during capture.
- **Serialized generations.** `LLMEngine.generate` holds a lock whenever a GPU backend is active: the
  GPU-resident passes keep the KV cache and recurrent state of one sequence, and the per-tensor path
  shares pooled buffers.
- **KV ownership.** The device KV cache holds the history of the state that ran last. When the
  conversation cache would resume a different state, `LLMEngine` starts a fresh state from position
  0 instead (`InferenceEngine.gpuHoldsHistoryOf`, `lastGpuResidentState` for the other engines); a
  forward at position > 0 for another state throws instead of attending over the wrong history.
- **Failure mid-sequence.** A GPU failure at position 0 falls back to the CPU for that token. Later
  it throws `GpuFailureException` after dropping the pass, because the CPU never received the KV
  cache or recurrent state of the earlier positions; `LLMEngine` retries the request on the CPU from
  position 0 when nothing has been streamed to the caller yet.

## MoE on the GPU

- `MoeAttentionCudaPass` runs the attention half of each Qwen3-MoE-family layer on the device: attn
  norm, Q/K/V through `Dp4aMatmul`, biases, QK-norm, RoPE, KV write, flash attention, Wo into the
  residual, FFN norm. One upload (residual and position) and one download (residual and FFN-normed
  input) per layer replace about five synchronous per-tensor round trips, and the attention itself
  leaves the CPU (Qwen3-Coder-30B: 197 → 37 ms per token). The KV-aware placement reserves its KV
  cache next to the attention weights, switching to FP16 KV when only that fits.
- `ExpertGpuCache` takes block geometry and kernel per projection (gate, up and down can have
  different types), runs the activation and the expert biases on the GPU, and synchronizes once per
  layer; K-quant experts are now on by default.
- `GraniteExpertGpu` uses `Dp4aMatmul`, which covers every dp4a type, quantizes the layer input once
  for all gate/up matmuls, and launches dp4a with its own geometry.

## Kernel compilation

NVRTC output is cached on disk (`~/.cache/llmplayer/cuda`, keyed by SHA-256 of source, options and
NVRTC version). When NVRTC supports the device's SM the kernels are compiled to cubin (`sm_XX`), so
the driver does not JIT PTX at start-up; otherwise PTX is generated for the newest virtual
architecture NVRTC supports that is not above the device. Passing the device's `compute_XX`
verbatim used to fail every compile on a GPU newer than the installed NVRTC.

## Runtime QKV fusion

Opt-in via `-Dcuda.fuse.qkv=true`. For models that ship separate Q/K/V weights (Llama, Qwen,
Mistral, Gemma, Granite), `CudaForwardPass` concatenates the three weight tensors byte-for-byte into
a single GPU buffer at construction time and activates the merged-matmul plus `split_qkv.cu` path
that was previously used only for natively packed architectures such as Phi-3/4. The synthetic
weight tensor is a `MergedQkvCudaTensor`, which is GPU-only — its CPU fallback throws.

This saves two kernel launches and two redundant Q8_1 input reads per layer per token. It measured
neutral on Llama-1B Q4_K_M in CUDA graph mode, since graph capture already amortizes launch overhead
via `cuGraphLaunch`, and about +2% in no-graph mode. It is disabled by default because it doubles
the QKV weight footprint in VRAM during the initialization device-to-device copy.

## CUDA kernel design patterns

All CUDA matmul kernels use one warp (32 threads) per output row with `__shfl_down_sync` reduction.
The grid is `ceil(rows / (blockSize/32))` blocks.

**Alignment constraints.** `__ldg((const unsigned int*)ptr)` requires a 4-byte aligned `ptr`. Block
sizes **not** divisible by 4 — Q8_0 (34B), Q4_0 (18B), Q5_0 (22B), Q6_K (210B), Q3_K (110B), IQ4_NL
(18B), IQ3_XXS (98B), IQ3_S (110B), IQ2_S (82B) — force those kernels to byte-level `__ldg` only.
Block sizes safe for uint32 `__ldg`: Q4_K (144B), Q5_K (176B), Q5_1 (24B), IQ4_XS (136B).

## cuBLAS acceleration (opt-in for decode)

The batched prefill uses cuBLAS by default (see above). For decode, an optional path using the
NVIDIA cuBLAS library for matmul is enabled with `-Dcuda.cublas=true`. It
pre-dequantizes Q4_K weights to FP16 (default) or FP32 at load time, then uses `cublasSgemv` and
`cublasGemmEx` for matrix-vector multiply.

Bindings live in `CublasBindings.java` (Panama FFM over `libcublas.so`) and `CublasMatmul.java`
(handle management, dequantization, gemv). The dequantization kernels are `dequant_q4_k_f16.cu`,
`dequant_q4_k_f32.cu`, and `convert_f32_to_f16.cu`.

The tradeoff: cuBLAS reaches higher bandwidth utilization (roughly 55–80%) than the custom Q4_K
kernels (roughly 22%), but FP16 weights are 3.5× larger than Q4_K. On bandwidth-limited GPUs such as
the RTX 4050 at 192 GB/s, custom Q4_K plus CUDA graph is faster. cuBLAS becomes competitive on GPUs
above roughly 500 GB/s (A100, H100) or when the weights are already FP16.

## Kernel inventory

`src/main/resources/kernels/cuda/` holds 81 `.cu` files, compiled at runtime through NVRTC by
`CudaContext`:

- **Matmul**, one per quantization type: Q3_K, Q4_0, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0, F32, BF16,
  F16, IQ2_S, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, MXFP4.
- **dp4a variants**: `matmul_q3_k_dp4a.cu`, `matmul_q4_k_dp4a.cu`, `matmul_q5_k_dp4a.cu`,
  `matmul_q6_k_dp4a.cu`, `matmul_q5_0_dp4a.cu`, `matmul_q8_0_dp4a.cu`, `matmul_iq4_nl_dp4a.cu`,
  `matmul_iq4_xs_dp4a.cu`.
- **Shared-memory variants**: `matmul_q5_k_smem.cu`, `matmul_q6_k_smem.cu`, `matmul_q6_k_tiled.cu`,
  `matmul_q5_0_dp4a_smem.cu`, `matmul_iq4_nl_dp4a_smem.cu`.
- **Other matmul experiments**: `matmul_q4_k_2warp.cu`, `matmul_q4_k_coalesced.cu`,
  `matmul_q4_k_dp4a_mr4.cu`, `matmul_iq4_xs_dp4a_mw.cu`.
- **DeltaNet**: `deltanet_fused.cu`, `deltanet_fused_v2.cu`.
- **Mamba-2**: `mamba2_scan.cu`, `mamba2_dt_softplus.cu`, `mamba2_gate_norm.cu`.
- **Attention**: `attention_flash.cu` (flash decoding, default), `attention.cu` and
  `attention_f16.cu` (KV writes, legacy `attention_full`).
- **Batched prefill**: `dequant_f16.cu` (row-range FP16 dequantization per type), `batch_ops.cu`
  (multi-token norm, RoPE, KV write, bias, axpy, FP16 conversion).
- **cuBLAS support**: `dequant_q4_k_f16.cu`, `dequant_q4_k_f32.cu`, `convert_f32_to_f16.cu`,
  `quantize_q8.cu`.
- **MoE**: `swiglu_oai.cu` (GPT-OSS expert activation in `ExpertGpuCache`).
- **Auxiliary**: RMSNorm, RoPE, attention, softmax, SiLU, argmax, `split_qkv`, `split_gate_up`,
  `fused_gate_up`, `rmsnorm_per_head`, `conv1d_short`, `conv1d_silu`, `alpha_beta_gates`,
  `deinterleave_q_gate`, `sigmoid_elementwise_mul`, `sqrelu`, `scale_inplace`, `silu_mul`, `gelu`.

`src/main/resources/kernels/` holds the 14 OpenCL `.cl` files, compiled on demand by `OpenCLContext`:
seven matmul variants (`matmul_f32`, `matmul_q3_k`, `matmul_q4_0`, `matmul_q4_k`, `matmul_q5_k`,
`matmul_q6_k`, `matmul_q8_0`) plus `rmsnorm`, `softmax`, `silu`, `saxpy`, `accumulate`,
`elementwise_mul`, and `fill_zero`.

MXFP4 now has a GPU tensor wrapper (`MXFP4CudaTensor`) routing matmul through `matmul_mxfp4.cu`;
previously MXFP4 was CPU-only at the tensor layer.
