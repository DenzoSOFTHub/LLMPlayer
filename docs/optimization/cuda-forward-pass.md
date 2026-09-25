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

The central design constraint is **zero-allocation hot paths**. All kernel parameter buffers
(`ParamBuffer`) and matmul launch descriptors (`MatmulLaunch`) are pre-allocated in the constructor.
`forwardLayer()` only writes parameter values in place and launches kernels — it must never
allocate.

## Internal split (v1.13.0)

`forwardLayer` is a thin wrapper around two private helpers:

- `forwardAttentionPart(layerIdx, position)` — steps 1–6c: attention norm, QKV, QK-norm, Granite
  scale, RoPE, KV cache, attention, Wo, post-attention norm.
- `forwardFFNPart(layerIdx)` — steps 7–10c: FFN norm, gate/up, `silu_mul`, down, post-FFN norm.

A public `forwardAttentionOnly(state, layerWeights, layerIdx, position, attention)` is exposed for
engines that own their own FFN path (the MoE architectures). It runs the attention half on GPU and
leaves the post-attention residual in `gpuX` for the caller to download, process on CPU or
elsewhere, and re-upload before the next layer's attention. The companion
`isSupportedForAttention(config, weights)` is a relaxed variant of `isSupported` that drops the MoE
block and the FFN-tensor requirement. The Qwen3MoE and DeepSeek2 engine wiring that will consume
this interface is scheduled for v1.14.

## Supported architectures

Running the full `CudaForwardPass`: Llama, Qwen2, Qwen3, Falcon3, OLMo2 (including Olmo 3),
Mistral3, Gemma 2 and Gemma 3 (post-norm), Phi-3/4 (packed FFN via `split_gate_up.cu`), Granite 3.3
(with scaling factors), ERNIE 4.5 (dense, explicit head_dim, RoPE NORM), and Nemotron-H / Granite
Hybrid via `NemotronHCudaForwardPass` (Granite Hybrid uses the full integrated-FFN path with all
scale factors on GPU). Per-head QK-norm (Qwen3) is handled by `rmsnorm_per_head.cu`.

Dedicated per-layer GPU-resident forward passes exist for **LFM2** (`LFM2CudaForwardPass`),
**Falcon-H1** (`FalconH1CudaForwardPass`), and **Gemma 4** (`Gemma4CudaForwardPass`), all added
2026-06-07.

**Not supported on GPU**, falling back to per-tensor matmul or CPU: the MoE architectures
(including Granite Hybrid MoE, whose experts run through `GraniteExpertGpu` while the dense parts
stay on CPU), and Gemma 3n, whose AltUp path runs on CPU via `Gemma4InferenceEngine`.

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

## Sliding-window attention on GPU

For Gemma 2, Gemma 3, and GPT-OSS, `CudaForwardPass` builds a per-layer `slidingWindowPerLayer[]`
table (0 meaning global/full attention) that mirrors `Attention.isGlobalLayer`, and passes the
active window size as the tenth argument of the attention kernel parameter buffer
(`attnPB.setInt(9, ...)`). There is no dedicated SWA kernel — the standard `attention.cu` kernel
applies the window mask when that parameter is nonzero. The table follows the per-architecture
pattern: Gemma 2 alternating, Gemma 3 every-sixth-global, GPT-OSS alternating. Architectures with a
single uniform window mark every layer as SWA.

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

## cuBLAS acceleration (opt-in)

An optional path using the NVIDIA cuBLAS library for matmul, enabled with `-Dcuda.cublas=true`. It
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

`src/main/resources/kernels/cuda/` holds 78 `.cu` files, compiled at runtime through NVRTC by
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
- **DeltaNet**: `deltanet_fused.cu`, `deltanet_fused_v2.cu`, `deltanet_recurrence.cu`.
- **Mamba-2**: `mamba2_scan.cu`, `mamba2_dt_softplus.cu`, `mamba2_gate_norm.cu`.
- **cuBLAS support**: `dequant_q4_k_f16.cu`, `dequant_q4_k_f32.cu`, `convert_f32_to_f16.cu`,
  `quantize_q8.cu`.
- **Auxiliary**: RMSNorm, RoPE, attention, softmax, SiLU, argmax, `split_qkv`, `split_gate_up`,
  `fused_gate_up`, `rmsnorm_per_head`, `conv1d_short`, `conv1d_silu`, `alpha_beta_gates`,
  `deinterleave_q_gate`, `sigmoid_elementwise_mul`, `sqrelu`, `scale_inplace`, `silu_mul`, `gelu`.

`src/main/resources/kernels/` holds the 14 OpenCL `.cl` files, compiled on demand by `OpenCLContext`:
seven matmul variants (`matmul_f32`, `matmul_q3_k`, `matmul_q4_0`, `matmul_q4_k`, `matmul_q5_k`,
`matmul_q6_k`, `matmul_q8_0`) plus `rmsnorm`, `softmax`, `silu`, `saxpy`, `accumulate`,
`elementwise_mul`, and `fill_zero`.

MXFP4 now has a GPU tensor wrapper (`MXFP4CudaTensor`) routing matmul through `matmul_mxfp4.cu`;
previously MXFP4 was CPU-only at the tensor layer.
