# LLMPlayer v1.11.0

Pure Java LLM inference engine for running GGUF models locally. Zero external dependencies — uses only the JDK. Supports 21 architectures including Llama, Qwen2/3/3.5, SmolLM3, DeepSeek2, Gemma 2/3/3n/4, Phi-3/4, Mistral3/Devstral, Falcon3, Granite 3.3, **Granite Hybrid**, **Nemotron-H** (hybrid Mamba-2 + Transformer), and **Olmo 3** (ChatML variant). 18 quantized formats with 16 dedicated CUDA kernels. Includes CUDA GPU acceleration with graph mode (~57 tok/s on RTX 4050 for Llama-3.2-1B), cuBLAS support (opt-in), thinking/reasoning mode, architecture-aware tool calling, HuggingFace model download, JMX runtime metrics with rolling window, smoke test suite for all architectures, automated kernel autosearch, and a built-in LoRA fine-tuning pipeline.

### What's new in v1.11.0 — Granite Hybrid full GPU + dp4a kernel fleet

Highlights:

- **dp4a expanded to Q5_0 / Q8_0 / IQ4_NL / IQ4_XS** — four new CUDA kernels (`matmul_{q5_0,q8_0,iq4_nl,iq4_xs}_dp4a.cu`) wired into `CudaForwardPass`, `Qwen35CudaForwardPass`, and `NemotronHCudaForwardPass`. Measured on RTX 4050: **Phi-3-mini IQ4_NL +42%** (8.4 → 11.9 tok/s), **Nemotron-3-Nano-4B +92%** (10.4 → 20.0), Qwen3-1.7B Q8_0 +8%, Gemma-3-1B Q5_0 +4%. The previously-anomalous models stuck at 15–32% of llama.cpp (Phi-3-mini IQ4_NL, Gemma-2-2B IQ4_XS, Gemma-3-1B Q5_0) were all falling through to FP32 kernels — fixed. PPL preserved bit-equivalent.
- **Granite Hybrid — full GPU path** — integrated SwiGLU FFN (inside Mamba/Attention layers) + all four scale factors (embedding/logit/residual/attention) on GPU via `runIntegratedFFN()`, `scale_inplace`, `accumulate`, and saxpy. `isSupported()` now returns true for Granite Hybrid; CUDA graph capture works. Bit-equivalent to CPU at ±2 ULP when dp4a is disabled. **Measured on granite-4.0-h-micro Q4_K_M: 35.6 tok/s (was ~8 CPU, +4.4×; 19% → 85% of llama.cpp).** Biggest single-model jump of the sprint.
- **Speculative decoding (scaffolding)** — new `it.denzosoft.llmplayer.spec.SpeculativeDecoder` (Leviathan et al.). Standalone class driving target + draft `LLMEngine` via `forwardSingleToken`. Enabled with `--draft-model <gguf>`. **Sequential verification only** (~1.14× max with K=4); real 2-3× requires a batched `forwardBatch` API that doesn't exist yet. Kept shipped for algorithmic correctness testing. See `docs/optimization/speculative-decoding.md`.
- **Optimization journal** — new `docs/optimization/` directory records kernel-level attempts with measured outcomes: cubin (Option A), cp.async prefetch (Option C), multi-warp Q4_K, mmvq port. All 0 to −22% on RTX 4050 Q4_K matvec at batch=1 — the hardware is at the bandwidth ceiling for this workload. `llamacpp-comparison.md` tracks the rolling tok/s vs llama.cpp across 17 models (avg ~72% of llama.cpp for standard Q4_K_M).

Still planned: Gemma 4 E2B benchmark, Q4_1 tensor support, dedicated `Gemma4CudaForwardPass` (PLE on GPU), MoE Granite Hybrid Tiny, Bonsai Q1_0 format, batched `forwardBatch` to unlock real speculative-decoding speedup.

### What's new in v1.10.2 — Gemma 4 fully working + Granite Hybrid GPU fix + Olmo 3 + autosearch

This release closes the long-standing **Gemma 4 / Gemma 3n** correctness gap (PPL 0.00 → 1.00 on canonical Q&A), fixes a GPU-only bug for **Granite Hybrid**, adds **Olmo 3** support, and ships an automated kernel-tuning tool.

- **Gemma 4 (PLE) — fully working at PPL 1.00** ✓ — root cause was two missing pieces vs `llama.cpp gemma4-iswa.cpp`: (1) **V-norm** (`ggml_rms_norm` without learnable scale on V projections) was disabled for Gemma 4; (2) **`layer_output_scale.weight`** per-layer scalar must be applied as a final multiplication of the residual stream (`cur *= out_scale`). Together these fixes turn semantically-related multilingual gibberish into coherent answers like *"The capital of France is **Paris**."*
- **Gemma 3n — also confirmed working at PPL 0.97-1.00** ✓ via the existing `forwardLayerGemma3nInner` AltUp + Laurel + sparsity path. K-norm `(1+w)` adjustment is now architecture-conditional (Gemma 3n only — Gemma 4 stores final values).
- **BPE decode bug fixed for Gemma 4 SentencePiece-mode tokens** — `BPETokenizer.decodeTokenPiece` was applying GPT-2 byte mapping unconditionally, leaving `▁` (U+2581) literal in output. Now: when `useGpt2ByteMapping=false` (gemma4 mode), `▁` is replaced with space and `<0xHH>` byte fallback tokens decode via UTF-8 byte coalescing.
- **Granite Hybrid GPU fix** — `NemotronHCudaForwardPass` was producing PPL 0.20 (junk output) for Granite Hybrid models because it didn't apply `embeddingScale`/`attentionScale`/`residualScale` on GPU. Now the GPU path is skipped (CPU fallback) when scaling is enabled — output goes back to PPL 1.00.
- **Olmo 3 ChatML detection** — Olmo 3 ships under the `olmo2` GGUF architecture but uses ChatML (`<|im_start|>...<|im_end|>`) instead of the legacy `<|user|>` format. `ChatTemplate` now auto-detects this from the chat_template metadata and switches format.
- **`autosearch.sh`** — Karpathy-style greedy coordinate-ascent over the full `-D` flag matrix (`cuda.dp4a`, `cuda.q4k.smem`, `cuda.q5k.smem`, `cuda.q6k.tiled`, `cuda.deltanet.v2`, `cuda.cublas`, `kv.q8`, `matmul.tiled`, `attn.flash`, etc.). Two KPIs: **tok/s** (max) and **PPL** (≥ threshold). Auto-discovers Pareto-optimal configurations per model.
- **22 models in CUDA-graph benchmark** including the new **Qwen3.6-Plus-Distill-4B-Thinking** community LoRA distillation (16.6 tok/s Q8_0).
- **Q4_1 documented as unsupported** at the tensor layer (no `Q4_1FloatTensor` implementation; community Qwen3.6 distill uses Q4_1 → use Q8_0 variant instead).

### What's new in v1.10.1 — Top-10 audit vs llama.cpp

Six commit's worth of correctness fixes, sampler additions, and a major memory optimization driven by an audit of LLMPlayer against the llama.cpp reference implementation. See `BENCHMARKS.md` for measured deltas.

- **Q8_0 KV cache** (`-Dkv.q8=true`) — block-quantized int8 KV with FP32 scales per 32-elem block. **3.56× memory reduction** on all 5 inference engines including the asymmetric MLA path for DeepSeek2 (separate K/V dims). Bit-identical greedy output on dense models. Headline: **DeepSeek-Coder-V2-Lite Q4_K_M CPU mode goes from 1080 MB → 303 MB KV and 0.7 → 0.9 tok/s** (faster, not just smaller — MLA is DRAM-bandwidth-bound).
- **Command-R LayerNorm fix** — `LayerNorm` (centered + scaled, no bias) now used in place of `RMSNorm` for Command-R / Cohere2 (matches llama.cpp `LLM_NORM`). Cohere2 also gets a separate enum, NoPE-on-global-layers handling, and the right ISWA pattern (`set_swa_pattern(4)`).
- **DS-V3 / GLM-4.7-Flash routing fix** — `exp_probs_b` is now applied AFTER sigmoid (not before), and only as a selection score; mix weights come from the unbiased `probs`. Matches llama.cpp `build_moe_ffn`.
- **New samplers** (opt-in): `--min-p`, `--mirostat 2 --mirostat-tau 5.0`, `--dry-multiplier 0.8`. Pipeline: `DRY → rep_penalty → temp → top-K → softmax → min-P → top-P → (mirostat | multinomial)`.
- **Wo bias / output bias loading** — `attn_output.bias` and top-level `output.bias` now loaded and applied for Qwen2 / SmolLM3 / Command-R variants that ship them.
- **GC churn reduction** — Qwen3.5 and Nemotron-H pre-cache all per-layer norm weights and pre-allocate conv1d output buffers.
- **MoE routing safety** — F16-epsilon clamp on weight-sum normalization in `Qwen3MoEInferenceEngine.moeFFN` (anti-NaN guardrail, matches llama.cpp `ggml_clamp`).
- **Gemma 3n / Gemma 4 broken-support warning** — at the time, PLE-only path produced random multi-language tokens. ~~Loud WARNING banner at engine creation~~ **Resolved in v1.10.2** (see above).
- **Documented but not enabled by default**: `-Dattn.flash=true` (FlashAttention online-softmax — bit-identical but ~10% slower on Java/CPU because the SIMD `VectorOps.softmax` is already fast); Nemotron-H CUDA graph (was hard-disabled, re-enabled but gives 0% speedup on this workload because Mamba-2 scan is compute-bound, not launch-bound).

### What's new in v1.10.0

- **Q5_1 CUDA kernel** — full GPU acceleration for Q5_1 quantization (24-byte aligned blocks, vectorized uint32 reads). Bit-exact with CPU.
- **Q4_K 2-warp kernel** — opt-in `-Dcuda.q4k.2warp=true` splits each output row across two warps for higher SM occupancy on small/medium matrices.
- **Refactored CudaFloatTensor** — base class now respects `getMatmulGridDim()` overrides on the non-graph path, enabling multi-warp-per-row kernels through the standard launch path.
- **JMX rolling window** — 60-second sliding window of generation samples for live tok/s monitoring. New attributes `RecentTokensPerSecond` and `RecentSampleCount` exposed via JMX MXBean and `/api/metrics`.
- **Architecture smoke test** — `test-architectures.sh` validates that each of the 18+ supported architectures loads and generates at least one valid token. Auto-detects Java 21+. Run before each release as a regression sanity check.
- **Bumped to v1.10.0** in pom.xml, LLMPlayer.java, README.md, web-ui.html (UI title is dynamic via `LLMPlayer.VERSION`).

### What's new in v1.9.0

- **Gemma 4 architecture**: Per-Layer Embeddings (PLE), dual headSize (SWA=256, full=512), shared KV cache, V-norm, dual RoPE, K-norm with (1+w), logit soft-capping. Dedicated `Gemma4InferenceEngine`.
- **Gemma 3n architecture**: PLE support via Gemma4 engine.
- **Granite Hybrid architecture**: Mamba-2 + Attention + FFN hybrid with integrated SwiGLU FFN per layer. Reuses NemotronH engine with embedding/attention/residual/logit scaling.
- **Granite 3.3 CUDA graph**: fixed GPU forward pass (was blocked, now 17x faster: 1.0 → 17.0 tok/s).
- **CUDA kernel optimizations**: dp4a enabled by default for Q4_K/Q5_K/Q6_K, Q5_K shared-memory kernel (+7%), Q6_K tiled kernel (+9%), DeltaNet v2 float4 vectorization (+4%). Qwen3.5-4B: 18 tok/s (+59%), Llama-1B: 56 tok/s (+17%).
- **JMX metrics**: runtime monitoring via `it.denzosoft.llmplayer:type=LLMPlayer` MXBean + REST at `/api/metrics`.
- 12 new models benchmarked across 4 new architectures. 34+ models total across 20 architectures.

## Requirements

- **Java 8** — base functionality (no SIMD, no GPU)
- **Java 21+** — adds SIMD (Vector API), optimized memory mapping (Panama FFI), GPU acceleration (CUDA + OpenCL)
- **Java 25** — adds advanced parallelism (StructuredTaskScope, virtual thread matmul)
- **NVIDIA GPU** — optional, for CUDA acceleration (requires NVIDIA driver + NVRTC)
- **OpenCL drivers** — alternative GPU backend (e.g. `libOpenCL.so` on Linux)
- **Maven 3.x** — for building
## Building

The project has three Maven profiles that include different source sets:

```bash
# java25 profile (default) — includes everything: java/ + java21/ + java25/
mvn clean compile

# java21 profile — includes java/ + java21/ (no StructuredTaskScope, no virtual thread matmul)
mvn clean compile -Pjava21

# java8 profile — includes only java/ (no Vector API, no Panama FFI, no GPU)
mvn clean compile -Pjava8
```

## Running

### With Java 8

After building with `-Pjava8`:

```bash
java -cp target/classes it.denzosoft.llmplayer.LLMPlayer [options]
```

No additional JVM flags needed. SIMD acceleration, GPU, and optimized memory mapping are not available — the system uses `MappedByteBuffer` and scalar operations as fallback.

### With Java 21+

After building with `-Pjava21`:

```bash
java --add-modules jdk.incubator.vector \
     --enable-native-access=ALL-UNNAMED \
     -cp target/classes \
     it.denzosoft.llmplayer.LLMPlayer [options]
```

The JVM flags are mandatory:
- `--add-modules jdk.incubator.vector` — enables the Vector API for SIMD tensor operations
- `--enable-native-access=ALL-UNNAMED` — enables Panama FFI for memory mapping and OpenCL bindings

### With Java 25

After building with the default profile:

```bash
java --add-modules jdk.incubator.vector \
     --enable-native-access=ALL-UNNAMED \
     --enable-preview \
     -cp target/classes \
     it.denzosoft.llmplayer.LLMPlayer [options]
```

The `--enable-preview` flag enables `StructuredTaskScope` for parallel batch generation and virtual threads for multi-threaded matmul.

### Launch scripts (Java 25)

These scripts are not included in the repository (excluded via `.gitignore`). Create them locally after cloning:

**`run.sh`** (Linux / macOS):
```bash
#!/bin/bash
# LLMPlayer launcher - includes all required JVM flags
DIR="$(cd "$(dirname "$0")" && pwd)"
exec java \
  --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  --enable-preview \
  -cp "$DIR/target/classes" \
  it.denzosoft.llmplayer.LLMPlayer "$@"
```

**`run.bat`** (Windows):
```batch
@echo off
REM LLMPlayer launcher - includes all required JVM flags
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED --enable-preview -cp "%~dp0target\classes" it.denzosoft.llmplayer.LLMPlayer %*
```

After creating, make the shell script executable: `chmod +x run.sh`

## Usage Modes

### Desktop GUI (default)

Launch without arguments to open the Swing interface:

```bash
./run.sh
```

The GUI allows you to:
- Select a GGUF model from a directory (default: `gguf/`)
- Configure temperature, top-k, top-p, repetition penalty, context length and max tokens
- Interactive chat with token-by-token streaming
- Real-time CPU and RAM monitoring
- Start/stop the integrated web server

### CLI — Single prompt

```bash
./run.sh --model path/to/model.gguf --prompt "Explain what artificial intelligence is" --max-tokens 512
```

Output is printed as a token-by-token stream. Statistics (tokens generated, speed, time) are shown at the end.

### CLI — Interactive chat

```bash
./run.sh --model path/to/model.gguf --interactive
```

Type your messages and press Enter. Special commands: `quit` or `exit` to leave, `info` for model details.

### CLI — Thinking/reasoning mode

```bash
./run.sh --model SmolLM3-Q4_K_M.gguf --thinking --prompt "What is 25 * 37?" --max-tokens 512
```

Enables extended reasoning for supported models. The model "thinks" step-by-step before answering:

| Model | Mechanism | How it works |
|-------|-----------|-------------|
| SmolLM3 | `/think` system message | Injects `/think` into the system prompt |
| Qwen3 | `<think>` suppressor removal | By default thinking is suppressed; `--thinking` removes the suppressor |
| Qwen3.5 | `<think>` suppressor removal | Same as Qwen3 |

Via the OpenAI API, pass `"thinking": true` in the request body.

### Web UI

```bash
./run.sh --web --port 8080 --gguf-dir ./models
```

Starts an HTTP server at `http://localhost:8080` with two web interfaces and REST APIs. The port is configurable (default: 8080).

- **`/`** — Model config page: load/unload GGUF models, configure GPU, monitor hardware
- **`/chat`** — Chat UI: persistent conversations with edit, regeneration, and branching support

The chat UI stores conversations as JSON files in the `chats/` directory (created automatically). Editing a user message or regenerating an assistant response creates a branch — alternative paths in the conversation tree, navigable with arrow controls.

### Download models from HuggingFace

```bash
# Download Q4_K_M (auto-selected) from a HuggingFace repo
./run.sh --download "bartowski/Llama-3.2-1B-Instruct-GGUF"

# Download a specific file
./run.sh --download "bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

# Use with custom directory and HuggingFace token for gated models
./run.sh --download "meta-llama/Llama-4-Scout-17B-16E-Instruct-GGUF" --gguf-dir models --hf-token hf_xxx
```

Skips download if the file already exists locally with matching size.

### Model info

```bash
./run.sh --model path/to/model.gguf --info
```

Prints model metadata (architecture, layers, dimensions, vocabulary) and exits.

## GPU Acceleration

Requires Java 21+ and a supported GPU. Two backends are available:

- **CUDA** (preferred) — calls `libcuda.so` + `libnvrtc.so` via Panama FFM. Kernels are `.cu` files compiled at runtime by NVRTC. Requires NVIDIA driver and NVRTC.
- **OpenCL** — calls `libOpenCL.so` via Panama FFM. Kernels are `.cl` files compiled at runtime. Works with any OpenCL driver.

Backend selection: `--gpu-backend auto` (default, prefers CUDA), `--gpu-backend cuda`, or `--gpu-backend opencl`.

### List available devices

```bash
./run.sh --gpu-list
```

Shows all detected CUDA and OpenCL devices with name and memory.

### Enable GPU

```bash
# Auto-detect best GPU (prefers CUDA)
./run.sh --model model.gguf --gpu --prompt "Hello" --max-tokens 256

# Use a specific device
./run.sh --model model.gguf --gpu-device 1 --prompt "Hello" --max-tokens 256

# Force a specific backend
./run.sh --model model.gguf --gpu --gpu-backend opencl --prompt "Hello"
```

### CUDA graph mode

For dense models that fit entirely in VRAM, CUDA graph mode captures all kernel launches (~210 for a 16-layer model) into a CUDA graph on the first token, then replays with a single `cuGraphLaunch` API call. This eliminates per-kernel Panama FFM launch overhead and delivers significant speedups.

Enabled by default when using CUDA with full GPU offload. Disable with `-Dcuda.nograph=true`.

### How it works

When GPU is enabled, the system:
1. Initializes CUDA or OpenCL context on the selected device
2. Registers a buffer manager in `TensorFactory`
3. For each tensor created during model loading, attempts the GPU variant first (e.g. `Q4_KCudaTensor`), then falls back to CPU
4. For dense models with all layers on GPU, enables the GPU-resident forward pass — activations stay on GPU between layers
5. Compute-heavy operations (matmul, RMSNorm, softmax, SiLU, etc.) are executed via GPU kernels compiled on-demand

GPU-supported quantized formats (CUDA): F32, Q3_K, Q4_0, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0, IQ3_XXS, IQ4_NL, IQ4_XS. Other formats automatically fall back to CPU.

If Java is < 21 or GPU drivers are not present, the system prints a warning and continues in CPU-only mode.

### MoE-optimized GPU placement

For MoE architectures (Qwen3MoE, DeepSeek2) with `--gpu-layers -1` (auto-detect, the default), the system uses an optimized placement strategy inspired by KTransformers (SOSP'25):

- **Attention tensors** go on GPU across **all** layers
- **Expert tensors** (`ffn_*_exps`, ~80-90% of layer weight) stay on CPU
- **Router and shared expert tensors** go on GPU (small)

This maximizes GPU utilization because expert tensors are large but only top-K are activated per token, while attention is compute-bound and benefits from GPU acceleration on every token. With 6 GB VRAM, standard first-N-layers fits only ~2/48 layers for Qwen3-Coder-30B, while MoE-optimized fits 100% of attention using just ~540 MB.

Explicit `--gpu-layers N` always uses first-N-layers to preserve backward compatibility.

## Java Profile Feature Comparison

| Feature | Java 8 | Java 21 | Java 25 |
|---|:---:|:---:|:---:|
| Base inference (all architectures) | Yes | Yes | Yes |
| Desktop GUI (Swing) | Yes | Yes | Yes |
| Web Server | Yes | Yes | Yes |
| All quantization formats (CPU) | Yes | Yes | Yes |
| SIMD tensor operations (Vector API) | No | Yes | Yes |
| Optimized memory mapping (Panama FFI) | No | Yes | Yes |
| GPU acceleration (CUDA + OpenCL) | No | Yes | Yes |
| CUDA graph mode | No | Yes | Yes |
| Virtual thread matmul | No | No | Yes |
| Parallel batch generation (StructuredTaskScope) | No | No | Yes |

Degradation is automatic: Java 21/25 classes are loaded via reflection (`Class.forName`). If unavailable, the system uses Java 8 fallbacks (scalar operations, `MappedByteBuffer`, standard thread pool) without errors.

## CLI Options

### General

| Option | Alias | Type | Default | Description |
|---|---|---|---|---|
| `--model` | `-m` | String | — | Path to the GGUF file |
| `--prompt` | `-p` | String | — | Prompt for single generation |
| `--interactive` | `-i` | Flag | false | Interactive chat mode |
| `--info` | — | Flag | false | Show model info and exit |
| `--web` | `-w` | Flag | false | Start the web server |
| `--port` | — | Integer | 8080 | Web server port |
| `--gguf-dir` | — | String | `gguf` | GGUF file directory |
| `--thinking` | — | Flag | false | Enable thinking/reasoning mode (SmolLM3, Qwen3, Qwen3.5) |
| `--draft-model` | — | String | — | Path to draft GGUF. Enables speculative decoding via `SpeculativeDecoder` (target = `--model`, draft = this). **Experimental**: sequential verification only (~1.14× max with K=4); kept for algorithmic correctness. See `docs/optimization/speculative-decoding.md`. |
| `--force` | `-y` | Flag | false | Skip confirmation prompts (e.g., RAM warning) |
| `--help` | `-h` | Flag | false | Show help |

### Generation Parameters

These flags control how tokens are generated. They affect output quality and creativity.

| Option | Alias | Type | Default | Description |
|---|---|---|---|---|
| `--max-tokens` | `-n` | Integer | 256 | Maximum number of tokens to generate. Generation stops at this limit or when the model produces an end-of-sequence token. |
| `--temperature` | `-t` | Float | 0.7 | Sampling temperature. 0 = greedy (always pick most probable), 0.1-0.5 = focused, 0.7-1.0 = creative, >1.0 = chaotic. Lower values produce more predictable output. |
| `--top-k` | — | Integer | 40 | After computing probabilities, keep only the K most probable tokens and renormalize. Prevents unlikely tokens from being selected. 0 = no top-K filtering. |
| `--top-p` | — | Float | 0.9 | Nucleus sampling: keep the smallest set of tokens whose cumulative probability exceeds P, then renormalize. 1.0 = no filtering. Applied after top-K. |
| `--repetition-penalty` | — | Float | 1.1 | Penalty applied to tokens that already appeared in the context. 1.0 = no penalty, 1.1-1.3 = light penalty, >1.5 = aggressive. Reduces loops and repetitive output. |
| `--min-p` | — | Float | 0 | min-P sampling: keep tokens with `p ≥ min_p × max_p`. Modern default in many clients. Applied after softmax, before top-P. 0 = disabled. Try 0.05 with temp=0.8 for a quality/diversity balance. |
| `--mirostat` | — | Integer | 0 | Mirostat sampler mode: `0` = off, `2` = Mirostat v2 (adaptive truncation by target entropy). Replaces multinomial sampling when enabled. |
| `--mirostat-tau` | — | Float | 5.0 | Mirostat target surprise (entropy in bits). Lower = more focused, higher = more diverse. |
| `--mirostat-eta` | — | Float | 0.1 | Mirostat learning rate for the running surprise estimate. |
| `--dry-multiplier` | — | Float | 0 | DRY (Don't Repeat Yourself) penalty multiplier. Penalizes tokens that would extend an existing n-gram match. 0 = disabled. Try 0.8 to break repetition loops. |
| `--dry-base` | — | Float | 1.75 | DRY exponential base. Penalty grows as `multiplier × base^(match_len − allowed_length)`. |
| `--dry-allowed-length` | — | Integer | 2 | DRY: minimum n-gram length before penalty kicks in. |
| `--dry-range` | — | Integer | 1024 | DRY: lookback window size in tokens. |
| `--seed` | — | Long | random | Random seed for reproducibility. Same seed + same prompt = same output (when temperature > 0). |
| `--threads` | — | Integer | num CPUs | Number of threads for parallel operations. By default uses all available CPU cores. |
| `--context-length` | `-c` | Integer | 2048 | Maximum context window in tokens. Larger values allow longer conversations but use more RAM. Model's maximum is shown in `--info` output. |

### GPU Flags

LLMPlayer **auto-detects** NVIDIA GPUs and enables CUDA when available. These flags override the default behavior.

| Option | Type | Default | Description |
|---|---|---|---|
| `--gpu` | Flag | false | Force GPU enablement. Not usually needed — auto-detection handles this. Use when auto-detect picks the wrong device or when you want to be explicit. |
| `--no-gpu` | Flag | false | **Disable GPU entirely**, force CPU-only inference. Useful for benchmarking or when GPU causes issues. Note: CPU-only is significantly slower (see Benchmarks). |
| `--gpu-device` | Integer | auto | Select GPU device by index. Use `--gpu-list` to see available devices and their indices. By default, the best available GPU is selected automatically. |
| `--gpu-backend` | String | `auto` | GPU compute backend: `auto` (prefers CUDA over OpenCL), `cuda` (NVIDIA only), `opencl` (any OpenCL device). |
| `--gpu-layers` | Integer | -1 | How many transformer layers to place on GPU. `-1` = auto-detect (fills VRAM optimally). `0` = no layers on GPU (equivalent to `--no-gpu`). Positive value N = put the first N layers on GPU. For MoE models with auto-detect, uses MoE-optimized placement (all attention on GPU, experts on CPU). |
| `--gpu-list` | Flag | false | List all detected CUDA and OpenCL devices with name, memory, and capabilities, then exit. |
| `--gpu-chain` | Flag | true | Enable GPU-resident forward pass (activations stay on GPU between layers). On by default. |
| `--no-gpu-chain` | Flag | false | Disable GPU-resident forward pass. Falls back to per-tensor GPU matmul (upload/compute/download per operation). Useful for debugging. |
| `--gpu-backend` | String | `auto` | GPU backend: `auto` (CUDA preferred), `cuda`, `opencl` |
| `--gpu-memory` | String | `device` | GPU memory allocation mode: `device` (GPU-only), `managed` (CUDA managed), `host-mapped` (host-pinned). |

**GPU auto-detection behavior:** When no GPU flags are specified, LLMPlayer probes for CUDA, then OpenCL. If an NVIDIA GPU is found, CUDA is enabled automatically with auto-detected layer count. CPU-only devices (PoCL) are skipped. To prevent this, use `--no-gpu`.

**Disable CUDA graph at runtime:** Use the JVM property `-Dcuda.nograph=true` to disable CUDA graph capture (useful for debugging or when shared memory is insufficient).

### CUDA JVM Tuning Properties

Advanced performance tuning via JVM system properties (`-Dproperty=value`). These do not require recompilation.

| Property | Default | Description |
|----------|---------|-------------|
| `-Dkv.q8=true` | `false` | **Q8_0 KV cache** — block-quantized int8 storage with FP32 scales (3.56× memory reduction). Bit-identical greedy output on dense models, plausibly-equivalent on MoE (router top-K sensitivity). **Always enable for DeepSeek2 / GLM-4.7-Flash**: MLA's large per-head K/V is DRAM-bandwidth bound, so Q8 is both smaller AND faster (DS-Coder-V2-Lite: −72% memory, +28% tok/s). CPU mode only — GPU forward passes use their own VRAM KV buffers. |
| `-Dattn.flash=true` | `false` | FlashAttention-style online-softmax single-pass attention. Bit-identical to the legacy 2-pass; on Java/CPU it's 6-15% **slower** than the SIMD-optimized 2-pass and is kept opt-in for future GPU HBM-bound use. |
| `-Dcuda.nograph=true` | `false` | Disable CUDA graph; use per-layer kernel launches |
| `-Dcuda.cublas=true` | `false` | Enable cuBLAS for matmul (dequantizes Q4_K to FP16, uses `libcublas.so`). Useful on high-bandwidth GPUs (A100, H100). On RTX 4050, custom Q4_K kernels + CUDA graph are faster |
| `-Dcuda.cublas.fp32=true` | `false` | Use FP32 instead of FP16 for cuBLAS (requires 7x VRAM vs Q4_K) |
| `-Dcuda.dp4a=true` | **`true`** | Enable `__dp4a` int8 dot product across `CudaForwardPass` + `Qwen35CudaForwardPass` + `NemotronHCudaForwardPass`. Covers Q4_K, Q5_K, Q5_0 (v1.11.0-dev), Q8_0 (v1.11.0-dev), IQ4_NL (v1.11.0-dev), IQ4_XS (v1.11.0-dev). Input is quantized to Q8_1 on the fly. **+34% on Llama-1B Q4_K, +42% on Phi-3-mini IQ4_NL, +92% on Nemotron-3-Nano-4B.** Q6_K stays on FP32 (dp4a slower due to unaligned 210-byte block). |
| `-Dcuda.dp4a.q6=true` | `false` | Force Q6_K dp4a kernel. Kept for correctness checks only — measures slower than the FP32 Q6_K kernel on Llama-1B (70.6 vs 72.85 tok/s). |
| `-Dcuda.dp4a.fused_gate_up=true` | `false` | Single fused kernel for gate+up dp4a matmul. Default OFF — marginal/regression (input is already L1-cached; fused kernel adds register pressure). |
| `-Dcuda.dp4a.mw=true` | `false` | llama.cpp-style multi-warp Q4_K dp4a kernel (4 warps × 32 lanes per row). Slower for small models (Llama-1B: 51 vs 70 tok/s); may help larger models with more super-blocks per row. |
| `-Dcuda.q4k.coalesced=true` | `false` | Alternative coalesced Q4_K kernel (all threads process same group) |
| `-Dcuda.q4k.smem=true` | `false` | Q4_K kernel with shared-memory input tiling |
| `-Dcuda.q4k.2warp=true` | `false` | 2-warp-per-row Q4_K (splits column groups with shared-memory partial-sum reduction) |
| `-Dcuda.q4k.cpasync=true` | `false` | cp.async input prefetch (Option C — measured −2.8% on RTX 4050; kept for future high-bandwidth GPU experiments) |
| `-Dcuda.q5k.smem=true` | **`true`** | Shared-memory input kernel for Q5_K (+7%) |
| `-Dcuda.q6k.tiled=true` | **`true`** | Tiled shared-memory Q6_K kernel (256-elem tiles, +9%) |
| `-Dcuda.q6k.smem=true` | `false` | Alternative shared-memory Q6_K (vs tiled) |
| `-Dcuda.deltanet.v2=true` | **`true`** | Float4-vectorized DeltaNet kernel (+4%) |
| `-Dcuda.prebuilt=true` | `false` | Load pre-built CUDA cubin instead of compiling via NVRTC (Option A — measured ±1%; no shipped artifacts) |
| `-Dcuda.profile=true` | `false` | Per-section GPU timing (adds sync barriers, ~15-25% overhead) |
| `-Dcpu.profile=true` | `false` | Per-layer CPU timing (standard `InferenceEngine` only) |
| `-Dgemma4.nople=true` | `false` | Disable PLE pre-computation in Gemma 4 forward pass (debug flag) |

### HuggingFace Download

| Option | Type | Default | Description |
|---|---|---|---|
| `--download` | String | — | Download GGUF model from HuggingFace. Format: `owner/repo` (auto-selects Q4_K_M) or `owner/repo/filename.gguf` (specific file). |
| `--hf-token` | String | — | HuggingFace API token for gated/private model access. |

### Fine-Tuning

| Option | Type | Default | Description |
|---|---|---|---|
| `--fine-tune` | Flag | false | Start LoRA fine-tuning pipeline |
| `--target-model` | String | — | Target model for fine-tuning |

See [FINE-TUNING.md](FINE-TUNING.md) for full fine-tuning documentation.

## REST API (web mode)

When started with `--web`, the server exposes the following APIs. Full documentation in `REST-API.md`.

### OpenAI-compatible API (`/v1/*`)

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Chat completion (streaming + non-streaming), tool calling, JSON mode |
| `/v1/embeddings` | POST | Text embeddings (L2-normalized vectors) |
| `/v1/models` | GET | List available/loaded models |

Works with standard OpenAI clients (Open WebUI, LangChain, LiteLLM, Cursor, Continue.dev, etc.). The `Authorization: Bearer <token>` header is accepted and ignored. Supports architecture-aware tool calling (SmolLM3 uses Hermes-style `<tool_call>` XML tags; other models use generic JSON format), JSON mode, and thinking/reasoning mode. See [TOOL-CALLING.md](TOOL-CALLING.md) for full tool calling documentation with examples.

### Anthropic Messages API (`/v1/messages`)

| Endpoint | Method | Description |
|---|---|---|
| `/v1/messages` | POST | Chat completion (streaming + non-streaming), Anthropic message format |
| `/v1/messages/count_tokens` | POST | Token counting for a message payload |

Compatible with Claude Code and other Anthropic API clients. The `x-api-key` header is accepted and ignored.

### Management API (`/api/*`)

| Endpoint | Method | Description |
|---|---|---|
| `/api/models` | GET | List GGUF files in the directory |
| `/api/models/load` | POST | Load a model: `{"path": "...", "contextLength": 2048}` |
| `/api/models/unload` | POST | Unload the current model |
| `/api/models/info` | GET | Loaded model metadata (includes `gpuLayers`, `gpuDeviceName`, `moeOptimizedGpu`) |
| `/api/chat` | POST | Generation with streaming (Server-Sent Events) |
| `/api/chat/stop` | POST | Stop the current generation |

### Chat Persistence API (`/api/chats/*`)

| Endpoint | Method | Description |
|---|---|---|
| `/api/chats` | GET | List conversations |
| `/api/chats` | POST | Create new conversation |
| `/api/chats/{id}` | GET | Get conversation with message tree |
| `/api/chats/{id}` | DELETE | Delete conversation |
| `/api/chats/{id}/title` | PUT | Rename conversation |
| `/api/chats/{id}/messages` | POST | Add message |
| `/api/chats/{id}/messages/{msgId}` | PUT | Edit message (creates branch) |
| `/api/chats/{id}/settings` | PUT | Update per-conversation settings |

## Java API

```java
// Loading
LLMEngine engine = LLMEngine.load(Path.of("model.gguf"), 2048);

// Single generation
GenerationResponse resp = engine.generate(
    GenerationRequest.builder()
        .prompt("Hello, how are you?")
        .maxTokens(256)
        .samplerConfig(SamplerConfig.builder()
            .temperature(0.7f)
            .topK(40)
            .topP(0.9f)
            .repetitionPenalty(1.1f)
            .build())
        .build()
);
System.out.println(resp.text());

// Streaming generation
engine.generate(request, (token, id) -> {
    System.out.print(token);
    return true;  // return false to stop
});

// Batch generation (uses StructuredTaskScope on Java 25, thread pool otherwise)
List<GenerationResponse> responses = engine.generateBatch(requestList);

// Loading with GPU
GpuConfig gpu = new GpuConfig();
gpu.setEnabled(true);
gpu.setDeviceId(0);
LLMEngine gpuEngine = LLMEngine.load(Path.of("model.gguf"), 2048, gpu);

// Cleanup
engine.close();
```

`LLMEngine` is thread-safe: model weights are immutable (memory-mapped) and each `generate()` call creates its own inference state.

## Supported Architectures

| Architecture | GGUF Key | Tokenizer | Chat Template |
|---|---|---|---|
| Llama (1/2/3) | `llama` | BPE (gpt2) / SentencePiece | `<\|start_header_id\|>user<\|end_header_id\|>` |
| Qwen2 | `qwen2` | BPE (gpt2) | `<\|im_start\|>user` |
| Qwen3 | `qwen3` | BPE (gpt2) | `<\|im_start\|>user` |
| Qwen3 MoE | `qwen3moe` | BPE (gpt2) | `<\|im_start\|>user` |
| Qwen3.5 | `qwen3` | BPE (gpt2) | `<\|im_start\|>user` |
| SmolLM3 | `smollm3` | BPE (gpt2) | `<\|im_start\|>user` |
| DeepSeek2 | `deepseek2` | BPE (gpt2) | `User: ... Assistant:` |
| GLM-4.7-Flash | `deepseek2` | SentencePiece | `[gMASK]<sop><\|user\|>` (auto-detected) |
| GLM4 | `glm4` | SentencePiece | `[gMASK]<sop><\|user\|>` |
| Gemma 2 | `gemma2` | SentencePiece | `<start_of_turn>user` |
| Gemma 3 | `gemma3` | SentencePiece | `<start_of_turn>user` |
| Gemma 3n (PLE) | `gemma3n` | SentencePiece | `<start_of_turn>user` |
| Gemma 4 (PLE, V-norm, layer_output_scale) | `gemma4` | BPE (gemma4 mode) | `<\|turn>...<turn\|>` |
| Phi-3/4 | `phi3` | BPE (gpt2) | `<\|user\|>` |
| Mistral3/Devstral | `mistral3` | SentencePiece | `[INST]` |
| Command-R/Cohere (Aya-23) | `command-r` | SentencePiece | `<\|START_OF_TURN_TOKEN\|><\|USER_TOKEN\|>` |
| OLMo2 | `olmo2` | BPE (gpt2) | `<\|user\|>` |
| Olmo 3 (ChatML variant) | `olmo2` (auto-detected) | BPE (gpt2) | `<\|im_start\|>user` |
| Falcon3 | `falcon3` | BPE (gpt2) | `<\|user\|>` |
| GPT-OSS (Sonar) | `llama` (MoE) | BPE (gpt2) | `<\|start_header_id\|>user<\|end_header_id\|>` |
| Granite 3.3 | `granite` | BPE (gpt2) | `<\|start_of_role\|>user<\|end_of_role\|>` |
| Granite Hybrid | `granitehybrid` | BPE (gpt2) | `<\|start_of_role\|>user<\|end_of_role\|>` |
| Nemotron-H (Mamba-2 hybrid) | `nemotron_h` | BPE (gpt2) | `<\|im_start\|>user` |

The architecture is automatically detected from the `general.architecture` field in GGUF metadata.

## Benchmarks

Hardware: Intel Core Ultra 7 155H + NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM, 192 GB/s), Java 25, SimdVectorOps.

**34+ models tested** across 20 architectures (Llama, Qwen2, Qwen3, Qwen3MoE, Qwen3.5, SmolLM3, DeepSeek2, GLM-4.7-Flash, GLM4, Gemma 2/3/3n/4, Phi-3/4, Mistral3/Devstral, Falcon3, Command-R/Cohere, OLMo2, GPT-OSS, Nemotron-H, Granite 3.3, Granite Hybrid).

### Top results — CUDA GPU (v1.11.0-dev sweep, ranked by tok/s)

Best-of-3 per model, T=0.0, 120 tokens, --context-length 512. All output is deterministic across the 3 runs (identical hash). Quality columns: PPL = normalized perplexity [0..1] · Coh = coherence [0..1] · Agg = composite quality score [0..1] (EXCELLENT ≥0.80, GOOD ≥0.55, FAIR ≥0.40).

| # | Model | Quant | Arch | GPU Path | tok/s | Agg |
|--:|-------|-------|------|----------|------:|----:|
| 1 | Qwen3-0.6B | Q8_0 | qwen3 | CUDA graph + dp4a (Q8_0) | **99.4** | 0.83 |
| 2 | granite-4.0-h-tiny | Q4_K_M | granite-hybrid | CUDA graph + dp4a (NemotronH) | **94.3** | 0.46 |
| 3 | OLMo-2-1B-Instruct | Q4_K_M | olmo2 | CUDA graph + dp4a (Q4_K) | **84.5** | 0.85 |
| 4 | Llama-3.2-1B-Instruct | Q4_K_M | llama | CUDA graph + dp4a (Q4_K) | **84.3** | 0.84 |
| 5 | granite-3.3-2b-instruct | Q4_K_M | granite | CUDA graph + dp4a (Q4_K) | **50.3** | 0.84 |
| 6 | Falcon3-3B-Instruct | Q4_K_M | falcon3 | CUDA graph + dp4a (Q4_K) | **44.1** | 0.84 |
| 7 | Qwen3-1.7B | Q8_0 | qwen3 | CUDA graph + dp4a (Q8_0) | **42.7** | 0.81 |
| 8 | Gemma-3-1B-it | Q4_K_M | gemma3 | CUDA graph + dp4a (Q4_K + Q5_0) | **41.2** | 0.51 |
| 9 | granite-4.0-h-micro | Q4_K_M | granite-hybrid | CUDA graph + dp4a + integrated SwiGLU | **34.1** | 0.60 |
| 10 | Llama-3.2-1B-Instruct | IQ4_NL | llama | CUDA graph + dp4a (IQ4_NL) | **32.6** | 0.85 |
| 11 | Qwen3-4B | Q4_K_M | qwen3 | CUDA graph + dp4a (Q4_K) | **32.3** | 0.85 |
| 12 | Phi-4-mini-Instruct | Q4_K_M | phi3 | CUDA graph + dp4a (Q4_K) | **32.1** | **0.98** |
| 13 | NVIDIA-Nemotron-3-Nano-4B | Q4_K_M | nemotron-h | Per-layer + dp4a (NemotronH) | **22.6** | 0.84 |
| 14 | Mistral-7B-Instruct-v0.3 | Q4_K_M | mistral3 | CUDA graph + dp4a (Q4_K) | **20.1** | 0.82 |
| 15 | Phi-3-mini-4k-instruct | IQ4_NL | phi3 | CUDA graph + dp4a (IQ4_NL) | **12.0** | 0.77 |
| 16 | Gemma-2-2b-it | IQ4_XS | gemma2 | CUDA graph + dp4a (IQ4_XS) | **10.8** | 0.66 |

Full results, methodology, headline gains and llama.cpp comparison in [BENCHMARKS.md](BENCHMARKS.md). Per-kernel profiling in [PERFORMANCE-ANALYSIS.md](PERFORMANCE-ANALYSIS.md). Sweep is reproducible via `bench-v1.11.0-dev.sh`.

### LLMPlayer vs llama.cpp (CPU comparison, v1.11.0-dev refresh)

LLMPlayer CUDA graph vs llama.cpp CPU-only (`llama-cpp-python`, same hardware, same GGUF files):

| Model | Params | Quant | LLMPlayer GPU | LLMPlayer CPU | llama.cpp CPU | GPU vs llama.cpp |
|-------|--------|-------|-------------:|-------------:|-------------:|---------:|
| Qwen3-0.6B | 0.6B | Q8_0 | **99.4** | 5.9 | 4.5 | **22.1×** |
| granite-4.0-h-tiny | ~400M | Q4_K_M | **94.3** | — | — | — |
| OLMo-2-1B | 1B | Q4_K_M | **84.5** | — | — | — |
| Llama-3.2-1B | 1B | Q4_K_M | **84.3** | 3.5 | 4.3 | **19.6×** |
| granite-3.3-2B | 2B | Q4_K_M | **50.3** | 1.3 | 1.8 | **27.9×** |
| Falcon3-3B | 3B | Q4_K_M | **44.1** | 1.3 | 4.8 | **9.2×** |
| Qwen3-1.7B | 1.7B | Q8_0 | **42.7** | 2.9 | 4.2 | **10.2×** |
| Gemma-3-1B | 1B | Q4_K_M | **41.2** | — | — | — |
| granite-4.0-h-micro | 1.8B | Q4_K_M | **34.1** | ~8 | 41.8 | **0.82×** (85%) |
| Qwen3-4B | 4B | Q4_K_M | **32.3** | 0.9 | 1.9 | **17.0×** |
| Phi-4-mini | 3.8B | Q4_K_M | **32.1** | — | — | — |
| Nemotron-3-Nano-4B | 4B (hybrid) | Q4_K_M | **22.6** | — | 35.6 | **0.63×** (63%) |
| Mistral-7B | 7B | Q4_K_M | **20.1** | — | — | — |

LLMPlayer GPU is now **9–28× faster** than llama.cpp CPU-only on dense Q4_K models (was 5–17× in v1.10.x), and the dp4a expansion + Granite Hybrid GPU path bring the two hybrid-arch outliers (Granite Hybrid, Nemotron-H) up to **63–85% of llama.cpp's GPU baseline** (was 19% and 29% respectively). On CPU-only, llama.cpp is still 1.2–3.7× faster than LLMPlayer (expected: C/C++ native SIMD vs Java Vector API).

### GPU strategy summary

| Strategy | When Used | VRAM Needed | Typical Speed |
|----------|-----------|-------------|---------------|
| Full offload + CUDA graph | Dense model fits in VRAM, supported architecture | 770–4794 MB | 8–52 tok/s |
| Full offload + CUDA graph (Qwen3.5) | Qwen3.5 hybrid DeltaNet+attention fits in VRAM | 1211–5417 MB | 7–28 tok/s |
| Full offload + per-layer (Nemotron-H) | Nemotron-H hybrid Mamba-2+attention+FFN | 2765 MB | ~10 tok/s |
| Full offload + per-tensor | Model fits in VRAM, architecture not supported for graph | 770–5000 MB | 1–7 tok/s |
| MoE-optimized + expert cache | MoE model, attention fits in VRAM | 517–913 MB | 0.7–2.5 tok/s |
| Partial offload | Dense/hybrid model, first-N layers on GPU | 4615–4909 MB | 0.3–0.7 tok/s |

## Fine-Tuning

Built-in LoRA fine-tuning pipeline. Full documentation in [FINE-TUNING.md](FINE-TUNING.md).

```bash
# Fine-tune from source code
./run.sh --fine-tune --target-model base.gguf --source ./my-codebase --model generator.gguf

# Fine-tune from documents
./run.sh --fine-tune --target-model base.gguf --documents ./docs --model generator.gguf

# Fine-tune from pre-built dataset
./run.sh --fine-tune --target-model base.gguf --train-dataset dataset.jsonl
```

The pipeline: analyze target model → chunk input data → generate Q&A pairs (using a generator LLM) → LoRA training → merge adapters → export as new GGUF file.
