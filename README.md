# LLMPlayer v1.10.1

Pure Java LLM inference engine for running GGUF models locally. Zero external dependencies — uses only the JDK. Supports 20 architectures including Llama, Qwen2/3/3.5, SmolLM3, DeepSeek2, Gemma 2/3/3n/4, Phi-3/4, Mistral3/Devstral, Falcon3, Granite 3.3, **Granite Hybrid**, and **Nemotron-H** (hybrid Mamba-2 + Transformer). 18 quantized formats with 16 dedicated CUDA kernels. Includes CUDA GPU acceleration with graph mode (~57 tok/s on RTX 4050 for Llama-3.2-1B), cuBLAS support (opt-in), thinking/reasoning mode, architecture-aware tool calling, HuggingFace model download, JMX runtime metrics with rolling window, smoke test suite for all architectures, and a built-in LoRA fine-tuning pipeline.

### What's new in v1.10.1 — Top-10 audit vs llama.cpp

Six commit's worth of correctness fixes, sampler additions, and a major memory optimization driven by an audit of LLMPlayer against the llama.cpp reference implementation. See `BENCHMARKS.md` for measured deltas.

- **Q8_0 KV cache** (`-Dkv.q8=true`) — block-quantized int8 KV with FP32 scales per 32-elem block. **3.56× memory reduction** on all 5 inference engines including the asymmetric MLA path for DeepSeek2 (separate K/V dims). Bit-identical greedy output on dense models. Headline: **DeepSeek-Coder-V2-Lite Q4_K_M CPU mode goes from 1080 MB → 303 MB KV and 0.7 → 0.9 tok/s** (faster, not just smaller — MLA is DRAM-bandwidth-bound).
- **Command-R LayerNorm fix** — `LayerNorm` (centered + scaled, no bias) now used in place of `RMSNorm` for Command-R / Cohere2 (matches llama.cpp `LLM_NORM`). Cohere2 also gets a separate enum, NoPE-on-global-layers handling, and the right ISWA pattern (`set_swa_pattern(4)`).
- **DS-V3 / GLM-4.7-Flash routing fix** — `exp_probs_b` is now applied AFTER sigmoid (not before), and only as a selection score; mix weights come from the unbiased `probs`. Matches llama.cpp `build_moe_ffn`.
- **New samplers** (opt-in): `--min-p`, `--mirostat 2 --mirostat-tau 5.0`, `--dry-multiplier 0.8`. Pipeline: `DRY → rep_penalty → temp → top-K → softmax → min-P → top-P → (mirostat | multinomial)`.
- **Wo bias / output bias loading** — `attn_output.bias` and top-level `output.bias` now loaded and applied for Qwen2 / SmolLM3 / Command-R variants that ship them.
- **GC churn reduction** — Qwen3.5 and Nemotron-H pre-cache all per-layer norm weights and pre-allocate conv1d output buffers.
- **MoE routing safety** — F16-epsilon clamp on weight-sum normalization in `Qwen3MoEInferenceEngine.moeFFN` (anti-NaN guardrail, matches llama.cpp `ggml_clamp`).
- **Gemma 3n / Gemma 4 broken-support warning** — empirically confirmed that PLE-only path (without AltUp + Laurel) produces random multi-language tokens. Loud WARNING banner at engine creation so users know the model is not yet usable.
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
| `-Dcuda.dp4a=true` | `false` | Enable `__dp4a` int8 dot product for Qwen3.5 matmul (~2% gain on consumer GPUs, more on datacenter GPUs with dedicated INT8 units) |
| `-Dcuda.q4k.coalesced=true` | `false` | Alternative coalesced Q4_K kernel (all threads process same group) |
| `-Dcuda.q4k.smem=true` | `false` | Q4_K kernel with shared-memory input tiling |
| `-Dcuda.profile=true` | `false` | Per-section GPU timing (adds sync barriers, ~15-25% overhead) |

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
| Phi-3/4 | `phi3` | BPE (gpt2) | `<\|user\|>` |
| Mistral3/Devstral | `mistral3` | SentencePiece | `[INST]` |
| Command-R/Cohere (Aya-23) | `command-r` | SentencePiece | `<\|START_OF_TURN_TOKEN\|><\|USER_TOKEN\|>` |
| OLMo2 | `olmo2` | BPE (gpt2) | `<\|user\|>` |
| GPT-OSS (Sonar) | `llama` (MoE) | BPE (gpt2) | `<\|start_header_id\|>user<\|end_header_id\|>` |
| Nemotron-H (Mamba-2 hybrid) | `nemotron_h` | BPE (gpt2) | `<\|im_start\|>user` |

The architecture is automatically detected from the `general.architecture` field in GGUF metadata.

## Benchmarks

Hardware: Intel Core Ultra 7 155H + NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM, 192 GB/s), Java 25, SimdVectorOps.

**34+ models tested** across 20 architectures (Llama, Qwen2, Qwen3, Qwen3MoE, Qwen3.5, SmolLM3, DeepSeek2, GLM-4.7-Flash, GLM4, Gemma 2/3/3n/4, Phi-3/4, Mistral3/Devstral, Falcon3, Command-R/Cohere, OLMo2, GPT-OSS, Nemotron-H, Granite 3.3, Granite Hybrid).

### Top results — CUDA GPU (ranked by tok/s)

| # | Model | Params | Quant | GPU Config | tok/s |
|--:|-------|--------|-------|------------|------:|
| 1 | OLMo-2-1B-Instruct | 1B | Q4_K_M | CUDA graph | 52.1 |
| 2 | Llama-3.2-1B-Instruct | 1B | Q4_K_M | CUDA graph | 48.8 |
| 3 | Qwen2.5-Coder-1.5B-Instruct | 1.5B | Q4_K_M | CUDA graph | 40.8 |
| 4 | Gemma-3-1B-it | 1B | Q4_K_M | CUDA graph | 33.1 |
| 5 | Qwen3.5-2B-Claude-4.6 | 2B | Q4_K_M | CUDA graph (Qwen35) | 28.2 |
| 6 | Llama-3.2-1B-Instruct | 1B | IQ4_NL | CUDA graph | 27.8 |
| 7 | SmolLM3-3B | 3B | Q4_K_M | CUDA graph | 22.9 |
| 8 | Llama-3.2-3B-Instruct | 3B | Q4_K_M | CUDA graph | 22.6 |
| 9 | Qwen2.5-Coder-3B-Instruct | 3B | Q4_K_M | CUDA graph | 21.5 |
| 10 | Gemma-3-4B-it | 4B | Q4_K_M | CUDA graph | 16.3 |
| 11 | Qwen3-4B | 4B | Q4_K_M | CUDA graph | 18.3 |
| 11 | Phi-4-mini-Instruct | 3.8B | Q4_K_M | CUDA graph | 14.5 |
| 12 | Qwen3.5-4B-Claude-4.6 | 4B | Q4_K_M | CUDA graph (Qwen35) | 13.7 |
| 13 | Mistral-7B-Instruct-v0.3 | 7B | Q4_K_M | CUDA graph | 11.8 |
| 14 | Qwen2.5-Coder-7B-Instruct | 7B | Q4_K_M | CUDA graph | 11.2 |
| 15 | NVIDIA-Nemotron-3-Nano-4B | 4B (hybrid) | Q4_K_M | Per-layer (NemotronH) | 9.9 |
| 16 | Qwen3.5-9B-Claude-4.6 | 9B | Q4_K_M | CUDA graph (Qwen35) | 7.9 |

Full benchmark results (27 models, CUDA GPU) in [BENCHMARKS.md](BENCHMARKS.md). Detailed performance analysis in [PERFORMANCE-ANALYSIS.md](PERFORMANCE-ANALYSIS.md).

### LLMPlayer vs llama.cpp (CPU comparison)

LLMPlayer CUDA graph vs llama.cpp CPU-only (`llama-cpp-python`, same hardware, same GGUF files):

| Model | Params | Quant | LLMPlayer GPU | LLMPlayer CPU | llama.cpp CPU | GPU vs llama.cpp |
|-------|--------|-------|-------------:|-------------:|-------------:|---------:|
| Qwen3-0.6B | 0.6B | Q8_0 | **74.6** | 5.9 | 4.5 | **16.6x** |
| Llama-3.2-1B | 1B | Q4_K_M | **52.8** | 3.5 | 4.3 | **12.3x** |
| Qwen3-1.7B | 1.7B | Q8_0 | **32.1** | 2.9 | 4.2 | **7.6x** |
| Granite-3.3-2B | 2B | Q4_K_M | **26.4** | 1.3 | 1.8 | **14.7x** |
| Falcon3-3B | 3B | Q4_K_M | **22.9** | 1.3 | 4.8 | **4.8x** |
| SmolLM3-3B | 3B | Q4_K_M | **22.4** | 1.3 | N/A | — |
| Qwen3-4B | 4B | Q4_K_M | **18.3** | 0.9 | 1.9 | **9.6x** |
| Qwen3-8B | 8B | Q4_K_M | **10.2** | 0.5 | 1.7 | **6.0x** |

LLMPlayer with CUDA graph is **5–17x faster** than llama.cpp running CPU-only on the same machine. On CPU-only, llama.cpp is 1.2–3.7x faster than LLMPlayer (expected: C/C++ native SIMD vs Java Vector API). The GPU acceleration closes this gap and provides a significant net advantage.

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
