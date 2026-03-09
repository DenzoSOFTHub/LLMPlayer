# LLMPlayer

Pure Java LLM inference engine for running GGUF models locally. Zero external dependencies — uses only the JDK. Supports Llama, Qwen2, Qwen3, Qwen3MoE, Qwen3.5, DeepSeek2, GLM4, Gemma 2/3, Phi-3/4, and Mistral3/Devstral architectures with quantized formats (Q2_K, Q3_K, Q4_0, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL, MXFP4, BF16, F16, F32). Includes GPU acceleration via CUDA and OpenCL (Panama FFM, zero native dependencies), CUDA graph mode for up to 55 tok/s on RTX 4050, HuggingFace model download, and a built-in LoRA fine-tuning pipeline.

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

| Option | Alias | Type | Default | Description |
|---|---|---|---|---|
| `--model` | `-m` | String | — | Path to the GGUF file |
| `--prompt` | `-p` | String | — | Prompt for single generation |
| `--interactive` | `-i` | Flag | false | Interactive chat mode |
| `--max-tokens` | `-n` | Integer | 256 | Maximum number of tokens to generate |
| `--temperature` | `-t` | Float | 0.7 | Sampling temperature (0 = deterministic, >1 = more random) |
| `--top-k` | — | Integer | 40 | Keep only the K most probable tokens |
| `--top-p` | — | Float | 0.9 | Nucleus sampling: cumulative probability threshold |
| `--repetition-penalty` | — | Float | 1.1 | Penalty for repeated tokens (>1 reduces repetition) |
| `--seed` | — | Long | random | Seed for reproducibility |
| `--threads` | — | Integer | num CPUs | Number of worker threads |
| `--context-length` | `-c` | Integer | 2048 | Maximum context length (tokens) |
| `--info` | — | Flag | false | Show model info and exit |
| `--web` | `-w` | Flag | false | Start the web server |
| `--port` | — | Integer | 8080 | Web server port |
| `--gguf-dir` | — | String | `gguf` | GGUF file directory |
| `--gpu` | — | Flag | false | Enable GPU acceleration |
| `--gpu-device` | — | Integer | 0 | GPU device index |
| `--gpu-backend` | — | String | `auto` | GPU backend: `auto`, `cuda`, `opencl` |
| `--gpu-layers` | — | Integer | -1 | GPU layers (-1 = auto-detect, 0 = CPU only) |
| `--gpu-list` | — | Flag | false | List GPU devices and exit |
| `--download` | — | String | — | Download GGUF model from HuggingFace (e.g. `owner/repo`) |
| `--hf-token` | — | String | — | HuggingFace API token for private/gated repos |
| `--no-gpu` | — | Flag | false | Disable GPU, use CPU only |
| `--fine-tune` | — | Flag | false | Start LoRA fine-tuning pipeline |
| `--target-model` | — | String | — | Target model for fine-tuning |
| `--help` | `-h` | Flag | false | Show help |

## REST API (web mode)

When started with `--web`, the server exposes the following APIs. Full documentation in `REST-API.md`.

### OpenAI-compatible API (`/v1/*`)

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Chat completion (streaming + non-streaming), tool calling, JSON mode |
| `/v1/embeddings` | POST | Text embeddings (L2-normalized vectors) |
| `/v1/models` | GET | List available/loaded models |

Works with standard OpenAI clients (Open WebUI, LangChain, LiteLLM, Cursor, Continue.dev, etc.). The `Authorization: Bearer <token>` header is accepted and ignored.

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
| DeepSeek2 | `deepseek2` | BPE (gpt2) | `User: ... Assistant:` |
| GLM4 | `glm4` | SentencePiece | `[gMASK]<sop><\|user\|>` |
| Gemma 2 | `gemma2` | SentencePiece | `<start_of_turn>user` |
| Gemma 3 | `gemma3` | SentencePiece | `<start_of_turn>user` |
| Phi-3/4 | `phi3` | BPE (gpt2) | `<\|user\|>` |
| Mistral3/Devstral | `mistral3` | SentencePiece | `[INST]` |

The architecture is automatically detected from the `general.architecture` field in GGUF metadata.

## Benchmarks

Hardware: Intel Core Ultra 7 155H + NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM, 192 GB/s), Java 25, SimdVectorOps.

### Top results (ranked by tok/s)

| # | Model | Params | Quant | GPU Config | tok/s |
|--:|-------|--------|-------|------------|------:|
| 1 | Llama-3.2-1B-Instruct | 1B | Q4_K_M | CUDA graph | 54.7 |
| 2 | Qwen2.5-Coder-1.5B-Instruct | 1.5B | Q4_K_M | CUDA graph | 41.5 |
| 3 | Gemma-3-1B-it | 1B | Q4_K_M | CUDA graph | 35.7 |
| 4 | Llama-3.2-1B-Instruct | 1B | IQ4_NL | CUDA graph | 28.7 |
| 5 | OLMo-2-1B-Instruct | 1B | Q4_K_M | CUDA graph | 25.5 |
| 6 | Llama-3.2-3B-Instruct | 3B | Q4_K_M | CUDA graph | 23.8 |
| 7 | Qwen2.5-Coder-3B-Instruct | 3B | Q4_K_M | CUDA graph | 21.8 |
| 8 | Qwen3-4B | 4B | Q4_K_M | CUDA graph | 19.0 |
| 9 | Qwen2.5-Coder-7B-Instruct | 7B | Q4_K_M | CUDA graph | 11.3 |
| 10 | DeepSeek-R1-Qwen3-8B | 8B | Q4_K_M | CUDA graph | 9.3 |

Full benchmark results (33 models) in [BENCHMARKS.md](BENCHMARKS.md). Detailed performance analysis in [PERFORMANCE-ANALYSIS.md](PERFORMANCE-ANALYSIS.md).

### GPU strategy summary

| Strategy | When Used | VRAM Needed | Typical Speed |
|----------|-----------|-------------|---------------|
| Full offload + CUDA graph | Dense model fits in VRAM, all tensors have CUDA kernels | 770–4822 MB | 7–55 tok/s |
| Full offload + per-tensor | Model fits in VRAM, architecture not supported for graph | 770–2600 MB | 6–12 tok/s |
| MoE-optimized | MoE model, attention fits in VRAM | 517–913 MB | 0.8–1.6 tok/s |
| Partial offload | Dense/hybrid model, first-N layers on GPU | 4615–4909 MB | 0.2–1.6 tok/s |

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
