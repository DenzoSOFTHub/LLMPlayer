# LLMPlayer

Pure Java LLM inference engine for running GGUF models locally. Zero external dependencies — uses only the JDK. Supports Llama, Qwen2, Qwen3, Qwen3MoE, DeepSeek2, GLM4, Phi-3/4, and Mistral3/Devstral architectures with quantized formats (Q2_K, Q3_K, Q4_0, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0, BF16, F16, F32).

## Requirements

- **Java 8** — base functionality (no SIMD, no GPU)
- **Java 21+** — adds SIMD (Vector API), optimized memory mapping (Panama FFI), GPU acceleration (OpenCL)
- **Java 25** — adds advanced parallelism (StructuredTaskScope, virtual thread matmul)
- **Maven 3.x** — for building
- **OpenCL drivers** — only needed for GPU usage (e.g. `libOpenCL.so` on Linux)

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

Starts an HTTP server at `http://localhost:8080` with a web interface and REST API. The port is configurable (default: 8080).

### Model info

```bash
./run.sh --model path/to/model.gguf --info
```

Prints model metadata (architecture, layers, dimensions, vocabulary) and exits.

## GPU Acceleration (OpenCL)

Requires Java 21+ and installed OpenCL drivers.

### List available devices

```bash
./run.sh --gpu-list
```

Shows all detected OpenCL devices (GPUs, OpenCL CPUs, accelerators) with name and memory. If no devices are found, verify that OpenCL drivers are installed (`libOpenCL.so` on Linux, vendor GPU drivers on Windows/macOS).

### Enable GPU

```bash
# Use the first GPU device (device 0)
./run.sh --model model.gguf --gpu --prompt "Hello" --max-tokens 256

# Use a specific device
./run.sh --model model.gguf --gpu-device 1 --prompt "Hello" --max-tokens 256
```

The `--gpu-device N` flag selects a device by index (as shown by `--gpu-list`) and automatically enables the GPU. The `--gpu` flag alone uses device 0.

### How it works

When GPU is enabled, the system:
1. Initializes an OpenCL context on the selected device
2. Registers a global `GpuBufferManager` in `TensorFactory`
3. For each tensor created during model loading, attempts the GPU variant first (e.g. `Q4_KGpuTensor`), then falls back to the CPU variant if unavailable
4. Compute-heavy operations (matmul, RMSNorm, softmax, SiLU, etc.) are executed via OpenCL kernels compiled on-demand

GPU-supported quantized formats: F32, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0. Other formats automatically fall back to CPU.

If Java is < 21 or OpenCL drivers are not present, the system prints a warning and continues in CPU-only mode.

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
| GPU acceleration (OpenCL) | No | Yes | Yes |
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
| `--gpu-list` | — | Flag | false | List GPU devices and exit |
| `--help` | `-h` | Flag | false | Show help |

## REST API (web mode)

When started with `--web`, the server exposes the following APIs:

### Models

| Endpoint | Method | Description |
|---|---|---|
| `/api/models` | GET | List GGUF files in the directory |
| `/api/models/load` | POST | Load a model: `{"path": "...", "contextLength": 2048}` |
| `/api/models/unload` | POST | Unload the current model |
| `/api/models/info` | GET | Loaded model metadata (includes `gpuLayers`, `gpuDeviceName`, `moeOptimizedGpu`) |

### Chat

| Endpoint | Method | Description |
|---|---|---|
| `/api/chat` | POST | Generation with streaming (Server-Sent Events) |
| `/api/chat/stop` | POST | Stop the current generation |

`/api/chat` request:
```json
{
  "prompt": "Your message",
  "systemMessage": "Optional system message",
  "temperature": 0.7,
  "maxTokens": 256,
  "topK": 40,
  "topP": 0.9,
  "repPenalty": 1.1
}
```

Response is an SSE stream:
```
data: {"token": "hello", "done": false}
data: {"token": " world", "done": false}
data: {"done": true, "stats": {"tokenCount": 10, "promptTokenCount": 5, "tokensPerSecond": 25.5, "timeMs": 392}}
```

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
| DeepSeek2 | `deepseek2` | BPE (gpt2) | `User: ... Assistant:` |
| GLM4 | `glm4` | SentencePiece | `[gMASK]<sop><\|user\|>` |
| Phi-3/4 | `phi3` | BPE (gpt2) | `<\|user\|>` |
| Mistral3/Devstral | `mistral3` | SentencePiece | `[INST]` |

The architecture is automatically detected from the `general.architecture` field in GGUF metadata.

## Benchmarks

### MoE-optimized GPU placement

Hardware: Intel Core Ultra 7 155H + NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM), Java 25, SimdVectorOps.

| Model | Type | GPU Strategy | Layers on GPU | VRAM Used | tok/s | Quality |
|-------|------|--------------|---------------|-----------|-------|---------|
| Qwen3-Coder-30B-A3B Q4_K_M | MoE (128 experts, top-8) | MoE-optimized | 48/48 attn | 540 MB | 1.7 | Perplexity 0.98, Coherence 0.99 |
| DeepSeek-Coder-V2-Lite Q4_K_M | MoE (64 experts, top-6+2shared) | MoE-optimized | 27/27 attn | 517 MB | 2.1 | Perplexity 0.85, Coherence 1.00 |
| Llama-3.2-3B Q3_K_L | Dense | Full offload | 28/28 all | 1731 MB | 5.9 | Perplexity 0.96, Coherence 0.95 |

Test: `--prompt "Write a Java class that calculates factorial" --max-tokens 40 --context-length 256 --gpu --gpu-device 1`.

Key takeaway: MoE-optimized placement puts 100% of attention on GPU using only ~540 MB VRAM for a 17.3 GB model. With standard first-N-layers, only ~2/48 layers would fit in 6 GB VRAM for Qwen3-Coder-30B.
