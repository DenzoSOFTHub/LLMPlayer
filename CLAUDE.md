# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLMPlayer is a pure Java LLM inference engine that runs GGUF models locally. Zero external dependencies — uses only the JDK. Supports Llama, Qwen2, Qwen3, Qwen3MoE, DeepSeek2, GLM4, Phi-3/4, and Mistral3/Devstral architectures with quantized formats (Q2_K through Q8_0, BF16, F16, F32).

## Build & Run Commands

```bash
# Compile (Java 25, default profile — includes java21 + java25 sources)
mvn clean compile

# Compile without Java 25 optimizations (no StructuredTaskScope, no virtual thread matmul)
mvn clean compile -Pjava21

# Compile for Java 8 (no Vector API, no GPU)
mvn clean compile -Pjava8

# Run via shell script (after compile)
./run.sh [options]

# Run via Maven
mvn exec:java

# Run directly (Java 25)
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED --enable-preview \
  -cp target/classes it.denzosoft.llmplayer.LLMPlayer [options]
```

Unit test directory exists but is empty. Integration tests for the OpenAI-compatible API are in `test-openai-api.sh` (requires a running server: `./run.sh --web`, then `bash test-openai-api.sh`). Tests cover 6 architectures (llama, qwen2, qwen3, phi3, deepseek2, mistral3) with streaming, non-streaming, multi-turn, system messages, CORS, error handling, and Bearer token acceptance.

`run.sh` (Linux/macOS) and `run.bat` (Windows) are pre-configured launcher scripts with all required JVM flags for Java 25.

**Note:** The `java21` Maven profile still uses `<release>25</release>` — it requires a Java 25 compiler but excludes the `java25/` source root and `--enable-preview`. Only the `java8` profile actually targets an older compiler release.

## JVM Requirements

Java 21+ builds require these flags at both compile and runtime:
- `--add-modules jdk.incubator.vector` (SIMD Vector API)
- `--enable-native-access=ALL-UNNAMED` (Panama FFI for mmap and OpenCL)

Java 25 builds additionally require:
- `--enable-preview` (StructuredTaskScope, virtual threads)

## Architecture

### Multi-source compilation and reflection loading

The project compiles from three source roots under the default `java25` profile:
- `src/main/java/` — core code, Java 8 compatible
- `src/main/java21/` — Java 21-only code (SIMD tensor ops, OpenCL GPU bindings)
- `src/main/java25/` — Java 25-only code (StructuredTaskScope batch generation, virtual thread matmul)

Classes in `java21/` and `java25/` are **never imported directly** from base code. They are loaded via `Class.forName()` reflection with try/catch fallbacks, allowing graceful degradation on older JVMs. There are six reflection loading sites:

| Loading site | Java 21+ class | Fallback |
|---|---|---|
| `VectorOpsFactory` static init | `SimdVectorOps` | `ScalarOps` |
| `TensorDataFactory.mapFile()` | `MemorySegmentTensorData` | `ByteBufferTensorData` |
| `TensorFactory.tryCreateGpuTensor()` | `Q4_KGpuTensor`, etc. | CPU tensor variant |
| `LLMEngine.initGpu()` | `OpenCLContext` + `GpuBufferManager` | CPU-only |
| `FloatTensor.tryVirtualThreadMatmul()` | `VirtualThreadMatmul` | `IntStream.parallel()` ForkJoinPool matmul |
| `LLMEngine.tryStructuredBatch()` | `StructuredBatchGenerator` | `ExecutorService` thread pool |

**When adding new java21/java25 features:** follow this pattern — put the implementation in the appropriate source root, load it via `Class.forName()` from the base code, and provide a Java 8-compatible fallback.

There is also a 7th reflection site in `CLIRunner.listGpuDevices()` which loads `OpenCLContext.enumerateDevices()` for the `--gpu-list` command.

### Package structure (`it.denzosoft.llmplayer`)

| Package | Purpose |
|---------|---------|
| `api` | Public facade — `LLMEngine` is the main entry point for programmatic use |
| `cli` | CLI argument parsing (`CLIOptions`) and interactive runner |
| `evaluator` | Response quality metrics (perplexity, coherence, length) |
| `gguf` | GGUF file format parser — memory-mapped with parallel preload |
| `gpu` | GPU config (base); OpenCL bindings and buffer management (java21) |
| `inference` | Transformer forward pass — `InferenceEngine` (standard) and `DeepSeek2InferenceEngine` (MLA + MoE) |
| `model` | Model loading, config extraction from GGUF metadata, weight structures |
| `sampler` | Token sampling (temperature, top-k, top-p, repetition penalty) |
| `tensor` | Tensor operations and quantization/dequantization; GPU variants in java21 |
| `tokenizer` | BPE and SentencePiece tokenizers, chat template formatting |
| `ui` | Swing desktop GUI |
| `web` | Embedded HTTP server with HTML web UI, OpenAI-compatible API (`OpenAIHandler`), management API (`ApiHandler`) |

### Key data flow

1. `GGUFParser` memory-maps the model file and extracts metadata + tensor info
2. `ModelLoader` builds `ModelConfig` from metadata, creates weight tensors via `TensorFactory`, and instantiates the tokenizer via `TokenizerFactory`
3. `LLMEngine.load()` wraps everything into the public API, choosing `InferenceEngine` or `DeepSeek2InferenceEngine` based on architecture
4. `generate()` tokenizes the prompt (with chat template if enabled), runs prefill, then auto-regressive decoding with the configured sampler
5. Each `generate()` call creates its own `InferenceState` — model weights are immutable mmap'd memory, making `LLMEngine` thread-safe

### Inference engine dispatch (three paths)

1. **Standard** (`InferenceEngine`): Llama, Qwen2, Qwen3, GLM4, Phi-3/4, Mistral3. Uses `TransformerBlock` → `Attention` (GQA with optional QK-norm/bias) + `SwiGLUFFN`.
2. **DeepSeek2** (`DeepSeek2InferenceEngine`): DeepSeek2 and GLM-4.7-Flash (also uses `deepseek2` GGUF arch). Uses MLA (Multi-Head Latent Attention) + MoE FFN with shared expert. Leading blocks use dense SwiGLU FFN.
3. **Qwen3 MoE** (`Qwen3MoEInferenceEngine`): Qwen3-Coder-30B-A3B and similar. Standard GQA attention with QK-norm + MoE FFN with shared expert. Leading blocks use dense SwiGLU FFN.

### Tensor system

`FloatTensor` is the core abstraction. Each quantization format has a dedicated subclass (e.g., `Q4_KFloatTensor`) that implements dequantization inline. `TensorFactory.create()` selects the implementation by `GGMLType` — trying a GPU variant first (if `GpuBufferManager` is registered), then falling back to CPU. GPU tensor classes live in `java21/` and delegate to OpenCL kernels in `src/main/resources/kernels/`.

### Tokenizer dispatch

`TokenizerFactory` reads `tokenizer.ggml.model` from GGUF metadata: `"gpt2"` or `"bpe"` → `BPETokenizer` (uses merge table, Llama 3 style pre-tokenization regex), anything else → `SentencePieceTokenizer` (score-based). Chat formatting is architecture-specific in `ChatTemplate`, with distinct templates for Llama (`<|start_header_id|>`), Qwen2/3/Qwen3MoE (`<|im_start|>`), GLM4 (`[gMASK]<sop>`), DeepSeek2 (`User: ... Assistant:`), Phi-3/4 (`<|user|>`), and Mistral3 (`[INST]`).

### Launch modes

- **No args** → Swing desktop GUI (`LLMPlayerUI`)
- **`--web`** → Embedded `com.sun.net.httpserver.HttpServer` with HTML web UI (default port 8080), OpenAI-compatible API at `/v1/*`, management API at `/api/*`
- **`--model <path>`** → CLI mode (single prompt with `--prompt` or `--interactive` chat)
- **`--gpu-list`** → enumerate OpenCL devices and exit

### REST API (web mode)

When running with `--web`, the server exposes two API groups. Full documentation in `REST-API.md`.

#### OpenAI-compatible API (`/v1/*`)

Follows the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat) spec. Works with standard OpenAI clients (Open WebUI, LangChain, LiteLLM, Cursor, Continue.dev, etc.). The `Authorization: Bearer <token>` header is accepted and ignored.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/chat/completions` | POST | Chat completion (streaming + non-streaming) |
| `/v1/models` | GET | List available/loaded models |

Implemented in `OpenAIHandler.java`. Uses `ChatTemplate.formatConversation()` for multi-turn message formatting. Supports `messages` array with `system`/`user`/`assistant` roles, `stream`, `temperature`, `max_tokens`, `top_p`, `top_k`, `stop`, `frequency_penalty`, `repetition_penalty`. The `model` field is accepted but ignored (uses the currently loaded model). Streaming sends SSE chunks in OpenAI format (`chat.completion.chunk`) ending with `data: [DONE]`. Non-streaming returns a full `chat.completion` JSON with `choices` and `usage`.

The web UI (`web-ui.html`) uses `/v1/chat/completions` for chat and `/api/*` for model management.

#### Management API (`/api/*`)

LLMPlayer-specific endpoints for model loading, GPU configuration, and diagnostics.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/models` | GET | List available GGUF files in the model directory |
| `/api/models/load` | POST | Load a model: `{"path": "gguf/model.gguf", "contextLength": 2048}` |
| `/api/models/unload` | POST | Unload the current model |
| `/api/models/info` | GET | Get loaded model metadata (includes GPU placement fields) |
| `/api/chat` | POST | Generate with SSE streaming (legacy format) |
| `/api/chat/stop` | POST | Stop current generation |
| `/api/gpu/devices` | GET | Enumerate OpenCL devices |
| `/api/memory/check` | POST | Check RAM availability for a model |
| `/api/hardware/plan` | POST | Build optimal hardware config plan |

### Adding a new model architecture

1. Add enum value to `ModelArchitecture` with its GGUF `general.architecture` string
2. Add any architecture-specific tensor name patterns to `ArchitectureRegistry` (standard names like `blk.{n}.attn_q.weight` are shared across most architectures)
3. Update `ModelConfig.fromMetadata()` if the architecture uses non-standard metadata keys for hyperparameters
4. Add a chat template branch in `ChatTemplate.formatUserMessage()` and `ChatTemplate.formatConversation()`
5. If the architecture's forward pass differs from standard transformer attention+FFN, create a dedicated inference engine class (see `DeepSeek2InferenceEngine` as reference)

### Resources

- `src/main/resources/kernels/` — 12 OpenCL kernel files (matmul variants for each quantization type, plus `rmsnorm.cl`, `softmax.cl`, `silu.cl`, `saxpy.cl`, `accumulate.cl`). Loaded and compiled on-demand by `OpenCLContext`.
- `src/main/resources/web-ui.html` — Single-page web UI served by `WebServer` in `--web` mode.

### GPU placement strategies

Two GPU offloading strategies are available, selected automatically based on model architecture and VRAM:

#### First-N-layers (dense models)
The default for dense architectures (Llama, Qwen2, GLM4, Phi, Mistral). The first N layers go entirely on GPU, the rest on CPU. N is calculated from available VRAM (`--gpu-layers -1` auto-detects, `--gpu-layers N` forces N layers).

#### MoE-optimized (MoE models)
For MoE architectures (Qwen3MoE, DeepSeek2) with `--gpu-layers -1` (auto-detect). Inspired by KTransformers (SOSP'25): places **all** attention tensors on GPU across every layer, while expert tensors (`ffn_*_exps`, ~80-90% of layer weight) stay on CPU. Router and shared expert tensors also go on GPU (small). This maximizes GPU utilization because:
- Expert tensors are large but only top-K are activated per token — GPU parallelism is wasted
- Attention is compute-bound and benefits from GPU acceleration on every token
- With 6 GB VRAM, standard first-N-layers fits ~2/48 layers, while MoE-optimized fits 100% of attention

**Auto-detection logic** (`LLMEngine.load()`):
1. Quick-parse GGUF to get `ModelConfig.expertCount()`
2. If `expertCount > 0`: sum non-expert tensor bytes for all layers via `sumNonExpertTensorBytes()`
3. If total ≤ 80% VRAM → enable MoE-optimized, `gpuLayers = blockCount`
4. Otherwise → fallback to first-N-layers

**Per-tensor GPU toggle** (`ModelLoader`): inside the layer loading loop, `TensorFactory.gpuBufferManager` is toggled on/off before each tensor group:
- GPU ON → attention tensors, norms, router, shared experts
- GPU OFF → expert tensors (`ffnGateExps`, `ffnUpExps`, `ffnDownExps`)
- GPU ON → restore for next layer

**Key files**: `GpuConfig.moeOptimized`, `ModelLoader.load(path, preload, gpuLayers, moeOptimizedGpu)`, `LLMEngine.sumNonExpertTensorBytes()`.

**Explicit `--gpu-layers N`** always uses first-N-layers (no MoE optimization) to preserve backward compatibility.

### GPU-virtual thread interaction

When GPU (OpenCL) is active, virtual thread matmul is force-disabled via `FloatTensor.disableVirtualThreadMatmul()` because PoCL's native threads conflict with the JVM's virtual thread carrier threads, causing segfaults. The system falls back to sequential `matmul()` in GPU mode.

### Adding a new quantization type

1. Add the type to `GGMLType` enum with block size and type size
2. Create a `FloatTensor` subclass implementing `getFloat()` and `dotProduct()` with the dequantization math
3. Add the type's case to `TensorFactory.create()`
4. Optionally: add a GPU variant in `java21/` with an OpenCL kernel in `src/main/resources/kernels/`

## Benchmarks

### MoE-optimized GPU placement (2025-02-15)

Hardware: Intel Core Ultra 7 155H + NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM), Java 25, SimdVectorOps.

| Model | Type | GPU Strategy | Layers on GPU | VRAM Used | tok/s | Quality |
|-------|------|--------------|---------------|-----------|-------|---------|
| Qwen3-Coder-30B-A3B Q4_K_M | MoE (128 experts, top-8) | MoE-optimized | 48/48 attn | 540 MB | 1.7 | Perplexity 0.98, Coherence 0.99 |
| DeepSeek-Coder-V2-Lite Q4_K_M | MoE (64 experts, top-6+2shared) | MoE-optimized | 27/27 attn | 517 MB | 2.1 | Perplexity 0.85, Coherence 1.00 |
| Llama-3.2-3B Q3_K_L | Dense | Full offload | 28/28 all | 1731 MB | 5.9 | Perplexity 0.96, Coherence 0.95 |

Test: `--prompt "Write a Java class that calculates factorial" --max-tokens 40 --context-length 256 --gpu --gpu-device 1`.

Key takeaway: MoE-optimized placement puts 100% of attention on GPU using only ~540 MB VRAM for a 17.3 GB model. With standard first-N-layers, only ~2/48 layers would fit in 6 GB VRAM for Qwen3-Coder-30B.
