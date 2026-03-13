# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLMPlayer is a pure Java LLM inference engine that runs GGUF models locally. Zero external dependencies — uses only the JDK. Supports Llama, Qwen2, Qwen3, Qwen3MoE, DeepSeek2, GLM4, Gemma 2, Gemma 3, Phi-3/4, and Mistral3/Devstral architectures with quantized formats (Q2_K through Q8_0, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL, MXFP4, BF16, F16, F32). Includes a built-in LoRA fine-tuning pipeline.

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

**Build environment note:** If the system JDK is not Java 25, set `JAVA_HOME` before running Maven (e.g. `export JAVA_HOME=/usr/lib/jvm/jdk-25.0.2+10 && export PATH=$JAVA_HOME/bin:$PATH`).

Unit test directory exists but is empty. Integration tests for the OpenAI-compatible API are in `test-openai-api.sh` (requires a running server: `./run.sh --web`, then `bash test-openai-api.sh`). Tests cover 6 architectures (llama, qwen2, qwen3, phi3, deepseek2, mistral3) with streaming, non-streaming, multi-turn, system messages, CORS, error handling, and Bearer token acceptance.

`run.sh` (Linux/macOS) and `run.bat` (Windows) are pre-configured launcher scripts with all required JVM flags for Java 25.

**Note:** The `java21` Maven profile still uses `<release>25</release>` — it requires a Java 25 compiler but excludes the `java25/` source root and `--enable-preview`. Only the `java8` profile actually targets an older compiler release.

## JVM Requirements

Java 21+ builds require these flags at both compile and runtime:
- `--add-modules jdk.incubator.vector` (SIMD Vector API)
- `--enable-native-access=ALL-UNNAMED` (Panama FFI for mmap, OpenCL, and CUDA)

Java 25 builds additionally require:
- `--enable-preview` (StructuredTaskScope, virtual threads)

## Architecture

### Multi-source compilation and reflection loading

The project compiles from three source roots under the default `java25` profile:
- `src/main/java/` — core code, Java 8 compatible
- `src/main/java21/` — Java 21-only code (SIMD tensor ops, SIMD-optimized quantized tensors, OpenCL GPU bindings, CUDA GPU bindings, GPU forward pass)
- `src/main/java25/` — Java 25-only code (StructuredTaskScope batch generation, virtual thread matmul, matmul benchmark)

Classes in `java21/` and `java25/` are **never imported directly** from base code. They are loaded via `Class.forName()` reflection with try/catch fallbacks, allowing graceful degradation on older JVMs. There are six reflection loading sites:

| Loading site | Java 21+ class | Fallback |
|---|---|---|
| `VectorOpsFactory` static init | `SimdVectorOps` | `ScalarOps` |
| `TensorDataFactory.mapFile()` | `MemorySegmentTensorData` | `ByteBufferTensorData` |
| `TensorFactory.tryCreateGpuTensor()` | `Q4_KGpuTensor`/`Q4_KCudaTensor`, etc. | CPU tensor variant |
| `LLMEngine.initGpu()` | `CudaContext` + `CudaBufferManager` (preferred) or `OpenCLContext` + `GpuBufferManager` | CPU-only |
| `TensorFactory.create()` Q4_K/Q8_0/Q6_K/Q5_0/Q5_K/Q3_K | `Simd*FloatTensor` variants | Scalar `*FloatTensor` variants |
| `FloatTensor.tryTiledMatmul()` | `TiledMatmul` | Standard `matmulParallel()` path |
| `FloatTensor.tryVirtualThreadMatmul()` | `VirtualThreadMatmul` | `IntStream.parallel()` ForkJoinPool matmul |
| `LLMEngine.tryStructuredBatch()` | `StructuredBatchGenerator` | `ExecutorService` thread pool |
| `InferenceEngine.tryInitGpuForwardPass()` | `CudaForwardPass` (preferred) or `GpuForwardPass` | CPU forward pass |
| `CLIRunner.listGpuDevices()` | `CudaContext.enumerateDevices()` + `OpenCLContext.enumerateDevices()` | empty list |

**When adding new java21/java25 features:** follow this pattern — put the implementation in the appropriate source root, load it via `Class.forName()` from the base code, and provide a Java 8-compatible fallback.

### Package structure (`it.denzosoft.llmplayer`)

| Package | Purpose |
|---------|---------|
| `api` | Public facade — `LLMEngine` is the main entry point for programmatic use |
| `cli` | CLI argument parsing (`CLIOptions`) and interactive runner |
| `evaluator` | Response quality metrics (perplexity, coherence, length) |
| `gguf` | GGUF file format parser — memory-mapped with parallel preload |
| `gpu` | GPU config (base); CUDA and OpenCL bindings and buffer management (java21) |
| `inference` | Transformer forward pass — `InferenceEngine` (standard), `DeepSeek2InferenceEngine` (MLA + MoE), `Qwen35InferenceEngine` (hybrid DeltaNet + attention) |
| `model` | Model loading, config extraction from GGUF metadata, weight structures |
| `sampler` | Token sampling (temperature, top-k, top-p, repetition penalty) |
| `tensor` | Tensor operations and quantization/dequantization; GPU variants in java21 |
| `tokenizer` | BPE and SentencePiece tokenizers, chat template formatting |
| `tuning` | LoRA fine-tuning pipeline — data chunking, Q&A dataset generation, training loop, LoRA merge, GGUF export |
| `ui` | Swing desktop GUI |
| `web` | Embedded HTTP server with HTML web UI, OpenAI-compatible API (`OpenAIHandler`), Anthropic Messages API (`AnthropicHandler`), management API (`ApiHandler`), chat persistence with branching (`ChatHandler`) |

### Key data flow

1. `GGUFParser` memory-maps the model file and extracts metadata + tensor info
2. `ModelLoader` builds `ModelConfig` from metadata, creates weight tensors via `TensorFactory`, and instantiates the tokenizer via `TokenizerFactory`
3. `LLMEngine.load()` wraps everything into the public API, choosing `InferenceEngine` or `DeepSeek2InferenceEngine` based on architecture
4. `generate()` tokenizes the prompt (with chat template if enabled), runs prefill, then auto-regressive decoding with the configured sampler
5. Each `generate()` call creates its own `InferenceState` — model weights are immutable mmap'd memory, making `LLMEngine` thread-safe

### Inference engine dispatch (four paths)

1. **Standard** (`InferenceEngine`): Llama, Qwen2, Qwen3, GLM4, Gemma 2, Gemma 3, Phi-3/4, Mistral3. Uses `TransformerBlock` → `Attention` (GQA with optional QK-norm/bias, sliding window, dual RoPE) + `SwiGLUFFN` (with GeGLU for Gemma). Gemma 2/3 use pre+post attention/FFN norms and embedding scaling.
2. **DeepSeek2** (`DeepSeek2InferenceEngine`): DeepSeek2 and GLM-4.7-Flash (also uses `deepseek2` GGUF arch). Uses MLA (Multi-Head Latent Attention) + MoE FFN with shared expert. Leading blocks use dense SwiGLU FFN.
3. **Qwen3 MoE** (`Qwen3MoEInferenceEngine`): Qwen3-Coder-30B-A3B and similar. Standard GQA attention with QK-norm + MoE FFN with shared expert. Leading blocks use dense SwiGLU FFN.
4. **Qwen3.5** (`Qwen35InferenceEngine`): Hybrid DeltaNet + full attention architecture. Alternates Gated DeltaNet (linear attention/SSM) and standard GQA layers in a 3:1 ratio (`full_attention_interval=4`). DeltaNet layers use recurrent state `S` with update rule `S_new = alpha*S + beta*outer(k, v - alpha*S^T@k)`, output `o = S^T_new @ q`. Full attention layers use a packed Q+gate projection where `wq` outputs interleaved `[Q_h0, gate_h0, Q_h1, gate_h1, ...]` — these must be deinterleaved into separate Q and gate arrays before use. Gate is applied as `sigmoid(gate) * attn_output`. Both layer types include short conv1d (width 4) on Q/K and use QK-norm. State is maintained per-layer in `Qwen35State`.

### Tensor system

`FloatTensor` is the core abstraction. Each quantization format has a dedicated subclass (e.g., `Q4_KFloatTensor`) that implements dequantization inline. `TensorFactory.create()` selects the implementation by `GGMLType` — trying a GPU variant first (CUDA or OpenCL, based on `TensorFactory.gpuBackend`), then falling back to CPU. GPU tensor classes live in `java21/` and delegate to CUDA kernels (`src/main/resources/kernels/cuda/`) or OpenCL kernels (`src/main/resources/kernels/`).

### Tokenizer dispatch

`TokenizerFactory` reads `tokenizer.ggml.model` from GGUF metadata: `"gpt2"` or `"bpe"` → `BPETokenizer` (uses merge table, Llama 3 style pre-tokenization regex), anything else → `SentencePieceTokenizer` (score-based). Chat formatting is architecture-specific in `ChatTemplate`, with distinct templates for Llama (`<|start_header_id|>`), Qwen2/3/Qwen3MoE (`<|im_start|>`), GLM4 (`[gMASK]<sop>`), DeepSeek2 (`User: ... Assistant:`), Phi-3/4 (`<|user|>`), and Mistral3 (`[INST]`).

### Launch modes

- **No args** → Swing desktop GUI (`LLMPlayerUI`)
- **`--web`** → Embedded `com.sun.net.httpserver.HttpServer` with model config UI at `/`, chat UI at `/chat`, OpenAI-compatible API at `/v1/*`, management API at `/api/*`, chat persistence API at `/api/chats/*`
- **`--model <path>`** → CLI mode (single prompt with `--prompt` or `--interactive` chat)
- **`--fine-tune`** → LoRA fine-tuning pipeline (requires `--target-model` and one of `--source`, `--documents`, `--data`, or `--train-dataset`)
- **`--gpu-list`** → enumerate CUDA and OpenCL devices and exit
- **`--download <repo>`** → download GGUF model from HuggingFace (`owner/repo` or `owner/repo/file.gguf`)

### REST API (web mode)

When running with `--web`, the server exposes four API groups. Full documentation in `REST-API.md`.

#### OpenAI-compatible API (`/v1/*`)

Follows the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat) spec. Works with standard OpenAI clients (Open WebUI, LangChain, LiteLLM, Cursor, Continue.dev, etc.). The `Authorization: Bearer <token>` header is accepted and ignored.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/chat/completions` | POST | Chat completion (streaming + non-streaming), tool calling, JSON mode |
| `/v1/embeddings` | POST | Text embeddings (L2-normalized vectors) |
| `/v1/models` | GET | List available/loaded models |

Implemented in `OpenAIHandler.java`. Uses `ChatTemplate.formatConversation()` for multi-turn message formatting. Supports `messages` array with `system`/`user`/`assistant` roles, `stream`, `temperature`, `max_tokens`, `top_p`, `top_k`, `stop`, `frequency_penalty`, `repetition_penalty`. The `model` field is accepted but ignored (uses the currently loaded model). Streaming sends SSE chunks in OpenAI format (`chat.completion.chunk`) ending with `data: [DONE]`. Non-streaming returns a full `chat.completion` JSON with `choices` and `usage`.

Additional OpenAI-compatible features:
- **Tool calling**: `tools` array in request → `tool_calls` in response with `finish_reason: "tool_calls"`. Architecture-aware: SmolLM3 uses Hermes-style XML format (`<tool_call>` tags, `<tool_response>` for results); other models use generic JSON prompt injection. Multi-tool-call parsing supported. Tool format logic lives in `ChatTemplate.formatToolsSystemPrompt()`, `formatToolResult()`, `formatAssistantToolCalls()`. Response parsing in `OpenAIHandler.tryParseToolCalls()`.
- **JSON mode**: `response_format: {type: "json_object"}` injects a system prompt instructing the model to produce JSON
- **Embeddings**: `/v1/embeddings` returns L2-normalized vectors with dimension = embeddingLength

The model config UI (`web-ui.html`, served at `/`) uses `/v1/chat/completions` for chat and `/api/*` for model management. The chat UI (`chat-ui.html`, served at `/chat`) uses `/v1/chat/completions` for streaming generation and `/api/chats/*` for conversation persistence.

#### Anthropic Messages API (`/v1/messages`)

Implements the Anthropic Messages API for compatibility with Claude Code and other Anthropic API clients. Implemented in `AnthropicHandler.java`.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/messages` | POST | Chat completion (streaming + non-streaming), Anthropic message format |
| `/v1/messages/count_tokens` | POST | Token counting for a message payload |

The `x-api-key` header is accepted and ignored (like `Authorization: Bearer` on the OpenAI endpoint).

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

#### Chat Persistence API (`/api/chats/*`)

Server-side conversation persistence with tree-based branching. Conversations are stored as JSON files in the `chats/` directory. Implemented in `ChatHandler.java`.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/chats` | GET | List conversations (id, title, created, updated, messageCount) |
| `/api/chats` | POST | Create new conversation |
| `/api/chats/{id}` | GET | Get full conversation with message tree |
| `/api/chats/{id}` | DELETE | Delete conversation |
| `/api/chats/{id}/title` | PUT | Rename conversation |
| `/api/chats/{id}/messages` | POST | Add message (user or assistant) |
| `/api/chats/{id}/messages/{msgId}` | PUT | Edit message (creates sibling branch) |
| `/api/chats/{id}/active-leaf` | PUT | Update active branch leaf |
| `/api/chats/{id}/settings` | PUT | Update per-conversation settings |
| `/api/chats/export/{id}` | GET | Export conversation as JSON |

Messages use a flat map (`id → message`) with `parentId`/`children` references forming a tree. Editing a message creates a new sibling with the same parent, enabling conversation branching. The chat UI (`/chat`) navigates branches with arrow controls.

### Fine-tuning pipeline (`tuning` package)

Pure Java LoRA fine-tuning with checkpoint/resume. Full documentation in `FINE-TUNING.md`. The pipeline has 6 stages:

1. **Analyze** (`TargetAnalyzer`): quick-parse target GGUF to get architecture, dimensions, layer count
2. **Chunk** (`DataChunker` / `CodeChunker` / `TextChunker`): split input data into context-sized chunks
3. **Generate** (`QAGenerator`): use a generator LLM to produce Q&A training pairs from chunks
4. **Train** (`TrainingLoop` + `LoRAAdapter`): LoRA fine-tuning via `LLMEngine.forwardSingleToken()` teacher forcing
5. **Merge** (`LoRAMerger`): merge LoRA adapters back into base model weights
6. **Export** (`GGUFWriter`): write merged weights as a new GGUF file

Three data scenarios auto-detected from CLI flags: `--source` (code), `--documents` (text), `--data`+`--schema` (structured/SQL). Dataset generation can be decoupled from training via `--dataset-only` and `--train-dataset`.

`LLMEngine.forwardSingleToken(token, position)` is the training-specific API: runs a single forward pass returning logits, with a lazily-created persistent inference state. Works with all three inference engine types (standard, DeepSeek2, Qwen3MoE).

### Adding a new model architecture

1. Add enum value to `ModelArchitecture` with its GGUF `general.architecture` string
2. Add any architecture-specific tensor name patterns to `ArchitectureRegistry` (standard names like `blk.{n}.attn_q.weight` are shared across most architectures)
3. Update `ModelConfig.fromMetadata()` if the architecture uses non-standard metadata keys for hyperparameters
4. Add a chat template branch in `ChatTemplate.formatUserMessage()` and `ChatTemplate.formatConversation()`
5. If the architecture's forward pass differs from standard transformer attention+FFN, create a dedicated inference engine class (see `DeepSeek2InferenceEngine` as reference)

### Resources

- `src/main/resources/kernels/` — 12 OpenCL kernel files (matmul variants for each quantization type, plus `rmsnorm.cl`, `softmax.cl`, `silu.cl`, `saxpy.cl`, `accumulate.cl`). Loaded and compiled on-demand by `OpenCLContext`.
- `src/main/resources/kernels/cuda/` — 31 CUDA kernel files (`.cu`). Matmul kernels for Q3_K, Q4_0, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0, F32, BF16, F16, IQ2_S, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS plus auxiliary kernels (RMSNorm, RoPE, attention, softmax, SiLU, argmax, split_qkv, split_gate_up, fused_gate_up, rmsnorm_per_head). Compiled at runtime via NVRTC by `CudaContext`. Includes `matmul_q4_k_coalesced.cu` (alternative coalesced kernel, opt-in via `-Dcuda.q4k.coalesced=true`).
- `src/main/resources/web-ui.html` — Model config web UI served at `/` by `WebServer` in `--web` mode.
- `src/main/resources/chat-ui.html` — Chat UI with conversation persistence and branching, served at `/chat`.

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

### CUDA GPU-resident forward pass

Two modes for keeping activations on GPU between transformer layers, reducing CPU↔GPU sync points:

1. **Per-layer mode** (`CudaForwardPass.forwardLayer()`): each layer runs entirely on GPU (RMSNorm → QKV → RoPE → KV cache → attention → Wo → FFN norm → gate/up → SiLU → down). Syncs only at `uploadX`/`downloadX` boundaries.

2. **CUDA graph mode** (`CudaForwardPass.forwardGraph()`): captures ALL kernel launches into a CUDA graph on the first token, then replays with a single `cuGraphLaunch` on subsequent tokens. Dynamic values (position, seqLen) are read from GPU-resident `tokenParams` buffer. Enabled by default; disable with `-Dcuda.nograph=true`.

Key design: **zero-allocation hot paths** — all kernel param buffers (`ParamBuffer`) and matmul launch descriptors (`MatmulLaunch`) are pre-allocated in the constructor. `forwardLayer()` only writes param values in-place and launches kernels.

**Supported architectures for CUDA forward pass**: Llama, Qwen2, Qwen3, Mistral3, Gemma 2/3 (post-norm), Phi-3/4 (packed FFN via `split_gate_up.cu`). Per-head QK-norm (Qwen3) via `rmsnorm_per_head.cu`. **Not supported**: MoE or hybrid architectures (Qwen3.5 DeltaNet).

### GPU-virtual thread interaction

When GPU (OpenCL or CUDA) is active, virtual thread matmul is force-disabled via `FloatTensor.disableVirtualThreadMatmul()` because native GPU threads can conflict with the JVM's virtual thread carrier threads. The system falls back to sequential `matmul()` in GPU mode.

### GPU backends: CUDA and OpenCL

LLMPlayer supports two GPU backends, both using Panama FFM (zero external dependencies):

- **CUDA** (`CudaBindings` + `CudaContext` + `CudaBufferManager`): calls `libcuda.so` + `libnvrtc.so` directly via Panama FFM. Kernels are `.cu` files compiled at runtime by NVRTC into PTX. Requires NVIDIA driver and NVRTC.
- **OpenCL** (`OpenCLBindings` + `OpenCLContext` + `GpuBufferManager`): calls `libOpenCL.so` via Panama FFM. Kernels are `.cl` files compiled at runtime by the OpenCL driver.

The `--gpu-backend` CLI flag controls backend selection: `auto` (default, prefers CUDA), `cuda`, or `opencl`. `GpuConfig.GpuBackend` enum (AUTO, CUDA, OPENCL) carries this through to `LLMEngine.initGpu()`.

`TensorFactory.gpuBackend` ("cuda" or "opencl") determines which tensor classes are created: `*CudaTensor` or `*GpuTensor`.

### GPU auto-detection priority

When `--gpu-device` is not specified, `LLMEngine.autoConfigureGpu()` tries backends in order: CUDA GPU > OpenCL GPU. CUDA is preferred because it provides native NVIDIA GPU access without the overhead of PoCL. Returns `null` (CPU SIMD path) if only OpenCL CPU devices (PoCL) are available — OpenCL on CPU adds marshaling overhead without compute benefit. Device type is detected via the `deviceType` field from `enumerateDevices()` (type "2" or containing "CPU"), with fallback to name-based heuristics ("cpu", "pocl").

### Adding a new quantization type

1. Add the type to `GGMLType` enum with block size and type size
2. Create a `FloatTensor` subclass implementing `getFloat()` and `dotProduct()` with the dequantization math
3. Add the type's case to `TensorFactory.create()`
4. Optionally: add GPU variants in `java21/` — both OpenCL (`*GpuTensor` + kernel in `kernels/*.cl`) and CUDA (`*CudaTensor` + kernel in `kernels/cuda/*.cu`)

## Benchmarks

See `BENCHMARKS.md` for full results (27 models tested) and `PERFORMANCE-ANALYSIS.md` for detailed per-kernel profiling.

Current best: Llama-3.2-1B Q4_K_M at **53-56 tok/s** (CUDA graph mode, RTX 4050 Laptop GPU).

### CUDA kernel design patterns

All CUDA matmul kernels use 1 warp (32 threads) per output row with `__shfl_down_sync` reduction. Grid: `ceil(rows / (blockSize/32))` blocks.

**CUDA alignment constraints**: `__ldg((const unsigned int*)ptr)` requires 4-byte aligned `ptr`. Block sizes NOT divisible by 4: Q8_0 (34B), Q4_0 (18B), Q6_K (210B), Q3_K (110B) — these kernels use byte-level `__ldg` only. Q4_K (144B) and Q5_K (176B) are safe for uint32 `__ldg`.

**GPU profiling**: enable via `-Dcuda.profile=true` for per-section timing with `cudaContext.finish()` barriers. **CPU profiling**: `-Dcpu.profile=true`.
