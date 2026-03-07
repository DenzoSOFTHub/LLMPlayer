# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLMPlayer is a pure Java LLM inference engine that runs GGUF models locally. Zero external dependencies — uses only the JDK. Supports Llama, Qwen2, Qwen3, Qwen3MoE, DeepSeek2, GLM4, Phi-3/4, and Mistral3/Devstral architectures with quantized formats (Q2_K through Q8_0, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL, MXFP4, BF16, F16, F32). Includes a built-in LoRA fine-tuning pipeline.

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
| `TensorFactory.create()` Q4_K/Q8_0 | `SimdQ4_KFloatTensor` / `SimdQ8_0FloatTensor` | `Q4_KFloatTensor` / `Q8_0FloatTensor` |
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

1. **Standard** (`InferenceEngine`): Llama, Qwen2, Qwen3, GLM4, Phi-3/4, Mistral3. Uses `TransformerBlock` → `Attention` (GQA with optional QK-norm/bias) + `SwiGLUFFN`.
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
- **Tool calling**: `tools` array in request → `tool_calls` in response with `finish_reason: "tool_calls"`
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
- `src/main/resources/kernels/cuda/` — 15 CUDA kernel files (`.cu`), parallel to OpenCL kernels. Compiled at runtime via NVRTC by `CudaContext`. Includes `matmul_q4_k_coalesced.cu` (alternative coalesced kernel, opt-in via `-Dcuda.q4k.coalesced=true`).
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

1. **Per-layer mode** (`CudaForwardPass.forwardLayer()`): each layer runs entirely on GPU (RMSNorm → QKV → RoPE → KV cache → attention → Wo → FFN norm → gate/up → SiLU → down). Syncs only at `uploadX`/`downloadX` boundaries. Falls back to CPU for final RMSNorm + output projection if output tensor isn't on GPU.

2. **CUDA graph mode** (`CudaForwardPass.forwardGraph()`): captures ALL kernel launches (all layers + output projection) into a CUDA graph on the first token, then replays with a single `cuGraphLaunch` API call on subsequent tokens. Dynamic values (position, seqLen) are read from GPU-resident `tokenParams` buffer updated via `updateTokenParams()` before each replay. Enabled by default; disable with `-Dcuda.nograph=true`. Requires shared memory ≤ 48 KB (limits maxSeqLen to ~12256 for graph mode).

The `CudaForwardPass` uses **zero-allocation hot paths**: all kernel param buffers (`ParamBuffer`) and matmul launch descriptors (`MatmulLaunch`) are pre-allocated in the constructor. `forwardLayer()` only writes param values in-place and launches kernels.

Only supported for dense pre-norm models with separate Q/K/V projections (Llama, Qwen2, Qwen3, Mistral3). Not supported for: MoE, merged QKV, post-norm, parallel FFN, or hybrid architectures (Qwen3.5 DeltaNet).

### GPU-virtual thread interaction

When GPU (OpenCL or CUDA) is active, virtual thread matmul is force-disabled via `FloatTensor.disableVirtualThreadMatmul()` because native GPU threads can conflict with the JVM's virtual thread carrier threads. The system falls back to sequential `matmul()` in GPU mode.

### GPU backends: CUDA and OpenCL

LLMPlayer supports two GPU backends, both using Panama FFM (zero external dependencies):

- **CUDA** (`CudaBindings` + `CudaContext` + `CudaBufferManager`): calls `libcuda.so` + `libnvrtc.so` directly via Panama FFM. Kernels are `.cu` files compiled at runtime by NVRTC into PTX. Requires NVIDIA driver and NVRTC.
- **OpenCL** (`OpenCLBindings` + `OpenCLContext` + `GpuBufferManager`): calls `libOpenCL.so` via Panama FFM. Kernels are `.cl` files compiled at runtime by the OpenCL driver.

The `--gpu-backend` CLI flag controls backend selection: `auto` (default, prefers CUDA), `cuda`, or `opencl`. `GpuConfig.GpuBackend` enum (AUTO, CUDA, OPENCL) carries this through to `LLMEngine.initGpu()`.

`TensorFactory.gpuBackend` ("cuda" or "opencl") determines which tensor classes are created: `*CudaTensor` or `*GpuTensor`.

### GPU auto-detection priority

When `--gpu-device` is not specified, `LLMEngine.autoConfigureGpu()` tries backends in order: CUDA GPU > OpenCL GPU > OpenCL CPU. CUDA is preferred because it provides native NVIDIA GPU access without the overhead of PoCL. Falls back to CPU OpenCL device only if no real GPU is available. Device type is detected via the `deviceType` field from `enumerateDevices()` (type "2" or containing "CPU"), with fallback to name-based heuristics ("cpu", "pocl").

### Adding a new quantization type

1. Add the type to `GGMLType` enum with block size and type size
2. Create a `FloatTensor` subclass implementing `getFloat()` and `dotProduct()` with the dequantization math
3. Add the type's case to `TensorFactory.create()`
4. Optionally: add GPU variants in `java21/` — both OpenCL (`*GpuTensor` + kernel in `kernels/*.cl`) and CUDA (`*CudaTensor` + kernel in `kernels/cuda/*.cu`)

## Benchmarks

### CUDA kernel optimizations (2026-03)

Hardware: Intel Core Ultra 7 155H + NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM, 192 GB/s peak bandwidth), Java 25, SimdVectorOps.

| Model | Quant | tok/s (before) | tok/s (after) | Improvement |
|-------|-------|-----------------|---------------|-------------|
| Llama-3.2-1B-Instruct | Q4_K_M | 34.0 | 53-56 | +56-65% |
| Llama-3.2-3B-Instruct | Q3_K_L | 15.9 | 8.4 | (regression, investigating) |

Test: `--prompt "Write a Java hello world" --max-tokens 200 --context-length 256 --gpu --gpu-backend cuda`.

Current best for Llama-3.2-1B Q4_K_M: **53-56 tok/s** (CUDA graph mode, 200 tokens). Profiled GPU compute: ~15 ms/tok at 31% of 192 GB/s peak bandwidth. Full performance analysis with per-section breakdown in `PERFORMANCE-ANALYSIS.md`.

**Key optimizations applied:**

1. **Coalesced Q6_K kernel** (`matmul_q6_k.cu`): Restructured from half-striped to position-parallel. All 32 warp threads process the same half-block with consecutive byte addresses → perfectly coalesced memory transactions. Output matmul dropped from 8.3 ms/tok to 2.7 ms/tok (3× improvement). This was the single biggest optimization.

2. **Warp-per-row kernels**: All CUDA matmul kernels use 1 warp (32 threads) per output row with `__shfl_down_sync` reduction. Grid: `ceil(rows / (blockSize/32))` blocks.

3. **Q4_K vectorized loads** (`matmul_q4_k.cu`): `__restrict__` + `__ldg` for read-only cache, `uint32` vectorized weight loads (safe: 144 bytes/block is 4-byte aligned), `float4` input loads. Group-level striping ensures full warp utilization.

4. **Q3_K efficient scale decode** (`matmul_q3_k.cu`): Only the 2 needed scales are decoded per sub-block (not all 16). Uses `__ldg` for texture cache.

5. **Q5_K with __ldg** (`matmul_q5_k.cu`): Added `__ldg` for all reads through texture cache.

6. **Cached kernel functions**: `CudaFloatTensor` caches compiled CUfunction to avoid hashmap lookup per matmul.

7. **Removed redundant finish()**: `cuMemcpyDtoH` is synchronous — explicit `cudaContext.finish()` before `readBuffer` was redundant.

**GPU profiling** (enabled via `-Dcuda.profile=true`): Per-section timing with `cudaContext.finish()` barriers. Steady-state breakdown for 1B Q4_K_M:
- Output Q6_K: 2.7 ms (was 8.3) | FFN Q4_K (GateUp+siluDown): 9.0 ms | Attn+norms: 5.1 ms
- Total profiled GPU: ~15.5 ms/tok | Wall time: ~21 ms/tok | Gap (~5.5 ms) is Panama FFM per-launch overhead

**CUDA alignment constraints**: `__ldg((const unsigned int*)ptr)` requires 4-byte aligned `ptr`. Block sizes NOT divisible by 4: Q8_0 (34B), Q4_0 (18B), Q6_K (210B), Q3_K (110B). These kernels use byte-level `__ldg` only. Q4_K (144B) and Q5_K (176B) are safe for uint32 `__ldg`.

**Approaches tested and rejected:**
- Shared memory cooperative loading: __syncthreads overhead outweighed coalescing benefit for small per-row data
- Concurrent CUDA streams (Gate+Up, K+V): recordEvent+streamWaitEvent sync overhead negated the concurrency gain
- Per-token Arena reuse: No measurable benefit (Arena.ofConfined allocation is already fast)
- Q4_K coalesced kernel (`matmul_q4_k_coalesced.cu`): All 32 threads process the same group with single byte/float reads instead of striping across groups with uint32/float4 vectorized reads. Result: 47.1 tok/s vs 51.0 tok/s original (−8%). Q4_K blocks are 144B (4-byte aligned), so vectorized loads are already optimal — coalescing with smaller reads loses instruction-level parallelism without improving bandwidth. Kernel kept as opt-in (`-Dcuda.q4k.coalesced=true`), default is original kernel.

### MoE-optimized GPU placement (2025-02-15)

Hardware: Intel Core Ultra 7 155H + NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM), Java 25, SimdVectorOps.

| Model | Type | GPU Strategy | Layers on GPU | VRAM Used | tok/s | Quality |
|-------|------|--------------|---------------|-----------|-------|---------|
| Qwen3-Coder-30B-A3B Q4_K_M | MoE (128 experts, top-8) | MoE-optimized | 48/48 attn | 540 MB | 1.7 | Perplexity 0.98, Coherence 0.99 |
| DeepSeek-Coder-V2-Lite Q4_K_M | MoE (64 experts, top-6+2shared) | MoE-optimized | 27/27 attn | 517 MB | 2.1 | Perplexity 0.85, Coherence 1.00 |
| Llama-3.2-3B Q3_K_L | Dense | Full offload | 28/28 all | 1731 MB | 5.9 | Perplexity 0.96, Coherence 0.95 |

Test: `--prompt "Write a Java class that calculates factorial" --max-tokens 40 --context-length 256 --gpu --gpu-device 1`.

Key takeaway: MoE-optimized placement puts 100% of attention on GPU using only ~540 MB VRAM for a 17.3 GB model. With standard first-N-layers, only ~2/48 layers would fit in 6 GB VRAM for Qwen3-Coder-30B.
