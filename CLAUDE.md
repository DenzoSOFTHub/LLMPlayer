# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLMPlayer is a pure Java LLM inference engine (v1.11.0) that runs GGUF models locally. Zero external dependencies — uses only the JDK. Supports 21 architectures (Llama, Qwen2, Qwen3, Qwen3MoE, Qwen3.5, SmolLM3, DeepSeek2, GLM4/GLM-4.7-Flash, Gemma 2/3/3n/4, Phi-3/4, Mistral3/Devstral, Command-R/Cohere, OLMo2 (incl. Olmo 3 ChatML variant), Falcon3, GPT-OSS/Sonar, Granite 3.3, Granite Hybrid, Nemotron-H hybrid Mamba-2) and 18 quantized formats (F32, F16, BF16, Q2_K, Q3_K, Q4_0, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0, IQ2_S, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, MXFP4). 16 of these have dedicated CUDA tensor classes for full GPU acceleration (all except Q2_K and MXFP4). Includes CUDA GPU acceleration with graph mode, thinking/reasoning mode, architecture-aware tool calling, HuggingFace model download, JMX metrics, automated kernel autosearch (`autosearch.sh`), and a built-in LoRA fine-tuning pipeline.

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

Unit test directory exists but is empty (no JUnit dependency to preserve zero-deps). Test scripts:
- `test-architectures.sh` — smoke test that loads each supported architecture (Llama, Qwen2/3/3.5, Gemma 2/3/3n/4, Phi-3/4, Mistral3, OLMo2, Falcon3, Granite 3.3, Granite Hybrid, Nemotron-H, SmolLM3, DeepSeek2, Qwen3MoE) and verifies it can generate at least one token. Auto-detects Java 21+. Run with `./test-architectures.sh` for CPU-only or `./test-architectures.sh --gpu` for GPU mode. Set `INCLUDE_LARGE=1` to include 30B+ MoE models.
- `test-openai-api.sh` — integration tests for the OpenAI-compatible API (requires a running server: `./run.sh --web`, then `bash test-openai-api.sh`). Tests cover 6 architectures with streaming, non-streaming, multi-turn, system messages, CORS, error handling, and Bearer token acceptance.
- `autosearch.sh <model.gguf> [runs] [min_ppl]` — Karpathy-style automated kernel-config search. Greedy coordinate ascent over all `-D` flags (CUDA kernel variants, KV cache quant, matmul mode, etc.). Two KPIs: tok/s and PPL. Writes a Pareto-optimal config to `autosearch-results.txt`.
- `test-ppl-sweep.sh` / `test-cpu-sweep.sh` — quality regression sweeps. Loads a list of GGUF models, runs them through a canonical prompt, prints aggregate PPL per model. Use to detect quality regressions across a release.

`run.sh` (Linux/macOS) and `run.bat` (Windows) are launcher scripts with all required JVM flags for Java 25, but they are **not checked into the repo** (excluded via `.gitignore`). Create them locally — see README.md for the script contents.

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

**Token embedding optimization**: across all architectures, the token embedding tensor is loaded on CPU (not GPU). It is only used for a single-element lookup per token (~16 KB), making GPU residency wasteful (~500+ MB of VRAM). When output weights are tied to embedding weights, the output tensor is reloaded separately on GPU for the matmul projection.

Classes in `java21/` and `java25/` are **never imported directly** from base code. They are loaded via `Class.forName()` reflection with try/catch fallbacks, allowing graceful degradation on older JVMs. Reflection loading sites:

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
| `inference` | Transformer forward pass — `InferenceEngine` (standard), `DeepSeek2InferenceEngine` (MLA + MoE), `Qwen3MoEInferenceEngine` (GQA + MoE), `Qwen35InferenceEngine` (hybrid DeltaNet + attention), `NemotronHInferenceEngine` (hybrid Mamba-2 + attention + FFN) |
| `model` | Model loading, config extraction from GGUF metadata, weight structures |
| `sampler` | Token sampling (temperature, top-k, top-p, repetition penalty) |
| `tensor` | Tensor operations and quantization/dequantization; GPU variants in java21 |
| `tokenizer` | BPE and SentencePiece tokenizers, chat template formatting |
| `tuning` | LoRA fine-tuning pipeline — data chunking, Q&A dataset generation, training loop, LoRA merge, GGUF export |
| `ui` | Swing desktop GUI |
| `web` | Embedded HTTP server with HTML web UI, OpenAI-compatible API (`OpenAIHandler`), Anthropic Messages API (`AnthropicHandler`), management API (`ApiHandler`), chat persistence with branching (`ChatHandler`) |
| `spec` | Speculative decoding (`SpeculativeDecoder`) — standalone class that drives two `LLMEngine` instances (target + draft) via `forwardSingleToken`. Sequential verification only (~1.1× max with K=4); real 2-3× speedup awaits a `forwardBatch` API. Enabled via `--draft-model <gguf>`. |

### Key data flow

1. `GGUFParser` memory-maps the model file and extracts metadata + tensor info
2. `ModelLoader` builds `ModelConfig` from metadata, creates weight tensors via `TensorFactory`, and instantiates the tokenizer via `TokenizerFactory`
3. `LLMEngine.load()` wraps everything into the public API, choosing `InferenceEngine` or `DeepSeek2InferenceEngine` based on architecture
4. `generate()` tokenizes the prompt (with chat template if enabled), runs prefill, then auto-regressive decoding with the configured sampler
5. Each `generate()` call creates its own `InferenceState` — model weights are immutable mmap'd memory, making `LLMEngine` thread-safe

### Inference engine dispatch (six paths)

1. **Standard** (`InferenceEngine`): Llama, Qwen2, Qwen3, SmolLM3, GLM4, Gemma 2, Gemma 3, Phi-3/4, Mistral3, Command-R, OLMo2, Falcon3, GPT-OSS, Granite 3.3. Uses `TransformerBlock` → `Attention` (GQA with optional QK-norm/bias, sliding window, dual RoPE) + `SwiGLUFFN` (with GeGLU for Gemma). Gemma 2/3 use pre+post attention/FFN norms and embedding scaling. Granite 3.3 uses four custom scaling factors (embedding, attention, residual, logit) read from GGUF `granite.*` metadata keys.
2. **DeepSeek2** (`DeepSeek2InferenceEngine`): DeepSeek2 and GLM-4.7-Flash (also uses `deepseek2` GGUF arch). Uses MLA (Multi-Head Latent Attention) + MoE FFN with shared expert. Leading blocks use dense SwiGLU FFN.
3. **Qwen3 MoE** (`Qwen3MoEInferenceEngine`): Qwen3-Coder-30B-A3B and similar. Standard GQA attention with QK-norm + MoE FFN with shared expert. Leading blocks use dense SwiGLU FFN.
4. **Qwen3.5** (`Qwen35InferenceEngine`): Hybrid DeltaNet + full attention architecture. Alternates Gated DeltaNet (linear attention/SSM) and standard GQA layers in a 3:1 ratio (`full_attention_interval=4`). DeltaNet layers use recurrent state `S` with update rule `S_new = alpha*S + beta*outer(k, v - alpha*S^T@k)`, output `o = S^T_new @ q`. Full attention layers use a packed Q+gate projection where `wq` outputs interleaved `[Q_h0, gate_h0, Q_h1, gate_h1, ...]` — these must be deinterleaved into separate Q and gate arrays before use. Gate is applied as `sigmoid(gate) * attn_output`. Both layer types include short conv1d (width 4) on Q/K and use QK-norm. State is maintained per-layer in `Qwen35State`.
5. **Nemotron-H / Granite Hybrid** (`NemotronHInferenceEngine`): Hybrid Mamba-2 SSM + GQA Attention + squared-ReLU FFN. Three distinct layer types (not combined like standard transformers). Per-layer arrays in GGUF metadata (`head_count_kv[]`, `feed_forward_length[]`) determine layer type: kvHeads>0 → Attention, ffnLength>0 → FFN, both 0 → Mamba-2. Mamba-2 uses SSD (Structured State Space Duality) with state `[nheads][headDim][stateSize]`, causal conv1d with bias+SiLU, and grouped RMSNorm with gate (norm_before_gate=False). GPU-resident forward pass via `NemotronHCudaForwardPass` with dedicated kernels (`mamba2_scan.cu`, `mamba2_dt_softplus.cu`, `mamba2_gate_norm.cu`, `sqrelu.cu`). **Granite Hybrid fully GPU-accelerated as of v1.11.0-dev** (2026-04-15): all four scale factors (embedding/logit/residual/attention) wired on GPU via `scale_inplace` + `accumulate` + saxpy, and the integrated SwiGLU FFN inside Mamba/Attention layers (`lw.ffnUp() != null`) runs on GPU via `runIntegratedFFN()` with fused RMSNorm + Q8_1 quant + gate/up/down dp4a + `silu_mul`. Validated bit-equivalent to CPU at ±2 ULP when dp4a is disabled; with dp4a on, the ~1-15% per-layer divergence is the expected Q8_1 quantization noise (same as all other dp4a-accelerated models). CUDA graph capture works.
6. **Gemma 4 / Gemma 3n** (`Gemma4InferenceEngine`): For PLE (Per-Layer Embeddings) models (E2B/E4B). Two sub-paths in one engine, dispatched by whether `Gemma3nWeights` has AltUp tensors loaded:
   - **Gemma 3n** (arch=GEMMA3N): full **AltUp** (4 parallel activation streams with learned router/predict/correct coefficients) + **Laurel** (low-rank residual branch) + **Gaussian top-k activation sparsity** (first 10 FFN layers) + PLE in `forwardLayerGemma3nInner` + `forwardLayerAltup`. K-norm uses `(1+w)` adjustment.
   - **Gemma 4** (arch=GEMMA4): simpler PLE-only path in `forwardLayer` (no AltUp/Laurel/sparsity). Per llama.cpp `gemma4-iswa.cpp`: **V-norm** (`ggml_rms_norm` on V, no learnable scale), **`layer_output_scale.weight`** per-layer scalar applied as final `cur *= scale` multiplication. K-norm stored as final values (no `(1+w)`).
   - Common to both: per-layer token embedding/projection tensors, dual headSize (SWA=256, full=512 per-layer), shared KV cache (layers ≥ blockCount-sharedKvLayers reuse earlier layers' KV), dual RoPE (SWA theta=10K, full theta=1M with proportional frequency factors), attention scale=1.0, logit soft-capping. Uses `Gemma4State` with variable-size buffers per layer.

### Tensor system

`FloatTensor` is the core abstraction. Each quantization format has a dedicated subclass (e.g., `Q4_KFloatTensor`) that implements dequantization inline. `TensorFactory.create()` selects the implementation by `GGMLType` — trying a GPU variant first (CUDA or OpenCL, based on `TensorFactory.gpuBackend`), then falling back to CPU. GPU tensor classes live in `java21/` and delegate to CUDA kernels (`src/main/resources/kernels/cuda/`) or OpenCL kernels (`src/main/resources/kernels/`).

### Tokenizer dispatch

`TokenizerFactory` reads `tokenizer.ggml.model` from GGUF metadata: `"gpt2"` or `"bpe"` → `BPETokenizer` (uses merge table, Llama 3 style pre-tokenization regex). The string `"gemma4"` also dispatches to `BPETokenizer` but with `useGpt2ByteMapping=false` — Gemma 4 vocab uses SentencePiece-style `▁` (U+2581) for spaces and `<0xHH>` byte fallback tokens, NOT GPT-2 byte mapping. Anything else → `SentencePieceTokenizer` (score-based). Chat formatting is architecture-specific in `ChatTemplate`, with distinct templates for Llama (`<|start_header_id|>`), Qwen2/3/Qwen3MoE (`<|im_start|>`), GLM4 (`[gMASK]<sop>`), DeepSeek2 (`User: ... Assistant:`), Phi-3/4 (`<|user|>`), Mistral3 (`[INST]`), Gemma 4 (`<|turn>...<turn|>`), and OLMo2 (`<|user|>...<|assistant|>`). **Olmo 3 detection**: when the chat_template metadata contains `<|im_start|>`, `ChatTemplate.isOlmo3ChatML` is set to true and the OLMo2 format method switches to ChatML output (`<|im_start|>user\n...<|im_end|>`).

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
| `/api/metrics` | GET | Runtime metrics: model info, generation stats, memory, GPU VRAM |

### JMX Monitoring

Runtime metrics are exposed via JMX MXBean at `it.denzosoft.llmplayer:type=LLMPlayer`. Connect with JConsole, VisualVM, or any JMX client.

**Interface:** `LLMPlayerMXBean.java` — **Implementation:** `LLMPlayerMetrics.java` (singleton, thread-safe atomics).

Attributes: `ModelName`, `Architecture`, `TotalGenerations`, `TotalTokensGenerated`, `LastTokensPerSecond`, `AverageTokensPerSecond`, `RecentTokensPerSecond` (60s rolling window), `RecentSampleCount`, `HeapUsedMB`, `HeapMaxMB`, `GpuVramTotalMB`, `GpuVramFreeMB`, `GpuLayersUsed`, `KvCacheEstimateMB`.

Also available via REST: `GET /api/metrics` returns the same data as JSON (model, generation, memory, GPU sections). The rolling window keeps the last 256 generations within the 60-second window for live tok/s monitoring.

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

### Thinking/reasoning mode

Enabled via `--thinking` CLI flag or `"thinking": true` in OpenAI API requests. Architecture-specific:
- **SmolLM3**: injects `/think` into the system prompt
- **Qwen3/Qwen3.5**: by default, thinking is suppressed (system prompt instructs no `<think>` blocks); `--thinking` removes the suppressor

### Tool calling

Architecture-aware tool calling via the OpenAI API (`tools` array in request). Two formats:
- **SmolLM3**: Hermes-style XML (`<tool_call>` tags, `<tool_response>` for results) — implemented in `ChatTemplate.formatToolsSystemPrompt()`, `formatToolResult()`, `formatAssistantToolCalls()`
- **Other models**: generic JSON prompt injection

Response parsing in `OpenAIHandler.tryParseToolCalls()`. Multi-tool-call parsing supported. Full documentation in `TOOL-CALLING.md`.

### Adding a new model architecture

1. Add enum value to `ModelArchitecture` with its GGUF `general.architecture` string
2. Add any architecture-specific tensor name patterns to `ArchitectureRegistry` (standard names like `blk.{n}.attn_q.weight` are shared across most architectures)
3. Update `ModelConfig.fromMetadata()` if the architecture uses non-standard metadata keys for hyperparameters
4. Add a chat template branch in `ChatTemplate.formatUserMessage()` and `ChatTemplate.formatConversation()`
5. If the architecture's forward pass differs from standard transformer attention+FFN, create a dedicated inference engine class (see `DeepSeek2InferenceEngine` for MLA+MoE, `Qwen35InferenceEngine` for DeltaNet+attention, `NemotronHInferenceEngine` for Mamba-2+attention+FFN)
6. If the architecture supports tool calling, add format methods in `ChatTemplate` (see SmolLM3's Hermes-style implementation as reference)
7. If the architecture supports thinking/reasoning, add handling in `ChatTemplate` and `CLIOptions`

### Resources

- `src/main/resources/kernels/` — 14 OpenCL kernel files: 7 matmul variants (`matmul_f32.cl`, `matmul_q3_k.cl`, `matmul_q4_0.cl`, `matmul_q4_k.cl`, `matmul_q5_k.cl`, `matmul_q6_k.cl`, `matmul_q8_0.cl`) plus `rmsnorm.cl`, `softmax.cl`, `silu.cl`, `saxpy.cl`, `accumulate.cl`, `elementwise_mul.cl`, `fill_zero.cl`. Loaded and compiled on-demand by `OpenCLContext`.
- `src/main/resources/kernels/cuda/` — CUDA kernel files (`.cu`). Matmul kernels for Q3_K, Q4_0, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0, F32, BF16, F16, IQ2_S, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, MXFP4 plus dp4a variants (`matmul_q4_k_dp4a.cu`, `matmul_q5_k_dp4a.cu`, `matmul_q6_k_dp4a.cu`), shared-memory variants (`matmul_q5_k_smem.cu`, `matmul_q6_k_smem.cu`, `matmul_q6_k_tiled.cu`), 2-warp variant (`matmul_q4_k_2warp.cu`), coalesced variant (`matmul_q4_k_coalesced.cu`), DeltaNet kernels (`deltanet_fused.cu`, `deltanet_fused_v2.cu`, `deltanet_recurrence.cu`), Mamba-2 kernels (`mamba2_scan.cu`, `mamba2_dt_softplus.cu`, `mamba2_gate_norm.cu`), cuBLAS support kernels (`dequant_q4_k_f16.cu`, `dequant_q4_k_f32.cu`, `convert_f32_to_f16.cu`, `quantize_q8.cu`), and auxiliary kernels (RMSNorm, RoPE, attention, softmax, SiLU, argmax, split_qkv, split_gate_up, fused_gate_up, rmsnorm_per_head, conv1d_short, conv1d_silu, alpha_beta_gates, deinterleave_q_gate, sigmoid_elementwise_mul, sqrelu, scale_inplace, silu_mul). Compiled at runtime via NVRTC by `CudaContext`. Note: `matmul_mxfp4.cu` exists but no `MXFP4CudaTensor` wrapper is wired up — MXFP4 is currently CPU-only at the tensor layer.
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

**Supported architectures for CUDA forward pass**: Llama, Qwen2, Qwen3, Falcon3, OLMo2 (incl. Olmo 3), Mistral3, Gemma 2/3 (post-norm), Phi-3/4 (packed FFN via `split_gate_up.cu`), Granite 3.3 (with scaling factors), and Nemotron-H / Granite Hybrid (via `NemotronHCudaForwardPass` — Granite Hybrid uses the full integrated-FFN path with all scale factors on GPU). Per-head QK-norm (Qwen3) via `rmsnorm_per_head.cu`. **Not supported on GPU**: MoE architectures, Gemma 4/3n (PLE — runs on CPU via `Gemma4InferenceEngine`). Standard architectures get full GPU acceleration.

**dp4a path** (default-on, see `cuda.dp4a` flag): inserts `quantize_q8` calls after each FP32 buffer is produced (post attn norm, post attention, post FFN norm, post silu_mul, post final norm) and routes Q4_K matmuls (QKV / Wo / Gate / Up / Down / Output) through `matmul_q4_k_dp4a.cu` reading int8 Q8_1 input. Q5_K uses `matmul_q5_k_dp4a.cu` similarly. Q6_K dp4a is opt-in only (broken, see `cuda.dp4a.q6`). Falls back to FP32 kernel for non-eligible types or when `cuda.dp4a=false`. Matches the long-standing pattern in `Qwen35CudaForwardPass` — was previously Qwen35-only despite the misleading "default true" doc.

#### Qwen3.5 CUDA forward pass (`Qwen35CudaForwardPass`)

Dedicated GPU-resident forward pass for the hybrid DeltaNet+attention architecture. Handles both DeltaNet layers (3/4) and full GQA attention layers (1/4) with CUDA graph support.

**DeltaNet-specific CUDA kernels:**
- `deltanet_fused.cu` — mega-kernel: recurrence + per-head RMSNorm + SiLU(gate) + gate multiply, with transposed S matrix `[dV][dQK]` for coalesced access and parallel L2 norm via warp-shuffle reduction
- `conv1d_silu.cu` — fused causal conv1d + SiLU activation
- `alpha_beta_gates.cu` — compute alpha (exp decay) and beta (sigmoid) gates from projections
- `deinterleave_q_gate.cu` — split packed Q+gate projection for attention layers
- `sigmoid_elementwise_mul.cu` — attention output gating: `xb2 *= sigmoid(gate)`

**Optimizations:**
- Fused FFN gate+up Q4_K kernel (reuses `matmul_q4_k_fused_gate_up.cu`)
- GPU-side argmax (`forwardGraphArgmax()`, `forwardFinalArgmax()`) — downloads 4 bytes instead of full logits
- Embedding loaded on CPU (frees ~500 MB VRAM, only used for 1-element lookup per token)
- VRAM budget corrected: subtracts non-layer tensor sizes before per-layer estimation, uses 90% of VRAM

#### cuBLAS acceleration (opt-in)

Optional path using NVIDIA cuBLAS library for matmul operations. Enabled via `-Dcuda.cublas=true`. Pre-dequantizes Q4_K weights to FP16 (default) or FP32 at load time, then uses `cublasSgemv`/`cublasGemmEx` for matrix-vector multiply.

**Bindings**: `CublasBindings.java` (Panama FFM for `libcublas.so`), `CublasMatmul.java` (handle management + dequant + gemv). Dequant kernels: `dequant_q4_k_f16.cu`, `dequant_q4_k_f32.cu`, `convert_f32_to_f16.cu`.

**Tradeoff**: cuBLAS achieves higher bandwidth utilization (~55-80%) than custom Q4_K kernels (~22%), but FP16 weights are 3.5x larger than Q4_K. On bandwidth-limited GPUs (RTX 4050: 192 GB/s), custom Q4_K + CUDA graph is faster. cuBLAS becomes competitive on GPUs with >500 GB/s bandwidth (A100, H100) or when weights are already FP16.

#### Nemotron-H CUDA forward pass (`NemotronHCudaForwardPass`)

GPU-resident forward pass for the Mamba-2 + Attention + FFN hybrid architecture. Handles all three layer types on GPU.

**Mamba-2 kernels:** `mamba2_scan.cu` (SSM state update per head, `[nheads][headDim][stateSize]`), `mamba2_dt_softplus.cu` (discretize timestep), `mamba2_gate_norm.cu` (fused gate+grouped RMSNorm, norm_before_gate=False), `sqrelu.cu` (squared ReLU for FFN layers). Reuses `conv1d_short.cu`, `silu.cu`, `rmsnorm.cu`, `rope.cu`, `attention.cu` for shared operations.

**CUDA graph**: now working — `NemotronH CUDA graph: captured 40 layers` confirmed on `granite-4.0-h-micro` (v1.11.0-dev). First generation may fall back to per-layer on a transient `cuMemcpyDtoH` error (906) during capture; subsequent generations replay the graph. Measured tok/s: **Nemotron-3-Nano-4B 20.0** (vs llama.cpp 35.6 = 56%), **Granite 4.0-h-micro 35.6** (vs llama.cpp 41.8 = 85%).

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

See `BENCHMARKS.md` for full results (34+ models tested across 20 architectures), `PERFORMANCE-ANALYSIS.md` for detailed per-kernel profiling, and `docs/optimization/llamacpp-comparison.md` for the rolling tok/s vs llama.cpp comparison plus a journal of optimization attempts (cubin, cp.async, multi-warp, mmvq, dp4a) with measured outcomes for each.

Current best: Llama-3.2-1B Q4_K_M at **55.8 tok/s** (CUDA graph mode, RTX 4050 Laptop GPU).

### CUDA kernel design patterns

All CUDA matmul kernels use 1 warp (32 threads) per output row with `__shfl_down_sync` reduction. Grid: `ceil(rows / (blockSize/32))` blocks.

**CUDA alignment constraints**: `__ldg((const unsigned int*)ptr)` requires 4-byte aligned `ptr`. Block sizes NOT divisible by 4: Q8_0 (34B), Q4_0 (18B), Q5_0 (22B), Q6_K (210B), Q3_K (110B), IQ4_NL (18B), IQ3_XXS (98B), IQ3_S (110B), IQ2_S (82B) — these kernels use byte-level `__ldg` only. Block sizes safe for uint32 `__ldg`: Q4_K (144B), Q5_K (176B), Q5_1 (24B), IQ4_XS (136B).

### JVM tuning properties

All properties are set via `-Dproperty=value` on the Java command line.

| Property | Default | Description |
|----------|---------|-------------|
| `cuda.nograph` | `false` | Disable CUDA graph capture; use per-layer kernel launches instead |
| `cuda.cublas` | `false` | Enable cuBLAS for Q4_K matmul (pre-dequantizes to FP16; requires `libcublas.so`) |
| `cuda.cublas.fp32` | `false` | Use FP32 instead of FP16 for cuBLAS dequantization (more VRAM, slower) |
| `cuda.dp4a` | `true` | Enable `__dp4a` integer dot product on CudaForwardPass + Qwen35CudaForwardPass + NemotronHCudaForwardPass. Covers Q4_K, Q5_K, Q5_0 (code 50), Q8_0 (80), IQ4_NL (41), IQ4_XS (42). Quantizes input to Q8_1 then uses int8 dp4a. **+34% on Llama-1B** (47→63 tok/s on RTX 4050), **+92% on Nemotron-3-Nano-4B** (10.4→20.0), **+42% on Phi-3-mini IQ4_NL**. Q6_K stays on FP32 fallback by default (see below). |
| `cuda.dp4a.q6` | `false` | Enable Q6_K dp4a kernel (rewritten 2026-04-14 — bit-equivalent to FP32). **Default OFF** because it's actually SLOWER than the FP32 Q6_K kernel (70.6 vs 72.85 tok/s on Llama-1B): Q6_K block is 210 bytes (not 4-byte aligned) so the dp4a kernel must use byte loads, and that overhead overwhelms the dp4a benefit. Kept for correctness checks. |
| `cuda.dp4a.fused_gate_up` | `false` | Use a single fused kernel for gate+up dp4a matmuls (`matmul_q4_k_dp4a_fused_gate_up.cu`). Default OFF — measured marginal/regression on Llama-1B (70.9 vs 73.5 separate-dp4a) because input was already L1-cached and the fused kernel adds register pressure. |
| `cuda.dp4a.mw` | `false` | Use llama.cpp-style multi-warp Q4_K dp4a kernel (`matmul_q4_k_dp4a_mw.cu`: 4 warps × 32 lanes per row, 1 row per block). Slower for small models (Llama-1B: 51 vs 70 tok/s) due to per-block overhead — may help larger models with more super-blocks per row. |
| `cuda.q4k.coalesced` | `false` | Use coalesced Q4_K kernel variant (all threads process same group) |
| `cuda.q4k.smem` | `false` | Use shared-memory input tiling Q4_K kernel variant |
| `cuda.q4k.2warp` | `false` | Use 2-warp-per-row Q4_K kernel (splits column groups between two warps with shared-memory partial-sum reduction) |
| `cuda.q5k.smem` | `true` | Use shared-memory input kernel for Q5_K (+7% throughput) |
| `cuda.q6k.tiled` | `true` | Use tiled shared-memory kernel for Q6_K (256-element tiles, +9% throughput) |
| `cuda.deltanet.v2` | `true` | Use float4-vectorized DeltaNet kernel (+4% throughput) |
| `cuda.q6k.smem` | `false` | Use shared-memory input kernel for Q6_K (alternative to tiled variant) |
| `cuda.profile` | `false` | Enable CUDA per-section timing (adds `finish()` barriers; ~15-25% overhead) |
| `cpu.profile` | `false` | Enable CPU per-layer timing (standard InferenceEngine only) |
| `matmul.tiled` | `false` | Enable cache-friendly tiled matmul on CPU (Java 21+) |
| `kv.q8` | `false` | Use Q8_0 block-quantized KV cache (1.125 vs 4 bytes/elem; **+28% faster** for DeepSeek2 MLA) |
| `attn.flash` | `false` | Enable FlashAttention online-softmax (single-pass; ~10% slower on Java/CPU; opt-in) |
| `gemma4.nople` | `false` | Disable PLE pre-computation in Gemma 4 forward pass (debug flag) |

**GPU profiling**: `-Dcuda.profile=true` for per-section timing with `cudaContext.finish()` barriers. Note: only instrumented in `CudaForwardPass` (standard architectures), not `Qwen35CudaForwardPass`. **CPU profiling**: `-Dcpu.profile=true` (standard `TransformerBlock` only).

### Automated kernel autosearch (`autosearch.sh`)

`./autosearch.sh <model.gguf> [runs_per_config] [min_ppl]` runs a Karpathy-style greedy coordinate-ascent search over the entire `-D` flag matrix above. For each flag, it toggles the value, runs N benchmarks (best tok/s wins), accepts the change only if **tok/s improves AND PPL ≥ min_ppl**, otherwise reverts. Output: optimal config + speedup vs baseline. Empirically on RTX 4050 + Qwen3-4B Q4_K_M: defaults are already Pareto-optimal (no flag toggle improves tok/s without degrading PPL). The CUDA-graph baseline is critical — disabling it crashes PPL from 1.00 to 0.07 even though tok/s changes by only ~10%.
