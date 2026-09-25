# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation style

When producing documentation (README, BENCHMARKS, REST-API, FINE-TUNING, TOOL-CALLING, docs under `docs/**`, Javadoc, and any `.md` authored for the project), always use normal prose — full sentences, articles, standard grammar. Do **not** use caveman / ultra-terse style in documentation, even if an active session has caveman mode enabled. Caveman style applies only to in-session assistant chat, not to persisted project artifacts.

## Project Overview

LLMPlayer is a pure Java LLM inference engine (v1.18.0) that runs GGUF models locally with **zero external dependencies** — only the JDK. It supports 29 architectures (plus aliased variants: Qwen2.5-VL, Qwen3-VL, LFM2-MoE, GLM4-MoE, the Qwen3-TTS talker) across nine inference engines and 18 quantized formats (F32, F16, BF16, Q2_K, Q3_K, Q4_0, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0, IQ2_S, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, MXFP4). Seventeen of those formats have dedicated CUDA tensor classes for full GPU acceleration (all except Q2_K).

Beyond text generation it includes image input for Qwen3-VL, Qwen3.5 and Qwen2.5-VL (llama.cpp `mmproj` files), Qwen3-TTS text-to-speech, CUDA GPU acceleration with graph mode, thinking/reasoning mode, architecture-aware tool calling, HuggingFace model download, JMX metrics, automated kernel autosearch, and a built-in LoRA fine-tuning pipeline.

The zero-dependency constraint is load-bearing: it is why there is no JUnit (`src/test/java` exists but is empty), why GPU access goes through Panama FFM rather than a binding library, and why JSON is parsed by hand in the `web` package. Do not add a dependency to `pom.xml` without raising it first.

## Build & Run

```bash
# Compile (Java 25, default profile — includes java21 + java25 sources)
mvn clean compile

# Compile without Java 25 optimizations (no StructuredTaskScope, no virtual thread matmul)
mvn clean compile -Pjava21

# Compile for Java 8 (no Vector API, no GPU)
mvn clean compile -Pjava8

# Run
./run.sh [options]        # launcher with all required JVM flags
mvn exec:java
```

**Build environment note:** if the system JDK is not Java 25, set `JAVA_HOME` first, e.g. `export JAVA_HOME=/usr/lib/jvm/jdk-25.0.2+10 && export PATH=$JAVA_HOME/bin:$PATH`.

**The `java21` profile still uses `<release>25</release>`** — it requires a Java 25 compiler but excludes the `java25/` source root and `--enable-preview`. Only the `java8` profile actually targets an older release.

**All `*.sh` and `*.bat` files in this repo are gitignored**, including `run.sh`, `run.bat`, and every test and benchmark script named below. They are local-only working files — a fresh clone will not have them, and they should not be committed. `.claude/` and `gguf/` are gitignored too. README.md carries the contents of the launcher scripts so they can be recreated.

### JVM flags

Java 21+ builds require these at both compile and runtime:

- `--add-modules jdk.incubator.vector` (SIMD Vector API)
- `--enable-native-access=ALL-UNNAMED` (Panama FFI for mmap, OpenCL, and CUDA)

Java 25 builds additionally require `--enable-preview` (StructuredTaskScope, virtual threads).

`run.sh` sets no heap limit, but the default heap is not enough for anything past a ~1B model. Pass `-Xmx16g` or more when invoking `java` directly; the batch scripts use `-Xmx16g` to `-Xmx28g`.

## Testing

There is no unit test framework. Verification is done by running models end to end.

**The inner loop — exercise one model directly.** This is what to run after touching a kernel, a tensor class, or an engine:

```bash
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED --enable-preview \
  -Xmx16g -cp target/classes it.denzosoft.llmplayer.LLMPlayer \
  --model gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --prompt "Write a Java method that reverses a linked list." \
  --max-tokens 32 --no-gpu
```

Add `--no-gpu` to force the CPU path; drop it to exercise the GPU path. Running both and diffing the output is the standard A/B for validating a new GPU kernel — a correct kernel should be bit-identical to CPU, or within the known Q8_1 quantization noise when dp4a is active. The run prints tok/s and an aggregate quality score, so a single invocation covers both a smoke test and a rough perplexity check.

Batch scripts (all gitignored, see above):

| Script | Purpose |
|---|---|
| `test-architectures.sh [--gpu]` | Smoke test — loads one model per supported architecture from `gguf/` and verifies it generates at least one token. Auto-detects a Java 21+ JDK. Set `INCLUDE_LARGE=1` to include 30B+ MoE models. Missing model files are skipped, not failed. |
| `test-openai-api.sh` | Integration tests for the OpenAI-compatible API. Requires a running server (`./run.sh --web` first). Covers 6 architectures with streaming, non-streaming, multi-turn, system messages, CORS, error handling, and Bearer token acceptance. |
| `test-ppl-sweep.sh` / `test-cpu-sweep.sh` | Quality regression sweeps — run a list of models through a canonical prompt and print aggregate PPL per model. Use these to detect quality regressions across a release. |
| `autosearch.sh <model.gguf> [runs] [min_ppl]` | Karpathy-style greedy coordinate ascent over the `-D` flag matrix, optimizing tok/s subject to a PPL floor. Writes a Pareto-optimal config to `autosearch-results.txt`. See [`docs/optimization/jvm-flags.md`](docs/optimization/jvm-flags.md). |

By convention GGUF models live in `gguf/` at the repo root; `--download` writes there by default. The `chats/` directory holds chat persistence JSON when running with `--web`.

## Architecture

### Multi-source compilation and reflection loading

The project compiles from three source roots under the default `java25` profile:

- `src/main/java/` — core code, Java 8 compatible
- `src/main/java21/` — SIMD tensor ops, SIMD quantized tensors, OpenCL and CUDA bindings, GPU forward passes
- `src/main/java25/` — StructuredTaskScope batch generation, virtual thread matmul, matmul benchmark

**Classes in `java21/` and `java25/` are never imported directly from base code.** They are loaded via `Class.forName()` reflection with try/catch fallbacks, which is what allows graceful degradation on older JVMs. When adding a new java21/java25 feature, follow this pattern: implementation in the appropriate source root, loaded reflectively from base code, with a Java 8 compatible fallback.

| Loading site | Java 21+ class | Fallback |
|---|---|---|
| `VectorOpsFactory` static init | `SimdVectorOps` | `ScalarOps` |
| `TensorDataFactory.mapFile()` | `MemorySegmentTensorData` | `ByteBufferTensorData` |
| `TensorFactory.tryCreateGpuTensor()` | `Q4_KGpuTensor` / `Q4_KCudaTensor`, etc. | CPU tensor variant |
| `LLMEngine.initGpu()` | `CudaContext` + `CudaBufferManager` (preferred) or `OpenCLContext` + `GpuBufferManager` | CPU-only |
| `TensorFactory.create()` for Q4_K/Q8_0/Q6_K/Q5_0/Q5_K/Q3_K | `Simd*FloatTensor` variants | scalar `*FloatTensor` variants |
| `FloatTensor.tryTiledMatmul()` | `TiledMatmul` | standard `matmulParallel()` |
| `FloatTensor.tryVirtualThreadMatmul()` | `VirtualThreadMatmul` (only when `MatmulPool` is off: GPU active or `-Dmatmul.pool=false`) | `IntStream.parallel()` ForkJoinPool matmul |
| `LLMEngine.tryStructuredBatch()` | `StructuredBatchGenerator` | `ExecutorService` thread pool |
| `InferenceEngine.tryInitGpuForwardPass()` | `CudaForwardPass` (preferred) or `GpuForwardPass` | CPU forward pass |
| `CLIRunner.listGpuDevices()` | `CudaContext.enumerateDevices()` + `OpenCLContext.enumerateDevices()` | empty list |

### Package structure (`it.denzosoft.llmplayer`)

| Package | Purpose |
|---------|---------|
| `api` | Public facade — `LLMEngine` is the entry point for programmatic use |
| `cli` | Argument parsing (`CLIOptions`), interactive runner, HuggingFace downloader |
| `evaluator` | Response quality metrics (perplexity, coherence, length) |
| `gguf` | GGUF format parser — memory-mapped with parallel preload |
| `gpu` | GPU config (base); CUDA and OpenCL bindings and buffer management (java21) |
| `inference` | Transformer forward pass — eight engines, plus `LayerPrefetcher` for async next-layer mmap page-in on lazy larger-than-RAM dense loads |
| `model` | Model loading, config extraction from GGUF metadata, weight structures |
| `sampler` | Token sampling (temperature, top-k, top-p, repetition penalty) |
| `tensor` | Tensor ops and quantization/dequantization; GPU variants in java21 |
| `tokenizer` | BPE and SentencePiece tokenizers, chat template formatting |
| `tuning` | LoRA fine-tuning — chunking, Q&A dataset generation, training loop, merge, GGUF export |
| `ui` | Swing desktop GUI |
| `web` | Embedded HTTP server, OpenAI API, Anthropic API, management API, chat persistence |
| `vision` | Image input: `ImagePreprocessor` (javax.imageio decode, smart resize, Pillow-style bicubic) and `VisionEncoder` (`qwen3vl_merger` ViT + merger + deepstack; `qwen2.5vl_merger` with window attention). Wired through `LLMEngine.loadVisionProjector` / `GenerationRequest.images()` |
| `tts` | Qwen3-TTS: `Qwen3Tts` (talker prompt + frame loop), `Qwen3TtsCodec` (code predictor + streaming code2wav decoder), `WavWriter`. CLI `--tts` |
| `spec` | Speculative decoding (`SpeculativeDecoder`) — drives two `LLMEngine` instances via `forwardSingleToken`. Sequential verification only (~1.1× max at K=4); real 2–3× awaits a working `forwardBatch`. Enabled with `--draft-model <gguf>`. |

### Key data flow

1. `GGUFParser` memory-maps the model file and extracts metadata plus tensor info.
2. `ModelLoader` builds `ModelConfig` from metadata, creates weight tensors via `TensorFactory`, and instantiates the tokenizer via `TokenizerFactory`.
3. `LLMEngine.load()` wraps everything into the public API and selects the inference engine by architecture.
4. `generate()` tokenizes the prompt (applying the chat template if enabled), runs prefill, then auto-regressive decoding with the configured sampler.
5. Each `generate()` call creates its own `InferenceState`. Model weights are immutable mmap'd memory, which is what makes `LLMEngine` thread-safe.

### Inference engine dispatch (nine paths)

Selected in `LLMEngine.load()` and `ModelLoader`. **Full per-engine detail is in [`docs/architecture/inference-engines.md`](docs/architecture/inference-engines.md)** — read the relevant section there before modifying an engine, because most of the behaviour mirrors a specific upstream llama.cpp implementation.

| Engine | Architectures | Shape |
|---|---|---|
| `InferenceEngine` | Llama, Qwen2, Qwen3, SmolLM3, GLM4, Gemma 2/3, Phi-3/4, Mistral3, Command-R, OLMo2, Falcon3, GPT-OSS, Granite 3.3, ERNIE 4.5, Qwen2.5-VL/Qwen3-VL text, Hunyuan dense, Nanbeige (looped), Spark2.5 | Standard `TransformerBlock` → GQA `Attention` + `SwiGLUFFN` |
| `DeepSeek2InferenceEngine` | DeepSeek2, GLM-4.7-Flash | MLA + MoE FFN with shared expert |
| `Qwen3MoEInferenceEngine` | Qwen3-Coder-30B-A3B, Llama 4 MoE, GLM4 MoE (`glm4moe`, GLM-4.5/4.6/4.7) | GQA with QK-norm + MoE FFN with shared expert; sigmoid + `exp_probs_b` routing for GLM4 MoE |
| `Qwen35InferenceEngine` | Qwen3.5 | Hybrid Gated DeltaNet + full attention, 3:1 ratio |
| `NemotronHInferenceEngine` | Nemotron-H, Granite Hybrid (incl. MoE) | Three distinct layer types: Mamba-2 SSM, GQA, squared-ReLU FFN |
| `Gemma4InferenceEngine` | Gemma 4 (PLE E2B/E4B and dense 12B/31B), Gemma 3n | PLE + dual head size + shared KV; AltUp/Laurel only for 3n |
| `LFM2InferenceEngine` | LFM2, LFM2-MoE | Gated short-conv mixers alternating with GQA, each + SwiGLU (routed experts past the leading dense blocks on LFM2-MoE) |
| `FalconH1InferenceEngine` | Falcon-H1 | **Parallel** attention + Mamba-2 over one shared normed input |
| `BailingMoE3InferenceEngine` | Ling 3.0 (`bailingmoe3`) | Kimi Delta Attention layers + gated MLA layers, group-limited MoE with shared expert; loads its own tensors |

### Tensor system

`FloatTensor` is the core abstraction; each quantization format has a subclass implementing dequantization inline. `TensorFactory.create()` selects an implementation by `GGMLType`, trying a GPU variant first (CUDA or OpenCL per `TensorFactory.gpuBackend`) and falling back to CPU.

Kernels may also override `matmulRows` (row range for one input, used by the pool) and `matmulRowsBatch` (row range for several inputs, used by batched prefill) to amortise per-input work; see [`docs/optimization/cpu-dispatch-and-kernels.md`](docs/optimization/cpu-dispatch-and-kernels.md). `SimdQ6_KFloatTensor` can also repack itself losslessly into an off-heap int8 layout for those two paths (`-Dq6k.repack=true`, opt-in): the kernel is 2.2× faster on one core, but multi-threaded decode is memory-bandwidth bound and the copy is 1.31× the bytes, so it measured slower end to end. A faster kernel is not a faster decode unless the bytes per weight stay the same. For CPU SIMD kernels, follow the B2I/I2F lane-parallel template in [`docs/optimization/simd-kernel-pattern.md`](docs/optimization/simd-kernel-pattern.md) — deviating from it is the usual cause of a "SIMD" kernel that is actually scalar in the hot loop.

### Tokenizer dispatch

`TokenizerFactory` reads `tokenizer.ggml.model`: `"gpt2"` or `"bpe"` → `BPETokenizer`; `"gemma4"` → `BPETokenizer` with `useGpt2ByteMapping=false`; anything else → `SentencePieceTokenizer`. Chat formatting is architecture-specific in `ChatTemplate`. Details and the full template list are in [`docs/architecture/inference-engines.md`](docs/architecture/inference-engines.md#tokenizer-and-chat-template-dispatch).

### Launch modes

- **No args** → Swing desktop GUI (`LLMPlayerUI`)
- **`--web`** → embedded `com.sun.net.httpserver.HttpServer` on port 8080 (`--port` to change)
- **`--model <path>`** → CLI mode, with `--prompt` or `--interactive`
- **`--fine-tune`** → LoRA pipeline (requires `--target-model` plus one of `--source`, `--documents`, `--data`, `--train-dataset`)
- **`--gpu-list`** → enumerate CUDA and OpenCL devices and exit
- **`--download <repo>`** → download GGUF from HuggingFace (`owner/repo` or `owner/repo/file.gguf`)

### Web APIs

Four API groups in `--web` mode. **Full endpoint documentation is in `REST-API.md`**; client setup (Continue.dev, Cursor, aider, Open WebUI) is in `CODING-ASSISTANTS.md`.

| Group | Handler | Notes |
|---|---|---|
| `/v1/chat/completions`, `/v1/embeddings`, `/v1/models` | `OpenAIHandler` | OpenAI-compatible. Tool calling, JSON mode, SSE streaming. |
| `/v1/messages`, `/v1/messages/count_tokens` | `AnthropicHandler` | Anthropic Messages API, for Claude Code and similar clients. |
| `/api/*` | `ApiHandler` | Model load/unload, GPU enumeration, memory check, hardware plan, `/api/metrics`. |
| `/api/chats/*` | `ChatHandler` | Conversation persistence with tree-based branching — messages form a flat `id → message` map with `parentId`/`children`, so editing a message creates a sibling branch. |

Cross-cutting behaviour worth knowing before editing a handler: the `model` field and all auth headers (`Authorization: Bearer`, `x-api-key`) are accepted and ignored, and **only one generation runs at a time — concurrent requests get HTTP 429**.

UI resources: `web-ui.html` (model config, served at `/`) and `chat-ui.html` (chat with branching, served at `/chat`).

Runtime metrics are also exposed via JMX at `it.denzosoft.llmplayer:type=LLMPlayer` (`LLMPlayerMXBean` / `LLMPlayerMetrics`, a thread-safe singleton), carrying the same data as `GET /api/metrics` with a 60-second rolling tok/s window.

### Fine-tuning pipeline (`tuning`)

Pure Java LoRA with checkpoint/resume, in six stages: analyze (`TargetAnalyzer`) → chunk (`DataChunker`/`CodeChunker`/`TextChunker`) → generate Q&A (`QAGenerator`) → train (`TrainingLoop` + `LoRAAdapter`) → merge (`LoRAMerger`) → export (`GGUFWriter`). Three data scenarios are auto-detected from the CLI flags: `--source` (code), `--documents` (text), `--data` + `--schema` (structured/SQL). Generation can be decoupled from training via `--dataset-only` and `--train-dataset`.

`LLMEngine.forwardSingleToken(token, position)` is the training-specific API: one forward pass returning logits, with a lazily created persistent inference state. It works with the standard, DeepSeek2, and Qwen3MoE engines. Full documentation in `FINE-TUNING.md`.

## GPU

### Backends

Two backends, both via Panama FFM so neither adds a dependency:

- **CUDA** (`CudaBindings` / `CudaContext` / `CudaBufferManager`) calls `libcuda.so` and `libnvrtc.so`; `.cu` kernels are compiled at runtime by NVRTC. Preferred.
- **OpenCL** (`OpenCLBindings` / `OpenCLContext` / `GpuBufferManager`) calls `libOpenCL.so`; `.cl` kernels are compiled by the driver.

`--gpu-backend` selects `auto` (default, prefers CUDA), `cuda`, or `opencl`. `TensorFactory.gpuBackend` then determines whether `*CudaTensor` or `*GpuTensor` classes are created.

**Auto-detection** (`LLMEngine.autoConfigureGpu()`) tries CUDA GPU, then OpenCL GPU, and returns `null` for the CPU SIMD path if only OpenCL CPU devices (PoCL) are available — OpenCL on CPU adds marshaling overhead with no compute benefit. Device type comes from the `deviceType` field of `enumerateDevices()` (type "2" or containing "CPU"), with a name-based fallback for "cpu" and "pocl".

### Placement strategies

**First-N-layers** is the default for dense architectures: the first N layers go entirely on GPU, the rest on CPU, with `--gpu-layers -1` auto-detecting N and `--gpu-layers N` forcing it. The budget is **KV-aware** — each GPU layer reserves its KV slice, which grows with context, via `N = (usableVram − nonLayerBytes) / (bytesPerLayer + kvPerLayer)`. When FP32 KV would not fit all layers but FP16 KV would, FP16 KV is auto-enabled.

**MoE-optimized** applies to MoE architectures with `--gpu-layers -1`. Inspired by KTransformers (SOSP'25), it places **all** attention tensors on GPU across every layer while expert tensors (`ffn_*_exps`, 80–90% of layer weight) stay on CPU; routers and shared experts also go on GPU. Expert tensors are large but only top-K activate per token, so GPU parallelism is wasted on them, whereas attention is compute-bound and pays off on every token. With 6 GB VRAM, first-N-layers fits about 2 of 48 layers while MoE-optimized fits 100% of attention. Detection: quick-parse for `expertCount > 0`, sum non-expert bytes via `sumNonExpertTensorBytes()`, and enable if that total is within 80% of VRAM. `ModelLoader` implements this by toggling `TensorFactory.gpuBufferManager` on and off around each tensor group inside the layer loop.

**Explicit `--gpu-layers N` always uses first-N-layers**, with no MoE optimization, for backward compatibility.

**SSD streaming for MoE models larger than RAM.** When `LLMEngine.load` skips the preload (model above 85% of RAM), the routed experts stay on disk. `MappedExpertCache` (java21, loaded reflectively via `ExpertCacheFactory`) caches whole expert slices in off-heap slots filled by `FileChannel.read(ByteBuffer, position)`, with least-frequently-used retention so the hot experts stay resident — gate, up and down are one slot, since the router never needs one without the others. Budget via `--expert-cache-size <MB>`. This is a MoE-only feature: a dense model needs every weight every token, so no cache policy can help and `LayerPrefetcher` already does the only useful thing. Measured 3.2× on Qwen3-Coder-30B. Batched MoE prefill uses 256-token chunks when the cache is active (bytes read scale with the number of chunks) and reads the next expert group while the current one computes; `findVictim` therefore protects the slots of the previous `prepare()` too, and slices must be resolved before the next `prepare()` starts. See [`docs/optimization/ssd-streaming-cache.md`](docs/optimization/ssd-streaming-cache.md), which also records why the mmap-hint approach (`MADV_WILLNEED`) cannot reach expert-granular bandwidth.

**`--auto-tune`** measures steady-state decode tok/s for the heuristic placement and for CPU-only and keeps the faster one. It exists to correct the partial-fit footgun where GPU placement is genuinely slower than CPU. One-time and opt-in, since it loads the model a couple of extra times (`CLIRunner.autoTune()` / `measurePlacement()`).

The matmul thread pool defaults to **physical** cores, not logical ones (`CLIOptions.detectPhysicalCores()` via Linux sysfs thread-sibling groups), because the matmul hot path is bandwidth-bound and two hyperthreads on one core only contend for its load/store ports. `--threads N` overrides.

Every placement decision rule, with thresholds and file:line citations, is in [`docs/optimization/autotuning-heuristics.md`](docs/optimization/autotuning-heuristics.md); the bandwidth cost model and roadmap behind it are in [`docs/optimization/placement-autotuning.md`](docs/optimization/placement-autotuning.md).

### GPU-resident forward pass

`CudaForwardPass` keeps activations on GPU between layers, in either per-layer mode or **CUDA graph mode** (captures all launches on the first token, then replays with a single `cuGraphLaunch`; default on, disable with `-Dcuda.nograph=true`). Dedicated passes exist for Qwen3.5, Nemotron-H, LFM2, Falcon-H1, and Gemma 4. The **dp4a int8 path is default-on** (`cuda.dp4a`) and covers Q3_K, Q4_K, Q5_K, Q5_0, Q8_0, IQ4_NL, and IQ4_XS.

The design constraint across all of these is **zero-allocation hot paths**: all `ParamBuffer` and `MatmulLaunch` objects are pre-allocated in the constructor, and `forwardLayer()` only writes parameter values in place.

Full detail — supported architectures per pass, the internal `forwardAttentionPart`/`forwardFFNPart` split, sliding-window handling, cuBLAS, kernel inventory, and per-kernel profiling findings — is in [`docs/optimization/cuda-forward-pass.md`](docs/optimization/cuda-forward-pass.md).

## Invariants that break things

These are the traps that have actually caused regressions. Check them before and after touching the relevant area.

**`attention_full` takes 10 parameters.** The tenth is `slidingWindow` (0 meaning full attention). Every caller's attention `ParamBuffer` must be size 10 and must set arg index 9. `CudaForwardPass`, `NemotronHCudaForwardPass`, and `Qwen35CudaForwardPass` all launch this kernel; when SWA added the tenth parameter the latter two were initially left at 9, causing a native `cuLaunchKernel` SIGSEGV on every Nemotron-H, Granite Hybrid, and Qwen3.5 GPU run (fixed 2026-06-07). If you change the arity, update all three.

**Gemma 4 hyperparameters are UINT32 scalar in some GGUFs and INT32 array in others.** `GGUFMetadata.getIntArray` returns non-null only for INT32 arrays, which is why `ModelConfig` falls back to the scalar when the per-layer array is absent. E2B/E4B ship `head_count_kv` as a scalar while the 12B ships it as an array; E4B ships `feed_forward_length` as a scalar while E2B/E4B ship it as an array. Removing either branch silently breaks part of the family.

**Token embeddings are deliberately loaded on CPU** across all architectures. They are used only for a single-element lookup per token (~16 KB), so GPU residency wastes 500+ MB of VRAM. When output weights are tied to embedding weights, the output tensor is reloaded separately on GPU for the projection matmul. Do not "fix" this by moving the embedding to GPU.

**The CPU hot path runs on `MatmulPool`** (base `tensor` package): persistent platform workers that claim row chunks dynamically, sized from `--threads` via `-Dmatmul.threads`. Matmuls, per-head attention and per-expert MoE loops all go through it (`MatmulPool.forEach` replaces `IntStream.parallel()`), and a dispatch made while it is busy runs inline, so nesting is safe. **It is force-disabled together with virtual thread matmul whenever a GPU is active**, via `FloatTensor.disableVirtualThreadMatmul()`, because native GPU threads conflicted with the JVM's virtual thread carrier threads; GPU mode keeps its previous dispatch. Batched prefill (`InferenceEngine.forwardPrefill`) is gated on the pool being enabled, which is also what keeps GPU tensors out of the multi-token kernels.

**The whole Gemma family (Gemma 2, 3, 3n, 4) uses NEOX RoPE**, as in llama.cpp, because the GGUF converter permutes Q/K only for Llama-style checkpoints. It was NORMAL until 2026-09-23, which left short prompts plausible and broke long ones (Gemma-3-1B produced no output past about 50 prompt tokens). Dense GLM4 (`glm4`) had the opposite error: it is NORM in llama.cpp but was NEOX until 2026-09-25 (GLM-4-9B perplexity 5.45 → 2.60 once fixed); only GLM-4.xV GGUFs with `rope.dimension_sections` stay NEOX. When adding an architecture, take the RoPE type from llama.cpp's `llama_model_rope_type`, and test with a prompt of a few hundred tokens: a wrong pairing is invisible at position 0.

**Batched prefill must keep warming the decode kernels.** It never calls the single-token kernels, so without `FloatTensor.warmUpRows` (run before the first batched prefill in `InferenceEngine`, `Qwen35InferenceEngine`, `Gemma4InferenceEngine`, `Qwen3MoEInferenceEngine` and `DeepSeek2InferenceEngine`; the MoE engines also call `FloatTensor.warmUpDot`, because decode computes routed experts with one `dot` per row) their C2 compilation starts only at decode and decode runs on C1 code, which does not intrinsify the Vector API — Llama-1B decode fell to 3.8–7.4 tok/s from 8–11.8. Any new engine that gets batched prefill needs the same call, and kernels must not allocate per call (use `KQuantInput.Scratch`).

**Vector API kernels can silently lose intrinsification.** A Q8_0 variant that differed from the shipped one only by using one accumulator instead of two ran 5× slower, and a packed Q6_K kernel fell to 0.8 Gelem/s. Holding constant vectors in `static final IntVector` fields also breaks it: `selectFrom` on such a field failed with "missing constant" and the IQ4_NL kernel ran 15× slower in most runs — build them from arrays inside the method. Benchmark every kernel change in place (in-process A/B, several JVM runs, since the failure can be intermittent), and keep the SIMD guard at `SPECIES_PREFERRED.length() < 8`: the kernels use 256-bit shapes explicitly, and a `!= 8` test sends AVX-512 hosts to the scalar path.

**Qwen-VL multi-axis RoPE gives text tokens `(p, p, p, 0)`, not `(p, p, p, p)`.** llama.cpp sets the fourth position to 0, and with Qwen3-VL's interleaved sections `[24, 20, 20, 0]` two pairs land in that section, so the text rotation is not exactly NEOX. `Attention.normAndRope` and `Qwen35InferenceEngine` take positions from `MRopePositions` whenever `ModelConfig.ropeSections()` is set; image tokens only work through that path, so `LLMEngine.loadVisionProjector` drops any GPU-resident pass (`disableGpuForwardPass`). Architectures whose layer math only the CPU `TransformerBlock` implements (Hunyuan, Spark2.5, looped Nanbeige, and M-RoPE layouts that are not plain NEOX for text — `ModelConfig.mropeDiffersFromNeox()`, i.e. Qwen3-VL and the Qwen3-TTS talker) must be declined by every GPU-resident pass via `ModelConfig.requiresCpuLayerPath()`.

**Two flags are measured dead** and must not be enabled on current hardware: `matmul.tiled` (-50% on Llama-1B CPU) and `cuda.q4k.cpasync` (-2.8%). They are retained only so the measurement is not repeated.

**A model's filename does not tell you its quantization mix.** Q4_K_M ships Q6_K for `output.weight` and for some `ffn_down`/`attn_v` tensors, and Gemma-3 Q4_K_M ships Q5_0 for Q/K/gate/up. Profile with JFR (CPU) or `-Dcuda.profile=true -Dcuda.nograph=true` (GPU) to find which kernel is actually hot before optimizing.

## Extending

### Adding a new model architecture

1. Add an enum value to `ModelArchitecture` with its GGUF `general.architecture` string.
2. Add any architecture-specific tensor name patterns to `ArchitectureRegistry` (standard names like `blk.{n}.attn_q.weight` are shared across most architectures).
3. Update `ModelConfig.fromMetadata()` if it uses non-standard metadata keys for hyperparameters.
4. Add a chat template branch in `ChatTemplate.formatUserMessage()` and `formatConversation()`.
5. If the forward pass differs from standard attention+FFN, create a dedicated inference engine — see `DeepSeek2InferenceEngine` for MLA+MoE, `Qwen35InferenceEngine` for DeltaNet+attention, `NemotronHInferenceEngine` for Mamba-2+attention+FFN.
6. If it supports tool calling, add format methods in `ChatTemplate` (SmolLM3's Hermes-style implementation is the reference).
7. If it supports thinking/reasoning, add handling in `ChatTemplate` and `CLIOptions`.

### Adding a new quantization type

1. Add the type to `GGMLType` with its block size and type size.
2. Create a `FloatTensor` subclass implementing `getFloat()` and `dotProduct()` with the dequantization math.
3. Add the case to `TensorFactory.create()`.
4. Optionally add GPU variants in `java21/` — CUDA (`*CudaTensor` + `kernels/cuda/*.cu`) and/or OpenCL (`*GpuTensor` + `kernels/*.cl`). Check the alignment constraints in [`docs/optimization/cuda-forward-pass.md`](docs/optimization/cuda-forward-pass.md#cuda-kernel-design-patterns) first — a block size not divisible by 4 forces byte-level `__ldg` and changes the performance you can expect.
5. Optionally add a CPU SIMD variant following [`docs/optimization/simd-kernel-pattern.md`](docs/optimization/simd-kernel-pattern.md).

### Thinking mode and tool calling

Thinking is enabled with `--thinking` or `"thinking": true` in an OpenAI API request. SmolLM3 injects `/think` into the system prompt; Qwen3 and Qwen3.5 suppress thinking by default and `--thinking` removes the suppressor.

Tool calling is architecture-aware: SmolLM3 uses Hermes-style XML (`<tool_call>` / `<tool_response>`), and other models use generic JSON prompt injection. Format logic is in `ChatTemplate.formatToolsSystemPrompt()`, `formatToolResult()`, and `formatAssistantToolCalls()`; parsing is in `OpenAIHandler.tryParseToolCalls()`, which handles multiple calls. Full documentation in `TOOL-CALLING.md`.

## Reference documents

| Document | Contents |
|---|---|
| `README.md` | User-facing overview, full CLI reference, launcher script contents |
| `WHATS-NEW.md` | Release-by-release deltas — the fastest way to find when a behaviour changed |
| `BENCHMARKS.md` | Results across 34+ models and 20 architectures |
| `REST-API.md` / `TOOL-CALLING.md` / `FINE-TUNING.md` | Per-subsystem user documentation |
| `CODING-ASSISTANTS.md` | Wiring LLMPlayer into Continue.dev, Cursor, aider, Open WebUI |
| `docs/architecture/inference-engines.md` | Per-engine internals for all nine paths, tokenizer dispatch |
| `docs/architecture/vision-and-tts.md` | Image input (Qwen3-VL/Qwen3.5 mmproj, multi-axis RoPE positions) and the Qwen3-TTS pipeline |
| `docs/optimization/cuda-forward-pass.md` | GPU-resident passes, dp4a, kernel conventions, cuBLAS, kernel inventory |
| `docs/optimization/jvm-flags.md` | Complete `-D` property matrix with measured effects |
| `docs/optimization/simd-kernel-pattern.md` | CPU SIMD B2I/I2F template and reference implementations |
| `docs/optimization/cpu-dispatch-and-kernels.md` | `MatmulPool`, the Q4_K/Q5_K/Q8_0/IQ4_NL/IQ4_XS kernel rewrites, batched prefill for the standard engine, the AVX-512, `--threads` and Gemma RoPE fixes, and the approaches measured and rejected (int8 path, packed Q6_K) |
| `docs/optimization/per-token-latency-analysis.md` | Measured phase profile of the CPU forward pass and a ranked list of remaining wins — including a verified defect where prefill computes and discards the output projection for every prompt token |
| `docs/optimization/ssd-streaming-cache.md` | Running MoE models larger than RAM by streaming experts from SSD — I/O measurements, the L0/L1/L2 design, and why mmap hints cannot reach expert-granular bandwidth |
| `docs/optimization/autotuning-heuristics.md` | Placement decision rules with file:line citations |
| `docs/optimization/placement-autotuning.md` | Bandwidth cost model, value-per-VRAM-byte ranking, roadmap |
| `docs/optimization/llamacpp-comparison.md` | Rolling tok/s gap vs llama.cpp plus a journal of optimization attempts and outcomes |
| `docs/optimization/speculative-decoding.md`, `qwen35-profile-analysis.md`, `option-a-ptx-attempt.md`, `option-c-cpasync-attempt.md`, `tier2-attempt.md` | Individual investigations |
| `docs/quantization/*.md` | One document per quantization format (18 files) — read before implementing a new tensor class |
| `docs/models/*.md` | Per-model reports (37 files) — check here first when debugging a specific model |
| `PERFORMANCE-ANALYSIS.md` | Detailed per-kernel profiling |

**`ANALYSIS.md` is a historical document** (roughly v1.2, early 2026) and its counts are stale — it says 21 architectures and 16 CUDA kernels against today's 25 and 17. Its own header says as much. Do not treat it as current state.

Current best measured throughput: Llama-3.2-1B Q4_K_M at **55.8 tok/s** in CUDA graph mode on an RTX 4050 Laptop GPU.
