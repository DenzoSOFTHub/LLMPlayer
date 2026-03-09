# LLMPlayer — What's New

## v1.5.0 (2026-03-08)

### BPE Tokenizer Fix — Non-ASCII Text Corruption

Fixed a critical bug in the GPT-2 byte-level BPE tokenizer that corrupted all non-ASCII text (accented characters, Cyrillic, Greek, CJK, emoji). The byte-to-Unicode mapping for bytes 127-160 used an incorrect formula, producing wrong codepoints (e.g., U+017F instead of U+0121 for byte 127). Both `byteToToken()` encoding and `tokenCharToByte()` decoding paths were corrected, along with missing identity-mapped byte ranges (161-172, 174-255). A buffer overflow in `decodeTokenPiece()` was also fixed (allocated `piece.length()` bytes but multi-byte UTF-8 chars could exceed that). Affects all BPE models: Llama 3, Mistral, Qwen2/3, OLMo, Aya, Yi-Coder.

### CUDA Forward Pass — Per-Head QK-Norm on GPU

Models with per-head QK-norm (Qwen3-4B, DeepSeek-R1-Qwen3-8B, Qwen3-8B) now run the full CUDA forward pass including QK-norm on GPU. New `rmsnorm_per_head.cu` kernel normalizes each attention head independently — one CUDA block per head. Previously these models fell back to per-tensor CUDA matmul; now they can use the GPU-resident forward pass with CUDA graph for maximum throughput.

### New CUDA Kernels — Q5_0, IQ4_NL, IQ4_XS, IQ3_XXS

Added dedicated CUDA GPU kernels for Q5_0, IQ4_NL, IQ4_XS, and IQ3_XXS quantization formats. Q5_0 uses split nibble layout with 5th-bit recovery from `qh` field (byte-level `__ldg` reads due to 22-byte block alignment). IQ4 kernels use the non-linear K-means lookup table (16 entries). IQ3_XXS uses grid codebook lookup (256 uint32 entries) with sign lookup tables in CUDA `__constant__` memory.

The Q5_0 kernel is critical for Gemma 2/3 models, which use Q5_0 for Q, K, gate, and up projections in Q4_K_M quantization. With Q5_0 on GPU, Gemma-3-1B now achieves **35.7 tok/s** with CUDA graph (previously 4.0 tok/s with per-tensor fallback — **9x improvement**).

IQ4_NL Llama-3.2-1B now achieves **28.7 tok/s** with CUDA graph (previously 4.4 tok/s — **6.5x improvement**).

### CUDA Forward Pass — Post-Norm (Gemma 2/3)

Models with post-attention and post-FFN normalization (Gemma 2, Gemma 3) now run the full CUDA forward pass on GPU. Post-norm layers apply RMSNorm after attention/FFN output before the residual add. Wo and Down matmuls write to a separate buffer; post-norm is applied in-place, then accumulated into the residual stream.

### CUDA Forward Pass — Merged QKV (Phi-3/4)

Added `split_qkv.cu` kernel for splitting concatenated QKV output into separate Q, K, V buffers. Note: Phi-3/4 models also use packed FFN (gate+up combined in a single weight matrix), which is not yet supported in the CUDA forward pass. These models currently use per-tensor CUDA matmul.

### CUDA Optimizations

- **Combined upload**: embedding vector + token params uploaded in a single `cuMemcpyHtoD` via contiguous GPU buffer, saving one Panama FFM call per token.
- **Fused gate+up kernel** (`matmul_q4_k_fused_gate_up.cu`): single kernel launch for both gate and up projections when both are Q4_K. Halves kernel launch count for FFN phase.
- **GPU-side argmax** (`argmax.cu`): two-phase parallel argmax on GPU logits. Downloads 4 bytes instead of 512 KB for greedy sampling.

### HuggingFace Model Download

New `--download` CLI command to download GGUF models directly from HuggingFace Hub. Supports downloading by repository name (auto-selects Q4_K_M variant) or by specific filename. Skips download if the file already exists locally with matching size. Supports `--hf-token` for private/gated repositories.

```bash
# Download Q4_K_M (auto-selected) from a HuggingFace repo
./run.sh --download "bartowski/Llama-3.2-1B-Instruct-GGUF"

# Download a specific file
./run.sh --download "bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

# Use with custom GGUF directory and HuggingFace token
./run.sh --download "meta-llama/Llama-4-Scout-17B-16E-Instruct-GGUF" --gguf-dir models --hf-token hf_xxx
```

### SentencePiece Tokenizer Robustness

- Fixed `_` (U+2581) prepending: only the first non-special text part receives the prefix, not every part after special token splits.
- Added warning when byte fallback token `<0xHH>` is missing from vocabulary.

### Llama4 iRoPE Support

Added conditional RoPE for Llama4's interleaved RoPE architecture: every Nth layer (N=4) is a NoPE layer that skips rotary position encoding. Implemented in `Attention`, `Qwen3MoEInferenceEngine`, and `ModelConfig`. Requires a Llama4 GGUF model to test.

### Q5_0 Dequantization Fix

Fixed element ordering in Q5_0 quantization: elements 0-15 use LOW nibbles of bytes 0-15 (qh bits 0-15), elements 16-31 use HIGH nibbles of bytes 0-15 (qh bits 16-31). The previous implementation used interleaved ordering. This was the root cause of Gemma 3 garbage output, since Gemma 3 Q4_K_M uses Q5_0 for Q, K, gate, and up projections.

### Comprehensive Benchmarks

All 33 GGUF models in the test suite verified on both CPU (SIMD) and GPU (CUDA). Full results in BENCHMARKS.md. Highlights:
- Llama-3.2-1B Q4_K_M: 54.7 tok/s GPU — CUDA graph (17x over CPU)
- Qwen2.5-Coder-1.5B Q4_K_M: 41.5 tok/s GPU — CUDA graph (19x)
- Gemma-3-1B Q4_K_M: 35.7 tok/s GPU — CUDA graph (9x, unlocked by Q5_0 kernel)
- Llama-3.2-1B IQ4_NL: 28.7 tok/s GPU — CUDA graph (6.5x, unlocked by IQ4_NL kernel)
- Qwen3-4B Q4_K_M: 19.0 tok/s GPU — CUDA graph (unlocked by QK-norm support)
- DeepSeek-R1-Qwen3-8B Q4_K_M: 9.3 tok/s GPU — CUDA graph (unlocked by QK-norm support)

---

## v1.4.0 (2026-03)

### CUDA GPU Backend

Full CUDA GPU backend via Panama FFM — zero native dependencies. Calls `libcuda.so` and `libnvrtc.so` directly. CUDA kernels (`.cu` files) compiled at runtime by NVRTC into PTX.

- **CUDA graph mode**: captures all kernel launches (~210 kernels for Llama 1B) into a CUDA graph on the first token, then replays with a single `cuGraphLaunch` call. Achieves 53-56 tok/s for Llama-3.2-1B Q4_K_M.
- **GPU-resident forward pass** (`CudaForwardPass`): activations stay on GPU between transformer layers, reducing CPU-GPU sync points. RMSNorm, QKV projections, RoPE, KV cache, attention, softmax, FFN all run on GPU.
- **Optimized CUDA kernels**: Q3_K, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, F32 matmul plus RMSNorm, RoPE, attention, softmax, SiLU. Warp-per-row design with `__shfl_down_sync` reduction. Coalesced Q6_K kernel (3x improvement for output projection). Vectorized Q4_K loads.
- **SIMD-optimized CPU tensors**: `SimdQ4_KFloatTensor` and `SimdQ8_0FloatTensor` use Java Vector API for 2-3x dot product speedup over scalar.

### Qwen3.5 Hybrid DeltaNet Architecture

New inference engine (`Qwen35InferenceEngine`) for Qwen3.5's hybrid architecture that alternates Gated DeltaNet (linear attention/SSM) and standard GQA layers in a 3:1 ratio. DeltaNet layers use recurrent state with update rule `S_new = alpha*S + beta*outer(k, v - alpha*S^T@k)`. Full attention layers use packed Q+gate projections that are deinterleaved at runtime.

### LoRA Fine-Tuning Pipeline

Pure Java LoRA fine-tuning with checkpoint/resume. 6-stage pipeline: analyze target model → chunk input data → generate Q&A pairs via LLM → LoRA training with teacher forcing → merge adapters → export as new GGUF. Supports code, text, and structured data inputs.

### Performance Analysis

Detailed GPU profiling (`-Dcuda.profile=true`) with per-section breakdown. Llama-3.2-1B Q4_K_M at 31% of peak 192 GB/s bandwidth utilization. Analysis documented in PERFORMANCE-ANALYSIS.md.

---

## v1.3.0 (2026-02)

JAR rebuild for jvm8, jvm21, jvm25 with all v1.2.0 features. No code changes.

---

## v1.2.0 (2026-02)

### Chat UI with Persistence and Branching

Full-featured chat interface at `/chat` with server-side conversation persistence:
- Tree-based message branching: edit user messages or regenerate assistant responses to explore alternative conversation paths
- Branch navigation with arrow controls
- Per-conversation settings (temperature, max tokens, top-k, top-p, repetition penalty)
- Markdown rendering with syntax-highlighted code blocks and copy button
- Responsive layout with collapsible sidebar
- Chat API at `/api/chats/*` with full CRUD operations

### New Quantization Formats

Added support for 6 new quantization types: IQ2_S, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, and MXFP4. IQ formats use precomputed grid lookup tables for dequantization.

### Anthropic Messages API

New `/v1/messages` endpoint implementing the Anthropic Messages API for compatibility with Claude Code and other Anthropic API clients. Supports streaming and non-streaming modes.

### OpenAI API Enhancements

- **Tool calling**: `tools` array in request → `tool_calls` in response with `finish_reason: "tool_calls"`
- **JSON mode**: `response_format: {type: "json_object"}` injects system prompt for structured output
- **Embeddings endpoint**: `/v1/embeddings` returns L2-normalized vectors

### ConversationCache

KV cache reuse across multi-turn conversations — avoids re-processing the full conversation history on each turn.

### Enhanced Qwen3MoE Inference

Added shared expert support for Qwen3MoE architecture, improving output quality for models like Qwen3-Coder-30B-A3B.

---

## v1.1.0 (2026-01)

### OpenAI-Compatible API

Added `/v1/chat/completions` (streaming and non-streaming) and `/v1/models` endpoints following the OpenAI Chat Completions API specification. Works with standard OpenAI clients (Open WebUI, LangChain, LiteLLM, Cursor, Continue.dev).

- Multi-turn conversation support via `ChatTemplate.formatConversation()` for all architectures
- SSE streaming in OpenAI `chat.completion.chunk` format
- Migrated web UI chat to use the OpenAI endpoint

### Qwen3 Inference Fix

Fixed `ArrayIndexOutOfBoundsException` in Qwen3 models where `embeddingLength < headCount * headSize` — Q and xb2 buffers were undersized.

---

## v1.0.0 (2026-01)

### Initial Release

Pure Java LLM inference engine that runs GGUF models locally with zero external dependencies.

**Supported architectures**: Llama, Qwen2, Qwen3, Qwen3MoE, DeepSeek2, GLM4, Phi-3/4, Mistral3/Devstral.

**Quantization formats**: Q2_K, Q3_K, Q4_0, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0, BF16, F16, F32.

**Key features**:
- Memory-mapped GGUF parser with parallel tensor preload
- Multi-Head / Grouped-Query Attention with RoPE
- SwiGLU FFN with GeGLU variant for Gemma
- MLA (Multi-Head Latent Attention) for DeepSeek2
- MoE FFN with shared expert and top-K routing
- Token sampling: temperature, top-k, top-p, repetition penalty
- BPE and SentencePiece tokenizers with architecture-specific chat templates
- OpenCL GPU backend via Panama FFM (zero native deps)
- MoE-optimized GPU placement: attention on GPU, expert tensors on CPU — enables 30B models on 6 GB VRAM
- Swing desktop GUI, embedded web server with model config UI
- Response quality evaluation (perplexity, coherence, length metrics)

**Multi-JVM support**: compiles for Java 8 (scalar), Java 21 (Vector API SIMD + GPU), and Java 25 (structured concurrency + virtual threads). Classes loaded via `Class.forName()` reflection with graceful fallback.
