# sonar-oss-20b-MXFP4Q8_MOE

## Model Info

| Field | Value |
|-------|-------|
| File | `sonar-oss-20b-MXFP4Q8_MOE.gguf` |
| Size | 11549 MB |
| Parameters | 20B |
| Architecture | GPT_OSS |
| Quantization | MXFP4 (microscaling FP4, 4-bit floating point with shared exponent) |
| Tokenizer | SentencePiece |

## Architecture

**Inference Engine:** `Qwen3MoEInferenceEngine` (MoE variant)

| Property | Value |
|----------|-------|
| Normalization | Pre-norm (RMSNorm) |
| Attention | GQA with sliding window |
| Sliding Window | Alternating: even layers = global (full attention), odd layers = local (128-token window) |
| Attention Sinks | Per-head learned biases |
| FFN | MoE with shared expert |

**Chat Template:** Multi-channel format (architecture-specific).

**Architecture notes:**
- GPT_OSS uses an alternating sliding window pattern distinct from other architectures: even-numbered layers have global (full) attention, odd-numbered layers have local attention with a 128-token window. This differs from Gemma 2 (even=local, odd=global) and Gemma 3 (every 6th layer global).
- Attention sinks are implemented as per-head learned biases that maintain attention to early tokens, improving long-context coherence.
- MXFP4 (microscaling FP4) is a 4-bit floating-point format with a shared exponent per block, providing better dynamic range than integer-based 4-bit quantization (Q4_K) at the same bit width.

## GPU Profile

- **VRAM fit:** MoE-optimized placement supported.
- **GPU strategy:** MoE-optimized -- attention layers on GPU, expert tensors on CPU.
- **CudaForwardPass:** NOT supported (MoE architecture).
- **Benchmark data:** None available.

## CPU Profile

No benchmark data available.

## Known Issues

- MXFP4 is a relatively uncommon quantization format. Ensure the MXFP4 tensor implementation is available (`GGMLType.MXFP4` in `TensorFactory`).
- The alternating sliding window pattern (even=global, odd=local) must not be confused with Gemma 2's pattern (even=local, odd=global) or Gemma 3's pattern (every 6th layer global).

## Version History

| Version | Notes |
|---------|-------|
| v1.5.0 | GPT_OSS architecture support with MXFP4 quantization |
