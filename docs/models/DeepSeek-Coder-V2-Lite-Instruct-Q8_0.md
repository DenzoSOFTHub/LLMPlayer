# DeepSeek-Coder-V2-Lite-Instruct-Q8_0

## Model Info

| Field | Value |
|-------|-------|
| File | `DeepSeek-Coder-V2-Lite-Instruct-Q8_0.gguf` |
| Size | 15929 MB |
| Parameters | 16B total (2.4B active) |
| Architecture | DEEPSEEK2 |
| Quantization | Q8_0 (8 bpw, 32 elements/block, 34 bytes/block) |
| Tokenizer | SentencePiece |

## Architecture

**Inference Engine:** `DeepSeek2InferenceEngine`

Same architecture as the Q4_K_M variant. See [DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M](DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.md) for full architecture details.

| Property | Value |
|----------|-------|
| Layers | 27 |
| Attention | Multi-Head Latent Attention (MLA) with KV LoRA rank 512 |
| FFN (leading blocks) | Dense SwiGLU |
| FFN (remaining blocks) | MoE: 64 experts, top-6 + 2 shared |

**Chat Template:** `User: {prompt}\n\nAssistant:`

## GPU Profile

- **VRAM fit:** MoE-optimized placement supported.
- **GPU strategy:** MoE-optimized -- all 27/27 attention layers on GPU, expert tensors remain on CPU.
- **VRAM used:** Higher than Q4_K_M variant (~517 MB) due to Q8_0 attention tensors being ~2x larger.
- **CudaForwardPass:** NOT supported (MLA architecture).
- **Benchmark data:** None available.

## CPU Profile

No benchmark data available.

## Known Issues

- Q8_0 block size (34 bytes) is not divisible by 4, so CUDA kernels use byte-level `__ldg` loads rather than vectorized uint32 loads.
- At 15.9 GB, this variant requires more RAM than Q4_K_M (9.9 GB). On 32 GB systems, context length may need to be limited to avoid memory pressure.
- Q8_0 provides higher quality than Q4_K_M with minimal perceptible quantization loss, at the cost of ~60% more memory.

## Version History

| Version | Notes |
|---------|-------|
| v1.4.0 | DeepSeek2 architecture support |
| v1.5.0 | MoE-optimized GPU placement strategy |
