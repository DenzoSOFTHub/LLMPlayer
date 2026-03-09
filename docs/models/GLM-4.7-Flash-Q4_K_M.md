# GLM-4.7-Flash-Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `GLM-4.7-Flash-Q4_K_M.gguf` |
| Size | 17465 MB |
| Parameters | 16B |
| Architecture | DEEPSEEK2 (uses `deepseek2` GGUF architecture despite "GLM" branding) |
| Quantization | Q4_K_M |
| Tokenizer | SentencePiece |
| Type | MoE |

## Architecture

**Inference Engine:** `DeepSeek2InferenceEngine`

| Property | Value |
|----------|-------|
| Normalization | Pre-norm (RMSNorm) |
| Attention | Multi-Head Latent Attention (MLA) |
| FFN | Hybrid dense (leading blocks) + MoE (remaining blocks) |

Despite its "GLM" name, this model uses the `deepseek2` GGUF architecture identifier and follows the same MLA + MoE pattern as DeepSeek-Coder-V2. It is processed by `DeepSeek2InferenceEngine`, not the standard `InferenceEngine` used by GLM4.

**Chat Template:** `[gMASK]<sop><|user|>\n{prompt}<|assistant|>\n`

**Key distinction from GLM-4-32B:**
- GLM-4-32B uses architecture `GLM4` with standard GQA attention and dense FFN.
- GLM-4.7-Flash uses architecture `DEEPSEEK2` with MLA attention and MoE FFN. These are fundamentally different inference paths.

## GPU Profile

- **VRAM fit:** MoE-optimized placement supported.
- **GPU strategy:** MoE-optimized -- all attention layers on GPU, expert tensors on CPU.
- **CudaForwardPass:** NOT supported (MLA architecture).
- **Benchmark data:** None available.

## CPU Profile

No benchmark data available.

## Known Issues

- The architecture mismatch between model name ("GLM") and GGUF architecture ("deepseek2") can cause confusion. The GGUF `general.architecture` field is the authoritative source for inference engine selection.
- At 17.5 GB, this model requires substantial RAM. On 32 GB systems, context length should be limited.
- MLA + MoE combination means neither CudaForwardPass nor standard GQA attention paths apply.

## Version History

| Version | Notes |
|---------|-------|
| v1.4.0 | Supported via DeepSeek2 architecture path |
| v1.5.0 | MoE-optimized GPU placement |
