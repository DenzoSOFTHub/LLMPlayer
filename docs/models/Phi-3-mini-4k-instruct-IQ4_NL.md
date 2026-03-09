# Phi-3 Mini 4K Instruct IQ4_NL

## Model Info

| Field | Value |
|-------|-------|
| File | `Phi-3-mini-4k-instruct.IQ4_NL.gguf` |
| Size | 2076 MB |
| Parameters | 3.8B |
| Quantization | IQ4_NL |
| Tokenizer | BPE |
| Chat Template | `<\|user\|>\nPrompt<\|end\|>\n<\|assistant\|>\n` |

## Architecture

| Parameter | Value |
|-----------|-------|
| Architecture | PHI3 |
| Layers | 32 |
| Attention Heads | 32 |
| KV Heads | 32 (MHA, not GQA) |
| Embedding Dim | 3072 |
| Head Size | 96 |
| FFN Dim | 8192 |
| GQA Ratio | 1:1 (multi-head attention) |
| RoPE | NEOX, base=10000 |
| Norm | Pre-norm (RMSNorm) |
| FFN | SwiGLU |

**Inference Engine:** `InferenceEngine` (standard). Same Phi-3 architecture as Phi-4 Mini with merged QKV and packed FFN weight layout. Key difference: Phi-3 uses full MHA (32 KV heads) whereas Phi-4 uses GQA (8 KV heads).

**Packed weight layout:** Same as Phi-4 -- merged `wqkv` weight and packed `wUp` with null `wGate`. See Phi-4 documentation for details.

## GPU Profile (RTX 4050 Laptop, 6140 MB VRAM)

| Version | tok/s | Mode | Notes |
|---------|-------|------|-------|
| v1.5.1 | 6.5 | Per-tensor CUDA | Packed FFN blocked CUDA graph |
| v1.6.0 | 8.8-8.9 | **CUDA graph** | Packed FFN now supported |

- **v1.5.1 to v1.6.0: +37% improvement** from enabling CUDA graph mode
- **VRAM usage:** 2076 MB (full offload, all 32 layers on GPU)
- **Features:** CUDA graph mode (v1.6.0+), `split_qkv` kernel, `split_gate_up.cu` kernel
- **Test config:** `--prompt "..." --max-tokens 200 --context-length 256 --gpu --gpu-backend cuda`

## CPU Profile

| Version | tok/s | Notes |
|---------|-------|-------|
| v1.6.0 | 0.8 | SIMD tensors active |

- **Hardware:** Intel Core Ultra 7 155H

## Known Issues

None.

## Version History

| Version | Change | tok/s |
|---------|--------|-------|
| v1.5.1 | Per-tensor CUDA only (packed FFN limitation) | 6.5 |
| v1.6.0 | CUDA graph with packed FFN support | 8.8-8.9 (+37%) |
