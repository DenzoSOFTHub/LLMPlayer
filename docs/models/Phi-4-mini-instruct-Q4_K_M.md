# Phi-4 Mini Instruct Q4_K_M

## Model Info

| Field | Value |
|-------|-------|
| File | `Phi-4-mini-instruct-Q4_K_M.gguf` |
| Size | 2377 MB |
| Parameters | 3.8B |
| Quantization | Q4_K_M |
| Tokenizer | BPE |
| Chat Template | `<\|user\|>\nPrompt<\|end\|>\n<\|assistant\|>\n` |

## Architecture

| Parameter | Value |
|-----------|-------|
| Architecture | PHI3 |
| Layers | 32 |
| Attention Heads | 32 |
| KV Heads | 8 |
| Embedding Dim | 3072 |
| Head Size | 96 |
| FFN Dim | 8192 |
| GQA Ratio | 4:1 (32 Q heads : 8 KV heads) |
| RoPE | NEOX, base=10000 |
| Norm | Pre-norm (RMSNorm) |
| FFN | SwiGLU |

**Inference Engine:** `InferenceEngine` (standard).

**Packed weight layout:** Phi-3/4 models use merged weight matrices that differ from standard separate Q/K/V and gate/up projections:

- **Merged QKV:** A single `wqkv` weight produces concatenated Q, K, V outputs. The CUDA forward pass splits these via the `split_qkv` kernel.
- **Packed FFN:** `wGate` is null. `wUp` outputs a `[2 * ffnDim]` vector containing interleaved gate and up projections. Split via `split_gate_up.cu` kernel on GPU.

## GPU Profile (RTX 4050 Laptop, 6140 MB VRAM)

| Version | tok/s | Mode | Notes |
|---------|-------|------|-------|
| v1.5.1 | 11.5 | Per-tensor CUDA | Packed FFN blocked CUDA graph |
| v1.6.0 | 15.2 | **CUDA graph** | Packed FFN now supported via `split_gate_up.cu` |

- **v1.5.1 to v1.6.0: +32% improvement** from enabling CUDA graph mode for packed FFN architectures
- **VRAM usage:** 2377 MB (full offload, all 32 layers on GPU)
- **Features:** CUDA graph mode (v1.6.0+), `split_qkv` kernel, `split_gate_up.cu` kernel
- **Test config:** `--prompt "..." --max-tokens 200 --context-length 256 --gpu --gpu-backend cuda`

## CPU Profile

| Version | tok/s | Notes |
|---------|-------|-------|
| v1.6.0 | 0.6-1.0 | SIMD tensors active |

- **Hardware:** Intel Core Ultra 7 155H

## Known Issues

None. The packed FFN limitation that prevented CUDA graph mode in v1.5.1 was resolved in v1.6.0.

## Version History

| Version | Change | tok/s |
|---------|--------|-------|
| v1.5.1 | Per-tensor CUDA only (packed FFN not in CudaForwardPass) | 11.5 |
| v1.6.0 | CUDA graph with packed FFN support (`split_gate_up.cu`) | 15.2 (+32%) |
