# Llama-3.2-1B-Instruct Q4_K_M

## Model Summary

| Field | Value |
|-------|-------|
| File | `Llama-3.2-1B-Instruct-Q4_K_M.gguf` |
| Size | 771 MB |
| Architecture | LLAMA |
| Parameters | 1B |
| Quantization | Q4_K_M (4.5 bpw) |
| Layers | 16 |
| Heads | 32 Q / 8 KV (GQA 4:1) |
| Embedding dim | 2048 |
| Head size | 64 |
| FFN dim | 8192 |

## Architecture Description

### Forward Pass

Standard pre-norm transformer (InferenceEngine):

```
RMSNorm -> Attention -> Residual -> RMSNorm -> SwiGLU FFN -> Residual
```

### Attention

- Grouped Query Attention (GQA): 32 query heads, 8 key-value heads (4:1 ratio)
- RoPE positional encoding: type NORMAL (consecutive pairs), base frequency 500000
- No QK-norm
- No bias terms
- No sliding window (full context attention on all layers)

### FFN

- SwiGLU activation: `SiLU(gate * x) * up`
- FFN intermediate dimension: 8192

### Normalization

- Pre-norm only (RMSNorm before attention and before FFN)
- No post-attention or post-FFN norms

### Tokenizer

- BPE (GPT-2 byte-level encoding)
- Llama 3 pre-tokenization regex

### Chat Template

```
<|start_header_id|>user<|end_header_id|>

Prompt<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

## Quantization Details

Q4_K_M uses a super-block structure:

- 256 elements per super-block
- 144 bytes per block (4-byte aligned, safe for `uint32` vectorized GPU loads)
- 4.5 bits per weight average
- K-quant grouping with per-group scales and minimums

## GPU Execution Profile

**Hardware:** NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM, 192 GB/s peak bandwidth)

| Metric | Value |
|--------|-------|
| Mode | CUDA graph (16/16 layers) |
| VRAM | 771 MB (full offload) |
| v1.5.1 | 54.7 tok/s |
| v1.6.0 | 52.7 tok/s |
| Features | `cuda_graph`, `fused_gateup` |

### Per-Section Breakdown (Steady State)

| Section | Time (ms/tok) |
|---------|---------------|
| Output projection (Q6_K) | 2.7 |
| FFN Q4_K (GateUp + SiLU + Down) | 9.0 |
| Attention + norms | 5.1 |
| **Total GPU compute** | **~15.5** |
| **Wall time** | **~21** |

The ~5.5 ms gap between GPU compute and wall time is Panama FFM per-launch overhead.

### Bandwidth Utilization

31% of the 192 GB/s peak memory bandwidth. The primary bottleneck is per-kernel launch overhead rather than memory throughput.

### CUDA Kernel Details

- Q4_K matmul: warp-per-row with `__restrict__` + `__ldg` read-only cache, `uint32` vectorized weight loads, `float4` input loads, group-level striping
- Q6_K matmul: coalesced position-parallel kernel (all 32 warp threads process the same half-block with consecutive byte addresses)
- Fused gate+up kernel: single launch for both gate and up projections when both are Q4_K, reads input vector once

## CPU Execution Profile

**Hardware:** Intel Core Ultra 7 155H

| Metric | Value |
|--------|-------|
| v1.6.0 | 2.5-2.6 tok/s |
| SIMD tensors | Q4_K, Q8_0, Q6_K, Q5_0, Q5_K, Q3_K |
| Prefill skip | Active (no output projection except last token) |
| RoPE mode | NORMAL (SIMD NEOX not applicable) |

All quantized tensor types use fused dequant+dot SIMD implementations via the Vector API (`jdk.incubator.vector`).

## Implementation Notes

- This is the reference benchmark model for LLMPlayer performance testing
- All 16 layers fit in GPU VRAM with room to spare
- CUDA graph captures all kernel launches on the first token and replays via `cuGraphLaunch` on subsequent tokens
- The fused gate+up kernel is auto-detected per-layer when both tensors are Q4_K
- GPU-side argmax available for greedy sampling (downloads 4 bytes instead of full logit vector)

## Known Issues

None. This model serves as the primary regression benchmark.

## Version History

| Version | tok/s (GPU) | Notes |
|---------|-------------|-------|
| v1.4.0 | 53-56 | Initial CUDA support with graph mode |
| v1.5.1 | 54.7 | Stable; fused gate+up kernel added |
| v1.6.0 | 52.7 | Within noise, no regression |
