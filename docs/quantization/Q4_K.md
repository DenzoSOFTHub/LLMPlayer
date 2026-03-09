# Q4_K -- 4-bit K-Quant

## Overview

| Property | Value |
|----------|-------|
| Full name | 4-bit K-Quantization |
| GGML type ID | 12 |
| Bits per weight | 4.5 bpw |
| Block size | 256 elements |
| Block bytes | 144 bytes |
| Compression ratio | 7.1x vs F32 |

## Data Layout

Each 144-byte super-block encodes 256 weights:

```
Offset  Size   Field    Description
------  -----  -------  -------------------------------------------
0       2      d        fp16 super-block scale
2       2      dmin     fp16 super-block minimum
4       12     scales   8 sub-block scale+min values, 6 bits each, packed
16      128    qs       256 x 4-bit quantized values (2 per byte)
------  -----  -------  -------------------------------------------
Total:  144 bytes
```

The 256 weights are divided into 4 groups of 64, each group containing 2 sub-blocks of 32. Each sub-block has a 6-bit scale and a 6-bit minimum.

### Scale Packing (12 bytes -> 8 scales + 8 mins)

The 12 scale bytes encode 8 scale/min pairs at 6 bits each:

```
For sub-blocks 0-3 (first 4):
    scale[i] = scaleBytes[i] & 0x3F
    min[i]   = scaleBytes[i + 4] & 0x3F

For sub-blocks 4-7 (last 4):
    scale[i] = (scaleBytes[i + 4] & 0x0F) | ((scaleBytes[i - 4] >> 6) << 4)
    min[i]   = ((scaleBytes[i + 4] >> 4) & 0x0F) | ((scaleBytes[i] >> 6) << 4)
```

### Nibble Layout

Within `qs`, each byte packs two 4-bit values. For a group of 64 weights (group index `g`):
- Weights 0-31 (low sub-block): low nibble of `qs[g*32 + l]`
- Weights 32-63 (high sub-block): high nibble of `qs[g*32 + l]`

## Dequantization Formula

```
group = j / 64
is_high = (j % 64) >= 32
l = j % 32
sub_block = group * 2 + (1 if is_high else 0)

qs_byte = qs[group * 32 + l]
q = high_nibble(qs_byte) if is_high else low_nibble(qs_byte)

value = d * scale[sub_block] * q - dmin * min[sub_block]
```

## CUDA Kernels

Q4_K has the most CUDA kernel variants of any quantization type:

| Kernel file | Purpose | Notes |
|-------------|---------|-------|
| `matmul_q4_k.cu` | Default matmul | Vectorized `uint32` weight loads, `float4` input loads, `__restrict__` + `__ldg`, group-level striping |
| `matmul_q4_k_coalesced.cu` | Alternative coalesced | All 32 threads process same group with single byte reads. 8% slower -- opt-in via `-Dcuda.q4k.coalesced=true` |
| `matmul_q4_k_fused_gate_up.cu` | Fused FFN gate+up | Single kernel launch for both gate and up projections when both are Q4_K. Reads input once, halves kernel launches for FFN phase |

**Alignment:** 144 bytes IS 4-byte aligned, enabling safe vectorized `uint32` `__ldg` loads. This is a significant advantage over Q3_K (110B) and Q6_K (210B) which must use byte-level reads.

## SIMD Optimization

| Property | Value |
|----------|-------|
| Fused SIMD class | `SimdQ4_KFloatTensor` (java21) |
| Available since | v1.5.0 |
| CPU dot path | Fused dequant+dot with SIMD vector operations |

## Performance Characteristics

Q4_K is the best-tested and best-optimized quantization type in LLMPlayer. Key performance numbers (NVIDIA RTX 4050 Laptop, 6 GB VRAM):

- **Llama-3.2-1B Q4_K_M**: 53-56 tok/s (CUDA graph mode)
- FFN phase (gate+up+SiLU+down): ~9.0 ms/tok for Q4_K layers
- The fused gate+up kernel saves one kernel launch per layer for models using Q4_K on both gate and up projections

The 144-byte block size is 4-byte aligned, allowing the CUDA kernel to use `uint32` vectorized loads. Combined with `float4` input loads and `__restrict__` hints, this makes Q4_K the most bandwidth-efficient quantized kernel.

The coalesced variant was tested and found 8% slower because Q4_K's 4-byte alignment already allows vectorized loads in the default kernel -- coalescing with smaller single-byte reads loses instruction-level parallelism without improving bandwidth utilization.

## Typical Usage

- Most Q4_K_M models: Llama, Qwen2, Qwen3, Phi-3, Phi-4, Mistral, GLM4, Gemma
- The dominant quantization type in the GGUF ecosystem
- Best balance of quality, speed, and model size
- A 7B parameter model in Q4_K uses roughly 4.0 GB
