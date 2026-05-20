# Q2_K -- 2-bit K-Quant

## Overview

| Property | Value |
|----------|-------|
| Full name | 2-bit K-Quantization |
| GGML type ID | 10 |
| Bits per weight | 2.625 bpw |
| Block size | 256 elements |
| Block bytes | 84 bytes |
| Compression ratio | 12.2x vs F32 |

## Data Layout

Each 84-byte super-block encodes 256 weights:

```
Offset  Size   Field    Description
------  -----  -------  -------------------------------------------
0       16     scales   16 x uint8, each byte packs two 4-bit values:
                          low nibble  = scale (sc, 0-15)
                          high nibble = minimum (m, 0-15)
16      64     qs       256 x 2-bit quantized values (4 per byte)
80      2      d        fp16 super-block scale
82      2      dmin     fp16 super-block minimum
------  -----  -------  -------------------------------------------
Total:  84 bytes
```

The 256 weights are divided into 16 sub-blocks of 16 weights each. Each sub-block has its own 4-bit scale and 4-bit minimum, stored in the corresponding `scales` byte. The sub-block index is `j / 16`.

## Dequantization Formula

For weight at position `j` within the super-block:

```
sub_block = j / 16
sc = scales[sub_block] & 0x0F        // 4-bit scale
m  = scales[sub_block] >> 4          // 4-bit minimum

qs_byte = qs[j / 4]
q = (qs_byte >> (2 * (j % 4))) & 0x03   // 2-bit quant value (0-3)

value = d * sc * q - dmin * m
```

The dot product implementation factors this into two sums per sub-block for efficiency:

```
result += d * sc * sum(q[i] * input[i]) - dmin * m * sum(input[i])
```

## CUDA Kernel

| Property | Value |
|----------|-------|
| Kernel file | **None** — no `matmul_q2_k.cu` exists on disk |
| Tensor wrapper | **None** — no `Q2_KCudaTensor` class |
| Status | CPU-only at both the kernel and the tensor layer |

As of v1.13.0, Q2_K is the only supported quantization without GPU acceleration. Every other quant in the supported set (17 of 18: F32, F16, BF16, Q3_K, Q4_0, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0, IQ2_S, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, MXFP4) has both a `matmul_*.cu` kernel and a `*CudaTensor` wrapper. MXFP4 was the previous gap and got both pieces wired in v1.13.0; Q2_K still lacks both, so any Q2_K model runs on the CPU SIMD path. Block size is 84 bytes (not 4-byte aligned), which would force byte-level `__ldg` if a GPU kernel were ever written.

## SIMD Optimization

No fused dequant+dot SIMD variant. The CPU path dequantizes into a temporary float buffer, then uses `VectorOpsFactory.get().dot()` for the SIMD dot product. Q2_K was not included in the v1.12.0 B2I/I2F lane-parallel sweep (the rewrite covered Q3_K, Q5_0, Q5_K, Q6_K, and Q8_0).

| Property | Value |
|----------|-------|
| Fused SIMD class | None |
| CPU dot path | Dequantize to buffer, then SIMD dot |

## Performance Characteristics

Q2_K provides the most aggressive compression among K-quants at 2.625 bits per weight. The 2-bit quantization grid is very coarse (only 4 levels per sub-block), so quality degrades significantly compared to higher-bit K-quants. The small block size relative to stored metadata (16 bytes of scales for 64 bytes of quant data) means the overhead ratio is higher than Q4_K or Q6_K.

Memory bandwidth is minimal -- a 7B parameter model in Q2_K uses roughly 2.3 GB. However, the quality loss typically makes Q3_K or Q4_K preferable unless memory is extremely constrained.

## Typical Usage

- Aggressive quantizations of larger models (30B+) where memory is the primary constraint
- Experimentation and testing where quality is secondary
- Not recommended for production use due to significant quality degradation
