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
| Kernel file | `matmul_q2_k.cu` |
| Strategy | Warp-per-row, `__ldg` texture cache reads |
| Alignment | 84 bytes -- not 4-byte aligned |

## SIMD Optimization

No fused dequant+dot SIMD variant. The CPU path dequantizes into a temporary float buffer, then uses `VectorOpsFactory.get().dot()` for the SIMD dot product.

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
