# Q5_K -- 5-bit K-Quant

## Overview

| Property | Value |
|----------|-------|
| Full name | 5-bit K-Quantization |
| GGML type ID | 13 |
| Bits per weight | 5.5 bpw |
| Block size | 256 elements |
| Block bytes | 176 bytes |
| Compression ratio | 5.8x vs F32 |

## Data Layout

Each 176-byte super-block encodes 256 weights:

```
Offset  Size   Field    Description
------  -----  -------  -------------------------------------------
0       2      d        fp16 super-block scale
2       2      dmin     fp16 super-block minimum
4       12     scales   8 sub-block scale+min values, 6 bits each, packed
16      32     qh       256 high bits (1 per weight, 8 per byte)
48      128    qs       256 x 4-bit low quants (2 per byte)
------  -----  -------  -------------------------------------------
Total:  176 bytes
```

Q5_K extends Q4_K by adding a 5th bit per weight. The layout is identical to Q4_K except for the additional 32-byte `qh` array that provides the high bit for each weight. The scale packing scheme is shared with Q4_K.

### 5-bit Value Reconstruction

For each weight, the 5-bit value is composed from two sources:
- **Low 4 bits** from `qs` (same nibble layout as Q4_K)
- **High 1 bit** from `qh` (bit-packed, 8 bits per byte)

```
group = j / 64
is_high = (j % 64) >= 32
l = j % 32

qs_nibble = high_nibble(qs[group*32+l]) if is_high else low_nibble(qs[group*32+l])

bit_pos = group * 2 + (1 if is_high else 0)
qh_bit = (qh[l] >> bit_pos) & 1

q5 = qs_nibble | (qh_bit << 4)    // 5-bit value: 0-31
```

## Dequantization Formula

```
value = d * scale[sub_block] * q5 - dmin * min[sub_block]
```

Same structure as Q4_K, but with the wider 5-bit quantization grid (32 levels instead of 16).

## CUDA Kernel

| Property | Value |
|----------|-------|
| Kernel file | `matmul_q5_k.cu` |
| Strategy | Warp-per-row |
| Cache hints | `__ldg` for all reads through texture cache |
| Alignment | 176 bytes IS 4-byte aligned -- safe for vectorized loads |

## SIMD Optimization

| Property | Value |
|----------|-------|
| Fused SIMD class | `SimdQ5_KFloatTensor` (java21) |
| Available since | v1.6.0 |
| CPU dot path | Fused dequant+dot with SIMD vector operations |

## Performance Characteristics

Q5_K provides noticeably better quality than Q4_K at the cost of 22% more memory (176 vs 144 bytes per block). The extra 32-byte `qh` array adds one additional memory read per block during dequantization.

The 176-byte block is 4-byte aligned (176 = 44 * 4), allowing the CUDA kernel to use vectorized loads similar to Q4_K. However, the additional `qh` reads and bit manipulation make Q5_K roughly 10-15% slower than Q4_K in practice.

A 7B model in Q5_K uses approximately 4.8 GB.

## Typical Usage

- Higher quality variants of dense models (Q5_K_S, Q5_K_M)
- When Q4_K quality is insufficient but Q6_K or Q8_0 memory cost is too high
- Common in mixed-quantization GGUF files where important tensors (attention Q/K) use Q5_K while less sensitive tensors use Q4_K
