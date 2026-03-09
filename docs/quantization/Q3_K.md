# Q3_K -- 3-bit K-Quant

## Overview

| Property | Value |
|----------|-------|
| Full name | 3-bit K-Quantization |
| GGML type ID | 11 |
| Bits per weight | 3.4375 bpw |
| Block size | 256 elements |
| Block bytes | 110 bytes |
| Compression ratio | 9.3x vs F32 |

## Data Layout

Each 110-byte super-block encodes 256 weights:

```
Offset  Size   Field    Description
------  -----  -------  -------------------------------------------
0       32     hmask    High bit mask: 1 bit per weight (256 bits = 32 bytes)
32      64     qs       Low 2 bits of each weight (4 weights per byte)
96      12     scales   16 x 6-bit sub-block scales, packed
108     2      d        fp16 super-block scale
------  -----  -------  -------------------------------------------
Total:  110 bytes
```

The 3-bit quant value is split across two fields: the low 2 bits come from `qs` and the high 1 bit comes from `hmask`.

### Scale Packing

The 16 sub-block scales are 6 bits each (0-63), packed into 12 bytes:

- Bytes 0-7: low 4 bits of scales 0-7 (low nibble) and scales 8-15 (high nibble)
- Bytes 8-11: high 2 bits of scales 0-15, packed 4 per byte (2 bits each)

Decoding all 16 scales:
```
for i in 0..7:
    sc[i]     = raw[i] & 0x0F
    sc[i + 8] = raw[i] >> 4
for i in 0..3:
    v = raw[8 + i]
    sc[i]      |= (v & 0x03) << 4
    sc[i + 4]  |= ((v >> 2) & 0x03) << 4
    sc[i + 8]  |= ((v >> 4) & 0x03) << 4
    sc[i + 12] |= ((v >> 6) & 0x03) << 4
```

### Quant Bit Extraction

The `qs` and `hmask` arrays use a complex layout organized by halves (0-127, 128-255) and pairs within each half:

```
half      = j / 128           // 0 or 1
pair      = (j % 128) / 32    // 0-3
which16   = (j % 32) / 16     // 0 or 1
l         = j % 16            // position within 16-element group

qs_byte_idx = half * 32 + which16 * 16 + l
qs_shift    = pair * 2
low_bits    = (qs[qs_byte_idx] >> qs_shift) & 0x03

hm_byte_idx = which16 * 16 + l
hm_bit      = half * 4 + pair
high_bit    = (hmask[hm_byte_idx] >> hm_bit) & 1

q = (low_bits | (high_bit << 2)) - 4     // signed 3-bit: range -4 to +3
```

## Dequantization Formula

```
sub_block = j / 16
value = d * (scales[sub_block] - 32) * q
```

The scale offset of 32 centers the 6-bit scale range (0-63) around zero, giving an effective signed scale of -32 to +31.

## CUDA Kernel

| Property | Value |
|----------|-------|
| Kernel file | `matmul_q3_k.cu` |
| Strategy | Warp-per-row, efficient scale decode |
| Key optimization | Only the 2 needed scales are decoded per sub-block, not all 16 |
| Cache hints | `__ldg` for texture cache |
| Alignment | 110 bytes -- NOT 4-byte aligned, uses byte-level `__ldg` only |

## SIMD Optimization

| Property | Value |
|----------|-------|
| Fused SIMD class | `SimdQ3_KFloatTensor` (java21) |
| Available since | v1.6.0 |
| CPU dot path | Fused dequant+dot with SIMD vector operations |

## Performance Characteristics

Q3_K sits between Q2_K and Q4_K in the quality-size trade-off. At 3.4375 bpw, a 7B model uses roughly 3.0 GB. The complex bit packing (3-bit values split across two arrays) makes the dequantization logic more involved than Q4_K, which can affect throughput.

The CUDA kernel's efficient scale decode -- extracting only the 2 scales needed per sub-block rather than all 16 -- reduces register pressure and improves performance. However, the 110-byte block size is not 4-byte aligned, preventing vectorized `uint32` loads in the kernel.

## Typical Usage

- Llama-3.2-3B Q3_K_L (used in LLMPlayer benchmarks)
- Mid-range quantization for models where Q4_K is too large but Q2_K quality is unacceptable
- Common in Q3_K_S (small), Q3_K_M (medium), Q3_K_L (large) GGUF variants where different tensor groups use Q2_K, Q3_K, or Q4_K
