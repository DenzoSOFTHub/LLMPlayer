# IQ4_XS -- Importance-Weighted 4-bit Extra-Small

## Overview

| Property | Value |
|----------|-------|
| Full name | Importance-Weighted 4-bit Extra-Small |
| GGML type ID | 23 |
| Bits per weight | 4.25 bpw |
| Block size | 256 elements |
| Block bytes | 136 bytes |
| Compression ratio | 7.5x vs F32 |

## Data Layout

Each 136-byte super-block encodes 256 weights:

```
Offset  Size   Field      Description
------  -----  ---------  -------------------------------------------
0       2      d          fp16 super-block scale
2       2      scales_h   uint16: high 2 bits of 8 sub-block scales
4       4      scales_l   Low 4 bits of 8 sub-block scales (2 per byte)
8       128    qs         256 x 4-bit nibbles (non-linear lookup)
------  -----  ---------  -------------------------------------------
Total:  136 bytes
```

The 256 weights are divided into 8 sub-blocks of 32 weights each, each with a 6-bit scale.

### 6-bit Scale Reconstruction

Each sub-block has a 6-bit scale, split across `scales_l` and `scales_h`:

```
scales_l_byte = scales_l[ib / 2]
low4 = low_nibble(scales_l_byte) if ib is even else high_nibble(scales_l_byte)
high2 = (scales_h >> (2 * ib)) & 3

ls = low4 | (high2 << 4)          // 6-bit value: 0-63
effective_scale = d * (ls - 32)    // centered around 32
```

### Nibble Layout

Within `qs`, the layout is interleaved (2 weights per byte):

```
For weight at position j within sub-block ib:
    in_sub = j % 32
    byte_offset = ib * 16 + in_sub / 2
    nibble = low nibble if in_sub is even, high nibble if odd
```

## Dequantization Formula

IQ4_XS uses the same non-linear lookup table as IQ4_NL:

```
dl = d * (ls - 32)
value = dl * KVALUES_IQ4NL[nibble]

where KVALUES_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
     1,  13,  25,  38,  53,  69,  89, 113
}
```

The non-linear mapping (derived from K-means clustering) provides better reconstruction than a uniform `(nibble - 8)` mapping, especially for weight distributions that are not uniform.

## CUDA Kernel

| Property | Value |
|----------|-------|
| Kernel file | `matmul_iq4_xs.cu` |
| Strategy | Warp-per-row |
| Available since | v1.5.1 |

## SIMD Optimization

| Property | Value |
|----------|-------|
| Fused SIMD class | None |
| CPU dot path | Dequantize via lookup table to buffer, then SIMD dot |

## Performance Characteristics

IQ4_XS provides a slight size advantage over Q4_K (4.25 vs 4.5 bpw) by using 256-element super-blocks with compact 6-bit scale encoding. The non-linear lookup table (shared with IQ4_NL) provides better reconstruction quality than linear dequantization at the same bit width.

Compared to Q4_K:
- **Smaller**: 136 bytes per 256 weights vs 144 bytes (5.6% saving)
- **Non-linear**: Lookup table adapts to typical weight distributions
- **Simpler**: No per-sub-block minimum (just scale), reducing metadata overhead

Compared to IQ4_NL:
- **Super-blocks**: 256-element blocks with per-sub-block scales (vs IQ4_NL's 32-element blocks with single scale)
- **More efficient**: Less scale overhead per weight

A 7B model in IQ4_XS uses approximately 3.7 GB.

## Typical Usage

- Gemma-2-2B IQ4_XS (used in LLMPlayer testing)
- Space-efficient alternative to Q4_K when the non-linear lookup table matches the model's weight distribution
- Models where the 5.6% size reduction vs Q4_K matters for fitting in VRAM
