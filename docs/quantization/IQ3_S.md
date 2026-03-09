# IQ3_S -- Importance-Weighted 3-bit Small

## Overview

| Property | Value |
|----------|-------|
| Full name | Importance-Weighted 3-bit (Small) |
| GGML type ID | 21 |
| Bits per weight | 3.4375 bpw |
| Block size | 256 elements |
| Block bytes | 110 bytes |
| Compression ratio | 9.3x vs F32 |

## Data Layout

Each 110-byte super-block encodes 256 weights:

```
Offset  Size   Field    Description
------  -----  -------  -------------------------------------------
0       2      d        fp16 super-block scale
2       64     qs       Grid indices, low 8 bits (8 per group of 32)
66      8      qh       Grid indices, high 1 bit (9th bit)
74      32     signs    Sign bits, 1 per weight (4 bytes per group of 32)
106     4      scales   4-bit sub-block scales (2 per byte, 1 per pair of 32-weight groups)
------  -----  -------  -------------------------------------------
Total:  110 bytes
```

The 256 weights are organized into 8 groups of 32, processed in pairs (groups 0+1, 2+3, 4+5, 6+7). Each pair shares a scale byte (low nibble for even group, high nibble for odd group).

## Grid-Based Dequantization

IQ3_S uses the `IQ3S_GRID` lookup table (512 entries, each a `uint32` encoding 4 unsigned byte values). A 9-bit grid index selects the entry. Sign bits are stored explicitly, one per weight.

### 9-bit Grid Index Construction

```
qs_low = qs[group * 8 + 2 * l + grid_quad]       // 8 low bits
qh_byte = qh[pair_index]                          // high bits for pair

if grid_quad == 0:
    high_bit = (qh_byte << (8 - 2*l)) & 256      // bit 8
else:
    high_bit = (qh_byte << (7 - 2*l)) & 256      // bit 8

grid_idx = qs_low | high_bit                       // 9-bit index (0-511)
```

Where `l` is the sub-group index (0-3) and `grid_quad` selects first/second half (0 or 1) of the sub-group.

## Dequantization Formula

```
// Scale (per pair of 32-weight groups)
scale_byte = scales[pair_index]
if even group:
    db = d * (1 + 2 * (scale_byte & 0x0F))
if odd group:
    db = d * (1 + 2 * ((scale_byte >> 4) & 0x0F))

// Grid lookup
grid_val = IQ3S_GRID[grid_idx]              // uint32: 4 unsigned bytes
gv = (grid_val >> (8 * byte_in_grid)) & 0xFF

// Sign
sign_byte = signs[group * 4 + l]
sign = -1.0 if (sign_byte & KMASK_IQ2XS[j_in_8]) else +1.0

value = db * gv * sign
```

The scale formula `(1 + 2 * nibble)` gives odd values 1, 3, 5, ..., 31 -- a different range than IQ2_S or IQ3_XXS.

## CUDA Kernel

| Property | Value |
|----------|-------|
| Kernel file | `matmul_iq3_s.cu` |
| Strategy | Warp-per-row |
| Lookup table | `__constant__` IQ3S_GRID (512 x uint32) |
| Available since | v1.6.0 |

## SIMD Optimization

| Property | Value |
|----------|-------|
| Fused SIMD class | None |
| CPU dot path | Dequantize via grid lookup to buffer, then SIMD dot |

## Performance Characteristics

IQ3_S provides higher quality than IQ3_XXS at the same 3.4375 bpw (same block byte count as Q3_K, coincidentally). The key differences from IQ3_XXS:

- Larger grid: 512 entries (9-bit) vs 256 entries (8-bit), providing finer granularity
- Explicit per-weight sign bits vs packed 7-bit sign indices
- Per-pair scales vs per-group scales

The explicit sign storage uses 32 bytes (vs IQ3_XXS's packed signs in 32 bytes of `scales_and_signs`), but the larger grid and simpler sign handling generally produce better reconstruction quality.

Compared to Q3_K at the same bpw, IQ3_S uses grid-based quantization which can better represent clustered weight distributions. Q3_K uses per-sub-block scales which can better handle varying weight magnitudes across the block.

A 7B model in IQ3_S uses approximately 3.0 GB.

## Typical Usage

- Mid-quality importance-weighted quantizations
- When IQ3_XXS quality is insufficient but Q4_K memory is too high
- Some GGUF files use IQ3_S as a compromise between quality and size
