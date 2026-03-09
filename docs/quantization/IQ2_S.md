# IQ2_S -- Importance-Weighted 2-bit

## Overview

| Property | Value |
|----------|-------|
| Full name | Importance-Weighted 2-bit (Small) |
| GGML type ID | 22 |
| Bits per weight | 2.5625 bpw |
| Block size | 256 elements |
| Block bytes | 82 bytes |
| Compression ratio | 12.5x vs F32 |

## Data Layout

Each 82-byte super-block encodes 256 weights:

```
Offset  Size   Field    Description
------  -----  -------  -------------------------------------------
0       2      d        fp16 super-block scale
2       32     qs       Grid index low 8 bits (4 bytes per group of 32)
34      32     signs    Sign bits, 1 per weight (4 bytes per group of 32)
66      8      qh       Grid index high 2 bits (1 byte per group of 32)
74      8      scales   4-bit sub-block scales (2 per byte, 1 per group of 32)
------  -----  -------  -------------------------------------------
Total:  82 bytes
```

The 256 weights are organized into 8 groups of 32, each group further divided into 4 sub-groups of 8.

## Grid-Based Dequantization

IQ2_S uses a precomputed lookup table (`IQ2S_GRID`) with 1024 entries, each a `uint64` encoding 8 unsigned byte values. A 10-bit grid index selects the entry, and explicit sign bits determine the polarity of each weight.

### 10-bit Grid Index Construction

```
qs_val  = qs[ib32 * 4 + l]                  // 8 low bits (per sub-group of 8)
qh_val  = qh[ib32]                          // shared across group of 32
grid_idx = qs_val | ((qh_val << (8 - 2*l)) & 0x300)   // 10-bit index
```

Where `ib32` is the group index (0-7) and `l` is the sub-group index (0-3).

## Dequantization Formula

```
// Scale computation
scale_byte = scales[ib32]
if l < 2:
    dl = d * (0.5 + (scale_byte & 0x0F)) * 0.25
else:
    dl = d * (0.5 + ((scale_byte >> 4) & 0x0F)) * 0.25

// Grid lookup
grid = IQ2S_GRID[grid_idx]                  // uint64: 8 unsigned bytes
grid_byte = (grid >> (8 * j_in_8)) & 0xFF   // value for position j within sub-group

// Sign
sign_byte = signs[ib32 * 4 + l]
sign = -1.0 if (sign_byte & KMASK_IQ2XS[j_in_8]) else +1.0

value = dl * grid_byte * sign
```

The `KMASK_IQ2XS` array provides bit masks for extracting individual sign bits from the packed sign byte.

## CUDA Kernel

| Property | Value |
|----------|-------|
| Kernel file | `matmul_iq2_s.cu` |
| Strategy | Warp-per-row |
| Grid table | `__device__ __constant__` IQ2S_GRID (1024 x uint64) |
| Available since | v1.6.0 |

The grid lookup table is stored in CUDA constant memory for fast cached access across all threads.

## SIMD Optimization

| Property | Value |
|----------|-------|
| Fused SIMD class | None |
| CPU dot path | Dequantize to buffer via grid lookup, then SIMD dot |

## Performance Characteristics

IQ2_S achieves extreme compression (2.5625 bpw) through importance-weighted grid quantization. The grid-based approach maps groups of 8 weights to one of 1024 precomputed patterns, which is more expressive than simple scalar quantization at the same bit rate.

The dequantization is relatively expensive due to:
1. Grid lookup (random access into 1024-entry table)
2. Sign bit extraction per weight
3. Bit manipulation for 10-bit index construction

However, the small memory footprint means more of the model fits in cache, which can offset the dequantization cost for memory-bandwidth-bound inference.

A 7B model in IQ2_S uses approximately 2.2 GB -- small enough to fit on devices with very limited memory.

## Typical Usage

- Ultra-compressed models for memory-constrained deployment
- Large models (30B+, 70B+) compressed to fit in limited VRAM or RAM
- Experimental and research use where maximum compression is needed
