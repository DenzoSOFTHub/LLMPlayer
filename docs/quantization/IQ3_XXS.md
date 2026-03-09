# IQ3_XXS -- Importance-Weighted 3-bit Extra-Extra-Small

## Overview

| Property | Value |
|----------|-------|
| Full name | Importance-Weighted 3-bit Extra-Extra-Small |
| GGML type ID | 18 |
| Bits per weight | 3.0625 bpw |
| Block size | 256 elements |
| Block bytes | 98 bytes |
| Compression ratio | 10.4x vs F32 |

## Data Layout

Each 98-byte super-block encodes 256 weights:

```
Offset  Size   Field              Description
------  -----  -----------------  -------------------------------------------
0       2      d                  fp16 super-block scale
2       64     qs                 8-bit grid indices (8 per group of 32)
66      32     scales_and_signs   uint32 per group: 4-bit scale + sign indices
------  -----  -----------------  -------------------------------------------
Total:  98 bytes
```

The 256 weights are organized into 8 groups of 32, each group divided into 4 sub-groups of 8.

## Grid-Based Dequantization

IQ3_XXS uses the `IQ3XXS_GRID` lookup table (256 entries, each a `uint32` encoding 4 unsigned byte values). An 8-bit grid index selects the entry. Sign information comes from a 7-bit index into the `KSIGNS_IQ2XS` lookup table.

### Packed scales_and_signs

Each group of 32 weights has a `uint32` value that encodes both the scale and sign information:

```
aux32 = scales_and_signs[ib32]     // uint32 for group ib32

scale_nibble = aux32 >>> 28        // top 4 bits: scale (0-15)
db = d * (0.5 + scale_nibble) * 0.5

// 4 sub-groups of 8 weights each:
for l in 0..3:
    sign_idx = (aux32 >>> (7 * l)) & 0x7F    // 7-bit sign index
    signs = KSIGNS_IQ2XS[sign_idx] & 0xFF    // 8 sign bits
```

### Grid Lookup

Each sub-group of 8 weights uses 2 grid indices (each covering 4 weights):

```
grid_idx_1 = qs[ib32 * 8 + 2 * l]       // first 4 weights
grid_idx_2 = qs[ib32 * 8 + 2 * l + 1]   // second 4 weights

grid_val = IQ3XXS_GRID[grid_idx]         // uint32: 4 unsigned bytes
byte_value = (grid_val >> (8 * j)) & 0xFF
```

## Dequantization Formula

```
value = db * grid_byte * sign

where:
    db = d * (0.5 + (aux32 >>> 28)) * 0.5
    grid_byte from IQ3XXS_GRID[8-bit index]
    sign from KSIGNS_IQ2XS[7-bit index]
```

## CUDA Kernel

| Property | Value |
|----------|-------|
| Kernel file | `matmul_iq3_xxs.cu` |
| Strategy | Warp-per-row |
| Lookup tables | `__constant__` IQ3XXS_GRID and KSIGNS_IQ2XS |
| Alignment | 98 bytes -- NOT 4-byte aligned |

### Known Issue

The kernel uses `*(unsigned int*)` casts for reading `scales_and_signs` data. Since the 98-byte block is not 4-byte aligned, this can cause CUDA error 716 (misaligned address) on some models depending on tensor alignment within the GGUF file.

## SIMD Optimization

| Property | Value |
|----------|-------|
| Fused SIMD class | None |
| CPU dot path | Dequantize via grid lookup to buffer, then SIMD dot |

## Performance Characteristics

IQ3_XXS achieves aggressive compression at 3.0625 bpw through grid-based quantization with shared sign tables. The "XXS" suffix indicates this is the smallest variant of 3-bit importance quantization.

Dequantization involves two lookup table accesses per sub-group of 8 weights:
1. `IQ3XXS_GRID` for weight magnitudes (256-entry table)
2. `KSIGNS_IQ2XS` for sign patterns (128-entry table)

The compact `scales_and_signs` packing is clever: by sharing a single `uint32` across scale and 4 sign indices, the format saves 12 bytes per group compared to storing them separately.

A 7B model in IQ3_XXS uses approximately 2.7 GB.

## Typical Usage

- Llama IQ3_XXS quantizations
- Aya IQ3_XXS quantizations
- Highly compressed models where IQ2_S quality is too low but Q3_K memory cost is too high
- Memory-constrained deployments of mid-size models
