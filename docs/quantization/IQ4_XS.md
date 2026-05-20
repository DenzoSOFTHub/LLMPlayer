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
| Kernel file (FP32) | `matmul_iq4_xs.cu` |
| Kernel file (dp4a) | `matmul_iq4_xs_dp4a.cu` (added v1.11.0) |
| Kernel file (dp4a multi-row) | `matmul_iq4_xs_dp4a_mw.cu` |
| Strategy | Warp-per-row |
| Alignment | 136 bytes IS 4-byte aligned -- safe for vectorized `uint32` `__ldg` |
| Available since | v1.5.1; dp4a since v1.11.0 |
| Default | dp4a path on via `-Dcuda.dp4a=true` |

### dp4a path (v1.11.0)

The dp4a variant quantizes the input vector to Q8_1 and uses `__dp4a` int8 dot products against the IQ4_XS weights after `KVALUES_IQ4NL` lookup reconstruction. Wired across all three GPU forward passes (`CudaForwardPass`, `Qwen35CudaForwardPass`, `NemotronHCudaForwardPass`), and as of v1.13.0 also through `launchOutputMatmul` for the final-output projection.

**Measured (RTX 4050 Laptop GPU):**
- Gemma-2-2B IQ4_XS: 8.6 → 9.0 tok/s (+5 %) -- modest because each row has only 9 super-blocks, limiting the parallelism the dp4a kernel can extract per row.

The multi-row dp4a kernel (`matmul_iq4_xs_dp4a_mw.cu`) trades per-block overhead for additional row-level parallelism and is available as an alternative dispatch path on Qwen35CudaForwardPass for workloads where it helps.

## SIMD Optimization

| Property | Value |
|----------|-------|
| Fused SIMD class | None |
| CPU dot path | Dequantize via lookup table to buffer, then SIMD dot |

IQ4_XS was explicitly **not** covered by the v1.12.0 CPU SIMD B2I/I2F sweep because, like IQ4_NL, its dequantization is a non-linear `KVALUES_IQ4NL[16]` table lookup. Going fully lane-parallel here requires `VectorShuffle.rearrange` over a pre-multiplied lookup table -- deferred until a workload makes the engineering cost worthwhile. GPU dp4a (above) is the recommended path for IQ4_XS workloads today.

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
