# IQ4_NL -- Importance-Weighted 4-bit Non-Linear

## Overview

| Property | Value |
|----------|-------|
| Full name | Importance-Weighted 4-bit Non-Linear |
| GGML type ID | 20 |
| Bits per weight | 4.5 bpw |
| Block size | 32 elements |
| Block bytes | 18 bytes |
| Compression ratio | 7.1x vs F32 |

## Data Layout

Each 18-byte block encodes 32 weights:

```
Offset  Size   Field    Description
------  -----  -------  -------------------------------------------
0       2      d        fp16 block scale
2       16     qs       16 bytes of 4-bit indices (non-linear lookup)
------  -----  -------  -------------------------------------------
Total:  18 bytes
```

### Split Nibble Layout

IQ4_NL uses a **SPLIT** nibble layout (same as Q5_0, different from Q4_0):

```
Elements  0-15:  LOW  nibbles of qs bytes 0-15
Elements 16-31:  HIGH nibbles of qs bytes 0-15
```

This matches llama.cpp's `dequantize_row_iq4_nl` which writes to `y[j+0]` and `y[j+QK4_NL/2]`.

## Non-Linear Lookup Table

Unlike Q4_0 which uses a linear `(nibble - 8)` mapping, IQ4_NL maps each 4-bit value through a precomputed non-linear table derived from K-means clustering:

```
KVALUES_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
}
```

The values are not uniformly spaced. The negative side has larger gaps at the extremes, reflecting the typical distribution of neural network weights (roughly Gaussian, with more density near zero).

## Dequantization Formula

```
// Split layout
if j < 16:
    nibble = qs[j] & 0x0F
else:
    nibble = (qs[j - 16] >> 4) & 0x0F

value = d * KVALUES_IQ4NL[nibble]
```

## CUDA Kernel

| Property | Value |
|----------|-------|
| Kernel file (FP32) | `matmul_iq4_nl.cu` |
| Kernel file (dp4a) | `matmul_iq4_nl_dp4a.cu` (added v1.11.0) |
| Kernel file (dp4a + smem) | `matmul_iq4_nl_dp4a_smem.cu` (added v1.13.0, **opt-in only**) |
| Strategy | Warp-per-row |
| Alignment | 18 bytes -- NOT 4-byte aligned, forces byte `__ldg` |
| Available since | v1.5.1; dp4a since v1.11.0 |
| Default | dp4a path on via `-Dcuda.dp4a=true`; smem variant off via `-Dcuda.iq4nl.smem=false` |

### dp4a path (v1.11.0)

The dp4a variant quantizes the input vector to Q8_1 and uses `__dp4a` int8 dot products against the IQ4_NL weights after lookup-table reconstruction. Wired across all three GPU forward passes (`CudaForwardPass`, `Qwen35CudaForwardPass`, `NemotronHCudaForwardPass`), and as of v1.13.0 also through `launchOutputMatmul` for the final-output projection.

**Measured (RTX 4050 Laptop GPU):**
- **Phi-3-mini IQ4_NL: 8.4 → 11.9 tok/s (+42 %)** -- the largest gain in the v1.11.0 dp4a wave, because Phi-3-mini stores essentially all its weights as IQ4_NL and the FP32 fallback was the dominant cost.

### Shared-memory variant (v1.13.0)

`matmul_iq4_nl_dp4a_smem.cu` caches the Q8_1 input vector in shared memory. It was written for the same reason as the Q5_0 smem variant, but measured **neutral** on Phi-3-mini in CUDA graph mode (12.0 vs 12.07 tok/s) -- graph mode already amortizes the kernel-launch overhead the smem variant was meant to save. The kernel is therefore kept opt-in only via `-Dcuda.iq4nl.smem=true`, retained for no-graph fallback and for hardware with different cache characteristics where it might still help.

## SIMD Optimization

| Property | Value |
|----------|-------|
| Fused SIMD class | None |
| CPU dot path | Dequantize via lookup table to buffer, then SIMD dot |

IQ4_NL was explicitly **not** covered by the v1.12.0 CPU SIMD B2I/I2F sweep because its dequantization is a non-linear table lookup (`KVALUES_IQ4NL[16]`) rather than a linear `(quant - zero) * scale` operation. Going fully lane-parallel here requires `VectorShuffle.rearrange` over a pre-multiplied lookup table -- not yet implemented. Phi-3-mini IQ4_NL at ~1.0 tok/s on CPU remains the worst CPU performer in the LLMPlayer test fleet for this reason; the v1.11.0 GPU dp4a kernel above is the practical path forward for this format.

## Performance Characteristics

IQ4_NL has the same block size and byte count as Q4_0 (18 bytes for 32 weights), but the non-linear lookup table provides better reconstruction quality. The K-means-derived quantization levels are optimized for typical neural network weight distributions.

The trade-off vs Q4_0:
- **Better quality**: Non-linear levels match weight distributions more closely
- **Same size**: Identical 18 bytes per block
- **Slightly more compute**: Lookup table access vs simple subtraction

The trade-off vs IQ4_XS:
- **Simpler**: 32-element blocks with single scale (vs 256-element super-blocks with per-sub-block 6-bit scales)
- **More overhead**: 2 bytes scale per 32 weights = 6.25% overhead (vs IQ4_XS's more efficient scale sharing)
- **Same lookup table**: Both use KVALUES_IQ4NL

## Typical Usage

- Llama IQ4_NL quantizations
- Phi-3 IQ4_NL quantizations
- When Q4_0 quality is insufficient at the same memory budget
- The base building block for IQ4_XS (which extends it with super-block structure)
