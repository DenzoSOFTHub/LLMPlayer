# MXFP4 -- Microscaling FP4

## Overview

| Property | Value |
|----------|-------|
| Full name | Microscaling FP4 (E2M1 with E8M0 shared exponent) |
| GGML type ID | 39 |
| Bits per weight | 4.25 bpw |
| Block size | 32 elements |
| Block bytes | 17 bytes |
| Compression ratio | 7.5x vs F32 |

## Data Layout

Each 17-byte block encodes 32 weights:

```
Offset  Size   Field    Description
------  -----  -------  -------------------------------------------
0       1      e        E8M0 shared block exponent (power-of-2 scale)
1       16     qs       16 bytes of FP4 E2M1 values (2 per byte)
------  -----  -------  -------------------------------------------
Total:  17 bytes
```

### Split Nibble Layout

MXFP4 uses a **SPLIT** nibble layout (same as IQ4_NL and Q5_0, not interleaved like Q4_0):

```
Elements  0-15:  LOW  nibbles of qs bytes 0-15
Elements 16-31:  HIGH nibbles of qs bytes 0-15
```

### E8M0 Shared Exponent

The block scale uses E8M0 format -- an 8-bit unsigned exponent with no mantissa:

```
scale = 2^(exponent - 127)

Special cases:
    exponent == 0:   scale = 0  (subnormal/zero block)
    exponent == 255: scale = 0  (NaN/invalid block, per OCP MX spec)
```

The E8M0 format gives powers of two only, which means the scale is always an exact power of 2. This is efficient for hardware implementation (just shift the exponent field) but less flexible than fp16 scales.

Implementation in Java:
```java
float scale = Float.intBitsToFloat(exp << 23);  // builds F32 with biased exponent = exp
```

### FP4 E2M1 Encoding

Each 4-bit weight value uses E2M1 floating-point format:

```
Bit layout: [sign (1 bit)] [exponent (2 bits)] [mantissa (1 bit)]

Representable values (16 total):
    Positive: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    Negative: -0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
```

Lookup table:
```
FP4_TABLE[16] = {
     0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
}
```

Note the non-uniform spacing: the gap between representable values increases with magnitude (0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 2.0). This is inherent to the floating-point representation and provides higher precision near zero.

## Dequantization Formula

```
scale = 2^(e - 127)       // E8M0 to float (or 0 for e=0 or e=255)

if j < 16:
    nibble = qs[j] & 0x0F
else:
    nibble = (qs[j - 16] >> 4) & 0x0F

value = FP4_TABLE[nibble] * scale
```

## CUDA Kernel

| Property | Value |
|----------|-------|
| Kernel file | None (no dedicated CUDA kernel) |
| Status | CPU-only |

MXFP4 does not have a dedicated CUDA matmul kernel. Inference uses CPU dequantization with SIMD dot product.

## SIMD Optimization

| Property | Value |
|----------|-------|
| Fused SIMD class | None |
| CPU dot path | Dequantize via FP4 lookup table to buffer, then SIMD dot |

## Performance Characteristics

MXFP4 is a hardware-oriented format defined by the Open Compute Project (OCP) Microscaling specification. Key characteristics:

- **Power-of-2 scaling**: E8M0 exponent means the block scale is always an exact power of 2, which is cheap to apply in hardware but loses the fine-grained scale precision of fp16 scales used by other formats
- **FP4 representation**: Only 16 distinct values per weight (8 positive, 8 negative including zero), with non-uniform spacing that provides higher precision near zero
- **Compact**: 17 bytes for 32 weights (4.25 bpw), slightly more efficient than Q4_0/IQ4_NL (18 bytes) due to the 1-byte E8M0 scale vs 2-byte fp16 scale

Compared to other 4-bit formats:
- vs Q4_0 (4.5 bpw): Smaller, but FP4's 16 non-uniform levels vs Q4_0's 16 uniform levels
- vs IQ4_NL (4.5 bpw): Smaller, but IQ4_NL's K-means-optimized levels may better match weight distributions
- vs IQ4_XS (4.25 bpw): Same bpw, but MXFP4 uses per-block (32) scales vs IQ4_XS's per-sub-block (32 within 256) scales

## Typical Usage

- sonar-oss-20b-MXFP4Q8_MOE (MoE model with MXFP4 expert weights)
- Models targeting hardware with native MXFP4 support (NVIDIA Blackwell, AMD MI300+)
- Future-oriented format as hardware MXFP4 support expands
