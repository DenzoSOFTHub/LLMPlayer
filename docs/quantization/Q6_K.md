# Q6_K -- 6-bit K-Quant

## Overview

| Property | Value |
|----------|-------|
| Full name | 6-bit K-Quantization |
| GGML type ID | 14 |
| Bits per weight | 6.5625 bpw |
| Block size | 256 elements |
| Block bytes | 210 bytes |
| Compression ratio | 4.9x vs F32 |

## Data Layout

Each 210-byte super-block encodes 256 weights:

```
Offset  Size   Field    Description
------  -----  -------  -------------------------------------------
0       128    ql       Lower 4 bits of 6-bit quants (complex packing)
128     64     qh       Upper 2 bits of 6-bit quants (complex packing)
192     16     scales   16 x int8 sub-block scales
208     2      d        fp16 super-block scale
------  -----  -------  -------------------------------------------
Total:  210 bytes
```

The 256 weights are divided into 16 sub-blocks of 16 weights each, each with its own signed int8 scale.

### Bit Packing Structure

The 6-bit value is split into 4 low bits (from `ql`) and 2 high bits (from `qh`), organized by halves (0-127, 128-255) and quadrants:

```
half     = j / 128           // 0 or 1
quadrant = (j % 128) / 32    // 0-3
l        = j % 32            // position within 32-element group

ql_base = half * 64
qh_base = 128 + half * 32

Quadrant 0: ql4 = ql[ql_base + l]      & 0x0F,  qh = (qh[qh_base + l] >> 0) & 3
Quadrant 1: ql4 = ql[ql_base + 32 + l] & 0x0F,  qh = (qh[qh_base + l] >> 2) & 3
Quadrant 2: ql4 = (ql[ql_base + l]      >> 4) & 0x0F,  qh = (qh[qh_base + l] >> 4) & 3
Quadrant 3: ql4 = (ql[ql_base + 32 + l] >> 4) & 0x0F,  qh = (qh[qh_base + l] >> 6) & 3
```

Each `ql` byte is shared between quadrants 0+2 (or 1+3): the low nibble serves one quadrant, the high nibble serves the other. Similarly, each `qh` byte provides 2-bit high parts for all 4 quadrants of the same position.

## Dequantization Formula

```
q6 = ql4 | (qh2 << 4)         // 6-bit value: 0-63

sub_block = j / 16
sc = scales[sub_block]          // signed int8 scale (-128 to +127)

value = d * sc * (q6 - 32)     // centered around 32
```

The zero-point of 32 centers the 6-bit range, giving an effective signed range of -32 to +31.

## CUDA Kernel

| Property | Value |
|----------|-------|
| Kernel file | `matmul_q6_k.cu` |
| Strategy | **Coalesced kernel** (restructured in v1.4.0) |
| Key optimization | All 32 warp threads process the same half-block with consecutive byte addresses |
| Alignment | 210 bytes -- NOT 4-byte aligned, uses byte-level `__ldg` |

### Performance Impact

The Q6_K kernel restructuring was **the single biggest optimization in LLMPlayer's CUDA backend**:

| Metric | Before (v1.3.0) | After (v1.4.0) | Improvement |
|--------|------------------|-----------------|-------------|
| Output matmul time | 8.3 ms/tok | 2.7 ms/tok | **3.1x faster** |

The original kernel used a half-striped access pattern where threads accessed non-contiguous memory. The coalesced rewrite ensures all 32 threads in a warp process consecutive bytes within the same half-block, resulting in perfectly coalesced memory transactions.

This optimization matters disproportionately because Q6_K is typically used for the **output projection** (embedding to logits), which has `vocabSize` rows -- often 32,000 to 128,000+. This makes the output matmul the single largest kernel launch per token.

## SIMD Optimization

| Property | Value |
|----------|-------|
| Fused SIMD class | `SimdQ6_KFloatTensor` (java21) |
| Available since | v1.6.0 |
| CPU dot path | Fused dequant+dot with SIMD vector operations |

## Performance Characteristics

Q6_K provides near-lossless quantization quality at 6.5625 bpw. The complex bit packing (4+2 split across two arrays, with quadrant-based indexing) makes the dequantization logic the most involved among K-quants.

The 210-byte block size is not 4-byte aligned (210 = 2 * 3 * 5 * 7), which prevents vectorized `uint32` loads in CUDA kernels. The coalesced memory access pattern compensates for this limitation.

**Critical role in inference:** Since Q6_K is almost always used for the output/embedding tensor (`token_embd.weight`), its kernel performance directly determines the per-token output projection cost. For models with large vocabularies (128K+ tokens), this single matmul can dominate total inference time.

## Typical Usage

- Output/embedding weights (`token_embd.weight`) in most quantized GGUF models
- Full Q6_K models for near-lossless quality at ~6.5 bpw
- The highest-quality K-quant commonly used in mixed-quantization files
- A 7B model fully in Q6_K uses approximately 5.7 GB
