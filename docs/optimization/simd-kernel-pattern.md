# CPU SIMD kernel pattern (B2I/I2F lane-parallel)

All block K-quant and Q8_0 SIMD kernels under `src/main/java21/.../tensor/Simd*FloatTensor.java`
share a common pattern as of v1.12.0-dev (2026-04-15). Before writing a new SIMD kernel — or before
reviewing an existing one for a performance regression — verify that it follows this template.

## The template

1. Read the packed bytes directly from the mapped `MemorySegment` via
   `ByteVector.fromMemorySegment(B_SPECIES, segment, offset, BYTE_ORDER)`. No `byte[]` scratch
   buffer, and no `MemorySegment.copy` inside the hot loop.
2. Widen with `vbyte.convertShape(VectorOperators.B2I, I_SPECIES, 0)` into an `IntVector` —
   sign-extended for signed quants such as Q8_0, or masked with `vand(0xFF)` for unsigned.
3. Extract nibbles and bit-packed fields lane-parallel with `.and(mask)`,
   `.lanewise(LSHR, shift)`, and `.lanewise(LSHL, shift)`. Never write a scalar `for j in F_LEN`
   loop feeding a `float[F_LEN]` scratch array — that is "SIMD only in the final FMA" and shows up
   as a hotspot in JFR.
4. Apply the dequantization offset in the `IntVector` domain, for example `.sub(vSub32)` for Q6_K,
   `.sub(vSub16)` for Q5_0, or `.sub(vSub4)` for Q3_K.
5. Convert with `q.convertShape(VectorOperators.I2F, F_SPECIES, 0)` into a `FloatVector`.
6. FMA against the input vector and the scale broadcast: `acc = vq.fma(vds.mul(in), acc)` or
   `acc = w.fma(in, acc)`.

Target `F_SPECIES = SPECIES_256` (8 floats, AVX2). Guard the kernel with

```java
if (FloatVector.SPECIES_PREFERRED.length() != 8 || length % BLOCK_SIZE != 0)
    return super.dot(...);
```

so it falls back to the scalar parent class on non-AVX2 hardware or for odd-sized tensors.

## Reference implementations

Cross-check these when adding a new quantization type:

| Class | What it demonstrates |
|---|---|
| `SimdQ4_KFloatTensor` | The canonical form, nibble extraction only |
| `SimdQ6_KFloatTensor` | Nibble plus 2-bit `qh` (4 sub-blocks × 2 halves) |
| `SimdQ5_KFloatTensor` | Nibble plus 1-bit `qh` (4 groups) |
| `SimdQ5_0FloatTensor` | Nibble plus 1-bit `qh` broadcast via a constant shift-vector |
| `SimdQ3_KFloatTensor` | 2-bit low plus 1-bit hmask (16 sub-blocks) |
| `SimdQ8_0FloatTensor` | No bit-packing — the simplest B2I/I2F chain |

## Measured results of the 2026-04-15 sweep

That sweep rewrote all five of Q6_K, Q8_0, Q5_K, Q5_0, and Q3_K to follow this template. Measured
CPU tok/s gains on small and medium models:

| Model | Gain |
|---|---|
| Llama-3.2-1B Q4_K_M | +80–100% |
| Qwen3-4B-Thinking Q8_0 | +327% |
| Qwen3-1.7B Q8_0 | +190% |
| Llama-3.2-3B Q3_K_L | +192% |
| gemma-3-1B | +127–293% |

PPL was bit-identical across the board.

JFR method sampling was the key tool for identifying which `Simd*FloatTensor.dot` was the active
hotspot for a given model, because the quantization mix inside a GGUF rarely matches its name. For
example, Q4_K_M ships Q6_K for `output.weight` and for some `ffn_down` and `attn_v` tensors, so Q6_K
dominated Llama-1B even though it is nominally a "Q4_K" model. Profile before optimizing.

## Types still on the scalar lookup-table path

IQ4_NL, IQ4_XS, IQ3_XXS, IQ3_S, and IQ2_S are **not** covered by the B2I/I2F template. They use
non-linear k-means centroids or grid codebooks, and going fully lane-parallel would require
`VectorShuffle.rearrange` over a pre-multiplied table.

This is why Phi-3-mini IQ4_NL at 1.0 tok/s is the worst CPU performer in the suite. The GPU dp4a
kernels added in v1.11.0 already cover these types, so the gap only shows on CPU-only runs.
