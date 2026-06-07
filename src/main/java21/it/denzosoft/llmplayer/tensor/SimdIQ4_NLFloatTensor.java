package it.denzosoft.llmplayer.tensor;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * SIMD-optimized IQ4_NL tensor with fused dequantization and dot product.
 * Eliminates ThreadLocal overhead, VectorOpsFactory dispatch, and intermediate buffer.
 *
 * IQ4_NL block layout (18 bytes, 32 elements):
 *   scale (fp16, 2 bytes)
 *   qs[16]: packed 4-bit nibbles (non-linear lookup)
 * Split nibble layout: low nibbles → elements 0-15, high nibbles → elements 16-31
 */
public class SimdIQ4_NLFloatTensor extends IQ4_NLFloatTensor {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int F_LEN = F_SPECIES.length();
    private static final int BLOCK_SIZE = 32;
    private static final int BLOCK_BYTES = 18;
    private static final ValueLayout.OfByte BYTE_LE = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED;

    private final MemorySegment segment;

    public SimdIQ4_NLFloatTensor(TensorData data, long size) {
        super(data, size);
        this.segment = ((MemorySegmentTensorData) data).segment();
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        int numBlocks = length / BLOCK_SIZE;
        long blockStart = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        int otherIdx = otherOffset;

        // 16 is block-clean only for F_LEN in {4,8,16}; fall back otherwise.
        if ((16 % F_LEN) != 0) return super.dot(thisOffset, other, otherOffset, length);

        FloatVector acc = FloatVector.zero(F_SPECIES);
        final byte[] qs = new byte[16];
        // Per-block scaled codebook: sk[i] = scale*KVALUES[i]. The nibble->value mapping then
        // becomes a SIMD gather (vgatherdps) over sk instead of 32 scalar table lookups.
        final float[] sk = new float[16];
        final int[] lo = new int[16];
        final int[] hi = new int[16];

        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float scale = Float.float16ToFloat(segment.get(SHORT_LE, bo));
            MemorySegment.copy(segment, BYTE_LE, bo + 2, qs, 0, 16);

            for (int i = 0; i < 16; i++) {
                sk[i] = scale * KVALUES_IQ4NL[i];
                int v = qs[i] & 0xFF;
                lo[i] = v & 0x0F;   // -> positions 0..15
                hi[i] = v >>> 4;    // -> positions 16..31
            }

            // Elements 0..15 (low nibbles): gather sk[lo[]] then FMA with input
            for (int j = 0; j < 16; j += F_LEN) {
                FloatVector vw = FloatVector.fromArray(F_SPECIES, sk, 0, lo, j);
                FloatVector vIn = FloatVector.fromArray(F_SPECIES, other, otherIdx + j);
                acc = vw.fma(vIn, acc);
            }
            // Elements 16..31 (high nibbles)
            for (int j = 0; j < 16; j += F_LEN) {
                FloatVector vw = FloatVector.fromArray(F_SPECIES, sk, 0, hi, j);
                FloatVector vIn = FloatVector.fromArray(F_SPECIES, other, otherIdx + 16 + j);
                acc = vw.fma(vIn, acc);
            }
            otherIdx += BLOCK_SIZE;
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }
}
