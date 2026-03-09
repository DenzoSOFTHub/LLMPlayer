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

        FloatVector acc = FloatVector.zero(F_SPECIES);
        byte[] qs = new byte[16];
        float[] dqLo = new float[16];
        float[] dqHi = new float[16];

        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float scale = Float.float16ToFloat(segment.get(SHORT_LE, bo));

            // Bulk copy packed nibbles
            MemorySegment.copy(segment, BYTE_LE, bo + 2, qs, 0, 16);

            // Dequantize using non-linear lookup table
            for (int i = 0; i < 16; i++) {
                int lo = qs[i] & 0x0F;
                int hi = (qs[i] >> 4) & 0x0F;
                dqLo[i] = scale * KVALUES_IQ4NL[lo];
                dqHi[i] = scale * KVALUES_IQ4NL[hi];
            }

            // SIMD FMA for elements 0..15
            int loopBound = F_SPECIES.loopBound(16);
            for (int j = 0; j < loopBound; j += F_LEN) {
                FloatVector vw = FloatVector.fromArray(F_SPECIES, dqLo, j);
                FloatVector vIn = FloatVector.fromArray(F_SPECIES, other, otherIdx + j);
                acc = vw.fma(vIn, acc);
            }
            for (int j = loopBound; j < 16; j++) {
                acc = acc.withLane(0, acc.lane(0) + dqLo[j] * other[otherIdx + j]);
            }

            // SIMD FMA for elements 16..31
            for (int j = 0; j < loopBound; j += F_LEN) {
                FloatVector vw = FloatVector.fromArray(F_SPECIES, dqHi, j);
                FloatVector vIn = FloatVector.fromArray(F_SPECIES, other, otherIdx + 16 + j);
                acc = vw.fma(vIn, acc);
            }
            for (int j = loopBound; j < 16; j++) {
                acc = acc.withLane(0, acc.lane(0) + dqHi[j] * other[otherIdx + 16 + j]);
            }

            otherIdx += BLOCK_SIZE;
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }
}
