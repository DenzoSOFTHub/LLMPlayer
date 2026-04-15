package it.denzosoft.llmplayer.tensor;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * SIMD-optimized Q4_0 tensor with fused dequantization and dot product.
 *
 * Q4_0 block layout (18 bytes, 32 elements):
 *   scale (fp16, 2 bytes)
 *   qs[16]: packed 4-bit quants, INTERLEAVED
 * Element mapping (byte i packs elements 2i and 2i+1):
 *   elem[2i]   = (low_nibble  - 8) * scale
 *   elem[2i+1] = (high_nibble - 8) * scale
 */
public class SimdQ4_0FloatTensor extends Q4_0FloatTensor {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int F_LEN = F_SPECIES.length();
    private static final int BLOCK_SIZE = 32;
    private static final int BLOCK_BYTES = 18;
    private static final ValueLayout.OfByte BYTE_LE = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED;

    private final MemorySegment segment;

    public SimdQ4_0FloatTensor(TensorData data, long size) {
        super(data, size);
        this.segment = ((MemorySegmentTensorData) data).segment();
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        int numBlocks = length / BLOCK_SIZE;
        long blockStart = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        int otherIdx = otherOffset;

        FloatVector acc = FloatVector.zero(F_SPECIES);
        float[] dq = new float[BLOCK_SIZE];
        byte[] qs = new byte[16];

        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float scale = Float.float16ToFloat(segment.get(SHORT_LE, bo));

            MemorySegment.copy(segment, BYTE_LE, bo + 2, qs, 0, 16);

            for (int j = 0; j < 16; j++) {
                int packed = qs[j] & 0xFF;
                int lo = packed & 0x0F;
                int hi = (packed >> 4) & 0x0F;
                dq[2 * j]     = (lo - 8) * scale;
                dq[2 * j + 1] = (hi - 8) * scale;
            }

            int loopBound = F_SPECIES.loopBound(BLOCK_SIZE);
            for (int j = 0; j < loopBound; j += F_LEN) {
                FloatVector vw = FloatVector.fromArray(F_SPECIES, dq, j);
                FloatVector vIn = FloatVector.fromArray(F_SPECIES, other, otherIdx + j);
                acc = vw.fma(vIn, acc);
            }
            for (int j = loopBound; j < BLOCK_SIZE; j++) {
                acc = acc.withLane(0, acc.lane(0) + dq[j] * other[otherIdx + j]);
            }

            otherIdx += BLOCK_SIZE;
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }
}
