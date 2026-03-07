package it.denzosoft.llmplayer.tensor;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * SIMD-optimized Q8_0 tensor with fused dequantization and dot product.
 * Uses direct MemorySegment access and Java Vector API to eliminate:
 * - ThreadLocal overhead
 * - VectorOpsFactory.get() virtual dispatch
 * - Intermediate tmp[32] buffer write+read
 */
public class SimdQ8_0FloatTensor extends Q8_0FloatTensor {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int F_LEN = F_SPECIES.length();
    private static final int BLOCK_SIZE = 32;
    private static final int BLOCK_BYTES = 34;
    private static final ValueLayout.OfByte BYTE_LE = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED;

    private final MemorySegment segment;

    public SimdQ8_0FloatTensor(TensorData data, long size) {
        super(data, size);
        this.segment = ((MemorySegmentTensorData) data).segment();
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        if (32 % F_LEN != 0 || length % BLOCK_SIZE != 0) {
            return super.dot(thisOffset, other, otherOffset, length);
        }

        int numBlocks = length / BLOCK_SIZE;
        long blockStart = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        int otherBase = otherOffset;

        FloatVector acc = FloatVector.zero(F_SPECIES);
        float[] dq = new float[F_LEN];

        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float scale = Float16.toFloat(segment.get(SHORT_LE, bo));
            FloatVector vScale = FloatVector.broadcast(F_SPECIES, scale);

            for (int i = 0; i < BLOCK_SIZE; i += F_LEN) {
                for (int j = 0; j < F_LEN; j++) {
                    dq[j] = (float) segment.get(BYTE_LE, bo + 2 + i + j);
                }
                FloatVector vq = FloatVector.fromArray(F_SPECIES, dq, 0);
                FloatVector w = vq.mul(vScale);
                FloatVector in = FloatVector.fromArray(F_SPECIES, other, otherBase + i);
                acc = w.fma(in, acc);
            }
            otherBase += BLOCK_SIZE;
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }
}
