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
        byte[] qs = new byte[32];

        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float scale = Float.float16ToFloat(segment.get(SHORT_LE, bo));
            FloatVector vScale = FloatVector.broadcast(F_SPECIES, scale);

            // Bulk copy 32 int8 quants (one bounds check instead of 32)
            MemorySegment.copy(segment, BYTE_LE, bo + 2, qs, 0, 32);

            for (int i = 0; i < BLOCK_SIZE; i += F_LEN) {
                for (int j = 0; j < F_LEN; j++) {
                    dq[j] = (float) qs[i + j];
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

    @Override
    public float dot(long thisOffset, FloatTensor other, long otherOffset, int length) {
        if (other instanceof SimdQ8_0FloatTensor) {
            return dotQ8Q8Simd(thisOffset, (SimdQ8_0FloatTensor) other, otherOffset, length);
        }
        return super.dot(thisOffset, other, otherOffset, length);
    }

    /**
     * SIMD-optimized Q8_0 × Q8_0 dot product using integer accumulation per block.
     * Integer accumulation avoids float conversion per element; only one float multiply per block.
     */
    private float dotQ8Q8Simd(long thisOffset, SimdQ8_0FloatTensor other, long otherOffset, int length) {
        int numBlocks = length / BLOCK_SIZE;
        long thisBlock = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        long otherBlock = (otherOffset / BLOCK_SIZE) * BLOCK_BYTES;
        MemorySegment otherSeg = other.segment;

        float result = 0f;
        byte[] qs0 = new byte[32];
        byte[] qs1 = new byte[32];

        for (int b = 0; b < numBlocks; b++) {
            float d0 = Float.float16ToFloat(segment.get(SHORT_LE, thisBlock));
            float d1 = Float.float16ToFloat(otherSeg.get(SHORT_LE, otherBlock));

            // Bulk copy both Q8_0 blocks (one bounds check each instead of 32)
            MemorySegment.copy(segment, BYTE_LE, thisBlock + 2, qs0, 0, 32);
            MemorySegment.copy(otherSeg, BYTE_LE, otherBlock + 2, qs1, 0, 32);

            // Integer accumulation: sum of int8*int8 products
            int isum = 0;
            for (int i = 0; i < BLOCK_SIZE; i++) {
                isum += qs0[i] * qs1[i];
            }
            result += d0 * d1 * isum;

            thisBlock += BLOCK_BYTES;
            otherBlock += BLOCK_BYTES;
        }
        return result;
    }
}
