package it.denzosoft.llmplayer.tensor;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * SIMD-optimized Q4_K tensor with fused dequantization and dot product.
 * Uses direct MemorySegment access and Java Vector API to eliminate:
 * - ThreadLocal overhead (5 lookups per dot call)
 * - VectorOpsFactory.get() virtual dispatch (1 per block)
 * - Intermediate tmp[256] buffer write+read (1 KB per block)
 * - TensorData.copyBytes() virtual dispatch
 */
public class SimdQ4_KFloatTensor extends Q4_KFloatTensor {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int F_LEN = F_SPECIES.length();
    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 144;
    private static final ValueLayout.OfByte BYTE_LE = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED;

    private final MemorySegment segment;

    public SimdQ4_KFloatTensor(TensorData data, long size) {
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
        byte[] qs = new byte[128];
        byte[] sb = new byte[12];
        int[] scales = new int[8];
        int[] mins = new int[8];
        float[] dq = new float[F_LEN];

        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float d = Float16.toFloat(segment.get(SHORT_LE, bo));
            float dmin = Float16.toFloat(segment.get(SHORT_LE, bo + 2));

            MemorySegment.copy(segment, BYTE_LE, bo + 4, sb, 0, 12);
            MemorySegment.copy(segment, BYTE_LE, bo + 16, qs, 0, 128);

            for (int i = 0; i < 4; i++) {
                scales[i] = Byte.toUnsignedInt(sb[i]) & 0x3F;
                mins[i] = Byte.toUnsignedInt(sb[i + 4]) & 0x3F;
            }
            for (int i = 4; i < 8; i++) {
                scales[i] = (Byte.toUnsignedInt(sb[i + 4]) & 0x0F)
                           | ((Byte.toUnsignedInt(sb[i - 4]) >> 6) << 4);
                mins[i] = ((Byte.toUnsignedInt(sb[i + 4]) >> 4) & 0x0F)
                         | ((Byte.toUnsignedInt(sb[i]) >> 6) << 4);
            }

            for (int group = 0; group < 4; group++) {
                float ds0 = d * scales[group * 2];
                float negDm0 = -(dmin * mins[group * 2]);
                float ds1 = d * scales[group * 2 + 1];
                float negDm1 = -(dmin * mins[group * 2 + 1]);

                FloatVector vds0 = FloatVector.broadcast(F_SPECIES, ds0);
                FloatVector vNegDm0 = FloatVector.broadcast(F_SPECIES, negDm0);
                FloatVector vds1 = FloatVector.broadcast(F_SPECIES, ds1);
                FloatVector vNegDm1 = FloatVector.broadcast(F_SPECIES, negDm1);

                int qsGroupBase = group * 32;
                int lowInputBase = otherBase + group * 64;
                int highInputBase = lowInputBase + 32;

                for (int l = 0; l < 32; l += F_LEN) {
                    // Dequant low nibbles + FMA with input
                    for (int j = 0; j < F_LEN; j++) {
                        dq[j] = (float) (Byte.toUnsignedInt(qs[qsGroupBase + l + j]) & 0x0F);
                    }
                    FloatVector vq0 = FloatVector.fromArray(F_SPECIES, dq, 0);
                    FloatVector w0 = vq0.fma(vds0, vNegDm0);
                    FloatVector in0 = FloatVector.fromArray(F_SPECIES, other, lowInputBase + l);
                    acc = w0.fma(in0, acc);

                    // Dequant high nibbles + FMA with input
                    for (int j = 0; j < F_LEN; j++) {
                        dq[j] = (float) ((Byte.toUnsignedInt(qs[qsGroupBase + l + j]) >> 4) & 0x0F);
                    }
                    FloatVector vq1 = FloatVector.fromArray(F_SPECIES, dq, 0);
                    FloatVector w1 = vq1.fma(vds1, vNegDm1);
                    FloatVector in1 = FloatVector.fromArray(F_SPECIES, other, highInputBase + l);
                    acc = w1.fma(in1, acc);
                }
            }
            otherBase += BLOCK_SIZE;
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }
}
