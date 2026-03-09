package it.denzosoft.llmplayer.tensor;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * SIMD-optimized Q5_K tensor with fused dequantization and dot product.
 * Eliminates 6 ThreadLocal lookups per dot call and intermediate tmp[256] buffer.
 *
 * Q5_K block layout (176 bytes, 256 elements):
 *   d (fp16, 2 bytes): super-block scale
 *   dmin (fp16, 2 bytes): super-block minimum
 *   scales[12]: 8 scale+min pairs packed in 6 bits
 *   qh[32]: high bit for each of 256 weights
 *   qs[128]: 256 x 4-bit low quants
 */
public class SimdQ5_KFloatTensor extends Q5_KFloatTensor {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int F_LEN = F_SPECIES.length();
    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 176;
    private static final ValueLayout.OfByte BYTE_LE = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED;

    private final MemorySegment segment;

    public SimdQ5_KFloatTensor(TensorData data, long size) {
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
        byte[] sb = new byte[12];
        byte[] qhBytes = new byte[32];
        byte[] qs = new byte[128];
        int[] scales = new int[8];
        int[] mins = new int[8];
        float[] dq = new float[F_LEN];

        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float d = Float.float16ToFloat(segment.get(SHORT_LE, bo));
            float dmin = Float.float16ToFloat(segment.get(SHORT_LE, bo + 2));

            MemorySegment.copy(segment, BYTE_LE, bo + 4, sb, 0, 12);
            MemorySegment.copy(segment, BYTE_LE, bo + 16, qhBytes, 0, 32);
            MemorySegment.copy(segment, BYTE_LE, bo + 48, qs, 0, 128);

            // Decode scales and mins (same packing as Q4_K)
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
                    // Low nibbles: elements group*64 + 0..31
                    for (int j = 0; j < F_LEN; j++) {
                        int idx = l + j;
                        int qsByte = Byte.toUnsignedInt(qs[qsGroupBase + idx]);
                        int qhByte = Byte.toUnsignedInt(qhBytes[idx]);
                        int ql = qsByte & 0x0F;
                        int qh = (qhByte >> (group * 2)) & 1;
                        dq[j] = (float) (ql | (qh << 4));
                    }
                    FloatVector vq0 = FloatVector.fromArray(F_SPECIES, dq, 0);
                    FloatVector w0 = vq0.fma(vds0, vNegDm0);
                    FloatVector in0 = FloatVector.fromArray(F_SPECIES, other, lowInputBase + l);
                    acc = w0.fma(in0, acc);

                    // High nibbles: elements group*64 + 32..63
                    for (int j = 0; j < F_LEN; j++) {
                        int idx = l + j;
                        int qsByte = Byte.toUnsignedInt(qs[qsGroupBase + idx]);
                        int qhByte = Byte.toUnsignedInt(qhBytes[idx]);
                        int ql = (qsByte >> 4) & 0x0F;
                        int qh = (qhByte >> (group * 2 + 1)) & 1;
                        dq[j] = (float) (ql | (qh << 4));
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
