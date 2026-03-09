package it.denzosoft.llmplayer.tensor;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * SIMD-optimized IQ2_S tensor with fused dequantization and dot product.
 * Eliminates ThreadLocal overhead, VectorOpsFactory dispatch, and intermediate buffer.
 *
 * IQ2_S block layout (82 bytes, 256 elements):
 *   d (fp16, 2 bytes): super-block scale
 *   qs (32 bytes): grid index low 8 bits
 *   signs (32 bytes): sign bits, 1 per weight
 *   qh (8 bytes): grid index high 2 bits (per group of 32)
 *   scales (8 bytes): 4-bit sub-block scales, 2 per byte
 */
public class SimdIQ2_SFloatTensor extends IQ2_SFloatTensor {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int F_LEN = F_SPECIES.length();
    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 82;
    private static final ValueLayout.OfByte BYTE_LE = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED;

    private final MemorySegment segment;

    public SimdIQ2_SFloatTensor(TensorData data, long size) {
        super(data, size);
        this.segment = ((MemorySegmentTensorData) data).segment();
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        int numBlocks = length / BLOCK_SIZE;
        long blockStart = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        int otherBase = otherOffset;

        FloatVector acc = FloatVector.zero(F_SPECIES);
        byte[] qsBytes = new byte[32];
        byte[] signBytesArr = new byte[32];
        byte[] qhBytes = new byte[8];
        byte[] scaleArr = new byte[8];
        float[] dq = new float[32];

        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float d = Float.float16ToFloat(segment.get(SHORT_LE, bo));

            // Bulk copy all block data
            MemorySegment.copy(segment, BYTE_LE, bo + 2, qsBytes, 0, 32);
            MemorySegment.copy(segment, BYTE_LE, bo + 34, signBytesArr, 0, 32);
            MemorySegment.copy(segment, BYTE_LE, bo + 66, qhBytes, 0, 8);
            MemorySegment.copy(segment, BYTE_LE, bo + 74, scaleArr, 0, 8);

            int outIdx = 0;

            for (int ib32 = 0; ib32 < 8; ib32++) {
                int scaleByte = Byte.toUnsignedInt(scaleArr[ib32]);
                float db0 = d * (0.5f + (scaleByte & 0xF)) * 0.25f;
                float db1 = d * (0.5f + ((scaleByte >> 4) & 0xF)) * 0.25f;

                int qhVal = Byte.toUnsignedInt(qhBytes[ib32]);

                int dqIdx = 0;
                for (int l = 0; l < 4; l++) {
                    float dl = (l < 2) ? db0 : db1;

                    int qsVal = Byte.toUnsignedInt(qsBytes[ib32 * 4 + l]);
                    int gridIdx = qsVal | ((qhVal << (8 - 2 * l)) & 0x300);
                    long grid = IQGridTables.IQ2S_GRID[gridIdx];

                    int signByte = Byte.toUnsignedInt(signBytesArr[ib32 * 4 + l]);

                    for (int j = 0; j < 8; j++) {
                        int gv = (int) ((grid >>> (8 * j)) & 0xFF);
                        float sign = ((signByte & IQGridTables.KMASK_IQ2XS[j]) != 0) ? -1.0f : 1.0f;
                        dq[dqIdx++] = dl * gv * sign;
                    }
                }

                // SIMD FMA for this group of 32
                int loopBound = F_SPECIES.loopBound(32);
                for (int j = 0; j < loopBound; j += F_LEN) {
                    FloatVector vw = FloatVector.fromArray(F_SPECIES, dq, j);
                    FloatVector vIn = FloatVector.fromArray(F_SPECIES, other, otherBase + outIdx + j);
                    acc = vw.fma(vIn, acc);
                }
                for (int j = loopBound; j < 32; j++) {
                    acc = acc.withLane(0, acc.lane(0) + dq[j] * other[otherBase + outIdx + j]);
                }
                outIdx += 32;
            }

            otherBase += BLOCK_SIZE;
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }
}
