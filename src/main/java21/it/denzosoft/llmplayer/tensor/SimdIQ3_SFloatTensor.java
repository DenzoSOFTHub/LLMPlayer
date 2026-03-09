package it.denzosoft.llmplayer.tensor;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * SIMD-optimized IQ3_S tensor with fused dequantization and dot product.
 * Eliminates ThreadLocal overhead, VectorOpsFactory dispatch, and intermediate buffer.
 *
 * IQ3_S block layout (110 bytes, 256 elements):
 *   d (fp16, 2 bytes)
 *   qs (64 bytes): grid indices low 8 bits
 *   qh (8 bytes): grid index high 1 bit (9th bit)
 *   signs (32 bytes): sign bits, 1 per weight
 *   scales (4 bytes): 4-bit sub-block scales, 2 per byte
 */
public class SimdIQ3_SFloatTensor extends IQ3_SFloatTensor {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int F_LEN = F_SPECIES.length();
    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 110;
    private static final ValueLayout.OfByte BYTE_LE = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED;

    private final MemorySegment segment;

    public SimdIQ3_SFloatTensor(TensorData data, long size) {
        super(data, size);
        this.segment = ((MemorySegmentTensorData) data).segment();
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        int numBlocks = length / BLOCK_SIZE;
        long blockStart = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        int otherBase = otherOffset;

        FloatVector acc = FloatVector.zero(F_SPECIES);
        byte[] qsBytes = new byte[64];
        byte[] qhBytes = new byte[8];
        byte[] signBytes = new byte[32];
        byte[] scaleBytes = new byte[4];
        float[] dq = new float[32];

        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float d = Float.float16ToFloat(segment.get(SHORT_LE, bo));

            // Bulk copy all block data
            MemorySegment.copy(segment, BYTE_LE, bo + 2, qsBytes, 0, 64);
            MemorySegment.copy(segment, BYTE_LE, bo + 66, qhBytes, 0, 8);
            MemorySegment.copy(segment, BYTE_LE, bo + 74, signBytes, 0, 32);
            MemorySegment.copy(segment, BYTE_LE, bo + 106, scaleBytes, 0, 4);

            int outIdx = 0;

            // Process in pairs of 32-weight groups (64 weights at a time)
            for (int ib32 = 0; ib32 < 8; ib32 += 2) {
                int scaleByte = Byte.toUnsignedInt(scaleBytes[ib32 / 2]);
                float db1 = d * (1 + 2 * (scaleByte & 0x0F));
                float db2 = d * (1 + 2 * ((scaleByte >> 4) & 0x0F));

                // First 32 weights of pair
                int qhByte0 = Byte.toUnsignedInt(qhBytes[ib32 / 2]);
                int qsBase1 = ib32 * 8;
                int signBase1 = ib32 * 4;

                int dqIdx = 0;
                for (int l = 0; l < 4; l++) {
                    int qs0 = Byte.toUnsignedInt(qsBytes[qsBase1 + 2 * l]);
                    int qs1 = Byte.toUnsignedInt(qsBytes[qsBase1 + 2 * l + 1]);
                    int grid1 = IQGridTables.IQ3S_GRID[qs0 | ((qhByte0 << (8 - 2 * l)) & 256)];
                    int grid2 = IQGridTables.IQ3S_GRID[qs1 | ((qhByte0 << (7 - 2 * l)) & 256)];
                    int signs = Byte.toUnsignedInt(signBytes[signBase1 + l]);

                    for (int j = 0; j < 4; j++) {
                        int gv = (grid1 >>> (8 * j)) & 0xFF;
                        float sign = ((signs & IQGridTables.KMASK_IQ2XS[j]) != 0) ? -1.0f : 1.0f;
                        dq[dqIdx++] = db1 * gv * sign;
                    }
                    for (int j = 0; j < 4; j++) {
                        int gv = (grid2 >>> (8 * j)) & 0xFF;
                        float sign = ((signs & IQGridTables.KMASK_IQ2XS[j + 4]) != 0) ? -1.0f : 1.0f;
                        dq[dqIdx++] = db1 * gv * sign;
                    }
                }

                // SIMD FMA for first 32
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

                // Second 32 weights of pair
                int qhByte1 = Byte.toUnsignedInt(qhBytes[ib32 / 2 + 1]);
                int qsBase2 = (ib32 + 1) * 8;
                int signBase2 = (ib32 + 1) * 4;

                dqIdx = 0;
                for (int l = 0; l < 4; l++) {
                    int qs0 = Byte.toUnsignedInt(qsBytes[qsBase2 + 2 * l]);
                    int qs1 = Byte.toUnsignedInt(qsBytes[qsBase2 + 2 * l + 1]);
                    int grid1 = IQGridTables.IQ3S_GRID[qs0 | ((qhByte1 << (8 - 2 * l)) & 256)];
                    int grid2 = IQGridTables.IQ3S_GRID[qs1 | ((qhByte1 << (7 - 2 * l)) & 256)];
                    int signs = Byte.toUnsignedInt(signBytes[signBase2 + l]);

                    for (int j = 0; j < 4; j++) {
                        int gv = (grid1 >>> (8 * j)) & 0xFF;
                        float sign = ((signs & IQGridTables.KMASK_IQ2XS[j]) != 0) ? -1.0f : 1.0f;
                        dq[dqIdx++] = db2 * gv * sign;
                    }
                    for (int j = 0; j < 4; j++) {
                        int gv = (grid2 >>> (8 * j)) & 0xFF;
                        float sign = ((signs & IQGridTables.KMASK_IQ2XS[j + 4]) != 0) ? -1.0f : 1.0f;
                        dq[dqIdx++] = db2 * gv * sign;
                    }
                }

                // SIMD FMA for second 32
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
