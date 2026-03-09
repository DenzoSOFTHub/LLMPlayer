package it.denzosoft.llmplayer.tensor;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * SIMD-optimized IQ3_XXS tensor with fused dequantization and dot product.
 * Eliminates ThreadLocal overhead, VectorOpsFactory dispatch, and intermediate buffer.
 *
 * IQ3_XXS block layout (98 bytes, 256 elements):
 *   d (fp16, 2 bytes): super-block scale
 *   qs (64 bytes): grid indices (8-bit each, 8 groups × 8 indices)
 *   scales_and_signs (32 bytes): 8 × uint32 (4-bit scale + 4 × 7-bit sign indices)
 */
public class SimdIQ3_XXSFloatTensor extends IQ3_XXSFloatTensor {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int F_LEN = F_SPECIES.length();
    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 98;
    private static final ValueLayout.OfByte BYTE_LE = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED;

    private final MemorySegment segment;

    public SimdIQ3_XXSFloatTensor(TensorData data, long size) {
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
        byte[] sasBytes = new byte[32];
        float[] dq = new float[32]; // dequantized group of 32

        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float d = Float.float16ToFloat(segment.get(SHORT_LE, bo));

            // Bulk copy all block data
            MemorySegment.copy(segment, BYTE_LE, bo + 2, qsBytes, 0, 64);
            MemorySegment.copy(segment, BYTE_LE, bo + 66, sasBytes, 0, 32);

            int outIdx = 0;
            for (int ib32 = 0; ib32 < 8; ib32++) {
                // Read uint32 from sasBytes
                int aux32 = (Byte.toUnsignedInt(sasBytes[ib32 * 4]))
                          | (Byte.toUnsignedInt(sasBytes[ib32 * 4 + 1]) << 8)
                          | (Byte.toUnsignedInt(sasBytes[ib32 * 4 + 2]) << 16)
                          | (Byte.toUnsignedInt(sasBytes[ib32 * 4 + 3]) << 24);
                float db = d * (0.5f + (aux32 >>> 28)) * 0.5f;

                int dqIdx = 0;
                for (int l = 0; l < 4; l++) {
                    int signIdx = (aux32 >>> (7 * l)) & 0x7F;
                    int signs = IQGridTables.KSIGNS_IQ2XS[signIdx] & 0xFF;

                    int gridIdx1 = Byte.toUnsignedInt(qsBytes[ib32 * 8 + 2 * l]);
                    int gridIdx2 = Byte.toUnsignedInt(qsBytes[ib32 * 8 + 2 * l + 1]);
                    int grid1 = IQGridTables.IQ3XXS_GRID[gridIdx1];
                    int grid2 = IQGridTables.IQ3XXS_GRID[gridIdx2];

                    for (int j = 0; j < 4; j++) {
                        int gv = (grid1 >>> (8 * j)) & 0xFF;
                        float sign = ((signs & IQGridTables.KMASK_IQ2XS[j]) != 0) ? -1.0f : 1.0f;
                        dq[dqIdx++] = db * gv * sign;
                    }
                    for (int j = 0; j < 4; j++) {
                        int gv = (grid2 >>> (8 * j)) & 0xFF;
                        float sign = ((signs & IQGridTables.KMASK_IQ2XS[j + 4]) != 0) ? -1.0f : 1.0f;
                        dq[dqIdx++] = db * gv * sign;
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
