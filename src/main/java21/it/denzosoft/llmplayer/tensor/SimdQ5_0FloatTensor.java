package it.denzosoft.llmplayer.tensor;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * SIMD-optimized Q5_0 tensor with fused dequantization and dot product.
 * Critical for Gemma 3 models which use Q5_0 for Q, K, gate, up projections.
 *
 * Q5_0 block layout (22 bytes, 32 elements):
 *   scale (fp16, 2 bytes)
 *   qh (uint32, 4 bytes): high bits for all 32 elements
 *   qs[16]: packed 4-bit low quants
 * Element mapping (SPLIT layout):
 *   Elements 0..15  = LOW nibbles of bytes 0..15, qh bits 0..15
 *   Elements 16..31 = HIGH nibbles of bytes 0..15, qh bits 16..31
 */
public class SimdQ5_0FloatTensor extends Q5_0FloatTensor {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int F_LEN = F_SPECIES.length();
    private static final int BLOCK_SIZE = 32;
    private static final int BLOCK_BYTES = 22;
    private static final ValueLayout.OfByte BYTE_LE = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED;
    private static final ValueLayout.OfInt INT_LE = ValueLayout.JAVA_INT_UNALIGNED;

    private final MemorySegment segment;

    public SimdQ5_0FloatTensor(TensorData data, long size) {
        super(data, size);
        this.segment = ((MemorySegmentTensorData) data).segment();
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        int numBlocks = length / BLOCK_SIZE;
        long blockStart = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        int otherIdx = otherOffset;

        FloatVector acc = FloatVector.zero(F_SPECIES);
        float[] dqLo = new float[16]; // dequantized elements 0..15
        float[] dqHi = new float[16]; // dequantized elements 16..31
        byte[] qs = new byte[16];     // bulk-copied packed nibbles

        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float scale = Float.float16ToFloat(segment.get(SHORT_LE, bo));
            int qh = segment.get(INT_LE, bo + 2);

            // Bulk copy 16 packed bytes (one bounds check instead of 16)
            MemorySegment.copy(segment, BYTE_LE, bo + 6, qs, 0, 16);

            // Dequantize all 32 elements in one pass
            for (int j = 0; j < 16; j++) {
                int lo4 = qs[j] & 0x0F;
                int hi4 = (qs[j] >> 4) & 0x0F;
                int hbitLo = (qh >> j) & 1;
                int hbitHi = (qh >> (j + 16)) & 1;
                dqLo[j] = ((lo4 | (hbitLo << 4)) - 16) * scale;
                dqHi[j] = ((hi4 | (hbitHi << 4)) - 16) * scale;
            }

            // SIMD FMA for elements 0..15
            int loopBound = F_SPECIES.loopBound(16);
            for (int j = 0; j < loopBound; j += F_LEN) {
                FloatVector vw = FloatVector.fromArray(F_SPECIES, dqLo, j);
                FloatVector vIn = FloatVector.fromArray(F_SPECIES, other, otherIdx + j);
                acc = vw.fma(vIn, acc);
            }
            // Scalar tail for elements 0..15
            for (int j = loopBound; j < 16; j++) {
                acc = acc.withLane(0, acc.lane(0) + dqLo[j] * other[otherIdx + j]);
            }

            // SIMD FMA for elements 16..31
            for (int j = 0; j < loopBound; j += F_LEN) {
                FloatVector vw = FloatVector.fromArray(F_SPECIES, dqHi, j);
                FloatVector vIn = FloatVector.fromArray(F_SPECIES, other, otherIdx + 16 + j);
                acc = vw.fma(vIn, acc);
            }
            // Scalar tail for elements 16..31
            for (int j = loopBound; j < 16; j++) {
                acc = acc.withLane(0, acc.lane(0) + dqHi[j] * other[otherIdx + 16 + j]);
            }

            otherIdx += BLOCK_SIZE;
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }
}
