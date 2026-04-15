package it.denzosoft.llmplayer.tensor;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * SIMD-optimized Q5_0 tensor using lane-parallel B2I/I2F path.
 *
 * Rewritten 2026-04-15 — critical for Gemma-3 models which use Q5_0 for Q/K/gate/up.
 * Old version extracted nibbles + qh bits with a scalar {@code for j} loop; this version
 * uses {@link ByteVector} + {@code B2I} + lane-wise masks/shifts.
 *
 * Q5_0 block layout (22 bytes, 32 elements):
 *   scale (fp16, 2 bytes)
 *   qh (uint32, 4 bytes): high bit for each of 32 elements
 *   qs[16]: packed 4-bit low quants
 * Element mapping (SPLIT layout):
 *   Elements 0..15  = LOW nibbles of bytes 0..15, qh bits 0..15
 *   Elements 16..31 = HIGH nibbles of bytes 0..15, qh bits 16..31
 */
public class SimdQ5_0FloatTensor extends Q5_0FloatTensor {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_256;
    private static final VectorSpecies<Integer> I_SPECIES = IntVector.SPECIES_256;
    private static final VectorSpecies<Byte> B_SPECIES = ByteVector.SPECIES_64;
    private static final int F_LEN = 8;
    private static final int BLOCK_SIZE = 32;
    private static final int BLOCK_BYTES = 22;
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED;
    private static final ValueLayout.OfInt INT_LE = ValueLayout.JAVA_INT_UNALIGNED;
    private static final ByteOrder BYTE_ORDER = ByteOrder.LITTLE_ENDIAN;

    private final MemorySegment segment;

    public SimdQ5_0FloatTensor(TensorData data, long size) {
        super(data, size);
        this.segment = ((MemorySegmentTensorData) data).segment();
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        if (FloatVector.SPECIES_PREFERRED.length() != 8 || length % BLOCK_SIZE != 0) {
            return super.dot(thisOffset, other, otherOffset, length);
        }

        int numBlocks = length / BLOCK_SIZE;
        long blockStart = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        int otherIdx = otherOffset;

        FloatVector acc = FloatVector.zero(F_SPECIES);
        IntVector vMask4 = IntVector.broadcast(I_SPECIES, 0x0F);
        IntVector vMask1 = IntVector.broadcast(I_SPECIES, 0x01);
        IntVector vSub16 = IntVector.broadcast(I_SPECIES, 16);

        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float scale = Float.float16ToFloat(segment.get(SHORT_LE, bo));
            int qh = segment.get(INT_LE, bo + 2);
            long qsBase = bo + 6;
            FloatVector vScale = FloatVector.broadcast(F_SPECIES, scale);

            // Process 16 qs bytes in 2 iterations of 8 bytes each
            // Each iteration handles 8 elements in LOW half + 8 elements in HIGH half
            for (int j = 0; j < 16; j += F_LEN) {
                ByteVector vqs = ByteVector.fromMemorySegment(B_SPECIES, segment, qsBase + j, BYTE_ORDER);
                IntVector vqsI = (IntVector) vqs.convertShape(VectorOperators.B2I, I_SPECIES, 0);

                // Extract 8 qh bits starting at position j (low elements 0-15)
                int qhLowBits = (qh >>> j) & 0xFF;
                // Build qhLo as 8 ints, each bit isolated
                IntVector qhLo = buildQhBits(qhLowBits, vMask1);
                // Extract 8 qh bits starting at position j+16 (high elements 16-31)
                int qhHighBits = (qh >>> (j + 16)) & 0xFF;
                IntVector qhHi = buildQhBits(qhHighBits, vMask1);

                // Low half: elements j..j+7 (base 0)
                IntVector lo4 = vqsI.and(vMask4);
                IntVector q0 = lo4.or(qhLo.lanewise(VectorOperators.LSHL, 4)).sub(vSub16);
                FloatVector vq0 = (FloatVector) q0.convertShape(VectorOperators.I2F, F_SPECIES, 0);
                FloatVector in0 = FloatVector.fromArray(F_SPECIES, other, otherIdx + j);
                acc = vq0.fma(vScale.mul(in0), acc);

                // High half: elements 16+j..16+j+7 (base 16)
                IntVector hi4 = vqsI.lanewise(VectorOperators.LSHR, 4).and(vMask4);
                IntVector q1 = hi4.or(qhHi.lanewise(VectorOperators.LSHL, 4)).sub(vSub16);
                FloatVector vq1 = (FloatVector) q1.convertShape(VectorOperators.I2F, F_SPECIES, 0);
                FloatVector in1 = FloatVector.fromArray(F_SPECIES, other, otherIdx + 16 + j);
                acc = vq1.fma(vScale.mul(in1), acc);
            }
            otherIdx += BLOCK_SIZE;
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    /**
     * Build an 8-lane IntVector where lane i = bit i of the given 8-bit value.
     * Uses a constant vector of shifts [0,1,2,3,4,5,6,7] + lane-wise LSHR + AND 1.
     */
    private static final int[] BIT_SHIFTS = {0, 1, 2, 3, 4, 5, 6, 7};
    private static final IntVector V_SHIFTS = IntVector.fromArray(I_SPECIES, BIT_SHIFTS, 0);

    private static IntVector buildQhBits(int eightBits, IntVector vMask1) {
        IntVector broadcast = IntVector.broadcast(I_SPECIES, eightBits);
        return broadcast.lanewise(VectorOperators.LSHR, V_SHIFTS).and(vMask1);
    }
}
