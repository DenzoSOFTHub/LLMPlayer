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
 * SIMD-optimized Q6_K tensor using lane-parallel B2I/I2F nibble extraction.
 *
 * Rewritten 2026-04-15 after JFR showed {@code SimdQ6_KFloatTensor.dot} as
 * the #1 CPU hotspot on Llama Q4_K_M (3× more samples than {@code SimdQ4_K.dot}).
 * Previous version extracted nibbles + shifted qh bits with a scalar
 * {@code for j in F_LEN} inner loop; the new version mirrors the {@link SimdQ4_KFloatTensor}
 * pattern: read {@link ByteVector} directly from the mapped segment, widen to
 * {@link IntVector} via {@code B2I}, extract low/high nibbles and qh 2-bit pairs
 * with masked shifts, then {@code I2F} + FMA.
 *
 * Q6_K block layout (210 bytes, 256 elements):
 *   ql[128] (offset 0):   lower 4 bits of 6-bit quants
 *   qh[64]  (offset 128): upper 2 bits of 6-bit quants (four 2-bit pairs per byte)
 *   sc[16]  (offset 192): int8 sub-block scales
 *   d (fp16, offset 208): super-block scale
 */
public class SimdQ6_KFloatTensor extends Q6_KFloatTensor {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_256;   // 8 floats
    private static final VectorSpecies<Integer> I_SPECIES = IntVector.SPECIES_256;   // 8 ints
    private static final VectorSpecies<Byte> B_SPECIES = ByteVector.SPECIES_64;      // 8 bytes
    private static final int F_LEN = 8;
    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 210;
    private static final ValueLayout.OfByte BYTE_LE = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED;
    private static final ByteOrder BYTE_ORDER = ByteOrder.LITTLE_ENDIAN;

    private final MemorySegment segment;

    public SimdQ6_KFloatTensor(TensorData data, long size) {
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
        int otherBase = otherOffset;

        FloatVector acc = FloatVector.zero(F_SPECIES);
        byte[] sc = new byte[16];
        IntVector vMask4 = IntVector.broadcast(I_SPECIES, 0x0F);
        IntVector vMask2 = IntVector.broadcast(I_SPECIES, 0x03);
        IntVector vSub32 = IntVector.broadcast(I_SPECIES, 32);

        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float d = Float.float16ToFloat(segment.get(SHORT_LE, bo + 208));
            MemorySegment.copy(segment, BYTE_LE, bo + 192, sc, 0, 16);

            for (int half = 0; half < 2; half++) {
                long qlBase = bo + half * 64L;                // 64 ql bytes for this half
                long qhBase = bo + 128L + half * 32L;         // 32 qh bytes for this half
                int scBase = half * 8;
                int elemBase = otherBase + half * 128;

                for (int l = 0; l < 32; l += F_LEN) {
                    // Load 8 qh bytes shared by all 4 sub-blocks
                    ByteVector vqh = ByteVector.fromMemorySegment(B_SPECIES, segment, qhBase + l, BYTE_ORDER);
                    IntVector vqhI = (IntVector) vqh.convertShape(VectorOperators.B2I, I_SPECIES, 0);

                    // Load 16 ql bytes: 8 at ql[qlBase+l] and 8 at ql[qlBase+32+l]
                    ByteVector vqlLo = ByteVector.fromMemorySegment(B_SPECIES, segment, qlBase + l, BYTE_ORDER);
                    ByteVector vqlHi = ByteVector.fromMemorySegment(B_SPECIES, segment, qlBase + 32L + l, BYTE_ORDER);
                    IntVector vqlLoI = (IntVector) vqlLo.convertShape(VectorOperators.B2I, I_SPECIES, 0);
                    IntVector vqlHiI = (IntVector) vqlHi.convertShape(VectorOperators.B2I, I_SPECIES, 0);

                    int scIdx = l >> 4;  // 0 or 1 (since l = 0, 8, 16, 24)

                    // Sub-block 0: low nibble of ql[qlBase..+32], qh bits 0-1
                    {
                        IntVector low4 = vqlLoI.and(vMask4);
                        IntVector high2 = vqhI.and(vMask2).lanewise(VectorOperators.LSHL, 4);
                        IntVector q = low4.or(high2).sub(vSub32);
                        FloatVector vq = (FloatVector) q.convertShape(VectorOperators.I2F, F_SPECIES, 0);
                        FloatVector vds = FloatVector.broadcast(F_SPECIES, d * sc[scBase + scIdx]);
                        FloatVector in = FloatVector.fromArray(F_SPECIES, other, elemBase + l);
                        acc = vq.fma(vds.mul(in), acc);
                    }

                    // Sub-block 1: low nibble of ql[qlBase+32..+64], qh bits 2-3
                    {
                        IntVector low4 = vqlHiI.and(vMask4);
                        IntVector high2 = vqhI.lanewise(VectorOperators.LSHR, 2).and(vMask2).lanewise(VectorOperators.LSHL, 4);
                        IntVector q = low4.or(high2).sub(vSub32);
                        FloatVector vq = (FloatVector) q.convertShape(VectorOperators.I2F, F_SPECIES, 0);
                        FloatVector vds = FloatVector.broadcast(F_SPECIES, d * sc[scBase + 2 + scIdx]);
                        FloatVector in = FloatVector.fromArray(F_SPECIES, other, elemBase + 32 + l);
                        acc = vq.fma(vds.mul(in), acc);
                    }

                    // Sub-block 2: high nibble of ql[qlBase..+32], qh bits 4-5
                    {
                        IntVector low4 = vqlLoI.lanewise(VectorOperators.LSHR, 4).and(vMask4);
                        IntVector high2 = vqhI.lanewise(VectorOperators.LSHR, 4).and(vMask2).lanewise(VectorOperators.LSHL, 4);
                        IntVector q = low4.or(high2).sub(vSub32);
                        FloatVector vq = (FloatVector) q.convertShape(VectorOperators.I2F, F_SPECIES, 0);
                        FloatVector vds = FloatVector.broadcast(F_SPECIES, d * sc[scBase + 4 + scIdx]);
                        FloatVector in = FloatVector.fromArray(F_SPECIES, other, elemBase + 64 + l);
                        acc = vq.fma(vds.mul(in), acc);
                    }

                    // Sub-block 3: high nibble of ql[qlBase+32..+64], qh bits 6-7
                    {
                        IntVector low4 = vqlHiI.lanewise(VectorOperators.LSHR, 4).and(vMask4);
                        IntVector high2 = vqhI.lanewise(VectorOperators.LSHR, 6).and(vMask2).lanewise(VectorOperators.LSHL, 4);
                        IntVector q = low4.or(high2).sub(vSub32);
                        FloatVector vq = (FloatVector) q.convertShape(VectorOperators.I2F, F_SPECIES, 0);
                        FloatVector vds = FloatVector.broadcast(F_SPECIES, d * sc[scBase + 6 + scIdx]);
                        FloatVector in = FloatVector.fromArray(F_SPECIES, other, elemBase + 96 + l);
                        acc = vq.fma(vds.mul(in), acc);
                    }
                }
            }
            otherBase += BLOCK_SIZE;
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }
}
