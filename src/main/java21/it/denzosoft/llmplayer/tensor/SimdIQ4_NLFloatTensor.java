package it.denzosoft.llmplayer.tensor;

import jdk.incubator.vector.FloatVector;
import java.nio.ByteOrder;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * SIMD-optimized IQ4_NL tensor with fused dequantization and dot product.
 * Eliminates ThreadLocal overhead, VectorOpsFactory dispatch, and intermediate buffer.
 *
 * IQ4_NL block layout (18 bytes, 32 elements):
 *   scale (fp16, 2 bytes)
 *   qs[16]: packed 4-bit nibbles (non-linear lookup)
 * Split nibble layout: low nibbles → elements 0-15, high nibbles → elements 16-31
 */
public class SimdIQ4_NLFloatTensor extends IQ4_NLFloatTensor {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int F_LEN = F_SPECIES.length();
    private static final int BLOCK_SIZE = 32;
    private static final int BLOCK_BYTES = 18;
    private static final ValueLayout.OfByte BYTE_LE = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED;

    private final MemorySegment segment;

    public SimdIQ4_NLFloatTensor(TensorData data, long size) {
        super(data, size);
        this.segment = ((MemorySegmentTensorData) data).segment();
    }

    // Codebook split in two 8-entry halves, stored as float bit patterns so a lane-wise
    // selectFrom (vpermd) yields the float value directly, without an int-to-float conversion.
    private static final VectorSpecies<Float> F256 = FloatVector.SPECIES_256;
    private static final VectorSpecies<Integer> I256 = IntVector.SPECIES_256;
    private static final VectorSpecies<Byte> B64 = ByteVector.SPECIES_64;
    static final int[] CODEBOOK_LO = new int[8];
    static final int[] CODEBOOK_HI = new int[8];
    static {
        for (int i = 0; i < 8; i++) {
            CODEBOOK_LO[i] = Float.floatToIntBits(KVALUES_IQ4NL[i]);
            CODEBOOK_HI[i] = Float.floatToIntBits(KVALUES_IQ4NL[i + 8]);
        }
    }

    /**
     * IQ4_NL dot product with an in-register codebook lookup. Each 4-bit index selects from two
     * 8-lane halves of the codebook ({@code selectFrom}, i.e. {@code vpermd}), blended on bit 3, so
     * the weights come straight out as floats. The raw {@code codebook[q]·x} of a block is formed in
     * two short chains and the block scale applied once per 32 weights. Measured 5.8–7.2 against
     * 1.0–1.3 Gelem/s for the previous per-block scalar table build plus gather (6 of 6 runs).
     *
     * <p>Keep the codebook vectors built from arrays inside this method and passed down: when they
     * were {@code static final IntVector} fields, C2 failed to intrinsify {@code selectFrom}
     * ("missing constant") and the kernel fell to 0.5 Gelem/s.
     */
    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        if (FloatVector.SPECIES_PREFERRED.length() < 8 || length % BLOCK_SIZE != 0) {
            return super.dot(thisOffset, other, otherOffset, length);
        }
        IntVector lo = IntVector.fromArray(I256, CODEBOOK_LO, 0);
        IntVector hi = IntVector.fromArray(I256, CODEBOOK_HI, 0);
        int numBlocks = length / BLOCK_SIZE;
        long blockStart = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        FloatVector acc0 = FloatVector.zero(F256);
        FloatVector acc1 = FloatVector.zero(F256);
        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            int xo = otherOffset + b * BLOCK_SIZE;
            IntVector q0 = (IntVector) ByteVector.fromMemorySegment(B64, segment, bo + 2, ByteOrder.LITTLE_ENDIAN)
                .convertShape(VectorOperators.B2I, I256, 0);
            IntVector q1 = (IntVector) ByteVector.fromMemorySegment(B64, segment, bo + 10, ByteOrder.LITTLE_ENDIAN)
                .convertShape(VectorOperators.B2I, I256, 0);
            // Low nibbles are elements 0..15, high nibbles 16..31.
            FloatVector p0 = lookup(q0.and(15), lo, hi).mul(FloatVector.fromArray(F256, other, xo));
            FloatVector p1 = lookup(q1.and(15), lo, hi).mul(FloatVector.fromArray(F256, other, xo + 8));
            p0 = lookup(q0.lanewise(VectorOperators.LSHR, 4).and(15), lo, hi).fma(FloatVector.fromArray(F256, other, xo + 16), p0);
            p1 = lookup(q1.lanewise(VectorOperators.LSHR, 4).and(15), lo, hi).fma(FloatVector.fromArray(F256, other, xo + 24), p1);
            FloatVector d = FloatVector.broadcast(F256, Float.float16ToFloat(segment.get(SHORT_LE, bo)));
            if ((b & 1) == 0) {
                acc0 = p0.add(p1).fma(d, acc0);
            } else {
                acc1 = p0.add(p1).fma(d, acc1);
            }
        }
        return acc0.add(acc1).reduceLanes(VectorOperators.ADD);
    }

    /** Codebook value for each 4-bit index lane; also used by {@link SimdIQ4_XSFloatTensor}. */
    static FloatVector lookup(IntVector idx, IntVector lo, IntVector hi) {
        IntVector i7 = idx.and(7);
        return i7.selectFrom(lo).blend(i7.selectFrom(hi), idx.compare(VectorOperators.GT, 7)).reinterpretAsFloats();
    }
}
