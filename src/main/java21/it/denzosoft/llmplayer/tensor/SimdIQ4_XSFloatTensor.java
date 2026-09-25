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
 * SIMD-optimized IQ4_XS tensor with fused dequantization and dot product.
 * Eliminates ThreadLocal overhead, VectorOpsFactory dispatch, and intermediate buffer.
 *
 * IQ4_XS block layout (136 bytes, 256 elements):
 *   d (fp16, 2 bytes): super-block scale
 *   scales_h (uint16, 2 bytes): high 2 bits of 8 sub-block scales
 *   scales_l (4 bytes): low 4 bits of 8 sub-block scales, packed 2 per byte
 *   qs (128 bytes): 256 x 4-bit nibbles (non-linear lookup)
 */
public class SimdIQ4_XSFloatTensor extends IQ4_XSFloatTensor {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int F_LEN = F_SPECIES.length();
    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 136;
    private static final ValueLayout.OfByte BYTE_LE = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED;
    private static final ValueLayout.OfInt INT_LE = ValueLayout.JAVA_INT_UNALIGNED;

    private final MemorySegment segment;

    public SimdIQ4_XSFloatTensor(TensorData data, long size) {
        super(data, size);
        this.segment = ((MemorySegmentTensorData) data).segment();
    }

    private static final VectorSpecies<Float> F256 = FloatVector.SPECIES_256;
    private static final VectorSpecies<Integer> I256 = IntVector.SPECIES_256;
    private static final VectorSpecies<Byte> B64 = ByteVector.SPECIES_64;

    /**
     * IQ4_XS dot product with the in-register codebook lookup of {@link SimdIQ4_NLFloatTensor#dot}:
     * each 32-weight sub-block is the IQ4_NL layout (low nibbles are elements 0..15, high nibbles
     * 16..31) with a 6-bit scale, so the raw {@code codebook[q]·x} is formed per sub-block and
     * {@code d·(ls − 32)} applied once. Replaces a per-sub-block scalar table build plus gather.
     * The codebook vectors are built here and passed down, as in IQ4_NL (see the note there).
     */
    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        if (FloatVector.SPECIES_PREFERRED.length() < 8 || length % BLOCK_SIZE != 0) {
            return super.dot(thisOffset, other, otherOffset, length);
        }
        IntVector lo = IntVector.fromArray(I256, SimdIQ4_NLFloatTensor.CODEBOOK_LO, 0);
        IntVector hi = IntVector.fromArray(I256, SimdIQ4_NLFloatTensor.CODEBOOK_HI, 0);
        int numBlocks = length / BLOCK_SIZE;
        long blockStart = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        FloatVector acc0 = FloatVector.zero(F256);
        FloatVector acc1 = FloatVector.zero(F256);
        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float d = Float.float16ToFloat(segment.get(SHORT_LE, bo));
            int scalesH = Short.toUnsignedInt(segment.get(SHORT_LE, bo + 2));
            int scalesL = segment.get(INT_LE, bo + 4);
            int xBase = otherOffset + b * BLOCK_SIZE;
            for (int ib = 0; ib < 8; ib++) {
                int ls = ((scalesL >>> (4 * ib)) & 0x0F) | (((scalesH >>> (2 * ib)) & 3) << 4);
                long qo = bo + 8 + ib * 16L;
                int xo = xBase + ib * 32;
                IntVector q0 = (IntVector) ByteVector.fromMemorySegment(B64, segment, qo, ByteOrder.LITTLE_ENDIAN)
                    .convertShape(VectorOperators.B2I, I256, 0);
                IntVector q1 = (IntVector) ByteVector.fromMemorySegment(B64, segment, qo + 8, ByteOrder.LITTLE_ENDIAN)
                    .convertShape(VectorOperators.B2I, I256, 0);
                FloatVector p0 = SimdIQ4_NLFloatTensor.lookup(q0.and(15), lo, hi).mul(FloatVector.fromArray(F256, other, xo));
                FloatVector p1 = SimdIQ4_NLFloatTensor.lookup(q1.and(15), lo, hi).mul(FloatVector.fromArray(F256, other, xo + 8));
                p0 = SimdIQ4_NLFloatTensor.lookup(q0.lanewise(VectorOperators.LSHR, 4).and(15), lo, hi)
                    .fma(FloatVector.fromArray(F256, other, xo + 16), p0);
                p1 = SimdIQ4_NLFloatTensor.lookup(q1.lanewise(VectorOperators.LSHR, 4).and(15), lo, hi)
                    .fma(FloatVector.fromArray(F256, other, xo + 24), p1);
                FloatVector dl = FloatVector.broadcast(F256, d * (ls - 32));
                if ((ib & 1) == 0) {
                    acc0 = p0.add(p1).fma(dl, acc0);
                } else {
                    acc1 = p0.add(p1).fma(dl, acc1);
                }
            }
        }
        return acc0.add(acc1).reduceLanes(VectorOperators.ADD);
    }
}
