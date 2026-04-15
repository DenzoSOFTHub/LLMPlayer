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
 * SIMD-optimized Q5_K tensor using lane-parallel B2I/I2F nibble + qh extraction.
 *
 * Rewritten 2026-04-15 — Qwen3.5-4B JFR showed {@code SimdQ5_KFloatTensor.dot}
 * at 4208 samples (2× Q4_K). The old kernel extracted nibbles + qh bits in a
 * scalar {@code for j in F_LEN} inner loop; this version reads {@link ByteVector}
 * directly from the mapped segment, widens via {@code B2I}, does all masking
 * and shifting lane-parallel, then {@code I2F} + FMA.
 *
 * Q5_K block layout (176 bytes, 256 elements):
 *   d    (fp16, 2 bytes): super-block scale
 *   dmin (fp16, 2 bytes): super-block minimum
 *   sb[12] (offset 4): 8×(6-bit scale) + 8×(6-bit min) packed like Q4_K
 *   qh[32] (offset 16): 1 high bit per element (2 bits per group's two halves)
 *   qs[128] (offset 48): 256 × 4-bit low quants
 */
public class SimdQ5_KFloatTensor extends Q5_KFloatTensor {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_256;   // 8 floats
    private static final VectorSpecies<Integer> I_SPECIES = IntVector.SPECIES_256;   // 8 ints
    private static final VectorSpecies<Byte> B_SPECIES = ByteVector.SPECIES_64;      // 8 bytes
    private static final int F_LEN = 8;
    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 176;
    private static final ValueLayout.OfByte BYTE_LE = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED;
    private static final ByteOrder BYTE_ORDER = ByteOrder.LITTLE_ENDIAN;

    private final MemorySegment segment;

    public SimdQ5_KFloatTensor(TensorData data, long size) {
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
        byte[] sb = new byte[12];
        int[] scales = new int[8];
        int[] mins = new int[8];
        IntVector vMask4 = IntVector.broadcast(I_SPECIES, 0x0F);
        IntVector vMask1 = IntVector.broadcast(I_SPECIES, 0x01);

        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float d = Float.float16ToFloat(segment.get(SHORT_LE, bo));
            float dmin = Float.float16ToFloat(segment.get(SHORT_LE, bo + 2));

            MemorySegment.copy(segment, BYTE_LE, bo + 4, sb, 0, 12);

            // Decode 8 scales + 8 mins (6 bits each, packed Q4_K-style)
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

            long qhBase = bo + 16;
            long qsBase = bo + 48;

            for (int group = 0; group < 4; group++) {
                float ds0 = d * scales[group * 2];
                float negDm0 = -(dmin * mins[group * 2]);
                float ds1 = d * scales[group * 2 + 1];
                float negDm1 = -(dmin * mins[group * 2 + 1]);
                FloatVector vds0 = FloatVector.broadcast(F_SPECIES, ds0);
                FloatVector vNegDm0 = FloatVector.broadcast(F_SPECIES, negDm0);
                FloatVector vds1 = FloatVector.broadcast(F_SPECIES, ds1);
                FloatVector vNegDm1 = FloatVector.broadcast(F_SPECIES, negDm1);

                long qsGroup = qsBase + (long) group * 32;
                int lowInputBase = otherBase + group * 64;
                int highInputBase = lowInputBase + 32;
                int qhShiftLow = group * 2;
                int qhShiftHigh = group * 2 + 1;

                for (int l = 0; l < 32; l += F_LEN) {
                    // Load 8 qs bytes and 8 qh bytes as ByteVectors, widen to IntVectors
                    ByteVector vqs = ByteVector.fromMemorySegment(B_SPECIES, segment, qsGroup + l, BYTE_ORDER);
                    ByteVector vqh = ByteVector.fromMemorySegment(B_SPECIES, segment, qhBase + l, BYTE_ORDER);
                    IntVector vqsI = (IntVector) vqs.convertShape(VectorOperators.B2I, I_SPECIES, 0);
                    IntVector vqhI = (IntVector) vqh.convertShape(VectorOperators.B2I, I_SPECIES, 0);

                    // Low nibbles + qh bit shift group*2
                    IntVector ql0 = vqsI.and(vMask4);
                    IntVector qh0 = vqhI.lanewise(VectorOperators.LSHR, qhShiftLow).and(vMask1).lanewise(VectorOperators.LSHL, 4);
                    IntVector q0 = ql0.or(qh0);
                    FloatVector vq0 = (FloatVector) q0.convertShape(VectorOperators.I2F, F_SPECIES, 0);
                    FloatVector w0 = vq0.fma(vds0, vNegDm0);
                    FloatVector in0 = FloatVector.fromArray(F_SPECIES, other, lowInputBase + l);
                    acc = w0.fma(in0, acc);

                    // High nibbles + qh bit shift group*2+1
                    IntVector ql1 = vqsI.lanewise(VectorOperators.LSHR, 4).and(vMask4);
                    IntVector qh1 = vqhI.lanewise(VectorOperators.LSHR, qhShiftHigh).and(vMask1).lanewise(VectorOperators.LSHL, 4);
                    IntVector q1 = ql1.or(qh1);
                    FloatVector vq1 = (FloatVector) q1.convertShape(VectorOperators.I2F, F_SPECIES, 0);
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
