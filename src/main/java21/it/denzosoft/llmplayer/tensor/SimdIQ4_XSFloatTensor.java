package it.denzosoft.llmplayer.tensor;

import jdk.incubator.vector.FloatVector;
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

    private final MemorySegment segment;

    public SimdIQ4_XSFloatTensor(TensorData data, long size) {
        super(data, size);
        this.segment = ((MemorySegmentTensorData) data).segment();
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        int numBlocks = length / BLOCK_SIZE;
        long blockStart = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        int otherBase = otherOffset;

        // 16 is block-clean only for F_LEN in {4,8,16}; fall back otherwise.
        if ((16 % F_LEN) != 0) return super.dot(thisOffset, other, otherOffset, length);

        FloatVector acc = FloatVector.zero(F_SPECIES);
        final byte[] qs = new byte[16];   // per sub-block packed nibbles
        final byte[] scl = new byte[4];   // scales_l bytes
        final float[] sk = new float[16]; // per-sub-block scaled codebook (dl*KVALUES[i])
        final int[] lo = new int[16];
        final int[] hi = new int[16];
        final float[] KV = IQ4_NLFloatTensor.KVALUES_IQ4NL;

        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float d = Float.float16ToFloat(segment.get(SHORT_LE, bo));
            int scalesH = Short.toUnsignedInt(segment.get(SHORT_LE, bo + 2));
            MemorySegment.copy(segment, BYTE_LE, bo + 4, scl, 0, 4);

            for (int ib = 0; ib < 8; ib++) {
                // Reconstruct 6-bit sub-block scale
                int scalesLByte = Byte.toUnsignedInt(scl[ib / 2]);
                int low4 = (ib % 2 == 0) ? (scalesLByte & 0x0F) : ((scalesLByte >> 4) & 0x0F);
                int high2 = (scalesH >> (2 * ib)) & 3;
                int ls = low4 | (high2 << 4);
                float dl = d * (ls - 32);

                MemorySegment.copy(segment, BYTE_LE, bo + 8 + (long) ib * 16, qs, 0, 16);

                // Scaled codebook + lane-parallel nibble split, then SIMD gather
                for (int i = 0; i < 16; i++) {
                    sk[i] = dl * KV[i];
                    int v = qs[i] & 0xFF;
                    lo[i] = v & 0x0F;
                    hi[i] = v >>> 4;
                }

                int elemBase = otherBase + ib * 32;
                for (int j = 0; j < 16; j += F_LEN) {
                    FloatVector vw = FloatVector.fromArray(F_SPECIES, sk, 0, lo, j);
                    FloatVector vIn = FloatVector.fromArray(F_SPECIES, other, elemBase + j);
                    acc = vw.fma(vIn, acc);
                }
                for (int j = 0; j < 16; j += F_LEN) {
                    FloatVector vw = FloatVector.fromArray(F_SPECIES, sk, 0, hi, j);
                    FloatVector vIn = FloatVector.fromArray(F_SPECIES, other, elemBase + 16 + j);
                    acc = vw.fma(vIn, acc);
                }
            }
            otherBase += BLOCK_SIZE;
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }
}
