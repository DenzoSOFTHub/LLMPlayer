package it.denzosoft.llmplayer.tensor;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * SIMD-optimized Q6_K tensor with fused dequantization and dot product.
 * Eliminates ThreadLocal overhead (4 lookups/dot), intermediate tmp[256] buffer,
 * and VectorOpsFactory.get() dispatch per block.
 *
 * Q6_K block layout (210 bytes, 256 elements):
 *   ql[128]: lower 4 bits of 6-bit quants
 *   qh[64]:  upper 2 bits of 6-bit quants
 *   scales[16]: int8 sub-block scales
 *   d (fp16, 2 bytes): super-block scale at offset 208
 */
public class SimdQ6_KFloatTensor extends Q6_KFloatTensor {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int F_LEN = F_SPECIES.length();
    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 210;
    private static final ValueLayout.OfByte BYTE_LE = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED;

    private final MemorySegment segment;

    public SimdQ6_KFloatTensor(TensorData data, long size) {
        super(data, size);
        this.segment = ((MemorySegmentTensorData) data).segment();
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        if (32 % F_LEN != 0 || length % BLOCK_SIZE != 0) {
            return super.dot(thisOffset, other, otherOffset, length);
        }

        int numBlocks = length / BLOCK_SIZE;
        long blockStart = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        int otherBase = otherOffset;

        FloatVector acc = FloatVector.zero(F_SPECIES);
        byte[] ql = new byte[128];
        byte[] qh = new byte[64];
        byte[] sc = new byte[16];
        float[] dq = new float[F_LEN];

        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float d = Float.float16ToFloat(segment.get(SHORT_LE, bo + 208));

            MemorySegment.copy(segment, BYTE_LE, bo, ql, 0, 128);
            MemorySegment.copy(segment, BYTE_LE, bo + 128, qh, 0, 64);
            MemorySegment.copy(segment, BYTE_LE, bo + 192, sc, 0, 16);

            // Process 2 halves × 4 sub-blocks, each sub-block with SIMD FMA.
            // Instead of fusing 4 quadrants per element (scalar), we process
            // each sub-block independently with vectorized multiply-accumulate.
            for (int half = 0; half < 2; half++) {
                int qlOff = half * 64;
                int qhOff = half * 32;
                int scBase = half * 8;
                int elemBase = otherBase + half * 128;

                // Sub-block 0: low nibbles of ql[0..31], qh bits 0-1
                for (int l = 0; l < 32; l += F_LEN) {
                    FloatVector vds = FloatVector.broadcast(F_SPECIES, d * sc[scBase + (l >> 4)]);
                    for (int j = 0; j < F_LEN; j++) {
                        int idx = l + j;
                        dq[j] = (float) (((Byte.toUnsignedInt(ql[qlOff + idx]) & 0x0F)
                                | (((Byte.toUnsignedInt(qh[qhOff + idx])) & 3) << 4)) - 32);
                    }
                    FloatVector vq = FloatVector.fromArray(F_SPECIES, dq, 0);
                    FloatVector in = FloatVector.fromArray(F_SPECIES, other, elemBase + l);
                    acc = vq.fma(vds.mul(in), acc);
                }

                // Sub-block 1: low nibbles of ql[32..63], qh bits 2-3
                for (int l = 0; l < 32; l += F_LEN) {
                    FloatVector vds = FloatVector.broadcast(F_SPECIES, d * sc[scBase + 2 + (l >> 4)]);
                    for (int j = 0; j < F_LEN; j++) {
                        int idx = l + j;
                        dq[j] = (float) (((Byte.toUnsignedInt(ql[qlOff + 32 + idx]) & 0x0F)
                                | (((Byte.toUnsignedInt(qh[qhOff + idx]) >> 2) & 3) << 4)) - 32);
                    }
                    FloatVector vq = FloatVector.fromArray(F_SPECIES, dq, 0);
                    FloatVector in = FloatVector.fromArray(F_SPECIES, other, elemBase + 32 + l);
                    acc = vq.fma(vds.mul(in), acc);
                }

                // Sub-block 2: high nibbles of ql[0..31], qh bits 4-5
                for (int l = 0; l < 32; l += F_LEN) {
                    FloatVector vds = FloatVector.broadcast(F_SPECIES, d * sc[scBase + 4 + (l >> 4)]);
                    for (int j = 0; j < F_LEN; j++) {
                        int idx = l + j;
                        dq[j] = (float) ((((Byte.toUnsignedInt(ql[qlOff + idx]) >> 4) & 0x0F)
                                | (((Byte.toUnsignedInt(qh[qhOff + idx]) >> 4) & 3) << 4)) - 32);
                    }
                    FloatVector vq = FloatVector.fromArray(F_SPECIES, dq, 0);
                    FloatVector in = FloatVector.fromArray(F_SPECIES, other, elemBase + 64 + l);
                    acc = vq.fma(vds.mul(in), acc);
                }

                // Sub-block 3: high nibbles of ql[32..63], qh bits 6-7
                for (int l = 0; l < 32; l += F_LEN) {
                    FloatVector vds = FloatVector.broadcast(F_SPECIES, d * sc[scBase + 6 + (l >> 4)]);
                    for (int j = 0; j < F_LEN; j++) {
                        int idx = l + j;
                        dq[j] = (float) ((((Byte.toUnsignedInt(ql[qlOff + 32 + idx]) >> 4) & 0x0F)
                                | (((Byte.toUnsignedInt(qh[qhOff + idx]) >> 6) & 3) << 4)) - 32);
                    }
                    FloatVector vq = FloatVector.fromArray(F_SPECIES, dq, 0);
                    FloatVector in = FloatVector.fromArray(F_SPECIES, other, elemBase + 96 + l);
                    acc = vq.fma(vds.mul(in), acc);
                }
            }
            otherBase += BLOCK_SIZE;
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }
}
