package it.denzosoft.llmplayer.tensor;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * SIMD-optimized Q3_K tensor with fused dequantization and dot product.
 * Eliminates 5 ThreadLocal lookups per dot call and intermediate tmp[256] buffer.
 *
 * Q3_K block layout (110 bytes, 256 elements):
 *   hmask[32]: high bit mask for 256 weights
 *   qs[64]: 256 x 2-bit low quants (packed)
 *   scales[12]: 16 sub-block scales packed in 6 bits
 *   d (fp16, 2 bytes): super-block scale at offset 108
 */
public class SimdQ3_KFloatTensor extends Q3_KFloatTensor {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int F_LEN = F_SPECIES.length();
    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 110;
    private static final ValueLayout.OfByte BYTE_LE = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED;

    private final MemorySegment segment;

    public SimdQ3_KFloatTensor(TensorData data, long size) {
        super(data, size);
        this.segment = ((MemorySegmentTensorData) data).segment();
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        if (length % BLOCK_SIZE != 0) {
            return super.dot(thisOffset, other, otherOffset, length);
        }

        int numBlocks = length / BLOCK_SIZE;
        long blockStart = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        int otherIdx = otherOffset;

        FloatVector acc = FloatVector.zero(F_SPECIES);
        byte[] hm = new byte[32];
        byte[] qs = new byte[64];
        byte[] raw = new byte[12];
        int[] sc = new int[16];
        float[] dq = new float[F_LEN];

        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float d = Float.float16ToFloat(segment.get(SHORT_LE, bo + 108));

            MemorySegment.copy(segment, BYTE_LE, bo, hm, 0, 32);
            MemorySegment.copy(segment, BYTE_LE, bo + 32, qs, 0, 64);
            MemorySegment.copy(segment, BYTE_LE, bo + 96, raw, 0, 12);

            // Decode all 16 scales
            for (int i = 0; i < 8; i++) {
                int v = Byte.toUnsignedInt(raw[i]);
                sc[i] = v & 0x0F;
                sc[i + 8] = v >> 4;
            }
            for (int i = 0; i < 4; i++) {
                int v = Byte.toUnsignedInt(raw[8 + i]);
                sc[i]      |= (v & 0x03) << 4;
                sc[i + 4]  |= ((v >> 2) & 0x03) << 4;
                sc[i + 8]  |= ((v >> 4) & 0x03) << 4;
                sc[i + 12] |= ((v >> 6) & 0x03) << 4;
            }

            int scaleIdx = 0;
            int hmBitPos = 0;

            for (int half = 0; half < 2; half++) {
                int qBase = half * 32;

                for (int pair = 0; pair < 4; pair++) {
                    int shift = pair * 2;
                    float dl0 = d * (sc[scaleIdx] - 32);
                    scaleIdx++;
                    float dl1 = d * (sc[scaleIdx] - 32);
                    scaleIdx++;

                    int elemBase = otherIdx + half * 128 + pair * 32;

                    // First 16 elements of this sub-block
                    for (int l = 0; l < 16; l += F_LEN) {
                        for (int j = 0; j < F_LEN; j++) {
                            int idx = l + j;
                            int qsByte = Byte.toUnsignedInt(qs[qBase + idx]);
                            int hmByte = Byte.toUnsignedInt(hm[idx]);
                            int lowBits = (qsByte >> shift) & 3;
                            int hbit = (hmByte >> hmBitPos) & 1;
                            int qVal = (lowBits | (hbit << 2)) - 4;
                            dq[j] = dl0 * qVal;
                        }
                        FloatVector vq = FloatVector.fromArray(F_SPECIES, dq, 0);
                        FloatVector vIn = FloatVector.fromArray(F_SPECIES, other, elemBase + l);
                        acc = vq.fma(vIn, acc);
                    }

                    // Second 16 elements
                    for (int l = 0; l < 16; l += F_LEN) {
                        for (int j = 0; j < F_LEN; j++) {
                            int idx = l + j;
                            int qsByte = Byte.toUnsignedInt(qs[qBase + 16 + idx]);
                            int hmByte = Byte.toUnsignedInt(hm[16 + idx]);
                            int lowBits = (qsByte >> shift) & 3;
                            int hbit = (hmByte >> hmBitPos) & 1;
                            int qVal = (lowBits | (hbit << 2)) - 4;
                            dq[j] = dl1 * qVal;
                        }
                        FloatVector vq = FloatVector.fromArray(F_SPECIES, dq, 0);
                        FloatVector vIn = FloatVector.fromArray(F_SPECIES, other, elemBase + 16 + l);
                        acc = vq.fma(vIn, acc);
                    }

                    hmBitPos++;
                }
            }

            otherIdx += BLOCK_SIZE;
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }
}
