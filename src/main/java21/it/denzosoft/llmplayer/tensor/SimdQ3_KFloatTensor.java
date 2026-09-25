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
 * SIMD-optimized Q3_K tensor using lane-parallel B2I/I2F path.
 *
 * Rewritten 2026-04-15 as part of the CPU doubling sweep — old version kept a
 * scalar {@code for j in F_LEN} inner loop to extract 2-bit quants and high
 * mask bits. New version: {@link ByteVector} from segment, {@code B2I},
 * lane-wise mask+shift, {@code I2F}, FMA.
 *
 * Q3_K block layout (110 bytes, 256 elements):
 *   hmask[32] (offset 0):   high bit mask
 *   qs[64]   (offset 32):   256 × 2-bit low quants
 *   sc[12]   (offset 96):   16 × 6-bit sub-block scales (packed)
 *   d        (fp16, offset 108): super-block scale
 */
public class SimdQ3_KFloatTensor extends Q3_KFloatTensor {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_256;
    private static final VectorSpecies<Integer> I_SPECIES = IntVector.SPECIES_256;
    private static final VectorSpecies<Byte> B_SPECIES = ByteVector.SPECIES_64;
    private static final int F_LEN = 8;
    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 110;
    private static final ValueLayout.OfByte BYTE_LE = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED;
    private static final ByteOrder BYTE_ORDER = ByteOrder.LITTLE_ENDIAN;

    private final MemorySegment segment;

    public SimdQ3_KFloatTensor(TensorData data, long size) {
        super(data, size);
        this.segment = ((MemorySegmentTensorData) data).segment();
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        if (FloatVector.SPECIES_PREFERRED.length() < 8 || length % BLOCK_SIZE != 0) {
            return super.dot(thisOffset, other, otherOffset, length);
        }

        int numBlocks = length / BLOCK_SIZE;
        long blockStart = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        int otherIdx = otherOffset;

        FloatVector acc = FloatVector.zero(F_SPECIES);
        byte[] raw = new byte[12];
        int[] sc = new int[16];
        IntVector vMask2 = IntVector.broadcast(I_SPECIES, 0x03);
        IntVector vMask1 = IntVector.broadcast(I_SPECIES, 0x01);
        IntVector vSub4 = IntVector.broadcast(I_SPECIES, 4);

        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float d = Float.float16ToFloat(segment.get(SHORT_LE, bo + 108));
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

            long hmBase = bo;
            long qsBase = bo + 32;

            int scaleIdx = 0;
            int hmBitPos = 0;
            for (int half = 0; half < 2; half++) {
                long qBase = qsBase + half * 32L;

                for (int pair = 0; pair < 4; pair++) {
                    int shift = pair * 2;
                    float dl0 = d * (sc[scaleIdx++] - 32);
                    float dl1 = d * (sc[scaleIdx++] - 32);
                    FloatVector vdl0 = FloatVector.broadcast(F_SPECIES, dl0);
                    FloatVector vdl1 = FloatVector.broadcast(F_SPECIES, dl1);
                    int elemBase = otherIdx + half * 128 + pair * 32;

                    // First 16 elements
                    for (int l = 0; l < 16; l += F_LEN) {
                        ByteVector vqs = ByteVector.fromMemorySegment(B_SPECIES, segment, qBase + l, BYTE_ORDER);
                        ByteVector vhm = ByteVector.fromMemorySegment(B_SPECIES, segment, hmBase + l, BYTE_ORDER);
                        IntVector vqsI = (IntVector) vqs.convertShape(VectorOperators.B2I, I_SPECIES, 0);
                        IntVector vhmI = (IntVector) vhm.convertShape(VectorOperators.B2I, I_SPECIES, 0);

                        IntVector lowBits = vqsI.lanewise(VectorOperators.LSHR, shift).and(vMask2);
                        IntVector hbit = vhmI.lanewise(VectorOperators.LSHR, hmBitPos).and(vMask1).lanewise(VectorOperators.LSHL, 2);
                        IntVector q = lowBits.or(hbit).sub(vSub4);
                        FloatVector vq = (FloatVector) q.convertShape(VectorOperators.I2F, F_SPECIES, 0);
                        FloatVector in = FloatVector.fromArray(F_SPECIES, other, elemBase + l);
                        acc = vq.fma(vdl0.mul(in), acc);
                    }

                    // Second 16 elements
                    for (int l = 0; l < 16; l += F_LEN) {
                        ByteVector vqs = ByteVector.fromMemorySegment(B_SPECIES, segment, qBase + 16 + l, BYTE_ORDER);
                        ByteVector vhm = ByteVector.fromMemorySegment(B_SPECIES, segment, hmBase + 16 + l, BYTE_ORDER);
                        IntVector vqsI = (IntVector) vqs.convertShape(VectorOperators.B2I, I_SPECIES, 0);
                        IntVector vhmI = (IntVector) vhm.convertShape(VectorOperators.B2I, I_SPECIES, 0);

                        IntVector lowBits = vqsI.lanewise(VectorOperators.LSHR, shift).and(vMask2);
                        IntVector hbit = vhmI.lanewise(VectorOperators.LSHR, hmBitPos).and(vMask1).lanewise(VectorOperators.LSHL, 2);
                        IntVector q = lowBits.or(hbit).sub(vSub4);
                        FloatVector vq = (FloatVector) q.convertShape(VectorOperators.I2F, F_SPECIES, 0);
                        FloatVector in = FloatVector.fromArray(F_SPECIES, other, elemBase + 16 + l);
                        acc = vq.fma(vdl1.mul(in), acc);
                    }

                    hmBitPos++;
                }
            }

            otherIdx += BLOCK_SIZE;
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    /**
     * Multi-token row-range matmul for batched prefill: the same per-8-weight dequantisation as
     * {@link #dot}, done once and FMA'd against four inputs. Leftover tokens use {@link #dot}.
     */
    @Override
    public void matmulRowsBatch(float[][] in, float[][] out, int n, int rowFrom, int rowTo, int cols) {
        if (FloatVector.SPECIES_PREFERRED.length() < 8 || cols % BLOCK_SIZE != 0) {
            super.matmulRowsBatch(in, out, n, rowFrom, rowTo, cols);
            return;
        }
        int numBlocks = cols / BLOCK_SIZE;
        byte[] raw = new byte[12];
        int[] sc = new int[16];
        float[] res = new float[4];
        for (int row = rowFrom; row < rowTo; row++) {
            long blockStart = (long) row * numBlocks * BLOCK_BYTES;
            // Groups of four; a short last group repeats its last input and keeps only the real
            // results. The single-token kernels are deliberately never called here: compiled by C2
            // during prefill from these few calls, they kept a poor profile and made the following
            // decode up to 40 % slower.
            for (int t = 0; t < n; t += 4) {
                int t1 = Math.min(t + 1, n - 1), t2 = Math.min(t + 2, n - 1), t3 = Math.min(t + 3, n - 1);
                dot4(blockStart, in[t], in[t1], in[t2], in[t3], numBlocks, raw, sc, res);
                out[t][row] += res[0];
                if (t + 1 < n) out[t + 1][row] += res[1];
                if (t + 2 < n) out[t + 2][row] += res[2];
                if (t + 3 < n) out[t + 3][row] += res[3];
            }
        }
    }

    private void dot4(long blockStart, float[] x0, float[] x1, float[] x2, float[] x3, int numBlocks,
                      byte[] raw, int[] sc, float[] res) {
        FloatVector a0 = FloatVector.zero(F_SPECIES);
        FloatVector a1 = a0, a2 = a0, a3 = a0;
        IntVector vMask2 = IntVector.broadcast(I_SPECIES, 0x03);
        IntVector vMask1 = IntVector.broadcast(I_SPECIES, 0x01);
        IntVector vSub4 = IntVector.broadcast(I_SPECIES, 4);
        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float d = Float.float16ToFloat(segment.get(SHORT_LE, bo + 108));
            MemorySegment.copy(segment, BYTE_LE, bo + 96, raw, 0, 12);
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
            long hmBase = bo;
            long qsBase = bo + 32;
            int scaleIdx = 0;
            int hmBitPos = 0;
            for (int half = 0; half < 2; half++) {
                long qBase = qsBase + half * 32L;
                for (int pair = 0; pair < 4; pair++) {
                    int shift = pair * 2;
                    FloatVector vdl0 = FloatVector.broadcast(F_SPECIES, d * (sc[scaleIdx++] - 32));
                    FloatVector vdl1 = FloatVector.broadcast(F_SPECIES, d * (sc[scaleIdx++] - 32));
                    int elemBase = b * BLOCK_SIZE + half * 128 + pair * 32;
                    for (int l = 0; l < 32; l += F_LEN) {
                        ByteVector vqs = ByteVector.fromMemorySegment(B_SPECIES, segment, qBase + l, BYTE_ORDER);
                        ByteVector vhm = ByteVector.fromMemorySegment(B_SPECIES, segment, hmBase + l, BYTE_ORDER);
                        IntVector vqsI = (IntVector) vqs.convertShape(VectorOperators.B2I, I_SPECIES, 0);
                        IntVector vhmI = (IntVector) vhm.convertShape(VectorOperators.B2I, I_SPECIES, 0);
                        IntVector lowBits = vqsI.lanewise(VectorOperators.LSHR, shift).and(vMask2);
                        IntVector hbit = vhmI.lanewise(VectorOperators.LSHR, hmBitPos).and(vMask1).lanewise(VectorOperators.LSHL, 2);
                        FloatVector w = ((FloatVector) lowBits.or(hbit).sub(vSub4).convertShape(VectorOperators.I2F, F_SPECIES, 0))
                            .mul(l < 16 ? vdl0 : vdl1);
                        int xo = elemBase + l;
                        a0 = w.fma(FloatVector.fromArray(F_SPECIES, x0, xo), a0);
                        a1 = w.fma(FloatVector.fromArray(F_SPECIES, x1, xo), a1);
                        a2 = w.fma(FloatVector.fromArray(F_SPECIES, x2, xo), a2);
                        a3 = w.fma(FloatVector.fromArray(F_SPECIES, x3, xo), a3);
                    }
                    hmBitPos++;
                }
            }
        }
        res[0] = a0.reduceLanes(VectorOperators.ADD);
        res[1] = a1.reduceLanes(VectorOperators.ADD);
        res[2] = a2.reduceLanes(VectorOperators.ADD);
        res[3] = a3.reduceLanes(VectorOperators.ADD);
    }
}
