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
        if (FloatVector.SPECIES_PREFERRED.length() < 8 || length % BLOCK_SIZE != 0) {
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

    /**
     * Row-range matmul with packed loads, following {@link SimdQ4_KFloatTensor#matmulRows}: the input
     * is permuted and its sub-block sums computed once per range; the 32 {@code qs} bytes of a group
     * and the 32 {@code qh} bytes of the block are each loaded as one 8-int vector, and the fifth bit
     * of element {@code 4k+p} is bit {@code 8p + 2*group} (low nibble) or {@code 8p + 2*group + 1}
     * (high nibble) of lane {@code k} of {@code qh}. Raw {@code q.x} is accumulated per sub-block and
     * {@code d*sc} applied once per 32 weights; {@code dmin*m*sum(x)} is scalar.
     */
    @Override
    public void matmulRows(float[] input, float[] out, int rowFrom, int rowTo, int cols) {
        if (FloatVector.SPECIES_PREFERRED.length() < 8 || cols % BLOCK_SIZE != 0) {
            super.matmulRows(input, out, rowFrom, rowTo, cols);
            return;
        }
        int numBlocks = cols / BLOCK_SIZE;
        KQuantInput.Scratch sc = KQuantInput.scratch();
        float[] sums = sc.sums(0, input, cols);
        float[] xp = sc.permuted(0, input, cols);
        int[] scales = sc.scales;
        int[] mins = sc.mins;
        for (int row = rowFrom; row < rowTo; row++) {
            out[row] += dotPacked((long) row * numBlocks * BLOCK_BYTES, xp, numBlocks, sums, scales, mins);
        }
    }

    private float dotPacked(long blockStart, float[] xp, int numBlocks, float[] sums, int[] scales, int[] mins) {
        IntVector m4 = IntVector.broadcast(I_SPECIES, 0x0F);
        IntVector m16 = IntVector.broadcast(I_SPECIES, 0x10);
        FloatVector acc0 = FloatVector.zero(F_SPECIES);
        FloatVector acc1 = FloatVector.zero(F_SPECIES);
        float minSum = 0f;
        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float d = Float.float16ToFloat(segment.get(SHORT_LE, bo));
            float dmin = Float.float16ToFloat(segment.get(SHORT_LE, bo + 2));
            KQuantInput.decodeScales(segment, bo + 4, scales, mins);
            int sb = b * 8;
            float m = 0f;
            for (int j = 0; j < 8; j++) m += mins[j] * sums[sb + j];
            minSum += dmin * m;

            IntVector qh = IntVector.fromMemorySegment(I_SPECIES, segment, bo + 16, BYTE_ORDER);
            int xBase = b * BLOCK_SIZE;
            // hg = qh >>> (2 * group), advanced by a constant 2 per group (constant shift counts:
            // see KQuantInput).
            IntVector hg = qh;
            for (int group = 0; group < 4; group++, hg = hg.lanewise(VectorOperators.LSHR, 2)) {
                IntVector q = IntVector.fromMemorySegment(I_SPECIES, segment, bo + 48 + group * 32L, BYTE_ORDER);
                int lo = xBase + group * 64;
                int hi = lo + 32;
                FloatVector aLo = FloatVector.zero(F_SPECIES);
                FloatVector aHi = FloatVector.zero(F_SPECIES);
                IntVector qs = q;
                IntVector h = hg;
                for (int p = 0; p < 4; p++, qs = qs.lanewise(VectorOperators.LSHR, 8), h = h.lanewise(VectorOperators.LSHR, 8)) {
                    IntVector vLo = qs.and(m4).or(h.lanewise(VectorOperators.LSHL, 4).and(m16));
                    IntVector vHi = qs.lanewise(VectorOperators.LSHR, 4).and(m4).or(h.lanewise(VectorOperators.LSHL, 3).and(m16));
                    aLo = i2f(vLo).fma(FloatVector.fromArray(F_SPECIES, xp, lo + 8 * p), aLo);
                    aHi = i2f(vHi).fma(FloatVector.fromArray(F_SPECIES, xp, hi + 8 * p), aHi);
                }
                acc0 = aLo.fma(FloatVector.broadcast(F_SPECIES, d * scales[group * 2]), acc0);
                acc1 = aHi.fma(FloatVector.broadcast(F_SPECIES, d * scales[group * 2 + 1]), acc1);
            }
        }
        return acc0.add(acc1).reduceLanes(VectorOperators.ADD) - minSum;
    }

    /**
     * Multi-token row-range matmul for batched prefill: four tokens at a time, each weight vector
     * unpacked, converted and scaled once. Leftover tokens use {@link #dotPacked}.
     */
    @Override
    public void matmulRowsBatch(float[][] in, float[][] out, int n, int rowFrom, int rowTo, int cols) {
        if (FloatVector.SPECIES_PREFERRED.length() < 8 || cols % BLOCK_SIZE != 0) {
            super.matmulRowsBatch(in, out, n, rowFrom, rowTo, cols);
            return;
        }
        int numBlocks = cols / BLOCK_SIZE;
        KQuantInput.Scratch sc = KQuantInput.scratch();
        for (int t = 0; t < n; t++) {
            sc.permuted(t, in[t], cols);
            sc.sums(t, in[t], cols);
        }
        float[][] xp = sc.xp;
        float[][] sums = sc.sums;
        int[] scales = sc.scales;
        int[] mins = sc.mins;
        float[] res = sc.res;
        for (int row = rowFrom; row < rowTo; row++) {
            long blockStart = (long) row * numBlocks * BLOCK_BYTES;
            // Groups of four; a short last group repeats its last input and keeps only the real
            // results. The single-token kernels are deliberately never called here: compiled by C2
            // during prefill from these few calls, they kept a poor profile and made the following
            // decode up to 40 % slower.
            for (int t = 0; t < n; t += 4) {
                int t1 = Math.min(t + 1, n - 1), t2 = Math.min(t + 2, n - 1), t3 = Math.min(t + 3, n - 1);
                dotPacked4(blockStart, xp[t], xp[t1], xp[t2], xp[t3],
                    sums[t], sums[t1], sums[t2], sums[t3], numBlocks, scales, mins, res);
                out[t][row] += res[0];
                if (t + 1 < n) out[t + 1][row] += res[1];
                if (t + 2 < n) out[t + 2][row] += res[2];
                if (t + 3 < n) out[t + 3][row] += res[3];
            }
        }
    }

    private void dotPacked4(long blockStart, float[] x0, float[] x1, float[] x2, float[] x3,
                            float[] s0, float[] s1, float[] s2, float[] s3,
                            int numBlocks, int[] scales, int[] mins, float[] res) {
        IntVector m4 = IntVector.broadcast(I_SPECIES, 0x0F);
        IntVector m16 = IntVector.broadcast(I_SPECIES, 0x10);
        FloatVector a0 = FloatVector.zero(F_SPECIES);
        FloatVector a1 = a0, a2 = a0, a3 = a0;
        float n0 = 0f, n1 = 0f, n2 = 0f, n3 = 0f;
        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float d = Float.float16ToFloat(segment.get(SHORT_LE, bo));
            float dmin = Float.float16ToFloat(segment.get(SHORT_LE, bo + 2));
            KQuantInput.decodeScales(segment, bo + 4, scales, mins);
            int sb = b * 8;
            for (int j = 0; j < 8; j++) {
                float dm = dmin * mins[j];
                n0 += dm * s0[sb + j];
                n1 += dm * s1[sb + j];
                n2 += dm * s2[sb + j];
                n3 += dm * s3[sb + j];
            }
            IntVector qh = IntVector.fromMemorySegment(I_SPECIES, segment, bo + 16, BYTE_ORDER);
            int xBase = b * BLOCK_SIZE;
            // hg = qh >>> (2 * group), advanced by a constant 2 per group (constant shift counts:
            // see KQuantInput).
            IntVector hg = qh;
            for (int group = 0; group < 4; group++, hg = hg.lanewise(VectorOperators.LSHR, 2)) {
                IntVector q = IntVector.fromMemorySegment(I_SPECIES, segment, bo + 48 + group * 32L, BYTE_ORDER);
                FloatVector ds0 = FloatVector.broadcast(F_SPECIES, d * scales[group * 2]);
                FloatVector ds1 = FloatVector.broadcast(F_SPECIES, d * scales[group * 2 + 1]);
                int lo = xBase + group * 64;
                int hi = lo + 32;
                IntVector qs = q;
                IntVector h = hg;
                for (int p = 0; p < 4; p++, qs = qs.lanewise(VectorOperators.LSHR, 8), h = h.lanewise(VectorOperators.LSHR, 8)) {
                    FloatVector wl = i2f(qs.and(m4).or(h.lanewise(VectorOperators.LSHL, 4).and(m16))).mul(ds0);
                    FloatVector wh = i2f(qs.lanewise(VectorOperators.LSHR, 4).and(m4).or(h.lanewise(VectorOperators.LSHL, 3).and(m16))).mul(ds1);
                    int ol = lo + 8 * p;
                    int oh = hi + 8 * p;
                    a0 = wh.fma(FloatVector.fromArray(F_SPECIES, x0, oh), wl.fma(FloatVector.fromArray(F_SPECIES, x0, ol), a0));
                    a1 = wh.fma(FloatVector.fromArray(F_SPECIES, x1, oh), wl.fma(FloatVector.fromArray(F_SPECIES, x1, ol), a1));
                    a2 = wh.fma(FloatVector.fromArray(F_SPECIES, x2, oh), wl.fma(FloatVector.fromArray(F_SPECIES, x2, ol), a2));
                    a3 = wh.fma(FloatVector.fromArray(F_SPECIES, x3, oh), wl.fma(FloatVector.fromArray(F_SPECIES, x3, ol), a3));
                }
            }
        }
        res[0] = a0.reduceLanes(VectorOperators.ADD) - n0;
        res[1] = a1.reduceLanes(VectorOperators.ADD) - n1;
        res[2] = a2.reduceLanes(VectorOperators.ADD) - n2;
        res[3] = a3.reduceLanes(VectorOperators.ADD) - n3;
    }

    private static FloatVector i2f(IntVector v) {
        return (FloatVector) v.convertShape(VectorOperators.I2F, F_SPECIES, 0);
    }
}
