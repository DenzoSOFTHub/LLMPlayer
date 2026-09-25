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
 * SIMD-optimized Q4_K tensor. Loads 8 quant bytes as a {@link ByteVector},
 * expands to {@link IntVector} via {@code B2I}, extracts low/high nibbles with
 * masked {@code LSHR}, then {@code I2F} + FMA — fully vectorized dequant path.
 *
 * <p>Reads scales and quant bytes directly from the mapped {@link MemorySegment},
 * avoiding per-block byte[] scratch buffers and {@code MemorySegment.copy} calls.
 *
 * <p>Uses 256-bit shapes (AVX2 and AVX-512 hosts). Falls back to the scalar
 * {@code Q4_KFloatTensor.dot} path when the hardware vector is narrower than 256 bits.
 *
 * <p>Measured: +62% tok/s over prior scalar-nibble SIMD variant on Llama-3.2-1B
 * Q4_K_M CPU (Intel Core Ultra 7 155H).
 */
public class SimdQ4_KFloatTensor extends Q4_KFloatTensor {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_256;   // 8 floats
    private static final VectorSpecies<Integer> I_SPECIES = IntVector.SPECIES_256;   // 8 ints
    private static final VectorSpecies<Byte> B_SPECIES = ByteVector.SPECIES_64;      // 8 bytes
    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 144;
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED;
    private static final ValueLayout.OfLong LONG_LE = ValueLayout.JAVA_LONG_UNALIGNED;
    private static final ValueLayout.OfInt INT_LE = ValueLayout.JAVA_INT_UNALIGNED;

    private static final boolean USABLE = FloatVector.SPECIES_PREFERRED.length() >= 8;

    private final MemorySegment segment;

    public SimdQ4_KFloatTensor(TensorData data, long size) {
        super(data, size);
        this.segment = ((MemorySegmentTensorData) data).segment();
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        if (!USABLE || length % BLOCK_SIZE != 0) {
            return super.dot(thisOffset, other, otherOffset, length);
        }

        int numBlocks = length / BLOCK_SIZE;
        long blockStart = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        int[] scales = new int[8];
        int[] mins = new int[8];
        IntVector vMaskLow = IntVector.broadcast(I_SPECIES, 0x0F);

        // Weight = d*sc*q - dmin*m, so a sub-block contributes d*sc*(q.x) - dmin*m*sum(x).
        // q.x and sum(x) are accumulated raw per sub-block in independent registers, and the
        // scale is applied once per 32 elements, instead of dequantising every element and
        // feeding one serial FMA chain (the previous kernel was FMA-latency bound).
        FloatVector acc = FloatVector.zero(F_SPECIES);
        FloatVector minAcc = FloatVector.zero(F_SPECIES);
        int otherBase = otherOffset;
        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float d = Float.float16ToFloat(segment.get(SHORT_LE, bo));
            float dmin = Float.float16ToFloat(segment.get(SHORT_LE, bo + 2));
            decodeScales(bo, scales, mins);

            long qsBase = bo + 16;
            for (int group = 0; group < 4; group++) {
                long qsGroup = qsBase + (long) group * 32;
                int lowInputBase = otherBase + group * 64;
                int highInputBase = lowInputBase + 32;

                FloatVector qxLo = FloatVector.zero(F_SPECIES);
                FloatVector qxHi = FloatVector.zero(F_SPECIES);
                FloatVector sxLo = FloatVector.zero(F_SPECIES);
                FloatVector sxHi = FloatVector.zero(F_SPECIES);
                for (int l = 0; l < 32; l += 8) {
                    IntVector vqInt = (IntVector) ByteVector.fromMemorySegment(B_SPECIES, segment, qsGroup + l, ByteOrder.LITTLE_ENDIAN)
                        .convertShape(VectorOperators.B2I, I_SPECIES, 0);
                    FloatVector lo = (FloatVector) vqInt.and(vMaskLow).convertShape(VectorOperators.I2F, F_SPECIES, 0);
                    FloatVector hi = (FloatVector) vqInt.lanewise(VectorOperators.LSHR, 4).and(vMaskLow)
                        .convertShape(VectorOperators.I2F, F_SPECIES, 0);
                    FloatVector in0 = FloatVector.fromArray(F_SPECIES, other, lowInputBase + l);
                    FloatVector in1 = FloatVector.fromArray(F_SPECIES, other, highInputBase + l);
                    qxLo = lo.fma(in0, qxLo);
                    qxHi = hi.fma(in1, qxHi);
                    sxLo = sxLo.add(in0);
                    sxHi = sxHi.add(in1);
                }
                acc = qxLo.fma(FloatVector.broadcast(F_SPECIES, d * scales[group * 2]), acc);
                acc = qxHi.fma(FloatVector.broadcast(F_SPECIES, d * scales[group * 2 + 1]), acc);
                minAcc = sxLo.fma(FloatVector.broadcast(F_SPECIES, dmin * mins[group * 2]), minAcc);
                minAcc = sxHi.fma(FloatVector.broadcast(F_SPECIES, dmin * mins[group * 2 + 1]), minAcc);
            }
            otherBase += BLOCK_SIZE;
        }

        return acc.sub(minAcc).reduceLanes(VectorOperators.ADD);
    }

    /**
     * Row-range matmul, with two pieces of per-input work done once for the whole range instead
     * of once per row:
     * <ul>
     *   <li>the input's per-sub-block sums, which turn the {@code dmin * m * sum(x)} term into eight
     *       scalar FMAs per 256-element block;</li>
     *   <li>a permuted copy of the input matching the "packed" quant load. The 32 quant bytes of a
     *       group are loaded as one {@link IntVector} of 8 ints, so lane {@code k} holds bytes
     *       {@code 4k..4k+3} and the nibble at byte {@code p} is element {@code 4k+p}. Shifting the
     *       whole vector by {@code 8p} yields 8 elements at once with no byte-to-int widening
     *       shuffle; the input is permuted to that order ({@code xp[p*8+k] = x[4k+p]} within every
     *       32-element sub-block), which leaves each sub-block's dot product unchanged.</li>
     * </ul>
     * Measured +33–38 % single-thread over the per-8-byte {@code B2I} kernel on AVX2.
     */
    @Override
    public void matmulRows(float[] input, float[] out, int rowFrom, int rowTo, int cols) {
        if (!USABLE || cols % BLOCK_SIZE != 0) {
            super.matmulRows(input, out, rowFrom, rowTo, cols);
            return;
        }
        int numBlocks = cols / BLOCK_SIZE;
        KQuantInput.Scratch sc = KQuantInput.scratch();
        float[] subSums = sc.sums(0, input, cols);
        float[] permuted = sc.permuted(0, input, cols);
        int[] scales = sc.scales;
        int[] mins = sc.mins;
        for (int row = rowFrom; row < rowTo; row++) {
            long blockStart = (long) row * numBlocks * BLOCK_BYTES;
            out[row] += dotPacked(blockStart, permuted, numBlocks, subSums, scales, mins);
        }
    }

    private float dotPacked(long blockStart, float[] xp, int numBlocks, float[] subSums,
                            int[] scales, int[] mins) {
        IntVector m4 = IntVector.broadcast(I_SPECIES, 0x0F);
        FloatVector acc0 = FloatVector.zero(F_SPECIES);
        FloatVector acc1 = FloatVector.zero(F_SPECIES);
        float minSum = 0f;
        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float d = Float.float16ToFloat(segment.get(SHORT_LE, bo));
            float dmin = Float.float16ToFloat(segment.get(SHORT_LE, bo + 2));
            decodeScales(bo, scales, mins);

            int sb = b * 8;
            float m = 0f;
            for (int j = 0; j < 8; j++) m += mins[j] * subSums[sb + j];
            minSum += dmin * m;

            long qsBase = bo + 16;
            int otherBase = b * BLOCK_SIZE;
            for (int group = 0; group < 4; group++) {
                IntVector q = IntVector.fromMemorySegment(I_SPECIES, segment, qsBase + group * 32L, ByteOrder.LITTLE_ENDIAN);
                int lo = otherBase + group * 64;
                int hi = lo + 32;
                // Low nibbles (bits 8p..8p+3) feed sub-block 2*group, high nibbles sub-block 2*group+1.
                // Two accumulators per sub-block keep the FMA chains short.
                FloatVector l0 = i2f(q.and(m4)).mul(FloatVector.fromArray(F_SPECIES, xp, lo));
                FloatVector h0 = i2f(q.lanewise(VectorOperators.LSHR, 4).and(m4)).mul(FloatVector.fromArray(F_SPECIES, xp, hi));
                FloatVector l1 = i2f(q.lanewise(VectorOperators.LSHR, 8).and(m4)).mul(FloatVector.fromArray(F_SPECIES, xp, lo + 8));
                FloatVector h1 = i2f(q.lanewise(VectorOperators.LSHR, 12).and(m4)).mul(FloatVector.fromArray(F_SPECIES, xp, hi + 8));
                l0 = i2f(q.lanewise(VectorOperators.LSHR, 16).and(m4)).fma(FloatVector.fromArray(F_SPECIES, xp, lo + 16), l0);
                h0 = i2f(q.lanewise(VectorOperators.LSHR, 20).and(m4)).fma(FloatVector.fromArray(F_SPECIES, xp, hi + 16), h0);
                l1 = i2f(q.lanewise(VectorOperators.LSHR, 24).and(m4)).fma(FloatVector.fromArray(F_SPECIES, xp, lo + 24), l1);
                h1 = i2f(q.lanewise(VectorOperators.LSHR, 28)).fma(FloatVector.fromArray(F_SPECIES, xp, hi + 24), h1);
                acc0 = l0.add(l1).fma(FloatVector.broadcast(F_SPECIES, d * scales[group * 2]), acc0);
                acc1 = h0.add(h1).fma(FloatVector.broadcast(F_SPECIES, d * scales[group * 2 + 1]), acc1);
            }
        }
        return acc0.add(acc1).reduceLanes(VectorOperators.ADD) - minSum;
    }

    /**
     * Multi-token row-range matmul for batched prefill. Tokens are taken four at a time: each
     * weight vector is dequantised once, with its sub-block scale folded in, and FMA'd against the
     * four inputs, so the unpack/convert work is shared four ways and the row's bytes are read
     * from L1 for every tile after the first. Measured ~2.2x per token single-thread vs the
     * original one-token kernel. Leftover tokens use {@link #dotPacked}.
     */
    @Override
    public void matmulRowsBatch(float[][] in, float[][] out, int n, int rowFrom, int rowTo, int cols) {
        if (!USABLE || cols % BLOCK_SIZE != 0) {
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
        FloatVector a0 = FloatVector.zero(F_SPECIES);
        FloatVector a1 = a0, a2 = a0, a3 = a0;
        float m0 = 0f, m1 = 0f, m2 = 0f, m3 = 0f;
        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float d = Float.float16ToFloat(segment.get(SHORT_LE, bo));
            float dmin = Float.float16ToFloat(segment.get(SHORT_LE, bo + 2));
            decodeScales(bo, scales, mins);
            int sb = b * 8;
            for (int j = 0; j < 8; j++) {
                float dm = dmin * mins[j];
                m0 += dm * s0[sb + j];
                m1 += dm * s1[sb + j];
                m2 += dm * s2[sb + j];
                m3 += dm * s3[sb + j];
            }
            long qsBase = bo + 16;
            int otherBase = b * BLOCK_SIZE;
            for (int group = 0; group < 4; group++) {
                IntVector q = IntVector.fromMemorySegment(I_SPECIES, segment, qsBase + group * 32L, ByteOrder.LITTLE_ENDIAN);
                FloatVector ds0 = FloatVector.broadcast(F_SPECIES, d * scales[group * 2]);
                FloatVector ds1 = FloatVector.broadcast(F_SPECIES, d * scales[group * 2 + 1]);
                int lo = otherBase + group * 64;
                int hi = lo + 32;
                // Shift by a constant 8 per step rather than by 8 * p: see KQuantInput on why the
                // shift count must stay a compile-time constant.
                IntVector qs = q;
                for (int p = 0; p < 4; p++, qs = qs.lanewise(VectorOperators.LSHR, 8)) {
                    FloatVector wl = i2f(qs.and(m4)).mul(ds0);
                    FloatVector wh = i2f(qs.lanewise(VectorOperators.LSHR, 4).and(m4)).mul(ds1);
                    int ol = lo + 8 * p;
                    int oh = hi + 8 * p;
                    a0 = wh.fma(FloatVector.fromArray(F_SPECIES, x0, oh), wl.fma(FloatVector.fromArray(F_SPECIES, x0, ol), a0));
                    a1 = wh.fma(FloatVector.fromArray(F_SPECIES, x1, oh), wl.fma(FloatVector.fromArray(F_SPECIES, x1, ol), a1));
                    a2 = wh.fma(FloatVector.fromArray(F_SPECIES, x2, oh), wl.fma(FloatVector.fromArray(F_SPECIES, x2, ol), a2));
                    a3 = wh.fma(FloatVector.fromArray(F_SPECIES, x3, oh), wl.fma(FloatVector.fromArray(F_SPECIES, x3, ol), a3));
                }
            }
        }
        res[0] = a0.reduceLanes(VectorOperators.ADD) - m0;
        res[1] = a1.reduceLanes(VectorOperators.ADD) - m1;
        res[2] = a2.reduceLanes(VectorOperators.ADD) - m2;
        res[3] = a3.reduceLanes(VectorOperators.ADD) - m3;
    }

    private static FloatVector i2f(IntVector v) {
        return (FloatVector) v.convertShape(VectorOperators.I2F, F_SPECIES, 0);
    }

    private static float[] permuteForPackedLoad(float[] input, int cols) {
        return KQuantInput.permute(input, cols);
    }

    private static float[] subBlockSums(float[] input, int cols) {
        return KQuantInput.subBlockSums(input, cols);
    }

    /** Unpacks the eight 6-bit scales and mins of the block at {@code bo}. */
    private void decodeScales(long bo, int[] scales, int[] mins) {
        KQuantInput.decodeScales(segment, bo + 4, scales, mins);
    }
}
