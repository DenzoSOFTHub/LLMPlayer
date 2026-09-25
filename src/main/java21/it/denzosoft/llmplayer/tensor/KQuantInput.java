package it.denzosoft.llmplayer.tensor;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Helpers shared by the Q4_K and Q5_K "packed" kernels: per-input preparation done once per row
 * range, and the 6-bit scale/min unpacking both formats use.
 *
 * <p>The packed kernels load the 32 quant bytes of a group as one 8-int vector. Lane {@code k} then
 * holds bytes {@code 4k..4k+3}, so the value at byte {@code p} is element {@code 4k+p}; the input is
 * permuted to that order once ({@link #permute}), which leaves every 32-element sub-block's dot
 * product unchanged.
 *
 * <p>Kernels using these helpers keep every vector shift count a compile-time constant (they advance
 * a shifted copy by a constant each step instead of shifting by {@code 8 * p}). With loop-variable
 * counts, profiles of the JDK's shared shift implementation could leave shifts un-intrinsified in
 * other kernels too — JFR showed {@code IntVector.lanewiseShiftTemplate} as a hot method under the
 * single-token Q4_K kernel after a batched prefill.
 */
final class KQuantInput {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_256;
    private static final ValueLayout.OfLong LONG_LE = ValueLayout.JAVA_LONG_UNALIGNED;
    private static final ValueLayout.OfInt INT_LE = ValueLayout.JAVA_INT_UNALIGNED;

    private KQuantInput() {}

    /**
     * Per-thread working buffers for the packed kernels. Allocating the permuted input and the
     * sub-block sums on every row range cost ~30 MB per decoded token and ~1000 young GCs over a
     * short run, each stopping the matmul threads; the pool's workers are long-lived platform
     * threads, so a thread-local buffer is reused for the life of the process.
     */
    static final class Scratch {
        float[][] xp = new float[0][];
        float[][] sums = new float[0][];
        final int[] scales = new int[8];
        final int[] mins = new int[8];
        final float[] res = new float[4];

        /** Permuted copy of {@code input} in slot {@code slot}. */
        float[] permuted(int slot, float[] input, int cols) {
            ensure(slot + 1, cols);
            float[] dst = xp[slot];
            for (int base = 0; base < cols; base += 32) {
                for (int k = 0; k < 8; k++) {
                    int src = base + 4 * k;
                    dst[base + k] = input[src];
                    dst[base + 8 + k] = input[src + 1];
                    dst[base + 16 + k] = input[src + 2];
                    dst[base + 24 + k] = input[src + 3];
                }
            }
            return dst;
        }

        /** Sub-block sums of {@code input} in slot {@code slot}. */
        float[] sums(int slot, float[] input, int cols) {
            ensure(slot + 1, cols);
            float[] dst = sums[slot];
            for (int j = 0; j < cols / 32; j++) {
                int base = j * 32;
                FloatVector v = FloatVector.fromArray(F_SPECIES, input, base)
                    .add(FloatVector.fromArray(F_SPECIES, input, base + 8))
                    .add(FloatVector.fromArray(F_SPECIES, input, base + 16).add(FloatVector.fromArray(F_SPECIES, input, base + 24)));
                dst[j] = v.reduceLanes(VectorOperators.ADD);
            }
            return dst;
        }

        private void ensure(int slots, int cols) {
            if (xp.length < slots) {
                xp = java.util.Arrays.copyOf(xp, slots);
                sums = java.util.Arrays.copyOf(sums, slots);
            }
            for (int i = 0; i < slots; i++) {
                if (xp[i] == null || xp[i].length < cols) {
                    xp[i] = new float[cols];
                    sums[i] = new float[cols / 32 + 1];
                }
            }
        }
    }

    private static final ThreadLocal<Scratch> SCRATCH = ThreadLocal.withInitial(Scratch::new);

    static Scratch scratch() {
        return SCRATCH.get();
    }

    /** Within every 32-element sub-block: {@code xp[p*8 + k] = x[4k + p]}. */
    static float[] permute(float[] input, int cols) {
        float[] xp = new float[cols];
        for (int base = 0; base < cols; base += 32) {
            for (int k = 0; k < 8; k++) {
                int src = base + 4 * k;
                xp[base + k] = input[src];
                xp[base + 8 + k] = input[src + 1];
                xp[base + 16 + k] = input[src + 2];
                xp[base + 24 + k] = input[src + 3];
            }
        }
        return xp;
    }

    /** Sum of every 32-element sub-block of the input: {@code cols / 32} values. */
    static float[] subBlockSums(float[] input, int cols) {
        float[] sums = new float[cols / 32];
        for (int j = 0; j < sums.length; j++) {
            int base = j * 32;
            FloatVector s = FloatVector.fromArray(F_SPECIES, input, base)
                .add(FloatVector.fromArray(F_SPECIES, input, base + 8))
                .add(FloatVector.fromArray(F_SPECIES, input, base + 16).add(FloatVector.fromArray(F_SPECIES, input, base + 24)));
            sums[j] = s.reduceLanes(VectorOperators.ADD);
        }
        return sums;
    }

    /** Unpacks the eight 6-bit scales and mins stored in the 12 bytes at {@code offset} (K-quant packing). */
    static void decodeScales(MemorySegment segment, long offset, int[] scales, int[] mins) {
        long sb_0_7 = segment.get(LONG_LE, offset);
        int  sb_8_11 = segment.get(INT_LE, offset + 8);
        for (int i = 0; i < 4; i++) {
            scales[i] = ((int)(sb_0_7 >>> (i * 8))) & 0x3F;
            mins[i]   = ((int)(sb_0_7 >>> ((i + 4) * 8))) & 0x3F;
        }
        for (int i = 4; i < 8; i++) {
            int b8plus = (sb_8_11 >>> ((i - 4) * 8)) & 0xFF;
            int b_minus_4 = ((int)(sb_0_7 >>> ((i - 4) * 8))) & 0xFF;
            int b_at_i = ((int)(sb_0_7 >>> (i * 8))) & 0xFF;
            scales[i] = (b8plus & 0x0F) | ((b_minus_4 >>> 6) << 4);
            mins[i]   = ((b8plus >>> 4) & 0x0F) | ((b_at_i >>> 6) << 4);
        }
    }
}
