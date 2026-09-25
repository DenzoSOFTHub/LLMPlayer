package it.denzosoft.llmplayer.tensor;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * SIMD-optimized Q6_K tensor using lane-parallel B2I/I2F nibble extraction.
 *
 * Rewritten 2026-04-15 after JFR showed {@code SimdQ6_KFloatTensor.dot} as
 * the #1 CPU hotspot on Llama Q4_K_M (3× more samples than {@code SimdQ4_K.dot}).
 * Previous version extracted nibbles + shifted qh bits with a scalar
 * {@code for j in F_LEN} inner loop; the new version mirrors the {@link SimdQ4_KFloatTensor}
 * pattern: read {@link ByteVector} directly from the mapped segment, widen to
 * {@link IntVector} via {@code B2I}, extract low/high nibbles and qh 2-bit pairs
 * with masked shifts, then {@code I2F} + FMA.
 *
 * Q6_K block layout (210 bytes, 256 elements):
 *   ql[128] (offset 0):   lower 4 bits of 6-bit quants
 *   qh[64]  (offset 128): upper 2 bits of 6-bit quants (four 2-bit pairs per byte)
 *   sc[16]  (offset 192): int8 sub-block scales
 *   d (fp16, offset 208): super-block scale
 */
public class SimdQ6_KFloatTensor extends Q6_KFloatTensor {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_256;   // 8 floats
    private static final VectorSpecies<Integer> I_SPECIES = IntVector.SPECIES_256;   // 8 ints
    private static final VectorSpecies<Byte> B_SPECIES = ByteVector.SPECIES_64;      // 8 bytes
    private static final int F_LEN = 8;
    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 210;
    private static final ValueLayout.OfByte BYTE_LE = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED;
    private static final ByteOrder BYTE_ORDER = ByteOrder.LITTLE_ENDIAN;

    private final MemorySegment segment;

    public SimdQ6_KFloatTensor(TensorData data, long size) {
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
        byte[] sc = new byte[16];
        IntVector vMask4 = IntVector.broadcast(I_SPECIES, 0x0F);
        IntVector vMask2 = IntVector.broadcast(I_SPECIES, 0x03);
        IntVector vSub32 = IntVector.broadcast(I_SPECIES, 32);

        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float d = Float.float16ToFloat(segment.get(SHORT_LE, bo + 208));
            MemorySegment.copy(segment, BYTE_LE, bo + 192, sc, 0, 16);

            for (int half = 0; half < 2; half++) {
                long qlBase = bo + half * 64L;                // 64 ql bytes for this half
                long qhBase = bo + 128L + half * 32L;         // 32 qh bytes for this half
                int scBase = half * 8;
                int elemBase = otherBase + half * 128;

                for (int l = 0; l < 32; l += F_LEN) {
                    // Load 8 qh bytes shared by all 4 sub-blocks
                    ByteVector vqh = ByteVector.fromMemorySegment(B_SPECIES, segment, qhBase + l, BYTE_ORDER);
                    IntVector vqhI = (IntVector) vqh.convertShape(VectorOperators.B2I, I_SPECIES, 0);

                    // Load 16 ql bytes: 8 at ql[qlBase+l] and 8 at ql[qlBase+32+l]
                    ByteVector vqlLo = ByteVector.fromMemorySegment(B_SPECIES, segment, qlBase + l, BYTE_ORDER);
                    ByteVector vqlHi = ByteVector.fromMemorySegment(B_SPECIES, segment, qlBase + 32L + l, BYTE_ORDER);
                    IntVector vqlLoI = (IntVector) vqlLo.convertShape(VectorOperators.B2I, I_SPECIES, 0);
                    IntVector vqlHiI = (IntVector) vqlHi.convertShape(VectorOperators.B2I, I_SPECIES, 0);

                    int scIdx = l >> 4;  // 0 or 1 (since l = 0, 8, 16, 24)

                    // Sub-block 0: low nibble of ql[qlBase..+32], qh bits 0-1
                    {
                        IntVector low4 = vqlLoI.and(vMask4);
                        IntVector high2 = vqhI.and(vMask2).lanewise(VectorOperators.LSHL, 4);
                        IntVector q = low4.or(high2).sub(vSub32);
                        FloatVector vq = (FloatVector) q.convertShape(VectorOperators.I2F, F_SPECIES, 0);
                        FloatVector vds = FloatVector.broadcast(F_SPECIES, d * sc[scBase + scIdx]);
                        FloatVector in = FloatVector.fromArray(F_SPECIES, other, elemBase + l);
                        acc = vq.fma(vds.mul(in), acc);
                    }

                    // Sub-block 1: low nibble of ql[qlBase+32..+64], qh bits 2-3
                    {
                        IntVector low4 = vqlHiI.and(vMask4);
                        IntVector high2 = vqhI.lanewise(VectorOperators.LSHR, 2).and(vMask2).lanewise(VectorOperators.LSHL, 4);
                        IntVector q = low4.or(high2).sub(vSub32);
                        FloatVector vq = (FloatVector) q.convertShape(VectorOperators.I2F, F_SPECIES, 0);
                        FloatVector vds = FloatVector.broadcast(F_SPECIES, d * sc[scBase + 2 + scIdx]);
                        FloatVector in = FloatVector.fromArray(F_SPECIES, other, elemBase + 32 + l);
                        acc = vq.fma(vds.mul(in), acc);
                    }

                    // Sub-block 2: high nibble of ql[qlBase..+32], qh bits 4-5
                    {
                        IntVector low4 = vqlLoI.lanewise(VectorOperators.LSHR, 4).and(vMask4);
                        IntVector high2 = vqhI.lanewise(VectorOperators.LSHR, 4).and(vMask2).lanewise(VectorOperators.LSHL, 4);
                        IntVector q = low4.or(high2).sub(vSub32);
                        FloatVector vq = (FloatVector) q.convertShape(VectorOperators.I2F, F_SPECIES, 0);
                        FloatVector vds = FloatVector.broadcast(F_SPECIES, d * sc[scBase + 4 + scIdx]);
                        FloatVector in = FloatVector.fromArray(F_SPECIES, other, elemBase + 64 + l);
                        acc = vq.fma(vds.mul(in), acc);
                    }

                    // Sub-block 3: high nibble of ql[qlBase+32..+64], qh bits 6-7
                    {
                        IntVector low4 = vqlHiI.lanewise(VectorOperators.LSHR, 4).and(vMask4);
                        IntVector high2 = vqhI.lanewise(VectorOperators.LSHR, 6).and(vMask2).lanewise(VectorOperators.LSHL, 4);
                        IntVector q = low4.or(high2).sub(vSub32);
                        FloatVector vq = (FloatVector) q.convertShape(VectorOperators.I2F, F_SPECIES, 0);
                        FloatVector vds = FloatVector.broadcast(F_SPECIES, d * sc[scBase + 6 + scIdx]);
                        FloatVector in = FloatVector.fromArray(F_SPECIES, other, elemBase + 96 + l);
                        acc = vq.fma(vds.mul(in), acc);
                    }
                }
            }
            otherBase += BLOCK_SIZE;
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    // ---------------------------------------------------------------------------------------
    // Lossless int8 repack for the dense matmul paths.
    //
    // A Q6_K weight is a 6-bit integer q in [0, 63] stored as 4 low bits in ql and 2 high bits in
    // qh, with value d * sc[j] * (q - 32) for sub-block j of 16. Unpacking the two bit fields costs
    // three widening loads and about ten integer ops per 8 weights, which kept this kernel at
    // ~2.5-3 Gelem/s while Q8_0 runs at ~7. Every (q - 32) fits in an int8, and d (fp16) times sc
    // (int8) is exact in float32, so the tensor can be rewritten once into a Q8-like layout with
    // bit-identical weight values:
    //
    //   per 256-weight block, REPACKED_BYTES = 276:  int8 q[256] (already minus 32) |
    //                                                int8 sc[16] | float32 d
    //
    // The kernel then has the Q8_0 shape (widen, convert, FMA; scale once per 16 weights).
    // Measured 7.6-7.7 vs 2.9 Gelem/s single-thread (in-process A/B). The copy is 1.31x the Q6_K
    // bytes and lives off-heap; the original mapped pages are no longer read by these paths and
    // can be dropped by the OS. Only matmulRows / matmulRowsBatch use it — dot() (MoE expert
    // slices, other callers) keeps reading the original layout.
    //
    // OPT-IN (-Dq6k.repack=true). End to end it did not pay on the reference machine: with all
    // threads running, decode is memory-bandwidth bound and the repacked bytes are 1.31x the Q6_K
    // bytes, so Llama-1B decode was slower in 9 of 10 paired repetitions even though the kernel is
    // 2.2x faster on one core; batched prefill (compute bound) gained 1.21x. Worth re-measuring on
    // hosts with more memory bandwidth per core. Never applied to lazy larger-than-RAM loads
    // (mmap.advise random/sequential), where the copy would compete with the page cache.
    // ---------------------------------------------------------------------------------------

    private static final int REPACKED_BYTES = 276;
    private static final ValueLayout.OfFloat FLOAT_LE = ValueLayout.JAVA_FLOAT_UNALIGNED;

    private volatile MemorySegment repacked;
    private volatile boolean repackDeclined;

    private static boolean repackAllowed() {
        if (!"true".equalsIgnoreCase(System.getProperty("q6k.repack", "false"))) return false;
        String advise = System.getProperty("mmap.advise");
        return advise == null || "none".equals(advise);
    }

    /** The repacked copy, created on first use; null when repacking is not allowed. */
    private MemorySegment repackedSegment() {
        MemorySegment r = repacked;
        if (r != null || repackDeclined) return r;
        synchronized (this) {
            if (repacked != null || repackDeclined) return repacked;
            if (FloatVector.SPECIES_PREFERRED.length() < 8 || size % BLOCK_SIZE != 0 || !repackAllowed()) {
                repackDeclined = true;
                return null;
            }
            long blocks = size / BLOCK_SIZE;
            MemorySegment dst;
            try {
                dst = Arena.ofAuto().allocate(blocks * REPACKED_BYTES, 32);
            } catch (Throwable oom) {
                repackDeclined = true;
                return null;
            }
            MatmulPool pool = MatmulPool.enabled() ? MatmulPool.get() : null;
            int units = (int) Math.min(Integer.MAX_VALUE, (blocks + 255) / 256);
            MatmulPool.RangeTask task = (from, to) -> {
                long bEnd = Math.min(blocks, (long) to * 256);
                for (long b = (long) from * 256; b < bEnd; b++) repackBlock(b, dst);
            };
            if (pool != null) pool.parallelFor(units, 1, task);
            else task.run(0, units);
            repacked = dst;
            return dst;
        }
    }

    private void repackBlock(long b, MemorySegment dst) {
        long bo = b * BLOCK_BYTES;
        long ro = b * REPACKED_BYTES;
        for (int half = 0; half < 2; half++) {
            long ql = bo + half * 64L;
            long qh = bo + 128L + half * 32L;
            long out = ro + half * 128L;
            for (int l = 0; l < 32; l++) {
                int lo = segment.get(BYTE_LE, ql + l) & 0xFF;
                int hi = segment.get(BYTE_LE, ql + 32 + l) & 0xFF;
                int h = segment.get(BYTE_LE, qh + l) & 0xFF;
                dst.set(BYTE_LE, out + l,      (byte) (((lo & 0x0F) | ((h & 3) << 4)) - 32));
                dst.set(BYTE_LE, out + 32 + l, (byte) (((hi & 0x0F) | (((h >> 2) & 3) << 4)) - 32));
                dst.set(BYTE_LE, out + 64 + l, (byte) (((lo >> 4) | (((h >> 4) & 3) << 4)) - 32));
                dst.set(BYTE_LE, out + 96 + l, (byte) (((hi >> 4) | (((h >> 6) & 3) << 4)) - 32));
            }
        }
        for (int j = 0; j < 16; j++) dst.set(BYTE_LE, ro + 256 + j, segment.get(BYTE_LE, bo + 192 + j));
        dst.set(FLOAT_LE, ro + 272, Float.float16ToFloat(segment.get(SHORT_LE, bo + 208)));
    }

    /** 8 int8 values at {@code offset} of {@code seg}, widened to float. */
    private static FloatVector int8x8(MemorySegment seg, long offset) {
        ByteVector v = ByteVector.fromMemorySegment(B_SPECIES, seg, offset, BYTE_ORDER);
        return (FloatVector) ((IntVector) v.convertShape(VectorOperators.B2I, I_SPECIES, 0))
            .convertShape(VectorOperators.I2F, F_SPECIES, 0);
    }

    @Override
    public void matmulRows(float[] input, float[] out, int rowFrom, int rowTo, int cols) {
        MemorySegment r = cols % BLOCK_SIZE == 0 ? repackedSegment() : null;
        if (r == null) {
            super.matmulRows(input, out, rowFrom, rowTo, cols);
            return;
        }
        int numBlocks = cols / BLOCK_SIZE;
        for (int row = rowFrom; row < rowTo; row++) {
            out[row] += dotRepacked(r, (long) row * numBlocks * REPACKED_BYTES, input, numBlocks);
        }
    }

    private static float dotRepacked(MemorySegment r, long start, float[] x, int numBlocks) {
        FloatVector acc0 = FloatVector.zero(F_SPECIES);
        FloatVector acc1 = FloatVector.zero(F_SPECIES);
        for (int b = 0; b < numBlocks; b++) {
            long ro = start + (long) b * REPACKED_BYTES;
            float d = r.get(FLOAT_LE, ro + 272);
            int xb = b * BLOCK_SIZE;
            for (int j = 0; j < 16; j += 2) {
                long o = ro + j * 16L;
                int xo = xb + j * 16;
                FloatVector p0 = int8x8(r, o).mul(FloatVector.fromArray(F_SPECIES, x, xo));
                FloatVector p1 = int8x8(r, o + 16).mul(FloatVector.fromArray(F_SPECIES, x, xo + 16));
                p0 = int8x8(r, o + 8).fma(FloatVector.fromArray(F_SPECIES, x, xo + 8), p0);
                p1 = int8x8(r, o + 24).fma(FloatVector.fromArray(F_SPECIES, x, xo + 24), p1);
                acc0 = p0.fma(FloatVector.broadcast(F_SPECIES, d * r.get(BYTE_LE, ro + 256 + j)), acc0);
                acc1 = p1.fma(FloatVector.broadcast(F_SPECIES, d * r.get(BYTE_LE, ro + 257 + j)), acc1);
            }
        }
        return acc0.add(acc1).reduceLanes(VectorOperators.ADD);
    }

    /**
     * Multi-token row-range matmul for batched prefill. Tokens are taken four at a time: each
     * 8-weight vector is converted and scaled once, then FMA'd against the four inputs. Uses the
     * repacked layout when available, the original one otherwise. Leftover tokens use the
     * single-token path.
     */
    @Override
    public void matmulRowsBatch(float[][] in, float[][] out, int n, int rowFrom, int rowTo, int cols) {
        if (FloatVector.SPECIES_PREFERRED.length() < 8 || cols % BLOCK_SIZE != 0) {
            super.matmulRowsBatch(in, out, n, rowFrom, rowTo, cols);
            return;
        }
        MemorySegment r = repackedSegment();
        int numBlocks = cols / BLOCK_SIZE;
        float[] res = new float[4];
        for (int row = rowFrom; row < rowTo; row++) {
            // Groups of four; a short last group repeats its last input and keeps only the real
            // results. The single-token kernels are deliberately never called here: compiled by C2
            // during prefill from these few calls, they kept a poor profile and made the following
            // decode up to 40 % slower.
            for (int t = 0; t < n; t += 4) {
                int t1 = Math.min(t + 1, n - 1), t2 = Math.min(t + 2, n - 1), t3 = Math.min(t + 3, n - 1);
                if (r != null) {
                    dot4Repacked(r, (long) row * numBlocks * REPACKED_BYTES, in[t], in[t1], in[t2], in[t3], numBlocks, res);
                } else {
                    dot4((long) row * numBlocks * BLOCK_BYTES, in[t], in[t1], in[t2], in[t3], numBlocks, res);
                }
                out[t][row] += res[0];
                if (t + 1 < n) out[t + 1][row] += res[1];
                if (t + 2 < n) out[t + 2][row] += res[2];
                if (t + 3 < n) out[t + 3][row] += res[3];
            }
        }
    }

    private static void dot4Repacked(MemorySegment r, long start, float[] x0, float[] x1, float[] x2, float[] x3,
                                     int numBlocks, float[] res) {
        FloatVector a0 = FloatVector.zero(F_SPECIES);
        FloatVector a1 = a0, a2 = a0, a3 = a0;
        for (int b = 0; b < numBlocks; b++) {
            long ro = start + (long) b * REPACKED_BYTES;
            float d = r.get(FLOAT_LE, ro + 272);
            int xb = b * BLOCK_SIZE;
            for (int j = 0; j < 16; j++) {
                float scale = d * r.get(BYTE_LE, ro + 256 + j);
                for (int k = 0; k < 16; k += F_LEN) {
                    FloatVector w = int8x8(r, ro + j * 16L + k).mul(scale);
                    int xo = xb + j * 16 + k;
                    a0 = w.fma(FloatVector.fromArray(F_SPECIES, x0, xo), a0);
                    a1 = w.fma(FloatVector.fromArray(F_SPECIES, x1, xo), a1);
                    a2 = w.fma(FloatVector.fromArray(F_SPECIES, x2, xo), a2);
                    a3 = w.fma(FloatVector.fromArray(F_SPECIES, x3, xo), a3);
                }
            }
        }
        res[0] = a0.reduceLanes(VectorOperators.ADD);
        res[1] = a1.reduceLanes(VectorOperators.ADD);
        res[2] = a2.reduceLanes(VectorOperators.ADD);
        res[3] = a3.reduceLanes(VectorOperators.ADD);
    }

    private void dot4(long blockStart, float[] x0, float[] x1, float[] x2, float[] x3, int numBlocks, float[] res) {
        IntVector vMask4 = IntVector.broadcast(I_SPECIES, 0x0F);
        IntVector vMask2 = IntVector.broadcast(I_SPECIES, 0x03);
        IntVector vSub32 = IntVector.broadcast(I_SPECIES, 32);
        FloatVector a0 = FloatVector.zero(F_SPECIES);
        FloatVector a1 = a0, a2 = a0, a3 = a0;
        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float d = Float.float16ToFloat(segment.get(SHORT_LE, bo + 208));
            for (int half = 0; half < 2; half++) {
                long qlBase = bo + half * 64L;
                long qhBase = bo + 128L + half * 32L;
                long scBase = bo + 192L + half * 8L;
                int elemBase = b * BLOCK_SIZE + half * 128;
                for (int l = 0; l < 32; l += F_LEN) {
                    IntVector qh = (IntVector) ByteVector.fromMemorySegment(B_SPECIES, segment, qhBase + l, BYTE_ORDER)
                        .convertShape(VectorOperators.B2I, I_SPECIES, 0);
                    IntVector qlLo = (IntVector) ByteVector.fromMemorySegment(B_SPECIES, segment, qlBase + l, BYTE_ORDER)
                        .convertShape(VectorOperators.B2I, I_SPECIES, 0);
                    IntVector qlHi = (IntVector) ByteVector.fromMemorySegment(B_SPECIES, segment, qlBase + 32L + l, BYTE_ORDER)
                        .convertShape(VectorOperators.B2I, I_SPECIES, 0);
                    int scIdx = l >> 4;
                    IntVector hs = qh;
                    for (int sub = 0; sub < 4; sub++, hs = hs.lanewise(VectorOperators.LSHR, 2)) {
                        IntVector low4 = (sub < 2 ? (sub == 0 ? qlLo : qlHi) : (sub == 2 ? qlLo : qlHi).lanewise(VectorOperators.LSHR, 4)).and(vMask4);
                        IntVector high2 = hs.and(vMask2).lanewise(VectorOperators.LSHL, 4);
                        FloatVector w = ((FloatVector) low4.or(high2).sub(vSub32).convertShape(VectorOperators.I2F, F_SPECIES, 0))
                            .mul(d * segment.get(BYTE_LE, scBase + 2 * sub + scIdx));
                        int xo = elemBase + 32 * sub + l;
                        a0 = w.fma(FloatVector.fromArray(F_SPECIES, x0, xo), a0);
                        a1 = w.fma(FloatVector.fromArray(F_SPECIES, x1, xo), a1);
                        a2 = w.fma(FloatVector.fromArray(F_SPECIES, x2, xo), a2);
                        a3 = w.fma(FloatVector.fromArray(F_SPECIES, x3, xo), a3);
                    }
                }
            }
        }
        res[0] = a0.reduceLanes(VectorOperators.ADD);
        res[1] = a1.reduceLanes(VectorOperators.ADD);
        res[2] = a2.reduceLanes(VectorOperators.ADD);
        res[3] = a3.reduceLanes(VectorOperators.ADD);
    }
}
