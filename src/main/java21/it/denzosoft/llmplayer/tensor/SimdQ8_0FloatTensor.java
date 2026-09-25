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
 * SIMD-optimized Q8_0 tensor using lane-parallel B2I/I2F path.
 *
 * Rewritten 2026-04-15 after JFR flagged {@code SimdQ8_0FloatTensor.dot} as THE
 * hotspot on Qwen3-4B-Thinking Q8_0 (15391 samples, 7× any other method). Old
 * version had a scalar {@code for j in F_LEN} inner loop converting int8 quants
 * to float via a {@code float[F_LEN]} scratch, then {@code FloatVector.fromArray}.
 * New version: {@code ByteVector.fromMemorySegment} → {@code B2I} → {@code I2F}
 * → FMA, zero scratch allocations, zero byte[] copies.
 *
 * Q8_0 block layout (34 bytes, 32 elements):
 *   scale (fp16, 2 bytes)
 *   qs[32]: signed int8 quants
 */
public class SimdQ8_0FloatTensor extends Q8_0FloatTensor {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_256;   // 8 floats
    private static final VectorSpecies<Integer> I_SPECIES = IntVector.SPECIES_256;   // 8 ints
    private static final VectorSpecies<Byte> B_SPECIES = ByteVector.SPECIES_64;      // 8 bytes
    private static final int F_LEN = 8;
    private static final int BLOCK_SIZE = 32;
    private static final int BLOCK_BYTES = 34;
    private static final ValueLayout.OfByte BYTE_LE = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED;
    private static final ByteOrder BYTE_ORDER = ByteOrder.LITTLE_ENDIAN;

    private final MemorySegment segment;

    public SimdQ8_0FloatTensor(TensorData data, long size) {
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

        // Raw q.x per block in two short chains, scale applied once per 32 elements; blocks
        // alternate between two accumulators. The previous form multiplied scale into every input
        // vector and fed a single serial FMA chain. Measured +50 % single-thread on AVX2.
        // Keep this shape: a single-accumulator variant of the same loop measured 5x slower
        // (the Vector API intrinsics stopped applying).
        FloatVector acc0 = FloatVector.zero(F_SPECIES);
        FloatVector acc1 = FloatVector.zero(F_SPECIES);
        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            int xo = otherOffset + b * BLOCK_SIZE;
            FloatVector p0 = quants(bo + 2).mul(FloatVector.fromArray(F_SPECIES, other, xo));
            FloatVector p1 = quants(bo + 10).mul(FloatVector.fromArray(F_SPECIES, other, xo + 8));
            p0 = quants(bo + 18).fma(FloatVector.fromArray(F_SPECIES, other, xo + 16), p0);
            p1 = quants(bo + 26).fma(FloatVector.fromArray(F_SPECIES, other, xo + 24), p1);
            FloatVector scale = FloatVector.broadcast(F_SPECIES, Float.float16ToFloat(segment.get(SHORT_LE, bo)));
            if ((b & 1) == 0) {
                acc0 = p0.add(p1).fma(scale, acc0);
            } else {
                acc1 = p0.add(p1).fma(scale, acc1);
            }
        }
        return acc0.add(acc1).reduceLanes(VectorOperators.ADD);
    }

    /**
     * Multi-token row-range matmul for batched prefill: each 8-weight vector is converted and
     * scaled once and FMA'd against four inputs. Leftover tokens use {@link #dot}.
     */
    @Override
    public void matmulRowsBatch(float[][] in, float[][] out, int n, int rowFrom, int rowTo, int cols) {
        if (FloatVector.SPECIES_PREFERRED.length() < 8 || cols % BLOCK_SIZE != 0) {
            super.matmulRowsBatch(in, out, n, rowFrom, rowTo, cols);
            return;
        }
        int numBlocks = cols / BLOCK_SIZE;
        float[] res = new float[4];
        for (int row = rowFrom; row < rowTo; row++) {
            long blockStart = (long) row * numBlocks * BLOCK_BYTES;
            // Groups of four; a short last group repeats its last input and keeps only the real
            // results. The single-token kernels are deliberately never called here: compiled by C2
            // during prefill from these few calls, they kept a poor profile and made the following
            // decode up to 40 % slower.
            for (int t = 0; t < n; t += 4) {
                int t1 = Math.min(t + 1, n - 1), t2 = Math.min(t + 2, n - 1), t3 = Math.min(t + 3, n - 1);
                dot4(blockStart, in[t], in[t1], in[t2], in[t3], numBlocks, res);
                out[t][row] += res[0];
                if (t + 1 < n) out[t + 1][row] += res[1];
                if (t + 2 < n) out[t + 2][row] += res[2];
                if (t + 3 < n) out[t + 3][row] += res[3];
            }
        }
    }

    private void dot4(long blockStart, float[] x0, float[] x1, float[] x2, float[] x3, int numBlocks, float[] res) {
        FloatVector a0 = FloatVector.zero(F_SPECIES);
        FloatVector a1 = a0, a2 = a0, a3 = a0;
        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float scale = Float.float16ToFloat(segment.get(SHORT_LE, bo));
            int xb = b * BLOCK_SIZE;
            for (int i = 0; i < BLOCK_SIZE; i += F_LEN) {
                FloatVector w = quants(bo + 2 + i).mul(scale);
                int xo = xb + i;
                a0 = w.fma(FloatVector.fromArray(F_SPECIES, x0, xo), a0);
                a1 = w.fma(FloatVector.fromArray(F_SPECIES, x1, xo), a1);
                a2 = w.fma(FloatVector.fromArray(F_SPECIES, x2, xo), a2);
                a3 = w.fma(FloatVector.fromArray(F_SPECIES, x3, xo), a3);
            }
        }
        res[0] = a0.reduceLanes(VectorOperators.ADD);
        res[1] = a1.reduceLanes(VectorOperators.ADD);
        res[2] = a2.reduceLanes(VectorOperators.ADD);
        res[3] = a3.reduceLanes(VectorOperators.ADD);
    }

    /** 8 int8 quants at {@code offset}, widened to float. */
    private FloatVector quants(long offset) {
        ByteVector vqb = ByteVector.fromMemorySegment(B_SPECIES, segment, offset, BYTE_ORDER);
        IntVector vqI = (IntVector) vqb.convertShape(VectorOperators.B2I, I_SPECIES, 0);
        return (FloatVector) vqI.convertShape(VectorOperators.I2F, F_SPECIES, 0);
    }

    @Override
    public float dot(long thisOffset, FloatTensor other, long otherOffset, int length) {
        if (other instanceof SimdQ8_0FloatTensor) {
            return dotQ8Q8Simd(thisOffset, (SimdQ8_0FloatTensor) other, otherOffset, length);
        }
        return super.dot(thisOffset, other, otherOffset, length);
    }

    /**
     * Q8_0 × Q8_0 with lane-parallel integer accumulation.
     */
    private float dotQ8Q8Simd(long thisOffset, SimdQ8_0FloatTensor other, long otherOffset, int length) {
        if (FloatVector.SPECIES_PREFERRED.length() < 8 || length % BLOCK_SIZE != 0) {
            return super.dot(thisOffset, (FloatTensor) other, otherOffset, length);
        }

        int numBlocks = length / BLOCK_SIZE;
        long thisBlock = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        long otherBlock = (otherOffset / BLOCK_SIZE) * BLOCK_BYTES;
        MemorySegment otherSeg = other.segment;

        float result = 0f;

        for (int b = 0; b < numBlocks; b++) {
            float d0 = Float.float16ToFloat(segment.get(SHORT_LE, thisBlock));
            float d1 = Float.float16ToFloat(otherSeg.get(SHORT_LE, otherBlock));
            long qs0 = thisBlock + 2;
            long qs1 = otherBlock + 2;

            IntVector isum = IntVector.zero(I_SPECIES);
            for (int i = 0; i < BLOCK_SIZE; i += F_LEN) {
                ByteVector v0 = ByteVector.fromMemorySegment(B_SPECIES, segment, qs0 + i, BYTE_ORDER);
                ByteVector v1 = ByteVector.fromMemorySegment(B_SPECIES, otherSeg, qs1 + i, BYTE_ORDER);
                IntVector i0 = (IntVector) v0.convertShape(VectorOperators.B2I, I_SPECIES, 0);
                IntVector i1 = (IntVector) v1.convertShape(VectorOperators.B2I, I_SPECIES, 0);
                isum = isum.add(i0.mul(i1));
            }
            result += d0 * d1 * isum.reduceLanes(VectorOperators.ADD);

            thisBlock += BLOCK_BYTES;
            otherBlock += BLOCK_BYTES;
        }
        return result;
    }
}
