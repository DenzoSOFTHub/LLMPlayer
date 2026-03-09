package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.GGMLType;
import it.denzosoft.llmplayer.tensor.MemorySegmentTensorData;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Cache-friendly tiled matrix-vector multiply.
 * Processes ROW_TILE rows simultaneously, sharing input vector reads from L1 cache.
 * Enabled via -Dmatmul.tiled=true. Falls back to standard matmul for unsupported types.
 *
 * Key insight: standard matmul processes one row at a time, reading the entire input vector
 * per row. Tiled matmul reads each input tile once and processes ROW_TILE rows with it,
 * improving L1/L2 cache reuse for the input vector and weight data.
 *
 * Loaded via reflection from FloatTensor. Requires Java 21+ (Vector API + MemorySegment).
 */
public final class TiledMatmul {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int F_LEN = F_SPECIES.length();
    private static final int ROW_TILE = 4; // process 4 rows simultaneously

    private static final ValueLayout.OfByte BYTE_LE = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED;

    private static final ExecutorService EXECUTOR = Executors.newVirtualThreadPerTaskExecutor();

    private TiledMatmul() {}

    /**
     * Tiled matmul entry point. Returns true if handled, false if unsupported type.
     * Called via reflection from FloatTensor.
     */
    public static boolean matmul(FloatTensor weights, float[] input, float[] out, int rows, int cols) {
        GGMLType type = weights.type();
        if (type == GGMLType.Q4_K) {
            tiledMatmulParallel(weights, input, out, rows, cols, 256, 144);
            return true;
        }
        if (type == GGMLType.Q8_0) {
            tiledMatmulParallel(weights, input, out, rows, cols, 32, 34);
            return true;
        }
        if (type == GGMLType.Q6_K) {
            tiledMatmulParallel(weights, input, out, rows, cols, 256, 210);
            return true;
        }
        // Unsupported type — fall back to standard matmul
        return false;
    }

    /**
     * Parallel tiled matmul: chunks rows across virtual threads, each chunk uses tiled processing.
     */
    private static void tiledMatmulParallel(FloatTensor weights, float[] input, float[] out,
                                             int rows, int cols, int blockSize, int blockBytes) {
        int chunkSize = Math.max(ROW_TILE, (rows / Runtime.getRuntime().availableProcessors() / ROW_TILE) * ROW_TILE);
        List<Future<?>> futures = new ArrayList<>();

        for (int start = 0; start < rows; start += chunkSize) {
            int from = start;
            int to = Math.min(start + chunkSize, rows);
            futures.add(EXECUTOR.submit(() -> {
                // Process ROW_TILE rows at a time within this chunk
                int row = from;
                for (; row + ROW_TILE <= to; row += ROW_TILE) {
                    tiledDotRows(weights, input, out, row, ROW_TILE, cols);
                }
                // Handle remaining rows (< ROW_TILE)
                for (; row < to; row++) {
                    out[row] += weights.dot((long) row * cols, input, 0, cols);
                }
            }));
        }

        for (Future<?> f : futures) {
            try { f.get(); } catch (Exception e) {
                throw new RuntimeException("Tiled matmul failed", e);
            }
        }
    }

    /**
     * Process ROW_TILE rows simultaneously using tiled dot product.
     * For each column block, reads the input tile once and processes all rows with it.
     */
    private static void tiledDotRows(FloatTensor weights, float[] input, float[] out,
                                      int startRow, int numRows, int cols) {
        GGMLType type = weights.type();
        if (type == GGMLType.Q4_K) {
            tiledDotQ4K(weights, input, out, startRow, numRows, cols);
        } else if (type == GGMLType.Q8_0) {
            tiledDotQ8_0(weights, input, out, startRow, numRows, cols);
        } else if (type == GGMLType.Q6_K) {
            tiledDotQ6K(weights, input, out, startRow, numRows, cols);
        }
    }

    // ==================== Q4_K Tiled Kernel ====================

    private static final int Q4K_BLOCK_SIZE = 256;
    private static final int Q4K_BLOCK_BYTES = 144;

    /**
     * Tiled Q4_K dot product: processes numRows rows simultaneously.
     * For each 256-element column block, reads input[256] once and computes
     * the dot product with numRows weight blocks, accumulating into numRows accumulators.
     */
    private static void tiledDotQ4K(FloatTensor weights, float[] input, float[] out,
                                     int startRow, int numRows, int cols) {
        MemorySegment segment = getSegment(weights);
        if (segment == null) {
            // Fallback for non-MemorySegment tensors
            for (int r = 0; r < numRows; r++) {
                int row = startRow + r;
                out[row] += weights.dot((long) row * cols, input, 0, cols);
            }
            return;
        }

        int numBlocks = cols / Q4K_BLOCK_SIZE;
        FloatVector[] accs = new FloatVector[numRows];
        for (int r = 0; r < numRows; r++) accs[r] = FloatVector.zero(F_SPECIES);

        byte[] qs = new byte[128];
        byte[] sb = new byte[12];
        int[] scales = new int[8];
        int[] mins = new int[8];
        float[] dq = new float[F_LEN];

        for (int b = 0; b < numBlocks; b++) {
            int inputBase = b * Q4K_BLOCK_SIZE;

            for (int r = 0; r < numRows; r++) {
                long bo = ((long) (startRow + r) * cols / Q4K_BLOCK_SIZE + b) * Q4K_BLOCK_BYTES;
                float d = it.denzosoft.llmplayer.tensor.Float16.toFloat(segment.get(SHORT_LE, bo));
                float dmin = it.denzosoft.llmplayer.tensor.Float16.toFloat(segment.get(SHORT_LE, bo + 2));

                MemorySegment.copy(segment, BYTE_LE, bo + 4, sb, 0, 12);
                MemorySegment.copy(segment, BYTE_LE, bo + 16, qs, 0, 128);

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

                for (int group = 0; group < 4; group++) {
                    float ds0 = d * scales[group * 2];
                    float negDm0 = -(dmin * mins[group * 2]);
                    float ds1 = d * scales[group * 2 + 1];
                    float negDm1 = -(dmin * mins[group * 2 + 1]);

                    FloatVector vds0 = FloatVector.broadcast(F_SPECIES, ds0);
                    FloatVector vNegDm0 = FloatVector.broadcast(F_SPECIES, negDm0);
                    FloatVector vds1 = FloatVector.broadcast(F_SPECIES, ds1);
                    FloatVector vNegDm1 = FloatVector.broadcast(F_SPECIES, negDm1);

                    int qsGroupBase = group * 32;
                    int lowInputBase = inputBase + group * 64;
                    int highInputBase = lowInputBase + 32;

                    for (int l = 0; l < 32; l += F_LEN) {
                        for (int j = 0; j < F_LEN; j++) {
                            dq[j] = (float) (Byte.toUnsignedInt(qs[qsGroupBase + l + j]) & 0x0F);
                        }
                        FloatVector vq0 = FloatVector.fromArray(F_SPECIES, dq, 0);
                        FloatVector w0 = vq0.fma(vds0, vNegDm0);
                        FloatVector in0 = FloatVector.fromArray(F_SPECIES, input, lowInputBase + l);
                        accs[r] = w0.fma(in0, accs[r]);

                        for (int j = 0; j < F_LEN; j++) {
                            dq[j] = (float) ((Byte.toUnsignedInt(qs[qsGroupBase + l + j]) >> 4) & 0x0F);
                        }
                        FloatVector vq1 = FloatVector.fromArray(F_SPECIES, dq, 0);
                        FloatVector w1 = vq1.fma(vds1, vNegDm1);
                        FloatVector in1 = FloatVector.fromArray(F_SPECIES, input, highInputBase + l);
                        accs[r] = w1.fma(in1, accs[r]);
                    }
                }
            }
        }

        for (int r = 0; r < numRows; r++) {
            out[startRow + r] += accs[r].reduceLanes(VectorOperators.ADD);
        }
    }

    // ==================== Q8_0 Tiled Kernel ====================

    private static final int Q8_BLOCK_SIZE = 32;
    private static final int Q8_BLOCK_BYTES = 34;

    private static void tiledDotQ8_0(FloatTensor weights, float[] input, float[] out,
                                      int startRow, int numRows, int cols) {
        MemorySegment segment = getSegment(weights);
        if (segment == null) {
            for (int r = 0; r < numRows; r++) {
                int row = startRow + r;
                out[row] += weights.dot((long) row * cols, input, 0, cols);
            }
            return;
        }

        int numBlocks = cols / Q8_BLOCK_SIZE;
        FloatVector[] accs = new FloatVector[numRows];
        for (int r = 0; r < numRows; r++) accs[r] = FloatVector.zero(F_SPECIES);

        for (int b = 0; b < numBlocks; b++) {
            int inputBase = b * Q8_BLOCK_SIZE;

            for (int r = 0; r < numRows; r++) {
                long bo = ((long) (startRow + r) * cols / Q8_BLOCK_SIZE + b) * Q8_BLOCK_BYTES;
                float scale = it.denzosoft.llmplayer.tensor.Float16.toFloat(segment.get(SHORT_LE, bo));
                FloatVector vScale = FloatVector.broadcast(F_SPECIES, scale);

                for (int i = 0; i < Q8_BLOCK_SIZE; i += F_LEN) {
                    // Read int8 quants and convert to float
                    float[] qf = new float[F_LEN];
                    for (int j = 0; j < F_LEN; j++) {
                        qf[j] = segment.get(BYTE_LE, bo + 2 + i + j); // signed byte
                    }
                    FloatVector vq = FloatVector.fromArray(F_SPECIES, qf, 0);
                    FloatVector w = vq.mul(vScale);
                    FloatVector in = FloatVector.fromArray(F_SPECIES, input, inputBase + i);
                    accs[r] = w.fma(in, accs[r]);
                }
            }
        }

        for (int r = 0; r < numRows; r++) {
            out[startRow + r] += accs[r].reduceLanes(VectorOperators.ADD);
        }
    }

    // ==================== Q6_K Tiled Kernel ====================

    private static final int Q6K_BLOCK_SIZE = 256;
    private static final int Q6K_BLOCK_BYTES = 210;

    private static void tiledDotQ6K(FloatTensor weights, float[] input, float[] out,
                                     int startRow, int numRows, int cols) {
        MemorySegment segment = getSegment(weights);
        if (segment == null) {
            for (int r = 0; r < numRows; r++) {
                int row = startRow + r;
                out[row] += weights.dot((long) row * cols, input, 0, cols);
            }
            return;
        }

        int numBlocks = cols / Q6K_BLOCK_SIZE;
        FloatVector[] accs = new FloatVector[numRows];
        for (int r = 0; r < numRows; r++) accs[r] = FloatVector.zero(F_SPECIES);

        byte[] ql = new byte[128];
        byte[] qh = new byte[64];
        byte[] sc = new byte[16];
        float[] dq = new float[F_LEN];

        for (int b = 0; b < numBlocks; b++) {
            int inputBase = b * Q6K_BLOCK_SIZE;

            for (int r = 0; r < numRows; r++) {
                long bo = ((long) (startRow + r) * cols / Q6K_BLOCK_SIZE + b) * Q6K_BLOCK_BYTES;
                float d = it.denzosoft.llmplayer.tensor.Float16.toFloat(segment.get(SHORT_LE, bo + 208));

                MemorySegment.copy(segment, BYTE_LE, bo, ql, 0, 128);
                MemorySegment.copy(segment, BYTE_LE, bo + 128, qh, 0, 64);
                MemorySegment.copy(segment, BYTE_LE, bo + 192, sc, 0, 16);

                for (int half = 0; half < 2; half++) {
                    int qlOff = half * 64;
                    int qhOff = half * 32;
                    int scBase = half * 8;
                    int elemBase = inputBase + half * 128;

                    for (int l = 0; l < 32; l += F_LEN) {
                        // Process F_LEN elements at a time across 4 quadrants
                        for (int j = 0; j < F_LEN; j++) {
                            int idx = l + j;
                            int qlByte0 = Byte.toUnsignedInt(ql[qlOff + idx]);
                            int qlByte1 = Byte.toUnsignedInt(ql[qlOff + 32 + idx]);
                            int qhByte = Byte.toUnsignedInt(qh[qhOff + idx]);

                            int q0 = (qlByte0 & 0x0F)         | (((qhByte >> 0) & 3) << 4);
                            int q1 = (qlByte1 & 0x0F)         | (((qhByte >> 2) & 3) << 4);
                            int q2 = ((qlByte0 >> 4) & 0x0F)  | (((qhByte >> 4) & 3) << 4);
                            int q3 = ((qlByte1 >> 4) & 0x0F)  | (((qhByte >> 6) & 3) << 4);

                            int scIdx = scBase + (idx / 16);
                            float ds0 = d * sc[scIdx];
                            float ds1 = d * sc[scIdx + 2];
                            float ds2 = d * sc[scIdx + 4];
                            float ds3 = d * sc[scIdx + 6];

                            // Accumulate all 4 quadrants for this element position
                            dq[j] = ds0 * (q0 - 32) * input[elemBase + idx]
                                   + ds1 * (q1 - 32) * input[elemBase + 32 + idx]
                                   + ds2 * (q2 - 32) * input[elemBase + 64 + idx]
                                   + ds3 * (q3 - 32) * input[elemBase + 96 + idx];
                        }
                        FloatVector vPartial = FloatVector.fromArray(F_SPECIES, dq, 0);
                        accs[r] = accs[r].add(vPartial);
                    }
                }
            }
        }

        for (int r = 0; r < numRows; r++) {
            out[startRow + r] += accs[r].reduceLanes(VectorOperators.ADD);
        }
    }

    // ==================== Utility ====================

    /**
     * Extract the MemorySegment from a tensor's TensorData.
     * Returns null if not a MemorySegmentTensorData.
     */
    private static MemorySegment getSegment(FloatTensor tensor) {
        if (tensor.data() instanceof MemorySegmentTensorData mstd) {
            return mstd.segment();
        }
        return null;
    }
}
