package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.tensor.FloatTensor;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Matrix-vector multiply using a dedicated virtual thread executor.
 * Avoids contention on ForkJoinPool.commonPool() when multiple components
 * parallelize concurrently (e.g. batch generation + matmul).
 *
 * Loaded via reflection from FloatTensor when running on Java 25+.
 */
public final class VirtualThreadMatmul {

    private static final ExecutorService EXECUTOR = Executors.newVirtualThreadPerTaskExecutor();

    private VirtualThreadMatmul() {}

    /**
     * Parallel matmul using virtual threads.
     * out[row] += dot(weights[row * cols ..], input, cols)
     */
    public static void matmul(FloatTensor weights, float[] input, float[] out, int rows, int cols) {
        int chunkSize = Math.max(1, rows / Runtime.getRuntime().availableProcessors());
        List<Future<?>> futures = new ArrayList<>();
        for (int start = 0; start < rows; start += chunkSize) {
            int from = start;
            int to = Math.min(start + chunkSize, rows);
            futures.add(EXECUTOR.submit(() -> {
                for (int row = from; row < to; row++) {
                    out[row] += weights.dot((long) row * cols, input, 0, cols);
                }
            }));
        }
        for (Future<?> f : futures) {
            try {
                f.get();
            } catch (Exception e) {
                throw new RuntimeException("Virtual thread matmul failed", e);
            }
        }
    }

    /**
     * Fused parallel matmul for two weight matrices sharing the same input.
     * Processes both gate and up (or any two projections) in a single parallel dispatch,
     * keeping the input vector in L1 cache and eliminating one synchronization barrier.
     */
    public static void fusedMatmul(FloatTensor w1, FloatTensor w2,
            float[] input, float[] out1, float[] out2, int rows, int cols) {
        int chunkSize = Math.max(1, rows / Runtime.getRuntime().availableProcessors());
        List<Future<?>> futures = new ArrayList<>();
        for (int start = 0; start < rows; start += chunkSize) {
            int from = start;
            int to = Math.min(start + chunkSize, rows);
            futures.add(EXECUTOR.submit(() -> {
                for (int row = from; row < to; row++) {
                    out1[row] += w1.dot((long) row * cols, input, 0, cols);
                    out2[row] += w2.dot((long) row * cols, input, 0, cols);
                }
            }));
        }
        for (Future<?> f : futures) {
            try {
                f.get();
            } catch (Exception e) {
                throw new RuntimeException("Virtual thread fused matmul failed", e);
            }
        }
    }

    /**
     * Fused parallel matmul for three weight matrices (Q, K, V) sharing the same input.
     * Handles different row counts (qDim != kvDim for GQA).
     */
    public static void fusedMatmulQKV(FloatTensor wq, FloatTensor wk, FloatTensor wv,
            float[] input, float[] q, float[] k, float[] v,
            int qRows, int kvRows, int cols) {
        int maxRows = Math.max(qRows, kvRows);
        int chunkSize = Math.max(1, maxRows / Runtime.getRuntime().availableProcessors());
        List<Future<?>> futures = new ArrayList<>();
        for (int start = 0; start < maxRows; start += chunkSize) {
            int from = start;
            int to = Math.min(start + chunkSize, maxRows);
            futures.add(EXECUTOR.submit(() -> {
                for (int row = from; row < to; row++) {
                    if (row < qRows) {
                        q[row] += wq.dot((long) row * cols, input, 0, cols);
                    }
                    if (row < kvRows) {
                        k[row] += wk.dot((long) row * cols, input, 0, cols);
                        v[row] += wv.dot((long) row * cols, input, 0, cols);
                    }
                }
            }));
        }
        for (Future<?> f : futures) {
            try {
                f.get();
            } catch (Exception e) {
                throw new RuntimeException("Virtual thread fused QKV matmul failed", e);
            }
        }
    }
}
