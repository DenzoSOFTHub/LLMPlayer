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
}
