package it.denzosoft.llmplayer.benchmark;

import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.GGMLType;
import it.denzosoft.llmplayer.tensor.TensorData;
import it.denzosoft.llmplayer.inference.VirtualThreadMatmul;

import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * Microbenchmark comparing matmul implementations:
 *   1. Sequential (single-thread) — baseline (simulates Java 1.8 without parallelism)
 *   2. ForkJoinPool (IntStream.parallel) — Java 8+ parallel path
 *   3. Virtual Threads (Java 25) — new dedicated executor path
 *   4. GPU (OpenCL) — if available
 *
 * Run: java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED --enable-preview \
 *        -cp target/classes it.denzosoft.llmplayer.benchmark.MatmulBenchmark
 */
public final class MatmulBenchmark {

    private static final int[][] SIZES = {
        {512, 512},
        {1024, 1024},
        {2048, 2048},
        {4096, 4096},
        {4096, 11008},  // typical Llama FFN
    };

    private static final int WARMUP = 8;
    private static final int ITERATIONS = 30;

    public static void main(String[] args) {
        System.out.println("╔══════════════════════════════════════════════════════════════╗");
        System.out.println("║          LLMPlayer Matmul Performance Benchmark             ║");
        System.out.println("╚══════════════════════════════════════════════════════════════╝");
        System.out.println();
        System.out.println("Java version:    " + System.getProperty("java.version"));
        System.out.println("VM:              " + System.getProperty("java.vm.name") + " " + System.getProperty("java.vm.version"));
        System.out.println("Processors:      " + Runtime.getRuntime().availableProcessors());
        System.out.println("Max memory:      " + (Runtime.getRuntime().maxMemory() / (1024 * 1024)) + " MB");
        System.out.println("Warmup iters:    " + WARMUP);
        System.out.println("Measured iters:  " + ITERATIONS);
        System.out.println();

        // Check GPU availability
        boolean gpuAvailable = false;
        Object gpuTensorFactory = null;
        try {
            // Try to initialize OpenCL
            Class<?> ctxClass = Class.forName("it.denzosoft.llmplayer.gpu.OpenCLContext");
            Method enumMethod = ctxClass.getMethod("enumerateDevices");
            java.util.List<?> devices = (java.util.List<?>) enumMethod.invoke(null);
            if (!devices.isEmpty()) {
                System.out.println("GPU detected:    " + devices.get(0));
                // Create context for device 0
                Method createMethod = ctxClass.getMethod("create", int.class);
                Object clContext = createMethod.invoke(null, 0);
                // Create GpuBufferManager
                Class<?> bufMgrClass = Class.forName("it.denzosoft.llmplayer.gpu.GpuBufferManager");
                gpuTensorFactory = bufMgrClass.getConstructor(ctxClass).newInstance(clContext);
                gpuAvailable = true;
            } else {
                System.out.println("GPU:             No OpenCL devices found");
            }
        } catch (ClassNotFoundException e) {
            System.out.println("GPU:             OpenCL classes not compiled (java21 sources missing)");
        } catch (Exception e) {
            String msg = e.getCause() != null ? e.getCause().getMessage() : e.getMessage();
            System.out.println("GPU:             Init failed — " + msg);
        }
        System.out.println();

        System.out.println("Legend:");
        System.out.println("  [SEQ]  Sequential single-thread (= Java 1.8 senza parallelismo, senza GPU)");
        System.out.println("  [FJP]  ForkJoinPool IntStream.parallel (= Java 8+/21 CPU path)");
        System.out.println("  [VT]   Virtual Threads dedicated executor (= Java 25 CPU path)");
        if (gpuAvailable) {
            System.out.println("  [GPU]  OpenCL GPU kernel");
        }
        System.out.println();

        // Print header
        System.out.printf("%-18s │ %10s │ %10s  %7s │ %10s  %7s  %7s",
            "Size", "SEQ (ms)", "FJP (ms)", "vs SEQ", "VT (ms)", "vs SEQ", "vs FJP");
        if (gpuAvailable) {
            System.out.printf(" │ %10s  %7s", "GPU (ms)", "vs SEQ");
        }
        System.out.println();
        System.out.println("─".repeat(gpuAvailable ? 115 : 90));

        for (int[] size : SIZES) {
            int rows = size[0];
            int cols = size[1];

            // Create test data
            Random rng = new Random(42);
            float[] weightData = new float[rows * cols];
            float[] input = new float[cols];
            for (int i = 0; i < weightData.length; i++) weightData[i] = (rng.nextFloat() - 0.5f) * 0.1f;
            for (int i = 0; i < cols; i++) input[i] = (rng.nextFloat() - 0.5f) * 0.1f;

            FloatTensor weights = createF32Tensor(weightData);

            // 1. Sequential
            double seqMs = benchmarkSequential(weights, input, rows, cols);

            // 2. ForkJoinPool
            double fjpMs = benchmarkForkJoin(weights, input, rows, cols);

            // 3. Virtual Threads
            double vtMs = benchmarkVirtualThreads(weights, input, rows, cols);

            // 4. GPU
            double gpuMs = -1;
            if (gpuAvailable) {
                gpuMs = benchmarkGpu(weightData, input, rows, cols, gpuTensorFactory);
            }

            // Verify correctness
            float[] outSeq = new float[rows];
            float[] outFjp = new float[rows];
            float[] outVt = new float[rows];
            weights.matmul(input, outSeq, rows, cols);
            IntStream.range(0, rows).parallel().forEach(row ->
                outFjp[row] += weights.dot((long) row * cols, input, 0, cols));
            VirtualThreadMatmul.matmul(weights, input, outVt, rows, cols);
            double maxErr = 0;
            for (int i = 0; i < rows; i++) {
                maxErr = Math.max(maxErr, Math.abs(outSeq[i] - outFjp[i]));
                maxErr = Math.max(maxErr, Math.abs(outSeq[i] - outVt[i]));
            }

            // Print row
            String sizeStr = rows + " x " + cols;
            System.out.printf("%-18s │ %10.2f │ %10.2f  %6.1fx  │ %10.2f  %6.1fx  %6.2fx",
                sizeStr, seqMs, fjpMs, seqMs / fjpMs, vtMs, seqMs / vtMs, fjpMs / vtMs);
            if (gpuAvailable && gpuMs > 0) {
                System.out.printf(" │ %10.2f  %6.1fx", gpuMs, seqMs / gpuMs);
            } else if (gpuAvailable) {
                System.out.printf(" │ %10s  %7s", "N/A", "-");
            }
            System.out.println();

            if (maxErr > 1e-4) {
                System.out.printf("  ⚠ max numerical error: %.2e%n", maxErr);
            }
        }

        System.out.println();
        System.out.println("Interpretation:");
        System.out.println("  vs SEQ > 1.0x = faster than sequential (simulates Java 1.8 without threads)");
        System.out.println("  vs FJP > 1.0x = Virtual Threads faster than ForkJoinPool");
        System.out.println("  vs FJP < 1.0x = ForkJoinPool faster (typical for pure compute-bound)");

        // Cleanup GPU
        if (gpuTensorFactory != null) {
            try { ((AutoCloseable) gpuTensorFactory).close(); } catch (Exception ignored) {}
        }
    }

    private static double benchmarkSequential(FloatTensor weights, float[] input, int rows, int cols) {
        float[] out = new float[rows];
        for (int i = 0; i < WARMUP; i++) {
            Arrays.fill(out, 0);
            weights.matmul(input, out, rows, cols);
        }
        long total = 0;
        for (int i = 0; i < ITERATIONS; i++) {
            Arrays.fill(out, 0);
            long start = System.nanoTime();
            weights.matmul(input, out, rows, cols);
            total += System.nanoTime() - start;
        }
        return total / (ITERATIONS * 1_000_000.0);
    }

    private static double benchmarkForkJoin(FloatTensor weights, float[] input, int rows, int cols) {
        float[] out = new float[rows];
        for (int i = 0; i < WARMUP; i++) {
            Arrays.fill(out, 0);
            IntStream.range(0, rows).parallel().forEach(row ->
                out[row] += weights.dot((long) row * cols, input, 0, cols));
        }
        long total = 0;
        for (int i = 0; i < ITERATIONS; i++) {
            Arrays.fill(out, 0);
            long start = System.nanoTime();
            IntStream.range(0, rows).parallel().forEach(row ->
                out[row] += weights.dot((long) row * cols, input, 0, cols));
            total += System.nanoTime() - start;
        }
        return total / (ITERATIONS * 1_000_000.0);
    }

    private static double benchmarkVirtualThreads(FloatTensor weights, float[] input, int rows, int cols) {
        float[] out = new float[rows];
        for (int i = 0; i < WARMUP; i++) {
            Arrays.fill(out, 0);
            VirtualThreadMatmul.matmul(weights, input, out, rows, cols);
        }
        long total = 0;
        for (int i = 0; i < ITERATIONS; i++) {
            Arrays.fill(out, 0);
            long start = System.nanoTime();
            VirtualThreadMatmul.matmul(weights, input, out, rows, cols);
            total += System.nanoTime() - start;
        }
        return total / (ITERATIONS * 1_000_000.0);
    }

    @SuppressWarnings("unchecked")
    private static double benchmarkGpu(float[] weightData, float[] input, int rows, int cols, Object bufMgr) {
        try {
            // Create an F32GpuTensor via reflection
            Class<?> gpuTensorClass = Class.forName("it.denzosoft.llmplayer.tensor.F32GpuTensor");
            Class<?> bufMgrClass = Class.forName("it.denzosoft.llmplayer.gpu.GpuBufferManager");
            Class<?> tensorDataClass = Class.forName("it.denzosoft.llmplayer.tensor.TensorData");

            // We need a TensorData for the weight data
            ByteBuffer buf = ByteBuffer.allocateDirect(weightData.length * 4).order(ByteOrder.LITTLE_ENDIAN);
            for (float f : weightData) buf.putFloat(f);
            buf.flip();

            TensorData td = createDirectTensorData(buf);

            FloatTensor gpuWeights = (FloatTensor) gpuTensorClass
                .getConstructor(tensorDataClass, long.class, bufMgrClass)
                .newInstance(td, (long) weightData.length, bufMgr);

            float[] out = new float[rows];
            // Warmup
            for (int i = 0; i < WARMUP; i++) {
                Arrays.fill(out, 0);
                gpuWeights.matmulParallel(input, out, rows, cols);
            }
            // Measure
            long total = 0;
            for (int i = 0; i < ITERATIONS; i++) {
                Arrays.fill(out, 0);
                long start = System.nanoTime();
                gpuWeights.matmulParallel(input, out, rows, cols);
                total += System.nanoTime() - start;
            }
            return total / (ITERATIONS * 1_000_000.0);
        } catch (Exception e) {
            return -1;
        }
    }

    private static FloatTensor createF32Tensor(float[] data) {
        ByteBuffer buf = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float f : data) buf.putFloat(f);
        buf.flip();

        TensorData td = createHeapTensorData(buf);

        return new FloatTensor(td, data.length) {
            @Override public float getFloat(long index) { return td.getFloatLE(index * 4); }
            @Override public GGMLType type() { return GGMLType.F32; }
        };
    }

    private static TensorData createHeapTensorData(ByteBuffer buf) {
        return new TensorData() {
            @Override public byte getByte(long offset) { return buf.get((int) offset); }
            @Override public short getShortLE(long offset) { return buf.getShort((int) offset); }
            @Override public int getIntLE(long offset) { return buf.getInt((int) offset); }
            @Override public long getLongLE(long offset) { return buf.getLong((int) offset); }
            @Override public float getFloatLE(long offset) { return buf.getFloat((int) offset); }
            @Override public double getDoubleLE(long offset) { return buf.getDouble((int) offset); }
            @Override public void copyBytes(long srcOffset, byte[] dst, int dstOffset, int length) {
                ByteBuffer dup = buf.duplicate();
                dup.position((int) srcOffset);
                dup.get(dst, dstOffset, length);
            }
            @Override public TensorData slice(long offset, long size) { return this; }
            @Override public long byteSize() { return buf.capacity(); }
        };
    }

    private static TensorData createDirectTensorData(ByteBuffer buf) {
        return new TensorData() {
            @Override public byte getByte(long offset) { return buf.get((int) offset); }
            @Override public short getShortLE(long offset) { return buf.getShort((int) offset); }
            @Override public int getIntLE(long offset) { return buf.getInt((int) offset); }
            @Override public long getLongLE(long offset) { return buf.getLong((int) offset); }
            @Override public float getFloatLE(long offset) { return buf.getFloat((int) offset); }
            @Override public double getDoubleLE(long offset) { return buf.getDouble((int) offset); }
            @Override public void copyBytes(long srcOffset, byte[] dst, int dstOffset, int length) {
                ByteBuffer dup = buf.duplicate();
                dup.position((int) srcOffset);
                dup.get(dst, dstOffset, length);
            }
            @Override public TensorData slice(long offset, long size) { return this; }
            @Override public long byteSize() { return buf.capacity(); }
        };
    }
}
