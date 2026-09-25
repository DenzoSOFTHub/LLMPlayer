package it.denzosoft.llmplayer.tensor;

import java.util.stream.IntStream;

public abstract class FloatTensor {

    protected final TensorData data;
    protected final long size; // number of float elements

    protected FloatTensor(TensorData data, long size) {
        this.data = data;
        this.size = size;
    }

    public long size() { return size; }
    public TensorData data() { return data; }

    public abstract float getFloat(long index);

    public abstract GGMLType type();

    /**
     * Dot product of this tensor starting at thisOffset with a float array.
     * This is the primary computation path - subclasses should override for performance.
     */
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        float sum = 0f;
        for (int i = 0; i < length; i++) {
            sum += getFloat(thisOffset + i) * other[otherOffset + i];
        }
        return sum;
    }

    /**
     * Dot product with another tensor (fallback path).
     */
    public float dot(long thisOffset, FloatTensor other, long otherOffset, int length) {
        float sum = 0f;
        for (int i = 0; i < length; i++) {
            sum += getFloat(thisOffset + i) * other.getFloat(otherOffset + i);
        }
        return sum;
    }

    /**
     * Matrix-vector multiply: out[row] += dot(weights[row], input, cols)
     * 'this' is the weight matrix with shape [rows, cols] stored row-major.
     */
    public void matmul(float[] input, float[] out, int rows, int cols) {
        matmulRows(input, out, 0, rows, cols);
    }

    /**
     * Row-range matrix-vector multiply: out[row] += dot(weights[row], input, cols) for every row in
     * [rowFrom, rowTo). This is the unit of work the parallel matmul hands to each thread. SIMD
     * kernels override it to amortise per-input work (e.g. Q4_K's per-sub-block input sums) across
     * all rows of the range instead of recomputing it inside every {@link #dot} call.
     */
    public void matmulRows(float[] input, float[] out, int rowFrom, int rowTo, int cols) {
        for (int row = rowFrom; row < rowTo; row++) {
            out[row] += dot((long) row * cols, input, 0, cols);
        }
    }

    /**
     * Multi-token row-range matmul: {@code out[t][row] += dot(weights[row], in[t])} for every token
     * {@code t < n} and row in [rowFrom, rowTo). Used by batched prefill. The default processes one
     * token at a time over a row range small enough to stay in cache, so each weight row comes from
     * memory once per range rather than once per token; SIMD kernels override it to also dequantise
     * each weight block once for several tokens.
     */
    public void matmulRowsBatch(float[][] in, float[][] out, int n, int rowFrom, int rowTo, int cols) {
        for (int t = 0; t < n; t++) {
            matmulRows(in[t], out[t], rowFrom, rowTo, cols);
        }
    }

    private static final java.util.Set<Class<?>> WARMED =
        java.util.Collections.newSetFromMap(new java.util.concurrent.ConcurrentHashMap<Class<?>, Boolean>());
    /** Upper bound on the time spent warming one kernel class, in ms (0 disables the warm-up). */
    private static final long WARM_MAX_MS = Long.getLong("matmul.warmup.ms", 1500);

    /**
     * Runs the single-token kernel ({@link #matmulRows}) of this tensor's class on a few rows until
     * it has been JIT-compiled — detected as an iteration costing less than half of the first —
     * or {@code -Dmatmul.warmup.ms} (default 1500) has passed. Once per tensor class.
     *
     * <p>Why: batched prefill never runs the single-token kernels, so their C2 compilation only
     * started when decode began, and on a machine whose cores were busy with the matmul itself it
     * took long enough that decode ran on C1 code (no Vector API intrinsics) for most of a short
     * generation: Llama-1B decode measured 3.8–7.4 tok/s after a batched prefill against 8–11.8
     * with the kernels compiled beforehand. Warming on one thread, while the pool is idle, leaves
     * the compiler threads free CPU.
     */
    public static void warmUpRows(FloatTensor t, int rows, int cols) {
        if (t == null || rows <= 0 || cols <= 0 || WARM_MAX_MS <= 0 || !WARMED.add(t.getClass())) return;
        int r = Math.min(rows, 64);
        float[] in = new float[cols];
        for (int i = 0; i < cols; i++) in[i] = ((i * 7919) % 17 - 8) * 0.01f;
        float[] out = new float[r];
        long start = System.nanoTime();
        long deadline = start + WARM_MAX_MS * 1_000_000L;
        long first = -1;
        for (int i = 0; ; i++) {
            long t0 = System.nanoTime();
            t.matmulRows(in, out, 0, r, cols);
            long dt = System.nanoTime() - t0;
            if (first < 0) first = dt;
            if ((i >= 50 && dt * 2 < first) || t0 + dt > deadline) break;
        }
    }

    private static final java.util.Set<Class<?>> WARMED_DOT =
        java.util.Collections.newSetFromMap(new java.util.concurrent.ConcurrentHashMap<Class<?>, Boolean>());

    /**
     * {@link #warmUpRows} for the per-row {@link #dot} kernel: runs {@code dot} over the first few
     * rows of this tensor until it has been JIT-compiled, once per tensor class. The MoE engines
     * compute routed experts in decode with one {@code dot} per row, which their batched prefill
     * (a row-range batch kernel per expert) never runs.
     */
    public static void warmUpDot(FloatTensor t, int rows, int cols) {
        if (t == null || rows <= 0 || cols <= 0 || WARM_MAX_MS <= 0 || !WARMED_DOT.add(t.getClass())) return;
        int r = Math.min(rows, 64);
        float[] in = new float[cols];
        for (int i = 0; i < cols; i++) in[i] = ((i * 7919) % 17 - 8) * 0.01f;
        float[] out = new float[r];
        long start = System.nanoTime();
        long deadline = start + WARM_MAX_MS * 1_000_000L;
        long first = -1;
        for (int i = 0; ; i++) {
            long t0 = System.nanoTime();
            for (int row = 0; row < r; row++) out[row] += t.dot((long) row * cols, in, 0, cols);
            long dt = System.nanoTime() - t0;
            if (first < 0) first = dt;
            if ((i >= 50 && dt * 2 < first) || t0 + dt > deadline) break;
        }
    }

    /** Smallest row chunk of a multi-token matmul handed to one pool thread (each row is n dots). */
    private static final int MIN_BATCH_CHUNK_ROWS = 16;

    /** Parallel {@link #matmulRowsBatch} over all rows. Callers must only use it when {@link MatmulPool#enabled()}. */
    public static void matmulBatchParallel(FloatTensor w, float[][] in, float[][] out, int n, int rows, int cols) {
        MatmulPool pool = MatmulPool.get();
        if (pool == null) {
            w.matmulRowsBatch(in, out, n, 0, rows, cols);
            return;
        }
        pool.parallelFor(rows, MIN_BATCH_CHUNK_ROWS, (from, to) -> w.matmulRowsBatch(in, out, n, from, to, cols));
    }

    /** Multi-token Q+K+V projection: the three matrices' rows form one unit space, as in {@link #fusedQKVMatmulParallel}. */
    public static void fusedQKVBatchParallel(FloatTensor wq, FloatTensor wk, FloatTensor wv,
            float[][] in, float[][] q, float[][] k, float[][] v, int n, int qRows, int kvRows, int cols) {
        int kEnd = qRows + kvRows;
        int total = kEnd + kvRows;
        MatmulPool.RangeTask task = (from, to) -> {
            segmentRowsBatch(wq, in, q, n, from, to, 0, qRows, cols);
            segmentRowsBatch(wk, in, k, n, from, to, qRows, kEnd, cols);
            segmentRowsBatch(wv, in, v, n, from, to, kEnd, total, cols);
        };
        MatmulPool pool = MatmulPool.get();
        if (pool == null) task.run(0, total);
        else pool.parallelFor(total, MIN_BATCH_CHUNK_ROWS, task);
    }

    /** Multi-token gate+up projection as one unit space of {@code 2 * rows}. */
    public static void fusedGateUpBatchParallel(FloatTensor wGate, FloatTensor wUp,
            float[][] in, float[][] outGate, float[][] outUp, int n, int rows, int cols) {
        MatmulPool.RangeTask task = (from, to) -> {
            segmentRowsBatch(wGate, in, outGate, n, from, to, 0, rows, cols);
            segmentRowsBatch(wUp, in, outUp, n, from, to, rows, 2 * rows, cols);
        };
        MatmulPool pool = MatmulPool.get();
        if (pool == null) task.run(0, 2 * rows);
        else pool.parallelFor(2 * rows, MIN_BATCH_CHUNK_ROWS, task);
    }

    private static void segmentRowsBatch(FloatTensor w, float[][] in, float[][] out, int n, int from, int to,
                                         int segStart, int segEnd, int cols) {
        int a = Math.max(from, segStart);
        int b = Math.min(to, segEnd);
        if (a < b) w.matmulRowsBatch(in, out, n, a - segStart, b - segStart, cols);
    }

    /**
     * Parallel matrix-vector multiply using ForkJoinPool,
     * or virtual threads on Java 25+ (avoids ForkJoinPool contention).
     * GpuFloatTensor overrides this to dispatch to GPU kernels directly.
     * With -Dmatmul.tiled=true, uses cache-friendly tiled matmul for supported types.
     */
    public void matmulParallel(float[] input, float[] out, int rows, int cols) {
        if (tryTiledMatmul(input, out, rows, cols)) return;
        MatmulPool pool = MatmulPool.get();
        if (pool != null && MatmulPool.enabled()) {
            pool.parallelFor(rows, MIN_CHUNK_ROWS, (from, to) -> matmulRows(input, out, from, to, cols));
            return;
        }
        if (tryVirtualThreadMatmul(input, out, rows, cols)) return;
        IntStream.range(0, rows).parallel().forEach(row ->
            out[row] += dot((long) row * cols, input, 0, cols)
        );
    }

    /**
     * Smallest row chunk handed to one pool thread. Kernels do some per-chunk input preparation
     * (Q4_K permutes the input and computes its sub-block sums), so tiny chunks would repeat that
     * work too often; 32 rows of a 2048-wide Q4_K matrix is ~10 µs of work per claim.
     */
    private static final int MIN_CHUNK_ROWS = Integer.getInteger("matmul.min.chunk", 32);

    /** Runs the part of {@code [from, to)} that falls inside segment {@code [segStart, segEnd)} as rows of {@code w}. */
    private static void segmentRows(FloatTensor w, float[] input, float[] out, int from, int to,
                                    int segStart, int segEnd, int cols) {
        int a = Math.max(from, segStart);
        int b = Math.min(to, segEnd);
        if (a < b) w.matmulRows(input, out, a - segStart, b - segStart, cols);
    }

    // Tiled matmul: cache-friendly multi-row processing. Enabled via -Dmatmul.tiled=true
    private static volatile Boolean tiledAvailable;
    private static volatile java.lang.reflect.Method cachedTiledMatmulMethod;

    private boolean tryTiledMatmul(float[] input, float[] out, int rows, int cols) {
        Boolean avail = tiledAvailable;
        if (avail != null && !avail) return false;

        java.lang.reflect.Method m = cachedTiledMatmulMethod;
        if (m == null) {
            if (!"true".equals(System.getProperty("matmul.tiled"))) {
                tiledAvailable = Boolean.FALSE;
                return false;
            }
            try {
                Class<?> cls = Class.forName("it.denzosoft.llmplayer.inference.TiledMatmul");
                m = cls.getMethod("matmul", FloatTensor.class, float[].class, float[].class, int.class, int.class);
                cachedTiledMatmulMethod = m;
                tiledAvailable = Boolean.TRUE;
                System.out.println("  Tiled matmul: enabled");
            } catch (ClassNotFoundException e) {
                tiledAvailable = Boolean.FALSE;
                return false;
            } catch (Exception e) {
                tiledAvailable = Boolean.FALSE;
                return false;
            }
        }

        try {
            return (boolean) m.invoke(null, this, input, out, rows, cols);
        } catch (Exception e) {
            tiledAvailable = Boolean.FALSE;
            cachedTiledMatmulMethod = null;
            return false;
        }
    }

    private static volatile Boolean virtualThreadAvailable;
    private static volatile java.lang.reflect.Method cachedMatmulMethod;

    /**
     * Force-disable virtual thread matmul (and the {@link MatmulPool}).
     * Called when GPU (OpenCL) is active because PoCL's native threads
     * conflict with the JVM's virtual thread carrier threads, causing segfaults.
     */
    public static void disableVirtualThreadMatmul() {
        MatmulPool.disable();
        virtualThreadAvailable = Boolean.FALSE;
        cachedMatmulMethod = null;
        fusedAvailable = Boolean.FALSE;
        cachedFusedMatmul = null;
        cachedFusedQKV = null;
    }

    private boolean tryVirtualThreadMatmul(float[] input, float[] out, int rows, int cols) {
        Boolean avail = virtualThreadAvailable;
        if (avail != null && !avail) return false;

        java.lang.reflect.Method m = cachedMatmulMethod;
        if (m == null) {
            try {
                Class<?> cls = Class.forName("it.denzosoft.llmplayer.inference.VirtualThreadMatmul");
                m = cls.getMethod("matmul", FloatTensor.class, float[].class, float[].class, int.class, int.class);
                cachedMatmulMethod = m;
                virtualThreadAvailable = Boolean.TRUE;
            } catch (ClassNotFoundException e) {
                virtualThreadAvailable = Boolean.FALSE;
                return false;
            } catch (Exception e) {
                virtualThreadAvailable = Boolean.FALSE;
                return false;
            }
        }

        try {
            m.invoke(null, this, input, out, rows, cols);
            return true;
        } catch (Exception e) {
            virtualThreadAvailable = Boolean.FALSE;
            cachedMatmulMethod = null;
            return false;
        }
    }

    // --- Fused parallel matmul (gate+up, Q+K+V) ---

    private static volatile Boolean fusedAvailable;
    private static volatile java.lang.reflect.Method cachedFusedMatmul;
    private static volatile java.lang.reflect.Method cachedFusedQKV;

    /**
     * Fused gate+up matmul: processes both projections in a single parallel dispatch.
     * Keeps input in L1 cache and eliminates one sync barrier vs two separate matmulParallel.
     */
    public static void fusedGateUpMatmulParallel(FloatTensor wGate, FloatTensor wUp,
            float[] input, float[] outGate, float[] outUp, int rows, int cols) {
        MatmulPool pool = MatmulPool.get();
        if (pool != null && MatmulPool.enabled()) {
            // Gate rows then up rows as one unit space: one dispatch, balanced chunks.
            pool.parallelFor(2 * rows, MIN_CHUNK_ROWS, (from, to) -> {
                segmentRows(wGate, input, outGate, from, to, 0, rows, cols);
                segmentRows(wUp, input, outUp, from, to, rows, 2 * rows, cols);
            });
            return;
        }
        if (tryFusedMatmul(wGate, wUp, input, outGate, outUp, rows, cols)) return;
        // Fallback: single ForkJoinPool dispatch over both tensors
        IntStream.range(0, rows).parallel().forEach(row -> {
            outGate[row] += wGate.dot((long) row * cols, input, 0, cols);
            outUp[row] += wUp.dot((long) row * cols, input, 0, cols);
        });
    }

    /**
     * Fused Q+K+V matmul: processes all three projections in a single parallel dispatch.
     * Handles GQA (qRows != kvRows).
     */
    public static void fusedQKVMatmulParallel(FloatTensor wq, FloatTensor wk, FloatTensor wv,
            float[] input, float[] q, float[] k, float[] v,
            int qRows, int kvRows, int cols) {
        MatmulPool pool = MatmulPool.get();
        if (pool != null && MatmulPool.enabled()) {
            // Q, K and V rows as one unit space. Chunks cost the same wherever they fall, which
            // removes the GQA imbalance of splitting max(qRows, kvRows) and doing 3 dots below kvRows.
            int kEnd = qRows + kvRows;
            pool.parallelFor(kEnd + kvRows, MIN_CHUNK_ROWS, (from, to) -> {
                segmentRows(wq, input, q, from, to, 0, qRows, cols);
                segmentRows(wk, input, k, from, to, qRows, kEnd, cols);
                segmentRows(wv, input, v, from, to, kEnd, kEnd + kvRows, cols);
            });
            return;
        }
        if (tryFusedQKVMatmul(wq, wk, wv, input, q, k, v, qRows, kvRows, cols)) return;
        // Fallback: single ForkJoinPool dispatch
        int maxRows = Math.max(qRows, kvRows);
        IntStream.range(0, maxRows).parallel().forEach(row -> {
            if (row < qRows) {
                q[row] += wq.dot((long) row * cols, input, 0, cols);
            }
            if (row < kvRows) {
                k[row] += wk.dot((long) row * cols, input, 0, cols);
                v[row] += wv.dot((long) row * cols, input, 0, cols);
            }
        });
    }

    private static boolean tryFusedMatmul(FloatTensor w1, FloatTensor w2,
            float[] input, float[] out1, float[] out2, int rows, int cols) {
        Boolean avail = fusedAvailable;
        if (avail != null && !avail) return false;

        java.lang.reflect.Method m = cachedFusedMatmul;
        if (m == null) {
            try {
                Class<?> cls = Class.forName("it.denzosoft.llmplayer.inference.VirtualThreadMatmul");
                m = cls.getMethod("fusedMatmul", FloatTensor.class, FloatTensor.class,
                    float[].class, float[].class, float[].class, int.class, int.class);
                cachedFusedMatmul = m;
                cachedFusedQKV = cls.getMethod("fusedMatmulQKV",
                    FloatTensor.class, FloatTensor.class, FloatTensor.class,
                    float[].class, float[].class, float[].class, float[].class,
                    int.class, int.class, int.class);
                fusedAvailable = Boolean.TRUE;
            } catch (Exception e) {
                fusedAvailable = Boolean.FALSE;
                return false;
            }
        }

        try {
            m.invoke(null, w1, w2, input, out1, out2, rows, cols);
            return true;
        } catch (Exception e) {
            fusedAvailable = Boolean.FALSE;
            cachedFusedMatmul = null;
            return false;
        }
    }

    private static boolean tryFusedQKVMatmul(FloatTensor wq, FloatTensor wk, FloatTensor wv,
            float[] input, float[] q, float[] k, float[] v,
            int qRows, int kvRows, int cols) {
        Boolean avail = fusedAvailable;
        if (avail != null && !avail) return false;

        java.lang.reflect.Method m = cachedFusedQKV;
        if (m == null) {
            // Force probe via tryFusedMatmul
            tryFusedMatmul(null, null, null, null, null, 0, 0);
            m = cachedFusedQKV;
            if (m == null) return false;
        }

        try {
            m.invoke(null, wq, wk, wv, input, q, k, v, qRows, kvRows, cols);
            return true;
        } catch (Exception e) {
            fusedAvailable = Boolean.FALSE;
            cachedFusedQKV = null;
            return false;
        }
    }

    /**
     * Dequantize a range of this tensor into a float array.
     */
    public void dequantize(float[] out, int outOffset, long srcOffset, int length) {
        for (int i = 0; i < length; i++) {
            out[outOffset + i] = getFloat(srcOffset + i);
        }
    }

    @Override
    public String toString() {
        return type() + "[" + size + "]";
    }
}
