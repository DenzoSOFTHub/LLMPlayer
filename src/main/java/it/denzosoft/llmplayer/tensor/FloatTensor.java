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
        for (int row = 0; row < rows; row++) {
            out[row] += dot((long) row * cols, input, 0, cols);
        }
    }

    /**
     * Parallel matrix-vector multiply using ForkJoinPool,
     * or virtual threads on Java 25+ (avoids ForkJoinPool contention).
     * GpuFloatTensor overrides this to dispatch to GPU kernels directly.
     * With -Dmatmul.tiled=true, uses cache-friendly tiled matmul for supported types.
     */
    public void matmulParallel(float[] input, float[] out, int rows, int cols) {
        if (tryTiledMatmul(input, out, rows, cols)) return;
        if (tryVirtualThreadMatmul(input, out, rows, cols)) return;
        IntStream.range(0, rows).parallel().forEach(row ->
            out[row] += dot((long) row * cols, input, 0, cols)
        );
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
     * Force-disable virtual thread matmul.
     * Called when GPU (OpenCL) is active because PoCL's native threads
     * conflict with the JVM's virtual thread carrier threads, causing segfaults.
     */
    public static void disableVirtualThreadMatmul() {
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
