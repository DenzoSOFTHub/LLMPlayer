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
     */
    public void matmulParallel(float[] input, float[] out, int rows, int cols) {
        if (tryVirtualThreadMatmul(input, out, rows, cols)) return;
        IntStream.range(0, rows).parallel().forEach(row ->
            out[row] += dot((long) row * cols, input, 0, cols)
        );
    }

    private static volatile Boolean virtualThreadAvailable;

    /**
     * Force-disable virtual thread matmul.
     * Called when GPU (OpenCL) is active because PoCL's native threads
     * conflict with the JVM's virtual thread carrier threads, causing segfaults.
     */
    public static void disableVirtualThreadMatmul() {
        virtualThreadAvailable = Boolean.FALSE;
    }

    private boolean tryVirtualThreadMatmul(float[] input, float[] out, int rows, int cols) {
        Boolean avail = virtualThreadAvailable;
        if (avail != null && !avail) return false;
        try {
            Class<?> cls = Class.forName("it.denzosoft.llmplayer.inference.VirtualThreadMatmul");
            java.lang.reflect.Method m = cls.getMethod("matmul", FloatTensor.class, float[].class, float[].class, int.class, int.class);
            m.invoke(null, this, input, out, rows, cols);
            virtualThreadAvailable = Boolean.TRUE;
            return true;
        } catch (ClassNotFoundException e) {
            virtualThreadAvailable = Boolean.FALSE;
            return false;
        } catch (Exception e) {
            virtualThreadAvailable = Boolean.FALSE;
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
