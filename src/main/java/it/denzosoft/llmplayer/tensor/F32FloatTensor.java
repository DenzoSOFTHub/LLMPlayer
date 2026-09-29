package it.denzosoft.llmplayer.tensor;

public class F32FloatTensor extends FloatTensor {

    private static final ThreadLocal<float[]> DOT_BUFFER = ThreadLocal.withInitial(() -> new float[0]);

    public F32FloatTensor(TensorData data, long size) {
        super(data, size);
    }

    @Override
    public GGMLType type() { return GGMLType.F32; }

    @Override
    public float getFloat(long index) {
        return data.getFloatLE(index * 4);
    }

    /**
     * Small F32 tensors (MoE routers, biases, norms: at most {@link #HEAP_COPY_MAX} elements) are
     * copied to the heap on the first dot, so the dot is one SIMD pass over a float[] instead of
     * one {@code getFloatLE} per element. The router is dotted every token in every MoE layer.
     */
    private static final long HEAP_COPY_MAX = 1L << 22;
    private volatile float[] heap;

    private float[] heapCopy() {
        float[] h = heap;
        if (h == null && size <= HEAP_COPY_MAX) {
            h = new float[(int) size];
            for (int i = 0; i < h.length; i++) h[i] = data.getFloatLE((long) i * 4);
            heap = h;
        }
        return h;
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        float[] h = heapCopy();
        if (h != null) return VectorOpsFactory.get().dot(h, (int) thisOffset, other, otherOffset, length);
        float[] buf = DOT_BUFFER.get();
        if (buf.length < length) {
            buf = new float[length];
            DOT_BUFFER.set(buf);
        }
        dequantize(buf, 0, thisOffset, length);
        return VectorOpsFactory.get().dot(buf, 0, other, otherOffset, length);
    }

    @Override
    public float dot(long thisOffset, FloatTensor other, long otherOffset, int length) {
        return super.dot(thisOffset, other, otherOffset, length);
    }
}
