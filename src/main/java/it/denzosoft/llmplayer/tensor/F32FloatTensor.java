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

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
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
