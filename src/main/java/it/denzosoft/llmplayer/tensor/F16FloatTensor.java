package it.denzosoft.llmplayer.tensor;

public class F16FloatTensor extends FloatTensor {

    public F16FloatTensor(TensorData data, long size) {
        super(data, size);
    }

    @Override
    public GGMLType type() { return GGMLType.F16; }

    @Override
    public float getFloat(long index) {
        short bits = data.getShortLE(index * 2);
        return Float16.toFloat(bits);
    }
}
