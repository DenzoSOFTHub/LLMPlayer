package it.denzosoft.llmplayer.tensor;

public class BF16FloatTensor extends FloatTensor {

    public BF16FloatTensor(TensorData data, long size) {
        super(data, size);
    }

    @Override
    public GGMLType type() { return GGMLType.BF16; }

    @Override
    public float getFloat(long index) {
        short bits = data.getShortLE(index * 2);
        // BF16 to F32: shift left 16 bits to get upper 16 bits of float32
        return Float.intBitsToFloat(((int) bits) << 16);
    }
}
