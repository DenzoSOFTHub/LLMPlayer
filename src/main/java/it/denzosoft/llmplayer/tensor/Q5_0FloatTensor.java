package it.denzosoft.llmplayer.tensor;

/**
 * Q5_0 quantization: 32 weights per block.
 * Block layout: [float16 scale (2 bytes)] [uint32 qh (4 bytes)] [16 bytes nibbles (32 x 4-bit)] = 22 bytes/block
 * value = ((low4 | (high_bit << 4)) - 16) * scale
 */
public class Q5_0FloatTensor extends FloatTensor {

    private static final int BLOCK_SIZE = 32;
    private static final int BLOCK_BYTES = 22;

    public Q5_0FloatTensor(TensorData data, long size) {
        super(data, size);
    }

    @Override
    public GGMLType type() { return GGMLType.Q5_0; }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int inBlockIndex = (int) (index % BLOCK_SIZE);
        long blockOffset = blockIndex * BLOCK_BYTES;

        short scaleBits = data.getShortLE(blockOffset);
        float scale = Float16.toFloat(scaleBits);

        int qh = data.getIntLE(blockOffset + 2);
        int highBit = (qh >> inBlockIndex) & 1;

        int bytePos = inBlockIndex / 2;
        byte packed = data.getByte(blockOffset + 6 + bytePos);
        int low4;
        if (inBlockIndex % 2 == 0) {
            low4 = packed & 0x0F;
        } else {
            low4 = (packed >> 4) & 0x0F;
        }

        int quant = low4 | (highBit << 4);
        return (quant - 16) * scale;
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        float result = 0f;
        int blocks = length / BLOCK_SIZE;
        long blockStart = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        int otherIdx = otherOffset;

        for (int b = 0; b < blocks; b++) {
            float scale = Float16.toFloat(data.getShortLE(blockStart));
            int qh = data.getIntLE(blockStart + 2);

            float blockSum = 0f;
            for (int i = 0; i < 16; i++) {
                byte packed = data.getByte(blockStart + 6 + i);
                int lo4 = packed & 0x0F;
                int hi4 = (packed >> 4) & 0x0F;

                int idx0 = i * 2;
                int idx1 = i * 2 + 1;
                int q0 = lo4 | (((qh >> idx0) & 1) << 4);
                int q1 = hi4 | (((qh >> idx1) & 1) << 4);

                blockSum += (q0 - 16) * other[otherIdx + idx0];
                blockSum += (q1 - 16) * other[otherIdx + idx1];
            }
            result += scale * blockSum;
            blockStart += BLOCK_BYTES;
            otherIdx += BLOCK_SIZE;
        }
        return result;
    }
}
