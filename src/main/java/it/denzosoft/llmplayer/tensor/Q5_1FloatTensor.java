package it.denzosoft.llmplayer.tensor;

/**
 * Q5_1 quantization: 32 weights per block.
 * Block layout: [float16 scale (2 bytes)] [float16 min (2 bytes)] [uint32 qh (4 bytes)] [16 bytes nibbles] = 24 bytes/block
 * Element mapping (SPLIT like Q5_0):
 *   Elements  0..15 = LOW nibbles of bytes 0..15, high bits from qh bits 0..15
 *   Elements 16..31 = HIGH nibbles of bytes 0..15, high bits from qh bits 16..31
 * value = (nibble | (high_bit << 4)) * scale + min
 */
public class Q5_1FloatTensor extends FloatTensor {

    private static final int BLOCK_SIZE = 32;
    private static final int BLOCK_BYTES = 24;

    public Q5_1FloatTensor(TensorData data, long size) {
        super(data, size);
    }

    @Override
    public GGMLType type() { return GGMLType.Q5_1; }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int inBlockIndex = (int) (index % BLOCK_SIZE);
        long blockOffset = blockIndex * BLOCK_BYTES;

        float scale = Float16.toFloat(data.getShortLE(blockOffset));
        float min = Float16.toFloat(data.getShortLE(blockOffset + 2));

        int qh = data.getIntLE(blockOffset + 4);

        int bytePos;
        int low4;
        if (inBlockIndex < 16) {
            bytePos = inBlockIndex;
            low4 = data.getByte(blockOffset + 8 + bytePos) & 0x0F;
        } else {
            bytePos = inBlockIndex - 16;
            low4 = (data.getByte(blockOffset + 8 + bytePos) >> 4) & 0x0F;
        }
        int highBit = (qh >> inBlockIndex) & 1;

        int quant = low4 | (highBit << 4);
        return quant * scale + min;
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        float result = 0f;
        int blocks = length / BLOCK_SIZE;
        long blockStart = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        int otherIdx = otherOffset;

        for (int b = 0; b < blocks; b++) {
            float scale = Float16.toFloat(data.getShortLE(blockStart));
            float min = Float16.toFloat(data.getShortLE(blockStart + 2));
            int qh = data.getIntLE(blockStart + 4);

            float blockSum = 0f;
            float otherSum = 0f;
            for (int j = 0; j < 16; j++) {
                byte packed = data.getByte(blockStart + 8 + j);
                int lo4 = packed & 0x0F;
                int hi4 = (packed >> 4) & 0x0F;

                int q0 = lo4 | (((qh >> j) & 1) << 4);
                int q1 = hi4 | (((qh >> (j + 16)) & 1) << 4);

                blockSum += q0 * other[otherIdx + j];
                blockSum += q1 * other[otherIdx + j + 16];
                otherSum += other[otherIdx + j] + other[otherIdx + j + 16];
            }
            result += scale * blockSum + min * otherSum;
            blockStart += BLOCK_BYTES;
            otherIdx += BLOCK_SIZE;
        }
        return result;
    }
}
