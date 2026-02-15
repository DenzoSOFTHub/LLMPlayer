package it.denzosoft.llmplayer.tensor;

/**
 * Q8_0 quantization: 32 weights per block.
 * Block layout: [float16 scale (2 bytes)] [32 x int8 quants (32 bytes)] = 34 bytes/block
 * value = scale * quant[i]
 */
public class Q8_0FloatTensor extends FloatTensor {

    private static final int BLOCK_SIZE = 32;
    private static final int BLOCK_BYTES = 34; // 2 (f16 scale) + 32 (int8 quants)
    private static final ThreadLocal<float[]> DOT_BUFFER = ThreadLocal.withInitial(() -> new float[BLOCK_SIZE]);

    public Q8_0FloatTensor(TensorData data, long size) {
        super(data, size);
    }

    @Override
    public GGMLType type() { return GGMLType.Q8_0; }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int inBlockIndex = (int) (index % BLOCK_SIZE);
        long blockOffset = blockIndex * BLOCK_BYTES;

        short scaleBits = data.getShortLE(blockOffset);
        float scale = Float16.toFloat(scaleBits);
        byte quant = data.getByte(blockOffset + 2 + inBlockIndex);
        return scale * quant;
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        float result = 0f;
        int numBlocks = length / BLOCK_SIZE;
        long blockStart = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        int otherBase = otherOffset;
        float[] tmp = DOT_BUFFER.get();

        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float scale = Float16.toFloat(data.getShortLE(bo));

            for (int i = 0; i < BLOCK_SIZE; i++) {
                tmp[i] = scale * data.getByte(bo + 2 + i);
            }

            result += VectorOpsFactory.get().dot(tmp, 0, other, otherBase, BLOCK_SIZE);
            otherBase += BLOCK_SIZE;
        }
        return result;
    }

    @Override
    public float dot(long thisOffset, FloatTensor other, long otherOffset, int length) {
        if (other instanceof Q8_0FloatTensor) {
            Q8_0FloatTensor q8other = (Q8_0FloatTensor) other;
            return dotQ8Q8(thisOffset, q8other, otherOffset, length);
        }
        return super.dot(thisOffset, other, otherOffset, length);
    }

    private float dotQ8Q8(long thisOffset, Q8_0FloatTensor other, long otherOffset, int length) {
        float result = 0f;
        int blocks = length / BLOCK_SIZE;
        long thisBlock = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        long otherBlock = (otherOffset / BLOCK_SIZE) * BLOCK_BYTES;

        for (int b = 0; b < blocks; b++) {
            float d0 = Float16.toFloat(data.getShortLE(thisBlock));
            float d1 = Float16.toFloat(other.data.getShortLE(otherBlock));

            int isum = 0;
            for (int i = 0; i < BLOCK_SIZE; i++) {
                isum += data.getByte(thisBlock + 2 + i) * other.data.getByte(otherBlock + 2 + i);
            }
            result += d0 * d1 * isum;
            thisBlock += BLOCK_BYTES;
            otherBlock += BLOCK_BYTES;
        }
        return result;
    }
}
