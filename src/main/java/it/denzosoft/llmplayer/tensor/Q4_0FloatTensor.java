package it.denzosoft.llmplayer.tensor;

/**
 * Q4_0 quantization: 32 weights per block.
 * Block layout: [float16 scale (2 bytes)] [16 bytes of nibbles (32 x 4-bit)] = 18 bytes/block
 * value = (nibble - 8) * scale
 */
public class Q4_0FloatTensor extends FloatTensor {

    private static final int BLOCK_SIZE = 32;
    private static final int BLOCK_BYTES = 18;
    private static final ThreadLocal<float[]> DOT_BUFFER = ThreadLocal.withInitial(() -> new float[BLOCK_SIZE]);

    public Q4_0FloatTensor(TensorData data, long size) {
        super(data, size);
    }

    @Override
    public GGMLType type() { return GGMLType.Q4_0; }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int inBlockIndex = (int) (index % BLOCK_SIZE);
        long blockOffset = blockIndex * BLOCK_BYTES;

        short scaleBits = data.getShortLE(blockOffset);
        float scale = Float16.toFloat(scaleBits);

        int bytePos = inBlockIndex / 2;
        byte packed = data.getByte(blockOffset + 2 + bytePos);
        int nibble;
        if (inBlockIndex % 2 == 0) {
            nibble = packed & 0x0F;
        } else {
            nibble = (packed >> 4) & 0x0F;
        }
        return (nibble - 8) * scale;
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

            for (int i = 0; i < 16; i++) {
                byte packed = data.getByte(bo + 2 + i);
                int lo = packed & 0x0F;
                int hi = (packed >> 4) & 0x0F;
                tmp[i * 2]     = (lo - 8) * scale;
                tmp[i * 2 + 1] = (hi - 8) * scale;
            }

            result += VectorOpsFactory.get().dot(tmp, 0, other, otherBase, BLOCK_SIZE);
            otherBase += BLOCK_SIZE;
        }
        return result;
    }

    @Override
    public float dot(long thisOffset, FloatTensor other, long otherOffset, int length) {
        if (other instanceof Q8_0FloatTensor) {
            Q8_0FloatTensor q8 = (Q8_0FloatTensor) other;
            return dotQ4Q8(thisOffset, q8, otherOffset, length);
        }
        return super.dot(thisOffset, other, otherOffset, length);
    }

    private float dotQ4Q8(long thisOffset, Q8_0FloatTensor q8, long otherOffset, int length) {
        float result = 0f;
        int blocks = length / BLOCK_SIZE;
        long thisBlock = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        long otherBlock = (otherOffset / BLOCK_SIZE) * 34;

        for (int b = 0; b < blocks; b++) {
            float d4 = Float16.toFloat(data.getShortLE(thisBlock));
            float d8 = Float16.toFloat(q8.data().getShortLE(otherBlock));

            int isum = 0;
            for (int i = 0; i < 16; i++) {
                byte packed = data.getByte(thisBlock + 2 + i);
                int lo = (packed & 0x0F) - 8;
                int hi = ((packed >> 4) & 0x0F) - 8;
                isum += lo * q8.data().getByte(otherBlock + 2 + i * 2);
                isum += hi * q8.data().getByte(otherBlock + 2 + i * 2 + 1);
            }
            result += d4 * d8 * isum;
            thisBlock += BLOCK_BYTES;
            otherBlock += 34;
        }
        return result;
    }
}
