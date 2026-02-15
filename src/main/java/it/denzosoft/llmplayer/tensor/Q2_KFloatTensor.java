package it.denzosoft.llmplayer.tensor;

/**
 * Q2_K quantization: 256 weights per super-block (84 bytes).
 * Layout:
 *   - scales (16 bytes): 16 x uint8 (4-bit scale + 4-bit min)
 *   - qs (64 bytes): 256 x 2-bit quants (4 per byte)
 *   - d (fp16, 2 bytes): super-block scale
 *   - dmin (fp16, 2 bytes): super-block minimum
 * Total: 16 + 64 + 2 + 2 = 84 bytes
 */
public class Q2_KFloatTensor extends FloatTensor {

    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 84;

    public Q2_KFloatTensor(TensorData data, long size) {
        super(data, size);
    }

    @Override
    public GGMLType type() { return GGMLType.Q2_K; }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int j = (int) (index % BLOCK_SIZE);
        long bo = blockIndex * BLOCK_BYTES;

        float d = Float16.toFloat(data.getShortLE(bo + 80));
        float dmin = Float16.toFloat(data.getShortLE(bo + 82));

        int subBlock = j / 16;
        int scByte = Byte.toUnsignedInt(data.getByte(bo + subBlock));
        int sc = scByte & 0x0F;
        int m = scByte >> 4;

        int qsByte = Byte.toUnsignedInt(data.getByte(bo + 16 + j / 4));
        int q = (qsByte >> (2 * (j % 4))) & 0x03;

        return d * sc * q - dmin * m;
    }

    @Override
    public float dot(long thisOffset, FloatTensor other, long otherOffset, int length) {
        if (other instanceof F32FloatTensor) {
            return dotQ2KF32(thisOffset, other, otherOffset, length);
        }
        return super.dot(thisOffset, other, otherOffset, length);
    }

    private float dotQ2KF32(long thisOffset, FloatTensor f32, long otherOffset, int length) {
        float result = 0f;
        int blocks = length / BLOCK_SIZE;
        long thisBlock = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;

        for (int b = 0; b < blocks; b++) {
            long bo = thisBlock;
            float d = Float16.toFloat(data.getShortLE(bo + 80));
            float dmin = Float16.toFloat(data.getShortLE(bo + 82));

            for (int sb = 0; sb < 16; sb++) {
                int scByte = Byte.toUnsignedInt(data.getByte(bo + sb));
                int sc = scByte & 0x0F;
                int m = scByte >> 4;

                float sumQ = 0f;
                float sumF = 0f;
                long fBase = otherOffset + (long) b * BLOCK_SIZE + sb * 16;

                for (int i = 0; i < 16; i++) {
                    int j = sb * 16 + i;
                    int qsByte = Byte.toUnsignedInt(data.getByte(bo + 16 + j / 4));
                    int q = (qsByte >> (2 * (j % 4))) & 0x03;
                    float f = f32.getFloat(fBase + i);
                    sumQ += q * f;
                    sumF += f;
                }
                result += d * sc * sumQ - dmin * m * sumF;
            }
            thisBlock += BLOCK_BYTES;
        }
        return result;
    }
}
