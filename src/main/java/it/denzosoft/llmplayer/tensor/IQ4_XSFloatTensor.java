package it.denzosoft.llmplayer.tensor;

/**
 * IQ4_XS quantization: 256 weights per super-block (4.25 bpw).
 * Layout (136 bytes):
 *   - d (fp16, 2 bytes): super-block scale
 *   - scales_h (uint16, 2 bytes): high 2 bits of 8 sub-block scales
 *   - scales_l (4 bytes): low 4 bits of 8 sub-block scales, packed 2 per byte
 *   - qs (128 bytes): 256 x 4-bit nibbles (non-linear lookup)
 * Total: 2 + 2 + 4 + 128 = 136 bytes
 *
 * Uses same IQ4_NL non-linear lookup table. Each sub-block has a 6-bit scale
 * reconstructed as (ls - 32) where ls = low4 | (high2 << 4).
 */
public class IQ4_XSFloatTensor extends FloatTensor {

    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 136;
    private static final ThreadLocal<float[]> DOT_BUFFER = ThreadLocal.withInitial(() -> new float[BLOCK_SIZE]);

    public IQ4_XSFloatTensor(TensorData data, long size) {
        super(data, size);
    }

    @Override
    public GGMLType type() { return GGMLType.IQ4_XS; }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int j = (int) (index % BLOCK_SIZE);
        long bo = blockIndex * BLOCK_BYTES;

        float d = Float16.toFloat(data.getShortLE(bo));

        // Decode sub-block scale (6 bits): ib = sub-block index (0..7)
        int ib = j / 32;
        int scalesH = Short.toUnsignedInt(data.getShortLE(bo + 2));
        int scalesLByte = Byte.toUnsignedInt(data.getByte(bo + 4 + ib / 2));
        int low4 = (ib % 2 == 0) ? (scalesLByte & 0x0F) : ((scalesLByte >> 4) & 0x0F);
        int high2 = (scalesH >> (2 * ib)) & 3;
        int ls = low4 | (high2 << 4);
        float dl = d * (ls - 32);

        // Decode nibble
        int inSub = j % 32;
        int qsOffset = ib * 16 + inSub / 2;
        byte packed = data.getByte(bo + 8 + qsOffset);
        int nibble = (inSub % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);

        return dl * IQ4_NLFloatTensor.KVALUES_IQ4NL[nibble];
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
            float d = Float16.toFloat(data.getShortLE(bo));
            int scalesH = Short.toUnsignedInt(data.getShortLE(bo + 2));

            for (int ib = 0; ib < 8; ib++) {
                // Reconstruct 6-bit sub-block scale
                int scalesLByte = Byte.toUnsignedInt(data.getByte(bo + 4 + ib / 2));
                int low4 = (ib % 2 == 0) ? (scalesLByte & 0x0F) : ((scalesLByte >> 4) & 0x0F);
                int high2 = (scalesH >> (2 * ib)) & 3;
                int ls = low4 | (high2 << 4);
                float dl = d * (ls - 32);

                // Dequantize 32 weights in this sub-block
                int baseIdx = ib * 32;
                long qsBase = bo + 8 + (long) ib * 16;
                for (int i = 0; i < 16; i++) {
                    byte packed = data.getByte(qsBase + i);
                    int lo = packed & 0x0F;
                    int hi = (packed >> 4) & 0x0F;
                    tmp[baseIdx + i]      = dl * IQ4_NLFloatTensor.KVALUES_IQ4NL[lo];
                    tmp[baseIdx + 16 + i] = dl * IQ4_NLFloatTensor.KVALUES_IQ4NL[hi];
                }
            }

            result += VectorOpsFactory.get().dot(tmp, 0, other, otherBase, BLOCK_SIZE);
            otherBase += BLOCK_SIZE;
        }
        return result;
    }
}
