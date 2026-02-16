package it.denzosoft.llmplayer.tensor;

/**
 * Q6_K quantization: 256 weights per super-block (210 bytes).
 * Layout:
 *   - ql (128 bytes): lower 4 bits of 6-bit quants (complex packing)
 *   - qh (64 bytes): upper 2 bits of 6-bit quants (complex packing)
 *   - scales (16 bytes): 16 x int8 sub-block scales
 *   - d (fp16, 2 bytes): super-block scale
 * Total: 128 + 64 + 16 + 2 = 210 bytes
 */
public class Q6_KFloatTensor extends FloatTensor {

    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 210;
    private static final ThreadLocal<float[]> DOT_BUFFER = ThreadLocal.withInitial(() -> new float[BLOCK_SIZE]);
    private static final ThreadLocal<byte[]> TL_QL = ThreadLocal.withInitial(() -> new byte[128]);
    private static final ThreadLocal<byte[]> TL_QH = ThreadLocal.withInitial(() -> new byte[64]);
    private static final ThreadLocal<byte[]> TL_SC = ThreadLocal.withInitial(() -> new byte[16]);

    public Q6_KFloatTensor(TensorData data, long size) {
        super(data, size);
    }

    @Override
    public GGMLType type() { return GGMLType.Q6_K; }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int j = (int) (index % BLOCK_SIZE);
        long bo = blockIndex * BLOCK_BYTES;

        float d = Float16.toFloat(data.getShortLE(bo + 208));

        int half = j / 128;
        int jLocal = j % 128;
        int quadrant = jLocal / 32;
        int l = jLocal % 32;

        long qlBase = bo + (long) half * 64;
        long qhBase = bo + 128 + (long) half * 32;

        int ql4;
        int qhBits;

        if (quadrant == 0) {
            ql4 = Byte.toUnsignedInt(data.getByte(qlBase + l)) & 0x0F;
            qhBits = (Byte.toUnsignedInt(data.getByte(qhBase + l)) >> 0) & 0x03;
        } else if (quadrant == 1) {
            ql4 = Byte.toUnsignedInt(data.getByte(qlBase + 32 + l)) & 0x0F;
            qhBits = (Byte.toUnsignedInt(data.getByte(qhBase + l)) >> 2) & 0x03;
        } else if (quadrant == 2) {
            ql4 = (Byte.toUnsignedInt(data.getByte(qlBase + l)) >> 4) & 0x0F;
            qhBits = (Byte.toUnsignedInt(data.getByte(qhBase + l)) >> 4) & 0x03;
        } else {
            ql4 = (Byte.toUnsignedInt(data.getByte(qlBase + 32 + l)) >> 4) & 0x0F;
            qhBits = (Byte.toUnsignedInt(data.getByte(qhBase + l)) >> 6) & 0x03;
        }

        int q = (ql4 | (qhBits << 4)) - 32;
        int subBlock = j / 16;
        int sc = data.getByte(bo + 192 + subBlock); // int8_t scale

        return d * sc * q;
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        float result = 0f;
        int numBlocks = length / BLOCK_SIZE;
        long blockStart = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        int otherBase = otherOffset;
        float[] tmp = DOT_BUFFER.get();
        byte[] ql = TL_QL.get();
        byte[] qh = TL_QH.get();
        byte[] sc = TL_SC.get();

        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float d = Float16.toFloat(data.getShortLE(bo + 208));

            data.copyBytes(bo, ql, 0, 128);
            data.copyBytes(bo + 128, qh, 0, 64);
            data.copyBytes(bo + 192, sc, 0, 16);

            for (int half = 0; half < 2; half++) {
                int qlOff = half * 64;
                int qhOff = half * 32;
                int scBase = half * 8;
                int elemBase = half * 128;

                for (int l = 0; l < 32; l++) {
                    int qlByte0 = Byte.toUnsignedInt(ql[qlOff + l]);
                    int qlByte1 = Byte.toUnsignedInt(ql[qlOff + 32 + l]);
                    int qhByte = Byte.toUnsignedInt(qh[qhOff + l]);

                    int q0 = (qlByte0 & 0x0F)         | (((qhByte >> 0) & 3) << 4);
                    int q1 = (qlByte1 & 0x0F)         | (((qhByte >> 2) & 3) << 4);
                    int q2 = ((qlByte0 >> 4) & 0x0F)  | (((qhByte >> 4) & 3) << 4);
                    int q3 = ((qlByte1 >> 4) & 0x0F)  | (((qhByte >> 6) & 3) << 4);

                    int scIdx = scBase + (l / 16);
                    float ds0 = d * sc[scIdx];
                    float ds1 = d * sc[scIdx + 2];
                    float ds2 = d * sc[scIdx + 4];
                    float ds3 = d * sc[scIdx + 6];

                    tmp[elemBase + l]       = ds0 * (q0 - 32);
                    tmp[elemBase + 32 + l]  = ds1 * (q1 - 32);
                    tmp[elemBase + 64 + l]  = ds2 * (q2 - 32);
                    tmp[elemBase + 96 + l]  = ds3 * (q3 - 32);
                }
            }

            result += VectorOpsFactory.get().dot(tmp, 0, other, otherBase, BLOCK_SIZE);
            otherBase += BLOCK_SIZE;
        }
        return result;
    }
}
