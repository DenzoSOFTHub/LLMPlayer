package it.denzosoft.llmplayer.tensor;

/**
 * IQ3_S quantization: 256 weights per super-block (3.4375 bpw).
 * Layout (110 bytes):
 *   - d (fp16, 2 bytes): super-block scale
 *   - qs (64 bytes): grid indices, low 8 bits
 *   - qh (8 bytes): grid indices, high 1 bit (9th bit)
 *   - signs (32 bytes): sign bits, 1 per weight
 *   - scales (4 bytes): 4-bit sub-block scales, 2 per byte
 *
 * Uses iq3s_grid (512 entries, 9-bit index). Explicit sign bits.
 * Scale formula: d * (1 + 2 * scale_nibble)
 */
public class IQ3_SFloatTensor extends FloatTensor {

    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 110;
    private static final ThreadLocal<float[]> DOT_BUFFER = ThreadLocal.withInitial(() -> new float[BLOCK_SIZE]);

    // Offsets within block
    private static final int OFF_D = 0;        // 2 bytes
    private static final int OFF_QS = 2;        // 64 bytes (256/4)
    private static final int OFF_QH = 66;       // 8 bytes (256/32)
    private static final int OFF_SIGNS = 74;    // 32 bytes (256/8)
    private static final int OFF_SCALES = 106;  // 4 bytes

    public IQ3_SFloatTensor(TensorData data, long size) {
        super(data, size);
    }

    @Override
    public GGMLType type() { return GGMLType.IQ3_S; }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int j = (int) (index % BLOCK_SIZE);
        long bo = blockIndex * BLOCK_BYTES;

        float d = Float16.toFloat(data.getShortLE(bo + OFF_D));

        // Which pair of 32-weight groups (ib32 = 0,2,4,6)
        int ib32 = (j / 32) & ~1; // round down to even
        boolean secondHalf = (j / 32) % 2 == 1;

        // Scale: each byte in scales[] holds two 4-bit scales
        int scaleByte = Byte.toUnsignedInt(data.getByte(bo + OFF_SCALES + ib32 / 2));
        float db;
        if (!secondHalf) {
            db = d * (1 + 2 * (scaleByte & 0x0F));
        } else {
            db = d * (1 + 2 * ((scaleByte >> 4) & 0x0F));
        }

        // Position within the 32-weight group
        int groupBase = (j / 32) * 32;
        int jInGroup = j % 32;
        int l = jInGroup / 8;
        int jIn8 = jInGroup % 8;
        int gridQuad = jIn8 / 4;

        // qh index: 2 bytes per pair of 32-weight groups
        int qhByteIdx = (j / 32) / 2; // which qh byte (0..3)... actually per 64 weights = one pair
        int qhOffset;
        if (!secondHalf) {
            qhOffset = 0;
        } else {
            qhOffset = 1;
        }

        // Compute grid index (9 bits)
        int qsIdx = (j / 32) * 8 + 2 * l + gridQuad;
        int qsLow = Byte.toUnsignedInt(data.getByte(bo + OFF_QS + qsIdx));

        int qhByte = Byte.toUnsignedInt(data.getByte(bo + OFF_QH + ib32 / 4 + qhOffset));
        // High bit extraction: same as llama.cpp (qh[x] << (8-2*l)) & 256 or (qh[x] << (7-2*l)) & 256
        int highBit;
        if (gridQuad == 0) {
            highBit = (qhByte << (8 - 2 * l)) & 256;
        } else {
            highBit = (qhByte << (7 - 2 * l)) & 256;
        }
        int gridIdx = qsLow | highBit;
        int gridVal = IQGridTables.IQ3S_GRID[gridIdx];

        // Extract grid byte
        int byteInGrid = jIn8 % 4;
        int gv = (gridVal >>> (8 * byteInGrid)) & 0xFF;

        // Sign bit
        int signByteIdx = (j / 32) * 4 + l;
        int signByte = Byte.toUnsignedInt(data.getByte(bo + OFF_SIGNS + signByteIdx));
        float sign = ((signByte & IQGridTables.KMASK_IQ2XS[jIn8]) != 0) ? -1.0f : 1.0f;

        return db * gv * sign;
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
            float d = Float16.toFloat(data.getShortLE(bo + OFF_D));

            int outIdx = 0;

            // Process in pairs of 32-weight groups (64 weights at a time)
            for (int ib32 = 0; ib32 < 8; ib32 += 2) {
                int scaleByte = Byte.toUnsignedInt(data.getByte(bo + OFF_SCALES + ib32 / 2));
                float db1 = d * (1 + 2 * (scaleByte & 0x0F));
                float db2 = d * (1 + 2 * ((scaleByte >> 4) & 0x0F));

                // First 32 weights of pair
                int qhByte0 = Byte.toUnsignedInt(data.getByte(bo + OFF_QH + ib32 / 4));
                int qsBase1 = ib32 * 8;
                int signBase1 = ib32 * 4;

                for (int l = 0; l < 4; l++) {
                    int qs0 = Byte.toUnsignedInt(data.getByte(bo + OFF_QS + qsBase1 + 2 * l));
                    int qs1 = Byte.toUnsignedInt(data.getByte(bo + OFF_QS + qsBase1 + 2 * l + 1));
                    int grid1 = IQGridTables.IQ3S_GRID[qs0 | ((qhByte0 << (8 - 2 * l)) & 256)];
                    int grid2 = IQGridTables.IQ3S_GRID[qs1 | ((qhByte0 << (7 - 2 * l)) & 256)];
                    int signs = Byte.toUnsignedInt(data.getByte(bo + OFF_SIGNS + signBase1 + l));

                    for (int j = 0; j < 4; j++) {
                        int gv = (grid1 >>> (8 * j)) & 0xFF;
                        float sign = ((signs & IQGridTables.KMASK_IQ2XS[j]) != 0) ? -1.0f : 1.0f;
                        tmp[outIdx++] = db1 * gv * sign;
                    }
                    for (int j = 0; j < 4; j++) {
                        int gv = (grid2 >>> (8 * j)) & 0xFF;
                        float sign = ((signs & IQGridTables.KMASK_IQ2XS[j + 4]) != 0) ? -1.0f : 1.0f;
                        tmp[outIdx++] = db1 * gv * sign;
                    }
                }

                // Second 32 weights of pair
                int qhByte1 = Byte.toUnsignedInt(data.getByte(bo + OFF_QH + ib32 / 4 + 1));
                int qsBase2 = (ib32 + 1) * 8;
                int signBase2 = (ib32 + 1) * 4;

                for (int l = 0; l < 4; l++) {
                    int qs0 = Byte.toUnsignedInt(data.getByte(bo + OFF_QS + qsBase2 + 2 * l));
                    int qs1 = Byte.toUnsignedInt(data.getByte(bo + OFF_QS + qsBase2 + 2 * l + 1));
                    int grid1 = IQGridTables.IQ3S_GRID[qs0 | ((qhByte1 << (8 - 2 * l)) & 256)];
                    int grid2 = IQGridTables.IQ3S_GRID[qs1 | ((qhByte1 << (7 - 2 * l)) & 256)];
                    int signs = Byte.toUnsignedInt(data.getByte(bo + OFF_SIGNS + signBase2 + l));

                    for (int j = 0; j < 4; j++) {
                        int gv = (grid1 >>> (8 * j)) & 0xFF;
                        float sign = ((signs & IQGridTables.KMASK_IQ2XS[j]) != 0) ? -1.0f : 1.0f;
                        tmp[outIdx++] = db2 * gv * sign;
                    }
                    for (int j = 0; j < 4; j++) {
                        int gv = (grid2 >>> (8 * j)) & 0xFF;
                        float sign = ((signs & IQGridTables.KMASK_IQ2XS[j + 4]) != 0) ? -1.0f : 1.0f;
                        tmp[outIdx++] = db2 * gv * sign;
                    }
                }
            }

            result += VectorOpsFactory.get().dot(tmp, 0, other, otherBase, BLOCK_SIZE);
            otherBase += BLOCK_SIZE;
        }
        return result;
    }
}
