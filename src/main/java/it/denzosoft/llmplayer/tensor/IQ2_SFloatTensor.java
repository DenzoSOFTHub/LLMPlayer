package it.denzosoft.llmplayer.tensor;

/**
 * IQ2_S quantization: 256 weights per super-block (2.5625 bpw).
 * Layout (82 bytes):
 *   - d (fp16, 2 bytes): super-block scale
 *   - qs (32 bytes): grid index low 8 bits
 *   - signs (32 bytes): sign bits, 1 per weight
 *   - qh (8 bytes): grid index high 2 bits (per group of 32)
 *   - scales (8 bytes): 4-bit sub-block scales, 2 per byte
 *
 * Uses iq2s_grid (1024 entries, 10-bit index). Explicit sign bits.
 * Scale formula: d * (0.5 + scale_nibble) * 0.25
 */
public class IQ2_SFloatTensor extends FloatTensor {

    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 82;
    private static final ThreadLocal<float[]> DOT_BUFFER = ThreadLocal.withInitial(() -> new float[BLOCK_SIZE]);

    // Offsets within block
    private static final int OFF_D = 0;        // 2 bytes
    private static final int OFF_QS = 2;        // 32 bytes
    private static final int OFF_SIGNS = 34;    // 32 bytes
    private static final int OFF_QH = 66;       // 8 bytes
    private static final int OFF_SCALES = 74;   // 8 bytes

    public IQ2_SFloatTensor(TensorData data, long size) {
        super(data, size);
    }

    @Override
    public GGMLType type() { return GGMLType.IQ2_S; }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int j = (int) (index % BLOCK_SIZE);
        long bo = blockIndex * BLOCK_BYTES;

        float d = Float16.toFloat(data.getShortLE(bo + OFF_D));

        int ib32 = j / 32; // which group of 32 (0..7)
        int jIn32 = j % 32;
        int l = jIn32 / 8;   // sub-group of 8 within the 32 (0..3)
        int jIn8 = jIn32 % 8;

        // Scale: low nibble for first 16 weights (l=0,1), high nibble for last 16 (l=2,3)
        int scaleByte = Byte.toUnsignedInt(data.getByte(bo + OFF_SCALES + ib32));
        float dl;
        if (l < 2) {
            dl = d * (0.5f + (scaleByte & 0xF)) * 0.25f;
        } else {
            dl = d * (0.5f + ((scaleByte >> 4) & 0xF)) * 0.25f;
        }

        // 10-bit grid index: 8 low bits from qs, 2 high bits from qh
        int qsVal = Byte.toUnsignedInt(data.getByte(bo + OFF_QS + ib32 * 4 + l));
        int qhVal = Byte.toUnsignedInt(data.getByte(bo + OFF_QH + ib32));
        int gridIdx = qsVal | ((qhVal << (8 - 2 * l)) & 0x300);

        // Look up grid (uint64 → 8 bytes)
        long grid = IQGridTables.IQ2S_GRID[gridIdx];
        int gridByte = (int) ((grid >>> (8 * jIn8)) & 0xFF);

        // Sign bit
        int signByte = Byte.toUnsignedInt(data.getByte(bo + OFF_SIGNS + ib32 * 4 + l));
        float sign = ((signByte & IQGridTables.KMASK_IQ2XS[jIn8]) != 0) ? -1.0f : 1.0f;

        return dl * gridByte * sign;
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

            for (int ib32 = 0; ib32 < 8; ib32++) {
                int scaleByte = Byte.toUnsignedInt(data.getByte(bo + OFF_SCALES + ib32));
                float db0 = d * (0.5f + (scaleByte & 0xF)) * 0.25f;
                float db1 = d * (0.5f + ((scaleByte >> 4) & 0xF)) * 0.25f;

                int qhVal = Byte.toUnsignedInt(data.getByte(bo + OFF_QH + ib32));

                for (int l = 0; l < 4; l++) {
                    float dl = (l < 2) ? db0 : db1;

                    int qsVal = Byte.toUnsignedInt(data.getByte(bo + OFF_QS + ib32 * 4 + l));
                    int gridIdx = qsVal | ((qhVal << (8 - 2 * l)) & 0x300);
                    long grid = IQGridTables.IQ2S_GRID[gridIdx];

                    int signByte = Byte.toUnsignedInt(data.getByte(bo + OFF_SIGNS + ib32 * 4 + l));

                    for (int j = 0; j < 8; j++) {
                        int gv = (int) ((grid >>> (8 * j)) & 0xFF);
                        float sign = ((signByte & IQGridTables.KMASK_IQ2XS[j]) != 0) ? -1.0f : 1.0f;
                        tmp[outIdx++] = dl * gv * sign;
                    }
                }
            }

            result += VectorOpsFactory.get().dot(tmp, 0, other, otherBase, BLOCK_SIZE);
            otherBase += BLOCK_SIZE;
        }
        return result;
    }
}
