package it.denzosoft.llmplayer.tensor;

/**
 * IQ3_XXS quantization: 256 weights per super-block (3.0625 bpw).
 * Layout (98 bytes):
 *   - d (fp16, 2 bytes): super-block scale
 *   - qs (96 bytes): 64 bytes grid indices + 32 bytes scales_and_signs
 *
 * Dequantization: For each group of 32 weights:
 *   - A uint32 from scales_and_signs provides a 4-bit scale and four 7-bit sign indices
 *   - Each 8-bit grid index looks up iq3xxs_grid (256 entries) for 4 unsigned byte values
 *   - Result: d * (0.5 + scale_nibble) * 0.5 * grid_value * sign
 */
public class IQ3_XXSFloatTensor extends FloatTensor {

    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 98;
    private static final ThreadLocal<float[]> DOT_BUFFER = ThreadLocal.withInitial(() -> new float[BLOCK_SIZE]);

    public IQ3_XXSFloatTensor(TensorData data, long size) {
        super(data, size);
    }

    @Override
    public GGMLType type() { return GGMLType.IQ3_XXS; }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int j = (int) (index % BLOCK_SIZE);
        long bo = blockIndex * BLOCK_BYTES;

        float d = Float16.toFloat(data.getShortLE(bo));

        // Grid indices start at bo+2, scales_and_signs at bo+2+64
        long qsBase = bo + 2;
        long sasBase = bo + 2 + 64; // scales_and_signs

        int ib32 = j / 32; // which group of 32
        int jIn32 = j % 32;
        int l = jIn32 / 8;  // which sub-group of 8 within the 32
        int jIn8 = jIn32 % 8;

        // Read uint32 scale_and_signs for this group of 32
        int aux32 = readIntLE(sasBase + 4 * ib32);
        float db = d * (0.5f + (aux32 >>> 28)) * 0.5f;

        // Get 7-bit sign index for sub-group l
        int signIdx = (aux32 >>> (7 * l)) & 0x7F;
        int signs = IQGridTables.KSIGNS_IQ2XS[signIdx] & 0xFF;

        // Get grid index (8-bit) and look up 4 values
        int gridQuad = jIn8 / 4; // 0 or 1 within the sub-group pair
        int gridIdx = Byte.toUnsignedInt(data.getByte(qsBase + ib32 * 8 + 2 * l + gridQuad));
        int gridVal = IQGridTables.IQ3XXS_GRID[gridIdx];

        // Extract the specific byte from the grid uint32
        int byteInGrid = jIn8 % 4;
        int gridByte = (gridVal >>> (8 * byteInGrid)) & 0xFF;

        // Apply sign
        float sign = ((signs & IQGridTables.KMASK_IQ2XS[jIn8]) != 0) ? -1.0f : 1.0f;

        return db * gridByte * sign;
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

            long qsBase = bo + 2;
            long sasBase = bo + 2 + 64;

            int outIdx = 0;
            for (int ib32 = 0; ib32 < 8; ib32++) {
                int aux32 = readIntLE(sasBase + 4 * ib32);
                float db = d * (0.5f + (aux32 >>> 28)) * 0.5f;

                for (int l = 0; l < 4; l++) {
                    int signIdx = (aux32 >>> (7 * l)) & 0x7F;
                    int signs = IQGridTables.KSIGNS_IQ2XS[signIdx] & 0xFF;

                    int gridIdx1 = Byte.toUnsignedInt(data.getByte(qsBase + ib32 * 8 + 2 * l));
                    int gridIdx2 = Byte.toUnsignedInt(data.getByte(qsBase + ib32 * 8 + 2 * l + 1));
                    int grid1 = IQGridTables.IQ3XXS_GRID[gridIdx1];
                    int grid2 = IQGridTables.IQ3XXS_GRID[gridIdx2];

                    for (int j = 0; j < 4; j++) {
                        int gv = (grid1 >>> (8 * j)) & 0xFF;
                        float sign = ((signs & IQGridTables.KMASK_IQ2XS[j]) != 0) ? -1.0f : 1.0f;
                        tmp[outIdx++] = db * gv * sign;
                    }
                    for (int j = 0; j < 4; j++) {
                        int gv = (grid2 >>> (8 * j)) & 0xFF;
                        float sign = ((signs & IQGridTables.KMASK_IQ2XS[j + 4]) != 0) ? -1.0f : 1.0f;
                        tmp[outIdx++] = db * gv * sign;
                    }
                }
            }

            result += VectorOpsFactory.get().dot(tmp, 0, other, otherBase, BLOCK_SIZE);
            otherBase += BLOCK_SIZE;
        }
        return result;
    }

    private int readIntLE(long offset) {
        int b0 = Byte.toUnsignedInt(data.getByte(offset));
        int b1 = Byte.toUnsignedInt(data.getByte(offset + 1));
        int b2 = Byte.toUnsignedInt(data.getByte(offset + 2));
        int b3 = Byte.toUnsignedInt(data.getByte(offset + 3));
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }
}
