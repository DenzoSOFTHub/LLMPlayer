package it.denzosoft.llmplayer.tensor;

import it.denzosoft.llmplayer.gpu.CudaBufferManager;

/**
 * CUDA GPU-accelerated IQ3_S tensor.
 * 256 weights per super-block, 110 bytes per block.
 * Layout: [d:fp16 (2B)][qs:64B][qh:8B][signs:32B][scales:4B]
 */
public class IQ3_SCudaTensor extends CudaFloatTensor {

    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 110;
    private static final ThreadLocal<float[]> DOT_BUFFER = ThreadLocal.withInitial(() -> new float[BLOCK_SIZE]);

    // Offsets within block
    private static final int OFF_D = 0;
    private static final int OFF_QS = 2;
    private static final int OFF_QH = 66;
    private static final int OFF_SIGNS = 74;
    private static final int OFF_SCALES = 106;

    public IQ3_SCudaTensor(TensorData data, long size, CudaBufferManager bufferManager) {
        super(data, size, bufferManager);
    }

    @Override
    public GGMLType type() { return GGMLType.IQ3_S; }

    @Override
    protected String kernelResourcePath() { return "kernels/cuda/matmul_iq3_s.cu"; }

    @Override
    protected String kernelName() { return "matmul_iq3_s"; }

    @Override
    protected int blockBytes() { return BLOCK_BYTES; }

    @Override
    protected int blockSize() { return BLOCK_SIZE; }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int j = (int) (index % BLOCK_SIZE);
        long bo = blockIndex * BLOCK_BYTES;

        float d = Float16.toFloat(data.getShortLE(bo + OFF_D));

        int ib32 = (j / 32) & ~1;
        boolean secondHalf = (j / 32) % 2 == 1;

        int scaleByte = Byte.toUnsignedInt(data.getByte(bo + OFF_SCALES + ib32 / 2));
        float db;
        if (!secondHalf) {
            db = d * (1 + 2 * (scaleByte & 0x0F));
        } else {
            db = d * (1 + 2 * ((scaleByte >> 4) & 0x0F));
        }

        int jInGroup = j % 32;
        int l = jInGroup / 8;
        int jIn8 = jInGroup % 8;
        int gridQuad = jIn8 / 4;

        int qsIdx = (j / 32) * 8 + 2 * l + gridQuad;
        int qsLow = Byte.toUnsignedInt(data.getByte(bo + OFF_QS + qsIdx));

        int qhByte = Byte.toUnsignedInt(data.getByte(bo + OFF_QH + ib32 / 4 + (secondHalf ? 1 : 0)));
        int highBit;
        if (gridQuad == 0) {
            highBit = (qhByte << (8 - 2 * l)) & 256;
        } else {
            highBit = (qhByte << (7 - 2 * l)) & 256;
        }
        int gridIdx = qsLow | highBit;
        int gridVal = IQGridTables.IQ3S_GRID[gridIdx];

        int byteInGrid = jIn8 % 4;
        int gv = (gridVal >>> (8 * byteInGrid)) & 0xFF;

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
            for (int ib32 = 0; ib32 < 8; ib32 += 2) {
                int scaleByte = Byte.toUnsignedInt(data.getByte(bo + OFF_SCALES + ib32 / 2));
                float db1 = d * (1 + 2 * (scaleByte & 0x0F));
                float db2 = d * (1 + 2 * ((scaleByte >> 4) & 0x0F));

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
