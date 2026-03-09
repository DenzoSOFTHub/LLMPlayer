package it.denzosoft.llmplayer.tensor;

import it.denzosoft.llmplayer.gpu.CudaBufferManager;

/**
 * CUDA GPU-accelerated IQ3_XXS tensor.
 * 256 weights per super-block, 98 bytes per block.
 * Layout: [d:fp16 (2B)][64B grid indices][32B scales_and_signs]
 */
public class IQ3_XXSCudaTensor extends CudaFloatTensor {

    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 98;
    private static final ThreadLocal<float[]> DOT_BUFFER = ThreadLocal.withInitial(() -> new float[BLOCK_SIZE]);

    public IQ3_XXSCudaTensor(TensorData data, long size, CudaBufferManager bufferManager) {
        super(data, size, bufferManager);
    }

    @Override
    public GGMLType type() { return GGMLType.IQ3_XXS; }

    @Override
    protected String kernelResourcePath() { return "kernels/cuda/matmul_iq3_xxs.cu"; }

    @Override
    protected String kernelName() { return "matmul_iq3_xxs"; }

    @Override
    protected int blockBytes() { return BLOCK_BYTES; }

    @Override
    protected int blockSize() { return BLOCK_SIZE; }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int j = (int) (index % BLOCK_SIZE);
        long bo = blockIndex * BLOCK_BYTES;

        float d = Float16.toFloat(data.getShortLE(bo));

        long qsBase = bo + 2;
        long sasBase = bo + 2 + 64;

        int ib32 = j / 32;
        int jIn32 = j % 32;
        int l = jIn32 / 8;
        int jIn8 = jIn32 % 8;

        int aux32 = readIntLE(sasBase + 4 * ib32);
        float db = d * (0.5f + (aux32 >>> 28)) * 0.5f;

        int signIdx = (aux32 >>> (7 * l)) & 0x7F;
        int signs = IQGridTables.KSIGNS_IQ2XS[signIdx] & 0xFF;

        int gridQuad = jIn8 / 4;
        int gridIdx = Byte.toUnsignedInt(data.getByte(qsBase + ib32 * 8 + 2 * l + gridQuad));
        int gridVal = IQGridTables.IQ3XXS_GRID[gridIdx];

        int byteInGrid = jIn8 % 4;
        int gridByte = (gridVal >>> (8 * byteInGrid)) & 0xFF;

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
