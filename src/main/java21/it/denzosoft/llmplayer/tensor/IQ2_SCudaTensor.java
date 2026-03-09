package it.denzosoft.llmplayer.tensor;

import it.denzosoft.llmplayer.gpu.CudaBufferManager;

/**
 * CUDA GPU-accelerated IQ2_S tensor.
 * 256 weights per super-block, 82 bytes per block.
 * Layout: [d:fp16 (2B)][qs:32B][signs:32B][qh:8B][scales:8B]
 */
public class IQ2_SCudaTensor extends CudaFloatTensor {

    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 82;
    private static final ThreadLocal<float[]> DOT_BUFFER = ThreadLocal.withInitial(() -> new float[BLOCK_SIZE]);

    // Offsets within block
    private static final int OFF_D = 0;
    private static final int OFF_QS = 2;
    private static final int OFF_SIGNS = 34;
    private static final int OFF_QH = 66;
    private static final int OFF_SCALES = 74;

    public IQ2_SCudaTensor(TensorData data, long size, CudaBufferManager bufferManager) {
        super(data, size, bufferManager);
    }

    @Override
    public GGMLType type() { return GGMLType.IQ2_S; }

    @Override
    protected String kernelResourcePath() { return "kernels/cuda/matmul_iq2_s.cu"; }

    @Override
    protected String kernelName() { return "matmul_iq2_s"; }

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

        int ib32 = j / 32;
        int jIn32 = j % 32;
        int l = jIn32 / 8;
        int jIn8 = jIn32 % 8;

        int scaleByte = Byte.toUnsignedInt(data.getByte(bo + OFF_SCALES + ib32));
        float dl;
        if (l < 2) {
            dl = d * (0.5f + (scaleByte & 0xF)) * 0.25f;
        } else {
            dl = d * (0.5f + ((scaleByte >> 4) & 0xF)) * 0.25f;
        }

        int qsVal = Byte.toUnsignedInt(data.getByte(bo + OFF_QS + ib32 * 4 + l));
        int qhVal = Byte.toUnsignedInt(data.getByte(bo + OFF_QH + ib32));
        int gridIdx = qsVal | ((qhVal << (8 - 2 * l)) & 0x300);

        long grid = IQGridTables.IQ2S_GRID[gridIdx];
        int gridByte = (int) ((grid >>> (8 * jIn8)) & 0xFF);

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
