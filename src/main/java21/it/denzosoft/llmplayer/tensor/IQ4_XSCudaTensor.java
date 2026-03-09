package it.denzosoft.llmplayer.tensor;

import it.denzosoft.llmplayer.gpu.CudaBufferManager;

/**
 * CUDA GPU-accelerated IQ4_XS tensor.
 * Non-linear 4-bit quantization with super-blocks (256 weights, 8 sub-blocks).
 */
public class IQ4_XSCudaTensor extends CudaFloatTensor {

    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 136;
    private static final ThreadLocal<float[]> DOT_BUFFER = ThreadLocal.withInitial(() -> new float[BLOCK_SIZE]);

    static final float[] KVALUES_IQ4NL = {
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113
    };

    public IQ4_XSCudaTensor(TensorData data, long size, CudaBufferManager bufferManager) {
        super(data, size, bufferManager);
    }

    @Override
    public GGMLType type() { return GGMLType.IQ4_XS; }

    @Override
    protected String kernelResourcePath() { return "kernels/cuda/matmul_iq4_xs.cu"; }

    @Override
    protected String kernelName() { return "matmul_iq4_xs"; }

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

        int ib = j / 32;
        int scalesH = Short.toUnsignedInt(data.getShortLE(bo + 2));
        int scalesLByte = Byte.toUnsignedInt(data.getByte(bo + 4 + ib / 2));
        int low4 = (ib % 2 == 0) ? (scalesLByte & 0x0F) : ((scalesLByte >> 4) & 0x0F);
        int high2 = (scalesH >> (2 * ib)) & 3;
        int ls = low4 | (high2 << 4);
        float dl = d * (ls - 32);

        int inSub = j % 32;
        int qsOffset = ib * 16 + inSub / 2;
        byte packed = data.getByte(bo + 8 + qsOffset);
        int nibble = (inSub % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);

        return dl * KVALUES_IQ4NL[nibble];
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
                int scalesLByte = Byte.toUnsignedInt(data.getByte(bo + 4 + ib / 2));
                int low4 = (ib % 2 == 0) ? (scalesLByte & 0x0F) : ((scalesLByte >> 4) & 0x0F);
                int high2 = (scalesH >> (2 * ib)) & 3;
                int ls = low4 | (high2 << 4);
                float dl = d * (ls - 32);

                int baseIdx = ib * 32;
                long qsBase = bo + 8 + (long) ib * 16;
                for (int i = 0; i < 16; i++) {
                    byte packed = data.getByte(qsBase + i);
                    int lo = packed & 0x0F;
                    int hi = (packed >> 4) & 0x0F;
                    tmp[baseIdx + i]      = dl * KVALUES_IQ4NL[lo];
                    tmp[baseIdx + 16 + i] = dl * KVALUES_IQ4NL[hi];
                }
            }

            result += VectorOpsFactory.get().dot(tmp, 0, other, otherBase, BLOCK_SIZE);
            otherBase += BLOCK_SIZE;
        }
        return result;
    }
}
