package it.denzosoft.llmplayer.tensor;

import it.denzosoft.llmplayer.gpu.CudaBufferManager;

/**
 * CUDA GPU-accelerated IQ4_NL tensor.
 * Non-linear 4-bit quantization with K-means-derived lookup table.
 * Split nibble layout: low nibbles → first 16 weights, high → second 16.
 */
public class IQ4_NLCudaTensor extends CudaFloatTensor {

    private static final int BLOCK_SIZE = 32;
    private static final int BLOCK_BYTES = 18;
    private static final ThreadLocal<float[]> DOT_BUFFER = ThreadLocal.withInitial(() -> new float[BLOCK_SIZE]);

    static final float[] KVALUES_IQ4NL = {
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113
    };

    public IQ4_NLCudaTensor(TensorData data, long size, CudaBufferManager bufferManager) {
        super(data, size, bufferManager);
    }

    @Override
    public GGMLType type() { return GGMLType.IQ4_NL; }

    @Override
    protected String kernelResourcePath() { return "kernels/cuda/matmul_iq4_nl.cu"; }

    @Override
    protected String kernelName() { return "matmul_iq4_nl"; }

    @Override
    protected int blockBytes() { return BLOCK_BYTES; }

    @Override
    protected int blockSize() { return BLOCK_SIZE; }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int inBlockIndex = (int) (index % BLOCK_SIZE);
        long blockOffset = blockIndex * BLOCK_BYTES;

        short scaleBits = data.getShortLE(blockOffset);
        float scale = Float16.toFloat(scaleBits);

        int nibble;
        if (inBlockIndex < 16) {
            byte packed = data.getByte(blockOffset + 2 + inBlockIndex);
            nibble = packed & 0x0F;
        } else {
            byte packed = data.getByte(blockOffset + 2 + inBlockIndex - 16);
            nibble = (packed >> 4) & 0x0F;
        }
        return scale * KVALUES_IQ4NL[nibble];
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
            float scale = Float16.toFloat(data.getShortLE(bo));

            for (int i = 0; i < 16; i++) {
                byte packed = data.getByte(bo + 2 + i);
                int lo = packed & 0x0F;
                int hi = (packed >> 4) & 0x0F;
                tmp[i]      = scale * KVALUES_IQ4NL[lo];
                tmp[i + 16] = scale * KVALUES_IQ4NL[hi];
            }

            result += VectorOpsFactory.get().dot(tmp, 0, other, otherBase, BLOCK_SIZE);
            otherBase += BLOCK_SIZE;
        }
        return result;
    }
}
