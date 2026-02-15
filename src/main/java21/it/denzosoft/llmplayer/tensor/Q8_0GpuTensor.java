package it.denzosoft.llmplayer.tensor;

import it.denzosoft.llmplayer.gpu.GpuBufferManager;

/**
 * GPU-accelerated Q8_0 tensor.
 * Delegates matmulParallel to GPU kernel, falls back to CPU on error.
 */
public class Q8_0GpuTensor extends GpuFloatTensor {

    private static final int BLOCK_SIZE = 32;
    private static final int BLOCK_BYTES = 34;
    private static final ThreadLocal<float[]> DOT_BUFFER = ThreadLocal.withInitial(() -> new float[BLOCK_SIZE]);

    public Q8_0GpuTensor(TensorData data, long size, GpuBufferManager bufferManager) {
        super(data, size, bufferManager);
    }

    @Override
    public GGMLType type() { return GGMLType.Q8_0; }

    @Override
    protected String kernelResourcePath() { return "kernels/matmul_q8_0.cl"; }

    @Override
    protected String kernelName() { return "matmul_q8_0"; }

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
        byte quant = data.getByte(blockOffset + 2 + inBlockIndex);
        return scale * quant;
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

            for (int i = 0; i < BLOCK_SIZE; i++) {
                tmp[i] = scale * data.getByte(bo + 2 + i);
            }

            result += VectorOpsFactory.get().dot(tmp, 0, other, otherBase, BLOCK_SIZE);
            otherBase += BLOCK_SIZE;
        }
        return result;
    }

    @Override
    public float dot(long thisOffset, FloatTensor other, long otherOffset, int length) {
        if (other instanceof Q8_0FloatTensor || other instanceof Q8_0GpuTensor) {
            return dotQ8Q8(thisOffset, other, otherOffset, length);
        }
        return super.dot(thisOffset, other, otherOffset, length);
    }

    private float dotQ8Q8(long thisOffset, FloatTensor other, long otherOffset, int length) {
        float result = 0f;
        int blocks = length / BLOCK_SIZE;
        long thisBlock = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        long otherBlock = (otherOffset / BLOCK_SIZE) * BLOCK_BYTES;

        for (int b = 0; b < blocks; b++) {
            float d0 = Float16.toFloat(data.getShortLE(thisBlock));
            float d1 = Float16.toFloat(other.data().getShortLE(otherBlock));

            int isum = 0;
            for (int i = 0; i < BLOCK_SIZE; i++) {
                isum += data.getByte(thisBlock + 2 + i) * other.data().getByte(otherBlock + 2 + i);
            }
            result += d0 * d1 * isum;
            thisBlock += BLOCK_BYTES;
            otherBlock += BLOCK_BYTES;
        }
        return result;
    }
}
