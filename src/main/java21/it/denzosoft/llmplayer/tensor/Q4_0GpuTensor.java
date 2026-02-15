package it.denzosoft.llmplayer.tensor;

import it.denzosoft.llmplayer.gpu.GpuBufferManager;

/**
 * GPU-accelerated Q4_0 tensor.
 * Delegates matmulParallel to GPU kernel, falls back to CPU on error.
 */
public class Q4_0GpuTensor extends GpuFloatTensor {

    private static final int BLOCK_SIZE = 32;
    private static final int BLOCK_BYTES = 18;
    private static final ThreadLocal<float[]> DOT_BUFFER = ThreadLocal.withInitial(() -> new float[BLOCK_SIZE]);

    public Q4_0GpuTensor(TensorData data, long size, GpuBufferManager bufferManager) {
        super(data, size, bufferManager);
    }

    @Override
    public GGMLType type() { return GGMLType.Q4_0; }

    @Override
    protected String kernelResourcePath() { return "kernels/matmul_q4_0.cl"; }

    @Override
    protected String kernelName() { return "matmul_q4_0"; }

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

        int bytePos = inBlockIndex / 2;
        byte packed = data.getByte(blockOffset + 2 + bytePos);
        int nibble;
        if (inBlockIndex % 2 == 0) {
            nibble = packed & 0x0F;
        } else {
            nibble = (packed >> 4) & 0x0F;
        }
        return (nibble - 8) * scale;
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
                tmp[i * 2]     = (lo - 8) * scale;
                tmp[i * 2 + 1] = (hi - 8) * scale;
            }

            result += VectorOpsFactory.get().dot(tmp, 0, other, otherBase, BLOCK_SIZE);
            otherBase += BLOCK_SIZE;
        }
        return result;
    }

    @Override
    public float dot(long thisOffset, FloatTensor other, long otherOffset, int length) {
        if (other instanceof Q8_0FloatTensor || other instanceof Q8_0GpuTensor) {
            return dotQ4Q8(thisOffset, other, otherOffset, length);
        }
        return super.dot(thisOffset, other, otherOffset, length);
    }

    private float dotQ4Q8(long thisOffset, FloatTensor other, long otherOffset, int length) {
        float result = 0f;
        int blocks = length / BLOCK_SIZE;
        long thisBlock = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        long otherBlock = (otherOffset / BLOCK_SIZE) * 34;

        for (int b = 0; b < blocks; b++) {
            float d4 = Float16.toFloat(data.getShortLE(thisBlock));
            float d8 = Float16.toFloat(other.data().getShortLE(otherBlock));

            int isum = 0;
            for (int i = 0; i < 16; i++) {
                byte packed = data.getByte(thisBlock + 2 + i);
                int lo = (packed & 0x0F) - 8;
                int hi = ((packed >> 4) & 0x0F) - 8;
                isum += lo * other.data().getByte(otherBlock + 2 + i * 2);
                isum += hi * other.data().getByte(otherBlock + 2 + i * 2 + 1);
            }
            result += d4 * d8 * isum;
            thisBlock += BLOCK_BYTES;
            otherBlock += 34;
        }
        return result;
    }
}
