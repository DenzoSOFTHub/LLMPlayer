package it.denzosoft.llmplayer.tensor;

import it.denzosoft.llmplayer.gpu.CudaBufferManager;

/**
 * CUDA GPU-accelerated Q5_1 tensor.
 * Delegates matmulParallel to CUDA GPU kernel, falls back to CPU on error.
 *
 * Q5_1: 32 weights per block, 24 bytes per block.
 * Layout: [fp16 scale (2B)] [fp16 min (2B)] [uint32 qh (4B)] [16 bytes nibbles]
 * Split element mapping (like Q5_0):
 *   Elements  0..15 = LOW nibbles of bytes 0..15, high bits from qh bits 0..15
 *   Elements 16..31 = HIGH nibbles of bytes 0..15, high bits from qh bits 16..31
 * value = (nibble | (high_bit << 4)) * scale + min
 */
public class Q5_1CudaTensor extends CudaFloatTensor {

    private static final int BLOCK_SIZE = 32;
    private static final int BLOCK_BYTES = 24;

    public Q5_1CudaTensor(TensorData data, long size, CudaBufferManager bufferManager) {
        super(data, size, bufferManager);
    }

    @Override
    public GGMLType type() { return GGMLType.Q5_1; }

    @Override
    protected String kernelResourcePath() { return "kernels/cuda/matmul_q5_1.cu"; }

    @Override
    protected String kernelName() { return "matmul_q5_1"; }

    @Override
    protected int blockBytes() { return BLOCK_BYTES; }

    @Override
    protected int blockSize() { return BLOCK_SIZE; }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int inBlockIndex = (int) (index % BLOCK_SIZE);
        long blockOffset = blockIndex * BLOCK_BYTES;

        float scale = Float16.toFloat(data.getShortLE(blockOffset));
        float min = Float16.toFloat(data.getShortLE(blockOffset + 2));

        int qh = data.getIntLE(blockOffset + 4);

        int bytePos;
        int low4;
        if (inBlockIndex < 16) {
            bytePos = inBlockIndex;
            low4 = data.getByte(blockOffset + 8 + bytePos) & 0x0F;
        } else {
            bytePos = inBlockIndex - 16;
            low4 = (data.getByte(blockOffset + 8 + bytePos) >> 4) & 0x0F;
        }
        int highBit = (qh >> inBlockIndex) & 1;

        int quant = low4 | (highBit << 4);
        return quant * scale + min;
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        float result = 0f;
        int blocks = length / BLOCK_SIZE;
        long blockStart = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        int otherIdx = otherOffset;

        for (int b = 0; b < blocks; b++) {
            float scale = Float16.toFloat(data.getShortLE(blockStart));
            float min = Float16.toFloat(data.getShortLE(blockStart + 2));
            int qh = data.getIntLE(blockStart + 4);

            float blockSum = 0f;
            float otherSum = 0f;
            for (int j = 0; j < 16; j++) {
                byte packed = data.getByte(blockStart + 8 + j);
                int lo4 = packed & 0x0F;
                int hi4 = (packed >> 4) & 0x0F;

                int q0 = lo4 | (((qh >> j) & 1) << 4);
                int q1 = hi4 | (((qh >> (j + 16)) & 1) << 4);

                blockSum += q0 * other[otherIdx + j];
                blockSum += q1 * other[otherIdx + j + 16];
                otherSum += other[otherIdx + j] + other[otherIdx + j + 16];
            }
            result += scale * blockSum + min * otherSum;
            blockStart += BLOCK_BYTES;
            otherIdx += BLOCK_SIZE;
        }
        return result;
    }
}
