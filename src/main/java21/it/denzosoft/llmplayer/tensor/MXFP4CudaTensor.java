package it.denzosoft.llmplayer.tensor;

import it.denzosoft.llmplayer.gpu.CudaBufferManager;

/**
 * CUDA GPU-accelerated MXFP4 tensor.
 * Delegates matmulParallel to CUDA GPU kernel, falls back to CPU on error.
 *
 * MXFP4 (Microscaling FP4 E2M1): 32 weights per block, 17 bytes per block.
 * Layout: [e8m0_scale:1B][qs:16B] — SPLIT nibble layout.
 *   - low  nibbles of qs[0..15] → positions 0..15
 *   - high nibbles of qs[0..15] → positions 16..31
 *
 * The `matmul_mxfp4` kernel takes FP32 input (no dp4a path — FP4 E2M1 values 0.5/1.5/... are
 * non-integer, so int8 dp4a isn't directly applicable without pre-scaling).
 * Block size 17 is NOT 4-byte aligned → kernel uses byte __ldg only.
 */
public class MXFP4CudaTensor extends CudaFloatTensor {

    private static final int BLOCK_SIZE = 32;
    private static final int BLOCK_BYTES = 17;

    public MXFP4CudaTensor(TensorData data, long size, CudaBufferManager bufferManager) {
        super(data, size, bufferManager);
    }

    @Override
    public GGMLType type() { return GGMLType.MXFP4; }

    @Override
    protected String kernelResourcePath() { return "kernels/cuda/matmul_mxfp4.cu"; }

    @Override
    protected String kernelName() { return "matmul_mxfp4"; }

    @Override
    protected int blockBytes() { return BLOCK_BYTES; }

    @Override
    protected int blockSize() { return BLOCK_SIZE; }

    // FP4 E2M1 lookup table (mirrors MXFP4FloatTensor / kernel FP4_TABLE).
    private static final float[] FP4_TABLE = {
         0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };

    private static float e8m0ToFloat(int exp) {
        if (exp == 0 || exp == 255) return 0f;
        return Float.intBitsToFloat(exp << 23);
    }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int inBlockIndex = (int) (index % BLOCK_SIZE);
        long blockOffset = blockIndex * BLOCK_BYTES;

        int scaleByte = Byte.toUnsignedInt(data.getByte(blockOffset));
        float scale = e8m0ToFloat(scaleByte);

        int byteIdx;
        int nibble;
        if (inBlockIndex < 16) {
            byteIdx = inBlockIndex;
            byte packed = data.getByte(blockOffset + 1 + byteIdx);
            nibble = packed & 0x0F;
        } else {
            byteIdx = inBlockIndex - 16;
            byte packed = data.getByte(blockOffset + 1 + byteIdx);
            nibble = (packed >> 4) & 0x0F;
        }

        return FP4_TABLE[nibble] * scale;
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        float result = 0f;
        int blocks = length / BLOCK_SIZE;
        long blockStart = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        int otherIdx = otherOffset;

        for (int b = 0; b < blocks; b++) {
            int scaleByte = Byte.toUnsignedInt(data.getByte(blockStart));
            float scale = e8m0ToFloat(scaleByte);

            float blockSum = 0f;
            for (int j = 0; j < 16; j++) {
                byte packed = data.getByte(blockStart + 1 + j);
                int lo = packed & 0x0F;
                int hi = (packed >> 4) & 0x0F;
                blockSum += FP4_TABLE[lo] * other[otherIdx + j];
                blockSum += FP4_TABLE[hi] * other[otherIdx + j + 16];
            }
            result += scale * blockSum;
            blockStart += BLOCK_BYTES;
            otherIdx += BLOCK_SIZE;
        }
        return result;
    }
}
