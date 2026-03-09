package it.denzosoft.llmplayer.tensor;

import it.denzosoft.llmplayer.gpu.CudaBufferManager;

/**
 * CUDA GPU-accelerated BF16 tensor.
 * Each element is 2 bytes (bfloat16). blockSize=1, blockBytes=2.
 */
public class BF16CudaTensor extends CudaFloatTensor {

    public BF16CudaTensor(TensorData data, long size, CudaBufferManager bufferManager) {
        super(data, size, bufferManager);
    }

    @Override
    public GGMLType type() { return GGMLType.BF16; }

    @Override
    protected String kernelResourcePath() { return "kernels/cuda/matmul_bf16.cu"; }

    @Override
    protected String kernelName() { return "matmul_bf16"; }

    @Override
    protected int blockBytes() { return 2; }

    @Override
    protected int blockSize() { return 1; }

    @Override
    public float getFloat(long index) {
        short bits = data.getShortLE(index * 2);
        return Float.intBitsToFloat(((int) bits) << 16);
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        float result = 0f;
        for (int i = 0; i < length; i++) {
            result += getFloat(thisOffset + i) * other[otherOffset + i];
        }
        return result;
    }
}
