package it.denzosoft.llmplayer.tensor;

import it.denzosoft.llmplayer.gpu.CudaBufferManager;

/**
 * CUDA GPU-accelerated F16 tensor.
 * Each element is 2 bytes (IEEE 754 half-precision). blockSize=1, blockBytes=2.
 */
public class F16CudaTensor extends CudaFloatTensor {

    public F16CudaTensor(TensorData data, long size, CudaBufferManager bufferManager) {
        super(data, size, bufferManager);
    }

    @Override
    public GGMLType type() { return GGMLType.F16; }

    @Override
    protected String kernelResourcePath() { return "kernels/cuda/matmul_f16.cu"; }

    @Override
    protected String kernelName() { return "matmul_f16"; }

    @Override
    protected int blockBytes() { return 2; }

    @Override
    protected int blockSize() { return 1; }

    @Override
    public float getFloat(long index) {
        short bits = data.getShortLE(index * 2);
        return Float16.toFloat(bits);
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
