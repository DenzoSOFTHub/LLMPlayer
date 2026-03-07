package it.denzosoft.llmplayer.tensor;

import it.denzosoft.llmplayer.gpu.CudaBufferManager;

/**
 * CUDA GPU-accelerated F32 tensor.
 * Delegates matmulParallel to CUDA GPU, falls back to CPU on error.
 */
public class F32CudaTensor extends CudaFloatTensor {

    private static final ThreadLocal<float[]> DOT_BUFFER = ThreadLocal.withInitial(() -> new float[0]);

    public F32CudaTensor(TensorData data, long size, CudaBufferManager bufferManager) {
        super(data, size, bufferManager);
    }

    @Override
    public GGMLType type() { return GGMLType.F32; }

    @Override
    protected String kernelResourcePath() { return "kernels/cuda/matmul_f32.cu"; }

    @Override
    protected String kernelName() { return "matmul_f32"; }

    @Override
    protected int blockBytes() { return 4; }

    @Override
    protected int blockSize() { return 1; }

    @Override
    public float getFloat(long index) {
        return data.getFloatLE(index * 4);
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        float[] buf = DOT_BUFFER.get();
        if (buf.length < length) {
            buf = new float[length];
            DOT_BUFFER.set(buf);
        }
        dequantize(buf, 0, thisOffset, length);
        return VectorOpsFactory.get().dot(buf, 0, other, otherOffset, length);
    }
}
