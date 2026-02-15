package it.denzosoft.llmplayer.tensor;

import it.denzosoft.llmplayer.gpu.GpuBufferManager;

/**
 * GPU-accelerated F32 tensor.
 * Delegates matmulParallel to GPU, falls back to CPU SIMD on error.
 * All other operations (getFloat, dot, dequantize) use CPU path from F32FloatTensor.
 */
public class F32GpuTensor extends GpuFloatTensor {

    private static final ThreadLocal<float[]> DOT_BUFFER = ThreadLocal.withInitial(() -> new float[0]);

    public F32GpuTensor(TensorData data, long size, GpuBufferManager bufferManager) {
        super(data, size, bufferManager);
    }

    @Override
    public GGMLType type() { return GGMLType.F32; }

    @Override
    protected String kernelResourcePath() { return "kernels/matmul_f32.cl"; }

    @Override
    protected String kernelName() { return "matmul_f32"; }

    @Override
    protected int blockBytes() { return 4; } // 4 bytes per float

    @Override
    protected int blockSize() { return 1; } // 1 element per "block"

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
