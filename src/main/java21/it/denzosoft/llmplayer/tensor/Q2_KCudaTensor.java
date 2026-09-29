package it.denzosoft.llmplayer.tensor;

import it.denzosoft.llmplayer.gpu.CudaBufferManager;

/**
 * CUDA Q2_K tensor ({@code kernels/cuda/matmul_q2_k.cu}, 256 weights per 84-byte block). Element
 * access on the CPU side delegates to {@link Q2_KFloatTensor}.
 */
public class Q2_KCudaTensor extends CudaFloatTensor {

    private final Q2_KFloatTensor cpu;

    public Q2_KCudaTensor(TensorData data, long size, CudaBufferManager bufferManager) {
        super(data, size, bufferManager);
        this.cpu = new Q2_KFloatTensor(data, size);
    }

    @Override
    public GGMLType type() { return GGMLType.Q2_K; }

    @Override
    protected String kernelResourcePath() { return "kernels/cuda/matmul_q2_k.cu"; }

    @Override
    protected String kernelName() { return "matmul_q2_k"; }

    @Override
    protected int blockBytes() { return 84; }

    @Override
    protected int blockSize() { return 256; }

    @Override
    public float getFloat(long index) { return cpu.getFloat(index); }

    @Override
    public void dequantize(float[] out, int outOffset, long srcOffset, int length) {
        cpu.dequantize(out, outOffset, srcOffset, length);
    }

    @Override
    protected float dotScalar(long thisOffset, float[] other, int otherOffset, int length) {
        return cpu.dot(thisOffset, other, otherOffset, length);
    }
}
