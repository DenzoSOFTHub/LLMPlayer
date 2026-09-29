package it.denzosoft.llmplayer.tensor;

import it.denzosoft.llmplayer.gpu.CudaBufferManager;

/**
 * CUDA IQ1_M tensor ({@code kernels/cuda/matmul_iq1_m.cu}, 256 weights per 56-byte block). Element
 * access on the CPU side delegates to {@link IQ1_MFloatTensor}.
 */
public class IQ1_MCudaTensor extends CudaFloatTensor {

    private final IQ1_MFloatTensor cpu;

    public IQ1_MCudaTensor(TensorData data, long size, CudaBufferManager bufferManager) {
        super(data, size, bufferManager);
        this.cpu = new IQ1_MFloatTensor(data, size);
    }

    @Override
    public GGMLType type() { return GGMLType.IQ1_M; }

    @Override
    protected String kernelResourcePath() { return "kernels/cuda/matmul_iq1_m.cu"; }

    @Override
    protected String kernelName() { return "matmul_iq1_m"; }

    @Override
    protected int blockBytes() { return 56; }

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
