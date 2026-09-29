package it.denzosoft.llmplayer.tensor;

import it.denzosoft.llmplayer.gpu.CudaBufferManager;

/**
 * CUDA IQ2_XXS tensor ({@code kernels/cuda/matmul_iq2_xxs.cu}, 256 weights per 66-byte block). Element
 * access on the CPU side delegates to {@link IQ2_XXSFloatTensor}.
 */
public class IQ2_XXSCudaTensor extends CudaFloatTensor {

    private final IQ2_XXSFloatTensor cpu;

    public IQ2_XXSCudaTensor(TensorData data, long size, CudaBufferManager bufferManager) {
        super(data, size, bufferManager);
        this.cpu = new IQ2_XXSFloatTensor(data, size);
    }

    @Override
    public GGMLType type() { return GGMLType.IQ2_XXS; }

    @Override
    protected String kernelResourcePath() { return "kernels/cuda/matmul_iq2_xxs.cu"; }

    @Override
    protected String kernelName() { return "matmul_iq2_xxs"; }

    @Override
    protected int blockBytes() { return 66; }

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
