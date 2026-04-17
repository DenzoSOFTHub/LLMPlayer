package it.denzosoft.llmplayer.tensor;

import it.denzosoft.llmplayer.gpu.CudaBufferManager;

/**
 * Runtime-fused QKV weight tensor (GPU-only).
 *
 * Wraps a pre-allocated GPU buffer that holds Q/K/V quantized weights concatenated
 * row-wise: rows 0..qDim-1 = Q, qDim..qDim+kvDim-1 = K, qDim+kvDim..qDim+2kvDim-1 = V.
 * Same underlying quantization type (Q4_K, Q8_0, etc) for all three — byte-level
 * concat works because K-quant / scalar-quant block layouts are row-independent.
 *
 * Enables the existing merged-QKV path (single matmul + split_qkv kernel) for
 * models that ship separate Q/K/V weights (Llama, Qwen, Mistral, Gemma ...), saving
 * 2 kernel launches + 2 redundant input reads per layer.
 *
 * The class delegates type/kernel info to a template tensor (typically wq) since the
 * quantization-specific CUDA kernel is shape-agnostic — it matmuls (rows × cols)
 * regardless of how many rows the merged matrix has.
 *
 * NOT usable on the CPU path — {@link #getFloat} / {@link #matmulParallel} throw.
 * If the GPU path fails, the caller must fall back to separate wq/wk/wv matmul.
 */
public final class MergedQkvCudaTensor extends CudaFloatTensor {

    private final CudaFloatTensor template;
    private final long mergedWeightsPtr;

    /**
     * @param template  one of wq/wk/wv (used for kernel selection + type info)
     * @param mergedPtr pre-allocated GPU device pointer with merged Q|K|V bytes
     * @param totalSize total element count across merged rows × cols
     * @param bm        buffer manager owning the GPU pointer
     */
    public MergedQkvCudaTensor(CudaFloatTensor template, long mergedPtr,
                               long totalSize, CudaBufferManager bm) {
        super(template.data, totalSize, bm);
        this.template = template;
        this.mergedWeightsPtr = mergedPtr;
    }

    @Override
    public GGMLType type() { return template.type(); }

    @Override
    protected String kernelResourcePath() { return template.kernelResourcePath(); }

    @Override
    protected String kernelName() { return template.kernelName(); }

    @Override
    protected int blockBytes() { return template.blockBytes(); }

    @Override
    protected int blockSize() { return template.blockSize(); }

    /** Return the pre-allocated merged GPU buffer pointer. Bypasses CPU upload path. */
    @Override
    public long getGpuWeights() { return mergedWeightsPtr; }

    @Override
    public int getMatmulBlockDim(int cols) { return template.getMatmulBlockDim(cols); }

    @Override
    public int getMatmulSharedMem(int cols) { return template.getMatmulSharedMem(cols); }

    // ---- CPU-fallback methods deliberately disabled ----

    @Override
    public float getFloat(long index) {
        throw new UnsupportedOperationException(
            "MergedQkvCudaTensor is GPU-only — CPU fallback should use wq/wk/wv separately");
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        throw new UnsupportedOperationException(
            "MergedQkvCudaTensor is GPU-only — CPU fallback should use wq/wk/wv separately");
    }

    @Override
    public void matmulParallel(float[] input, float[] out, int rows, int cols) {
        throw new UnsupportedOperationException(
            "MergedQkvCudaTensor is GPU-only — CPU fallback should use wq/wk/wv separately");
    }
}
