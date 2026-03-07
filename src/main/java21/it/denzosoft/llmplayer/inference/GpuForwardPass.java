package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.gpu.GpuBufferManager;
import it.denzosoft.llmplayer.gpu.OpenCLBindings;
import it.denzosoft.llmplayer.gpu.OpenCLContext;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.model.ModelWeights;
import it.denzosoft.llmplayer.model.TransformerLayerWeights;
import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.GpuFloatTensor;

import java.lang.foreign.*;
import java.util.Arrays;

/**
 * GPU-resident forward pass for dense pre-norm transformer architectures.
 * Keeps activation buffers on the GPU throughout each layer's computation,
 * only syncing to CPU for the attention score phase (KV cache + softmax).
 *
 * This reduces blocking sync points from ~30 per layer to 2:
 * 1. Download Q/K/V after projections
 * 2. Upload attention output before Wo projection
 *
 * Supports: Llama, Qwen2, Qwen3, Mistral3, Phi-4-mini (pre-norm, separate Q/K/V, SiLU activation).
 * Falls back to CPU path for: post-norm, parallel FFN, merged QKV, GeGLU, packed FFN, MoE.
 */
public class GpuForwardPass implements AutoCloseable {

    private final OpenCLContext clContext;
    private final GpuBufferManager bufferManager;
    private final ModelConfig config;
    private final Arena arena;

    // GPU-resident activation buffers (allocated once, reused every token)
    private final MemorySegment gpuX;      // main activation [dim]
    private final MemorySegment gpuXb;     // scratch for post-norm / FFN output [dim]
    private final MemorySegment gpuXb2;    // scratch for attention output [qDim]
    private final MemorySegment gpuHb;     // FFN hidden [ffnDim]
    private final MemorySegment gpuHb2;    // FFN gate [ffnDim]
    private final MemorySegment gpuQ;      // Q projection output [qDim]
    private final MemorySegment gpuK;      // K projection output [kvDim]
    private final MemorySegment gpuV;      // V projection output [kvDim]
    private final MemorySegment gpuPartialSums; // for rmsnorm reduction

    // Per-layer norm weight buffers on GPU (uploaded once at init)
    private final MemorySegment[] gpuAttnNormWeights;
    private final MemorySegment[] gpuFfnNormWeights;

    // Pre-compiled kernels (fetched once)
    private final MemorySegment rmsnormSumsqKernel;
    private final MemorySegment rmsnormNormalizeKernel;
    private final MemorySegment siluKernel;
    private final MemorySegment accumulateKernel;
    private final MemorySegment elementwiseMulKernel;
    private final MemorySegment fillZeroKernel;

    // Dimensions
    private final int dim;
    private final int qDim;
    private final int kvDim;
    private final int ffnDim;
    private final float normEps;
    private final long localWorkSize;
    private final int numWorkGroups; // for rmsnorm reduction

    // Host-side staging buffers (pinned in arena for async transfers)
    private final MemorySegment hostQ;
    private final MemorySegment hostK;
    private final MemorySegment hostV;
    private final MemorySegment hostXb2;
    private final MemorySegment hostX;

    /**
     * Create a GPU forward pass for the given model.
     * All layer weights must be GpuFloatTensor instances (full GPU offload).
     */
    public GpuForwardPass(ModelConfig config, ModelWeights weights, GpuBufferManager bufferManager) {
        this.config = config;
        this.bufferManager = bufferManager;
        this.clContext = bufferManager.getClContext();
        this.arena = Arena.ofShared();

        this.dim = config.embeddingLength();
        this.qDim = config.headCount() * config.headSize();
        this.kvDim = config.kvDim();
        this.ffnDim = config.intermediateSize();
        this.normEps = config.normEps();

        long maxWg = clContext.getDeviceInfo().maxWorkGroupSize();
        this.localWorkSize = Math.min(256, maxWg);
        this.numWorkGroups = (int) ((dim + localWorkSize - 1) / localWorkSize);

        // Allocate GPU activation buffers
        long floatBytes = Float.BYTES;
        gpuX = bufferManager.createBuffer(dim * floatBytes, OpenCLBindings.CL_MEM_READ_WRITE);
        gpuXb = bufferManager.createBuffer(dim * floatBytes, OpenCLBindings.CL_MEM_READ_WRITE);
        gpuXb2 = bufferManager.createBuffer(qDim * floatBytes, OpenCLBindings.CL_MEM_READ_WRITE);
        gpuHb = bufferManager.createBuffer(ffnDim * floatBytes, OpenCLBindings.CL_MEM_READ_WRITE);
        gpuHb2 = bufferManager.createBuffer(ffnDim * floatBytes, OpenCLBindings.CL_MEM_READ_WRITE);
        gpuQ = bufferManager.createBuffer(qDim * floatBytes, OpenCLBindings.CL_MEM_READ_WRITE);
        gpuK = bufferManager.createBuffer(kvDim * floatBytes, OpenCLBindings.CL_MEM_READ_WRITE);
        gpuV = bufferManager.createBuffer(kvDim * floatBytes, OpenCLBindings.CL_MEM_READ_WRITE);
        gpuPartialSums = bufferManager.createBuffer((long) numWorkGroups * floatBytes, OpenCLBindings.CL_MEM_READ_WRITE);

        // Allocate host staging buffers
        hostQ = arena.allocate(ValueLayout.JAVA_FLOAT, qDim);
        hostK = arena.allocate(ValueLayout.JAVA_FLOAT, kvDim);
        hostV = arena.allocate(ValueLayout.JAVA_FLOAT, kvDim);
        hostXb2 = arena.allocate(ValueLayout.JAVA_FLOAT, qDim);
        hostX = arena.allocate(ValueLayout.JAVA_FLOAT, dim);

        // Compile kernels
        rmsnormSumsqKernel = clContext.compileKernel("kernels/rmsnorm.cl", "rmsnorm_sumsq");
        rmsnormNormalizeKernel = clContext.compileKernel("kernels/rmsnorm.cl", "rmsnorm_normalize");
        siluKernel = clContext.compileKernel("kernels/silu.cl", "silu");
        accumulateKernel = clContext.compileKernel("kernels/accumulate.cl", "accumulate");
        elementwiseMulKernel = clContext.compileKernel("kernels/elementwise_mul.cl", "elementwise_mul");
        fillZeroKernel = clContext.compileKernel("kernels/fill_zero.cl", "fill_zero");

        // Upload per-layer norm weights to GPU
        int blockCount = config.blockCount();
        gpuAttnNormWeights = new MemorySegment[blockCount];
        gpuFfnNormWeights = new MemorySegment[blockCount];
        for (int i = 0; i < blockCount; i++) {
            TransformerLayerWeights layer = weights.layers()[i];
            if (layer.attnNorm() != null) {
                gpuAttnNormWeights[i] = uploadNormWeights(layer.attnNorm(), dim);
            }
            if (layer.ffnNorm() != null) {
                gpuFfnNormWeights[i] = uploadNormWeights(layer.ffnNorm(), dim);
            }
        }
    }

    private MemorySegment uploadNormWeights(FloatTensor normTensor, int size) {
        float[] w = new float[size];
        for (int i = 0; i < size; i++) {
            w[i] = normTensor.getFloat(i);
        }
        return bufferManager.uploadNormWeights(w);
    }

    /**
     * Check if the GPU forward pass can handle this model configuration.
     * Returns true for pre-norm dense models with separate Q/K/V and SiLU activation.
     */
    public static boolean isSupported(ModelConfig config, ModelWeights weights) {
        // Must be dense (no MoE)
        if (config.expertCount() > 0) return false;

        // Check first layer to determine configuration
        TransformerLayerWeights firstLayer = weights.layers()[0];

        // Must have pre-norm (attnNorm present)
        if (firstLayer.attnNorm() == null) return false;

        // Must have separate FFN norm (not parallel FFN like Command-R)
        if (firstLayer.ffnNorm() == null) return false;

        // Must have separate Q/K/V (not merged QKV like Phi3/Phi4)
        if (firstLayer.wqkv() != null) return false;

        // Must not have post-norms (not GLM4/Gemma2/OLMo2 style)
        if (firstLayer.postAttnNorm() != null) return false;
        if (firstLayer.postFfnNorm() != null) return false;

        // Must have separate gate/up (not packed like GLM4)
        if (firstLayer.wGate() == null) return false;

        // All weight tensors for the first layer must be GpuFloatTensor (full GPU offload)
        if (!(firstLayer.wq() instanceof GpuFloatTensor)) return false;
        if (!(firstLayer.wk() instanceof GpuFloatTensor)) return false;
        if (!(firstLayer.wv() instanceof GpuFloatTensor)) return false;
        if (!(firstLayer.wo() instanceof GpuFloatTensor)) return false;
        if (!(firstLayer.wGate() instanceof GpuFloatTensor)) return false;
        if (!(firstLayer.wUp() instanceof GpuFloatTensor)) return false;
        if (!(firstLayer.wDown() instanceof GpuFloatTensor)) return false;

        return true;
    }

    /**
     * Upload host x[] to GPU.
     */
    public void uploadX(float[] x) {
        MemorySegment.copy(x, 0, hostX, ValueLayout.JAVA_FLOAT, 0, dim);
        clContext.writeBuffer(gpuX, hostX, (long) dim * Float.BYTES);
    }

    /**
     * Download GPU x to host array.
     */
    public void downloadX(float[] x) {
        clContext.finish();
        clContext.readBuffer(gpuX, hostX, (long) dim * Float.BYTES);
        MemorySegment.copy(hostX, ValueLayout.JAVA_FLOAT, 0, x, 0, dim);
    }

    /**
     * Execute one transformer layer on GPU with minimal host synchronization.
     *
     * Flow:
     * GPU: rmsnorm(x) -> Q/K/V matmul -> download q,k,v (sync 1)
     * CPU: RoPE, KV cache, attention scores, softmax, weighted sum -> upload xb2 (sync 2)
     * GPU: Wo matmul -> accumulate -> rmsnorm -> gate/up matmul -> silu -> mul -> down matmul -> accumulate
     */
    public void forwardLayer(InferenceState state, TransformerLayerWeights layerWeights,
                              int layerIdx, int position, Attention attention) {
        try (Arena tempArena = Arena.ofConfined()) {
            // === GPU Phase 1: Attention projections ===

            // 1. RMSNorm: gpuXb = rmsnorm(gpuX, attnNormWeights)
            gpuRmsNorm(gpuXb, gpuX, gpuAttnNormWeights[layerIdx], dim, tempArena);

            // 2. Q/K/V projections (all stay on GPU, no sync)
            gpuFillZero(gpuQ, qDim, tempArena);
            gpuFillZero(gpuK, kvDim, tempArena);
            gpuFillZero(gpuV, kvDim, tempArena);
            ((GpuFloatTensor) layerWeights.wq()).gpuMatmulBuffered(gpuXb, gpuQ, qDim, dim, tempArena);
            ((GpuFloatTensor) layerWeights.wk()).gpuMatmulBuffered(gpuXb, gpuK, kvDim, dim, tempArena);
            ((GpuFloatTensor) layerWeights.wv()).gpuMatmulBuffered(gpuXb, gpuV, kvDim, dim, tempArena);

            // === Sync point 1: Download Q/K/V to CPU ===
            clContext.finish();
            clContext.readBuffer(gpuQ, hostQ, (long) qDim * Float.BYTES);
            clContext.readBuffer(gpuK, hostK, (long) kvDim * Float.BYTES);
            clContext.readBuffer(gpuV, hostV, (long) kvDim * Float.BYTES);
            MemorySegment.copy(hostQ, ValueLayout.JAVA_FLOAT, 0, state.q, 0, qDim);
            MemorySegment.copy(hostK, ValueLayout.JAVA_FLOAT, 0, state.k, 0, kvDim);
            MemorySegment.copy(hostV, ValueLayout.JAVA_FLOAT, 0, state.v, 0, kvDim);

            // === CPU Phase: Attention scores (RoPE, KV cache, softmax, weighted sum) ===
            // Reuse the Attention object's CPU logic for this part
            attention.forwardFromProjections(state, layerWeights, layerIdx, position);
            // After this, state.xb contains Wo * xb2 (output projection done on CPU by attention)
            // But we want to do Wo on GPU. So we need xb2 (the attention-weighted output).
            // Let's adjust: we upload xb2 and do Wo on GPU.

            // === Sync point 2: Upload attention output (xb2) to GPU ===
            MemorySegment.copy(state.xb2, 0, hostXb2, ValueLayout.JAVA_FLOAT, 0, qDim);
            clContext.writeBufferAsync(gpuXb2, hostXb2, (long) qDim * Float.BYTES);

            // === GPU Phase 2: Output projection + FFN (all chained, no sync) ===

            // 3. Wo matmul: gpuXb = Wo * gpuXb2
            gpuFillZero(gpuXb, dim, tempArena);
            ((GpuFloatTensor) layerWeights.wo()).gpuMatmulBuffered(gpuXb2, gpuXb, dim, qDim, tempArena);

            // 4. Residual connection: gpuX += gpuXb
            gpuAccumulate(gpuX, gpuXb, dim, tempArena);

            // 5. FFN norm: gpuXb = rmsnorm(gpuX, ffnNormWeights)
            gpuRmsNorm(gpuXb, gpuX, gpuFfnNormWeights[layerIdx], dim, tempArena);

            // 6. Gate and Up projections
            gpuFillZero(gpuHb, ffnDim, tempArena);
            gpuFillZero(gpuHb2, ffnDim, tempArena);
            ((GpuFloatTensor) layerWeights.wGate()).gpuMatmulBuffered(gpuXb, gpuHb, ffnDim, dim, tempArena);
            ((GpuFloatTensor) layerWeights.wUp()).gpuMatmulBuffered(gpuXb, gpuHb2, ffnDim, dim, tempArena);

            // 7. SiLU activation: gpuHb = silu(gpuHb)
            gpuSilu(gpuHb, ffnDim, tempArena);

            // 8. Element-wise multiply: gpuHb *= gpuHb2
            gpuElementwiseMul(gpuHb, gpuHb2, ffnDim, tempArena);

            // 9. Down projection: gpuXb = wDown * gpuHb
            gpuFillZero(gpuXb, dim, tempArena);
            ((GpuFloatTensor) layerWeights.wDown()).gpuMatmulBuffered(gpuHb, gpuXb, dim, ffnDim, tempArena);

            // 10. Residual connection: gpuX += gpuXb
            gpuAccumulate(gpuX, gpuXb, dim, tempArena);

            // No sync — gpuX carries forward to next layer
        }
    }

    // --- GPU kernel wrappers ---

    private void gpuRmsNorm(MemorySegment gpuOut, MemorySegment gpuIn,
                             MemorySegment gpuWeights, int size, Arena tempArena) {
        // Phase 1: compute partial sums of squares
        long globalWS = numWorkGroups * localWorkSize;
        long localMemBytes = localWorkSize * Float.BYTES;

        clContext.setKernelArgMem(rmsnormSumsqKernel, 0, gpuIn, tempArena);
        clContext.setKernelArgMem(rmsnormSumsqKernel, 1, gpuPartialSums, tempArena);
        clContext.setKernelArgInt(rmsnormSumsqKernel, 2, size, tempArena);
        clContext.setKernelArgLocal(rmsnormSumsqKernel, 3, localMemBytes);
        clContext.enqueueKernel1D(rmsnormSumsqKernel, globalWS, localWorkSize, tempArena);

        // Phase 2: normalize
        long normalizeGWS = ((size + localWorkSize - 1) / localWorkSize) * localWorkSize;
        clContext.setKernelArgMem(rmsnormNormalizeKernel, 0, gpuOut, tempArena);
        clContext.setKernelArgMem(rmsnormNormalizeKernel, 1, gpuIn, tempArena);
        clContext.setKernelArgMem(rmsnormNormalizeKernel, 2, gpuWeights, tempArena);
        clContext.setKernelArgMem(rmsnormNormalizeKernel, 3, gpuPartialSums, tempArena);
        clContext.setKernelArgInt(rmsnormNormalizeKernel, 4, size, tempArena);
        clContext.setKernelArgFloat(rmsnormNormalizeKernel, 5, normEps);
        clContext.setKernelArgInt(rmsnormNormalizeKernel, 6, numWorkGroups, tempArena);
        clContext.enqueueKernel1D(rmsnormNormalizeKernel, normalizeGWS, localWorkSize, tempArena);
    }

    private void gpuSilu(MemorySegment gpuBuf, int size, Arena tempArena) {
        clContext.setKernelArgMem(siluKernel, 0, gpuBuf, tempArena);
        clContext.setKernelArgInt(siluKernel, 1, size, tempArena);
        long gws = ((size + localWorkSize - 1) / localWorkSize) * localWorkSize;
        clContext.enqueueKernel1D(siluKernel, gws, localWorkSize, tempArena);
    }

    private void gpuAccumulate(MemorySegment gpuY, MemorySegment gpuXarg, int size, Arena tempArena) {
        clContext.setKernelArgMem(accumulateKernel, 0, gpuY, tempArena);
        clContext.setKernelArgMem(accumulateKernel, 1, gpuXarg, tempArena);
        clContext.setKernelArgInt(accumulateKernel, 2, size, tempArena);
        long gws = ((size + localWorkSize - 1) / localWorkSize) * localWorkSize;
        clContext.enqueueKernel1D(accumulateKernel, gws, localWorkSize, tempArena);
    }

    private void gpuElementwiseMul(MemorySegment gpuA, MemorySegment gpuB, int size, Arena tempArena) {
        clContext.setKernelArgMem(elementwiseMulKernel, 0, gpuA, tempArena);
        clContext.setKernelArgMem(elementwiseMulKernel, 1, gpuB, tempArena);
        clContext.setKernelArgInt(elementwiseMulKernel, 2, size, tempArena);
        long gws = ((size + localWorkSize - 1) / localWorkSize) * localWorkSize;
        clContext.enqueueKernel1D(elementwiseMulKernel, gws, localWorkSize, tempArena);
    }

    private void gpuFillZero(MemorySegment gpuBuf, int sizeFloats, Arena tempArena) {
        clContext.fillBufferZero(gpuBuf, (long) sizeFloats * Float.BYTES);
    }

    @Override
    public void close() {
        // Release GPU activation buffers
        try { OpenCLBindings.releaseMemObject(gpuX); } catch (Exception ignored) {}
        try { OpenCLBindings.releaseMemObject(gpuXb); } catch (Exception ignored) {}
        try { OpenCLBindings.releaseMemObject(gpuXb2); } catch (Exception ignored) {}
        try { OpenCLBindings.releaseMemObject(gpuHb); } catch (Exception ignored) {}
        try { OpenCLBindings.releaseMemObject(gpuHb2); } catch (Exception ignored) {}
        try { OpenCLBindings.releaseMemObject(gpuQ); } catch (Exception ignored) {}
        try { OpenCLBindings.releaseMemObject(gpuK); } catch (Exception ignored) {}
        try { OpenCLBindings.releaseMemObject(gpuV); } catch (Exception ignored) {}
        try { OpenCLBindings.releaseMemObject(gpuPartialSums); } catch (Exception ignored) {}
        for (MemorySegment buf : gpuAttnNormWeights) {
            if (buf != null) try { OpenCLBindings.releaseMemObject(buf); } catch (Exception ignored) {}
        }
        for (MemorySegment buf : gpuFfnNormWeights) {
            if (buf != null) try { OpenCLBindings.releaseMemObject(buf); } catch (Exception ignored) {}
        }
        arena.close();
    }
}
