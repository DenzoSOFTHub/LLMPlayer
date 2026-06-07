package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.gpu.CudaBindings;
import it.denzosoft.llmplayer.gpu.CudaBufferManager;
import it.denzosoft.llmplayer.gpu.CudaContext;
import it.denzosoft.llmplayer.model.LFM2LayerWeights;
import it.denzosoft.llmplayer.model.LFM2Weights;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.tensor.CudaFloatTensor;
import it.denzosoft.llmplayer.tensor.FloatTensor;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * GPU-resident per-layer forward pass for LFM2 (gated short-conv + GQA hybrid).
 * Keeps activations on the GPU across a whole layer (no per-matmul CPU round-trips like the
 * per-tensor path), running every op — conv, attention, RoPE, QK-norm, SwiGLU — as a CUDA kernel.
 *
 * Reuses the standard kernels (rmsnorm, rmsnorm_per_head, rope, attention, conv1d_short,
 * silu_mul, elementwise_mul, accumulate) and each weight tensor's own FP32 matmul kernel.
 * No CUDA-graph capture and no dp4a in this first version (a future optimization); the win is
 * eliminating the per-matmul host↔device transfers of the per-tensor fallback.
 *
 * Gated by {@link #isSupported}: if any matmul weight is not GPU-resident the engine falls back
 * to the per-tensor path, so this can never regress correctness.
 */
public class LFM2CudaForwardPass implements AutoCloseable {

    private final CudaContext cudaContext;
    private final CudaBufferManager bufferManager;
    private final Arena arena;
    private final MemorySegment defaultStream;
    private final LFM2Weights weights;

    private final int dim, vocabSize, blockCount, maxSeqLen;
    private final int headCount, headCountKV, headSize, kvDim, qDim, ffnDim;
    private final int lCache, histSize, halfRope, ropeType;
    private final float normEps;
    private final long blockSize;
    private final boolean[] isAttn;

    // gpuCombined = [gpuX (dim floats)][tokenParams (2 ints)]
    private final long gpuCombined, gpuX, gpuTokenParams;
    private final long gpuNorm, gpuBcx, gpuBx, gpuQ, gpuK, gpuV, gpuAttnOut, gpuGate, gpuUp;
    private final long gpuLogits, gpuLogitsBytes;
    private final long gpuCosTable, gpuSinTable;
    private final MemorySegment hostCombined, hostX, hostLogits;

    private final long[] gpuOpNorm, gpuFfnNorm, gpuQNorm, gpuKNorm;     // per-layer norm weights
    private final long[] gpuConvW, gpuConvState;                        // conv layers
    private final long[] gpuKeyCache, gpuValueCache;                    // attention layers

    private final MemorySegment rmsnormFunc, perHeadNormFunc, ropeFunc, kvUpdateFunc, attnFunc;
    private final MemorySegment convFunc, siluMulFunc, elemMulFunc, accumFunc;

    // dp4a (int8) matmul path: quantize FP32 input -> Q8_1, then per-type dp4a kernel. Default on
    // (-Dcuda.dp4a), with FP32 fallback for ineligible types (Q6_K/F32/...) and on disable.
    private final boolean useDp4a = !"false".equals(System.getProperty("cuda.dp4a", "true"));
    private final MemorySegment quantizeFunc, dp4aQ4kFunc, dp4aQ5kFunc, dp4aQ50Func, dp4aQ80Func,
                                dp4aQ3kFunc, dp4aIq4nlFunc, dp4aIq4xsFunc;
    private final long gpuQ8In;   // Q8_1 input scratch, sized for the largest matmul input
    private final PB quantPB, dp4aPB;

    private final int normSharedMem, perHeadBlockDim, perHeadSharedMem;
    private final int ropeQGrid, ropeKGrid, kvGrid, convGrid, accumGrid;

    private static final class PB {
        final MemorySegment args, ptrs;
        PB(Arena a, int n) {
            args = a.allocate(n * 8L, 8);
            ptrs = a.allocate(ValueLayout.ADDRESS, n);
            for (int i = 0; i < n; i++) ptrs.setAtIndex(ValueLayout.ADDRESS, i, args.asSlice(i * 8L, 8));
        }
        void setLong(int i, long v) { args.set(ValueLayout.JAVA_LONG, i * 8L, v); }
        void setInt(int i, int v) { args.set(ValueLayout.JAVA_INT, i * 8L, v); }
        void setFloat(int i, float v) { args.set(ValueLayout.JAVA_FLOAT, i * 8L, v); }
    }

    private final PB matmulPB, normPB, perHeadPB, ropePB, kvPB, attnPB, convPB, siluMulPB, elemMulPB, accumPB;

    public LFM2CudaForwardPass(ModelConfig config, LFM2Weights weights,
                               CudaBufferManager bufferManager, int maxSeqLen) {
        this.cudaContext = bufferManager.getCudaContext();
        this.bufferManager = bufferManager;
        this.weights = weights;
        this.arena = Arena.ofShared();
        this.defaultStream = cudaContext.getStream();
        this.maxSeqLen = maxSeqLen;

        this.dim = config.embeddingLength();
        this.vocabSize = config.vocabSize();
        this.blockCount = config.blockCount();
        this.headCount = config.headCount();
        this.headCountKV = config.headCountKV();
        this.headSize = config.headSize();
        this.kvDim = config.kvDim();
        this.qDim = headCount * headSize;
        this.ffnDim = config.intermediateSize();
        this.lCache = config.ssmConvKernel();
        this.histSize = Math.max(1, lCache - 1);
        this.normEps = config.normEps();
        long maxWg = cudaContext.getDeviceInfo().maxWorkGroupSize();
        this.blockSize = Math.min(256, maxWg);

        RoPE rope = new RoPE(headSize, config.ropeDimensionCount(), maxSeqLen,
            config.ropeFreqBase(), config.ropeType(), weights.ropeFreqFactors());
        this.halfRope = rope.getRopeDimCount() / 2;
        this.ropeType = rope.getRopeType();

        this.isAttn = new boolean[blockCount];
        for (int i = 0; i < blockCount; i++) isAttn[i] = config.lfm2IsAttentionLayer(i);

        long fb = Float.BYTES;

        // Kernels
        rmsnormFunc     = cudaContext.compileKernel("kernels/cuda/rmsnorm.cu", "rmsnorm_fused");
        perHeadNormFunc = cudaContext.compileKernel("kernels/cuda/rmsnorm_per_head.cu", "rmsnorm_per_head");
        ropeFunc        = cudaContext.compileKernel("kernels/cuda/rope.cu", "rope_apply");
        kvUpdateFunc    = cudaContext.compileKernel("kernels/cuda/attention.cu", "kv_cache_update");
        attnFunc        = cudaContext.compileKernel("kernels/cuda/attention.cu", "attention_full");
        convFunc        = cudaContext.compileKernel("kernels/cuda/conv1d_short.cu", "conv1d_short");
        siluMulFunc     = cudaContext.compileKernel("kernels/cuda/silu_mul.cu", "silu_mul");
        elemMulFunc     = cudaContext.compileKernel("kernels/cuda/elementwise_mul.cu", "elementwise_mul");
        accumFunc       = cudaContext.compileKernel("kernels/cuda/accumulate.cu", "accumulate");

        if (useDp4a) {
            quantizeFunc  = cudaContext.compileKernel("kernels/cuda/quantize_q8.cu", "quantize_q8");
            dp4aQ4kFunc   = cudaContext.compileKernel("kernels/cuda/matmul_q4_k_dp4a.cu", "matmul_q4_k_dp4a");
            dp4aQ5kFunc   = cudaContext.compileKernel("kernels/cuda/matmul_q5_k_dp4a.cu", "matmul_q5_k_dp4a");
            dp4aQ50Func   = cudaContext.compileKernel("kernels/cuda/matmul_q5_0_dp4a.cu", "matmul_q5_0_dp4a");
            dp4aQ80Func   = cudaContext.compileKernel("kernels/cuda/matmul_q8_0_dp4a.cu", "matmul_q8_0_dp4a");
            dp4aQ3kFunc   = cudaContext.compileKernel("kernels/cuda/matmul_q3_k_dp4a.cu", "matmul_q3_k_dp4a");
            dp4aIq4nlFunc = cudaContext.compileKernel("kernels/cuda/matmul_iq4_nl_dp4a.cu", "matmul_iq4_nl_dp4a");
            dp4aIq4xsFunc = cudaContext.compileKernel("kernels/cuda/matmul_iq4_xs_dp4a.cu", "matmul_iq4_xs_dp4a");
        } else {
            quantizeFunc = dp4aQ4kFunc = dp4aQ5kFunc = dp4aQ50Func = dp4aQ80Func
                = dp4aQ3kFunc = dp4aIq4nlFunc = dp4aIq4xsFunc = null;
        }

        // Buffers
        long combinedBytes = dim * fb + 8;
        gpuCombined = bufferManager.createBuffer(combinedBytes);
        gpuX = gpuCombined;
        gpuTokenParams = gpuCombined + dim * fb;
        hostCombined = arena.allocate(combinedBytes, 8);
        hostX = arena.allocate(ValueLayout.JAVA_FLOAT, dim);

        gpuNorm = bufferManager.createBuffer(dim * fb);
        gpuBcx  = bufferManager.createBuffer(3L * dim * fb);
        gpuBx   = bufferManager.createBuffer(dim * fb);
        gpuQ    = bufferManager.createBuffer((long) qDim * fb);
        gpuK    = bufferManager.createBuffer((long) kvDim * fb);
        gpuV    = bufferManager.createBuffer((long) kvDim * fb);
        gpuAttnOut = bufferManager.createBuffer((long) qDim * fb);
        gpuGate = bufferManager.createBuffer((long) ffnDim * fb);
        gpuUp   = bufferManager.createBuffer((long) ffnDim * fb);
        // Q8_1 input scratch: 40 bytes per 32-element block; sized for the largest matmul input.
        int maxIn = Math.max(dim, ffnDim);
        gpuQ8In = useDp4a ? bufferManager.createBuffer((long) ((maxIn + 31) / 32) * 40) : 0;

        gpuCosTable = uploadFloatArray(rope.getCosTable());
        gpuSinTable = uploadFloatArray(rope.getSinTable());

        // Per-layer weights + state
        gpuOpNorm = new long[blockCount];
        gpuFfnNorm = new long[blockCount];
        gpuQNorm = new long[blockCount];
        gpuKNorm = new long[blockCount];
        gpuConvW = new long[blockCount];
        gpuConvState = new long[blockCount];
        gpuKeyCache = new long[blockCount];
        gpuValueCache = new long[blockCount];
        long kvBytes = (long) maxSeqLen * kvDim * fb;
        long convBytes = (long) histSize * dim * fb;
        for (int i = 0; i < blockCount; i++) {
            LFM2LayerWeights lw = weights.layers()[i];
            gpuOpNorm[i] = uploadNormWeights(lw.operatorNorm(), dim);
            gpuFfnNorm[i] = uploadNormWeights(lw.ffnNorm(), dim);
            if (isAttn[i]) {
                gpuQNorm[i] = uploadNormWeights(lw.qNorm(), headSize);
                gpuKNorm[i] = uploadNormWeights(lw.kNorm(), headSize);
                gpuKeyCache[i] = bufferManager.createBuffer(kvBytes);
                gpuValueCache[i] = bufferManager.createBuffer(kvBytes);
            } else {
                gpuConvW[i] = uploadTensorAsFloats(lw.conv(), lCache * dim);
                gpuConvState[i] = bufferManager.createBuffer(convBytes);
            }
        }

        // Output (tied to token_embd) — must be GPU-resident
        FloatTensor out = weights.output();
        gpuLogits = bufferManager.createBuffer((long) vocabSize * fb);
        gpuLogitsBytes = (long) vocabSize * fb;
        hostLogits = arena.allocate(ValueLayout.JAVA_FLOAT, vocabSize);
        long outNorm = uploadNormWeights(weights.outputNorm(), dim);

        // Param buffers
        matmulPB = new PB(arena, 6);
        quantPB = new PB(arena, 3);
        dp4aPB = new PB(arena, 6);
        normPB = new PB(arena, 5);
        normPB.setLong(0, gpuNorm); normPB.setLong(1, gpuX); normPB.setInt(3, dim); normPB.setFloat(4, normEps);
        this.gpuOutputNorm = outNorm;

        perHeadPB = new PB(arena, 4);
        perHeadPB.setInt(2, headSize); perHeadPB.setFloat(3, normEps);

        ropePB = new PB(arena, 8);
        ropePB.setLong(1, gpuCosTable); ropePB.setLong(2, gpuSinTable);
        ropePB.setInt(4, headSize); ropePB.setInt(5, halfRope);
        ropePB.setLong(6, gpuTokenParams); ropePB.setInt(7, ropeType);

        kvPB = new PB(arena, 6);
        kvPB.setLong(2, gpuK); kvPB.setLong(3, gpuV); kvPB.setInt(4, kvDim); kvPB.setLong(5, gpuTokenParams);

        attnPB = new PB(arena, 10);
        attnPB.setLong(0, gpuAttnOut); attnPB.setLong(1, gpuQ);
        attnPB.setInt(4, headCount); attnPB.setInt(5, headCountKV);
        attnPB.setInt(6, headSize); attnPB.setInt(7, kvDim); attnPB.setLong(8, gpuTokenParams);
        attnPB.setInt(9, 0);

        convPB = new PB(arena, 6);
        convPB.setLong(0, gpuBcx); convPB.setInt(3, dim); convPB.setInt(4, lCache); convPB.setLong(5, gpuTokenParams);

        siluMulPB = new PB(arena, 3);
        siluMulPB.setLong(0, gpuGate); siluMulPB.setLong(1, gpuUp); siluMulPB.setInt(2, ffnDim);

        elemMulPB = new PB(arena, 3);
        elemMulPB.setInt(2, dim);

        accumPB = new PB(arena, 3);
        accumPB.setLong(0, gpuX); accumPB.setLong(1, gpuBx); accumPB.setInt(2, dim);

        int normNumWarps = (int) (blockSize / 32);
        this.normSharedMem = (normNumWarps + 1) * Float.BYTES;
        this.perHeadBlockDim = (int) Math.min(Math.max(32, ((headSize + 31) / 32) * 32), blockSize);
        this.perHeadSharedMem = ((perHeadBlockDim / 32) + 1) * Float.BYTES;
        this.ropeQGrid = (int) ((headCount * halfRope + blockSize - 1) / blockSize);
        this.ropeKGrid = (int) ((headCountKV * halfRope + blockSize - 1) / blockSize);
        this.kvGrid = (int) ((kvDim + blockSize - 1) / blockSize);
        this.convGrid = (int) ((dim + blockSize - 1) / blockSize);
        this.accumGrid = (int) ((dim + blockSize - 1) / blockSize);
    }

    private final long gpuOutputNorm;

    public static boolean isSupported(ModelConfig config, LFM2Weights weights) {
        if (weights.layers().length == 0) return false;
        if (!(weights.output() instanceof CudaFloatTensor)) return false;
        for (LFM2LayerWeights lw : weights.layers()) {
            FloatTensor[] mm = lw.isAttention()
                ? new FloatTensor[]{lw.wq(), lw.wk(), lw.wv(), lw.wo(), lw.ffnGate(), lw.ffnUp(), lw.ffnDown()}
                : new FloatTensor[]{lw.convInProj(), lw.convOutProj(), lw.ffnGate(), lw.ffnUp(), lw.ffnDown()};
            for (FloatTensor t : mm) if (!(t instanceof CudaFloatTensor)) return false;
        }
        return true;
    }

    public int getGpuLayerCount() { return blockCount; }

    public void uploadXAndUpdateParams(float[] x, int position) {
        long embBytes = (long) dim * Float.BYTES;
        MemorySegment.copy(x, 0, hostCombined, ValueLayout.JAVA_FLOAT, 0, dim);
        hostCombined.set(ValueLayout.JAVA_INT, embBytes, position);
        hostCombined.set(ValueLayout.JAVA_INT, embBytes + 4, position + 1);
        cudaContext.writeBuffer(gpuCombined, hostCombined, embBytes + 8);
    }

    public void downloadX(float[] x) {
        cudaContext.readBuffer(gpuX, hostX, (long) dim * Float.BYTES);
        MemorySegment.copy(hostX, ValueLayout.JAVA_FLOAT, 0, x, 0, dim);
    }

    public void forwardLayer(int li, int position) {
        LFM2LayerWeights lw = weights.layers()[li];
        long fb = Float.BYTES;

        // operator_norm: gpuX -> gpuNorm
        normPB.setLong(2, gpuOpNorm[li]);
        launch(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB);

        if (isAttn[li]) {
            matmul((CudaFloatTensor) lw.wq(), gpuNorm, gpuQ, qDim, dim);
            matmul((CudaFloatTensor) lw.wk(), gpuNorm, gpuK, kvDim, dim);
            matmul((CudaFloatTensor) lw.wv(), gpuNorm, gpuV, kvDim, dim);
            // per-head QK-norm (before RoPE)
            perHeadPB.setLong(0, gpuQ); perHeadPB.setLong(1, gpuQNorm[li]);
            launch(perHeadNormFunc, headCount, perHeadBlockDim, perHeadSharedMem, perHeadPB);
            perHeadPB.setLong(0, gpuK); perHeadPB.setLong(1, gpuKNorm[li]);
            launch(perHeadNormFunc, headCountKV, perHeadBlockDim, perHeadSharedMem, perHeadPB);
            // RoPE (NEOX)
            ropePB.setLong(0, gpuQ); ropePB.setInt(3, headCount);
            launch(ropeFunc, ropeQGrid, (int) blockSize, 0, ropePB);
            ropePB.setLong(0, gpuK); ropePB.setInt(3, headCountKV);
            launch(ropeFunc, ropeKGrid, (int) blockSize, 0, ropePB);
            // KV cache update + attention
            kvPB.setLong(0, gpuKeyCache[li]); kvPB.setLong(1, gpuValueCache[li]);
            launch(kvUpdateFunc, kvGrid, (int) blockSize, 0, kvPB);
            attnPB.setLong(2, gpuKeyCache[li]); attnPB.setLong(3, gpuValueCache[li]);
            int attnSM = (position + 1 + 32) * Float.BYTES;
            launch(attnFunc, headCount, Math.min(256, (int) blockSize), attnSM, attnPB);
            // wo: gpuAttnOut -> gpuBx
            matmul((CudaFloatTensor) lw.wo(), gpuAttnOut, gpuBx, dim, qDim);
        } else {
            // in_proj -> gpuBcx [b | c | x]
            matmul((CudaFloatTensor) lw.convInProj(), gpuNorm, gpuBcx, 3 * dim, dim);
            // bx = b * x  (in place: b-region *= x-region)
            elemMulPB.setLong(0, gpuBcx); elemMulPB.setLong(1, gpuBcx + 2L * dim * fb);
            launch(elemMulFunc, convGrid, (int) blockSize, 0, elemMulPB);
            // depthwise causal conv1d (in place on bx region)
            convPB.setLong(1, gpuConvState[li]); convPB.setLong(2, gpuConvW[li]);
            launch(convFunc, convGrid, (int) blockSize, 0, convPB);
            // y = c * conv_out  (in place: bx-region(=conv_out) *= c-region)
            elemMulPB.setLong(0, gpuBcx); elemMulPB.setLong(1, gpuBcx + (long) dim * fb);
            launch(elemMulFunc, convGrid, (int) blockSize, 0, elemMulPB);
            // out_proj: gpuBcx(y) -> gpuBx
            matmul((CudaFloatTensor) lw.convOutProj(), gpuBcx, gpuBx, dim, dim);
        }
        // residual: gpuX += gpuBx
        launch(accumFunc, accumGrid, (int) blockSize, 0, accumPB);

        // FFN: ffn_norm -> SwiGLU -> residual
        normPB.setLong(2, gpuFfnNorm[li]);
        launch(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB);
        matmul((CudaFloatTensor) lw.ffnGate(), gpuNorm, gpuGate, ffnDim, dim);
        matmul((CudaFloatTensor) lw.ffnUp(), gpuNorm, gpuUp, ffnDim, dim);
        launch(siluMulFunc, (int) ((ffnDim + blockSize - 1) / blockSize), (int) blockSize, 0, siluMulPB); // gpuGate=silu(gpuGate)*gpuUp
        matmul((CudaFloatTensor) lw.ffnDown(), gpuGate, gpuBx, dim, ffnDim);
        launch(accumFunc, accumGrid, (int) blockSize, 0, accumPB);
    }

    public boolean forwardFinalLogits(float[] logits) {
        normPB.setLong(2, gpuOutputNorm);
        launch(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB);
        matmul((CudaFloatTensor) weights.output(), gpuNorm, gpuLogits, vocabSize, dim);
        cudaContext.readBuffer(gpuLogits, hostLogits, gpuLogitsBytes);
        MemorySegment.copy(hostLogits, ValueLayout.JAVA_FLOAT, 0, logits, 0, vocabSize);
        return true;
    }

    private void matmul(CudaFloatTensor t, long in, long out, int rows, int cols) {
        MemorySegment dp4a = useDp4a ? dp4aFunc(t) : null;
        if (dp4a != null) {
            // quantize FP32 input[cols] -> Q8_1, then int8 dp4a matmul
            quantPB.setLong(0, in); quantPB.setLong(1, gpuQ8In); quantPB.setInt(2, cols);
            launch(quantizeFunc, (((cols + 31) / 32) + 7) / 8, 256, 0, quantPB);
            dp4aPB.setLong(0, t.getGpuWeights()); dp4aPB.setLong(1, gpuQ8In); dp4aPB.setLong(2, out);
            dp4aPB.setInt(3, rows); dp4aPB.setInt(4, cols); dp4aPB.setInt(5, 0);
            launch(dp4a, t.getMatmulGridDim(rows, cols), t.getMatmulBlockDim(cols), 0, dp4aPB);
            return;
        }
        matmulPB.setLong(0, t.getGpuWeights()); matmulPB.setLong(1, in); matmulPB.setLong(2, out);
        matmulPB.setInt(3, rows); matmulPB.setInt(4, cols); matmulPB.setInt(5, 0); // write mode
        launch(t.getCudaFunction(), t.getMatmulGridDim(rows, cols), t.getMatmulBlockDim(cols),
               t.getMatmulSharedMem(cols), matmulPB);
    }

    /** dp4a kernel for the tensor's quant type, or null if not dp4a-eligible (FP32 fallback). */
    private MemorySegment dp4aFunc(CudaFloatTensor t) {
        switch (t.type()) {
            case Q4_K:   return dp4aQ4kFunc;
            case Q5_K:   return dp4aQ5kFunc;
            case Q5_0:   return dp4aQ50Func;
            case Q8_0:   return dp4aQ80Func;
            case Q3_K:   return dp4aQ3kFunc;
            case IQ4_NL: return dp4aIq4nlFunc;
            case IQ4_XS: return dp4aIq4xsFunc;
            default:     return null;   // Q6_K / F32 / etc. -> FP32 kernel
        }
    }

    private void launch(MemorySegment fn, int grid, int block, int sm, PB params) {
        int err = CudaBindings.launchKernel(fn, grid, 1, 1, block, 1, 1, sm, defaultStream, params.ptrs, MemorySegment.NULL);
        if (err != CudaBindings.CUDA_SUCCESS) throw new RuntimeException("LFM2 CUDA error: " + err);
    }

    private long uploadNormWeights(FloatTensor t, int size) {
        float[] w = new float[size]; for (int i = 0; i < size; i++) w[i] = t.getFloat(i);
        return bufferManager.uploadNormWeights(w);
    }

    private long uploadTensorAsFloats(FloatTensor t, int size) {
        float[] w = new float[size]; for (int i = 0; i < size; i++) w[i] = t.getFloat(i);
        return uploadFloatArray(w);
    }

    private long uploadFloatArray(float[] data) {
        long bytes = (long) data.length * Float.BYTES;
        long ptr = bufferManager.createBuffer(bytes);
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment host = temp.allocate(ValueLayout.JAVA_FLOAT, data.length);
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_FLOAT, 0, data.length);
            cudaContext.writeBuffer(ptr, host, bytes);
        }
        return ptr;
    }

    @Override
    public void close() { arena.close(); }
}
