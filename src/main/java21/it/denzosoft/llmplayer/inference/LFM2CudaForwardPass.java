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
 * Reuses the standard kernels (rmsnorm, rmsnorm_per_head, rope, flash attention, conv1d_short,
 * silu_mul, elementwise_mul, accumulate). Matmuls take the dp4a int8 path (default on, one Q8_1
 * quantization per input buffer) with each tensor's FP32 kernel as the fallback, and from the
 * second token the whole pass replays as a CUDA graph.
 *
 * Gated by {@link #isSupported}: if any matmul weight is not GPU-resident the engine falls back
 * to the per-tensor path, so this can never regress correctness.
 */
public class LFM2CudaForwardPass implements LayerGpuForwardPass {

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

    private final MemorySegment rmsnormFunc, perHeadNormFunc, ropeFunc, kvUpdateFunc;
    // FP16 KV cache (-Dcuda.kv.fp16, also set by the KV-aware VRAM budget); inline-init so the
    // constructor body sees it when sizing the KV buffers.
    private final boolean useFp16Kv = "true".equals(System.getProperty("cuda.kv.fp16", "false"));
    private FlashAttention flashAttn;
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

    private final PB matmulPB, normPB, perHeadPB, ropePB, kvPB, convPB, siluMulPB, elemMulPB, accumPB;

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
        kvUpdateFunc    = useFp16Kv
            ? cudaContext.compileKernel("kernels/cuda/attention_f16.cu", "kv_cache_update_f16")
            : cudaContext.compileKernel("kernels/cuda/attention.cu", "kv_cache_update");
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
        long kvBytes = (long) maxSeqLen * kvDim * (useFp16Kv ? 2L : fb);
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

        // LFM2-MoE: routed experts on the GPU (every expert tensor GPU-resident). The router runs
        // on the GPU; its `experts` logits are downloaded for the top-K selection, which stays on
        // the CPU exactly as LFM2InferenceEngine.moe does it.
        this.config = config;
        this.hasMoE = config.expertCount() > 0;
        if (hasMoE) {
            experts = config.expertCount();
            efd = config.expertFfnLength();
            gpuRouter = bufferManager.createBuffer((long) experts * fb);
            hostRouter = arena.allocate((long) experts * fb, 16);
            gpuExpGate = bufferManager.createBuffer((long) efd * fb);
            gpuExpUp = bufferManager.createBuffer((long) efd * fb);
            gpuExpOut = bufferManager.createBuffer((long) dim * fb);
            moeInMm = new it.denzosoft.llmplayer.gpu.Dp4aMatmul(cudaContext, bufferManager, arena, dim);
            moeDownMm = new it.denzosoft.llmplayer.gpu.Dp4aMatmul(cudaContext, bufferManager, arena, efd);
            axpyFunc = cudaContext.compileKernel("kernels/cuda/batch_ops.cu", "axpy");
            axpyPB = new PB(arena, 4);
            moeSiluPB = new PB(arena, 3);
            probs = new float[experts];
            routerIn = new float[dim];
            sel = new float[experts];
            expBias = new float[blockCount][];
            for (int i = 0; i < blockCount; i++) {
                LFM2LayerWeights lw = weights.layers()[i];
                if (lw.isMoE() && lw.expProbsBias() != null) {
                    expBias[i] = new float[experts];
                    for (int e = 0; e < experts; e++) expBias[i][e] = lw.expProbsBias().getFloat(e);
                }
            }
        }

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

        flashAttn = new FlashAttention(cudaContext, bufferManager, arena, headCount, headSize, maxSeqLen,
            useFp16Kv, gpuTokenParams);

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

    // LFM2-MoE (null / 0 for dense LFM2)
    private ModelConfig config;
    private boolean hasMoE;
    private int experts, efd;
    private long gpuRouter, gpuExpGate, gpuExpUp, gpuExpOut;
    private MemorySegment hostRouter, axpyFunc;
    private it.denzosoft.llmplayer.gpu.Dp4aMatmul moeInMm, moeDownMm;
    private PB axpyPB, moeSiluPB;
    private float[] probs, sel, routerIn;
    private float[][] expBias;
    private final int[] moeIds = new int[64];
    private final float[] moeW = new float[64];

    public static boolean isSupported(ModelConfig config, LFM2Weights weights) {
        if (weights.layers().length == 0) return false;
        if (!(weights.output() instanceof CudaFloatTensor)) {
            if (Boolean.getBoolean("cuda.debug")) System.err.println("LFM2 CUDA pass: output weight not on the GPU");
            return false;
        }
        for (LFM2LayerWeights lw : weights.layers()) {
            java.util.List<FloatTensor> mm = new java.util.ArrayList<>();
            if (lw.isAttention()) { mm.add(lw.wq()); mm.add(lw.wk()); mm.add(lw.wv()); mm.add(lw.wo()); }
            else { mm.add(lw.convInProj()); mm.add(lw.convOutProj()); }
            if (lw.isMoE()) {
                // LFM2-MoE runs here only with every expert GPU-resident (the model fits in VRAM)
                // (the router may stay on the CPU: it is a tiny F32 matrix)
                mm.add(lw.gateExps()); mm.add(lw.upExps()); mm.add(lw.downExps());
            } else {
                mm.add(lw.ffnGate()); mm.add(lw.ffnUp()); mm.add(lw.ffnDown());
            }
            for (FloatTensor t : mm) {
                if (!(t instanceof CudaFloatTensor)) {
                    if (Boolean.getBoolean("cuda.debug")) {
                        System.err.println("LFM2 CUDA pass: not supported, a " + (t == null ? "missing" : t.getClass().getSimpleName())
                            + " weight in a " + (lw.isAttention() ? "attention" : "conv") + (lw.isMoE() ? "/MoE" : "") + " layer");
                    }
                    return false;
                }
            }
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
        if (li == blockCount - 1 && position >= 0) warmedUp = true;
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
            flashAttn.launch(defaultStream, gpuAttnOut, gpuQ, gpuKeyCache[li], gpuValueCache[li],
                headCountKV, headSize, kvDim, 0, (float) (1.0 / Math.sqrt(headSize)), 0f, position);
            q8CachedIn = 0; // attention rewrote gpuAttnOut, a matmul input
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

        // FFN: ffn_norm -> SwiGLU (or routed experts) -> residual
        normPB.setLong(2, gpuFfnNorm[li]);
        launch(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB);
        if (lw.isMoE()) {
            moeFfn(li, lw);
            launch(accumFunc, accumGrid, (int) blockSize, 0, accumPB);
            return;
        }
        matmul((CudaFloatTensor) lw.ffnGate(), gpuNorm, gpuGate, ffnDim, dim);
        matmul((CudaFloatTensor) lw.ffnUp(), gpuNorm, gpuUp, ffnDim, dim);
        launch(siluMulFunc, (int) ((ffnDim + blockSize - 1) / blockSize), (int) blockSize, 0, siluMulPB); // gpuGate=silu(gpuGate)*gpuUp
        matmul((CudaFloatTensor) lw.ffnDown(), gpuGate, gpuBx, dim, ffnDim);
        launch(accumFunc, accumGrid, (int) blockSize, 0, accumPB);
    }

    /**
     * Routed-expert FFN of one LFM2-MoE layer into gpuBx: router on the GPU, top-K on the CPU (the
     * same selection, normalisation and scale as LFM2InferenceEngine.moe), then per selected expert
     * gate/up (dp4a, the layer input quantized once), SiLU·up, down, and gpuBx += w · out.
     */
    private void moeFfn(int li, LFM2LayerWeights lw) {
        long fb = Float.BYTES;
        if (lw.gateInp() instanceof CudaFloatTensor) {
            matmul((CudaFloatTensor) lw.gateInp(), gpuNorm, gpuRouter, experts, dim);
            cudaContext.readBuffer(gpuRouter, hostRouter, (long) experts * fb); // waits for the router
            MemorySegment.copy(hostRouter, ValueLayout.JAVA_FLOAT, 0, probs, 0, experts);
        } else {
            // CPU router: download the normed input (dim floats) and project it on the host
            cudaContext.readBuffer(gpuNorm, hostX, (long) dim * fb);
            MemorySegment.copy(hostX, ValueLayout.JAVA_FLOAT, 0, routerIn, 0, dim);
            java.util.Arrays.fill(probs, 0f);
            lw.gateInp().matmul(routerIn, probs, experts, dim);
        }
        if (config.expertGatingFunc() == 2) {
            for (int e = 0; e < experts; e++) probs[e] = 1.0f / (1.0f + (float) Math.exp(-probs[e]));
        } else {
            float max = Float.NEGATIVE_INFINITY;
            for (int e = 0; e < experts; e++) max = Math.max(max, probs[e]);
            float sum = 0f;
            for (int e = 0; e < experts; e++) { probs[e] = (float) Math.exp(probs[e] - max); sum += probs[e]; }
            for (int e = 0; e < experts; e++) probs[e] /= sum;
        }
        float[] bias = expBias[li];
        for (int e = 0; e < experts; e++) sel[e] = probs[e] + (bias != null ? bias[e] : 0f);
        int k = MoERouting.effectiveTopK(config.expertUsedCount());
        for (int j = 0; j < k; j++) {
            int best = -1;
            for (int e = 0; e < experts; e++) {
                boolean taken = false;
                for (int q = 0; q < j; q++) if (moeIds[q] == e) { taken = true; break; }
                if (!taken && (best < 0 || sel[e] > sel[best])) best = e;
            }
            moeIds[j] = best;
            moeW[j] = probs[best];
        }
        float sum = 0f;
        for (int j = 0; j < k; j++) sum += moeW[j];
        sum = Math.max(sum, 6.103515625e-5f);
        float scale = config.expertWeightsScale();
        float mul = (scale != 0f && scale != 1f) ? scale / sum : 1f / sum;

        CudaFloatTensor g = (CudaFloatTensor) lw.gateExps(), u = (CudaFloatTensor) lw.upExps(),
                        d = (CudaFloatTensor) lw.downExps();
        long gB = g.getWeightsBytes() / experts, uB = u.getWeightsBytes() / experts, dB = d.getWeightsBytes() / experts;
        cudaContext.fillBufferZero(gpuBx, (long) dim * fb);
        moeInMm.invalidate(); // new layer input in gpuNorm
        for (int j = 0; j < k; j++) {
            int e = moeIds[j];
            moeInMm.matmul(g, (long) e * gB, gpuNorm, gpuExpGate, efd, dim, false);
            moeInMm.matmul(u, (long) e * uB, gpuNorm, gpuExpUp, efd, dim, false);
            moeSiluPB.setLong(0, gpuExpGate); moeSiluPB.setLong(1, gpuExpUp); moeSiluPB.setInt(2, efd);
            launch(siluMulFunc, (int) ((efd + blockSize - 1) / blockSize), (int) blockSize, 0, moeSiluPB);
            moeDownMm.invalidate();
            moeDownMm.matmul(d, (long) e * dB, gpuExpGate, gpuExpOut, dim, efd, false);
            axpyPB.setLong(0, gpuBx); axpyPB.setLong(1, gpuExpOut); axpyPB.setFloat(2, moeW[j] * mul); axpyPB.setInt(3, dim);
            launch(axpyFunc, accumGrid, (int) blockSize, 0, axpyPB);
        }
    }

    public boolean forwardFinalLogits(float[] logits) {
        launchFinal();
        outputWarm = true;
        cudaContext.readBuffer(gpuLogits, hostLogits, gpuLogitsBytes);
        MemorySegment.copy(hostLogits, ValueLayout.JAVA_FLOAT, 0, logits, 0, vocabSize);
        return true;
    }

    private void launchFinal() {
        normPB.setLong(2, gpuOutputNorm);
        launch(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB);
        matmul((CudaFloatTensor) weights.output(), gpuNorm, gpuLogits, vocabSize, dim);
    }

    // --- CUDA graph: every kernel reads the position from gpuTokenParams, so one capture replays
    // for every token. Two executables: all layers + output projection (decode), layers only
    // (prefill tokens whose logits are discarded). A failed capture is not retried.
    private MemorySegment graphExec, graphExecLayers;
    private boolean graphAvailable;
    private boolean graphInit;
    // Weights upload to the GPU and kernels compile lazily on first use, and neither cuMemAlloc
    // nor module loading may happen inside a capture: the first token runs per-layer.
    private boolean warmedUp;     // every layer ran once outside a capture
    private boolean outputWarm;   // the output projection ran once outside a capture

    private boolean ensureGraph(boolean withOutput) {
        if (!graphInit) {
            graphInit = true;
            graphAvailable = !Boolean.getBoolean("cuda.nograph") && cudaContext.isGraphApiAvailable()
                && flashAttn.graphCompatible()
                && !hasMoE; // the MoE router decides on the host every layer
        }
        if (!graphAvailable || !warmedUp || (withOutput && !outputWarm)) return false;
        if ((withOutput ? graphExec : graphExecLayers) != null) return true;
        boolean capturing = false;
        try {
            cudaContext.beginCapture();
            capturing = true;
            for (int li = 0; li < blockCount; li++) forwardLayer(li, -1);
            if (withOutput) launchFinal();
            MemorySegment graph = cudaContext.endCapture();
            capturing = false;
            try {
                MemorySegment exec = cudaContext.instantiateGraph(graph);
                if (withOutput) graphExec = exec; else graphExecLayers = exec;
            } finally {
                cudaContext.destroyGraph(graph);
            }
            System.err.println("LFM2 CUDA graph: captured " + blockCount + " layers"
                + (withOutput ? " + output projection" : " (prefill)"));
            return true;
        } catch (Exception e) {
            if (capturing) {
                try {
                    MemorySegment partial = cudaContext.endCapture();
                    if (partial != null && partial.address() != 0) cudaContext.destroyGraph(partial);
                } catch (Exception ignored) {}
            }
            graphAvailable = false;
            System.err.println("LFM2 CUDA graph: capture failed — " + e.getMessage() + ", using per-layer mode");
            return false;
        }
    }

    @Override
    public boolean forwardGraph(float[] logits) {
        if (!ensureGraph(true)) return false;
        cudaContext.launchGraph(graphExec);
        cudaContext.readBuffer(gpuLogits, hostLogits, gpuLogitsBytes);
        MemorySegment.copy(hostLogits, ValueLayout.JAVA_FLOAT, 0, logits, 0, vocabSize);
        return true;
    }

    @Override
    public boolean forwardGraphPrefill() {
        if (!ensureGraph(false)) return false;
        cudaContext.launchGraph(graphExecLayers);
        return true;
    }

    // Q8_1 quantization cache: consecutive dp4a matmuls on the same input (Q/K/V, gate/up) reuse
    // one quantization. Any other kernel launch invalidates it (see launch()), as does a matmul
    // that writes the quantized buffer itself.
    private long q8CachedIn;
    private int q8CachedCols;

    private void matmul(CudaFloatTensor t, long in, long out, int rows, int cols) {
        MemorySegment dp4a = useDp4a ? dp4aFunc(t) : null;
        if (dp4a != null) {
            // quantize FP32 input[cols] -> Q8_1 (skipped when already quantized), then int8 dp4a matmul
            if (q8CachedIn != in || q8CachedCols != cols) {
                quantPB.setLong(0, in); quantPB.setLong(1, gpuQ8In); quantPB.setInt(2, cols);
                launchRaw(quantizeFunc, (((cols + 31) / 32) + 7) / 8, 256, 0, quantPB);
                q8CachedIn = in;
                q8CachedCols = cols;
            }
            dp4aPB.setLong(0, t.getGpuWeights()); dp4aPB.setLong(1, gpuQ8In); dp4aPB.setLong(2, out);
            dp4aPB.setInt(3, rows); dp4aPB.setInt(4, cols); dp4aPB.setInt(5, 0);
            launchRaw(dp4a, t.getMatmulGridDim(rows, cols), t.getMatmulBlockDim(cols), 0, dp4aPB);
            if (out == q8CachedIn) q8CachedIn = 0;
            return;
        }
        matmulPB.setLong(0, t.getGpuWeights()); matmulPB.setLong(1, in); matmulPB.setLong(2, out);
        matmulPB.setInt(3, rows); matmulPB.setInt(4, cols); matmulPB.setInt(5, 0); // write mode
        launchRaw(t.getCudaFunction(), t.getMatmulGridDim(rows, cols), t.getMatmulBlockDim(cols),
               t.getMatmulSharedMem(cols), matmulPB);
        if (out == q8CachedIn) q8CachedIn = 0;
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

    /** Launch a non-matmul kernel; it may overwrite a matmul input, so the Q8_1 cache is dropped. */
    private void launch(MemorySegment fn, int grid, int block, int sm, PB params) {
        q8CachedIn = 0;
        launchRaw(fn, grid, block, sm, params);
    }

    private void launchRaw(MemorySegment fn, int grid, int block, int sm, PB params) {
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
    public void close() {
        if (graphExec != null) try { cudaContext.destroyGraphExec(graphExec); } catch (Exception ignored) {}
        if (graphExecLayers != null) try { cudaContext.destroyGraphExec(graphExecLayers); } catch (Exception ignored) {}
        arena.close();
    }
}
