package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.gpu.CudaBindings;
import it.denzosoft.llmplayer.gpu.CudaBufferManager;
import it.denzosoft.llmplayer.gpu.CudaContext;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.model.ModelWeights;
import it.denzosoft.llmplayer.model.TransformerLayerWeights;
import it.denzosoft.llmplayer.tensor.CudaFloatTensor;
import it.denzosoft.llmplayer.tensor.FloatTensor;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * GPU-resident per-layer forward pass for the Gemma 4 architecture (PLE-only path —
 * NOT the Gemma 3n AltUp/Laurel path, which stays on the CPU engine).
 *
 * <p>Ports {@link Gemma4InferenceEngine#forwardLayer} exactly to the GPU, keeping the residual
 * stream resident in {@code gpuX} across a whole layer (no per-matmul host↔device round-trips).
 * Reuses the standard kernels (rmsnorm, rmsnorm_per_head, rope, attention, gelu, elementwise_mul,
 * accumulate, scale_inplace) and each weight tensor's own FP32 / dp4a matmul kernel.
 *
 * <p>Gemma-4-specific machinery (vs the LFM2 reference this is modelled on):
 * <ul>
 *   <li><b>Dual per-layer head size</b> — SWA layers use {@code config.headSize()}, full-attention
 *       layers use {@code config.keyLength()}. Q/K/V/attnOut buffers are allocated at the MAX size;
 *       the rope/attention/per-head param buffers' headSize/kvDim fields are rewritten per layer.</li>
 *   <li><b>Dual RoPE</b> — SWA theta (default 10K) vs full theta (1M with freq factors); two cos/sin
 *       tables uploaded; the matching one is selected per layer.</li>
 *   <li><b>Attention scale = 1.0</b> — the {@code attention_full} kernel hard-codes
 *       {@code 1/sqrt(headSize)}, so after RoPE on Q we multiply Q by {@code sqrt(headSize)} to cancel
 *       it back to an effective scale of 1.0.</li>
 *   <li><b>V-norm</b> — RMS without learnable scale, per head: reuse {@code rmsnorm_per_head} with a
 *       buffer of all-1.0 weights.</li>
 *   <li><b>Shared KV</b> — the last {@code sharedKvLayers} layers reuse an earlier layer's KV cache;
 *       only own-KV layers allocate / populate a cache.</li>
 *   <li><b>Pre + post norms</b> (Gemma), <b>GeGLU</b> (gelu instead of silu), <b>PLE injection</b>,
 *       and a per-layer <b>layer_output_scale</b> multiply.</li>
 * </ul>
 *
 * <p>Embedding lookup + scaling, the PLE pre-computation ({@code pleCombined}), and the final logit
 * soft-cap stay on the CPU side in the engine. This class only runs the per-layer GPU work + the
 * final RMSNorm + output projection (raw logits, no soft-cap).
 *
 * <p>No CUDA-graph capture in this first version (per-layer headSize varies → graph capture would
 * need per-layer param rewrites; deferred). dp4a is default-on ({@code -Dcuda.dp4a}). Gated by
 * {@link #isSupported}: if any matmul weight (incl. PLE tensors) is not GPU-resident the engine
 * falls back to the per-tensor / CPU path, so this can never regress correctness.
 */
public class Gemma4CudaForwardPass implements AutoCloseable {

    private final CudaContext cudaContext;
    private final CudaBufferManager bufferManager;
    private final Arena arena;
    private final MemorySegment defaultStream;
    private final ModelWeights weights;

    private final int dim, vocabSize, blockCount, maxSeqLen, ffnDim;
    private final int headCount, headCountKV, kvMul;
    private final int headSizeSwa, headSizeFull, maxHeadSize, maxQDim, maxKvDim;
    private final int slidingWindow, sharedKvLayers, pleDim;
    private final float normEps;
    private final long blockSize;

    private final int ropeTypeSwa, halfRopeSwa, ropeTypeFull, halfRopeFull;

    private final boolean[] isSwa;
    private final int[] kvSourceLayer;
    private final FloatTensor[] pleInpGate, pleProj;
    private final float[] layerOutputScale;

    // gpuCombined = [gpuX (dim floats)][tokenParams (2 ints)]
    private final long gpuCombined, gpuX, gpuTokenParams;
    private final long gpuNorm, gpuQ, gpuK, gpuV, gpuAttnOut, gpuGate, gpuUp, gpuBx, gpuAttnRes;
    private final long gpuPleGate, gpuPleOut, gpuPleCombined;
    private final long gpuOnes;
    private final long gpuLogits, gpuLogitsBytes;
    private final long gpuCosTableSwa, gpuSinTableSwa, gpuCosTableFull, gpuSinTableFull;
    private final long gpuOutputNorm;
    private final MemorySegment hostCombined, hostX, hostLogits, hostPle;

    // Per-layer norm weights + KV cache
    private final long[] gpuAttnNorm, gpuFfnNorm, gpuQNorm, gpuKNorm, gpuPostAttnNorm, gpuPostFfnNorm;
    private final long[] gpuPlePostNorm;
    private final long[] gpuKeyCache, gpuValueCache;

    private final MemorySegment rmsnormFunc, perHeadNormFunc, ropeFunc, kvUpdateFunc, attnFunc;
    private final MemorySegment geluFunc, elemMulFunc, accumFunc, scaleFunc;

    // dp4a (int8) matmul path: quantize FP32 input -> Q8_1, then per-type dp4a kernel. Default on
    // (-Dcuda.dp4a), with FP32 fallback for ineligible types (Q6_K/F32/...) and on disable.
    private final boolean useDp4a = !"false".equals(System.getProperty("cuda.dp4a", "true"));
    private final MemorySegment quantizeFunc, dp4aQ4kFunc, dp4aQ5kFunc, dp4aQ50Func, dp4aQ80Func,
                                dp4aQ3kFunc, dp4aIq4nlFunc, dp4aIq4xsFunc;
    private final long gpuQ8In;   // Q8_1 input scratch, sized for the largest matmul input
    private final PB quantPB, dp4aPB;

    private final int normSharedMem, perHeadBlockDim, perHeadSharedMem;

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

    private final PB matmulPB, normPB, perHeadPB, ropePB, kvPB, attnPB, geluPB, elemMulPB, accumPB, scalePB;

    public Gemma4CudaForwardPass(ModelConfig config, ModelWeights weights,
                                 CudaBufferManager bufferManager, int maxSeqLen,
                                 FloatTensor[] pleInpGate, FloatTensor[] pleProj,
                                 float[][] plePostNorm, float[] layerOutputScale,
                                 float[] ropeFreqFactors, int pleDim) {
        this.cudaContext = bufferManager.getCudaContext();
        this.bufferManager = bufferManager;
        this.weights = weights;
        this.arena = Arena.ofShared();
        this.defaultStream = cudaContext.getStream();
        this.maxSeqLen = maxSeqLen;

        this.dim = config.embeddingLength();
        this.vocabSize = config.vocabSize();
        this.blockCount = config.blockCount();
        this.ffnDim = config.intermediateSize();
        this.headCount = config.headCount();
        this.headCountKV = config.headCountKV();
        this.kvMul = headCount / headCountKV;
        this.normEps = config.normEps();
        this.slidingWindow = config.slidingWindow();
        this.sharedKvLayers = config.sharedKvLayers();
        this.pleDim = pleDim;
        this.pleInpGate = pleInpGate;
        this.pleProj = pleProj;
        this.layerOutputScale = layerOutputScale;
        long maxWg = cudaContext.getDeviceInfo().maxWorkGroupSize();
        this.blockSize = Math.min(256, maxWg);

        // Dual head size: SWA layers use headSize(), full-attention layers use keyLength().
        this.headSizeSwa = config.headSize();
        this.headSizeFull = config.keyLength() > 0 ? config.keyLength() : headSizeSwa;
        this.maxHeadSize = Math.max(headSizeSwa, headSizeFull);
        this.maxQDim = headCount * maxHeadSize;
        this.maxKvDim = headCountKV * maxHeadSize;

        // SWA pattern (mirrors Gemma4InferenceEngine.isSwaLayer).
        boolean[] swaPattern = config.slidingWindowPattern();
        this.isSwa = new boolean[blockCount];
        for (int i = 0; i < blockCount; i++) {
            isSwa[i] = (swaPattern != null && i < swaPattern.length) ? swaPattern[i] : (i % 6 != 5);
        }

        // Shared KV layer mapping (mirrors Gemma4InferenceEngine).
        this.kvSourceLayer = new int[blockCount];
        int firstShared = blockCount - sharedKvLayers;
        for (int i = 0; i < blockCount; i++) {
            if (i < firstShared) kvSourceLayer[i] = i;
            else kvSourceLayer[i] = isSwa[i] ? (firstShared - 2) : (firstShared - 1);
        }

        // Dual RoPE: SWA uses theta=10K (or ropeFreqBaseSwa), full uses theta=1M with freq factors.
        float swaTheta = config.ropeFreqBaseSwa() > 0 ? config.ropeFreqBaseSwa() : 10000f;
        float fullTheta = config.ropeFreqBase();
        RoPE ropeSwa = new RoPE(headSizeSwa, headSizeSwa, maxSeqLen, swaTheta, config.ropeType(), null);
        RoPE ropeFull = new RoPE(headSizeFull, headSizeFull, maxSeqLen, fullTheta, config.ropeType(), ropeFreqFactors);
        this.ropeTypeSwa = ropeSwa.getRopeType();
        this.halfRopeSwa = ropeSwa.getRopeDimCount() / 2;
        this.ropeTypeFull = ropeFull.getRopeType();
        this.halfRopeFull = ropeFull.getRopeDimCount() / 2;

        long fb = Float.BYTES;

        // Kernels
        rmsnormFunc     = cudaContext.compileKernel("kernels/cuda/rmsnorm.cu", "rmsnorm_fused");
        perHeadNormFunc = cudaContext.compileKernel("kernels/cuda/rmsnorm_per_head.cu", "rmsnorm_per_head");
        ropeFunc        = cudaContext.compileKernel("kernels/cuda/rope.cu", "rope_apply");
        kvUpdateFunc    = cudaContext.compileKernel("kernels/cuda/attention.cu", "kv_cache_update");
        attnFunc        = cudaContext.compileKernel("kernels/cuda/attention.cu", "attention_full");
        geluFunc        = cudaContext.compileKernel("kernels/cuda/gelu.cu", "gelu");
        elemMulFunc     = cudaContext.compileKernel("kernels/cuda/elementwise_mul.cu", "elementwise_mul");
        accumFunc       = cudaContext.compileKernel("kernels/cuda/accumulate.cu", "accumulate");
        scaleFunc       = cudaContext.compileKernel("kernels/cuda/scale_inplace.cu", "scale_inplace");

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

        gpuNorm    = bufferManager.createBuffer(dim * fb);
        gpuQ       = bufferManager.createBuffer((long) maxQDim * fb);
        gpuK       = bufferManager.createBuffer((long) maxKvDim * fb);
        gpuV       = bufferManager.createBuffer((long) maxKvDim * fb);
        gpuAttnOut = bufferManager.createBuffer((long) maxQDim * fb);
        gpuGate    = bufferManager.createBuffer((long) ffnDim * fb);
        gpuUp      = bufferManager.createBuffer((long) ffnDim * fb);
        gpuBx      = bufferManager.createBuffer(dim * fb);      // scratch (Wo out / Down out)
        gpuAttnRes = bufferManager.createBuffer(dim * fb);      // attn residual (attn_out = wo + x)
        gpuPleGate = pleDim > 0 ? bufferManager.createBuffer((long) pleDim * fb) : 0;
        gpuPleOut  = pleDim > 0 ? bufferManager.createBuffer(dim * fb) : 0;
        gpuPleCombined = pleDim > 0 ? bufferManager.createBuffer((long) pleDim * blockCount * fb) : 0;
        hostPle = pleDim > 0 ? arena.allocate(ValueLayout.JAVA_FLOAT, (long) pleDim * blockCount) : null;

        // All-ones buffer for V-norm (rmsnorm_per_head with no learnable scale).
        float[] ones = new float[maxHeadSize];
        java.util.Arrays.fill(ones, 1.0f);
        gpuOnes = uploadFloatArray(ones);

        // Q8_1 input scratch: 40 bytes per 32-element block; sized for the largest matmul input
        // (input cols are dim or ffnDim; pleDim/blockCount inputs are <= dim).
        int maxIn = Math.max(dim, ffnDim);
        gpuQ8In = useDp4a ? bufferManager.createBuffer((long) ((maxIn + 31) / 32) * 40) : 0;

        gpuCosTableSwa  = uploadFloatArray(ropeSwa.getCosTable());
        gpuSinTableSwa  = uploadFloatArray(ropeSwa.getSinTable());
        gpuCosTableFull = uploadFloatArray(ropeFull.getCosTable());
        gpuSinTableFull = uploadFloatArray(ropeFull.getSinTable());

        // Per-layer weights + KV cache
        gpuAttnNorm     = new long[blockCount];
        gpuFfnNorm      = new long[blockCount];
        gpuQNorm        = new long[blockCount];
        gpuKNorm        = new long[blockCount];
        gpuPostAttnNorm = new long[blockCount];
        gpuPostFfnNorm  = new long[blockCount];
        gpuPlePostNorm  = new long[blockCount];
        gpuKeyCache     = new long[blockCount];
        gpuValueCache   = new long[blockCount];
        for (int i = 0; i < blockCount; i++) {
            TransformerLayerWeights lw = weights.layers()[i];
            int hs = isSwa[i] ? headSizeSwa : headSizeFull;
            int kvDim = headCountKV * hs;
            gpuAttnNorm[i] = uploadNormWeights(lw.attnNorm(), dim);
            gpuFfnNorm[i]  = uploadNormWeights(lw.ffnNorm(), dim);
            if (lw.qNorm() != null) gpuQNorm[i] = uploadNormWeights(lw.qNorm(), hs);
            if (lw.kNorm() != null) gpuKNorm[i] = uploadNormWeights(lw.kNorm(), hs);
            if (lw.postAttnNorm() != null) gpuPostAttnNorm[i] = uploadNormWeights(lw.postAttnNorm(), dim);
            if (lw.postFfnNorm() != null) gpuPostFfnNorm[i] = uploadNormWeights(lw.postFfnNorm(), dim);
            if (plePostNorm != null && plePostNorm[i] != null) gpuPlePostNorm[i] = uploadFloatArray(plePostNorm[i]);
            // Allocate KV cache only for own-KV layers (shared layers reuse an earlier layer's).
            if (i < firstShared) {
                long kvBytes = (long) maxSeqLen * kvDim * fb;
                gpuKeyCache[i]   = bufferManager.createBuffer(kvBytes);
                gpuValueCache[i] = bufferManager.createBuffer(kvBytes);
            }
        }

        // Output projection (final logits) — must be GPU-resident
        gpuLogits = bufferManager.createBuffer((long) vocabSize * fb);
        gpuLogitsBytes = (long) vocabSize * fb;
        hostLogits = arena.allocate(ValueLayout.JAVA_FLOAT, vocabSize);
        gpuOutputNorm = uploadNormWeights(weights.outputNorm(), dim);

        // Param buffers
        matmulPB = new PB(arena, 6);
        quantPB = new PB(arena, 3);
        dp4aPB = new PB(arena, 6);

        normPB = new PB(arena, 5);
        normPB.setLong(0, gpuNorm); normPB.setLong(1, gpuX); normPB.setInt(3, dim); normPB.setFloat(4, normEps);

        // per-head norm: [vec, weights, headSize, eps] — headSize set per layer
        perHeadPB = new PB(arena, 4);
        perHeadPB.setFloat(3, normEps);

        // rope: [vec, cosTable, sinTable, nHeads, headSize, halfRope, tokenParams, ropeType]
        // — cos/sin/headSize/halfRope/ropeType/nHeads set per layer
        ropePB = new PB(arena, 8);
        ropePB.setLong(6, gpuTokenParams);

        // kv_cache_update: [keyCache, valueCache, k, v, kvDim, tokenParams] — kvDim/caches per layer
        kvPB = new PB(arena, 6);
        kvPB.setLong(2, gpuK); kvPB.setLong(3, gpuV); kvPB.setLong(5, gpuTokenParams);

        // attention_full: [output, q, keyCache, valueCache, headCount, headCountKV, headSize, kvDim,
        //                  tokenParams, slidingWindow] — headSize/kvDim/caches/window per layer
        attnPB = new PB(arena, 10);
        attnPB.setLong(0, gpuAttnOut); attnPB.setLong(1, gpuQ);
        attnPB.setInt(4, headCount); attnPB.setInt(5, headCountKV);
        attnPB.setLong(8, gpuTokenParams);

        // gelu: [x, size] — x/size set per use
        geluPB = new PB(arena, 2);

        // elementwise_mul: [a, b, size] — a/b/size set per use
        elemMulPB = new PB(arena, 3);

        // accumulate: [y, x, size] — y/x/size set per use
        accumPB = new PB(arena, 3);

        // scale_inplace: [x, scale, size] — x/scale/size set per use
        scalePB = new PB(arena, 3);

        int normNumWarps = (int) (blockSize / 32);
        this.normSharedMem = (normNumWarps + 1) * Float.BYTES;
        this.perHeadBlockDim = (int) Math.min(Math.max(32, ((maxHeadSize + 31) / 32) * 32), blockSize);
        this.perHeadSharedMem = ((perHeadBlockDim / 32) + 1) * Float.BYTES;
    }

    public static boolean isSupported(ModelConfig config, ModelWeights weights,
                                      FloatTensor[] pleInpGate, FloatTensor[] pleProj) {
        if (weights.layers().length == 0) return false;
        if (!(weights.output() instanceof CudaFloatTensor)) return false;
        for (TransformerLayerWeights lw : weights.layers()) {
            FloatTensor[] mm = { lw.wq(), lw.wk(), lw.wv(), lw.wo(), lw.wGate(), lw.wUp(), lw.wDown() };
            for (FloatTensor t : mm) if (!(t instanceof CudaFloatTensor)) return false;
        }
        if (pleInpGate != null) for (FloatTensor t : pleInpGate) if (!(t instanceof CudaFloatTensor)) return false;
        if (pleProj != null) for (FloatTensor t : pleProj) if (!(t instanceof CudaFloatTensor)) return false;
        return true;
    }

    public int getGpuLayerCount() { return blockCount; }

    /** Upload the precomputed [pleDim*blockCount] PLE combined vector for this token. */
    public void uploadPleCombined(float[] pleCombined) {
        if (pleDim <= 0 || gpuPleCombined == 0) return;
        int n = pleDim * blockCount;
        MemorySegment.copy(pleCombined, 0, hostPle, ValueLayout.JAVA_FLOAT, 0, n);
        cudaContext.writeBuffer(gpuPleCombined, hostPle, (long) n * Float.BYTES);
    }

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
        TransformerLayerWeights lw = weights.layers()[li];
        long fb = Float.BYTES;

        boolean swa = isSwa[li];
        int hs = swa ? headSizeSwa : headSizeFull;
        int qDim = headCount * hs;
        int kvDim = headCountKV * hs;
        boolean hasOwnKv = li < blockCount - sharedKvLayers;
        int kvLayer = kvSourceLayer[li];
        long cosTable = swa ? gpuCosTableSwa : gpuCosTableFull;
        long sinTable = swa ? gpuSinTableSwa : gpuSinTableFull;
        int halfRope = swa ? halfRopeSwa : halfRopeFull;
        int ropeType = swa ? ropeTypeSwa : ropeTypeFull;

        int qDimGrid  = (int) ((qDim + blockSize - 1) / blockSize);
        int kvDimGrid = (int) ((kvDim + blockSize - 1) / blockSize);
        int dimGrid   = (int) ((dim + blockSize - 1) / blockSize);
        int ffnGrid   = (int) ((ffnDim + blockSize - 1) / blockSize);
        int ropeQGrid = (int) ((headCount * halfRope + blockSize - 1) / blockSize);
        int ropeKGrid = (int) ((headCountKV * halfRope + blockSize - 1) / blockSize);

        // === 1. Pre-attention RMSNorm: gpuX -> gpuNorm ===
        normPB.setLong(0, gpuNorm); normPB.setLong(1, gpuX); normPB.setLong(2, gpuAttnNorm[li]);
        normPB.setInt(3, dim);
        launch(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB);

        // === 2. Q projection ===
        matmul((CudaFloatTensor) lw.wq(), gpuNorm, gpuQ, qDim, dim);
        // QK-norm on Q (per-head, with learnable scale)
        if (gpuQNorm[li] != 0) {
            perHeadPB.setLong(0, gpuQ); perHeadPB.setLong(1, gpuQNorm[li]); perHeadPB.setInt(2, hs);
            launch(perHeadNormFunc, headCount, perHeadBlockDim, perHeadSharedMem, perHeadPB);
        }
        // RoPE on Q
        ropePB.setLong(0, gpuQ); ropePB.setLong(1, cosTable); ropePB.setLong(2, sinTable);
        ropePB.setInt(3, headCount); ropePB.setInt(4, hs); ropePB.setInt(5, halfRope); ropePB.setInt(7, ropeType);
        launch(ropeFunc, ropeQGrid, (int) blockSize, 0, ropePB);
        // Attention scale = 1.0: cancel the kernel's 1/sqrt(hs) by scaling Q by sqrt(hs).
        scalePB.setLong(0, gpuQ); scalePB.setFloat(1, (float) Math.sqrt((double) hs)); scalePB.setInt(2, qDim);
        launch(scaleFunc, qDimGrid, (int) blockSize, 0, scalePB);

        // === 3. K/V projection (own-KV layers only) ===
        if (hasOwnKv) {
            matmul((CudaFloatTensor) lw.wk(), gpuNorm, gpuK, kvDim, dim);
            matmul((CudaFloatTensor) lw.wv(), gpuNorm, gpuV, kvDim, dim);
            // QK-norm on K (per-head, with learnable scale)
            if (gpuKNorm[li] != 0) {
                perHeadPB.setLong(0, gpuK); perHeadPB.setLong(1, gpuKNorm[li]); perHeadPB.setInt(2, hs);
                launch(perHeadNormFunc, headCountKV, perHeadBlockDim, perHeadSharedMem, perHeadPB);
            }
            // V-norm: RMS without learnable scale (all-ones weights), per head
            perHeadPB.setLong(0, gpuV); perHeadPB.setLong(1, gpuOnes); perHeadPB.setInt(2, hs);
            launch(perHeadNormFunc, headCountKV, perHeadBlockDim, perHeadSharedMem, perHeadPB);
            // RoPE on K
            ropePB.setLong(0, gpuK); ropePB.setInt(3, headCountKV);
            launch(ropeFunc, ropeKGrid, (int) blockSize, 0, ropePB);
            // KV cache update into THIS layer's cache
            kvPB.setLong(0, gpuKeyCache[li]); kvPB.setLong(1, gpuValueCache[li]); kvPB.setInt(4, kvDim);
            launch(kvUpdateFunc, kvDimGrid, (int) blockSize, 0, kvPB);
        }

        // === 4. Attention: reads source layer's KV cache (same hs/kvDim as this layer) ===
        attnPB.setLong(2, gpuKeyCache[kvLayer]); attnPB.setLong(3, gpuValueCache[kvLayer]);
        attnPB.setInt(6, hs); attnPB.setInt(7, kvDim);
        attnPB.setInt(9, (swa && slidingWindow > 0) ? slidingWindow : 0);
        int attnSM = (position + 1 + 32) * Float.BYTES;
        launch(attnFunc, headCount, Math.min(256, (int) blockSize), attnSM, attnPB);

        // === 5. Wo projection: gpuAttnOut -> gpuBx ===
        matmul((CudaFloatTensor) lw.wo(), gpuAttnOut, gpuBx, dim, qDim);

        // === 6. Post-attention norm (in place on gpuBx, if present) ===
        if (gpuPostAttnNorm[li] != 0) {
            normPB.setLong(0, gpuBx); normPB.setLong(1, gpuBx); normPB.setLong(2, gpuPostAttnNorm[li]); normPB.setInt(3, dim);
            launch(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB);
        }

        // === 7. Attention residual: gpuAttnRes = gpuBx + gpuX ===
        // (copy gpuX into gpuAttnRes, then accumulate gpuBx)
        cudaContext.copyBufferDtoD(gpuAttnRes, gpuX, (long) dim * fb);
        accumPB.setLong(0, gpuAttnRes); accumPB.setLong(1, gpuBx); accumPB.setInt(2, dim);
        launch(accumFunc, dimGrid, (int) blockSize, 0, accumPB);

        // === 8. Pre-FFN RMSNorm: gpuAttnRes -> gpuNorm ===
        normPB.setLong(0, gpuNorm); normPB.setLong(1, gpuAttnRes); normPB.setLong(2, gpuFfnNorm[li]); normPB.setInt(3, dim);
        launch(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB);

        // === 9. GeGLU FFN: gate=wGate@norm, up=wUp@norm, gelu(gate), gate*=up, down=wDown@gate ===
        matmul((CudaFloatTensor) lw.wGate(), gpuNorm, gpuGate, ffnDim, dim);
        matmul((CudaFloatTensor) lw.wUp(), gpuNorm, gpuUp, ffnDim, dim);
        geluPB.setLong(0, gpuGate); geluPB.setInt(1, ffnDim);
        launch(geluFunc, ffnGrid, (int) blockSize, 0, geluPB);
        elemMulPB.setLong(0, gpuGate); elemMulPB.setLong(1, gpuUp); elemMulPB.setInt(2, ffnDim);
        launch(elemMulFunc, ffnGrid, (int) blockSize, 0, elemMulPB);
        matmul((CudaFloatTensor) lw.wDown(), gpuGate, gpuBx, dim, ffnDim);

        // === 10. Post-FFN norm (in place on gpuBx, if present) ===
        if (gpuPostFfnNorm[li] != 0) {
            normPB.setLong(0, gpuBx); normPB.setLong(1, gpuBx); normPB.setLong(2, gpuPostFfnNorm[li]); normPB.setInt(3, dim);
            launch(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB);
        }

        // === 11. FFN residual: gpuX = gpuBx + gpuAttnRes ===
        // (copy gpuAttnRes into gpuX, then accumulate gpuBx)
        cudaContext.copyBufferDtoD(gpuX, gpuAttnRes, (long) dim * fb);
        accumPB.setLong(0, gpuX); accumPB.setLong(1, gpuBx); accumPB.setInt(2, dim);
        launch(accumFunc, dimGrid, (int) blockSize, 0, accumPB);

        // === 12. PLE injection ===
        if (pleDim > 0 && gpuPleCombined != 0 && pleInpGate != null
                && li < pleInpGate.length && pleInpGate[li] != null) {
            int pleGrid = (int) ((pleDim + blockSize - 1) / blockSize);
            // pleGate = inp_gate @ x  ([dim -> pleDim])
            matmul((CudaFloatTensor) pleInpGate[li], gpuX, gpuPleGate, pleDim, dim);
            // gelu(pleGate)
            geluPB.setLong(0, gpuPleGate); geluPB.setInt(1, pleDim);
            launch(geluFunc, pleGrid, (int) blockSize, 0, geluPB);
            // pleGate *= pleCombined[layer*pleDim ..]
            elemMulPB.setLong(0, gpuPleGate);
            elemMulPB.setLong(1, gpuPleCombined + (long) li * pleDim * fb);
            elemMulPB.setInt(2, pleDim);
            launch(elemMulFunc, pleGrid, (int) blockSize, 0, elemMulPB);
            // pleOut = proj @ pleGate  ([pleDim -> dim])
            matmul((CudaFloatTensor) pleProj[li], gpuPleGate, gpuPleOut, dim, pleDim);
            // post-PLE norm (in place, if present)
            if (gpuPlePostNorm[li] != 0) {
                normPB.setLong(0, gpuPleOut); normPB.setLong(1, gpuPleOut);
                normPB.setLong(2, gpuPlePostNorm[li]); normPB.setInt(3, dim);
                launch(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB);
            }
            // x += pleOut
            accumPB.setLong(0, gpuX); accumPB.setLong(1, gpuPleOut); accumPB.setInt(2, dim);
            launch(accumFunc, dimGrid, (int) blockSize, 0, accumPB);
        }

        // === 13. Layer output scale: x *= out_scale ===
        if (layerOutputScale != null && li < layerOutputScale.length) {
            float scale = layerOutputScale[li];
            if (scale != 1.0f && scale != 0f) {
                scalePB.setLong(0, gpuX); scalePB.setFloat(1, scale); scalePB.setInt(2, dim);
                launch(scaleFunc, dimGrid, (int) blockSize, 0, scalePB);
            }
        }
    }

    /** Final RMSNorm + output projection. Returns RAW logits (logit soft-cap applied by caller). */
    public boolean forwardFinalLogits(float[] logits) {
        normPB.setLong(0, gpuNorm); normPB.setLong(1, gpuX); normPB.setLong(2, gpuOutputNorm); normPB.setInt(3, dim);
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
        if (err != CudaBindings.CUDA_SUCCESS) throw new RuntimeException("Gemma4 CUDA error: " + err);
    }

    private long uploadNormWeights(FloatTensor t, int size) {
        float[] w = new float[size]; for (int i = 0; i < size; i++) w[i] = t.getFloat(i);
        return bufferManager.uploadNormWeights(w);
    }

    private float[] tensorToFloats(FloatTensor t) {
        int n = (int) t.size();
        float[] w = new float[n]; for (int i = 0; i < n; i++) w[i] = t.getFloat(i);
        return w;
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
