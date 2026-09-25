package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.model.ModelWeights;
import it.denzosoft.llmplayer.model.TransformerLayerWeights;
import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Inference engine for Gemma 4 architecture.
 * Features:
 * - Per-Layer Embeddings (PLE): parallel token-specific information injected at each layer
 * - Shared KV cache: top N layers reuse KV from earlier layers (no K/V projections)
 * - Dual RoPE: SWA layers use theta=10K, full attention layers use theta=1M
 * - Pre+post attention/FFN norms (Gemma-style)
 * - QK-norm on Q and K (K-norm has no learnable scale)
 * - V-norm without learnable scale
 * - GeGLU activation (GELU instead of SiLU)
 * - Layer output scaling
 * - Logit soft-capping
 */
public class Gemma4InferenceEngine {

    private final ModelConfig config;
    private final ModelWeights weights;
    private final int maxSeqLen;

    // Cached config
    private final int dim;
    private final int ffnDim;
    private final int vocabSize;
    private final int blockCount;
    private final float normEps;
    private final int headCount;
    private final int headCountKV;
    private final int headSizeSwa;   // 256 for SWA layers
    private final int headSizeFull;  // 512 for full attention layers
    private final int maxQDim;       // max(headCount * headSizeSwa, headCount * headSizeFull)
    private final int maxKvDim;      // max(headCountKV * headSizeSwa, headCountKV * headSizeFull)
    private final int kvMul;
    private final int slidingWindow;
    private final int sharedKvLayers;
    private final int pleDim; // per-layer embedding dim (256 for E4B)
    private final float embeddingScale;
    private final float finalLogitSoftCap;

    // RoPE (dual: SWA and full attention)
    private final RoPE ropeSwa;   // for sliding window layers (theta=10K)
    private final RoPE ropeFull;  // for full attention layers (theta=1M)
    private final boolean[] slidingWindowPattern; // true=SWA, false=full

    // PLE global weights
    private final FloatTensor pleTokenEmbd;    // [pleDim*blockCount, vocabSize]
    private final FloatTensor pleModelProj;    // [dim, pleDim*blockCount]
    private final float[] pleProjNormWeights;  // [pleDim]

    // PLE per-layer weights
    private final FloatTensor[] pleInpGate;    // [dim, pleDim] per layer
    private final FloatTensor[] pleProj;       // [pleDim, dim] per layer
    private final float[][] plePostNorm;       // [dim] per layer
    private final float[] layerOutputScale;    // scalar per layer

    // H10: Gemma 3n AltUp + Laurel — null if model is plain PLE without altup machinery
    private final it.denzosoft.llmplayer.model.Gemma3nWeights g3n;

    // Cached norm weights
    private final float[][] attnNormCache;
    private final float[][] ffnNormCache;
    private final float[][] postAttnNormCache;
    private final float[][] postFfnNormCache;
    private final float[][] qNormCache;
    private final float[][] kNormCache;
    private final float[] outputNormCache;

    // Shared KV mapping: for layer L, kvSourceLayer[L] = which layer's KV cache to use
    private final int[] kvSourceLayer;

    // Inference state
    private Gemma4State state;

    // RoPE freq factors (kept for the GPU forward pass to rebuild its own cos/sin tables)
    private final float[] ropeFreqFactorsArr;

    // GPU-resident forward pass (Gemma 4 PLE-only; null on CPU / Gemma 3n AltUp / unsupported)
    private AutoCloseable gpuForwardPass;
    private int gpuLayerCount;
    private Method gpuUploadX, gpuUploadPle, gpuForwardLayer, gpuForwardFinalLogits, gpuDownloadX;

    public Gemma4InferenceEngine(ModelConfig config, ModelWeights weights, int maxSeqLen,
                                  FloatTensor pleTokenEmbd, FloatTensor pleModelProj,
                                  float[] pleProjNormWeights,
                                  FloatTensor[] pleInpGate, FloatTensor[] pleProj,
                                  float[][] plePostNorm, float[] layerOutputScale,
                                  float[] ropeFreqFactors) {
        this(config, weights, maxSeqLen, pleTokenEmbd, pleModelProj, pleProjNormWeights,
             pleInpGate, pleProj, plePostNorm, layerOutputScale, ropeFreqFactors, null);
    }

    public Gemma4InferenceEngine(ModelConfig config, ModelWeights weights, int maxSeqLen,
                                  FloatTensor pleTokenEmbd, FloatTensor pleModelProj,
                                  float[] pleProjNormWeights,
                                  FloatTensor[] pleInpGate, FloatTensor[] pleProj,
                                  float[][] plePostNorm, float[] layerOutputScale,
                                  float[] ropeFreqFactors,
                                  it.denzosoft.llmplayer.model.Gemma3nWeights g3n) {
        this.g3n = g3n;
        this.config = config;
        this.weights = weights;
        this.maxSeqLen = maxSeqLen;

        this.dim = config.embeddingLength();
        this.ffnDim = config.intermediateSize();
        this.vocabSize = config.vocabSize();
        this.blockCount = config.blockCount();
        this.normEps = config.normEps();
        this.headCount = config.headCount();
        this.headCountKV = config.headCountKV();
        // Gemma 4: SWA layers use headSize=key_length_swa, full layers use headSize=key_length
        this.headSizeSwa = config.headSize();  // already clamped to key_length_swa in ModelConfig
        this.headSizeFull = config.keyLength() > 0 ? config.keyLength() : headSizeSwa;
        this.maxQDim = headCount * Math.max(headSizeSwa, headSizeFull);
        this.maxKvDim = headCountKV * Math.max(headSizeSwa, headSizeFull);
        this.kvMul = headCount / headCountKV;
        this.slidingWindow = config.slidingWindow();
        this.sharedKvLayers = config.sharedKvLayers();
        this.pleDim = config.embeddingLengthPerLayer();
        this.embeddingScale = config.embeddingScale();
        this.finalLogitSoftCap = config.finalLogitSoftCap();
        this.slidingWindowPattern = config.slidingWindowPattern();

        this.pleTokenEmbd = pleTokenEmbd;
        this.pleModelProj = pleModelProj;
        this.pleProjNormWeights = pleProjNormWeights;
        this.pleInpGate = pleInpGate;
        this.pleProj = pleProj;
        this.plePostNorm = plePostNorm;
        this.layerOutputScale = layerOutputScale;

        // Dual RoPE: SWA uses theta=10K, full uses theta=1M with freq factors
        // RoPE dim = headSize for each layer type
        float swaTheta = config.ropeFreqBaseSwa() > 0 ? config.ropeFreqBaseSwa() : 10000f;
        float fullTheta = config.ropeFreqBase();
        this.ropeSwa = new RoPE(headSizeSwa, headSizeSwa, maxSeqLen, swaTheta, config.ropeType(), null);
        this.ropeFull = new RoPE(headSizeFull, headSizeFull, maxSeqLen, fullTheta, config.ropeType(), ropeFreqFactors);
        this.ropeFreqFactorsArr = ropeFreqFactors;

        // Cache norm weights for fast access
        attnNormCache = new float[blockCount][];
        ffnNormCache = new float[blockCount][];
        postAttnNormCache = new float[blockCount][];
        postFfnNormCache = new float[blockCount][];
        qNormCache = new float[blockCount][];
        kNormCache = new float[blockCount][];
        for (int i = 0; i < blockCount; i++) {
            TransformerLayerWeights lw = weights.layers()[i];
            attnNormCache[i] = RMSNorm.cacheWeights(lw.attnNorm(), dim);
            ffnNormCache[i] = RMSNorm.cacheWeights(lw.ffnNorm(), dim);
            if (lw.postAttnNorm() != null) postAttnNormCache[i] = RMSNorm.cacheWeights(lw.postAttnNorm(), dim);
            if (lw.postFfnNorm() != null) postFfnNormCache[i] = RMSNorm.cacheWeights(lw.postFfnNorm(), dim);
            // QK-norm size depends on layer type (SWA=256, full=512)
            int layerHeadSize = isSwaLayer(i) ? headSizeSwa : headSizeFull;
            if (lw.qNorm() != null) {
                qNormCache[i] = RMSNorm.cacheWeights(lw.qNorm(), layerHeadSize);
            }
            if (lw.kNorm() != null) {
                kNormCache[i] = RMSNorm.cacheWeights(lw.kNorm(), layerHeadSize);
                // Gemma 3n: K-norm weights in GGUF are raw w (NOT pre-computed 1+w).
                // Gemma 4: weights are stored as final values (Q≈0.98, K≈0.13). Don't adjust.
                if (config.architecture() == it.denzosoft.llmplayer.model.ModelArchitecture.GEMMA3N) {
                    for (int j = 0; j < layerHeadSize; j++) kNormCache[i][j] += 1.0f;
                }
            }
        }
        outputNormCache = RMSNorm.cacheWeights(weights.outputNorm(), dim);

        // Build shared KV layer mapping: ALL shared layers reuse the LAST non-shared layer of same type
        // SWA shared layers → last SWA before shared boundary (firstShared - 2)
        // Full attention shared layers → last full before shared boundary (firstShared - 1)
        kvSourceLayer = new int[blockCount];
        int firstShared = blockCount - sharedKvLayers;
        for (int i = 0; i < blockCount; i++) {
            if (i < firstShared) {
                kvSourceLayer[i] = i; // own KV cache
            } else {
                kvSourceLayer[i] = isSwaLayer(i) ? (firstShared - 2) : (firstShared - 1);
            }
        }
    }

    private boolean isSwaLayer(int layer) {
        if (slidingWindowPattern != null && layer < slidingWindowPattern.length) {
            return slidingWindowPattern[layer];
        }
        return layer % 6 != 5; // fallback: Gemma 3 pattern
    }

    private int layerQDim(int layer) {
        return headCount * (isSwaLayer(layer) ? headSizeSwa : headSizeFull);
    }

    /** Per-layer K/V width: see {@link #attentionCore} for why the KV head count is per layer. */
    private int layerKvDim(int layer) {
        return config.layerKvHeads(layer) * (isSwaLayer(layer) ? headSizeSwa : headSizeFull);
    }

    /** Try to enable the GPU-resident forward pass (Gemma 4 PLE-only; not Gemma 3n AltUp). */
    public void tryInitGpuForwardPass(Object bufferManager) {
        if (g3n != null && g3n.isFullyLoaded()) return; // Gemma 3n AltUp stays on CPU
        try {
            Class<?> cls = Class.forName("it.denzosoft.llmplayer.inference.Gemma4CudaForwardPass");
            Method isSup = cls.getMethod("isSupported", ModelConfig.class, ModelWeights.class,
                FloatTensor[].class, FloatTensor[].class);
            if (!(Boolean) isSup.invoke(null, config, weights, pleInpGate, pleProj)) return;
            Object fwd = cls.getConstructor(ModelConfig.class, ModelWeights.class,
                    bufferManager.getClass(), int.class, FloatTensor[].class, FloatTensor[].class,
                    float[][].class, float[].class, float[].class, int.class)
                .newInstance(config, weights, bufferManager, maxSeqLen,
                    pleInpGate, pleProj, plePostNorm, layerOutputScale, ropeFreqFactorsArr, pleDim);
            gpuUploadX = cls.getMethod("uploadXAndUpdateParams", float[].class, int.class);
            gpuUploadPle = cls.getMethod("uploadPleCombined", float[].class);
            gpuForwardLayer = cls.getMethod("forwardLayer", int.class, int.class);
            gpuDownloadX = cls.getMethod("downloadX", float[].class);
            gpuForwardFinalLogits = cls.getMethod("forwardFinalLogits", float[].class);
            gpuLayerCount = (Integer) cls.getMethod("getGpuLayerCount").invoke(fwd);
            gpuForwardPass = (AutoCloseable) fwd;
            System.err.println("Gemma 4 CUDA forward pass: enabled (" + gpuLayerCount + "/" + blockCount + " layers)");
        } catch (Throwable e) {
            System.err.println("Gemma 4 CUDA forward pass: unavailable — " + e.getMessage());
        }
    }

    private float[] forwardGpu(Gemma4State state, int position, boolean computeLogits, boolean doPle) throws Exception {
        gpuUploadX.invoke(gpuForwardPass, state.x, position);
        if (doPle) gpuUploadPle.invoke(gpuForwardPass, state.pleCombined);
        for (int i = 0; i < gpuLayerCount; i++) gpuForwardLayer.invoke(gpuForwardPass, i, position);

        // Partial offload (e.g. the 12B at 7 GB on a 6 GB card): the first gpuLayerCount layers ran
        // on the GPU; finish the remaining layers on the CPU engine. The GPU layers keep their KV
        // in GPU buffers and the CPU layers in the CPU KVCache — disjoint sets, each self-consistent
        // across tokens (Gemma 4 dense uses no shared KV).
        if (gpuLayerCount < blockCount) {
            gpuDownloadX.invoke(gpuForwardPass, state.x);   // activation after the GPU layers
            for (int layer = gpuLayerCount; layer < blockCount; layer++) {
                forwardLayer(state, layer, position, doPle);
            }
            if (!computeLogits) return null;
            // The final output projection weight is GPU-resident; re-upload the CPU-finished
            // activation so forwardFinalLogits can project it on the GPU.
            gpuUploadX.invoke(gpuForwardPass, state.x, position);
        }

        if (!computeLogits) return null;
        gpuForwardFinalLogits.invoke(gpuForwardPass, state.logits);
        if (finalLogitSoftCap > 0f) {
            for (int i = 0; i < vocabSize; i++) {
                state.logits[i] = finalLogitSoftCap * (float) Math.tanh(state.logits[i] / finalLogitSoftCap);
            }
        }
        return state.logits;
    }

    public Gemma4State createState() {
        // Use max dimensions for buffers (full attention layers have larger Q/K/V).
        // If g3n is loaded, also allocate AltUp + Laurel buffers (4-stream activations).
        if (g3n != null && g3n.isFullyLoaded()) {
            // laurel rank is the inner dim of laurel_l (shape [n_embd, laurel_rank])
            int laurelRank = (int) g3n.laurelL[0].size() / dim;
            return new Gemma4State(config, maxSeqLen, pleDim, maxQDim, maxKvDim,
                g3n.nAltup, laurelRank);
        }
        return new Gemma4State(config, maxSeqLen, pleDim, maxQDim, maxKvDim);
    }

    /**
     * Forward pass for a single token.
     */
    public float[] forward(int token, int position, boolean computeLogits) {
        if (state == null) state = createState();

        // 1. Token embedding lookup + scaling
        for (int i = 0; i < dim; i++) {
            state.x[i] = weights.tokenEmbedding().getFloat((long) token * dim + i);
        }
        if (embeddingScale > 0f) {
            for (int i = 0; i < dim; i++) state.x[i] *= embeddingScale;
        }

        // 2. PLE pre-computation
        boolean hasPle = pleTokenEmbd != null && pleDim > 0;
        if (hasPle) computePleInput(token, state);
        boolean doPle = hasPle && !Boolean.getBoolean("gemma4.nople"); // can be toggled for debug

        // H10: Gemma 3n AltUp + Laurel forward path
        boolean useAltup = g3n != null && g3n.isFullyLoaded() && state.altupStreams != null;
        if (useAltup) {
            // 3a. Initialize 4 streams from x via altup_proj
            altupInit(state);
            // 3b. Per layer: predict → run active stream → correct → first_pred inject
            for (int layer = 0; layer < blockCount; layer++) {
                forwardLayerAltup(state, layer, position, doPle);
            }
            // 3c. Merge streams back to single x via altup_unembd_proj + averaging
            altupMerge(state);
        } else if (gpuForwardPass != null) {
            // 3b. GPU-resident PLE-only path (falls back to CPU on any failure)
            try {
                return forwardGpu(state, position, computeLogits, doPle);
            } catch (Exception e) {
                System.err.println("Gemma 4 GPU forward failed: " + e.getMessage());
                gpuForwardPass = null;
                for (int layer = 0; layer < blockCount; layer++) forwardLayer(state, layer, position, doPle);
            }
        } else {
            // 3. Forward through all layers (PLE-only path, no AltUp)
            for (int layer = 0; layer < blockCount; layer++) {
                forwardLayer(state, layer, position, doPle);
            }
        }

        if (!computeLogits) return null;
        return finishLogits(state);
    }

    /** Final norm, output projection and logit soft-capping of {@code state.x}. */
    private float[] finishLogits(Gemma4State state) {
        // 4. Final RMSNorm
        VectorOpsFactory.get().rmsnorm(state.xb, state.x, outputNormCache, dim, normEps);

        // 5. Output projection
        Arrays.fill(state.logits, 0);
        weights.output().matmulParallel(state.xb, state.logits, vocabSize, dim);

        // 6. Logit soft-capping: logits = softCap * tanh(logits / softCap)
        if (finalLogitSoftCap > 0f) {
            for (int i = 0; i < vocabSize; i++) {
                state.logits[i] = finalLogitSoftCap * (float) Math.tanh(state.logits[i] / finalLogitSoftCap);
            }
        }

        return state.logits;
    }

    // ==================== Batched prefill ====================

    private static final int PREFILL_BATCH = Integer.getInteger("prefill.batch", 64);
    private static final boolean PREFILL_BATCHED =
        !"false".equalsIgnoreCase(System.getProperty("prefill.batched", "true"));

    // Indices into Gemma4State.prefillBuffers
    private static final int B_X = 0, B_XB = 1, B_ATTN = 2, B_Q = 3, B_K = 4, B_V = 5, B_OUT = 6,
        B_HB = 7, B_HB2 = 8, B_PLE = 9, B_PLE_GATE = 10;

    /**
     * Prefill {@code tokens[fromPos..toPos)} at positions {@code fromPos..toPos-1} and return the
     * logits of the last one.
     *
     * <p>On the CPU path without AltUp (Gemma 4 E2B/E4B and dense) the prompt goes through each
     * layer in chunks of {@link #PREFILL_BATCH} tokens, and every projection — the PLE model
     * projection, Q, K, V, O, FFN gate, up and down, the PLE gate and PLE projection — runs as one
     * multi-token matmul ({@code FloatTensor.matmulRowsBatch}). The order-dependent part
     * ({@link #attentionCore}: norms, RoPE, KV store, attention) still runs token by token in
     * position order with the same code as the one-token path; only the summation order inside the
     * matmul kernels differs. Gemma 3n (AltUp) and the GPU pass keep the one-token loop. Disable
     * with {@code -Dprefill.batched=false}.
     */
    public float[] forwardPrefill(Gemma4State st, int[] tokens, int fromPos, int toPos) {
        this.state = st;
        int count = toPos - fromPos;
        if (count <= 0) return null;
        boolean useAltup = g3n != null && g3n.isFullyLoaded() && st.altupStreams != null;
        if (count == 1 || !PREFILL_BATCHED || useAltup || gpuForwardPass != null
                || !it.denzosoft.llmplayer.tensor.MatmulPool.enabled()) {
            float[] logits = null;
            for (int i = fromPos; i < toPos; i++) {
                logits = forward(tokens[i], i, i == toPos - 1);
            }
            return logits;
        }
        warmDecodeKernels();
        boolean hasPle = pleTokenEmbd != null && pleDim > 0;
        boolean doPle = hasPle && !Boolean.getBoolean("gemma4.nople");
        int totalPleDim = hasPle ? pleDim * blockCount : 0;
        int cap = Math.max(1, PREFILL_BATCH);
        if (st.prefillBuffers == null) {
            int maxFfn = 0;
            for (int l = 0; l < blockCount; l++) maxFfn = Math.max(maxFfn, config.layerFfnLength(l));
            st.prefillBuffers = new float[][][] {
                new float[cap][dim], new float[cap][dim], new float[cap][dim],
                new float[cap][maxQDim], new float[cap][maxKvDim], new float[cap][maxKvDim],
                new float[cap][maxQDim], new float[cap][maxFfn], new float[cap][maxFfn],
                new float[cap][totalPleDim], new float[cap][pleDim]
            };
        }
        float[][][] b = st.prefillBuffers;
        for (int base = fromPos; base < toPos; base += cap) {
            int n = Math.min(cap, toPos - base);
            for (int t = 0; t < n; t++) {
                int token = tokens[base + t];
                float[] x = b[B_X][t];
                for (int i = 0; i < dim; i++) {
                    x[i] = weights.tokenEmbedding().getFloat((long) token * dim + i);
                }
                if (embeddingScale > 0f) {
                    for (int i = 0; i < dim; i++) x[i] *= embeddingScale;
                }
            }
            if (hasPle) {
                // B_PLE first holds each token's raw projection, then its combined PLE input
                for (int t = 0; t < n; t++) Arrays.fill(b[B_PLE][t], 0f);
                FloatTensor.matmulBatchParallel(pleModelProj, b[B_X], b[B_PLE], n, totalPleDim, dim);
                for (int t = 0; t < n; t++) {
                    combinePle(tokens[base + t], b[B_PLE][t], st.pleEmb, b[B_PLE][t]);
                }
            }
            for (int layer = 0; layer < blockCount; layer++) {
                forwardLayerBatch(st, b, layer, base, n, doPle);
            }
            if (base + n == toPos) System.arraycopy(b[B_X][n - 1], 0, st.x, 0, dim);
        }
        return finishLogits(st);
    }

    /** See {@link FloatTensor#warmUpRows}: batched prefill skips the kernels decode will use. */
    private void warmDecodeKernels() {
        for (int layer = 0; layer < blockCount; layer++) {
            TransformerLayerWeights lw = weights.layers()[layer];
            int qDim = layerQDim(layer), kvDim = layerKvDim(layer), ffn = config.layerFfnLength(layer);
            FloatTensor.warmUpRows(lw.wq(), qDim, dim);
            if (layer < blockCount - sharedKvLayers) {
                FloatTensor.warmUpRows(lw.wk(), kvDim, dim);
                FloatTensor.warmUpRows(lw.wv(), kvDim, dim);
            }
            FloatTensor.warmUpRows(lw.wo(), dim, qDim);
            FloatTensor.warmUpRows(lw.wGate(), ffn, dim);
            FloatTensor.warmUpRows(lw.wUp(), ffn, dim);
            FloatTensor.warmUpRows(lw.wDown(), dim, ffn);
            if (pleInpGate[layer] != null) FloatTensor.warmUpRows(pleInpGate[layer], pleDim, dim);
            if (pleProj[layer] != null) FloatTensor.warmUpRows(pleProj[layer], dim, pleDim);
        }
        if (pleModelProj != null) FloatTensor.warmUpRows(pleModelProj, pleDim * blockCount, dim);
        FloatTensor.warmUpRows(weights.output(), vocabSize, dim);
    }

    /** Multi-token {@link #forwardLayer}: the same steps, with every projection batched. */
    private void forwardLayerBatch(Gemma4State st, float[][][] b, int layer, int basePos, int n, boolean doPle) {
        TransformerLayerWeights lw = weights.layers()[layer];
        boolean hasOwnKv = layer < blockCount - sharedKvLayers;
        int qDim = layerQDim(layer);
        int kvDim = layerKvDim(layer);
        int layerFfn = config.layerFfnLength(layer);
        float[][] x = b[B_X], xb = b[B_XB], attnOut = b[B_ATTN];

        // Pre-attention norm, then Q/K/V projections
        for (int t = 0; t < n; t++) {
            RMSNorm.apply(xb[t], x[t], attnNormCache[layer], dim, normEps);
            Arrays.fill(b[B_Q][t], 0, qDim, 0f);
            if (hasOwnKv) {
                Arrays.fill(b[B_K][t], 0, kvDim, 0f);
                Arrays.fill(b[B_V][t], 0, kvDim, 0f);
            }
        }
        if (hasOwnKv && lw.wv() != null) {
            FloatTensor.fusedQKVBatchParallel(lw.wq(), lw.wk(), lw.wv(), xb, b[B_Q], b[B_K], b[B_V],
                n, qDim, kvDim, dim);
        } else {
            FloatTensor.matmulBatchParallel(lw.wq(), xb, b[B_Q], n, qDim, dim);
            if (hasOwnKv) FloatTensor.matmulBatchParallel(lw.wk(), xb, b[B_K], n, kvDim, dim);
        }

        // Attention, token by token in position order
        for (int t = 0; t < n; t++) {
            System.arraycopy(b[B_Q][t], 0, st.qLarge, 0, qDim);
            if (hasOwnKv) {
                System.arraycopy(b[B_K][t], 0, st.kLarge, 0, kvDim);
                if (lw.wv() != null) System.arraycopy(b[B_V][t], 0, st.vLarge, 0, kvDim);
            }
            attentionCore(st, layer, basePos + t);
            System.arraycopy(st.xb2Large, 0, b[B_OUT][t], 0, qDim);
            Arrays.fill(xb[t], 0f);
        }
        FloatTensor.matmulBatchParallel(lw.wo(), b[B_OUT], xb, n, dim, qDim);

        // Post-attention norm, attention residual, pre-FFN norm
        for (int t = 0; t < n; t++) {
            if (postAttnNormCache[layer] != null) {
                RMSNorm.apply(xb[t], xb[t], postAttnNormCache[layer], dim, normEps);
            }
            for (int i = 0; i < dim; i++) attnOut[t][i] = xb[t][i] + x[t][i];
            RMSNorm.apply(xb[t], attnOut[t], ffnNormCache[layer], dim, normEps);
            Arrays.fill(b[B_HB][t], 0, layerFfn, 0f);
            Arrays.fill(b[B_HB2][t], 0, layerFfn, 0f);
        }

        // GeGLU FFN
        if (lw.wGate() != null) {
            FloatTensor.fusedGateUpBatchParallel(lw.wGate(), lw.wUp(), xb, b[B_HB], b[B_HB2], n, layerFfn, dim);
        } else {
            FloatTensor.matmulBatchParallel(lw.wUp(), xb, b[B_HB2], n, layerFfn, dim);
        }
        for (int t = 0; t < n; t++) {
            float[] hb = b[B_HB][t], hb2 = b[B_HB2][t];
            for (int i = 0; i < layerFfn; i++) {
                float v = hb[i];
                hb[i] = 0.5f * v * (1.0f + (float) Math.tanh(
                        0.7978845608028654f * (v + 0.044715f * v * v * v)));
                hb[i] *= hb2[i];
            }
            Arrays.fill(xb[t], 0f);
        }
        FloatTensor.matmulBatchParallel(lw.wDown(), b[B_HB], xb, n, dim, layerFfn);

        // Post-FFN norm and FFN residual
        for (int t = 0; t < n; t++) {
            if (postFfnNormCache[layer] != null) {
                RMSNorm.apply(xb[t], xb[t], postFfnNormCache[layer], dim, normEps);
            }
            for (int i = 0; i < dim; i++) x[t][i] = xb[t][i] + attnOut[t][i];
        }

        // PLE injection: x += post_norm(proj @ (gelu(inp_gate @ x) * ple_layer))
        if (doPle && pleInpGate[layer] != null) {
            float[][] gate = b[B_PLE_GATE];
            for (int t = 0; t < n; t++) Arrays.fill(gate[t], 0f);
            FloatTensor.matmulBatchParallel(pleInpGate[layer], x, gate, n, pleDim, dim);
            int pleOffset = layer * pleDim;
            for (int t = 0; t < n; t++) {
                float[] g = gate[t], ple = b[B_PLE][t];
                for (int i = 0; i < pleDim; i++) {
                    float v = g[i];
                    g[i] = 0.5f * v * (1.0f + (float) Math.tanh(
                            0.7978845608028654f * (v + 0.044715f * v * v * v)));
                    g[i] *= ple[pleOffset + i];
                }
                Arrays.fill(xb[t], 0f);
            }
            FloatTensor.matmulBatchParallel(pleProj[layer], gate, xb, n, dim, pleDim);
            for (int t = 0; t < n; t++) {
                if (plePostNorm[layer] != null) {
                    RMSNorm.apply(xb[t], xb[t], plePostNorm[layer], dim, normEps);
                }
                for (int i = 0; i < dim; i++) x[t][i] += xb[t][i];
            }
        }

        // Layer output scale
        if (layerOutputScale != null && layer < layerOutputScale.length) {
            float scale = layerOutputScale[layer];
            if (scale != 1.0f && scale != 0f) {
                for (int t = 0; t < n; t++) {
                    for (int i = 0; i < dim; i++) x[t][i] *= scale;
                }
            }
        }
    }

    /**
     * Pre-compute PLE input for this token.
     * Combines token-identity embedding with context-aware projection.
     */
    private void computePleInput(int token, Gemma4State state) {
        int totalPleDim = pleDim * blockCount;
        // Context-aware projection: per_layer_model_proj @ x → [totalPleDim]
        Arrays.fill(state.pleProjected, 0, totalPleDim, 0f);
        pleModelProj.matmulParallel(state.x, state.pleProjected, totalPleDim, dim);
        combinePle(token, state.pleProjected, state.pleEmb, state.pleCombined);
    }

    /**
     * Turns the raw {@code per_layer_model_proj @ x} of one token ({@code projected}, modified in
     * place) into its combined PLE input: scale and per-layer-norm the projection, add the scaled
     * token-identity embedding (looked up into the {@code emb} scratch) and scale the sum.
     */
    private void combinePle(int token, float[] projected, float[] emb, float[] combined) {
        int totalPleDim = pleDim * blockCount;

        // Token-identity embedding lookup → [totalPleDim], scaled by sqrt(pleDim)
        float pleScale = (float) Math.sqrt(pleDim);
        for (int i = 0; i < totalPleDim; i++) {
            emb[i] = pleTokenEmbd.getFloat((long) token * totalPleDim + i) * pleScale;
        }

        // Projection scaled by 1/sqrt(dim)
        float projScale = 1.0f / (float) Math.sqrt(dim);
        for (int i = 0; i < totalPleDim; i++) projected[i] *= projScale;

        // Apply per-layer RMSNorm to each pleDim-sized chunk of projected
        for (int l = 0; l < blockCount; l++) {
            RMSNorm.apply(projected, l * pleDim, projected, l * pleDim,
                    pleProjNormWeights, pleDim, normEps);
        }

        // Combine and scale: (projected + emb) * (1/sqrt(2))
        float combineScale = 1.0f / (float) Math.sqrt(2.0);
        for (int i = 0; i < totalPleDim; i++) {
            combined[i] = (projected[i] + emb[i]) * combineScale;
        }
    }

    /**
     * Forward pass for a single transformer layer.
     */
    private void forwardLayer(Gemma4State state, int layer, int position, boolean doPle) {
        TransformerLayerWeights lw = weights.layers()[layer];
        boolean hasPle = pleTokenEmbd != null && pleDim > 0;
        boolean hasOwnKv = layer < blockCount - sharedKvLayers;
        int qDim = layerQDim(layer);
        int kvDim = layerKvDim(layer);

        // === 1. Pre-attention RMSNorm ===
        RMSNorm.apply(state.xb, state.x, attnNormCache[layer], dim, normEps);

        // === 2. Q/K/V projections === (use large buffers that fit both SWA and full layers)
        Arrays.fill(state.qLarge, 0, qDim, 0f);
        lw.wq().matmulParallel(state.xb, state.qLarge, qDim, dim);
        if (hasOwnKv) {
            Arrays.fill(state.kLarge, 0, kvDim, 0f);
            lw.wk().matmulParallel(state.xb, state.kLarge, kvDim, dim);
            if (lw.wv() != null) {
                Arrays.fill(state.vLarge, 0, kvDim, 0f);
                lw.wv().matmulParallel(state.xb, state.vLarge, kvDim, dim);
            }
        }

        // === 3-4. QK-norm, V-norm, RoPE, KV store, attention → xb2Large ===
        attentionCore(state, layer, position);
        final float[] xb2Buf = state.xb2Large;

        // === 5. Wo projection ===
        Arrays.fill(state.xb, 0);
        lw.wo().matmulParallel(xb2Buf, state.xb, dim, qDim);

        // === 6. Post-attention norm ===
        if (postAttnNormCache[layer] != null) {
            RMSNorm.apply(state.xb, state.xb, postAttnNormCache[layer], dim, normEps);
        }

        // === 7. Attention residual: attn_out = xb + x ===
        float[] attnOut = state.xb3; // use separate buffer for attn_out
        for (int i = 0; i < dim; i++) attnOut[i] = state.xb[i] + state.x[i];

        // === 8. Pre-FFN RMSNorm ===
        RMSNorm.apply(state.xb, attnOut, ffnNormCache[layer], dim, normEps);

        // === 9. GeGLU FFN: gate * up with GELU activation ===
        // Gemma 4 E2B/E4B use a per-layer "double-wide MLP": feed_forward_length is a per-layer
        // array, so the gate/up/down weights of the wider layers have more rows than the narrower
        // ones. Use the per-layer width — the scalar max would over-read the narrow layers.
        int layerFfn = config.layerFfnLength(layer);
        Arrays.fill(state.hb, 0, layerFfn, 0f);
        Arrays.fill(state.hb2, 0, layerFfn, 0f);
        if (lw.wGate() != null) {
            lw.wGate().matmulParallel(state.xb, state.hb, layerFfn, dim);
        }
        lw.wUp().matmulParallel(state.xb, state.hb2, layerFfn, dim);

        // GELU activation on gate, then element-wise multiply
        for (int i = 0; i < layerFfn; i++) {
            float x = state.hb[i];
            // GELU with tanh approximation (gelu_pytorch_tanh)
            state.hb[i] = 0.5f * x * (1.0f + (float) Math.tanh(
                    0.7978845608028654f * (x + 0.044715f * x * x * x)));
            state.hb[i] *= state.hb2[i];
        }

        // Down projection
        Arrays.fill(state.xb, 0);
        lw.wDown().matmulParallel(state.hb, state.xb, dim, layerFfn);

        // === 10. Post-FFN norm (post_ffw_norm.weight) ===
        if (postFfnNormCache[layer] != null) {
            RMSNorm.apply(state.xb, state.xb, postFfnNormCache[layer], dim, normEps);
        }

        // === 11. FFN residual: x = xb + attn_out ===
        for (int i = 0; i < dim; i++) state.x[i] = state.xb[i] + attnOut[i];

        // === 12. PLE injection (after full layer) ===
        if (doPle && pleInpGate[layer] != null) {
            float[] pleGate = state.pleGate;
            Arrays.fill(pleGate, 0, pleDim, 0f);

            // gate = gelu(inp_gate @ x) — inp_gate maps [dim → pleDim]
            pleInpGate[layer].matmulParallel(state.x, pleGate, pleDim, dim);
            for (int i = 0; i < pleDim; i++) {
                float g = pleGate[i];
                pleGate[i] = 0.5f * g * (1.0f + (float) Math.tanh(
                        0.7978845608028654f * (g + 0.044715f * g * g * g)));
            }

            // Element-wise multiply with pre-computed PLE for this layer
            int pleOffset = layer * pleDim;
            for (int i = 0; i < pleDim; i++) {
                pleGate[i] *= state.pleCombined[pleOffset + i];
            }

            // Up-project: proj @ gated_ple → [dim]
            float[] pleOut = state.pleOut;
            Arrays.fill(pleOut, 0, dim, 0f);
            pleProj[layer].matmulParallel(pleGate, pleOut, dim, pleDim);

            // Post-PLE norm (post_norm.weight)
            if (plePostNorm[layer] != null) {
                RMSNorm.apply(pleOut, pleOut, plePostNorm[layer], dim, normEps);
            }

            // PLE residual: cur = pe_in + post_norm(per_layer_proj @ ...)
            for (int i = 0; i < dim; i++) state.x[i] += pleOut[i];
        }

        // === 13. Layer output scale: cur *= out_scale ===
        // Per llama.cpp gemma4-iswa.cpp, the entire layer output is multiplied by
        // a learned per-layer scalar before being passed to the next layer.
        if (layerOutputScale != null && layer < layerOutputScale.length) {
            float scale = layerOutputScale[layer];
            if (scale != 1.0f && scale != 0f) {
                for (int i = 0; i < dim; i++) state.x[i] *= scale;
            }
        }
    }

    /**
     * The order-dependent part of an attention layer: QK-norm, V-norm, RoPE, KV store and the
     * attention itself for one token at {@code position}. Reads the raw projections from
     * {@code state.qLarge} (and {@code kLarge}/{@code vLarge} on layers with their own KV) and
     * writes the attention output, before Wo, to {@code state.xb2Large}. Shared by the one-token
     * path and batched prefill, which feeds it one token at a time in position order.
     */
    private void attentionCore(Gemma4State state, int layer, int position) {
        TransformerLayerWeights lw = weights.layers()[layer];
        boolean hasOwnKv = layer < blockCount - sharedKvLayers;
        boolean isSwa = isSwaLayer(layer);
        RoPE rope = isSwa ? ropeSwa : ropeFull;

        // Per-layer dimensions: SWA and full attention layers have different headSize,
        // and Gemma 4 dense (12B) also varies the KV head count per layer — the full count
        // on SWA layers, a reduced count (often 1) on the every-6th global layers. Using the
        // scalar max here would size the K/V matmul past the actual wk/wv rows on global
        // layers and read out of bounds.
        int headSize = isSwa ? headSizeSwa : headSizeFull;
        int layerKvHeads = config.layerKvHeads(layer);
        int layerKvMul = headCount / layerKvHeads;
        int kvDim = layerKvHeads * headSize;

        final float[] qBuf = state.qLarge;
        final float[] kBuf = state.kLarge;
        final float[] vBuf = state.vLarge;
        final float[] xb2Buf = state.xb2Large;

        // QK-norm on Q (with learnable scale)
        if (qNormCache[layer] != null) {
            for (int h = 0; h < headCount; h++) {
                RMSNorm.apply(qBuf, h * headSize, qBuf, h * headSize,
                        qNormCache[layer], headSize, normEps);
            }
        }

        // RoPE on Q
        rope.applyAllHeads(qBuf, headCount, position);

        // === K/V (conditional on shared KV) ===
        int kvLayer = kvSourceLayer[layer]; // which layer's KV cache to use
        if (hasOwnKv) {
            if (lw.wv() == null) {
                // Gemma 4 "alternative attention": the global (full-attention) layers ship no V
                // projection — llama.cpp gemma4.cpp marks wv TENSOR_NOT_REQUIRED and uses
                // `Vcur = wv ? wv@x : Kcur`, i.e. the raw K projection becomes V. Capture kBuf
                // here, before the K-norm/RoPE below mutate it (V then gets only its own V-norm).
                System.arraycopy(kBuf, 0, vBuf, 0, kvDim);
            }

            // QK-norm on K (with learnable scale)
            if (kNormCache[layer] != null) {
                for (int h = 0; h < layerKvHeads; h++) {
                    RMSNorm.apply(kBuf, h * headSize, kBuf, h * headSize,
                            kNormCache[layer], headSize, normEps);
                }
            }

            // V-norm: rms_norm without learnable scale (both Gemma 3n and Gemma 4
            // per llama.cpp gemma4-iswa.cpp: Vcur = ggml_rms_norm(ctx0, Vcur, eps)).
            for (int h = 0; h < layerKvHeads; h++) {
                RMSNorm.applyNoScale(vBuf, h * headSize, headSize, normEps);
            }

            // RoPE on K
            rope.applyAllHeads(kBuf, layerKvHeads, position);

            // Store K/V in per-layer cache (kvDim varies per layer type)
            int kvCacheDim = state.gemma4KvCache.kvDim(layer);
            System.arraycopy(kBuf, 0, state.gemma4KvCache.keyLayer(layer),
                    position * kvCacheDim, kvDim);
            System.arraycopy(vBuf, 0, state.gemma4KvCache.valueLayer(layer),
                    position * kvCacheDim, kvDim);
        }
        // else: shared KV — use kvLayer's cache (already populated)

        // === Attention ===
        final int hs = headSize; // capture for lambda
        // Gemma 4: attention scale = 1.0 (model handles scaling via QK-norm internally)
        final float attnScale = 1.0f;
        final int startPos = (isSwa && slidingWindow > 0)
                ? Math.max(0, position - slidingWindow + 1) : 0;
        final int seqLen = position + 1;
        final int kvCacheDim = state.gemma4KvCache.kvDim(kvLayer);
        final float[] keyCache = state.gemma4KvCache.keyLayer(kvLayer);
        final float[] valueCache = state.gemma4KvCache.valueLayer(kvLayer);

        // Parallel attention over heads
        final int kvMulLayer = layerKvMul;
        final it.denzosoft.llmplayer.tensor.VectorOps ops = VectorOpsFactory.get();
        it.denzosoft.llmplayer.tensor.MatmulPool.forEach(headCount, h -> {
            int kvHead = h / kvMulLayer;
            int qOffset = h * hs;

            // Compute attention scores
            for (int t = startPos; t < seqLen; t++) {
                float score = ops.dot(qBuf, qOffset, keyCache, t * kvCacheDim + kvHead * hs, hs);
                state.att[h * maxSeqLen + t] = score * attnScale;
            }

            // Softmax
            float maxVal = Float.NEGATIVE_INFINITY;
            for (int t = startPos; t < seqLen; t++) {
                if (state.att[h * maxSeqLen + t] > maxVal) maxVal = state.att[h * maxSeqLen + t];
            }
            float sum = 0f;
            for (int t = startPos; t < seqLen; t++) {
                float v = (float) Math.exp(state.att[h * maxSeqLen + t] - maxVal);
                state.att[h * maxSeqLen + t] = v;
                sum += v;
            }
            float invSum = 1.0f / sum;
            for (int t = startPos; t < seqLen; t++) {
                state.att[h * maxSeqLen + t] *= invSum;
            }

            // Weighted V sum → xb2 (per element, still accumulated in position order)
            int outOffset = h * hs;
            Arrays.fill(xb2Buf, outOffset, outOffset + hs, 0f);
            for (int t = startPos; t < seqLen; t++) {
                ops.saxpy(state.att[h * maxSeqLen + t], valueCache, t * kvCacheDim + kvHead * hs,
                        xb2Buf, outOffset, hs);
            }
        });
    }

    // === Public API matching InferenceEngine pattern ===

    public float[] prefill(Gemma4State st, int[] tokens) {
        this.state = st;
        for (int i = 0; i < tokens.length - 1; i++) {
            forward(tokens[i], i, false);
        }
        return forward(tokens[tokens.length - 1], tokens.length - 1, true);
    }

    public float[] forwardSingleToken(int token, int position) {
        if (state == null) state = createState();
        return forward(token, position, true);
    }

    public void setState(Gemma4State st) { this.state = st; }
    public Gemma4State getState() { return state; }

    // ============================================================================
    // H10: Gemma 3n AltUp + Laurel implementation
    // ============================================================================
    // Reference: llama.cpp src/models/gemma3n-iswa.cpp
    //
    // Architecture overview:
    //   - 4 parallel activation streams (n_altup=4)
    //   - Active stream (i_altup_act=0) goes through normal attention + FFN
    //   - Other 3 streams are "predicted" via learned linear combinations
    //   - After each layer, predictions are "corrected" using actual layer output
    //   - Per-layer first_prediction (active stream → PLE space → back) is added to
    //     the 3 non-active streams to share information across them
    //   - At the end, all 4 streams are averaged with magnitude rescaling
    //
    // Intentional simplifications in this implementation:
    //   - Activation sparsity (gaussian_topk on first n_layer_sparsity FFN gates) NOT applied
    //   - V-norm (rms_norm without scale on V) NOT applied — uses standard projection
    //   - has_kv per-layer logic uses existing kvSourceLayer mapping
    //
    // These simplifications mean the output may not be bit-exact with llama.cpp,
    // but should be MUCH better than the "random multi-language tokens" failure
    // mode of the PLE-only path.

    /**
     * AltUp init: project the active stream into 4 streams via altup_proj.
     * Input: state.x (active stream, [dim])
     * Output: state.altupStreams[4][dim], with stream 0 = state.x and streams 1..3
     *         = altup_proj @ state.x with magnitude rescaling.
     */
    private void altupInit(Gemma4State state) {
        int nAltup = g3n.nAltup;
        int iAct = g3n.iAltupAct;
        // Stream 0 = active = current state.x
        System.arraycopy(state.x, 0, state.altupStreams[iAct], 0, dim);

        // Compute target magnitude of active stream
        float targetMag = magnitude(state.x);

        // For each non-active stream, project active via altup_proj[stream_idx - 1]
        // altup_proj has shape [dim, dim, n_altup-1] in the GGUF — slice (n_altup-1) matrices.
        // We use slice index s = streamIdx - (streamIdx > iAct ? 1 : 0) into the n_altup-1 slices.
        for (int s = 0; s < nAltup; s++) {
            if (s == iAct) continue;
            int sliceIdx = s > iAct ? s - 1 : s;
            // matmul: [dim] = altup_proj[sliceIdx] @ x
            float[] target = state.altupStreams[s];
            java.util.Arrays.fill(target, 0f);
            long sliceOffset = (long) sliceIdx * dim * dim;
            for (int row = 0; row < dim; row++) {
                float sum = 0f;
                for (int col = 0; col < dim; col++) {
                    sum += g3n.altupProj.getFloat(sliceOffset + (long) row * dim + col) * state.x[col];
                }
                target[row] = sum;
            }
            // Rescale to target magnitude
            float currentMag = magnitude(target);
            if (currentMag > 1e-8f) {
                float scale = targetMag / currentMag;
                for (int i = 0; i < dim; i++) target[i] *= scale;
            }
        }
    }

    /** L2 magnitude of a vector. */
    private static float magnitude(float[] v) {
        float sum = 0f;
        for (float x : v) sum += x * x;
        return (float) Math.sqrt(sum);
    }

    /**
     * AltUp merge: collapse 4 streams to single x via altup_unembd_proj + averaging.
     * Input: state.altupStreams[4][dim]
     * Output: state.x (averaged + rescaled)
     */
    private void altupMerge(Gemma4State state) {
        int nAltup = g3n.nAltup;
        int iAct = g3n.iAltupAct;
        // Target magnitude from active stream
        float targetMag = magnitude(state.altupStreams[iAct]);

        // Project non-active streams via altup_unembd_proj and rescale to target mag
        // Then average all 4 (active + 3 unembedded) into state.x
        // Start x = active stream
        System.arraycopy(state.altupStreams[iAct], 0, state.x, 0, dim);

        float[] tmp = state.firstPredFullDim; // reuse as scratch
        for (int s = 0; s < nAltup; s++) {
            if (s == iAct) continue;
            int sliceIdx = s > iAct ? s - 1 : s;
            // tmp = altup_unembd_proj[sliceIdx] @ altupStreams[s]
            java.util.Arrays.fill(tmp, 0f);
            long sliceOffset = (long) sliceIdx * dim * dim;
            for (int row = 0; row < dim; row++) {
                float sum = 0f;
                for (int col = 0; col < dim; col++) {
                    sum += g3n.altupUnembdProj.getFloat(sliceOffset + (long) row * dim + col)
                         * state.altupStreams[s][col];
                }
                tmp[row] = sum;
            }
            // Rescale to target mag
            float curMag = magnitude(tmp);
            if (curMag > 1e-8f) {
                float scale = targetMag / curMag;
                for (int i = 0; i < dim; i++) tmp[i] *= scale;
            }
            // Add to x
            for (int i = 0; i < dim; i++) state.x[i] += tmp[i];
        }
        // Average: divide by n_altup
        float invN = 1.0f / nAltup;
        for (int i = 0; i < dim; i++) state.x[i] *= invN;
    }

    /**
     * Compute router modalities for AltUp predict/correct.
     * router_input = rmsnorm(activated, altup_router_norm) * (1/dim)
     * modalities = tanh(altup_router @ router_input)  → [n_altup]
     */
    private void altupComputeRouterModalities(float[] activated, int layer, float[] outModalities) {
        int nAltup = g3n.nAltup;
        // RMSNorm of activated with altup_router_norm
        RMSNorm.apply(state.altupRouterInput, activated, g3n.altupRouterNorm[layer], dim, normEps);
        // Scale by 1/dim
        float invDim = 1.0f / dim;
        for (int i = 0; i < dim; i++) state.altupRouterInput[i] *= invDim;
        // matmul: outModalities[nAltup] = altup_router @ router_input
        // altup_router shape [dim, n_altup] in GGUF (rows=dim, cols=n_altup)
        // To compute output[k] = sum(router[i,k] * input[i]) we treat the tensor as [n_altup, dim]
        // because GGUF stores as [d0, d1] where d0=dim, d1=n_altup, and matmul is output_rows × inputs.
        // Actually based on llama.cpp `ggml_mul_mat(altup_router, router_inputs)`, the output is
        // [n_altup, n_tokens], so altup_router is treated as [dim, n_altup].
        java.util.Arrays.fill(outModalities, 0f);
        FloatTensor router = g3n.altupRouter[layer];
        for (int k = 0; k < nAltup; k++) {
            float sum = 0f;
            for (int i = 0; i < dim; i++) {
                sum += router.getFloat((long) k * dim + i) * state.altupRouterInput[i];
            }
            outModalities[k] = (float) Math.tanh(sum);
        }
    }

    /**
     * AltUp predict: compute next-state predictions for all 4 streams.
     * Input: state.altupStreams[4][dim]
     * Output: state.altupPredictions[4][dim]
     */
    private void altupPredict(Gemma4State state, int layer) {
        int nAltup = g3n.nAltup;
        int iAct = g3n.iAltupAct;
        float[] activated = state.altupStreams[iAct];
        altupComputeRouterModalities(activated, layer, state.altupModalities);

        // all_coefs [nAltup * nAltup] = altup_predict_coef[layer] @ modalities[nAltup]
        // altup_predict_coef shape [nAltup, nAltup*nAltup] in GGUF
        FloatTensor predictCoef = g3n.altupPredictCoef[layer];
        java.util.Arrays.fill(state.altupAllCoefs, 0f);
        for (int row = 0; row < nAltup * nAltup; row++) {
            float sum = 0f;
            for (int col = 0; col < nAltup; col++) {
                sum += predictCoef.getFloat((long) row * nAltup + col) * state.altupModalities[col];
            }
            state.altupAllCoefs[row] = sum;
        }
        // all_coefs is now reshaped as [nAltup][nAltup]: rows = output stream, cols = input stream

        // For each output stream s_out, compute:
        //   prediction[s_out] = sum over s_in of streams[s_in] * all_coefs[s_out][s_in]
        // Then add residual: prediction += streams (so it's an UPDATE not a replacement)
        for (int sOut = 0; sOut < nAltup; sOut++) {
            float[] pred = state.altupPredictions[sOut];
            java.util.Arrays.fill(pred, 0f);
            for (int sIn = 0; sIn < nAltup; sIn++) {
                float coef = state.altupAllCoefs[sOut * nAltup + sIn];
                float[] src = state.altupStreams[sIn];
                for (int i = 0; i < dim; i++) pred[i] += src[i] * coef;
            }
            // Residual
            float[] selfStream = state.altupStreams[sOut];
            for (int i = 0; i < dim; i++) pred[i] += selfStream[i];
        }
    }

    /**
     * AltUp correct: update predictions using actual layer output.
     * Input:
     *   state.altupPredictions[4][dim] — pre-layer predictions
     *   activatedActual[dim] — actual layer output for active stream
     * Output: state.altupCorrected[4][dim]
     */
    private void altupCorrect(Gemma4State state, float[] activatedActual, int layer) {
        int nAltup = g3n.nAltup;
        int iAct = g3n.iAltupAct;
        altupComputeRouterModalities(activatedActual, layer, state.altupModalities);

        // innovation = activatedActual - predictions[iAct]
        float[] activePred = state.altupPredictions[iAct];
        for (int i = 0; i < dim; i++) state.altupInnovation[i] = activatedActual[i] - activePred[i];

        // all_coefs [nAltup] = altup_correct_coef[layer] @ modalities[nAltup]; then +1.0
        FloatTensor correctCoef = g3n.altupCorrectCoef[layer];
        for (int row = 0; row < nAltup; row++) {
            float sum = 0f;
            for (int col = 0; col < nAltup; col++) {
                sum += correctCoef.getFloat((long) row * nAltup + col) * state.altupModalities[col];
            }
            state.altupAllCoefs[row] = sum + 1.0f;
        }

        // For each stream: corrected[s] = innovation * all_coefs[s] + predictions[s]
        for (int s = 0; s < nAltup; s++) {
            float coef = state.altupAllCoefs[s];
            float[] dst = state.altupCorrected[s];
            float[] pred = state.altupPredictions[s];
            for (int i = 0; i < dim; i++) {
                dst[i] = state.altupInnovation[i] * coef + pred[i];
            }
        }
    }

    /**
     * Laurel low-rank residual branch.
     * Input: cur[dim] (post-attention input)
     * Output: state.laurelOut[dim] = (laurel_r @ laurel_l @ cur) + cur (with post_norm)
     */
    private void laurel(float[] cur, int layer) {
        int laurelRank = state.laurelTmpRank.length;
        // tmp = laurel_l @ cur ([dim, rank] @ [dim] → [rank])
        java.util.Arrays.fill(state.laurelTmpRank, 0f);
        FloatTensor lL = g3n.laurelL[layer];
        for (int row = 0; row < laurelRank; row++) {
            float sum = 0f;
            for (int col = 0; col < dim; col++) {
                sum += lL.getFloat((long) row * dim + col) * cur[col];
            }
            state.laurelTmpRank[row] = sum;
        }
        // out = laurel_r @ tmp ([rank, dim] @ [rank] → [dim])
        java.util.Arrays.fill(state.laurelOut, 0f);
        FloatTensor lR = g3n.laurelR[layer];
        for (int row = 0; row < dim; row++) {
            float sum = 0f;
            for (int col = 0; col < laurelRank; col++) {
                sum += lR.getFloat((long) row * laurelRank + col) * state.laurelTmpRank[col];
            }
            state.laurelOut[row] = sum;
        }
        // Post-norm + residual: out = rmsnorm(out, laurel_post_norm) + cur
        RMSNorm.apply(state.laurelOut, state.laurelOut, g3n.laurelPostNorm[layer], dim, normEps);
        for (int i = 0; i < dim; i++) state.laurelOut[i] += cur[i];
    }

    /**
     * Forward one layer in AltUp mode.
     * Conceptually:
     *   1. predictions = altup_predict(streams)
     *   2. active = predictions[iAct]; norm; laurel; attention; post-norm; +active
     *   3. attn_laurel = (cur + laurel_out) / sqrt(2)
     *   4. norm; ffn; post-norm; +attn_laurel  → activatedActual
     *   5. corrected = altup_correct(predictions, activatedActual)
     *   6. first_pred from corrected[iAct] → PLE space → back-projected → injected into corrected[1..3]
     *   7. streams = corrected
     *
     * For attention/FFN, this delegates to forwardLayer() via state.x temporarily.
     */
    private void forwardLayerAltup(Gemma4State state, int layer, int position, boolean doPle) {
        int iAct = g3n.iAltupAct;
        int nAltup = g3n.nAltup;

        // 1. Predict next state for all streams (uses current state.altupStreams[*])
        altupPredict(state, layer);

        // 2. Active stream goes through dedicated Gemma 3n layer compute, in the
        //    correct order: laurel branched off pre-attention, merged BETWEEN
        //    attention and FFN. Result is the "activatedActual" used by altupCorrect.
        float[] activePrediction = state.altupPredictions[iAct];

        // forwardLayerGemma3nInner writes the layer output into state.xb3 (so it doesn't
        // disturb the residual stream state.x of the standard path).
        forwardLayerGemma3nInner(state, layer, activePrediction, position);
        // state.xb3 now contains the activatedActual

        // 5. Correct predictions using actual active output
        altupCorrect(state, state.xb3, layer);

        // 6. First prediction injection (only if PLE is loaded)
        if (doPle && pleProj[layer] != null && pleInpGate[layer] != null) {
            // first_pred = corrected[iAct] * altup_correct_scale[layer]
            float[] firstPred = state.firstPredFullDim;
            float[] active = state.altupCorrected[iAct];
            for (int i = 0; i < dim; i++) firstPred[i] = active[i] * g3n.altupCorrectScale[layer][i];
            // first_pred_altup = per_layer_inp_gate[layer] @ first_pred  ([dim] → [pleDim])
            java.util.Arrays.fill(state.firstPredAltup, 0f);
            pleInpGate[layer].matmul(firstPred, state.firstPredAltup, pleDim, dim);
            // gelu
            float SQRT_2_OVER_PI = (float) Math.sqrt(2.0 / Math.PI);
            for (int i = 0; i < pleDim; i++) {
                float v = state.firstPredAltup[i];
                state.firstPredAltup[i] = 0.5f * v * (1.0f + (float) Math.tanh(
                    SQRT_2_OVER_PI * (v + 0.044715f * v * v * v)));
            }
            // Multiply with this layer's per-layer input (from pleCombined)
            for (int i = 0; i < pleDim; i++) {
                state.firstPredAltup[i] *= state.pleCombined[layer * pleDim + i];
            }
            // Project back to dim space: per_layer_proj[layer] @ first_pred_altup
            java.util.Arrays.fill(firstPred, 0f);
            pleProj[layer].matmul(state.firstPredAltup, firstPred, dim, pleDim);
            // Apply per_layer_post_norm
            if (plePostNorm[layer] != null) {
                RMSNorm.apply(firstPred, firstPred, plePostNorm[layer], dim, normEps);
            }
            // Add to non-active streams: corrected[s != iAct] += first_pred
            for (int s = 0; s < nAltup; s++) {
                if (s == iAct) continue;
                float[] dst = state.altupCorrected[s];
                for (int i = 0; i < dim; i++) dst[i] += firstPred[i];
            }
        }

        // 7. Update streams = corrected
        for (int s = 0; s < nAltup; s++) {
            System.arraycopy(state.altupCorrected[s], 0, state.altupStreams[s], 0, dim);
        }
    }

    /**
     * Dedicated single-layer forward for Gemma 3n active stream.
     *
     * <p>Correct order per llama.cpp gemma3n-iswa.cpp:
     * <pre>
     *   cur = norm(active_prediction, attn_norm)
     *   laurel_out = laurel(cur)         ← computed on the SAME pre-norm output
     *   Q,K,V = wq/wk/wv @ cur (or shared KV if !has_kv)
     *   Q = q_norm(Q) per-head
     *   K = k_norm(K) per-head
     *   V = rms_norm(V, no scale) per-head    ← V-norm (NEW)
     *   apply RoPE to Q,K
     *   attention(Q,K,V) → attn_out
     *   attn_out = wo @ attn_out
     *   attn_out = post_attn_norm(attn_out)
     *   cur = attn_out + active_prediction         ← residual with PREDICTION not state.x
     *   attn_laurel = (cur + laurel_out) / sqrt(2) ← laurel merged HERE
     *   cur = norm(attn_laurel, ffn_norm)          ← FFN takes attn_laurel as input
     *   gate = wGate @ cur
     *   if (layer &lt; n_layer_sparsity) gate = gaussianTopK(gate)  ← activation sparsity
     *   gate = gelu(gate)
     *   up = wUp @ cur
     *   h = gate * up
     *   cur = wDown @ h
     *   cur = post_ffn_norm(cur)
     *   activatedActual = cur + attn_laurel        ← residual with attn_laurel
     * </pre>
     *
     * <p>Output is written to {@code state.xb3} (so the standard {@code state.x}
     * residual stream is not touched).
     */
    private void forwardLayerGemma3nInner(Gemma4State state, int layer, float[] activePrediction, int position) {
        TransformerLayerWeights lw = weights.layers()[layer];
        boolean hasOwnKv = g3n.hasKv != null ? g3n.hasKv[layer] : (layer < blockCount - sharedKvLayers);
        boolean isSwa = isSwaLayer(layer);
        RoPE rope = isSwa ? ropeSwa : ropeFull;
        int headSize = isSwa ? headSizeSwa : headSizeFull;
        int qDim = headCount * headSize;
        int kvDim = headCountKV * headSize;

        // === 1. Pre-attention norm of active prediction (into state.xb) ===
        RMSNorm.apply(state.xb, activePrediction, attnNormCache[layer], dim, normEps);

        // === 2. Laurel branch (computed on the same pre-norm output) ===
        // Writes to state.laurelOut. Must happen BEFORE attention because attention reuses xb buffers.
        laurel(state.xb, layer);
        // Save laurel out to a stable buffer (state.firstPredFullDim is unused at this point)
        float[] savedLaurel = state.firstPredFullDim;
        System.arraycopy(state.laurelOut, 0, savedLaurel, 0, dim);

        // === 3. Q projection ===
        final float[] qBuf = state.qLarge;
        final float[] kBuf = state.kLarge;
        final float[] vBuf = state.vLarge;
        final float[] xb2Buf = state.xb2Large;
        Arrays.fill(qBuf, 0, qDim, 0f);
        lw.wq().matmulParallel(state.xb, qBuf, qDim, dim);

        // QK-norm on Q (per-head)
        if (qNormCache[layer] != null) {
            for (int h = 0; h < headCount; h++) {
                RMSNorm.apply(qBuf, h * headSize, qBuf, h * headSize, qNormCache[layer], headSize, normEps);
            }
        }
        rope.applyAllHeads(qBuf, headCount, position);

        // === 4. K/V projection (conditional on hasOwnKv) ===
        int kvLayer = kvSourceLayer[layer];
        if (hasOwnKv) {
            Arrays.fill(kBuf, 0, kvDim, 0f);
            Arrays.fill(vBuf, 0, kvDim, 0f);
            lw.wk().matmulParallel(state.xb, kBuf, kvDim, dim);
            lw.wv().matmulParallel(state.xb, vBuf, kvDim, dim);

            // QK-norm on K
            if (kNormCache[layer] != null) {
                for (int h = 0; h < headCountKV; h++) {
                    RMSNorm.apply(kBuf, h * headSize, kBuf, h * headSize, kNormCache[layer], headSize, normEps);
                }
            }
            // V-norm: rms_norm without learnable scale (Gemma 3n specific)
            for (int h = 0; h < headCountKV; h++) {
                RMSNorm.applyNoScale(vBuf, h * headSize, headSize, normEps);
            }
            rope.applyAllHeads(kBuf, headCountKV, position);

            // Store K/V in per-layer cache
            int kvCacheDim = state.gemma4KvCache.kvDim(layer);
            System.arraycopy(kBuf, 0, state.gemma4KvCache.keyLayer(layer), position * kvCacheDim, kvDim);
            System.arraycopy(vBuf, 0, state.gemma4KvCache.valueLayer(layer), position * kvCacheDim, kvDim);
        }

        // === 5. Attention ===
        final int hs = headSize;
        final int startPos = (isSwa && slidingWindow > 0) ? Math.max(0, position - slidingWindow + 1) : 0;
        final int seqLen = position + 1;
        final int kvCacheDim = state.gemma4KvCache.kvDim(kvLayer);
        final float[] keyCache = state.gemma4KvCache.keyLayer(kvLayer);
        final float[] valueCache = state.gemma4KvCache.valueLayer(kvLayer);

        final float attnScale = 1.0f;
        final it.denzosoft.llmplayer.tensor.VectorOps ops = VectorOpsFactory.get();
        it.denzosoft.llmplayer.tensor.MatmulPool.forEach(headCount, h -> {
            int kvHead = h / kvMul;
            int qOffset = h * hs;
            for (int t = startPos; t < seqLen; t++) {
                float score = ops.dot(qBuf, qOffset, keyCache, t * kvCacheDim + kvHead * hs, hs);
                state.att[h * maxSeqLen + t] = score * attnScale;
            }
            // Softmax
            float maxVal = Float.NEGATIVE_INFINITY;
            for (int t = startPos; t < seqLen; t++) {
                if (state.att[h * maxSeqLen + t] > maxVal) maxVal = state.att[h * maxSeqLen + t];
            }
            float sum = 0f;
            for (int t = startPos; t < seqLen; t++) {
                float v = (float) Math.exp(state.att[h * maxSeqLen + t] - maxVal);
                state.att[h * maxSeqLen + t] = v;
                sum += v;
            }
            float invSum = 1.0f / sum;
            for (int t = startPos; t < seqLen; t++) state.att[h * maxSeqLen + t] *= invSum;
            // Weighted V sum (per element, still accumulated in position order)
            int outOffset = h * hs;
            Arrays.fill(xb2Buf, outOffset, outOffset + hs, 0f);
            for (int t = startPos; t < seqLen; t++) {
                ops.saxpy(state.att[h * maxSeqLen + t], valueCache, t * kvCacheDim + kvHead * hs,
                        xb2Buf, outOffset, hs);
            }
        });

        // === 6. Wo projection (into state.xb) ===
        Arrays.fill(state.xb, 0);
        lw.wo().matmulParallel(xb2Buf, state.xb, dim, qDim);

        // === 7. Post-attention norm ===
        if (postAttnNormCache[layer] != null) {
            RMSNorm.apply(state.xb, state.xb, postAttnNormCache[layer], dim, normEps);
        }

        // === 8. Residual with active_prediction ===
        // cur = post_attn_norm(attn_out) + active_prediction
        for (int i = 0; i < dim; i++) state.xb[i] += activePrediction[i];

        // === 9. Laurel merge: attn_laurel = (cur + laurel_out) / sqrt(2) ===
        // Stash attn_laurel in state.xb3 — we need it both as FFN input and as final residual.
        float invSqrt2 = 1.0f / (float) Math.sqrt(2.0);
        for (int i = 0; i < dim; i++) state.xb3[i] = (state.xb[i] + savedLaurel[i]) * invSqrt2;

        // === 10. Pre-FFN norm (norm of attn_laurel into state.xb) ===
        RMSNorm.apply(state.xb, state.xb3, ffnNormCache[layer], dim, normEps);

        // === 11. GeGLU FFN with optional activation sparsity ===
        Arrays.fill(state.hb, 0, ffnDim, 0f);
        Arrays.fill(state.hb2, 0, ffnDim, 0f);
        if (lw.wGate() != null) {
            lw.wGate().matmulParallel(state.xb, state.hb, ffnDim, dim);
        }
        lw.wUp().matmulParallel(state.xb, state.hb2, ffnDim, dim);

        // Activation sparsity (gaussian top-k) on the gate, only for the first n_layer_sparsity layers
        if (layer < g3n.nLayerSparsity) {
            gaussianTopK(state.hb, ffnDim, g3n.fSparsityStdMul);
        }

        // GELU on gate then element-wise multiply with up
        for (int i = 0; i < ffnDim; i++) {
            float xv = state.hb[i];
            state.hb[i] = 0.5f * xv * (1.0f + (float) Math.tanh(
                0.7978845608028654f * (xv + 0.044715f * xv * xv * xv)));
            state.hb[i] *= state.hb2[i];
        }

        // Down projection (into state.xb)
        Arrays.fill(state.xb, 0);
        lw.wDown().matmulParallel(state.hb, state.xb, dim, ffnDim);

        // === 12. Post-FFN norm ===
        if (postFfnNormCache[layer] != null) {
            RMSNorm.apply(state.xb, state.xb, postFfnNormCache[layer], dim, normEps);
        }

        // === 13. Final residual: activatedActual = post_ffn_norm(ffn_out) + attn_laurel ===
        // attn_laurel is in state.xb3; we want activatedActual in state.xb3 too.
        for (int i = 0; i < dim; i++) state.xb3[i] += state.xb[i];
        // state.xb3 now contains activatedActual — caller (forwardLayerAltup) reads from here.
    }

    /**
     * Gaussian top-k activation sparsity (Gemma 3n FFN).
     *
     * <p>For a vector {@code x} of length n:
     * <pre>
     *   mean = sum(x) / n
     *   variance = sum((x - mean)^2) / (n - 1)   // Bessel's correction
     *   std = sqrt(variance)
     *   cutoff = mean + std * f_sparsity_std_mul
     *   x[i] = max(0, x[i] - cutoff)             // ReLU
     * </pre>
     * Effectively keeps only neurons in the top tail of the Gaussian.
     */
    private static void gaussianTopK(float[] x, int n, float fSparsityStdMul) {
        // 1. mean
        float sum = 0f;
        for (int i = 0; i < n; i++) sum += x[i];
        float mean = sum / n;
        // 2. variance with Bessel's correction (n-1)
        float varSum = 0f;
        for (int i = 0; i < n; i++) {
            float d = x[i] - mean;
            varSum += d * d;
        }
        float std = (float) Math.sqrt(varSum / (n - 1));
        // 3. cutoff
        float cutoff = mean + std * fSparsityStdMul;
        // 4. relu(x - cutoff)
        for (int i = 0; i < n; i++) {
            float v = x[i] - cutoff;
            x[i] = v > 0f ? v : 0f;
        }
    }
}
