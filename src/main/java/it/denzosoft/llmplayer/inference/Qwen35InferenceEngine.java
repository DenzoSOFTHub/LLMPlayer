package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.model.Qwen35LayerWeights;
import it.denzosoft.llmplayer.model.Qwen35Weights;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

import java.lang.reflect.Method;
import it.denzosoft.llmplayer.tensor.FloatTensor;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Inference engine for Qwen3.5 architecture.
 * Hybrid model: alternating Gated DeltaNet (linear attention) and full GQA layers.
 * Pattern: 3 DeltaNet + 1 Full Attention, controlled by full_attention_interval.
 *
 * DeltaNet layers: recurrent state-space model with gated updates, no KV cache.
 * Full Attention layers: standard GQA with KV cache, QK-norm, RoPE.
 */
public class Qwen35InferenceEngine {

    private final ModelConfig config;
    private final Qwen35Weights weights;
    private final int maxSeqLen;

    // Cached config values
    private final int dim;
    private final int ffnDim;
    private final int vocabSize;
    private final int blockCount;
    private final int timeStepRank;
    private final int stateSize;
    private final int groupCount;
    private final int innerSize;
    private final int convKernel;
    private final int fullAttnInterval;
    private final float normEps;

    // Attention parameters for full attention layers
    private final int headCount;
    private final int headCountKV;
    private final int headSize;
    private final int kvDim;
    private final int kvMul;  // headCount / headCountKV for GQA

    // DeltaNet parameters
    private final int dQK;   // Q/K dimension per head = stateSize (head_k_dim)
    private final int dV;    // V dimension per head = stateSize (head_v_dim)
    private final int qkvDim; // total QKV projection dim

    // Pre-cached norm weights
    private final float[] outputNormCache;
    // E11: pre-cache all per-layer norm weights at construction time so the forward pass
    // doesn't re-dequantize + allocate a new float[dim] every token. Before the fix, each
    // forward pass called cacheWeightsInline() several times per layer, generating ~800 KB
    // of garbage per token on the 4B model.
    private final float[][] attnNormPerLayer;    // [blockCount][dim]
    private final float[][] postAttnNormPerLayer; // [blockCount][dim] (full-attention layers)
    private final float[][] ssmNormPerLayer;     // [blockCount][stateSize] (DeltaNet layers)

    // RoPE for full attention layers
    private final RoPE rope;
    private final int[] mropeMap; // IMROPE section per rotated pair, used for image tokens

    // Pre-cached per-layer norm weights for attention layers
    private final float[][] qNormCache;
    private final float[][] kNormCache;

    // GPU forward pass (loaded via reflection from java21/, null if unavailable)
    private AutoCloseable gpuForwardPass;
    private int gpuLayerCount;
    private Method gpuUploadXAndUpdateParams;
    private Method gpuForwardLayer;
    private Method gpuForwardGraph;
    private Method gpuForwardGraphArgmax;
    private Method gpuForwardFinalLogits;
    private Method gpuForwardFinalArgmax;
    private Method gpuForwardGraphPrefill;
    private Method gpuDownloadX;

    public Qwen35InferenceEngine(ModelConfig config, Qwen35Weights weights, int maxSeqLen,
                                  float[] ropeFreqFactors) {
        this.config = config;
        this.weights = weights;
        this.maxSeqLen = maxSeqLen;

        this.dim = config.embeddingLength();
        this.ffnDim = config.intermediateSize();
        this.vocabSize = config.vocabSize();
        this.blockCount = config.blockCount();
        this.timeStepRank = config.ssmTimeStepRank();
        this.stateSize = config.ssmStateSize();
        this.groupCount = config.ssmGroupCount();
        this.innerSize = config.ssmInnerSize();
        this.convKernel = config.ssmConvKernel();
        this.fullAttnInterval = config.fullAttentionInterval();
        this.normEps = config.normEps();

        this.headCount = config.headCount();
        this.headCountKV = config.headCountKV();
        this.headSize = config.headSize();
        this.kvDim = config.kvDim();
        this.kvMul = headCount / headCountKV;

        this.dQK = stateSize;  // head_k_dim = stateSize
        this.dV = stateSize;   // head_v_dim = stateSize
        this.qkvDim = groupCount * stateSize * 2 + timeStepRank * stateSize;

        // Cache output norm weights
        this.outputNormCache = new float[dim];
        for (int i = 0; i < dim; i++) {
            outputNormCache[i] = weights.outputNorm().getFloat(i);
        }

        // RoPE for full attention layers
        int ropeDimCount = config.ropeDimensionCount();
        this.rope = new RoPE(headSize, ropeDimCount, maxSeqLen, config.ropeFreqBase(),
            config.ropeType(), ropeFreqFactors);
        this.mropeMap = config.ropeSections() != null
            ? RoPE.mropeSectionMap(config.ropeSections(), config.ropeSectionsInterleaved(), ropeDimCount / 2)
            : null;

        // Cache QK norm weights for attention layers
        this.qNormCache = new float[blockCount][];
        this.kNormCache = new float[blockCount][];
        for (int layer = 0; layer < blockCount; layer++) {
            Qwen35LayerWeights lw = weights.layers()[layer];
            if (!lw.isDeltaNet() && lw.qNorm() != null) {
                qNormCache[layer] = cacheWeights(lw.qNorm(), headSize);
                kNormCache[layer] = cacheWeights(lw.kNorm(), headSize);
            }
        }

        // E11: pre-cache per-layer norm weights (attn_norm always, post_attn + ssm_norm per type)
        this.attnNormPerLayer = new float[blockCount][];
        this.postAttnNormPerLayer = new float[blockCount][];
        this.ssmNormPerLayer = new float[blockCount][];
        for (int layer = 0; layer < blockCount; layer++) {
            Qwen35LayerWeights lw = weights.layers()[layer];
            if (lw.attnNorm() != null) {
                attnNormPerLayer[layer] = cacheWeights(lw.attnNorm(), dim);
            }
            if (lw.postAttnNorm() != null) {
                postAttnNormPerLayer[layer] = cacheWeights(lw.postAttnNorm(), dim);
            }
            if (lw.isDeltaNet() && lw.ssmNorm() != null) {
                ssmNormPerLayer[layer] = cacheWeights(lw.ssmNorm(), stateSize);
            }
        }
    }

    private float[] cacheWeights(it.denzosoft.llmplayer.tensor.FloatTensor tensor, int size) {
        float[] cache = new float[size];
        for (int i = 0; i < size; i++) cache[i] = tensor.getFloat(i);
        return cache;
    }

    /**
     * Try to initialize GPU forward pass via reflection (java21/ Qwen35CudaForwardPass).
     * Call after construction with the CudaBufferManager from LLMEngine.
     */
    public void tryInitGpuForwardPass(Object bufferManager) {
        try {
            Class<?> fwdClass = Class.forName("it.denzosoft.llmplayer.inference.Qwen35CudaForwardPass");

            // Check isSupported
            Method isSup = fwdClass.getMethod("isSupported", ModelConfig.class, Qwen35Weights.class);
            Boolean supported = (Boolean) isSup.invoke(null, config, weights);
            if (!supported) {
                System.err.println("Qwen35 CUDA forward pass: not supported (weights not on GPU)");
                return;
            }

            // Construct
            Object fwd = fwdClass.getConstructor(ModelConfig.class, Qwen35Weights.class,
                    bufferManager.getClass(), int.class)
                .newInstance(config, weights, bufferManager, maxSeqLen);

            // Cache methods
            gpuUploadXAndUpdateParams = fwdClass.getMethod("uploadXAndUpdateParams", float[].class, int.class);
            gpuForwardLayer = fwdClass.getMethod("forwardLayer", int.class, int.class);
            gpuForwardGraph = fwdClass.getMethod("forwardGraph", float[].class);
            gpuForwardGraphArgmax = fwdClass.getMethod("forwardGraphArgmax");
            gpuForwardFinalLogits = fwdClass.getMethod("forwardFinalLogits", float[].class);
            gpuForwardFinalArgmax = fwdClass.getMethod("forwardFinalArgmax");
            gpuForwardGraphPrefill = fwdClass.getMethod("forwardGraphPrefill");
            gpuDownloadX = fwdClass.getMethod("downloadX", float[].class);

            Method getGpuLayers = fwdClass.getMethod("getGpuLayerCount");
            gpuLayerCount = (Integer) getGpuLayers.invoke(fwd);
            gpuForwardPass = (AutoCloseable) fwd;

            System.err.println("Qwen35 CUDA forward pass: enabled (" + gpuLayerCount + "/" + blockCount + " layers)");
        } catch (Throwable e) {
            System.err.println("Qwen35 CUDA forward pass: unavailable — " + e.getMessage());
            gpuForwardPass = null;
        }
    }

    /** See {@link InferenceEngine#disableGpuForwardPass}. */
    public void disableGpuForwardPass() {
        if (gpuForwardPass != null) {
            try { gpuForwardPass.close(); } catch (Exception ignored) { }
            gpuForwardPass = null;
            System.out.println("Qwen3.5 GPU forward pass: disabled (vision input needs the CPU layer path)");
        }
    }

    public Qwen35State createState(int maxSeqLen) {
        return new Qwen35State(config, maxSeqLen);
    }

    public float[] forward(Qwen35State state, int token, int position) {
        return forwardInternal(state, token, position, true);
    }

    public void forwardNoOutput(Qwen35State state, int token, int position) {
        forwardInternal(state, token, position, false);
    }

    private float[] forwardInternal(Qwen35State state, int token, int position, boolean computeLogits) {
        // 1. Token embedding
        for (int i = 0; i < dim; i++) {
            state.x[i] = weights.tokenEmbedding().getFloat((long) token * dim + i);
        }

        // 2. Try GPU forward pass
        if (gpuForwardPass != null) {
            try {
                return forwardGpu(state, position, computeLogits);
            } catch (Exception e) {
                System.err.println("Qwen35 GPU forward failed, falling back to CPU: " + e.getMessage());
                gpuForwardPass = null;
            }
        }

        // 3. CPU forward through all layers
        forwardLayersCpu(state, position);

        if (!computeLogits) return null;
        return finishLogits(state);
    }

    /** All layers for the residual stream already in {@code state.x}, on the CPU. */
    private void forwardLayersCpu(Qwen35State state, int position) {
        long t0 = 0, t1;
        for (int layer = 0; layer < blockCount; layer++) {
            Qwen35LayerWeights lw = weights.layers()[layer];
            if (lw.isDeltaNet()) {
                if (cpuProfile) t0 = System.nanoTime();
                forwardDeltaNet(state, lw, layer);
                if (cpuProfile) { t1 = System.nanoTime(); profDeltaNetNs += t1 - t0; }
            } else {
                if (cpuProfile) t0 = System.nanoTime();
                forwardAttention(state, lw, layer, position);
                if (cpuProfile) { t1 = System.nanoTime(); profAttnNs += t1 - t0; }
            }
        }
    }

    /**
     * Prefill precomputed input embeddings (vision tokens, [n][dim]) at KV slots
     * {@code kvStart .. kvStart + n - 1}; rope positions come from {@code state.mrope}. Batched like
     * {@link #forwardPrefill} when possible. Logits are not computed.
     */
    public void prefillEmbeddings(Qwen35State state, float[][] embds, int kvStart) {
        int n = embds.length;
        if (n > 1 && PREFILL_BATCHED && gpuForwardPass == null
                && it.denzosoft.llmplayer.tensor.MatmulPool.enabled()) {
            float[][][] b = prefillBuffers(state);
            int cap = b[B_X].length;
            for (int base = 0; base < n; base += cap) {
                int m = Math.min(cap, n - base);
                for (int t = 0; t < m; t++) System.arraycopy(embds[base + t], 0, b[B_X][t], 0, dim);
                prefillLayers(state, b, kvStart + base, m);
            }
            return;
        }
        for (int t = 0; t < n; t++) {
            System.arraycopy(embds[t], 0, state.x, 0, dim);
            forwardLayersCpu(state, kvStart + t);
        }
    }

    /** Final norm and output projection for the residual stream in {@code state.x}. */
    private float[] finishLogits(Qwen35State state) {
        long t0 = 0;
        if (cpuProfile) t0 = System.nanoTime();
        VectorOpsFactory.get().rmsnorm(state.xb, state.x, outputNormCache, dim, normEps);
        Arrays.fill(state.logits, 0);
        weights.output().matmulParallel(state.xb, state.logits, vocabSize, dim);
        if (cpuProfile) {
            profOutputNs += System.nanoTime() - t0;
            profTokenCount++;
            if (profTokenCount % 10 == 0) printProfile();
        }

        return state.logits;
    }

    private final boolean cpuProfile = "true".equals(System.getProperty("cpu.profile"));
    private long profDeltaNetNs, profAttnNs, profOutputNs;
    private int profTokenCount;

    private void printProfile() {
        int n = profTokenCount;
        double ms = 1e6;
        long total = profDeltaNetNs + profAttnNs + profOutputNs;
        System.out.printf("[cpu-profile Qwen35] %d tokens, per-token avg (ms): deltanet=%.1f attn(GQA)=%.1f output=%.1f | total=%.1f%n",
            n, profDeltaNetNs / ms / n, profAttnNs / ms / n, profOutputNs / ms / n, total / ms / n);
    }

    private float[] forwardGpu(Qwen35State state, int position, boolean computeLogits) throws Exception {
        // Upload embedding + token params
        gpuUploadXAndUpdateParams.invoke(gpuForwardPass, state.x, position);

        // Try CUDA graph (use generation graph for both prefill and generation)
        if (gpuLayerCount == blockCount) {
            Boolean graphOk = (Boolean) gpuForwardGraph.invoke(gpuForwardPass, state.logits);
            if (graphOk) return computeLogits ? state.logits : null;
        }

        // Per-layer GPU forward
        for (int layer = 0; layer < gpuLayerCount; layer++) {
            gpuForwardLayer.invoke(gpuForwardPass, layer, position);
        }
        // Profiling hook (no-op unless -Dqwen35.profile=true)
        try {
            java.lang.reflect.Method m = gpuForwardPass.getClass().getMethod("profileTokenComplete");
            m.invoke(gpuForwardPass);
        } catch (NoSuchMethodException ignored) {}

        // If not all layers on GPU, download X and continue on CPU
        if (gpuLayerCount < blockCount) {
            gpuDownloadX.invoke(gpuForwardPass, state.x);
            for (int layer = gpuLayerCount; layer < blockCount; layer++) {
                Qwen35LayerWeights lw = weights.layers()[layer];
                if (lw.isDeltaNet()) {
                    forwardDeltaNet(state, lw, layer);
                } else {
                    forwardAttention(state, lw, layer, position);
                }
            }
            if (!computeLogits) return null;
            VectorOpsFactory.get().rmsnorm(state.xb, state.x, outputNormCache, dim, normEps);
            Arrays.fill(state.logits, 0);
            weights.output().matmulParallel(state.xb, state.logits, vocabSize, dim);
            return state.logits;
        }

        if (!computeLogits) return null;

        // All layers on GPU — try GPU output projection
        Boolean logitsOk = (Boolean) gpuForwardFinalLogits.invoke(gpuForwardPass, state.logits);
        if (logitsOk) return state.logits;

        // Fallback: download X and compute output on CPU
        gpuDownloadX.invoke(gpuForwardPass, state.x);
        VectorOpsFactory.get().rmsnorm(state.xb, state.x, outputNormCache, dim, normEps);
        Arrays.fill(state.logits, 0);
        weights.output().matmulParallel(state.xb, state.logits, vocabSize, dim);
        return state.logits;
    }

    public float[] prefill(Qwen35State state, int[] tokens) {
        for (int i = 0; i < tokens.length - 1; i++) {
            forwardNoOutput(state, tokens[i], i);
        }
        return forward(state, tokens[tokens.length - 1], tokens.length - 1);
    }

    // ==================== Batched prefill ====================

    private static final int PREFILL_BATCH = Integer.getInteger("prefill.batch", 64);
    private static final boolean PREFILL_BATCHED =
        !"false".equalsIgnoreCase(System.getProperty("prefill.batched", "true"));

    // Indices into Qwen35State.prefillBuffers
    private static final int B_X = 0, B_XB = 1, B_QKV = 2, B_ALPHA = 3, B_BETA = 4, B_GATE = 5,
        B_OUT = 6, B_Q = 7, B_K = 8, B_V = 9, B_HB = 10, B_HB2 = 11;

    /**
     * Prefill {@code tokens[fromPos..toPos)} and return the logits of the last one.
     *
     * <p>On the CPU the prompt goes through each layer in chunks of {@link #PREFILL_BATCH} tokens,
     * and every projection — DeltaNet QKV, alpha, beta, output gate and {@code ssm_out}; attention
     * Q+gate, K, V and O; FFN gate, up and down — runs as one multi-token matmul
     * ({@code FloatTensor.matmulRowsBatch}). The order-dependent parts ({@link #deltaNetCore}: conv1d
     * and the DeltaNet recurrence; {@link #attentionCore}: RoPE, KV store and attention) still run
     * token by token in position order with the same code as the one-token path, so their state
     * evolves exactly as before; only the summation order inside the matmul kernels differs.
     * Disable with {@code -Dprefill.batched=false}.
     */
    public float[] forwardPrefill(Qwen35State state, int[] tokens, int fromPos, int toPos) {
        int count = toPos - fromPos;
        if (count <= 0) return null;
        if (count == 1 || !PREFILL_BATCHED || gpuForwardPass != null
                || !it.denzosoft.llmplayer.tensor.MatmulPool.enabled()) {
            float[] logits = null;
            for (int i = fromPos; i < toPos; i++) {
                if (i < toPos - 1) forwardNoOutput(state, tokens[i], i);
                else logits = forward(state, tokens[i], i);
            }
            return logits;
        }
        float[][][] b = prefillBuffers(state);
        int cap = b[B_X].length;
        for (int base = fromPos; base < toPos; base += cap) {
            int n = Math.min(cap, toPos - base);
            for (int t = 0; t < n; t++) {
                int token = tokens[base + t];
                for (int i = 0; i < dim; i++) {
                    b[B_X][t][i] = weights.tokenEmbedding().getFloat((long) token * dim + i);
                }
            }
            prefillLayers(state, b, base, n);
            if (base + n == toPos) System.arraycopy(b[B_X][n - 1], 0, state.x, 0, dim);
        }
        return finishLogits(state);
    }

    /** Batched-prefill buffers, created on first use (warms the decode kernels once). */
    private float[][][] prefillBuffers(Qwen35State state) {
        warmDecodeKernels();
        int cap = Math.max(1, PREFILL_BATCH);
        int qGateDim = 2 * headCount * headSize;
        if (state.prefillBuffers == null) {
            state.prefillBuffers = new float[][][] {
                new float[cap][dim], new float[cap][dim], new float[cap][qkvDim],
                new float[cap][timeStepRank], new float[cap][timeStepRank], new float[cap][innerSize],
                new float[cap][Math.max(innerSize, headCount * headSize)], new float[cap][qGateDim],
                new float[cap][kvDim], new float[cap][kvDim], new float[cap][ffnDim], new float[cap][ffnDim]
            };
        }
        return state.prefillBuffers;
    }

    /** All layers for the chunk whose inputs are in {@code b[B_X][0..n)}, at KV slots base.. */
    private void prefillLayers(Qwen35State state, float[][][] b, int base, int n) {
        for (int layer = 0; layer < blockCount; layer++) {
            Qwen35LayerWeights lw = weights.layers()[layer];
            if (lw.isDeltaNet()) {
                deltaNetBatch(state, b, lw, layer, n);
            } else {
                attentionBatch(state, b, lw, layer, base, n);
            }
            ffnBatch(b, lw, layer, n);
        }
    }

    /** See {@link FloatTensor#warmUpRows}: batched prefill skips the kernels decode will use. */
    private void warmDecodeKernels() {
        int qDim = headCount * headSize;
        for (Qwen35LayerWeights lw : weights.layers()) {
            if (lw.isDeltaNet()) {
                FloatTensor.warmUpRows(lw.attnQkv(), qkvDim, dim);
                FloatTensor.warmUpRows(lw.attnGate(), innerSize, dim);
                FloatTensor.warmUpRows(lw.ssmOut(), dim, innerSize);
            } else {
                FloatTensor.warmUpRows(lw.wq(), 2 * qDim, dim);
                FloatTensor.warmUpRows(lw.wk(), kvDim, dim);
                FloatTensor.warmUpRows(lw.wo(), dim, qDim);
            }
            FloatTensor.warmUpRows(lw.ffnGate(), ffnDim, dim);
            FloatTensor.warmUpRows(lw.ffnDown(), dim, ffnDim);
        }
        FloatTensor.warmUpRows(weights.output(), vocabSize, dim);
    }

    private void deltaNetBatch(Qwen35State state, float[][][] b, Qwen35LayerWeights lw, int layer, int n) {
        for (int t = 0; t < n; t++) {
            VectorOpsFactory.get().rmsnorm(b[B_XB][t], b[B_X][t], attnNormPerLayer[layer], dim, normEps);
            Arrays.fill(b[B_QKV][t], 0f);
            Arrays.fill(b[B_ALPHA][t], 0f);
            Arrays.fill(b[B_BETA][t], 0f);
            Arrays.fill(b[B_GATE][t], 0f);
        }
        FloatTensor.matmulBatchParallel(lw.attnQkv(), b[B_XB], b[B_QKV], n, qkvDim, dim);
        FloatTensor.matmulBatchParallel(lw.ssmAlpha(), b[B_XB], b[B_ALPHA], n, timeStepRank, dim);
        FloatTensor.matmulBatchParallel(lw.ssmBeta(), b[B_XB], b[B_BETA], n, timeStepRank, dim);
        FloatTensor.matmulBatchParallel(lw.attnGate(), b[B_XB], b[B_GATE], n, innerSize, dim);
        for (int t = 0; t < n; t++) {
            System.arraycopy(b[B_QKV][t], 0, state.qkv, 0, qkvDim);
            System.arraycopy(b[B_ALPHA][t], 0, state.alpha, 0, timeStepRank);
            System.arraycopy(b[B_BETA][t], 0, state.beta, 0, timeStepRank);
            System.arraycopy(b[B_GATE][t], 0, state.gate, 0, innerSize);
            deltaNetCore(state, lw, layer);
            System.arraycopy(state.deltaOut, 0, b[B_OUT][t], 0, innerSize);
            Arrays.fill(b[B_XB][t], 0f);
        }
        FloatTensor.matmulBatchParallel(lw.ssmOut(), b[B_OUT], b[B_XB], n, dim, innerSize);
        for (int t = 0; t < n; t++) {
            for (int i = 0; i < dim; i++) b[B_X][t][i] += b[B_XB][t][i];
        }
    }

    private void attentionBatch(Qwen35State state, float[][][] b, Qwen35LayerWeights lw, int layer, int basePos, int n) {
        int qDim = headCount * headSize;
        int qGateDim = 2 * qDim;
        for (int t = 0; t < n; t++) {
            VectorOpsFactory.get().rmsnorm(b[B_XB][t], b[B_X][t], attnNormPerLayer[layer], dim, normEps);
            Arrays.fill(b[B_Q][t], 0f);
            Arrays.fill(b[B_K][t], 0f);
            Arrays.fill(b[B_V][t], 0f);
        }
        FloatTensor.matmulBatchParallel(lw.wq(), b[B_XB], b[B_Q], n, qGateDim, dim);
        FloatTensor.matmulBatchParallel(lw.wk(), b[B_XB], b[B_K], n, kvDim, dim);
        FloatTensor.matmulBatchParallel(lw.wv(), b[B_XB], b[B_V], n, kvDim, dim);
        for (int t = 0; t < n; t++) {
            System.arraycopy(b[B_Q][t], 0, state.q, 0, qGateDim);
            System.arraycopy(b[B_K][t], 0, state.k, 0, kvDim);
            System.arraycopy(b[B_V][t], 0, state.v, 0, kvDim);
            attentionCore(state, layer, basePos + t);
            System.arraycopy(state.xb2, 0, b[B_OUT][t], 0, qDim);
            Arrays.fill(b[B_XB][t], 0f);
        }
        FloatTensor.matmulBatchParallel(lw.wo(), b[B_OUT], b[B_XB], n, dim, qDim);
        for (int t = 0; t < n; t++) {
            for (int i = 0; i < dim; i++) b[B_X][t][i] += b[B_XB][t][i];
        }
    }

    /** Multi-token {@link #forwardFFN}: same norm, activation formula and residual per token. */
    private void ffnBatch(float[][][] b, Qwen35LayerWeights lw, int layer, int n) {
        for (int t = 0; t < n; t++) {
            VectorOpsFactory.get().rmsnorm(b[B_XB][t], b[B_X][t], postAttnNormPerLayer[layer], dim, normEps);
            Arrays.fill(b[B_HB][t], 0f);
            Arrays.fill(b[B_HB2][t], 0f);
        }
        FloatTensor.fusedGateUpBatchParallel(lw.ffnGate(), lw.ffnUp(), b[B_XB], b[B_HB], b[B_HB2], n, ffnDim, dim);
        for (int t = 0; t < n; t++) {
            float[] hb = b[B_HB][t], hb2 = b[B_HB2][t];
            for (int i = 0; i < ffnDim; i++) {
                float val = hb[i];
                hb[i] = (val * sigmoid(val)) * hb2[i];
            }
            Arrays.fill(b[B_XB][t], 0f);
        }
        FloatTensor.matmulBatchParallel(lw.ffnDown(), b[B_HB], b[B_XB], n, dim, ffnDim);
        for (int t = 0; t < n; t++) {
            for (int i = 0; i < dim; i++) b[B_X][t][i] += b[B_XB][t][i];
        }
    }

    // ==================== DeltaNet Forward Pass ====================

    private void forwardDeltaNet(Qwen35State state, Qwen35LayerWeights lw, int layer) {
        // Pre-attention RMSNorm (uses pre-cached weights, no per-token allocation)
        VectorOpsFactory.get().rmsnorm(state.xb, state.x, attnNormPerLayer[layer], dim, normEps);

        // Projections from xb: QKV, alpha and beta gates, output gate. They depend only on xb,
        // so computing all four before the conv/recurrence is the same as interleaving them.
        Arrays.fill(state.qkv, 0);
        lw.attnQkv().matmulParallel(state.xb, state.qkv, qkvDim, dim);
        Arrays.fill(state.alpha, 0);
        lw.ssmAlpha().matmulParallel(state.xb, state.alpha, timeStepRank, dim);
        Arrays.fill(state.beta, 0);
        lw.ssmBeta().matmulParallel(state.xb, state.beta, timeStepRank, dim);
        Arrays.fill(state.gate, 0);
        lw.attnGate().matmulParallel(state.xb, state.gate, innerSize, dim);

        deltaNetCore(state, lw, layer);

        // Output projection: ssm_out @ deltaOut -> xb
        Arrays.fill(state.xb, 0);
        lw.ssmOut().matmulParallel(state.deltaOut, state.xb, dim, innerSize);

        // Residual connection (attention)
        for (int i = 0; i < dim; i++) {
            state.x[i] += state.xb[i];
        }

        // Post-attention norm + FFN + residual
        forwardFFN(state, lw, layer);
    }

    /**
     * The per-token, order-dependent part of a DeltaNet layer: from the raw projections in
     * {@code state.qkv / alpha / beta / gate} to the gated output in {@code state.deltaOut}. It
     * advances the layer's conv and SSM state, so tokens must go through it in position order.
     */
    private void deltaNetCore(Qwen35State state, Qwen35LayerWeights lw, int layer) {
        // Causal Conv1D
        applyConv1d(state, lw, layer);

        // Apply SiLU activation to QKV after conv
        for (int i = 0; i < qkvDim; i++) {
            float val = state.qkv[i];
            state.qkv[i] = val * sigmoid(val); // SiLU = x * sigmoid(x)
        }

        // Alpha (decay): exp(negExpA * softplus(alpha_proj + dt_bias))
        // ssm_a stores -exp(A_log) (pre-computed by GGUF converter)
        // Beta (update gate): sigmoid(beta_proj)
        for (int h = 0; h < timeStepRank; h++) {
            float negExpA = lw.ssmA().getFloat(h);
            float dtBias = lw.ssmDtBias().getFloat(h);
            float g = negExpA * softplus(state.alpha[h] + dtBias);
            state.alpha[h] = (float) Math.exp(g);
            state.beta[h] = sigmoid(state.beta[h]);
        }

        // Output gate: SiLU(attn_gate @ xb)
        for (int i = 0; i < innerSize; i++) {
            float val = state.gate[i];
            state.gate[i] = val * sigmoid(val);
        }

        // Split QKV and run DeltaNet recurrence per head
        deltaNetRecurrence(state, layer);

        // Apply ssm_norm (per-head RMSNorm on the d_v output) — pre-cached
        applyPerHeadNorm(state.deltaOut, ssmNormPerLayer[layer], timeStepRank, dV, normEps);

        // Gate the output: deltaOut *= gate
        for (int i = 0; i < innerSize; i++) {
            state.deltaOut[i] *= state.gate[i];
        }
    }

    private void applyConv1d(Qwen35State state, Qwen35LayerWeights lw, int layer) {
        float[][] convBuf = state.convState[layer];
        int histSize = convKernel - 1; // 3

        int pos = state.convStatePos[layer]; // number of values stored so far

        // Depthwise conv1d: for each channel, multiply kernel weights by history + current
        // GGUF stores conv1d as [kernel_size, channels] with ne[0]=kernel_size, ne[1]=channels
        // So element for (channel, kernel_pos) is at offset: channel * convKernel + kernel_pos
        // PyTorch Conv1d convention: weight[0] = oldest input, weight[K-1] = current input
        // For causal conv: output[t] = sum_k(weight[K-1-k] * input[t-k]) for k=0..K-1
        // E21: use pre-allocated state.convResult (was: new float[qkvDim] every token)
        float[] result = state.convResult;
        for (int ch = 0; ch < qkvDim; ch++) {
            float sum = 0;
            // Current value (k=0): weight[convKernel-1] * input[t]
            sum += lw.ssmConv1d().getFloat((long) ch * convKernel + (convKernel - 1)) * state.qkv[ch];
            // History (k=1..convKernel-1): weight[convKernel-1-k] * input[t-k]
            for (int k = 1; k < convKernel; k++) {
                if (pos - k >= 0) { // only if we have enough history
                    int histIdx = (pos - k) % histSize;
                    sum += lw.ssmConv1d().getFloat((long) ch * convKernel + (convKernel - 1 - k)) * convBuf[histIdx][ch];
                }
            }
            result[ch] = sum;
        }

        // Store current QKV into circular buffer AFTER conv computation
        System.arraycopy(state.qkv, 0, convBuf[pos % histSize], 0, qkvDim);
        state.convStatePos[layer] = pos + 1;

        System.arraycopy(result, 0, state.qkv, 0, qkvDim);
    }

    private void deltaNetRecurrence(Qwen35State state, int layer) {
        // Split QKV:
        // Q: [groupCount * dQK], K: [groupCount * dQK], V: [timeStepRank * dV]
        int qSize = groupCount * dQK;
        int kSize = groupCount * dQK;
        int vOffset = qSize + kSize;

        // Process per head (timeStepRank heads)
        // Q/K are repeated: head h uses Q/K group (h % groupCount)
        it.denzosoft.llmplayer.tensor.MatmulPool.forEach(timeStepRank, h -> {
            int group = h % groupCount;
            float alphaH = state.alpha[h];
            float betaH = state.beta[h];

            // Get Q and K for this head's group (shared across heads in the group)
            int qOff = group * dQK;
            int kOff = qSize + group * dQK;

            // L2-normalize Q and apply scaling (1/sqrt(head_k_dim))
            float[] qNorm = new float[dQK];
            float qLen = 0;
            for (int i = 0; i < dQK; i++) {
                qNorm[i] = state.qkv[qOff + i];
                qLen += qNorm[i] * qNorm[i];
            }
            float qScale = 1.0f / (float) Math.sqrt(qLen + 1e-12f) * (1.0f / (float) Math.sqrt(dQK));
            for (int i = 0; i < dQK; i++) qNorm[i] *= qScale;

            // L2-normalize K
            float[] kNormalized = new float[dQK];
            float kLen = 0;
            for (int i = 0; i < dQK; i++) {
                kNormalized[i] = state.qkv[kOff + i];
                kLen += kNormalized[i] * kNormalized[i];
            }
            kLen = (float) Math.sqrt(kLen + 1e-12f);
            for (int i = 0; i < dQK; i++) kNormalized[i] /= kLen;

            // Get V for this head
            int vOff = vOffset + h * dV;
            float[] vH = new float[dV];
            for (int i = 0; i < dV; i++) vH[i] = state.qkv[vOff + i];

            // State S is [dQK, dV] for this head
            float[] S = state.ssmState[layer][h];

            // DeltaNet recurrence:
            // S_new = alpha*S + beta * outer(k, v - alpha * S^T @ k)
            // o = S_new^T @ q

            // Compute S^T @ k -> sK [dV]
            float[] sK = new float[dV];
            for (int j = 0; j < dV; j++) {
                float sum = 0;
                for (int i = 0; i < dQK; i++) {
                    sum += S[i * dV + j] * kNormalized[i];
                }
                sK[j] = sum;
            }

            // Update S
            for (int i = 0; i < dQK; i++) {
                for (int j = 0; j < dV; j++) {
                    int idx = i * dV + j;
                    S[idx] = alphaH * S[idx]
                           - alphaH * betaH * sK[j] * kNormalized[i]
                           + betaH * vH[j] * kNormalized[i];
                }
            }

            // Compute output: o = S^T @ q -> o_h [dV]
            int outOff = h * dV;
            for (int j = 0; j < dV; j++) {
                float sum = 0;
                for (int i = 0; i < dQK; i++) {
                    sum += S[i * dV + j] * qNorm[i];
                }
                state.deltaOut[outOff + j] = sum;
            }
        });
    }

    // ==================== Full Attention Forward Pass ====================

    private void forwardAttention(Qwen35State state, Qwen35LayerWeights lw, int layer, int position) {
        // Pre-attention RMSNorm (pre-cached)
        VectorOpsFactory.get().rmsnorm(state.xb, state.x, attnNormPerLayer[layer], dim, normEps);

        int qDim = headCount * headSize;
        int qGateDim = qDim * 2; // Q projection packs Q + gate

        // Q projection: outputs [head0_Q(headSize), head0_gate(headSize), head1_Q(headSize), head1_gate(headSize), ...]
        Arrays.fill(state.q, 0, qGateDim, 0);
        lw.wq().matmulParallel(state.xb, state.q, qGateDim, dim);

        // K, V projections
        Arrays.fill(state.k, 0, kvDim, 0);
        lw.wk().matmulParallel(state.xb, state.k, kvDim, dim);
        Arrays.fill(state.v, 0, kvDim, 0);
        lw.wv().matmulParallel(state.xb, state.v, kvDim, dim);

        attentionCore(state, layer, position);

        // Output projection: wo @ xb2 -> xb
        Arrays.fill(state.xb, 0);
        lw.wo().matmulParallel(state.xb2, state.xb, dim, qDim);

        // Residual connection
        for (int i = 0; i < dim; i++) {
            state.x[i] += state.xb[i];
        }

        // Post-attention norm + FFN + residual
        forwardFFN(state, lw, layer);
    }

    /**
     * The per-token part of a full-attention layer: from the raw projections in {@code state.q}
     * (interleaved Q and gate), {@code state.k} and {@code state.v} to the gated attention output in
     * {@code state.xb2}. Writes this position's KV entry, so tokens must go through it in order.
     */
    private void attentionCore(Qwen35State state, int layer, int position) {
        int qDim = headCount * headSize;
        // Deinterleave Q and gate per-head: raw layout is [Q_h0(headSize), gate_h0(headSize), Q_h1, gate_h1, ...]
        // Extract gates first (to separate array), then compact Q in forward order to avoid overlap
        for (int h = 0; h < headCount; h++) {
            System.arraycopy(state.q, h * headSize * 2 + headSize, state.attnGate, h * headSize, headSize);
        }
        for (int h = 1; h < headCount; h++) {
            System.arraycopy(state.q, h * headSize * 2, state.q, h * headSize, headSize);
        }

        // Per-head QK normalization (only on Q, not on gate)
        if (qNormCache[layer] != null) {
            applyPerHeadNorm(state.q, qNormCache[layer], headCount, headSize, normEps);
            applyPerHeadNorm(state.k, kNormCache[layer], headCountKV, headSize, normEps);
        }

        // RoPE (multi-axis when the sequence contains image tokens; for text all axes are equal
        // and the IMROPE sections cover every pair, so the 1D tables give the same rotation)
        if (state.mrope != null && mropeMap != null) {
            state.mrope.get(position, state.ropePos4);
            if (state.mropeCos == null) {
                state.mropeCos = new float[rope.getRopeDimCount() / 2];
                state.mropeSin = new float[rope.getRopeDimCount() / 2];
            }
            rope.applyMrope(state.q, headCount, mropeMap, state.ropePos4, state.mropeCos, state.mropeSin);
            rope.applyMrope(state.k, headCountKV, mropeMap, state.ropePos4, state.mropeCos, state.mropeSin);
        } else {
            rope.applyAllHeads(state.q, headCount, position);
            rope.applyAllHeads(state.k, headCountKV, position);
        }

        // Store K, V in cache (transparently quantizes in Q8 mode)
        state.kvCache.storeK(layer, position, state.k, kvDim);
        state.kvCache.storeV(layer, position, state.v, kvDim);

        // Multi-head attention
        final float invSqrt = 1.0f / (float) Math.sqrt(headSize);
        final KVCache kv = state.kvCache;
        final int layerFinal = layer;
        final int positionFinal = position;
        final int headSizeFinal = headSize;
        it.denzosoft.llmplayer.tensor.MatmulPool.forEach(headCount, h -> {
            int kvHead = h / kvMul;
            int qOff = h * headSizeFinal;
            int kvHeadOff = kvHead * headSizeFinal;

            // Attention scores
            for (int t = 0; t <= positionFinal; t++) {
                float score = kv.dotK(layerFinal, t, kvHeadOff, headSizeFinal, state.q, qOff);
                state.att[h * maxSeqLen + t] = score * invSqrt;
            }

            // Softmax
            softmax(state.att, h * maxSeqLen, positionFinal + 1);

            // Weighted sum of values
            int outOff = h * headSizeFinal;
            Arrays.fill(state.xb2, outOff, outOff + headSizeFinal, 0);
            for (int t = 0; t <= positionFinal; t++) {
                float a = state.att[h * maxSeqLen + t];
                kv.saxpyV(layerFinal, t, kvHeadOff, headSizeFinal, a, state.xb2, outOff);
            }
        });

        // Apply attention output gate: output *= sigmoid(gate)
        for (int i = 0; i < qDim; i++) {
            state.xb2[i] *= sigmoid(state.attnGate[i]);
        }
    }

    // ==================== SwiGLU FFN ====================

    private void forwardFFN(Qwen35State state, Qwen35LayerWeights lw, int layer) {
        // Post-attention / FFN norm (pre-cached)
        VectorOpsFactory.get().rmsnorm(state.xb, state.x, postAttnNormPerLayer[layer], dim, normEps);

        // SwiGLU FFN: h = SiLU(gate @ xb) * (up @ xb), output = down @ h
        Arrays.fill(state.hb, 0);
        lw.ffnGate().matmulParallel(state.xb, state.hb, ffnDim, dim);
        Arrays.fill(state.hb2, 0);
        lw.ffnUp().matmulParallel(state.xb, state.hb2, ffnDim, dim);

        // SiLU(gate) * up
        for (int i = 0; i < ffnDim; i++) {
            float val = state.hb[i];
            state.hb[i] = (val * sigmoid(val)) * state.hb2[i];
        }

        // Down projection
        Arrays.fill(state.xb, 0);
        lw.ffnDown().matmulParallel(state.hb, state.xb, dim, ffnDim);

        // Residual connection
        for (int i = 0; i < dim; i++) {
            state.x[i] += state.xb[i];
        }
    }

    // ==================== Utility ====================

    private static float sigmoid(float x) {
        return 1.0f / (1.0f + (float) Math.exp(-x));
    }

    private static float softplus(float x) {
        if (x > 20) return x;
        return (float) Math.log(1.0 + Math.exp(x));
    }

    private static void softmax(float[] x, int offset, int size) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < size; i++) {
            if (x[offset + i] > max) max = x[offset + i];
        }
        float sum = 0;
        for (int i = 0; i < size; i++) {
            x[offset + i] = (float) Math.exp(x[offset + i] - max);
            sum += x[offset + i];
        }
        for (int i = 0; i < size; i++) {
            x[offset + i] /= sum;
        }
    }

    private static void applyPerHeadNorm(float[] data, float[] normWeights, int numHeads,
                                          int headDim, float eps) {
        for (int h = 0; h < numHeads; h++) {
            int off = h * headDim;
            float ss = 0;
            for (int i = 0; i < headDim; i++) {
                ss += data[off + i] * data[off + i];
            }
            ss = 1.0f / (float) Math.sqrt(ss / headDim + eps);
            for (int i = 0; i < headDim; i++) {
                data[off + i] = data[off + i] * ss * normWeights[i];
            }
        }
    }

    private float[] cacheWeightsInline(it.denzosoft.llmplayer.tensor.FloatTensor tensor, int size) {
        float[] cache = new float[size];
        for (int i = 0; i < size; i++) cache[i] = tensor.getFloat(i);
        return cache;
    }

    public ModelConfig getConfig() { return config; }
}
