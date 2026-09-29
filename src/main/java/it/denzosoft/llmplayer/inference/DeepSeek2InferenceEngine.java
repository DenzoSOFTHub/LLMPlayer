package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.DeepSeek2LayerWeights;
import it.denzosoft.llmplayer.model.DeepSeek2Weights;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

import java.util.Arrays;

/**
 * Inference engine for DeepSeek2 architecture.
 *
 * Forward pass pipeline:
 * 1. Token embedding lookup
 * 2. For each layer:
 *    a. RMSNorm → MLA Attention → Residual
 *    b. RMSNorm → (Dense SwiGLU FFN or MoE FFN) → Residual
 * 3. Final RMSNorm
 * 4. Output projection → logits
 */
public class DeepSeek2InferenceEngine {

    private final ModelConfig config;
    private final DeepSeek2Weights weights;
    private final MLAAttention mlaAttention;
    private final MoEFFN moeFFN;
    private final float[][] cachedAttnNorm;
    private final float[][] cachedFfnNorm;
    private final float[] outputNormCache;
    private final int maxSeqLen;

    // Phase timing (see DecodeProfile): attention and expert phases always, the rest with -Dcpu.profile
    private static final int P_ATTN_NORM = 0, P_ATTN = 1, P_FFN_NORM = 2, P_DENSE = 3, P_MOE = 4,
        P_RESIDUAL = 5, P_OUTPUT = 6;
    private final DecodeProfile prof = new DecodeProfile("DS2", new String[] {
        "attn_norm", "attn(MLA)", "ffn_norm", "dense_ffn", "moe_ffn", "residual", "output" }, P_ATTN, P_MOE, P_OUTPUT);
    private final boolean cpuProfile = prof.detailed;
    private final java.util.function.Supplier<String> cacheStatsSupplier = this::getExpertCacheStats;
    private OutputRouter outputRouter;
    /** SSD-streaming expert cache (also held by {@link MoEFFN}), or null when the model is resident. */
    private it.denzosoft.llmplayer.tensor.ExpertCache expertCache;

    public DeepSeek2InferenceEngine(ModelConfig config, DeepSeek2Weights weights, int maxSeqLen,
                                     float[] ropeFreqFactors) {
        this.config = config;
        this.weights = weights;
        this.maxSeqLen = maxSeqLen;

        // RoPE for MLA: operates on ropeDimensionCount (64) dimensions
        // headSize for RoPE in MLA context = ropeDimensionCount (the rope portion)
        int ropeDim = config.ropeDimensionCount();
        RoPE.YarnParams yarnParams = null;
        if (config.ropeScalingFactor() > 1.0f) {
            yarnParams = new RoPE.YarnParams(
                config.ropeScalingFactor(), config.ropeOrigContextLength(),
                config.yarnLogMultiplier());
        }
        RoPE rope = new RoPE(ropeDim, ropeDim, maxSeqLen, config.ropeFreqBase(),
            config.ropeType(), ropeFreqFactors, yarnParams);
        this.mlaRope = rope;

        this.mlaAttention = new MLAAttention(config, rope, weights.layers());
        this.moeFFN = new MoEFFN(config);
        this.moeFFN.setProfile(prof);

        // Pre-cache norm weights
        int dim = config.embeddingLength();
        int blockCount = config.blockCount();
        this.cachedAttnNorm = new float[blockCount][];
        this.cachedFfnNorm = new float[blockCount][];
        for (int i = 0; i < blockCount; i++) {
            cachedAttnNorm[i] = RMSNorm.cacheWeights(weights.layers()[i].attnNorm(), dim);
            cachedFfnNorm[i] = RMSNorm.cacheWeights(weights.layers()[i].ffnNorm(), dim);
        }

        this.outputRouter = new OutputRouter(weights.output(), "Output");
        this.outputNormCache = new float[dim];
        for (int i = 0; i < dim; i++) {
            outputNormCache[i] = weights.outputNorm().getFloat(i);
        }
    }

    /** Attach the SSD-streaming expert cache (models larger than RAM). */
    public void setExpertCache(it.denzosoft.llmplayer.tensor.ExpertCache cache) {
        this.expertCache = cache;
        moeFFN.setExpertCache(cache);
    }

    private final RoPE mlaRope;
    // GPU-resident MLA attention (MlaAttentionCudaPass); null when unavailable or disabled
    private volatile GpuAttentionPass gpuAttention;

    /**
     * Run the MLA attention half of every layer with GPU-resident attention weights on the device,
     * with its expanded KV cache; the MoE FFN stays on the CPU. Disable with
     * {@code -Dmoe.gpu.attention=false}.
     */
    public void tryInitGpuAttention(Object bufferManager) {
        if ("false".equals(System.getProperty("moe.gpu.attention", "true"))) return;
        try {
            Class<?> cls = Class.forName("it.denzosoft.llmplayer.inference.MlaAttentionCudaPass");
            java.lang.reflect.Method isSup = cls.getMethod("isSupported", ModelConfig.class,
                it.denzosoft.llmplayer.model.DeepSeek2Weights.class);
            if (!(Boolean) isSup.invoke(null, config, weights)) return;
            gpuAttention = (GpuAttentionPass) cls.getConstructor(ModelConfig.class,
                    it.denzosoft.llmplayer.model.DeepSeek2Weights.class, bufferManager.getClass(), RoPE.class, int.class)
                .newInstance(config, weights, bufferManager, mlaRope, maxSeqLen);
        } catch (Throwable e) {
            System.err.println("MLA GPU attention: unavailable — " + GpuFailureException.describe(e));
            gpuAttention = null;
        }
    }

    /** GPU hot-expert cache for the routed experts (see GpuExpertCache). */
    public void initExpertGpuCache(Object cudaContext, long maxCacheBytes) {
        it.denzosoft.llmplayer.tensor.FloatTensor[][] ex = new it.denzosoft.llmplayer.tensor.FloatTensor[weights.layers().length][];
        for (int i = 0; i < ex.length; i++) {
            DeepSeek2LayerWeights lw = weights.layers()[i];
            if (lw.ffnGateExps() != null) ex[i] = new it.denzosoft.llmplayer.tensor.FloatTensor[] {
                lw.ffnGateExps(), lw.ffnUpExps(), lw.ffnDownExps() };
        }
        moeFFN.setGpuExpertCache(GpuExpertCache.create(cudaContext, maxCacheBytes, ex, config.expertFfnLength(),
            config.embeddingLength(), config.expertCount()));
    }

    /** Expert GPU cache statistics, or null when the cache is not active. */
    public String getExpertCacheStats() {
        GpuExpertCache c = moeFFN.gpuExpertCache();
        return c == null ? null : c.getStats();
    }

    /** The GPU expert cache, or null (for metrics). */
    public GpuExpertCache getGpuExpertCache() {
        return moeFFN.gpuExpertCache() != null ? moeFFN.gpuExpertCache() : parkedCache;
    }

    /**
     * Upload (and compile the kernels of) every GPU tensor the CPU-side code reaches through the
     * per-tensor path — shared experts, dense leading layers — by running one matmul on each. Done
     * before the expert cache is sized: lazily, on the first forward, they would be allocated after
     * the cache has taken the free VRAM, which under WDDM means shared system memory.
     */
    public void warmGpuTensors() {
        int dim = config.embeddingLength();
        int sharedFfn = config.expertSharedCount() * config.expertFfnLength();
        int ffn = config.intermediateSize();
        for (int i = 0; i < weights.layers().length; i++) {
            DeepSeek2LayerWeights lw = weights.layers()[i];
            warmGpu(lw.ffnGateShexp(), sharedFfn, dim);
            warmGpu(lw.ffnUpShexp(), sharedFfn, dim);
            warmGpu(lw.ffnDownShexp(), dim, sharedFfn);
            if (i < config.leadingDenseBlockCount()) {
                warmGpu(lw.wGate(), ffn, dim);
                warmGpu(lw.wUp(), ffn, dim);
                warmGpu(lw.wDown(), dim, ffn);
            }
        }
    }

    private static void warmGpu(FloatTensor t, int rows, int cols) {
        if (t == null || !t.isGpuResident() || rows <= 0 || cols <= 0) return;
        try {
            t.matmulParallel(new float[cols], new float[rows], rows, cols);
        } catch (RuntimeException ignored) {
            // the tensor falls back to its CPU twin on its own
        }
    }

    private GpuAttentionPass parkedAttention;
    private GpuExpertCache parkedCache;

    /** Park (or restore) the GPU attention pass (placement calibrator); see Qwen3MoEInferenceEngine. */
    public synchronized void setGpuAttentionParked(boolean park) {
        if (park && gpuAttention != null) { parkedAttention = gpuAttention; gpuAttention = null; }
        else if (!park && parkedAttention != null) { gpuAttention = parkedAttention; parkedAttention = null; }
    }

    public boolean hasParkedAttention() { return parkedAttention != null; }

    /** Park (or restore) the GPU expert cache (placement calibrator). */
    public synchronized void setExpertGpuCacheParked(boolean park) {
        if (park && moeFFN.gpuExpertCache() != null) { parkedCache = moeFFN.gpuExpertCache(); moeFFN.setGpuExpertCache(null); }
        else if (!park && parkedCache != null) { moeFFN.setGpuExpertCache(parkedCache); parkedCache = null; }
    }

    /** True when a GPU-resident attention pass (which owns the device KV cache) is active. */
    public boolean hasGpuForwardPass() {
        return gpuAttention != null;
    }

    private void gpuAttentionFailed(RuntimeException e, int position, int layer) {
        GpuAttentionPass failed = gpuAttention;
        gpuAttention = null;
        System.err.println("MLA GPU attention failed, disabling it — " + GpuFailureException.describe(e));
        try { failed.close(); } catch (Exception ignored) { }
        it.denzosoft.llmplayer.gpu.GpuActivity.gpuPathDisabled();
        // Only the very first GPU call of a sequence can move to the CPU (see Qwen3MoEInferenceEngine)
        if (position > 0 || layer > 0) {
            throw new GpuFailureException("MLA GPU attention failed at layer " + layer + ", position " + position, e);
        }
    }

    public DeepSeek2State createState(int maxSeqLen) {
        return new DeepSeek2State(config, maxSeqLen);
    }

    public float[] forward(DeepSeek2State state, int token, int position) {
        return forwardInternal(state, token, position, true);
    }

    public void forwardNoOutput(DeepSeek2State state, int token, int position) {
        forwardInternal(state, token, position, false);
    }

    private float[] forwardInternal(DeepSeek2State state, int token, int position, boolean computeLogits) {
        embedToken(state, token);
        for (int layer = 0; layer < config.blockCount(); layer++) {
            forwardLayer(state, layer, position);
        }
        if (!computeLogits) return null;
        return outputProjection(state);
    }

    /** Tokens per layer-outer prefill chunk; see {@link #forwardPrefill}. */
    private static final int PREFILL_BATCH = Integer.getInteger("prefill.batch", 64);
    private static final boolean PREFILL_BATCHED =
        !"false".equals(System.getProperty("prefill.batched", "true"));

    /**
     * Prefill positions {@code fromPos..toPos-1} with the layers in the outer loop, returning the
     * logits for the last position. Same rationale and same guarantees as
     * {@code Qwen3MoEInferenceEngine.forwardPrefill}: identical arithmetic in an identical
     * per-(layer, token) order, so the output is bit-identical, but the expert working set collapses
     * from {@code layers × top-K} to the union selected by one chunk at one layer — which is what
     * makes prefill affordable when the experts are streamed from SSD.
     *
     * Disable with {@code -Dprefill.batched=false}; chunk size via {@code -Dprefill.batch}.
     */
    // F4: whether any layer weight is GPU-resident (scanned once). The batched CPU prefill used to
    // be gated on the matmul pool, which GPU init switched off; the pool now stays on.
    private volatile Boolean layersGpuResident;

    private boolean layersGpuResident() {
        Boolean b = layersGpuResident;
        if (b == null) layersGpuResident = b = it.denzosoft.llmplayer.tensor.FloatTensor.anyGpuResident(weights.layers());
        // GPU matmuls switched off (the placement calibrator's CPU candidate): the tensors run
        // on their CPU twins, so the batched prefill applies as in a CPU-only run
        return b && it.denzosoft.llmplayer.tensor.FloatTensor.gpuMatmulEnabled();
    }

    public float[] forwardPrefill(DeepSeek2State state, int[] tokens, int fromPos, int toPos) {
        int count = toPos - fromPos;
        if (count <= 0) return null;
        prof.startGeneration();
        if (!PREFILL_BATCHED || count == 1) {
            float[] logits = null;
            for (int i = fromPos; i < toPos; i++) {
                if (i < toPos - 1) forwardInternal(state, tokens[i], i, false);
                else logits = forwardInternal(state, tokens[i], i, true);
            }
            return logits;
        }
        // Batched prefill: CPU layers run MLA against the CPU KV cache, GPU-resident layers the
        // pass's batched attention (their KV lives on the device); the routed experts run batched
        // on the CPU. A pass without batched attention keeps the per-token path below.
        GpuAttentionPass gpu = gpuAttention;
        // Without an attention pass, a first-N placement (explicit --gpu-layers) keeps its GPU
        // layers on the per-token path (F4 residency gate).
        if (MoEFFN.batchAvailable() && (gpu == null ? !layersGpuResident() : gpu.maxBatchTokens() > 0)) {
            return prefillBatched(state, tokens, fromPos, toPos);
        }
        // Per-token path (GPU mode): warm the decode kernels here too, see Qwen3MoEInferenceEngine
        warmDecodeKernels();

        int dim = config.embeddingLength();
        int blockCount = config.blockCount();
        for (int base = fromPos; base < toPos; base += PREFILL_BATCH) {
            int n = Math.min(PREFILL_BATCH, toPos - base);
            float[][] xs = new float[n][];
            for (int t = 0; t < n; t++) {
                embedToken(state, tokens[base + t]);
                xs[t] = state.x.clone();
            }
            for (int layer = 0; layer < blockCount; layer++) {
                for (int t = 0; t < n; t++) {
                    System.arraycopy(xs[t], 0, state.x, 0, dim);
                    forwardLayer(state, layer, base + t);
                    System.arraycopy(state.x, 0, xs[t], 0, dim);
                }
            }
            System.arraycopy(xs[n - 1], 0, state.x, 0, dim);
        }
        return outputProjection(state);
    }

    // Indices into DeepSeek2State.prefillBuffers
    private static final int B_X = 0, B_XN = 1, B_XB = 2, B_HB = 3, B_HB2 = 4;

    /**
     * Layer-outer prefill with multi-token matmuls, on the CPU path: per chunk and layer, MLA
     * ({@link MLAAttention#forwardBatch}) and the MoE FFN ({@link MoEFFN#forwardBatch}) batch every
     * projection over the chunk — each routed expert runs once over all the tokens routed to it —
     * while attention runs token by token in position order. Disable with {@code -Dmoe.expert.rows=false}
     * (layer-outer order with one-token matmuls) or {@code -Dprefill.batched=false} (token by token).
     */
    private float[] prefillBatched(DeepSeek2State state, int[] tokens, int fromPos, int toPos) {
        warmDecodeKernels();
        int dim = config.embeddingLength();
        int cap = ExpertViews.prefillChunk(expertCache, PREFILL_BATCH);
        GpuAttentionPass gpu0 = gpuAttention;
        if (gpu0 != null) cap = Math.min(cap, gpu0.maxBatchTokens());
        if (state.prefillBuffers == null || state.prefillBuffers[B_X].length < cap) {
            int ffn = Math.max(1, config.intermediateSize());
            state.prefillBuffers = new float[][][] {
                new float[cap][dim], new float[cap][dim], new float[cap][dim], new float[cap][ffn], new float[cap][ffn]
            };
        }
        float[][][] b = state.prefillBuffers;
        float[][] x = b[B_X], xn = b[B_XN], xb = b[B_XB];
        for (int base = fromPos; base < toPos; base += cap) {
            int n = Math.min(cap, toPos - base);
            for (int t = 0; t < n; t++) {
                embedToken(state, tokens[base + t]);
                System.arraycopy(state.x, 0, x[t], 0, dim);
            }
            for (int layer = 0; layer < config.blockCount(); layer++) {
                DeepSeek2LayerWeights lw = weights.layers()[layer];
                boolean onGpu = false;
                GpuAttentionPass gpu = gpuAttention;
                if (gpu != null && gpu.isLayerOnGpu(layer)) {
                    try {
                        gpu.attentionLayerBatch(layer, x, xn, base, n); // x += attn, xn = ffnNorm(x)
                        onGpu = true;
                    } catch (RuntimeException e) {
                        gpuAttentionFailed(e, base, layer); // throws unless it is the first GPU call
                    }
                }
                if (!onGpu) {
                    for (int t = 0; t < n; t++) {
                        RMSNorm.apply(xn[t], x[t], cachedAttnNorm[layer], dim, config.normEps());
                    }
                    mlaAttention.forwardBatch(state, lw, layer, base, n, xn, xb);
                    for (int t = 0; t < n; t++) {
                        VectorOpsFactory.get().accumulate(x[t], xb[t], dim);
                        RMSNorm.apply(xn[t], x[t], cachedFfnNorm[layer], dim, config.normEps());
                    }
                }
                if (layer < config.leadingDenseBlockCount()) {
                    denseFFNBatch(b, lw, n);
                } else {
                    moeFFN.forwardBatch(state, lw, layer, n, xn, xb);
                }
                for (int t = 0; t < n; t++) {
                    VectorOpsFactory.get().accumulate(x[t], xb[t], dim);
                }
            }
            if (base + n == toPos) System.arraycopy(x[n - 1], 0, state.x, 0, dim);
        }
        return outputProjection(state);
    }

    /** Multi-token {@link #denseFFN}: {@code xb[t] = down(silu(gate(xn[t])) * up(xn[t]))}. */
    private void denseFFNBatch(float[][][] b, DeepSeek2LayerWeights lw, int n) {
        int dim = config.embeddingLength();
        int ffn = config.intermediateSize();
        float[][] hb = b[B_HB], hb2 = b[B_HB2];
        for (int t = 0; t < n; t++) {
            Arrays.fill(hb[t], 0, ffn, 0f);
            Arrays.fill(hb2[t], 0, ffn, 0f);
        }
        FloatTensor.fusedGateUpBatchParallel(lw.wGate(), lw.wUp(), b[B_XN], hb, hb2, n, ffn, dim);
        for (int t = 0; t < n; t++) {
            VectorOpsFactory.get().silu(hb[t], ffn);
            VectorOpsFactory.get().elementwiseMul(hb[t], hb2[t], hb[t], ffn);
            Arrays.fill(b[B_XB][t], 0, dim, 0f);
        }
        FloatTensor.matmulBatchParallel(lw.wDown(), hb, b[B_XB], n, dim, ffn);
    }

    /** See {@link FloatTensor#warmUpRows}: batched prefill skips the kernels decode will use. */
    private void warmDecodeKernels() {
        int dim = config.embeddingLength();
        int ffn = config.intermediateSize();
        for (int layer = 0; layer < config.blockCount(); layer++) {
            DeepSeek2LayerWeights lw = weights.layers()[layer];
            mlaAttention.warmUp(lw);
            if (layer < config.leadingDenseBlockCount()) {
                FloatTensor.warmUpRows(lw.wGate(), ffn, dim);
                FloatTensor.warmUpRows(lw.wUp(), ffn, dim);
                FloatTensor.warmUpRows(lw.wDown(), dim, ffn);
            } else {
                moeFFN.warmUp(lw);
            }
        }
        FloatTensor.warmUpRows(weights.output(), config.vocabSize(), dim);
    }

    /** Load a token's embedding into the residual stream. */
    private void embedToken(DeepSeek2State state, int token) {
        int dim = config.embeddingLength();
        for (int i = 0; i < dim; i++) {
            state.x[i] = weights.tokenEmbedding().getFloat((long) token * dim + i);
        }
    }

    /** One transformer block (MLA attention + dense or MoE FFN), reading and writing {@code state.x}. */
    private void forwardLayer(DeepSeek2State state, int layer, int position) {
        int dim = config.embeddingLength();
        int leadingDenseCount = config.leadingDenseBlockCount();
        final boolean d = cpuProfile;
        DeepSeek2LayerWeights layerWeights = weights.layers()[layer];
        long t0 = System.nanoTime(), t1;

        GpuAttentionPass gpu = gpuAttention;
        if (gpu != null && gpu.isLayerOnGpu(layer)) {
            // Whole MLA half on the GPU: x += Attn(norm(x)), xb = ffnNorm(x)
            try {
                gpu.attentionLayer(layer, state.x, state.xb, position);
            } catch (RuntimeException e) {
                gpuAttentionFailed(e, position, layer);
                forwardLayer(state, layer, position);
                return;
            }
            t1 = System.nanoTime(); prof.add(P_ATTN, t1 - t0); t0 = t1;
        } else {
            RMSNorm.apply(state.xb, state.x, cachedAttnNorm[layer], dim, config.normEps());
            if (d) { t1 = System.nanoTime(); prof.add(P_ATTN_NORM, t1 - t0); t0 = t1; }

            mlaAttention.forward(state, layerWeights, layer, position);
            t1 = System.nanoTime(); prof.add(P_ATTN, t1 - t0); t0 = t1;

            VectorOpsFactory.get().accumulate(state.x, state.xb, dim);
            if (d) { t1 = System.nanoTime(); prof.add(P_RESIDUAL, t1 - t0); t0 = t1; }

            RMSNorm.apply(state.xb, state.x, cachedFfnNorm[layer], dim, config.normEps());
            if (d) { t1 = System.nanoTime(); prof.add(P_FFN_NORM, t1 - t0); t0 = t1; }
        }

        if (layer < leadingDenseCount) {
            denseFFN(state, layerWeights);
            if (d) { t1 = System.nanoTime(); prof.add(P_DENSE, t1 - t0); t0 = t1; }
        } else {
            System.arraycopy(state.xb, 0, state.xbSaved, 0, dim);
            GpuExpertCache gc = moeFFN.gpuExpertCache();
            if (gc != null) gc.noteToken(position);
            GpuAttentionPass g = gpuAttention;
            GpuAttentionPass sharedSource = g != null && g.isLayerOnGpu(layer) ? g : null;
            try {
                moeFFN.forward(state, layerWeights, layer, sharedSource);
            } catch (RuntimeException e) {
                if (sharedSource == null) throw e;
                // The shared expert's download failed: a GPU failure of the pass. Past position 0 /
                // layer 0 this throws GpuFailureException; otherwise redo the FFN on the CPU.
                System.arraycopy(state.xbSaved, 0, state.xb, 0, dim);
                gpuAttentionFailed(e, position, layer);
                moeFFN.forward(state, layerWeights, layer, (GpuAttentionPass) null);
            }
            t1 = System.nanoTime(); prof.add(P_MOE, t1 - t0); t0 = t1;
        }

        VectorOpsFactory.get().accumulate(state.x, state.xb, dim);
        if (d) prof.add(P_RESIDUAL, System.nanoTime() - t0);
    }

    /** Final norm + logit projection over the current residual stream. */
    private float[] outputProjection(DeepSeek2State state) {
        int dim = config.embeddingLength();
        long t0 = System.nanoTime();
        RMSNorm.apply(state.xb, state.x, outputNormCache, dim, config.normEps());
        int vocabSize = config.vocabSize();
        Arrays.fill(state.logits, 0);
        outputRouter.matmul(state.xb, state.logits, vocabSize, dim);
        prof.endToken(System.nanoTime() - t0, cacheStatsSupplier); // drops the prompt at the first call
        return state.logits;
    }

    /**
     * Dense SwiGLU FFN for leading dense blocks.
     */
    private void denseFFN(DeepSeek2State state, DeepSeek2LayerWeights weights) {
        int dim = config.embeddingLength();
        int ffnDim = config.intermediateSize();

        Arrays.fill(state.hb, 0, ffnDim, 0f);
        weights.wGate().matmulParallel(state.xb, state.hb, ffnDim, dim);

        Arrays.fill(state.hb2, 0, ffnDim, 0f);
        weights.wUp().matmulParallel(state.xb, state.hb2, ffnDim, dim);

        VectorOpsFactory.get().silu(state.hb, ffnDim);
        VectorOpsFactory.get().elementwiseMul(state.hb, state.hb2, state.hb, ffnDim);

        Arrays.fill(state.xb, 0);
        weights.wDown().matmulParallel(state.hb, state.xb, dim, ffnDim);
    }

    public float[] prefill(DeepSeek2State state, int[] tokens) {
        long t0 = System.currentTimeMillis();
        for (int i = 0; i < tokens.length - 1; i++) {
            forwardNoOutput(state, tokens[i], i);
            long elapsed = System.currentTimeMillis() - t0;
            System.out.printf("[prefill] token %d/%d (%.1fs)%n", i + 1, tokens.length, elapsed / 1000.0);
        }
        float[] logits = forward(state, tokens[tokens.length - 1], tokens.length - 1);
        long total = System.currentTimeMillis() - t0;
        System.out.printf("[prefill] done: %d tokens in %.1fs%n", tokens.length, total / 1000.0);
        return logits;
    }

    public ModelConfig getConfig() { return config; }
}
