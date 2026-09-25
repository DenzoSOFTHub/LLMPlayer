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

    private final boolean cpuProfile;
    /** SSD-streaming expert cache (also held by {@link MoEFFN}), or null when the model is resident. */
    private it.denzosoft.llmplayer.tensor.ExpertCache expertCache;
    private long profAttnNormNs, profAttnNs, profFfnNormNs, profDenseFfnNs, profMoeFfnNs, profResidualNs, profOutputNs;
    private int profTokenCount;

    public DeepSeek2InferenceEngine(ModelConfig config, DeepSeek2Weights weights, int maxSeqLen,
                                     float[] ropeFreqFactors) {
        this.cpuProfile = "true".equals(System.getProperty("cpu.profile"));
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

        this.mlaAttention = new MLAAttention(config, rope, weights.layers());
        this.moeFFN = new MoEFFN(config);

        // Pre-cache norm weights
        int dim = config.embeddingLength();
        int blockCount = config.blockCount();
        this.cachedAttnNorm = new float[blockCount][];
        this.cachedFfnNorm = new float[blockCount][];
        for (int i = 0; i < blockCount; i++) {
            cachedAttnNorm[i] = RMSNorm.cacheWeights(weights.layers()[i].attnNorm(), dim);
            cachedFfnNorm[i] = RMSNorm.cacheWeights(weights.layers()[i].ffnNorm(), dim);
        }

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
    public float[] forwardPrefill(DeepSeek2State state, int[] tokens, int fromPos, int toPos) {
        int count = toPos - fromPos;
        if (count <= 0) return null;
        if (!PREFILL_BATCHED || count == 1) {
            float[] logits = null;
            for (int i = fromPos; i < toPos; i++) {
                if (i < toPos - 1) forwardInternal(state, tokens[i], i, false);
                else logits = forwardInternal(state, tokens[i], i, true);
            }
            return logits;
        }
        if (MoEFFN.batchAvailable()) return prefillBatched(state, tokens, fromPos, toPos);

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
                for (int t = 0; t < n; t++) {
                    RMSNorm.apply(xn[t], x[t], cachedAttnNorm[layer], dim, config.normEps());
                }
                mlaAttention.forwardBatch(state, lw, layer, base, n, xn, xb);
                for (int t = 0; t < n; t++) {
                    VectorOpsFactory.get().accumulate(x[t], xb[t], dim);
                    RMSNorm.apply(xn[t], x[t], cachedFfnNorm[layer], dim, config.normEps());
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
        long t0 = 0, t1;
        {
            DeepSeek2LayerWeights layerWeights = weights.layers()[layer];

            if (cpuProfile) t0 = System.nanoTime();
            RMSNorm.apply(state.xb, state.x, cachedAttnNorm[layer], dim, config.normEps());
            if (cpuProfile) { t1 = System.nanoTime(); profAttnNormNs += t1 - t0; t0 = t1; }

            mlaAttention.forward(state, layerWeights, layer, position);
            if (cpuProfile) { t1 = System.nanoTime(); profAttnNs += t1 - t0; t0 = t1; }

            VectorOpsFactory.get().accumulate(state.x, state.xb, dim);
            if (cpuProfile) { t1 = System.nanoTime(); profResidualNs += t1 - t0; t0 = t1; }

            RMSNorm.apply(state.xb, state.x, cachedFfnNorm[layer], dim, config.normEps());
            if (cpuProfile) { t1 = System.nanoTime(); profFfnNormNs += t1 - t0; t0 = t1; }

            if (layer < leadingDenseCount) {
                denseFFN(state, layerWeights);
                if (cpuProfile) { t1 = System.nanoTime(); profDenseFfnNs += t1 - t0; t0 = t1; }
            } else {
                System.arraycopy(state.xb, 0, state.xbSaved, 0, dim);
                moeFFN.forward(state, layerWeights, layer);
                if (cpuProfile) { t1 = System.nanoTime(); profMoeFfnNs += t1 - t0; t0 = t1; }
            }

            VectorOpsFactory.get().accumulate(state.x, state.xb, dim);
            if (cpuProfile) { t1 = System.nanoTime(); profResidualNs += t1 - t0; }
        }
    }

    /** Final norm + logit projection over the current residual stream. */
    private float[] outputProjection(DeepSeek2State state) {
        int dim = config.embeddingLength();
        long t0 = 0;
        if (cpuProfile) t0 = System.nanoTime();
        RMSNorm.apply(state.xb, state.x, outputNormCache, dim, config.normEps());
        int vocabSize = config.vocabSize();
        Arrays.fill(state.logits, 0);
        weights.output().matmulParallel(state.xb, state.logits, vocabSize, dim);
        if (cpuProfile) {
            profOutputNs += System.nanoTime() - t0;
            profTokenCount++;
            if (profTokenCount % 10 == 0) printProfile();
        }

        return state.logits;
    }

    private void printProfile() {
        int n = profTokenCount;
        double ms = 1e6;
        long total = profAttnNormNs + profAttnNs + profFfnNormNs + profDenseFfnNs + profMoeFfnNs + profResidualNs + profOutputNs;
        System.out.printf("[cpu-profile DS2] %d tokens, per-token avg (ms): attn_norm=%.1f attn(MLA)=%.1f ffn_norm=%.1f dense_ffn=%.1f moe_ffn=%.1f residual=%.1f output=%.1f | total=%.1f%n",
            n, profAttnNormNs / ms / n, profAttnNs / ms / n, profFfnNormNs / ms / n,
            profDenseFfnNs / ms / n, profMoeFfnNs / ms / n, profResidualNs / ms / n,
            profOutputNs / ms / n, total / ms / n);
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
