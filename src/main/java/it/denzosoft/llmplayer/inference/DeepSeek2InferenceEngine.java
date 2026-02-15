package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.DeepSeek2LayerWeights;
import it.denzosoft.llmplayer.model.DeepSeek2Weights;
import it.denzosoft.llmplayer.model.ModelConfig;
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

    public DeepSeek2State createState(int maxSeqLen) {
        return new DeepSeek2State(config, maxSeqLen);
    }

    public float[] forward(DeepSeek2State state, int token, int position) {
        int dim = config.embeddingLength();
        int vocabSize = config.vocabSize();
        int leadingDenseCount = config.leadingDenseBlockCount();

        // 1. Token embedding lookup
        for (int i = 0; i < dim; i++) {
            state.x[i] = weights.tokenEmbedding().getFloat((long) token * dim + i);
        }

        // 2. Forward through all layers
        for (int layer = 0; layer < config.blockCount(); layer++) {
            DeepSeek2LayerWeights layerWeights = weights.layers()[layer];

            // Attention norm
            RMSNorm.apply(state.xb, state.x, cachedAttnNorm[layer], dim, config.normEps());

            // MLA Attention
            mlaAttention.forward(state, layerWeights, layer, position);

            // Residual
            VectorOpsFactory.get().accumulate(state.x, state.xb, dim);

            // FFN norm
            RMSNorm.apply(state.xb, state.x, cachedFfnNorm[layer], dim, config.normEps());

            if (layer < leadingDenseCount) {
                // Dense SwiGLU FFN
                denseFFN(state, layerWeights);
            } else {
                // Save xb before MoE (since MoE uses xb as accumulator)
                System.arraycopy(state.xb, 0, state.xbSaved, 0, dim);
                // MoE FFN
                moeFFN.forward(state, layerWeights);
            }

            // Residual
            VectorOpsFactory.get().accumulate(state.x, state.xb, dim);
        }

        // 3. Final RMSNorm
        RMSNorm.apply(state.xb, state.x, outputNormCache, dim, config.normEps());

        // 4. Output projection
        Arrays.fill(state.logits, 0);
        weights.output().matmulParallel(state.xb, state.logits, vocabSize, dim);

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
        float[] logits = null;
        long t0 = System.currentTimeMillis();
        for (int i = 0; i < tokens.length; i++) {
            logits = forward(state, tokens[i], i);
            long elapsed = System.currentTimeMillis() - t0;
            System.out.printf("[prefill] token %d/%d (%.1fs)%n", i + 1, tokens.length, elapsed / 1000.0);
        }
        long total = System.currentTimeMillis() - t0;
        System.out.printf("[prefill] done: %d tokens in %.1fs%n", tokens.length, total / 1000.0);
        return logits;
    }

    public ModelConfig getConfig() { return config; }
}
