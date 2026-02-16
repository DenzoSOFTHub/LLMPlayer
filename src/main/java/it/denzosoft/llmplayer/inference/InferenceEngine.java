package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.ModelArchitecture;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.model.ModelWeights;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

import java.util.Arrays;

/**
 * Orchestrates the complete forward pass of the transformer model.
 *
 * Forward pass pipeline:
 * 1. Token embedding lookup
 * 2. For each layer: RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual
 * 3. Final RMSNorm
 * 4. Output projection -> logits
 */
public class InferenceEngine {

    private final ModelConfig config;
    private final ModelWeights weights;
    private final TransformerBlock block;
    private final float[] normWeightCache;
    private final int maxSeqLen;
    private final float finalLogitSoftCap;
    private final float logitScale;
    private final float embeddingScale;

    public InferenceEngine(ModelConfig config, ModelWeights weights, int maxSeqLen) {
        this(config, weights, maxSeqLen, null);
    }

    public InferenceEngine(ModelConfig config, ModelWeights weights, int maxSeqLen, float[] ropeFreqFactors) {
        this.config = config;
        this.weights = weights;
        this.maxSeqLen = maxSeqLen;

        int ropeDimCount = config.ropeDimensionCount();
        RoPE rope = new RoPE(config.headSize(), ropeDimCount, maxSeqLen, config.ropeFreqBase(),
            config.ropeType(), ropeFreqFactors);
        Attention attention = new Attention(config, rope);
        attention.initNormCachesIfNeeded(weights.layers());
        SwiGLUFFN ffn = new SwiGLUFFN(config);
        this.block = new TransformerBlock(config, attention, ffn, weights.layers());

        this.finalLogitSoftCap = config.finalLogitSoftCap();
        this.logitScale = config.logitScale();

        // Gemma models scale embeddings by sqrt(dim)
        ModelArchitecture arch = config.architecture();
        if (arch == ModelArchitecture.GEMMA2 || arch == ModelArchitecture.GEMMA3) {
            this.embeddingScale = (float) Math.sqrt(config.embeddingLength());
        } else {
            this.embeddingScale = 0f;
        }

        // Pre-cache output norm weights
        int dim = config.embeddingLength();
        this.normWeightCache = new float[dim];
        for (int i = 0; i < dim; i++) {
            normWeightCache[i] = weights.outputNorm().getFloat(i);
        }
    }

    /**
     * Create a new inference state for this model.
     */
    public InferenceState createState(int maxSeqLen) {
        return new InferenceState(config, maxSeqLen);
    }

    /**
     * Run forward pass for a single token at the given position.
     * Returns logits array (part of InferenceState, do not modify externally).
     */
    public float[] forward(InferenceState state, int token, int position) {
        int dim = config.embeddingLength();
        int vocabSize = config.vocabSize();

        // 1. Token embedding lookup
        for (int i = 0; i < dim; i++) {
            state.x[i] = weights.tokenEmbedding().getFloat((long) token * dim + i);
        }

        // Gemma: scale embedding by sqrt(dim)
        if (embeddingScale > 0f) {
            for (int i = 0; i < dim; i++) {
                state.x[i] *= embeddingScale;
            }
        }

        // 2. Forward through all transformer layers
        for (int layer = 0; layer < config.blockCount(); layer++) {
            block.forward(state, weights.layers()[layer], layer, position);
        }

        // 3. Final RMSNorm
        VectorOpsFactory.get().rmsnorm(state.xb, state.x, normWeightCache, dim, config.normEps());

        // 4. Output projection: logits = output_weight * xb
        Arrays.fill(state.logits, 0);
        weights.output().matmulParallel(state.xb, state.logits, vocabSize, dim);

        // 5. Logit scaling (Command-R): logits *= logitScale
        if (logitScale > 0f) {
            for (int i = 0; i < vocabSize; i++) {
                state.logits[i] *= logitScale;
            }
        }

        // 6. Logit soft-capping (Gemma2/3): logits = softCap * tanh(logits / softCap)
        if (finalLogitSoftCap > 0f) {
            for (int i = 0; i < vocabSize; i++) {
                state.logits[i] = finalLogitSoftCap * (float) Math.tanh(state.logits[i] / finalLogitSoftCap);
            }
        }

        return state.logits;
    }

    /**
     * Prefill: process multiple tokens (prompt) and return logits for the last token.
     */
    public float[] prefill(InferenceState state, int[] tokens) {
        float[] logits = null;
        for (int i = 0; i < tokens.length; i++) {
            logits = forward(state, tokens[i], i);
        }
        return logits;
    }

    /**
     * Run forward pass with a limited number of layers (for debugging).
     */
    public float[] forwardLayers(InferenceState state, int token, int position, int numLayers) {
        int dim = config.embeddingLength();
        int vocabSize = config.vocabSize();

        for (int i = 0; i < dim; i++) {
            state.x[i] = weights.tokenEmbedding().getFloat((long) token * dim + i);
        }

        for (int layer = 0; layer < numLayers; layer++) {
            block.forward(state, weights.layers()[layer], layer, position);
        }

        VectorOpsFactory.get().rmsnorm(state.xb, state.x, normWeightCache, dim, config.normEps());
        Arrays.fill(state.logits, 0);
        weights.output().matmulParallel(state.xb, state.logits, vocabSize, dim);

        return state.logits;
    }

    public ModelConfig getConfig() { return config; }
}
