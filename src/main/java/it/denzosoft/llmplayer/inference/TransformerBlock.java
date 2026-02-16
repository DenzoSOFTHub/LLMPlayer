package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.model.TransformerLayerWeights;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

/**
 * A single transformer block supporting three normalization modes:
 * 1. Pre-norm (standard): attn_norm -> attention -> residual -> ffn_norm -> ffn -> residual
 * 2. Pre+Post-norm (GLM4/Gemma2): adds post-attention and post-FFN norms
 * 3. Post-norm only (OLMo2): attention -> post_attn_norm -> residual -> ffn -> post_ffn_norm -> residual
 * 4. Parallel FFN (Command-R): attn_norm -> {attention || ffn} -> residual (no ffn_norm)
 */
public class TransformerBlock {

    private final ModelConfig config;
    private final Attention attention;
    private final SwiGLUFFN ffn;
    private final float[][] cachedAttnNorm;
    private final float[][] cachedFfnNorm;
    private final float[][] cachedPostAttnNorm; // null entries if not used
    private final float[][] cachedPostFfnNorm;  // null entries if not used

    public TransformerBlock(ModelConfig config, Attention attention, SwiGLUFFN ffn,
                            TransformerLayerWeights[] allLayers) {
        this.config = config;
        this.attention = attention;
        this.ffn = ffn;

        int dim = config.embeddingLength();
        int blockCount = allLayers.length;
        this.cachedAttnNorm = new float[blockCount][];
        this.cachedFfnNorm = new float[blockCount][];
        this.cachedPostAttnNorm = new float[blockCount][];
        this.cachedPostFfnNorm = new float[blockCount][];
        for (int i = 0; i < blockCount; i++) {
            if (allLayers[i].attnNorm() != null) {
                cachedAttnNorm[i] = RMSNorm.cacheWeights(allLayers[i].attnNorm(), dim);
            }
            if (allLayers[i].ffnNorm() != null) {
                cachedFfnNorm[i] = RMSNorm.cacheWeights(allLayers[i].ffnNorm(), dim);
            }
            if (allLayers[i].postAttnNorm() != null) {
                cachedPostAttnNorm[i] = RMSNorm.cacheWeights(allLayers[i].postAttnNorm(), dim);
            }
            if (allLayers[i].postFfnNorm() != null) {
                cachedPostFfnNorm[i] = RMSNorm.cacheWeights(allLayers[i].postFfnNorm(), dim);
            }
        }
    }

    public void forward(InferenceState state, TransformerLayerWeights weights, int layer, int position) {
        int dim = config.embeddingLength();

        // Parallel FFN mode (Command-R): no ffn_norm, FFN uses same pre-norm as attention
        boolean parallelFfn = cachedFfnNorm[layer] == null && cachedAttnNorm[layer] != null;

        // Pre-attention norm (standard/GLM4/Gemma2/Command-R) or pass-through (OLMo2)
        if (cachedAttnNorm[layer] != null) {
            RMSNorm.apply(state.xb, state.x, cachedAttnNorm[layer], dim, config.normEps());
        } else {
            System.arraycopy(state.x, 0, state.xb, 0, dim);
        }

        // Save pre-norm output for parallel FFN (reuse logits buffer, always large enough)
        if (parallelFfn) {
            System.arraycopy(state.xb, 0, state.logits, 0, dim);
        }

        // Attention (writes to xb via output projection)
        attention.forward(state, weights, layer, position);

        // Post-attention norm (GLM4/Gemma2/OLMo2)
        if (cachedPostAttnNorm[layer] != null) {
            RMSNorm.apply(state.xb, state.xb, cachedPostAttnNorm[layer], dim, config.normEps());
        }

        // Residual connection: x += attention_output
        VectorOpsFactory.get().accumulate(state.x, state.xb, dim);

        // FFN input: depends on mode
        if (parallelFfn) {
            // Restore pre-norm output for FFN (Command-R parallel mode)
            System.arraycopy(state.logits, 0, state.xb, 0, dim);
        } else if (cachedFfnNorm[layer] != null) {
            // Standard pre-FFN norm
            RMSNorm.apply(state.xb, state.x, cachedFfnNorm[layer], dim, config.normEps());
        } else {
            // Post-norm only (OLMo2): FFN takes x directly
            System.arraycopy(state.x, 0, state.xb, 0, dim);
        }

        // FFN (writes to xb)
        ffn.forward(state, weights);

        // Post-FFN norm (GLM4/Gemma2/OLMo2)
        if (cachedPostFfnNorm[layer] != null) {
            RMSNorm.apply(state.xb, state.xb, cachedPostFfnNorm[layer], dim, config.normEps());
        }

        // Residual connection: x += ffn_output
        VectorOpsFactory.get().accumulate(state.x, state.xb, dim);
    }
}
