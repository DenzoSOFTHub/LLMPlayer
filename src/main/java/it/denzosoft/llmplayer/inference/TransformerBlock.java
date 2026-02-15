package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.model.TransformerLayerWeights;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

/**
 * A single transformer block: attention-norm -> attention -> [post-attn-norm] -> residual -> ffn-norm -> ffn -> [post-ffn-norm] -> residual
 * Post-norms are optional (used by GLM4).
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
            cachedAttnNorm[i] = RMSNorm.cacheWeights(allLayers[i].attnNorm(), dim);
            cachedFfnNorm[i] = RMSNorm.cacheWeights(allLayers[i].ffnNorm(), dim);
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

        // Attention norm
        RMSNorm.apply(state.xb, state.x, cachedAttnNorm[layer], dim, config.normEps());

        // Attention (writes to xb via output projection)
        attention.forward(state, weights, layer, position);

        // Post-attention norm (GLM4)
        if (cachedPostAttnNorm[layer] != null) {
            RMSNorm.apply(state.xb, state.xb, cachedPostAttnNorm[layer], dim, config.normEps());
        }

        // Residual connection: x += attention_output (which is in xb)
        VectorOpsFactory.get().accumulate(state.x, state.xb, dim);

        // FFN norm
        RMSNorm.apply(state.xb, state.x, cachedFfnNorm[layer], dim, config.normEps());

        // FFN (writes to xb)
        ffn.forward(state, weights);

        // Post-FFN norm (GLM4)
        if (cachedPostFfnNorm[layer] != null) {
            RMSNorm.apply(state.xb, state.xb, cachedPostFfnNorm[layer], dim, config.normEps());
        }

        // Residual connection: x += ffn_output
        VectorOpsFactory.get().accumulate(state.x, state.xb, dim);
    }
}
