package it.denzosoft.llmplayer.model;

import it.denzosoft.llmplayer.tensor.FloatTensor;

/**
 * Top-level weights container for Qwen3 MoE models.
 */
public final class Qwen3MoEWeights {
    private final FloatTensor tokenEmbedding;
    private final FloatTensor outputNorm;
    private final FloatTensor output;
    private final Qwen3MoELayerWeights[] layers;
    private final float[] ropeFreqFactors;

    public Qwen3MoEWeights(FloatTensor tokenEmbedding, FloatTensor outputNorm, FloatTensor output,
                            Qwen3MoELayerWeights[] layers, float[] ropeFreqFactors) {
        this.tokenEmbedding = tokenEmbedding;
        this.outputNorm = outputNorm;
        this.output = output;
        this.layers = layers;
        this.ropeFreqFactors = ropeFreqFactors;
    }

    public FloatTensor tokenEmbedding() { return tokenEmbedding; }
    public FloatTensor outputNorm() { return outputNorm; }
    public FloatTensor output() { return output; }
    public Qwen3MoELayerWeights[] layers() { return layers; }
    public float[] ropeFreqFactors() { return ropeFreqFactors; }
}
