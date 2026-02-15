package it.denzosoft.llmplayer.model;

import it.denzosoft.llmplayer.tensor.FloatTensor;

public final class DeepSeek2Weights {
    private final FloatTensor tokenEmbedding;
    private final FloatTensor outputNorm;
    private final FloatTensor output;
    private final DeepSeek2LayerWeights[] layers;
    private final float[] ropeFreqFactors;

    public DeepSeek2Weights(FloatTensor tokenEmbedding, FloatTensor outputNorm, FloatTensor output,
                            DeepSeek2LayerWeights[] layers, float[] ropeFreqFactors) {
        this.tokenEmbedding = tokenEmbedding;
        this.outputNorm = outputNorm;
        this.output = output;
        this.layers = layers;
        this.ropeFreqFactors = ropeFreqFactors;
    }

    public FloatTensor tokenEmbedding() { return tokenEmbedding; }
    public FloatTensor outputNorm() { return outputNorm; }
    public FloatTensor output() { return output; }
    public DeepSeek2LayerWeights[] layers() { return layers; }
    public float[] ropeFreqFactors() { return ropeFreqFactors; }
}
