package it.denzosoft.llmplayer.model;

import it.denzosoft.llmplayer.tensor.FloatTensor;

public final class ModelWeights {
    private final FloatTensor tokenEmbedding;
    private final FloatTensor outputNorm;
    private final FloatTensor output;
    private final TransformerLayerWeights[] layers;
    private final float[] ropeFreqFactors;

    public ModelWeights(FloatTensor tokenEmbedding, FloatTensor outputNorm, FloatTensor output,
                        TransformerLayerWeights[] layers, float[] ropeFreqFactors) {
        this.tokenEmbedding = tokenEmbedding;
        this.outputNorm = outputNorm;
        this.output = output;
        this.layers = layers;
        this.ropeFreqFactors = ropeFreqFactors;
    }

    public FloatTensor tokenEmbedding() { return tokenEmbedding; }
    public FloatTensor outputNorm() { return outputNorm; }
    public FloatTensor output() { return output; }
    public TransformerLayerWeights[] layers() { return layers; }
    public float[] ropeFreqFactors() { return ropeFreqFactors; }
}
