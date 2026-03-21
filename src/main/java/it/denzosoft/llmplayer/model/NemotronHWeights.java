package it.denzosoft.llmplayer.model;

import it.denzosoft.llmplayer.tensor.FloatTensor;

public final class NemotronHWeights {
    private final FloatTensor tokenEmbedding;
    private final FloatTensor outputNorm;
    private final FloatTensor output;
    private final NemotronHLayerWeights[] layers;
    private final float[] ropeFreqFactors;

    public NemotronHWeights(FloatTensor tokenEmbedding, FloatTensor outputNorm, FloatTensor output,
                             NemotronHLayerWeights[] layers, float[] ropeFreqFactors) {
        this.tokenEmbedding = tokenEmbedding;
        this.outputNorm = outputNorm;
        this.output = output;
        this.layers = layers;
        this.ropeFreqFactors = ropeFreqFactors;
    }

    public FloatTensor tokenEmbedding() { return tokenEmbedding; }
    public FloatTensor outputNorm() { return outputNorm; }
    public FloatTensor output() { return output; }
    public NemotronHLayerWeights[] layers() { return layers; }
    public float[] ropeFreqFactors() { return ropeFreqFactors; }
}
