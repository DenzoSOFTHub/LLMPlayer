package it.denzosoft.llmplayer.model;

import it.denzosoft.llmplayer.tensor.FloatTensor;

public final class LFM2Weights {
    private final FloatTensor tokenEmbedding;
    private final FloatTensor outputNorm;   // token_embd_norm (final norm)
    private final FloatTensor output;        // tied to tokenEmbedding when absent
    private final LFM2LayerWeights[] layers;
    private final float[] ropeFreqFactors;

    public LFM2Weights(FloatTensor tokenEmbedding, FloatTensor outputNorm, FloatTensor output,
                       LFM2LayerWeights[] layers, float[] ropeFreqFactors) {
        this.tokenEmbedding = tokenEmbedding;
        this.outputNorm = outputNorm;
        this.output = output;
        this.layers = layers;
        this.ropeFreqFactors = ropeFreqFactors;
    }

    public FloatTensor tokenEmbedding() { return tokenEmbedding; }
    public FloatTensor outputNorm() { return outputNorm; }
    public FloatTensor output() { return output; }
    public LFM2LayerWeights[] layers() { return layers; }
    public float[] ropeFreqFactors() { return ropeFreqFactors; }
}
