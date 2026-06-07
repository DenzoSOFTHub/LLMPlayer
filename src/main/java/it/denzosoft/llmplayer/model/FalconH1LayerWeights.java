package it.denzosoft.llmplayer.model;

import it.denzosoft.llmplayer.tensor.FloatTensor;

/**
 * Weights for a single Falcon-H1 layer. Every layer is a PARALLEL hybrid: attention and a
 * Mamba-2 SSM both run on the same pre-normed input (attn_norm), their outputs are summed,
 * then a SwiGLU FFN follows. Per llama.cpp src/models/falcon-h1.cpp.
 */
public final class FalconH1LayerWeights {
    private final FloatTensor attnNorm;                       // shared by attn + mamba
    // attention
    private final FloatTensor wq, wk, wv, wo, woBias;
    // mamba-2
    private final FloatTensor ssmIn, ssmConv1d, ssmConv1dBias, ssmDtBias, ssmA, ssmD, ssmNorm, ssmOut;
    // ffn (SwiGLU, optional biases)
    private final FloatTensor ffnNorm, ffnGate, ffnGateBias, ffnUp, ffnUpBias, ffnDown, ffnDownBias;

    public FalconH1LayerWeights(FloatTensor attnNorm,
            FloatTensor wq, FloatTensor wk, FloatTensor wv, FloatTensor wo, FloatTensor woBias,
            FloatTensor ssmIn, FloatTensor ssmConv1d, FloatTensor ssmConv1dBias, FloatTensor ssmDtBias,
            FloatTensor ssmA, FloatTensor ssmD, FloatTensor ssmNorm, FloatTensor ssmOut,
            FloatTensor ffnNorm, FloatTensor ffnGate, FloatTensor ffnGateBias,
            FloatTensor ffnUp, FloatTensor ffnUpBias, FloatTensor ffnDown, FloatTensor ffnDownBias) {
        this.attnNorm = attnNorm;
        this.wq = wq; this.wk = wk; this.wv = wv; this.wo = wo; this.woBias = woBias;
        this.ssmIn = ssmIn; this.ssmConv1d = ssmConv1d; this.ssmConv1dBias = ssmConv1dBias; this.ssmDtBias = ssmDtBias;
        this.ssmA = ssmA; this.ssmD = ssmD; this.ssmNorm = ssmNorm; this.ssmOut = ssmOut;
        this.ffnNorm = ffnNorm; this.ffnGate = ffnGate; this.ffnGateBias = ffnGateBias;
        this.ffnUp = ffnUp; this.ffnUpBias = ffnUpBias; this.ffnDown = ffnDown; this.ffnDownBias = ffnDownBias;
    }

    public FloatTensor attnNorm() { return attnNorm; }
    public FloatTensor wq() { return wq; }
    public FloatTensor wk() { return wk; }
    public FloatTensor wv() { return wv; }
    public FloatTensor wo() { return wo; }
    public FloatTensor woBias() { return woBias; }
    public FloatTensor ssmIn() { return ssmIn; }
    public FloatTensor ssmConv1d() { return ssmConv1d; }
    public FloatTensor ssmConv1dBias() { return ssmConv1dBias; }
    public FloatTensor ssmDtBias() { return ssmDtBias; }
    public FloatTensor ssmA() { return ssmA; }
    public FloatTensor ssmD() { return ssmD; }
    public FloatTensor ssmNorm() { return ssmNorm; }
    public FloatTensor ssmOut() { return ssmOut; }
    public FloatTensor ffnNorm() { return ffnNorm; }
    public FloatTensor ffnGate() { return ffnGate; }
    public FloatTensor ffnGateBias() { return ffnGateBias; }
    public FloatTensor ffnUp() { return ffnUp; }
    public FloatTensor ffnUpBias() { return ffnUpBias; }
    public FloatTensor ffnDown() { return ffnDown; }
    public FloatTensor ffnDownBias() { return ffnDownBias; }
}
