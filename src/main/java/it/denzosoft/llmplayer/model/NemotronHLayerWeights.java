package it.denzosoft.llmplayer.model;

import it.denzosoft.llmplayer.tensor.FloatTensor;

/**
 * Weights for a single Nemotron-H / Granite Hybrid layer.
 * Layer types: Mamba-2 SSM, Attention (GQA), FFN, or combined (Mamba+FFN, Attention+FFN).
 */
public final class NemotronHLayerWeights {
    public static final int TYPE_MAMBA = 0;
    public static final int TYPE_ATTENTION = 1;
    public static final int TYPE_FFN = 2;

    private final int layerType;
    private final FloatTensor attnNorm;
    private final FloatTensor ssmIn, ssmConv1d, ssmConv1dBias, ssmDtBias, ssmA, ssmD, ssmNorm, ssmOut;
    private final FloatTensor wq, wk, wv, wo;
    private final FloatTensor ffnNorm, ffnGate, ffnUp, ffnDown;

    // All-fields constructor
    private NemotronHLayerWeights(int type, FloatTensor attnNorm,
            FloatTensor ssmIn, FloatTensor ssmConv1d, FloatTensor ssmConv1dBias,
            FloatTensor ssmDtBias, FloatTensor ssmA, FloatTensor ssmD, FloatTensor ssmNorm, FloatTensor ssmOut,
            FloatTensor wq, FloatTensor wk, FloatTensor wv, FloatTensor wo,
            FloatTensor ffnNorm, FloatTensor ffnGate, FloatTensor ffnUp, FloatTensor ffnDown) {
        this.layerType = type; this.attnNorm = attnNorm;
        this.ssmIn = ssmIn; this.ssmConv1d = ssmConv1d; this.ssmConv1dBias = ssmConv1dBias;
        this.ssmDtBias = ssmDtBias; this.ssmA = ssmA; this.ssmD = ssmD; this.ssmNorm = ssmNorm; this.ssmOut = ssmOut;
        this.wq = wq; this.wk = wk; this.wv = wv; this.wo = wo;
        this.ffnNorm = ffnNorm; this.ffnGate = ffnGate; this.ffnUp = ffnUp; this.ffnDown = ffnDown;
    }

    /** Pure Mamba-2 layer (Nemotron-H) */
    public NemotronHLayerWeights(FloatTensor attnNorm,
            FloatTensor ssmIn, FloatTensor ssmConv1d, FloatTensor ssmConv1dBias,
            FloatTensor ssmDtBias, FloatTensor ssmA, FloatTensor ssmD, FloatTensor ssmNorm, FloatTensor ssmOut) {
        this(TYPE_MAMBA, attnNorm, ssmIn, ssmConv1d, ssmConv1dBias, ssmDtBias, ssmA, ssmD, ssmNorm, ssmOut,
             null, null, null, null, null, null, null, null);
    }

    /** Mamba-2 + SwiGLU FFN layer (Granite Hybrid) */
    public static NemotronHLayerWeights mambaWithFFN(FloatTensor attnNorm,
            FloatTensor ssmIn, FloatTensor ssmConv1d, FloatTensor ssmConv1dBias,
            FloatTensor ssmDtBias, FloatTensor ssmA, FloatTensor ssmD, FloatTensor ssmNorm, FloatTensor ssmOut,
            FloatTensor ffnNorm, FloatTensor ffnGate, FloatTensor ffnUp, FloatTensor ffnDown) {
        return new NemotronHLayerWeights(TYPE_MAMBA, attnNorm, ssmIn, ssmConv1d, ssmConv1dBias,
                ssmDtBias, ssmA, ssmD, ssmNorm, ssmOut, null, null, null, null, ffnNorm, ffnGate, ffnUp, ffnDown);
    }

    /** Attention layer (with optional FFN) */
    public static NemotronHLayerWeights attention(FloatTensor attnNorm,
            FloatTensor wq, FloatTensor wk, FloatTensor wv, FloatTensor wo,
            FloatTensor ffnNorm, FloatTensor ffnGate, FloatTensor ffnUp, FloatTensor ffnDown) {
        return new NemotronHLayerWeights(TYPE_ATTENTION, attnNorm, null, null, null, null, null, null, null, null,
                wq, wk, wv, wo, ffnNorm, ffnGate, ffnUp, ffnDown);
    }

    /** FFN-only layer */
    public NemotronHLayerWeights(FloatTensor attnNorm, FloatTensor ffnUp, FloatTensor ffnDown) {
        this(TYPE_FFN, attnNorm, null, null, null, null, null, null, null, null,
             null, null, null, null, null, null, ffnUp, ffnDown);
    }

    public int layerType() { return layerType; }
    public boolean isMamba() { return layerType == TYPE_MAMBA; }
    public boolean isAttention() { return layerType == TYPE_ATTENTION; }
    public boolean isFFN() { return layerType == TYPE_FFN; }

    public FloatTensor attnNorm() { return attnNorm; }
    public FloatTensor ssmIn() { return ssmIn; }
    public FloatTensor ssmConv1d() { return ssmConv1d; }
    public FloatTensor ssmConv1dBias() { return ssmConv1dBias; }
    public FloatTensor ssmDtBias() { return ssmDtBias; }
    public FloatTensor ssmA() { return ssmA; }
    public FloatTensor ssmD() { return ssmD; }
    public FloatTensor ssmNorm() { return ssmNorm; }
    public FloatTensor ssmOut() { return ssmOut; }
    public FloatTensor wq() { return wq; }
    public FloatTensor wk() { return wk; }
    public FloatTensor wv() { return wv; }
    public FloatTensor wo() { return wo; }
    public FloatTensor ffnNorm() { return ffnNorm; }
    public FloatTensor ffnGate() { return ffnGate; }
    public FloatTensor ffnUp() { return ffnUp; }
    public FloatTensor ffnDown() { return ffnDown; }
}
