package it.denzosoft.llmplayer.model;

import it.denzosoft.llmplayer.tensor.FloatTensor;

/**
 * Weights for a single Nemotron-H layer.
 * Three types: Mamba-2 SSM, Attention (GQA), or FFN (squared ReLU).
 * Type determined by per-layer arrays in ModelConfig.
 */
public final class NemotronHLayerWeights {
    public static final int TYPE_MAMBA = 0;
    public static final int TYPE_ATTENTION = 1;
    public static final int TYPE_FFN = 2;

    private final int layerType;
    private final FloatTensor attnNorm;  // RMSNorm before each layer (all types)

    // Mamba-2 tensors (null for attention/FFN layers)
    private final FloatTensor ssmIn;       // input projection [dim, projDim]
    private final FloatTensor ssmConv1d;   // conv1d weight [kernelSize, channels]
    private final FloatTensor ssmConv1dBias; // conv1d bias [channels]
    private final FloatTensor ssmDtBias;   // dt bias [timeStepRank]
    private final FloatTensor ssmA;        // log(A) [1, timeStepRank]
    private final FloatTensor ssmD;        // D residual [1, timeStepRank]
    private final FloatTensor ssmNorm;     // grouped RMSNorm [innerSize/groupCount, groupCount]
    private final FloatTensor ssmOut;      // output projection [innerSize, dim]

    // Attention tensors (null for Mamba/FFN layers)
    private final FloatTensor wq;
    private final FloatTensor wk;
    private final FloatTensor wv;
    private final FloatTensor wo;

    // FFN tensors (null for Mamba/attention layers)
    private final FloatTensor ffnUp;
    private final FloatTensor ffnDown;

    /** Mamba-2 layer */
    public NemotronHLayerWeights(FloatTensor attnNorm,
                                  FloatTensor ssmIn, FloatTensor ssmConv1d, FloatTensor ssmConv1dBias,
                                  FloatTensor ssmDtBias, FloatTensor ssmA, FloatTensor ssmD,
                                  FloatTensor ssmNorm, FloatTensor ssmOut) {
        this.layerType = TYPE_MAMBA;
        this.attnNorm = attnNorm;
        this.ssmIn = ssmIn; this.ssmConv1d = ssmConv1d; this.ssmConv1dBias = ssmConv1dBias;
        this.ssmDtBias = ssmDtBias; this.ssmA = ssmA; this.ssmD = ssmD;
        this.ssmNorm = ssmNorm; this.ssmOut = ssmOut;
        this.wq = null; this.wk = null; this.wv = null; this.wo = null;
        this.ffnUp = null; this.ffnDown = null;
    }

    /** Attention layer */
    public NemotronHLayerWeights(FloatTensor attnNorm,
                                  FloatTensor wq, FloatTensor wk, FloatTensor wv, FloatTensor wo) {
        this.layerType = TYPE_ATTENTION;
        this.attnNorm = attnNorm;
        this.wq = wq; this.wk = wk; this.wv = wv; this.wo = wo;
        this.ssmIn = null; this.ssmConv1d = null; this.ssmConv1dBias = null;
        this.ssmDtBias = null; this.ssmA = null; this.ssmD = null;
        this.ssmNorm = null; this.ssmOut = null;
        this.ffnUp = null; this.ffnDown = null;
    }

    /** FFN layer */
    public NemotronHLayerWeights(FloatTensor attnNorm,
                                  FloatTensor ffnUp, FloatTensor ffnDown) {
        this.layerType = TYPE_FFN;
        this.attnNorm = attnNorm;
        this.ffnUp = ffnUp; this.ffnDown = ffnDown;
        this.ssmIn = null; this.ssmConv1d = null; this.ssmConv1dBias = null;
        this.ssmDtBias = null; this.ssmA = null; this.ssmD = null;
        this.ssmNorm = null; this.ssmOut = null;
        this.wq = null; this.wk = null; this.wv = null; this.wo = null;
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
    public FloatTensor ffnUp() { return ffnUp; }
    public FloatTensor ffnDown() { return ffnDown; }
}
