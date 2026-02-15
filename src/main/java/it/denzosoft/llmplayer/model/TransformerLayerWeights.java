package it.denzosoft.llmplayer.model;

import it.denzosoft.llmplayer.tensor.FloatTensor;

public final class TransformerLayerWeights {
    private final FloatTensor attnNorm;
    private final FloatTensor ffnNorm;
    private final FloatTensor wq;
    private final FloatTensor wk;
    private final FloatTensor wv;
    private final FloatTensor wo;
    private final FloatTensor wqkv;  // merged QKV (Phi3/Phi4): null if Q,K,V are separate
    private final FloatTensor wGate;
    private final FloatTensor wUp;
    private final FloatTensor wDown;
    private final FloatTensor qBias;
    private final FloatTensor kBias;
    private final FloatTensor vBias;
    private final FloatTensor qNorm;
    private final FloatTensor kNorm;
    private final FloatTensor postAttnNorm;
    private final FloatTensor postFfnNorm;

    public TransformerLayerWeights(FloatTensor attnNorm, FloatTensor ffnNorm,
                                   FloatTensor wq, FloatTensor wk, FloatTensor wv, FloatTensor wo,
                                   FloatTensor wGate, FloatTensor wUp, FloatTensor wDown,
                                   FloatTensor qBias, FloatTensor kBias, FloatTensor vBias,
                                   FloatTensor qNorm, FloatTensor kNorm,
                                   FloatTensor postAttnNorm, FloatTensor postFfnNorm) {
        this(attnNorm, ffnNorm, wq, wk, wv, wo, null, wGate, wUp, wDown,
             qBias, kBias, vBias, qNorm, kNorm, postAttnNorm, postFfnNorm);
    }

    public TransformerLayerWeights(FloatTensor attnNorm, FloatTensor ffnNorm,
                                   FloatTensor wq, FloatTensor wk, FloatTensor wv, FloatTensor wo,
                                   FloatTensor wqkv,
                                   FloatTensor wGate, FloatTensor wUp, FloatTensor wDown,
                                   FloatTensor qBias, FloatTensor kBias, FloatTensor vBias,
                                   FloatTensor qNorm, FloatTensor kNorm,
                                   FloatTensor postAttnNorm, FloatTensor postFfnNorm) {
        this.attnNorm = attnNorm;
        this.ffnNorm = ffnNorm;
        this.wq = wq;
        this.wk = wk;
        this.wv = wv;
        this.wo = wo;
        this.wqkv = wqkv;
        this.wGate = wGate;
        this.wUp = wUp;
        this.wDown = wDown;
        this.qBias = qBias;
        this.kBias = kBias;
        this.vBias = vBias;
        this.qNorm = qNorm;
        this.kNorm = kNorm;
        this.postAttnNorm = postAttnNorm;
        this.postFfnNorm = postFfnNorm;
    }

    public FloatTensor attnNorm() { return attnNorm; }
    public FloatTensor ffnNorm() { return ffnNorm; }
    public FloatTensor wq() { return wq; }
    public FloatTensor wk() { return wk; }
    public FloatTensor wv() { return wv; }
    public FloatTensor wo() { return wo; }
    public FloatTensor wqkv() { return wqkv; }
    public FloatTensor wGate() { return wGate; }
    public FloatTensor wUp() { return wUp; }
    public FloatTensor wDown() { return wDown; }
    public FloatTensor qBias() { return qBias; }
    public FloatTensor kBias() { return kBias; }
    public FloatTensor vBias() { return vBias; }
    public FloatTensor qNorm() { return qNorm; }
    public FloatTensor kNorm() { return kNorm; }
    public FloatTensor postAttnNorm() { return postAttnNorm; }
    public FloatTensor postFfnNorm() { return postFfnNorm; }
}
