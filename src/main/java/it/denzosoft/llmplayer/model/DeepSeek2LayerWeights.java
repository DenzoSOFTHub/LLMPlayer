package it.denzosoft.llmplayer.model;

import it.denzosoft.llmplayer.tensor.FloatTensor;

public final class DeepSeek2LayerWeights {
    private final FloatTensor attnNorm;
    private final FloatTensor ffnNorm;
    private final FloatTensor wq;
    private final FloatTensor wkvA;
    private final FloatTensor kvANorm;
    private final FloatTensor wkvB;
    private final FloatTensor wo;
    private final FloatTensor wGate;
    private final FloatTensor wUp;
    private final FloatTensor wDown;
    private final FloatTensor ffnGateInp;
    private final FloatTensor ffnGateExps;
    private final FloatTensor ffnUpExps;
    private final FloatTensor ffnDownExps;
    private final FloatTensor ffnGateShexp;
    private final FloatTensor ffnUpShexp;
    private final FloatTensor ffnDownShexp;

    public DeepSeek2LayerWeights(FloatTensor attnNorm, FloatTensor ffnNorm,
                                 FloatTensor wq, FloatTensor wkvA, FloatTensor kvANorm,
                                 FloatTensor wkvB, FloatTensor wo,
                                 FloatTensor wGate, FloatTensor wUp, FloatTensor wDown,
                                 FloatTensor ffnGateInp, FloatTensor ffnGateExps,
                                 FloatTensor ffnUpExps, FloatTensor ffnDownExps,
                                 FloatTensor ffnGateShexp, FloatTensor ffnUpShexp,
                                 FloatTensor ffnDownShexp) {
        this.attnNorm = attnNorm;
        this.ffnNorm = ffnNorm;
        this.wq = wq;
        this.wkvA = wkvA;
        this.kvANorm = kvANorm;
        this.wkvB = wkvB;
        this.wo = wo;
        this.wGate = wGate;
        this.wUp = wUp;
        this.wDown = wDown;
        this.ffnGateInp = ffnGateInp;
        this.ffnGateExps = ffnGateExps;
        this.ffnUpExps = ffnUpExps;
        this.ffnDownExps = ffnDownExps;
        this.ffnGateShexp = ffnGateShexp;
        this.ffnUpShexp = ffnUpShexp;
        this.ffnDownShexp = ffnDownShexp;
    }

    public FloatTensor attnNorm() { return attnNorm; }
    public FloatTensor ffnNorm() { return ffnNorm; }
    public FloatTensor wq() { return wq; }
    public FloatTensor wkvA() { return wkvA; }
    public FloatTensor kvANorm() { return kvANorm; }
    public FloatTensor wkvB() { return wkvB; }
    public FloatTensor wo() { return wo; }
    public FloatTensor wGate() { return wGate; }
    public FloatTensor wUp() { return wUp; }
    public FloatTensor wDown() { return wDown; }
    public FloatTensor ffnGateInp() { return ffnGateInp; }
    public FloatTensor ffnGateExps() { return ffnGateExps; }
    public FloatTensor ffnUpExps() { return ffnUpExps; }
    public FloatTensor ffnDownExps() { return ffnDownExps; }
    public FloatTensor ffnGateShexp() { return ffnGateShexp; }
    public FloatTensor ffnUpShexp() { return ffnUpShexp; }
    public FloatTensor ffnDownShexp() { return ffnDownShexp; }

    public boolean isMoELayer() {
        return ffnGateInp != null;
    }
}
