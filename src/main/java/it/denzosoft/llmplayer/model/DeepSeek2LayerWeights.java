package it.denzosoft.llmplayer.model;

import it.denzosoft.llmplayer.tensor.FloatTensor;

public final class DeepSeek2LayerWeights {
    private final FloatTensor attnNorm;
    private final FloatTensor ffnNorm;
    private final FloatTensor wq;       // null when Q-LoRA is used (wqA/wqANorm/wqB instead)
    private final FloatTensor wkvA;
    private final FloatTensor kvANorm;
    private final FloatTensor wkvB;     // combined K+V decompression (null when separate wkB/wvB)
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
    // Q-LoRA (GLM-4.7-Flash / DeepSeek-V3): decompose Q into Q_A * norm * Q_B
    private final FloatTensor wqA;
    private final FloatTensor wqANorm;
    private final FloatTensor wqB;
    // Separate K_B and V_B (3D per-head tensors, GLM-4.7-Flash / DeepSeek-V3)
    private final FloatTensor wkB;
    private final FloatTensor wvB;
    // Expert probability bias (GLM-4.7-Flash)
    private final FloatTensor expProbsBias;

    public DeepSeek2LayerWeights(FloatTensor attnNorm, FloatTensor ffnNorm,
                                 FloatTensor wq, FloatTensor wkvA, FloatTensor kvANorm,
                                 FloatTensor wkvB, FloatTensor wo,
                                 FloatTensor wGate, FloatTensor wUp, FloatTensor wDown,
                                 FloatTensor ffnGateInp, FloatTensor ffnGateExps,
                                 FloatTensor ffnUpExps, FloatTensor ffnDownExps,
                                 FloatTensor ffnGateShexp, FloatTensor ffnUpShexp,
                                 FloatTensor ffnDownShexp) {
        this(attnNorm, ffnNorm, wq, wkvA, kvANorm, wkvB, wo,
             wGate, wUp, wDown, ffnGateInp, ffnGateExps, ffnUpExps, ffnDownExps,
             ffnGateShexp, ffnUpShexp, ffnDownShexp,
             null, null, null, null, null, null);
    }

    public DeepSeek2LayerWeights(FloatTensor attnNorm, FloatTensor ffnNorm,
                                 FloatTensor wq, FloatTensor wkvA, FloatTensor kvANorm,
                                 FloatTensor wkvB, FloatTensor wo,
                                 FloatTensor wGate, FloatTensor wUp, FloatTensor wDown,
                                 FloatTensor ffnGateInp, FloatTensor ffnGateExps,
                                 FloatTensor ffnUpExps, FloatTensor ffnDownExps,
                                 FloatTensor ffnGateShexp, FloatTensor ffnUpShexp,
                                 FloatTensor ffnDownShexp,
                                 FloatTensor wqA, FloatTensor wqANorm, FloatTensor wqB,
                                 FloatTensor wkB, FloatTensor wvB,
                                 FloatTensor expProbsBias) {
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
        this.wqA = wqA;
        this.wqANorm = wqANorm;
        this.wqB = wqB;
        this.wkB = wkB;
        this.wvB = wvB;
        this.expProbsBias = expProbsBias;
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
    public FloatTensor wqA() { return wqA; }
    public FloatTensor wqANorm() { return wqANorm; }
    public FloatTensor wqB() { return wqB; }
    public FloatTensor wkB() { return wkB; }
    public FloatTensor wvB() { return wvB; }
    public FloatTensor expProbsBias() { return expProbsBias; }

    public boolean isMoELayer() {
        return ffnGateInp != null;
    }

    public boolean hasQLoRA() {
        return wqA != null;
    }

    public boolean hasSeparateKVB() {
        return wkB != null;
    }
}
