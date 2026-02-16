package it.denzosoft.llmplayer.model;

import it.denzosoft.llmplayer.tensor.FloatTensor;

/**
 * Per-layer weights for Qwen3 MoE architecture.
 * Combines standard GQA attention (with QK-norm) with either dense SwiGLU FFN
 * or Mixture-of-Experts FFN depending on the layer.
 * Also supports GPT-OSS attention/expert biases.
 */
public final class Qwen3MoELayerWeights {
    // Norms
    private final FloatTensor attnNorm;
    private final FloatTensor ffnNorm;

    // Standard GQA attention
    private final FloatTensor wq;
    private final FloatTensor wk;
    private final FloatTensor wv;
    private final FloatTensor wo;

    // Qwen3 per-head QK normalization
    private final FloatTensor qNorm;
    private final FloatTensor kNorm;

    // Dense FFN (for leading dense blocks only, null for MoE layers)
    private final FloatTensor wGate;
    private final FloatTensor wUp;
    private final FloatTensor wDown;

    // MoE FFN (for MoE layers only, null for dense layers)
    private final FloatTensor ffnGateInp;     // router weights [expertCount, dim]
    private final FloatTensor ffnGateExps;    // all experts gate [expertCount * expertFfnDim * dim]
    private final FloatTensor ffnUpExps;      // all experts up
    private final FloatTensor ffnDownExps;    // all experts down
    private final FloatTensor ffnGateShexp;   // shared expert gate
    private final FloatTensor ffnUpShexp;     // shared expert up
    private final FloatTensor ffnDownShexp;   // shared expert down

    // Optional biases (GPT-OSS)
    private final FloatTensor wqBias;
    private final FloatTensor wkBias;
    private final FloatTensor wvBias;
    private final FloatTensor woBias;
    private final FloatTensor ffnGateInpBias;
    private final FloatTensor ffnGateExpsBias;
    private final FloatTensor ffnUpExpsBias;
    private final FloatTensor ffnDownExpsBias;

    // Attention sinks (GPT-OSS): per-head learned bias for softmax denominator
    private final FloatTensor attnSinks;

    public Qwen3MoELayerWeights(
            FloatTensor attnNorm, FloatTensor ffnNorm,
            FloatTensor wq, FloatTensor wk, FloatTensor wv, FloatTensor wo,
            FloatTensor qNorm, FloatTensor kNorm,
            FloatTensor wGate, FloatTensor wUp, FloatTensor wDown,
            FloatTensor ffnGateInp, FloatTensor ffnGateExps,
            FloatTensor ffnUpExps, FloatTensor ffnDownExps,
            FloatTensor ffnGateShexp, FloatTensor ffnUpShexp,
            FloatTensor ffnDownShexp) {
        this(attnNorm, ffnNorm, wq, wk, wv, wo, qNorm, kNorm,
             wGate, wUp, wDown, ffnGateInp, ffnGateExps, ffnUpExps, ffnDownExps,
             ffnGateShexp, ffnUpShexp, ffnDownShexp,
             null, null, null, null, null, null, null, null, null);
    }

    public Qwen3MoELayerWeights(
            FloatTensor attnNorm, FloatTensor ffnNorm,
            FloatTensor wq, FloatTensor wk, FloatTensor wv, FloatTensor wo,
            FloatTensor qNorm, FloatTensor kNorm,
            FloatTensor wGate, FloatTensor wUp, FloatTensor wDown,
            FloatTensor ffnGateInp, FloatTensor ffnGateExps,
            FloatTensor ffnUpExps, FloatTensor ffnDownExps,
            FloatTensor ffnGateShexp, FloatTensor ffnUpShexp,
            FloatTensor ffnDownShexp,
            FloatTensor wqBias, FloatTensor wkBias, FloatTensor wvBias, FloatTensor woBias,
            FloatTensor ffnGateInpBias,
            FloatTensor ffnGateExpsBias, FloatTensor ffnUpExpsBias, FloatTensor ffnDownExpsBias,
            FloatTensor attnSinks) {
        this.attnNorm = attnNorm;
        this.ffnNorm = ffnNorm;
        this.wq = wq;
        this.wk = wk;
        this.wv = wv;
        this.wo = wo;
        this.qNorm = qNorm;
        this.kNorm = kNorm;
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
        this.wqBias = wqBias;
        this.wkBias = wkBias;
        this.wvBias = wvBias;
        this.woBias = woBias;
        this.ffnGateInpBias = ffnGateInpBias;
        this.ffnGateExpsBias = ffnGateExpsBias;
        this.ffnUpExpsBias = ffnUpExpsBias;
        this.ffnDownExpsBias = ffnDownExpsBias;
        this.attnSinks = attnSinks;
    }

    public FloatTensor attnNorm() { return attnNorm; }
    public FloatTensor ffnNorm() { return ffnNorm; }
    public FloatTensor wq() { return wq; }
    public FloatTensor wk() { return wk; }
    public FloatTensor wv() { return wv; }
    public FloatTensor wo() { return wo; }
    public FloatTensor qNorm() { return qNorm; }
    public FloatTensor kNorm() { return kNorm; }
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

    // Bias accessors (null if absent)
    public FloatTensor wqBias() { return wqBias; }
    public FloatTensor wkBias() { return wkBias; }
    public FloatTensor wvBias() { return wvBias; }
    public FloatTensor woBias() { return woBias; }
    public FloatTensor ffnGateInpBias() { return ffnGateInpBias; }
    public FloatTensor ffnGateExpsBias() { return ffnGateExpsBias; }
    public FloatTensor ffnUpExpsBias() { return ffnUpExpsBias; }
    public FloatTensor ffnDownExpsBias() { return ffnDownExpsBias; }
    public FloatTensor attnSinks() { return attnSinks; }

    public boolean isMoELayer() {
        return ffnGateInp != null;
    }
}
