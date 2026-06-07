package it.denzosoft.llmplayer.model;

import it.denzosoft.llmplayer.tensor.FloatTensor;

/**
 * Weights for a single LFM2 layer. Every layer has an "operator_norm" (stored under the
 * attn_norm name), an FFN (SwiGLU), and EITHER a GQA attention mixer OR a gated short-conv mixer.
 */
public final class LFM2LayerWeights {
    private final boolean attention;
    // operator norm (applied before the mixer) + ffn norm
    private final FloatTensor operatorNorm, ffnNorm;
    // attention mixer (null on conv layers)
    private final FloatTensor wq, wk, wv, wo, qNorm, kNorm;
    // short-conv mixer (null on attention layers)
    private final FloatTensor convInProj, conv, convOutProj;
    // FFN (all layers)
    private final FloatTensor ffnGate, ffnUp, ffnDown;

    private LFM2LayerWeights(boolean attention, FloatTensor operatorNorm, FloatTensor ffnNorm,
            FloatTensor wq, FloatTensor wk, FloatTensor wv, FloatTensor wo, FloatTensor qNorm, FloatTensor kNorm,
            FloatTensor convInProj, FloatTensor conv, FloatTensor convOutProj,
            FloatTensor ffnGate, FloatTensor ffnUp, FloatTensor ffnDown) {
        this.attention = attention;
        this.operatorNorm = operatorNorm; this.ffnNorm = ffnNorm;
        this.wq = wq; this.wk = wk; this.wv = wv; this.wo = wo; this.qNorm = qNorm; this.kNorm = kNorm;
        this.convInProj = convInProj; this.conv = conv; this.convOutProj = convOutProj;
        this.ffnGate = ffnGate; this.ffnUp = ffnUp; this.ffnDown = ffnDown;
    }

    public static LFM2LayerWeights attention(FloatTensor operatorNorm, FloatTensor ffnNorm,
            FloatTensor wq, FloatTensor wk, FloatTensor wv, FloatTensor wo, FloatTensor qNorm, FloatTensor kNorm,
            FloatTensor ffnGate, FloatTensor ffnUp, FloatTensor ffnDown) {
        return new LFM2LayerWeights(true, operatorNorm, ffnNorm, wq, wk, wv, wo, qNorm, kNorm,
                null, null, null, ffnGate, ffnUp, ffnDown);
    }

    public static LFM2LayerWeights conv(FloatTensor operatorNorm, FloatTensor ffnNorm,
            FloatTensor convInProj, FloatTensor conv, FloatTensor convOutProj,
            FloatTensor ffnGate, FloatTensor ffnUp, FloatTensor ffnDown) {
        return new LFM2LayerWeights(false, operatorNorm, ffnNorm, null, null, null, null, null, null,
                convInProj, conv, convOutProj, ffnGate, ffnUp, ffnDown);
    }

    public boolean isAttention() { return attention; }
    public FloatTensor operatorNorm() { return operatorNorm; }
    public FloatTensor ffnNorm() { return ffnNorm; }
    public FloatTensor wq() { return wq; }
    public FloatTensor wk() { return wk; }
    public FloatTensor wv() { return wv; }
    public FloatTensor wo() { return wo; }
    public FloatTensor qNorm() { return qNorm; }
    public FloatTensor kNorm() { return kNorm; }
    public FloatTensor convInProj() { return convInProj; }
    public FloatTensor conv() { return conv; }
    public FloatTensor convOutProj() { return convOutProj; }
    public FloatTensor ffnGate() { return ffnGate; }
    public FloatTensor ffnUp() { return ffnUp; }
    public FloatTensor ffnDown() { return ffnDown; }
}
