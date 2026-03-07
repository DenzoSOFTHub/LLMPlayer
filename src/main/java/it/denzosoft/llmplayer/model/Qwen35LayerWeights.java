package it.denzosoft.llmplayer.model;

import it.denzosoft.llmplayer.tensor.FloatTensor;

/**
 * Weights for a single Qwen3.5 layer.
 * Qwen3.5 has two types of layers:
 * - DeltaNet layers (3/4): attn_gate, attn_qkv, ssm_* tensors
 * - Full attention layers (1/4): attn_q, attn_k, attn_v, attn_output, qk norms
 * Both types share: attn_norm, post_attention_norm, ffn_gate, ffn_up, ffn_down
 */
public final class Qwen35LayerWeights {
    // Common
    private final FloatTensor attnNorm;
    private final FloatTensor postAttnNorm;
    private final FloatTensor ffnGate;
    private final FloatTensor ffnUp;
    private final FloatTensor ffnDown;
    private final boolean isDeltaNet;

    // DeltaNet layer tensors (null for full attention layers)
    private final FloatTensor attnGate;     // output gate [embed, inner_size]
    private final FloatTensor attnQkv;      // QKV projection [embed, qkv_dim]
    private final FloatTensor ssmA;         // decay parameter [time_step_rank]
    private final FloatTensor ssmAlpha;     // alpha gate [embed, time_step_rank]
    private final FloatTensor ssmBeta;      // beta gate [embed, time_step_rank]
    private final FloatTensor ssmConv1d;    // conv kernel [kernel_size, channels]
    private final FloatTensor ssmDtBias;    // dt bias [time_step_rank]
    private final FloatTensor ssmNorm;      // output norm [state_size]
    private final FloatTensor ssmOut;       // output projection [inner_size, embed]

    // Full attention layer tensors (null for DeltaNet layers)
    private final FloatTensor wq;
    private final FloatTensor wk;
    private final FloatTensor wv;
    private final FloatTensor wo;
    private final FloatTensor qNorm;
    private final FloatTensor kNorm;

    /** Constructor for DeltaNet layers */
    public Qwen35LayerWeights(FloatTensor attnNorm, FloatTensor postAttnNorm,
                               FloatTensor ffnGate, FloatTensor ffnUp, FloatTensor ffnDown,
                               FloatTensor attnGate, FloatTensor attnQkv,
                               FloatTensor ssmA, FloatTensor ssmAlpha, FloatTensor ssmBeta,
                               FloatTensor ssmConv1d, FloatTensor ssmDtBias,
                               FloatTensor ssmNorm, FloatTensor ssmOut) {
        this.attnNorm = attnNorm;
        this.postAttnNorm = postAttnNorm;
        this.ffnGate = ffnGate;
        this.ffnUp = ffnUp;
        this.ffnDown = ffnDown;
        this.isDeltaNet = true;
        this.attnGate = attnGate;
        this.attnQkv = attnQkv;
        this.ssmA = ssmA;
        this.ssmAlpha = ssmAlpha;
        this.ssmBeta = ssmBeta;
        this.ssmConv1d = ssmConv1d;
        this.ssmDtBias = ssmDtBias;
        this.ssmNorm = ssmNorm;
        this.ssmOut = ssmOut;
        this.wq = null;
        this.wk = null;
        this.wv = null;
        this.wo = null;
        this.qNorm = null;
        this.kNorm = null;
    }

    /** Constructor for Full Attention layers */
    public Qwen35LayerWeights(FloatTensor attnNorm, FloatTensor postAttnNorm,
                               FloatTensor ffnGate, FloatTensor ffnUp, FloatTensor ffnDown,
                               FloatTensor wq, FloatTensor wk, FloatTensor wv, FloatTensor wo,
                               FloatTensor qNorm, FloatTensor kNorm) {
        this.attnNorm = attnNorm;
        this.postAttnNorm = postAttnNorm;
        this.ffnGate = ffnGate;
        this.ffnUp = ffnUp;
        this.ffnDown = ffnDown;
        this.isDeltaNet = false;
        this.attnGate = null;
        this.attnQkv = null;
        this.ssmA = null;
        this.ssmAlpha = null;
        this.ssmBeta = null;
        this.ssmConv1d = null;
        this.ssmDtBias = null;
        this.ssmNorm = null;
        this.ssmOut = null;
        this.wq = wq;
        this.wk = wk;
        this.wv = wv;
        this.wo = wo;
        this.qNorm = qNorm;
        this.kNorm = kNorm;
    }

    public boolean isDeltaNet() { return isDeltaNet; }

    // Common accessors
    public FloatTensor attnNorm() { return attnNorm; }
    public FloatTensor postAttnNorm() { return postAttnNorm; }
    public FloatTensor ffnGate() { return ffnGate; }
    public FloatTensor ffnUp() { return ffnUp; }
    public FloatTensor ffnDown() { return ffnDown; }

    // DeltaNet accessors
    public FloatTensor attnGate() { return attnGate; }
    public FloatTensor attnQkv() { return attnQkv; }
    public FloatTensor ssmA() { return ssmA; }
    public FloatTensor ssmAlpha() { return ssmAlpha; }
    public FloatTensor ssmBeta() { return ssmBeta; }
    public FloatTensor ssmConv1d() { return ssmConv1d; }
    public FloatTensor ssmDtBias() { return ssmDtBias; }
    public FloatTensor ssmNorm() { return ssmNorm; }
    public FloatTensor ssmOut() { return ssmOut; }

    // Full attention accessors
    public FloatTensor wq() { return wq; }
    public FloatTensor wk() { return wk; }
    public FloatTensor wv() { return wv; }
    public FloatTensor wo() { return wo; }
    public FloatTensor qNorm() { return qNorm; }
    public FloatTensor kNorm() { return kNorm; }
}
