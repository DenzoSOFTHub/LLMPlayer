package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.ModelConfig;

/**
 * Mutable state for Qwen3.5 inference.
 * Contains:
 * - Standard activation buffers (x, xb, etc.)
 * - KV cache for full attention layers only
 * - DeltaNet recurrent state (S matrices) per DeltaNet layer
 * - Conv1d state buffer per DeltaNet layer
 */
public class Qwen35State {

    public final float[] x;      // current activation [embeddingLength]
    public final float[] xb;     // activation after rmsnorm [embeddingLength]
    public final float[] xb2;    // second buffer
    public final float[] hb;     // FFN hidden buffer [intermediateSize]
    public final float[] hb2;    // FFN hidden buffer 2 [intermediateSize]
    public final float[] logits; // output logits [vocabSize]

    // Full attention buffers (only used for attention layers)
    public final float[] q;
    public final float[] k;
    public final float[] v;
    public final float[] att;
    public final KVCache kvCache;

    // DeltaNet buffers
    public final float[] qkv;      // QKV projection output [qkvDim]
    public final float[] gate;     // gate output [innerSize]
    public final float[] alpha;    // alpha gate [timeStepRank]
    public final float[] beta;     // beta gate [timeStepRank]
    public final float[] deltaOut; // DeltaNet output [innerSize]

    // DeltaNet persistent state per layer
    // ssmState[layer][head][d_qk * d_v] — recurrent state matrices
    public final float[][][] ssmState;
    // convState[layer][convKernel-1][channels] — causal conv1d buffer
    public final float[][][] convState;
    public final int[] convStatePos; // circular buffer position per layer

    private final int blockCount;
    private final int fullAttnInterval;

    // Attention output gate buffer (for full attention layers with Q+gate packing)
    public final float[] attnGate;

    public Qwen35State(ModelConfig config, int maxSeqLen) {
        this.blockCount = config.blockCount();
        this.fullAttnInterval = config.fullAttentionInterval();

        int dim = config.embeddingLength();
        int ffnDim = config.intermediateSize();
        int timeStepRank = config.ssmTimeStepRank();
        int innerSize = config.ssmInnerSize();
        int stateSize = config.ssmStateSize();
        int convKernel = config.ssmConvKernel();

        // QKV dim: for DeltaNet, from attn_qkv weight second dimension
        // groupCount * stateSize (Q) + groupCount * stateSize (K) + timeStepRank * stateSize (V)
        int groupCount = config.ssmGroupCount();
        int qkvDim = groupCount * stateSize * 2 + timeStepRank * stateSize;

        this.x = new float[dim];
        this.xb = new float[dim];
        int qDim = config.headCount() * config.headSize();
        // Q projection outputs 2x (Q + gate packed together)
        int qGateDim = qDim * 2;
        this.xb2 = new float[Math.max(dim, qDim)];
        this.hb = new float[ffnDim];
        this.hb2 = new float[ffnDim];
        this.logits = new float[config.vocabSize()];

        // Full attention buffers (q holds Q+gate from projection, attnGate holds the gate)
        int kvDim = config.kvDim();
        this.q = new float[Math.max(dim, qGateDim)];
        this.attnGate = new float[qDim];
        this.k = new float[kvDim];
        this.v = new float[kvDim];
        this.att = new float[config.headCount() * maxSeqLen];
        // KV cache only for full attention layers (every fullAttnInterval-th layer)
        this.kvCache = new KVCache(blockCount, kvDim, maxSeqLen);

        // DeltaNet buffers
        this.qkv = new float[qkvDim];
        this.gate = new float[innerSize];
        this.alpha = new float[timeStepRank];
        this.beta = new float[timeStepRank];
        this.deltaOut = new float[innerSize];

        // Persistent DeltaNet state
        // Count DeltaNet layers
        this.ssmState = new float[blockCount][][];
        this.convState = new float[blockCount][][];
        this.convStatePos = new int[blockCount];

        for (int layer = 0; layer < blockCount; layer++) {
            if (isDeltaNetLayer(layer)) {
                // State per head: timeStepRank heads, each with [head_k_dim, head_v_dim]
                // head_k_dim = stateSize (key heads share Q/K across groups)
                // head_v_dim = stateSize
                int d_qk = stateSize;
                int d_v = stateSize;
                this.ssmState[layer] = new float[timeStepRank][d_qk * d_v];
                this.convState[layer] = new float[convKernel - 1][qkvDim];
            }
        }
    }

    public boolean isDeltaNetLayer(int layer) {
        return fullAttnInterval > 0 && ((layer + 1) % fullAttnInterval != 0);
    }

    public boolean isAttentionLayer(int layer) {
        return fullAttnInterval > 0 && ((layer + 1) % fullAttnInterval == 0);
    }
}
