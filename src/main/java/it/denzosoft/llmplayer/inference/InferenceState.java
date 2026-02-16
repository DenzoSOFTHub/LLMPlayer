package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.ModelConfig;

/**
 * Pre-allocated mutable state for a single inference request.
 * All buffers are allocated once and reused across tokens to avoid GC pressure.
 */
public class InferenceState {

    public final float[] x;      // current activation [embeddingLength]
    public final float[] xb;     // activation after rmsnorm [embeddingLength]
    public final float[] xb2;    // second buffer [max(embeddingLength, headCount*headSize)]
    public final float[] hb;     // FFN hidden buffer [intermediateSize]
    public final float[] hb2;    // FFN hidden buffer 2 [intermediateSize]
    public final float[] hbPacked; // FFN packed buffer [2 * intermediateSize] - for GLM4 packed FFN
    public final float[] q;      // query [max(embeddingLength, headCount*headSize)]
    public final float[] k;      // key [kvDim]
    public final float[] v;      // value [kvDim]
    public final float[] att;    // attention scores [headCount * maxSeqLen]
    public final float[] logits; // output logits [vocabSize]
    public final KVCache kvCache;

    public InferenceState(ModelConfig config, int maxSeqLen) {
        int dim = config.embeddingLength();
        int kvDim = config.kvDim();
        int ffnDim = config.intermediateSize();

        this.x = new float[dim];
        this.xb = new float[dim];
        int qDim = Math.max(dim, config.headCount() * config.headSize());
        this.xb2 = new float[qDim];
        this.hb = new float[ffnDim];
        this.hb2 = new float[ffnDim];
        this.hbPacked = new float[2 * ffnDim];
        this.q = new float[qDim];
        this.k = new float[kvDim];
        this.v = new float[kvDim];
        this.att = new float[config.headCount() * maxSeqLen];
        this.logits = new float[config.vocabSize()];
        this.kvCache = new KVCache(config.blockCount(), kvDim, maxSeqLen);
    }

    /**
     * Clear activation buffers (not KV cache) for next token.
     */
    public void clearActivations() {
        java.util.Arrays.fill(x, 0);
        java.util.Arrays.fill(logits, 0);
    }
}
