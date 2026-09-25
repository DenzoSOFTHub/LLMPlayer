package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.ModelConfig;

/**
 * Per-token activation buffers for batched prefill in the standard engine: one row per token of
 * the chunk being processed. Allocated lazily once per {@link InferenceState} and reused across
 * chunks and requests, so batched prefill adds no per-token allocation.
 */
final class PrefillBatch {

    final int capacity;
    final float[][] x;    // residual stream [n][dim]
    final float[][] xb;   // normed input / block output [n][dim]
    final float[][] q;    // [n][qDim]
    final float[][] k;    // [n][kvDim]
    final float[][] v;    // [n][kvDim]
    final float[][] att;  // attention output before Wo [n][qDim]
    final float[][] hb;   // FFN gate [n][ffnDim]
    final float[][] hb2;  // FFN up [n][ffnDim]
    private float[][] qkv; // merged-QKV output [n][qDim + 2 * kvDim], created on first use

    float[][] qkv(int qkvDim) {
        if (qkv == null || qkv[0].length < qkvDim) qkv = new float[capacity][qkvDim];
        return qkv;
    }

    PrefillBatch(ModelConfig config, int capacity) {
        int dim = config.embeddingLength();
        int qDim = config.headCount() * config.headSize();
        int kvDim = config.kvDim();
        int ffnDim = config.intermediateSize();
        this.capacity = capacity;
        this.x = new float[capacity][dim];
        this.xb = new float[capacity][dim];
        this.q = new float[capacity][qDim];
        this.k = new float[capacity][kvDim];
        this.v = new float[capacity][kvDim];
        this.att = new float[capacity][qDim];
        this.hb = new float[capacity][ffnDim];
        this.hb2 = new float[capacity][ffnDim];
    }
}
