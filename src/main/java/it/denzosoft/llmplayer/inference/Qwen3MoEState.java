package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.ModelConfig;

/**
 * Pre-allocated mutable state for Qwen3 MoE inference.
 * Uses standard GQA attention buffers + MoE routing/expert buffers.
 */
public class Qwen3MoEState {

    // Standard transformer buffers
    public final float[] x;      // [dim]
    public final float[] xb;     // [dim] - activation after norm, also accumulator for MoE output
    public final float[] xb2;    // [dim] - attention output before Wo projection
    public final float[] logits; // [vocabSize]

    // Standard GQA attention buffers
    public final float[] q;      // [dim] = headCount * headSize
    public final float[] k;      // [kvDim] = headCountKV * headSize
    public final float[] v;      // [kvDim]
    public final float[] att;    // [headCount * maxSeqLen]
    public final KVCache kvCache;

    // Dense FFN buffers (for leading dense blocks)
    public final float[] hb;     // [ffnDim]
    public final float[] hb2;    // [ffnDim]

    // MoE buffers
    public final float[] xbSaved;         // [dim] - saved input for MoE
    public final float[] routerLogits;    // [expertCount]
    public final int[] selectedExperts;   // [expertUsedCount]
    public final float[] selectedWeights; // [expertUsedCount]
    public final float[] expertOut;       // [dim] - single expert output
    public final float[] sharedHb;        // [sharedFfnDim]
    public final float[] sharedHb2;       // [sharedFfnDim]
    // Per-expert parallel buffers
    public final float[][] moeHbPerExpert;    // [expertUsedCount][expertFfnDim]
    public final float[][] moeHb2PerExpert;   // [expertUsedCount][expertFfnDim]
    public final float[][] expertOutPerExpert; // [expertUsedCount][dim]

    public Qwen3MoEState(ModelConfig config, int maxSeqLen) {
        int dim = config.embeddingLength();
        int kvDim = config.kvDim();
        int ffnDim = config.intermediateSize();
        int expertCount = config.expertCount();
        int expertUsedCount = config.expertUsedCount();
        int expertFfnDim = config.expertFfnLength();
        int sharedFfnDim = config.expertSharedCount() * expertFfnDim;

        // qDim may differ from dim when headSize != dim/headCount (e.g., Qwen3-Coder-30B)
        int qDim = config.headCount() * config.headSize();

        // Standard buffers
        this.x = new float[dim];
        this.xb = new float[dim];
        this.xb2 = new float[Math.max(dim, qDim)];
        this.logits = new float[config.vocabSize()];

        // Attention
        this.q = new float[qDim];
        this.k = new float[kvDim];
        this.v = new float[kvDim];
        this.att = new float[config.headCount() * maxSeqLen];
        this.kvCache = new KVCache(config.blockCount(), kvDim, maxSeqLen);

        // Dense FFN
        this.hb = new float[ffnDim];
        this.hb2 = new float[ffnDim];

        // MoE
        this.xbSaved = new float[dim];
        this.routerLogits = new float[Math.max(expertCount, 1)];
        this.selectedExperts = new int[Math.max(expertUsedCount, 1)];
        this.selectedWeights = new float[Math.max(expertUsedCount, 1)];
        this.expertOut = new float[dim];
        this.sharedHb = new float[Math.max(sharedFfnDim, 1)];
        this.sharedHb2 = new float[Math.max(sharedFfnDim, 1)];

        int eu = Math.max(expertUsedCount, 1);
        int efd = Math.max(expertFfnDim, 1);
        this.moeHbPerExpert = new float[eu][efd];
        this.moeHb2PerExpert = new float[eu][efd];
        this.expertOutPerExpert = new float[eu][dim];
    }
}
