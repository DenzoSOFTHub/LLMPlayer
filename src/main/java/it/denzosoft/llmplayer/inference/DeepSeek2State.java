package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.ModelConfig;

/**
 * Pre-allocated mutable state for DeepSeek2 inference.
 * DeepSeek2 has different buffer requirements due to MLA attention and MoE FFN:
 * - Separate key/value cache dimensions (keyLength != valueLength)
 * - MLA intermediate buffers (compressed KV, decompressed KV)
 * - MoE routing and expert buffers
 */
public class DeepSeek2State {

    // Standard transformer buffers
    public final float[] x;      // [dim]
    public final float[] xb;     // [dim] - activation after norm, also accumulator for MoE output
    public final float[] xb2;    // [headCount * valueLength] - attention output before Wo projection
    public final float[] logits; // [vocabSize]

    // MLA attention buffers
    public final float[] q;              // [headCount * keyLength]
    public final float[] k;              // [headCount * keyLength] - assembled full key
    public final float[] v;              // [headCount * valueLength]
    public final float[] kvCompressed;   // [kvLoraRank + ropeDim] - output of wkvA
    public final float[] kvLatentNormed; // [kvLoraRank] - normed latent
    public final float[] kvDecompressed; // [headCount * (keyNope + valueLen)] - output of wkvB
    public final float[] kRopeTemp;      // [ropeDim] - temporary for k_rope rotation
    public final float[] att;            // [headCount * maxSeqLen]

    // KV cache: separate key and value dimensions
    public final float[][] keyCache;     // [layers][maxSeqLen * headCount * keyLength]
    public final float[][] valueCache;   // [layers][maxSeqLen * headCount * valueLength]

    // Dense FFN buffers (for leading dense blocks)
    public final float[] hb;    // [ffnDim]
    public final float[] hb2;   // [ffnDim]

    // MoE buffers
    public final float[] xbSaved;        // [dim] - copy of xb for MoE (since xb is reused as output)
    public final float[] routerLogits;   // [expertCount]
    public final int[] selectedExperts;  // [expertUsedCount]
    public final float[] selectedWeights;// [expertUsedCount]
    public final float[] moeHb;          // [expertFfnLength] - per-expert gate/silu buffer
    public final float[] moeHb2;         // [expertFfnLength] - per-expert up buffer
    public final float[] expertOut;      // [dim] - single expert output
    public final float[] sharedHb;       // [sharedFfnDim] - shared expert gate buffer
    public final float[] sharedHb2;      // [sharedFfnDim] - shared expert up buffer
    // Per-expert parallel buffers
    public final float[][] moeHbPerExpert;   // [expertUsedCount][expertFfnDim]
    public final float[][] moeHb2PerExpert;  // [expertUsedCount][expertFfnDim]
    public final float[][] expertOutPerExpert; // [expertUsedCount][dim]

    public DeepSeek2State(ModelConfig config, int maxSeqLen) {
        int dim = config.embeddingLength();
        int headCount = config.headCount();
        int keyLength = config.keyLength();
        int valueLength = config.valueLength();
        int kvLoraRank = config.kvLoraRank();
        int ropeDim = config.ropeDimensionCount();
        int keyNope = keyLength - ropeDim;
        int ffnDim = config.intermediateSize();
        int expertCount = config.expertCount();
        int expertUsedCount = config.expertUsedCount();
        int expertFfnDim = config.expertFfnLength();
        int sharedFfnDim = config.expertSharedCount() * expertFfnDim;
        int blockCount = config.blockCount();

        int totalKeyDim = headCount * keyLength;
        int totalValDim = headCount * valueLength;

        // Standard
        this.x = new float[dim];
        this.xb = new float[dim];
        this.xb2 = new float[totalValDim];
        this.logits = new float[config.vocabSize()];

        // MLA
        this.q = new float[totalKeyDim];
        this.k = new float[totalKeyDim];
        this.v = new float[totalValDim];
        this.kvCompressed = new float[kvLoraRank + ropeDim];
        this.kvLatentNormed = new float[kvLoraRank];
        this.kvDecompressed = new float[headCount * (keyNope + valueLength)];
        this.kRopeTemp = new float[ropeDim];
        this.att = new float[headCount * maxSeqLen];

        // KV caches
        this.keyCache = new float[blockCount][maxSeqLen * totalKeyDim];
        this.valueCache = new float[blockCount][maxSeqLen * totalValDim];

        // Dense FFN
        this.hb = new float[ffnDim];
        this.hb2 = new float[ffnDim];

        // MoE
        this.xbSaved = new float[dim];
        this.routerLogits = new float[Math.max(expertCount, 1)];
        this.selectedExperts = new int[Math.max(expertUsedCount, 1)];
        this.selectedWeights = new float[Math.max(expertUsedCount, 1)];
        this.moeHb = new float[Math.max(expertFfnDim, 1)];
        this.moeHb2 = new float[Math.max(expertFfnDim, 1)];
        this.expertOut = new float[dim];
        this.sharedHb = new float[Math.max(sharedFfnDim, 1)];
        this.sharedHb2 = new float[Math.max(sharedFfnDim, 1)];

        // Per-expert buffers for parallel MoE execution
        int eu = Math.max(expertUsedCount, 1);
        int efd = Math.max(expertFfnDim, 1);
        this.moeHbPerExpert = new float[eu][efd];
        this.moeHb2PerExpert = new float[eu][efd];
        this.expertOutPerExpert = new float[eu][dim];
    }
}
