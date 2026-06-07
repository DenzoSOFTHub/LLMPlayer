package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.ModelConfig;

/**
 * Mutable per-sequence state for LFM2 inference: activation buffers, attention KV cache
 * (attention layers), and per-conv-layer rolling short-conv state.
 */
public class LFM2State {
    public final float[] x;        // [dim] residual stream
    public final float[] nrm;      // [dim] operator/ffn norm output
    public final float[] out;      // [dim] mixer / ffn output (added back to x)
    public final float[] logits;   // [vocabSize]

    // short-conv buffers
    public final float[] bcx;      // [3*dim] in_proj output
    public final float[] bx;       // [dim] b*x gate
    public final float[] convOut;  // [dim] conv result
    // conv rolling state: convState[layer][lCache-1][dim], only allocated for conv layers
    public final float[][][] convState;
    public final int[] convStatePos;

    // attention buffers
    public final float[] q;        // [headCount*headSize]
    public final float[] k;        // [kvDim]
    public final float[] v;        // [kvDim]
    public final float[] att;      // [headCount*maxSeqLen]
    public final float[] attOut;   // [headCount*headSize]
    public final KVCache kvCache;

    // ffn buffers
    public final float[] gate;     // [ffnDim]
    public final float[] up;       // [ffnDim]

    public LFM2State(ModelConfig config, int maxSeqLen) {
        int dim = config.embeddingLength();
        int vocabSize = config.vocabSize();
        int headCount = config.headCount();
        int headSize = config.headSize();
        int kvDim = config.kvDim();
        int ffnDim = config.intermediateSize();
        int blockCount = config.blockCount();
        int lCache = config.ssmConvKernel();         // shortconv l_cache

        this.x = new float[dim];
        this.nrm = new float[dim];
        this.out = new float[dim];
        this.logits = new float[vocabSize];

        this.bcx = new float[3 * dim];
        this.bx = new float[dim];
        this.convOut = new float[dim];

        int qDim = headCount * headSize;
        this.q = new float[qDim];
        this.k = new float[kvDim];
        this.v = new float[kvDim];
        this.att = new float[headCount * maxSeqLen];
        this.attOut = new float[qDim];

        KVCache.Mode kvMode = "true".equals(System.getProperty("kv.q8"))
            && (kvDim % KVCache.Q8_BLOCK == 0) ? KVCache.Mode.Q8_0 : KVCache.Mode.FLOAT32;
        this.kvCache = new KVCache(blockCount, kvDim, maxSeqLen, kvMode);

        this.gate = new float[ffnDim];
        this.up = new float[ffnDim];

        int hist = Math.max(1, lCache - 1);
        this.convState = new float[blockCount][][];
        this.convStatePos = new int[blockCount];
        for (int i = 0; i < blockCount; i++) {
            if (!config.lfm2IsAttentionLayer(i)) {
                this.convState[i] = new float[hist][dim];
            }
        }
    }
}
