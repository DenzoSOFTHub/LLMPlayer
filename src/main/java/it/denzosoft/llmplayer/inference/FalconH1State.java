package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.ModelConfig;

/**
 * Mutable per-sequence state for Falcon-H1. Each layer runs attention + Mamba-2 in parallel,
 * so every layer holds both an SSM recurrent state / conv ring buffer and shares the KV cache.
 */
public class FalconH1State {
    public final float[] x;        // [dim] residual stream
    public final float[] nrm;      // [dim] shared attn_norm output (fed to both paths)
    public final float[] attnOut;  // [dim] attention path output
    public final float[] ssmOut;   // [dim] mamba path output
    public final float[] logits;

    // mamba-2 buffers
    public final float[] zxBCdt;   // [ssmInner + convChannels + nheads]
    public final float[] xBC;      // [convChannels]
    public final float[] ssm_x;    // [ssmInner]
    public final float[] ssm_y;    // [ssmInner]
    public final float[] convResult;
    public final float[][][] ssmState;   // [layer][nheads][headDim*stateSize]
    public final float[][][] convState;  // [layer][hist][convChannels]
    public final int[] convStatePos;

    // attention buffers
    public final float[] q, k, v, att, attBuf;
    public final KVCache kvCache;

    // ffn buffers
    public final float[] gate, up;

    public FalconH1State(ModelConfig config, int maxSeqLen) {
        int dim = config.embeddingLength();
        int vocabSize = config.vocabSize();
        int blockCount = config.blockCount();
        int ssmInner = config.ssmInnerSize();
        int ssmState_ = config.ssmStateSize();
        int ssmGroups = config.ssmGroupCount();
        int nheads = config.ssmTimeStepRank();
        int ssmConv = config.ssmConvKernel();
        int headDim = ssmInner / nheads;
        int convChannels = ssmInner + 2 * ssmGroups * ssmState_;
        int projDim = ssmInner + convChannels + nheads;

        int headCount = config.headCount();
        int headSize = config.headSize();
        int kvDim = config.kvDim();
        int ffnDim = config.intermediateSize();

        this.x = new float[dim];
        this.nrm = new float[dim];
        this.attnOut = new float[dim];
        this.ssmOut = new float[dim];
        this.logits = new float[vocabSize];

        this.zxBCdt = new float[projDim];
        this.xBC = new float[convChannels];
        this.ssm_x = new float[ssmInner];
        this.ssm_y = new float[ssmInner];
        this.convResult = new float[convChannels];

        int qDim = headCount * headSize;
        this.q = new float[qDim];
        this.k = new float[kvDim];
        this.v = new float[kvDim];
        this.att = new float[headCount * maxSeqLen];
        this.attBuf = new float[qDim];
        KVCache.Mode kvMode = "true".equals(System.getProperty("kv.q8"))
            && (kvDim % KVCache.Q8_BLOCK == 0) ? KVCache.Mode.Q8_0 : KVCache.Mode.FLOAT32;
        this.kvCache = new KVCache(blockCount, kvDim, maxSeqLen, kvMode);

        this.gate = new float[ffnDim];
        this.up = new float[ffnDim];

        int hist = ssmConv - 1;
        this.ssmState = new float[blockCount][nheads][headDim * ssmState_];
        this.convState = new float[blockCount][hist][convChannels];
        this.convStatePos = new int[blockCount];
    }
}
