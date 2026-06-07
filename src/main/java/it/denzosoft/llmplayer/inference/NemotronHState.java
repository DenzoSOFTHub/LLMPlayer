package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.ModelConfig;

/**
 * Mutable state for Nemotron-H inference.
 * Contains activation buffers, KV cache for attention layers,
 * Mamba-2 recurrent state and conv1d buffers for Mamba layers.
 */
public class NemotronHState {
    public final float[] x;       // [dim] current activation
    public final float[] xb;      // [dim] after norm
    public final float[] logits;  // [vocabSize]

    // Mamba-2 buffers
    public final float[] zxBCdt;  // [ssmProjectionDim] ssm_in output (z + xBC + dt concatenated)
    public final float[] xBC;     // [ssmInnerSize + 2*ssmGroupCount*ssmStateSize] after conv+SiLU
    public final float[] ssm_x;   // [ssmInnerSize] x portion of xBC
    public final float[] ssm_y;   // [ssmInnerSize] SSM output
    public final float[] convResult; // [convChannels] — pre-allocated conv1d output buffer (E21)

    // Mamba-2 persistent state per Mamba layer
    // ssmState[layer][head][stateSize] — recurrent state
    public final float[][][] ssmState;
    // convState[layer][histSize][convChannels] — conv1d circular buffer
    public final float[][][] convState;
    public final int[] convStatePos;

    // Attention buffers
    public final float[] q;
    public final float[] k;
    public final float[] v;
    public final float[] att;
    public final float[] xb2;     // attention output
    public final KVCache kvCache;

    // FFN buffers
    public final float[] hb;      // [ffnDim] after ffn_up

    // Granite Hybrid MoE buffers (null when not MoE)
    public final float[] routerLogits;       // [expertCount]
    public final int[] selectedExperts;      // [expertUsedCount]
    public final float[] selectedWeights;    // [expertUsedCount]
    public final float[][] moeGatePerExpert; // [expertUsedCount][expertFfnDim]
    public final float[][] moeUpPerExpert;   // [expertUsedCount][expertFfnDim]
    public final float[][] moeOutPerExpert;  // [expertUsedCount][dim]
    public final float[] sharedHb;           // [sharedFfnDim]
    public final float[] sharedHb2;          // [sharedFfnDim]
    public final float[] moeOut;             // [dim] accumulated MoE output

    private final int blockCount;
    private final ModelConfig config;

    public NemotronHState(ModelConfig config, int maxSeqLen) {
        this.config = config;
        this.blockCount = config.blockCount();

        int dim = config.embeddingLength();
        int vocabSize = config.vocabSize();
        int ssmInnerSize = config.ssmInnerSize();
        int ssmStateSize = config.ssmStateSize();
        int ssmGroupCount = config.ssmGroupCount();
        int ssmTimeStepRank = config.ssmTimeStepRank();
        int ssmConvKernel = config.ssmConvKernel();
        int headCount = config.headCount();
        int headSize = config.headSize();

        // z + xBC + dt = ssmInnerSize + (ssmInnerSize + 2*ssmGroupCount*ssmStateSize) + ssmTimeStepRank
        int convChannels = ssmInnerSize + 2 * ssmGroupCount * ssmStateSize;
        int projDim = ssmInnerSize + convChannels + ssmTimeStepRank;

        this.x = new float[dim];
        this.xb = new float[dim];
        this.logits = new float[vocabSize];

        // Mamba buffers
        this.zxBCdt = new float[projDim];
        this.xBC = new float[convChannels];
        this.ssm_x = new float[ssmInnerSize];
        this.ssm_y = new float[ssmInnerSize];
        this.convResult = new float[convChannels]; // E21: pre-allocated, no per-token alloc

        // Attention buffers
        int qDim = headCount * headSize;
        int kvDim = config.kvDim();
        this.q = new float[qDim];
        this.k = new float[kvDim];
        this.v = new float[kvDim];
        this.att = new float[headCount * maxSeqLen];
        this.xb2 = new float[Math.max(qDim, dim)];
        // KV cache honors -Dkv.q8=true for the attention-layer KV
        KVCache.Mode nhKvMode = "true".equals(System.getProperty("kv.q8"))
            && (kvDim % KVCache.Q8_BLOCK == 0)
            ? KVCache.Mode.Q8_0 : KVCache.Mode.FLOAT32;
        this.kvCache = new KVCache(blockCount, kvDim, maxSeqLen, nhKvMode);

        // FFN buffers
        this.hb = new float[config.intermediateSize()];

        // MoE buffers (Granite Hybrid MoE)
        int expertCount = config.expertCount();
        if (expertCount > 0) {
            int eu = config.expertUsedCount();
            int eFfn = config.expertFfnLength() > 0 ? config.expertFfnLength() : config.intermediateSize();
            int shFfn = Math.max(config.expertSharedFeedForwardLength(), 1);
            this.routerLogits = new float[expertCount];
            this.selectedExperts = new int[eu];
            this.selectedWeights = new float[eu];
            this.moeGatePerExpert = new float[eu][eFfn];
            this.moeUpPerExpert = new float[eu][eFfn];
            this.moeOutPerExpert = new float[eu][dim];
            this.sharedHb = new float[shFfn];
            this.sharedHb2 = new float[shFfn];
            this.moeOut = new float[dim];
        } else {
            this.routerLogits = null; this.selectedExperts = null; this.selectedWeights = null;
            this.moeGatePerExpert = null; this.moeUpPerExpert = null; this.moeOutPerExpert = null;
            this.sharedHb = null; this.sharedHb2 = null; this.moeOut = null;
        }

        // Persistent Mamba state
        int histSize = ssmConvKernel - 1;
        this.ssmState = new float[blockCount][][];
        this.convState = new float[blockCount][][];
        this.convStatePos = new int[blockCount];

        for (int i = 0; i < blockCount; i++) {
            if (config.nemotronLayerType(i) == 0) { // Mamba
                // State per head: [nheads][headDim * stateSize] — each head_dim element has independent state
                int headDim = ssmInnerSize / ssmTimeStepRank;
                this.ssmState[i] = new float[ssmTimeStepRank][headDim * ssmStateSize];
                this.convState[i] = new float[histSize][convChannels];
            }
        }
    }
}
