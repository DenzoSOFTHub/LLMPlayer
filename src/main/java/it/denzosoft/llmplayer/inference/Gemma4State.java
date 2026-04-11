package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.ModelConfig;

/**
 * Inference state for Gemma 4 architecture.
 * Handles dual headSize: SWA layers use headSize=256, full attention layers use headSize=512.
 * Buffers are allocated at the MAX size to accommodate both layer types.
 * KV cache uses per-layer dimensions (different kvDim per layer type).
 */
public class Gemma4State extends InferenceState {
    // Additional buffer for attention residual (attn_out = attention_result + x)
    public final float[] xb3;

    // PLE buffers
    public final float[] pleEmb;        // [pleDim * blockCount] — token-identity embedding
    public final float[] pleProjected;  // [pleDim * blockCount] — context-aware projection
    public final float[] pleCombined;   // [pleDim * blockCount] — combined PLE input
    public final float[] pleGate;       // [pleDim] — per-layer gate values
    public final float[] pleOut;        // [dim] — per-layer PLE output

    // Per-layer KV cache with variable dimensions (SWA vs full attention)
    public final Gemma4KVCache gemma4KvCache;

    // Override Q/K/V/xb2 with larger buffers for full attention layers
    public final float[] qLarge;
    public final float[] kLarge;
    public final float[] vLarge;
    public final float[] xb2Large;

    // H10: AltUp 4-stream activation buffers + scratch.
    // streams[s][d] = stream s of dim d. nAltup typically 4 for Gemma 3n.
    public final float[][] altupStreams;        // [nAltup][dim]
    public final float[][] altupPredictions;    // [nAltup][dim] — output of altup_predict
    public final float[][] altupCorrected;      // [nAltup][dim] — output of altup_correct
    public final float[] altupRouterInput;      // [dim] — RMSNorm of active stream for router
    public final float[] altupModalities;       // [nAltup] — tanh(router @ active_normed)
    public final float[] altupAllCoefs;         // [nAltup * nAltup] — predict OR correct coefs
    public final float[] altupInnovation;       // [dim] — actual - active_prediction
    public final float[] laurelTmpRank;         // [laurel_rank] — intermediate of laurel_l @ x
    public final float[] laurelOut;             // [dim] — laurel(x)
    public final float[] firstPredAltup;        // [pleDim] — first_prediction in altup space
    public final float[] firstPredFullDim;      // [dim]    — first_prediction back-projected

    public Gemma4State(ModelConfig config, int maxSeqLen, int pleDim, int maxQDim, int maxKvDim) {
        this(config, maxSeqLen, pleDim, maxQDim, maxKvDim, 0, 0);
    }

    public Gemma4State(ModelConfig config, int maxSeqLen, int pleDim, int maxQDim, int maxKvDim,
                       int nAltup, int laurelRank) {
        super(config, maxSeqLen);
        int dim = config.embeddingLength();
        int blockCount = config.blockCount();

        // Allocate larger Q/K/V/xb2 for full attention layers
        this.qLarge = new float[maxQDim];
        this.kLarge = new float[maxKvDim];
        this.vLarge = new float[maxKvDim];
        this.xb2Large = new float[maxQDim];
        this.xb3 = new float[dim];

        int totalPleDim = pleDim * blockCount;
        this.pleEmb = totalPleDim > 0 ? new float[totalPleDim] : new float[0];
        this.pleProjected = totalPleDim > 0 ? new float[totalPleDim] : new float[0];
        this.pleCombined = totalPleDim > 0 ? new float[totalPleDim] : new float[0];
        this.pleGate = pleDim > 0 ? new float[pleDim] : new float[0];
        this.pleOut = new float[dim];

        // Create per-layer KV cache with variable dimensions
        this.gemma4KvCache = new Gemma4KVCache(config, maxSeqLen);

        // H10: AltUp + Laurel buffers (allocated only when nAltup > 0)
        if (nAltup > 0) {
            this.altupStreams     = new float[nAltup][dim];
            this.altupPredictions = new float[nAltup][dim];
            this.altupCorrected   = new float[nAltup][dim];
            this.altupRouterInput = new float[dim];
            this.altupModalities  = new float[nAltup];
            this.altupAllCoefs    = new float[nAltup * nAltup];
            this.altupInnovation  = new float[dim];
            this.laurelTmpRank    = laurelRank > 0 ? new float[laurelRank] : new float[0];
            this.laurelOut        = new float[dim];
            this.firstPredAltup   = pleDim > 0 ? new float[pleDim] : new float[0];
            this.firstPredFullDim = new float[dim];
        } else {
            this.altupStreams = null;
            this.altupPredictions = null;
            this.altupCorrected = null;
            this.altupRouterInput = null;
            this.altupModalities = null;
            this.altupAllCoefs = null;
            this.altupInnovation = null;
            this.laurelTmpRank = null;
            this.laurelOut = null;
            this.firstPredAltup = null;
            this.firstPredFullDim = null;
        }
    }

    /**
     * KV cache with per-layer variable dimensions.
     * SWA layers use kvDim=headCountKV*headSizeSwa, full layers use kvDim=headCountKV*headSizeFull.
     */
    public static class Gemma4KVCache {
        private final float[][] keyLayers;
        private final float[][] valueLayers;
        private final int[] kvDimPerLayer;
        private final int maxSeqLen;

        public Gemma4KVCache(ModelConfig config, int maxSeqLen) {
            int blockCount = config.blockCount();
            int headCountKV = config.headCountKV();
            int headSizeSwa = config.headSize();  // SWA headSize (256)
            int headSizeFull = config.keyLength() > 0 ? config.keyLength() : headSizeSwa; // full headSize (512)
            boolean[] swaPattern = config.slidingWindowPattern();
            this.maxSeqLen = maxSeqLen;

            keyLayers = new float[blockCount][];
            valueLayers = new float[blockCount][];
            kvDimPerLayer = new int[blockCount];

            for (int i = 0; i < blockCount; i++) {
                boolean isSwa = (swaPattern != null && i < swaPattern.length) ? swaPattern[i] : (i % 6 != 5);
                int kvDim = headCountKV * (isSwa ? headSizeSwa : headSizeFull);
                kvDimPerLayer[i] = kvDim;
                keyLayers[i] = new float[maxSeqLen * kvDim];
                valueLayers[i] = new float[maxSeqLen * kvDim];
            }
        }

        public float[] keyLayer(int layer) { return keyLayers[layer]; }
        public float[] valueLayer(int layer) { return valueLayers[layer]; }
        public int kvDim(int layer) { return kvDimPerLayer[layer]; }
        public int offset(int position) { return position; } // offset is position * kvDim — caller multiplies
    }
}
