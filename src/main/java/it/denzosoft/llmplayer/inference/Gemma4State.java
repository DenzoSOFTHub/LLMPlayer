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

    public Gemma4State(ModelConfig config, int maxSeqLen, int pleDim, int maxQDim, int maxKvDim) {
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
