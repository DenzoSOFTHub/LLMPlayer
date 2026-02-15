package it.denzosoft.llmplayer.inference;

/**
 * Key-Value cache for autoregressive generation.
 * Pre-allocated per layer, stores projected K and V vectors for all past positions.
 */
public class KVCache {

    private final float[][] keyCache;   // [layer][position * kvDim]
    private final float[][] valueCache; // [layer][position * kvDim]
    private final int kvDim;
    private final int maxSeqLen;

    public KVCache(int blockCount, int kvDim, int maxSeqLen) {
        this.kvDim = kvDim;
        this.maxSeqLen = maxSeqLen;
        this.keyCache = new float[blockCount][maxSeqLen * kvDim];
        this.valueCache = new float[blockCount][maxSeqLen * kvDim];
    }

    public float[] keyLayer(int layer) { return keyCache[layer]; }
    public float[] valueLayer(int layer) { return valueCache[layer]; }

    public int getKvDim() { return kvDim; }
    public int getMaxSeqLen() { return maxSeqLen; }

    /**
     * Get offset into the cache for a given position.
     */
    public int offset(int position) {
        return position * kvDim;
    }
}
