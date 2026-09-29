package it.denzosoft.llmplayer.inference;

/**
 * GPU-resident attention half of an MoE layer, for engines whose FFN (router + experts) stays on
 * the CPU. Implemented in the java21 source root and created reflectively.
 *
 * <p>The pass owns the KV cache of the layers it runs; see {@link GpuFailureException}.
 */
public interface GpuAttentionPass extends AutoCloseable {

    /** Whether {@code layer}'s attention runs on the GPU (its weights are GPU-resident). */
    boolean isLayerOnGpu(int layer);

    /**
     * {@code x += Attention(RMSNorm(x))} for {@code layer} at {@code position}, then
     * {@code xbOut = FFNNorm(x)}. {@code x} is read and written; {@code xbOut} is written.
     */
    void attentionLayer(int layer, float[] x, float[] xbOut, int position);

    /**
     * When the pass queued {@code layer}'s shared expert on the FFN-normed input at the end of the
     * last {@link #attentionLayer} (after its download, so the GPU computes it while the CPU runs
     * the router and the routed experts), downloads its output into {@code out} and returns true;
     * otherwise returns false and the engine computes it. Call it after the routed experts: the
     * download synchronises the stream. A failure here is a GPU failure of the pass (the engine
     * must treat it like one from {@link #attentionLayer}).
     */
    default boolean takeSharedExpert(int layer, float[] out) { return false; }

    /** Largest token count {@link #attentionLayerBatch} accepts; 0 when the pass has no batched prefill. */
    default int maxBatchTokens() { return 0; }

    /**
     * Batched prefill: for tokens {@code t < n} at positions {@code basePos + t},
     * {@code x[t] += Attention(RMSNorm(x[t]))} over the device KV cache (which receives their K/V),
     * then {@code xbOut[t] = FFNNorm(x[t])}. The tokens attend causally to each other and to every
     * earlier position.
     */
    default void attentionLayerBatch(int layer, float[][] x, float[][] xbOut, int basePos, int n) {
        throw new UnsupportedOperationException("no batched prefill in this pass");
    }
}
