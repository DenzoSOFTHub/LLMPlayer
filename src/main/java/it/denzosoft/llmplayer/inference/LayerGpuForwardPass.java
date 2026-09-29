package it.denzosoft.llmplayer.inference;

/**
 * Contract between the hybrid/alternative engines (Qwen3.5, Nemotron-H, LFM2, Falcon-H1, Gemma 4)
 * and their GPU-resident forward passes in the java21 source root. The passes are created
 * reflectively (this Java 8 compatible code never links against them) and then called through this
 * interface, so the per-layer, per-token hot path has no {@code Method.invoke} and no boxing.
 *
 * <p>The pass owns the device-side KV cache and recurrent state of its layers.
 */
public interface LayerGpuForwardPass extends AutoCloseable {

    int getGpuLayerCount();

    void uploadXAndUpdateParams(float[] x, int position);

    void forwardLayer(int layerIdx, int position);

    void downloadX(float[] x);

    /** Final norm + output projection on the GPU; false if not possible. */
    boolean forwardFinalLogits(float[] logits);

    /** Every layer plus the output projection as one graph replay; false to fall back. */
    default boolean forwardGraph(float[] logits) { return false; }

    /** Every layer as one graph replay, no output projection (prefill); false to fall back. */
    default boolean forwardGraphPrefill() { return false; }

    default int forwardGraphArgmax() { return -1; }

    default int forwardFinalArgmax() { return -1; }

    /** Gemma 4 per-layer-embedding input for the current token. */
    default void uploadPleCombined(float[] pleCombined) {
        throw new UnsupportedOperationException("no per-layer embeddings in this pass");
    }

    /**
     * Routed-expert half of a mixture-of-experts FFN, computed by the engine on the CPU for a pass
     * that runs everything else of the layer on the GPU (Qwen3.5-MoE).
     */
    interface MoeFfn {
        /**
         * {@code out = sum_k w_k * expert_k(xn)} for {@code layer}; plus the gated shared expert
         * when {@code withShared}. {@code xn} is the FFN-normed input.
         */
        void routed(int layer, float[] xn, float[] out, boolean withShared);
    }

    /** Engine callback for the MoE layers of the pass (called from {@link #forwardLayer}). */
    default void setMoeFfn(MoeFfn ffn) { }

    /** Batched variant of {@link MoeFfn} for {@link #prefillBatch}: routed experts and the gated shared expert. */
    interface MoeFfnBatch {
        /** {@code out[t] = MoE(xn[t])} for {@code t < n}, shared expert included. */
        void routedBatch(int layer, float[][] xn, int n, float[][] out);
    }

    /** Engine callback for the MoE layers during {@link #prefillBatch}. */
    default void setMoeFfnBatch(MoeFfnBatch ffn) { }

    /** Largest chunk {@link #prefillBatch} accepts, or 0 when the pass has no batched prefill. */
    default int maxBatchTokens() { return 0; }

    /**
     * Run every layer over {@code n} prompt tokens at positions {@code basePos..basePos+n-1}
     * ({@code x[t]} their embeddings), leaving the device KV cache and recurrent state as {@code n}
     * {@link #forwardLayer} calls would, and the last token's hidden state where
     * {@link #forwardFinalLogits} reads it. Full offload only.
     */
    default void prefillBatch(float[][] x, int basePos, int n) {
        throw new UnsupportedOperationException("no batched prefill in this pass");
    }

    /** Profiling hook, called once per token. */
    default void profileTokenComplete() { }
}
