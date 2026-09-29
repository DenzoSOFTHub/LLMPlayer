package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.TransformerLayerWeights;

/**
 * Contract between {@link InferenceEngine} and a GPU-resident forward pass for dense
 * transformers ({@code CudaForwardPass}, {@code GpuForwardPass} in the java21 source root).
 *
 * <p>The implementations are still created reflectively, so this Java 8 compatible code never
 * links against them, but once created they are called through this interface: the per-token
 * hot path then has no {@code Method.invoke}, no boxed arguments and no varargs array.
 *
 * <p>The pass owns the KV cache of its layers. The GPU keeps the history of exactly one sequence;
 * see {@link InferenceEngine#gpuHoldsHistoryOf}.
 */
public interface DenseGpuForwardPass extends AutoCloseable {

    /** Layers 0..n-1 run on the GPU; the rest (partial offload) run on the CPU. */
    int getGpuLayerCount();

    /** Upload the input embedding. */
    void uploadX(float[] x);

    /** Upload the input embedding together with the token position used by the kernels. */
    default void uploadXAndUpdateParams(float[] x, int position) {
        uploadX(x);
    }

    /** Run one GPU layer (per-layer mode). */
    void forwardLayer(InferenceState state, TransformerLayerWeights layerWeights,
                      int layerIdx, int position, Attention attention);

    /** Download the residual stream after the last GPU layer. */
    void downloadX(float[] x);

    /**
     * All GPU layers plus final norm and output projection in one replay, logits downloaded into
     * {@code logits}. Only valid with every layer on the GPU. Returns false to fall back to the
     * per-layer path.
     */
    default boolean forwardGraph(float[] logits) { return false; }

    /**
     * All GPU layers in one replay, without the output projection: prefill tokens whose logits
     * are discarded, and the GPU prefix of a partial offload. Returns false to fall back.
     */
    default boolean forwardGraphLayers() { return false; }

    /** Final norm and output projection on the GPU after per-layer mode; false if not possible. */
    default boolean forwardFinalLogits(float[] logits) { return false; }

    /**
     * Graph replay of all layers plus the output projection, then an on-device argmax: only the
     * winning token id is downloaded. Returns -1 when unavailable.
     */
    default int forwardGraphArgmax() { return -1; }

    /**
     * Largest token count {@link #prefillBatch} accepts; 0 when the pass has no batched prefill.
     * Only meaningful with every layer on the GPU.
     */
    default int maxPrefillBatch() { return 0; }

    /**
     * Run every layer for {@code n} consecutive tokens (positions {@code startPos ..}) whose input
     * embeddings are packed in {@code embeddings} ({@code n * dim} floats): the weights are read
     * once per chunk instead of once per token. Writes the tokens' K/V; computes no logits.
     */
    default void prefillBatch(float[] embeddings, int startPos, int n) {
        throw new UnsupportedOperationException("no batched prefill");
    }

    /** Argmax of the logits of the last {@link #forwardFinalLogits}-style computation; -1 if unavailable. */
    default int forwardFinalArgmax() { return -1; }
}
