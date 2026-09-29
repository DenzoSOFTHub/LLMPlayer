package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.tensor.FloatTensor;

/**
 * GPU expert FFN for an MoE layer (router top-K on the CPU, routed + shared experts on the GPU).
 * Implemented in the java21 source root and created reflectively; called through this interface
 * so the per-layer hot path has no {@code Method.invoke} and no boxing.
 */
public interface GpuMoeExperts extends AutoCloseable {

    /**
     * @param input   normed FFN input, length dim
     * @param sel     selected expert indices (first {@code used})
     * @param weights routing weights of the selected experts
     * @param out     overwritten with routed + shared expert output, length dim
     */
    void computeMoE(FloatTensor gateExps, FloatTensor upExps, FloatTensor downExps,
                    FloatTensor gateShexp, FloatTensor upShexp, FloatTensor downShexp,
                    float[] input, int[] sel, float[] weights, int used, float[] out);
}
