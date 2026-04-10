package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.DeepSeek2LayerWeights;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Mixture of Experts Feed-Forward Network for DeepSeek2, GLM-4.7-Flash, and DeepSeek-V3.
 *
 * Gating modes:
 * - expertGatingFunc=0 (default): softmax over router logits, no renormalization
 * - expertGatingFunc=2 (GLM-4.7-Flash): sigmoid per expert, with exp_probs_b bias added to logits,
 *   L2 normalization of selected weights, then multiply by expertWeightsScale
 *
 * Flow:
 * 1. Router: logits = x * ffnGateInp (+ exp_probs_b bias if present) → gating → top-K expert selection
 * 2. For each selected expert: SwiGLU with sliced 3D weight tensors
 * 3. Output = weighted sum of expert outputs + shared expert output
 */
public class MoEFFN {

    private final ModelConfig config;

    public MoEFFN(ModelConfig config) {
        this.config = config;
    }

    /**
     * Forward pass for MoE FFN.
     * Reads from xb (normalized input), writes result to xb (for residual addition).
     */
    public void forward(DeepSeek2State state, DeepSeek2LayerWeights weights) {
        int dim = config.embeddingLength();
        int expertCount = config.expertCount();
        int expertUsedCount = config.expertUsedCount();
        int expertFfnDim = config.expertFfnLength();
        int sharedFfnDim = config.expertSharedCount() * expertFfnDim;

        // 1. Router: compute raw expert logits
        float[] routerLogits = state.routerLogits;
        Arrays.fill(routerLogits, 0, expertCount, 0f);
        weights.ffnGateInp().matmul(state.xb, routerLogits, expertCount, dim);

        // 2. Gating function → probs (UNBIASED). Overwrites routerLogits in-place.
        //    See llama.cpp build_moe_ffn (llama-graph.cpp ~1230): sigmoid/softmax is applied
        //    to raw logits BEFORE any bias. The resulting `probs` is preserved unbiased for
        //    later use as the mix weights.
        int gatingFunc = config.expertGatingFunc();
        if (gatingFunc == 2) {
            // Sigmoid gating (DeepSeek-V3, GLM-4.7-Flash)
            for (int i = 0; i < expertCount; i++) {
                routerLogits[i] = 1.0f / (1.0f + (float) Math.exp(-routerLogits[i]));
            }
        } else {
            // Softmax gating (default, DeepSeek-V2)
            VectorOpsFactory.get().softmax(routerLogits, 0, expertCount);
        }

        // 3. Build selection scores: selection_probs = probs + exp_probs_b (DS-V3 only).
        //    The biased scores are used ONLY for top-K selection; the unbiased probs drive
        //    the mix weights. See llama.cpp:1249-1255: "leave probs unbiased as it's later
        //    used to get expert weights".
        float[] selectionScores;
        if (weights.expProbsBias() != null) {
            selectionScores = state.selectionScores;
            FloatTensor bias = weights.expProbsBias();
            for (int i = 0; i < expertCount; i++) {
                selectionScores[i] = routerLogits[i] + bias.getFloat(i);
            }
        } else {
            selectionScores = routerLogits;
        }

        // 4. Top-K on selection_probs; mix weights come from UNBIASED probs (routerLogits).
        int[] selectedExperts = state.selectedExperts;
        float[] selectedWeights = state.selectedWeights;
        selectTopKWithWeightSource(selectionScores, routerLogits, expertCount, expertUsedCount,
            selectedExperts, selectedWeights);

        // 5. Post-selection weight normalization (for sigmoid gating with expert_weights_norm=true).
        if (gatingFunc == 2) {
            // L2 normalization of selected weights, then scale
            // Note: llama.cpp does sum-normalization (ggml_sum_rows + ggml_div, clamped to 6.1e-5).
            // LLMPlayer uses L2 — preserved here to avoid regressing GLM-4.7-Flash.
            // This is a separate issue (see audit M/cross-cutting MoE) to be fixed once verified.
            float l2Norm = 0f;
            for (int k = 0; k < expertUsedCount; k++) {
                l2Norm += selectedWeights[k] * selectedWeights[k];
            }
            l2Norm = (float) Math.sqrt(l2Norm);
            if (l2Norm > 6.103515625e-5f) { // F16-epsilon clamp guards against NaN
                float scale = config.expertWeightsScale() / l2Norm;
                for (int k = 0; k < expertUsedCount; k++) {
                    selectedWeights[k] *= scale;
                }
            }
        }
        // Note: DeepSeek-V2 has norm_topk_prob=false, so we do NOT renormalize weights.
        // The sub-1.0 sum acts as intentional gating.

        // 2. Compute routed expert outputs — parallel across experts
        Arrays.fill(state.xb, 0);  // Will accumulate weighted expert outputs here

        IntStream.range(0, expertUsedCount).parallel().forEach(k -> {
            int e = selectedExperts[k];

            // Per-expert buffers (no contention)
            float[] gate = state.moeHbPerExpert[k];
            float[] up = state.moeHb2PerExpert[k];
            float[] out = state.expertOutPerExpert[k];

            Arrays.fill(gate, 0, expertFfnDim, 0f);
            Arrays.fill(up, 0, expertFfnDim, 0f);

            expertMatmul(weights.ffnGateExps(), state.xbSaved, gate, e, dim, expertFfnDim);
            expertMatmul(weights.ffnUpExps(), state.xbSaved, up, e, dim, expertFfnDim);

            VectorOpsFactory.get().silu(gate, expertFfnDim);
            VectorOpsFactory.get().elementwiseMul(gate, up, gate, expertFfnDim);

            Arrays.fill(out, 0, dim, 0f);
            expertMatmul(weights.ffnDownExps(), gate, out, e, expertFfnDim, dim);
        });

        // Sequential accumulation of weighted expert outputs
        for (int k = 0; k < expertUsedCount; k++) {
            VectorOpsFactory.get().saxpy(selectedWeights[k], state.expertOutPerExpert[k], 0, state.xb, 0, dim);
        }

        // 3. Shared expert: standard SwiGLU with shared weights
        float[] shGate = state.sharedHb;
        float[] shUp = state.sharedHb2;
        Arrays.fill(shGate, 0, sharedFfnDim, 0f);
        Arrays.fill(shUp, 0, sharedFfnDim, 0f);

        weights.ffnGateShexp().matmulParallel(state.xbSaved, shGate, sharedFfnDim, dim);
        weights.ffnUpShexp().matmulParallel(state.xbSaved, shUp, sharedFfnDim, dim);

        VectorOpsFactory.get().silu(shGate, sharedFfnDim);
        VectorOpsFactory.get().elementwiseMul(shGate, shUp, shGate, sharedFfnDim);

        float[] sharedOut = state.expertOut;
        Arrays.fill(sharedOut, 0, dim, 0f);
        weights.ffnDownShexp().matmulParallel(shGate, sharedOut, dim, sharedFfnDim);

        // Add shared expert output
        VectorOpsFactory.get().accumulate(state.xb, sharedOut, dim);
    }

    /**
     * Matrix-vector multiply for a single expert slice from a 3D tensor.
     * The 3D tensor has shape [inDim, outDim, numExperts] in GGUF layout.
     * Expert e starts at offset e * outDim * inDim.
     */
    private static void expertMatmul(FloatTensor weights3D, float[] input, float[] output,
                                      int expert, int inDim, int outDim) {
        long expertOffset = (long) expert * outDim * inDim;
        for (int row = 0; row < outDim; row++) {
            output[row] += weights3D.dot(expertOffset + (long) row * inDim, input, 0, inDim);
        }
    }

    /**
     * Select top-K indices from {@code scores} and return the corresponding values from
     * {@code weightSource}. When {@code scores == weightSource} this behaves like a classic
     * top-K (old behavior); when they differ, the top-K is decided by scores but the stored
     * values come from weightSource — used to implement DS-V3's "biased selection, unbiased
     * mix weights" pattern.
     */
    private static void selectTopKWithWeightSource(float[] scores, float[] weightSource,
                                                    int n, int k,
                                                    int[] outIndices, float[] outValues) {
        Arrays.fill(outIndices, 0, k, -1);
        // Track the min score *within the selected set* to decide replacement
        float[] selectedScores = new float[k];
        Arrays.fill(selectedScores, Float.NEGATIVE_INFINITY);
        Arrays.fill(outValues, 0, k, Float.NEGATIVE_INFINITY);

        int minPos = 0;
        float minVal = Float.NEGATIVE_INFINITY;

        for (int i = 0; i < n; i++) {
            if (scores[i] > minVal) {
                selectedScores[minPos] = scores[i];
                outIndices[minPos] = i;
                outValues[minPos] = weightSource[i];
                // Rescan for new minimum of the selected-score set
                minPos = 0;
                minVal = selectedScores[0];
                for (int j = 1; j < k; j++) {
                    if (selectedScores[j] < minVal) {
                        minPos = j;
                        minVal = selectedScores[j];
                    }
                }
            }
        }
    }
}
