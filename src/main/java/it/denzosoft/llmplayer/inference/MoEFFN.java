package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.DeepSeek2LayerWeights;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Mixture of Experts Feed-Forward Network for DeepSeek2.
 *
 * Flow:
 * 1. Router: logits = x * ffnGateInp → softmax → top-K expert selection
 * 2. For each selected expert: SwiGLU with sliced 3D weight tensors
 * 3. Output = weighted sum of expert outputs + shared expert output
 *
 * The 3D expert tensors (ffnGateExps, ffnUpExps, ffnDownExps) contain all experts
 * packed along the 3rd dimension. Expert e's weights start at offset e * dim0 * dim1.
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

        // 1. Router: compute expert logits and select top-K
        float[] routerLogits = state.routerLogits;
        Arrays.fill(routerLogits, 0, expertCount, 0f);
        weights.ffnGateInp().matmul(state.xb, routerLogits, expertCount, dim);

        // Softmax over all expert logits
        VectorOpsFactory.get().softmax(routerLogits, 0, expertCount);

        // Select top-K experts
        int[] selectedExperts = state.selectedExperts;
        float[] selectedWeights = state.selectedWeights;
        selectTopK(routerLogits, expertCount, expertUsedCount, selectedExperts, selectedWeights);

        // Note: DeepSeek-V2 has norm_topk_prob=false, so we do NOT renormalize weights.
        // The sub-1.0 sum acts as intentional gating (tokens that don't strongly match
        // any expert get attenuated FFN output). DeepSeek-V3 uses renormalization.

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
     * Select top-K indices and values from logits.
     */
    private static void selectTopK(float[] logits, int n, int k,
                                    int[] outIndices, float[] outValues) {
        // Simple selection: find k largest values
        Arrays.fill(outIndices, 0, k, -1);
        Arrays.fill(outValues, 0, k, Float.NEGATIVE_INFINITY);

        for (int i = 0; i < n; i++) {
            // Find the minimum in our top-K buffer
            int minIdx = 0;
            for (int j = 1; j < k; j++) {
                if (outValues[j] < outValues[minIdx]) {
                    minIdx = j;
                }
            }
            // Replace if current is larger
            if (logits[i] > outValues[minIdx]) {
                outValues[minIdx] = logits[i];
                outIndices[minIdx] = i;
            }
        }
    }
}
