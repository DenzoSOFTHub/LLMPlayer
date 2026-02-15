package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.model.TransformerLayerWeights;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

import java.util.Arrays;

/**
 * Feed-Forward Network with SwiGLU activation.
 * Standard: FFN(x) = wDown * (silu(wGate * x) * (wUp * x))
 * Packed (GLM4): wUp produces [2*ffnDim], first half is gate, second half is up.
 */
public class SwiGLUFFN {

    private final ModelConfig config;

    public SwiGLUFFN(ModelConfig config) {
        this.config = config;
    }

    /**
     * Forward pass. Reads from xb (normalized input), writes result to xb (for residual addition).
     */
    public void forward(InferenceState state, TransformerLayerWeights weights) {
        int dim = config.embeddingLength();
        int ffnDim = config.intermediateSize();

        if (weights.wGate() != null) {
            // Standard mode (Llama, Qwen): gate and up are separate tensors
            // gate = wGate * xb
            Arrays.fill(state.hb, 0);
            weights.wGate().matmulParallel(state.xb, state.hb, ffnDim, dim);

            // up = wUp * xb
            Arrays.fill(state.hb2, 0);
            weights.wUp().matmulParallel(state.xb, state.hb2, ffnDim, dim);
        } else {
            // Packed mode (GLM4): wUp has shape [2*ffnDim, dim]
            // Single matmul produces [gate, up] concatenated
            Arrays.fill(state.hbPacked, 0);
            weights.wUp().matmulParallel(state.xb, state.hbPacked, 2 * ffnDim, dim);

            // Split: first half = gate (hb), second half = up (hb2)
            System.arraycopy(state.hbPacked, 0, state.hb, 0, ffnDim);
            System.arraycopy(state.hbPacked, ffnDim, state.hb2, 0, ffnDim);
        }

        // hb = silu(gate) * up
        VectorOpsFactory.get().silu(state.hb, ffnDim);
        VectorOpsFactory.get().elementwiseMul(state.hb, state.hb2, state.hb, ffnDim);

        // xb = wDown * hb
        Arrays.fill(state.xb, 0);
        weights.wDown().matmulParallel(state.hb, state.xb, dim, ffnDim);
    }
}
