package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.ModelArchitecture;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.model.TransformerLayerWeights;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

import java.util.Arrays;

/**
 * Feed-Forward Network with SwiGLU or GeGLU activation.
 * SwiGLU (Llama, Qwen, etc.): FFN(x) = wDown * (silu(wGate * x) * (wUp * x))
 * GeGLU (Gemma2/3):            FFN(x) = wDown * (gelu(wGate * x) * (wUp * x))
 * Packed (GLM4): wUp produces [2*ffnDim], first half is gate, second half is up.
 */
public class SwiGLUFFN {

    private static final float SQRT_2_OVER_PI = (float) Math.sqrt(2.0 / Math.PI);

    private final ModelConfig config;
    private final boolean useGelu;

    public SwiGLUFFN(ModelConfig config) {
        this.config = config;
        // Gemma2/3 uses GELU activation instead of SiLU
        this.useGelu = config.architecture() == ModelArchitecture.GEMMA2
                     || config.architecture() == ModelArchitecture.GEMMA3;
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

        // hb = activation(gate) * up
        if (useGelu) {
            gelu(state.hb, ffnDim);
        } else {
            VectorOpsFactory.get().silu(state.hb, ffnDim);
        }
        VectorOpsFactory.get().elementwiseMul(state.hb, state.hb2, state.hb, ffnDim);

        // xb = wDown * hb
        Arrays.fill(state.xb, 0);
        weights.wDown().matmulParallel(state.hb, state.xb, dim, ffnDim);
    }

    /**
     * GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
     */
    private static void gelu(float[] x, int size) {
        for (int i = 0; i < size; i++) {
            float v = x[i];
            x[i] = 0.5f * v * (1.0f + (float) Math.tanh(SQRT_2_OVER_PI * (v + 0.044715f * v * v * v)));
        }
    }
}
