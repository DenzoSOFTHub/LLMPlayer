package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

/**
 * Root Mean Square Layer Normalization.
 */
public class RMSNorm {

    private RMSNorm() {}

    /**
     * Apply RMSNorm: out = (x / rms(x)) * weight
     * weight is a pre-dequantized float array.
     */
    public static void apply(float[] out, float[] x, float[] weight, int size, float eps) {
        VectorOpsFactory.get().rmsnorm(out, x, weight, size, eps);
    }

    /**
     * Pre-dequantize a weight tensor to a float array.
     */
    public static float[] cacheWeights(FloatTensor weights, int size) {
        float[] cached = new float[size];
        for (int i = 0; i < size; i++) {
            cached[i] = weights.getFloat(i);
        }
        return cached;
    }
}
