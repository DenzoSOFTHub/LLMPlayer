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
     * Apply RMSNorm with offset: out[outOff..] = (x[inOff..] / rms(x[inOff..])) * weight
     */
    public static void apply(float[] out, int outOff, float[] x, int inOff,
                              float[] weight, int size, float eps) {
        float ss = 0f;
        for (int i = 0; i < size; i++) {
            float v = x[inOff + i];
            ss += v * v;
        }
        ss = 1.0f / (float) Math.sqrt(ss / size + eps);
        for (int i = 0; i < size; i++) {
            out[outOff + i] = x[inOff + i] * ss * weight[i];
        }
    }

    /**
     * Apply RMSNorm without learnable scale (V-norm in Gemma 4): out = x / rms(x)
     */
    public static void applyNoScale(float[] x, int offset, int size, float eps) {
        float ss = 0f;
        for (int i = 0; i < size; i++) {
            float v = x[offset + i];
            ss += v * v;
        }
        ss = 1.0f / (float) Math.sqrt(ss / size + eps);
        for (int i = 0; i < size; i++) {
            x[offset + i] *= ss;
        }
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
