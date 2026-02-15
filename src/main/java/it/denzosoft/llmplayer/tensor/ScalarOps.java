package it.denzosoft.llmplayer.tensor;

/**
 * Scalar (non-SIMD) implementation of VectorOps.
 * Used as fallback when Vector API is not available.
 */
public final class ScalarOps implements VectorOps {

    @Override
    public float dot(float[] a, int aOff, float[] b, int bOff, int len) {
        float sum = 0f;
        for (int i = 0; i < len; i++) {
            sum += a[aOff + i] * b[bOff + i];
        }
        return sum;
    }

    @Override
    public void saxpy(float a, float[] x, int xOff, float[] y, int yOff, int len) {
        for (int i = 0; i < len; i++) {
            y[yOff + i] += a * x[xOff + i];
        }
    }

    @Override
    public void rmsnorm(float[] out, float[] x, float[] w, int size, float eps) {
        float ss = 0f;
        for (int i = 0; i < size; i++) {
            ss += x[i] * x[i];
        }
        ss = 1.0f / (float) Math.sqrt(ss / size + eps);
        for (int i = 0; i < size; i++) {
            out[i] = x[i] * ss * w[i];
        }
    }

    @Override
    public void softmax(float[] logits, int offset, int size) {
        float maxVal = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < size; i++) {
            if (logits[offset + i] > maxVal) maxVal = logits[offset + i];
        }
        float sum = 0f;
        for (int i = 0; i < size; i++) {
            logits[offset + i] = (float) Math.exp(logits[offset + i] - maxVal);
            sum += logits[offset + i];
        }
        float invSum = 1.0f / sum;
        for (int i = 0; i < size; i++) {
            logits[offset + i] *= invSum;
        }
    }

    @Override
    public void scale(float[] x, int offset, int size, float scale) {
        for (int i = 0; i < size; i++) {
            x[offset + i] *= scale;
        }
    }

    @Override
    public void elementwiseMul(float[] a, float[] b, float[] out, int size) {
        for (int i = 0; i < size; i++) {
            out[i] = a[i] * b[i];
        }
    }

    @Override
    public void silu(float[] x, int size) {
        for (int i = 0; i < size; i++) {
            x[i] = x[i] / (1.0f + (float) Math.exp(-x[i]));
        }
    }

    @Override
    public void accumulate(float[] y, float[] x, int size) {
        for (int i = 0; i < size; i++) {
            y[i] += x[i];
        }
    }
}
