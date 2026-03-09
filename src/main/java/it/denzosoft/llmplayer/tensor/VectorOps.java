package it.denzosoft.llmplayer.tensor;

/**
 * Interface for vectorizable math operations.
 * On Java 21+ with Vector API, uses SIMD.
 * On older JVMs, uses scalar fallback.
 */
public interface VectorOps {

    float dot(float[] a, int aOff, float[] b, int bOff, int len);

    void saxpy(float a, float[] x, int xOff, float[] y, int yOff, int len);

    void rmsnorm(float[] out, float[] x, float[] w, int size, float eps);

    void softmax(float[] logits, int offset, int size);

    void scale(float[] x, int offset, int size, float scale);

    void elementwiseMul(float[] a, float[] b, float[] out, int size);

    void silu(float[] x, int size);

    void accumulate(float[] y, float[] x, int size);

    /**
     * In-place scale with weights: x[xOff+i] = x[xOff+i] * scale * w[i] for i in [0, size).
     * Used for per-head QK-norm scaling phase.
     */
    default void scaleWeighted(float[] x, int xOff, float[] w, float scale, int size) {
        for (int i = 0; i < size; i++) {
            x[xOff + i] = x[xOff + i] * scale * w[i];
        }
    }

    /**
     * Apply RoPE rotation in NEOX (split-half) mode for one head.
     * vec[off+i] and vec[off+halfRope+i] are the paired elements.
     * cos/sin are read from tables at cosOff+i.
     */
    default void ropeNeox(float[] vec, int off, float[] cos, float[] sin, int cosOff, int halfRope) {
        for (int i = 0; i < halfRope; i++) {
            float c = cos[cosOff + i];
            float s = sin[cosOff + i];
            float v0 = vec[off + i];
            float v1 = vec[off + halfRope + i];
            vec[off + i] = v0 * c - v1 * s;
            vec[off + halfRope + i] = v0 * s + v1 * c;
        }
    }
}
