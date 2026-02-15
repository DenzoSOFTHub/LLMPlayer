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
}
