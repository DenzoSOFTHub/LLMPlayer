package it.denzosoft.llmplayer.inference;

/**
 * Standard Layer Normalization (mean-centered + variance scaling) with optional weight and bias.
 *
 * <p>Formula: y = ((x - mean(x)) / sqrt(var(x) + eps)) * weight + bias
 *
 * <p>Used by architectures that load tensors with {@code LLM_NORM} (not {@code LLM_NORM_RMS}) in
 * llama.cpp — currently Command-R and Cohere2. The variant without bias matches
 * {@code build_norm(..., NULL, LLM_NORM, ...)}.
 *
 * <p>Note: this is mathematically distinct from {@link RMSNorm}. RMSNorm omits the mean centering
 * and uses {@code rms(x) = sqrt(mean(x^2))} instead of {@code sqrt(var(x))}.
 */
public class LayerNorm {

    private LayerNorm() {}

    /**
     * Apply LayerNorm without bias: out = ((x - mean) / sqrt(var + eps)) * weight
     */
    public static void apply(float[] out, float[] x, float[] weight, int size, float eps) {
        // 1. mean
        float sum = 0f;
        for (int i = 0; i < size; i++) sum += x[i];
        float mean = sum / size;

        // 2. variance (centered second moment)
        float var = 0f;
        for (int i = 0; i < size; i++) {
            float c = x[i] - mean;
            var += c * c;
        }
        var /= size;

        // 3. scale
        float invStd = 1.0f / (float) Math.sqrt(var + eps);

        // 4. centered + scaled + weight
        for (int i = 0; i < size; i++) {
            out[i] = (x[i] - mean) * invStd * weight[i];
        }
    }

    /**
     * Apply LayerNorm per-head: each head of size headSize is independently normalized.
     * The weights are shared across heads (single array of length headSize).
     */
    public static void applyPerHead(float[] vec, float[] weight, int nHeads, int headSize, float eps) {
        for (int h = 0; h < nHeads; h++) {
            int off = h * headSize;
            // mean
            float sum = 0f;
            for (int i = 0; i < headSize; i++) sum += vec[off + i];
            float mean = sum / headSize;
            // variance
            float var = 0f;
            for (int i = 0; i < headSize; i++) {
                float c = vec[off + i] - mean;
                var += c * c;
            }
            var /= headSize;
            float invStd = 1.0f / (float) Math.sqrt(var + eps);
            // centered + scaled + weight
            for (int i = 0; i < headSize; i++) {
                vec[off + i] = (vec[off + i] - mean) * invStd * weight[i];
            }
        }
    }
}
