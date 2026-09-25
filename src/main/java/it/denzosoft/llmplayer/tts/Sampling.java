package it.denzosoft.llmplayer.tts;

import java.util.Arrays;
import java.util.Random;

/** Top-k / top-p / temperature sampling over a logits prefix, for the TTS codebooks. */
final class Sampling {

    private Sampling() {}

    /**
     * Samples an index in {@code [0, n)}. Entries equal to {@code -Infinity} are never chosen.
     * {@code topK <= 0} keeps every entry, {@code topP >= 1} disables nucleus filtering and
     * {@code temp <= 0} is greedy.
     */
    static int sample(float[] logits, int n, int topK, float topP, float temp, Random rng) {
        if (temp <= 0f) {
            int best = 0;
            for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
            return best;
        }
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Float.compare(logits[b], logits[a]));
        int k = (topK > 0 && topK < n) ? topK : n;
        double max = logits[idx[0]];
        double[] p = new double[k];
        double sum = 0;
        for (int i = 0; i < k; i++) {
            float l = logits[idx[i]];
            p[i] = Float.isInfinite(l) && l < 0 ? 0 : Math.exp((l - max) / temp);
            sum += p[i];
        }
        int keep = k;
        if (topP < 1f) {
            double cum = 0;
            for (int i = 0; i < k; i++) {
                cum += p[i] / sum;
                if (cum >= topP) { keep = i + 1; break; }
            }
            sum = 0;
            for (int i = 0; i < keep; i++) sum += p[i];
        }
        double r = rng.nextDouble() * sum;
        for (int i = 0; i < keep; i++) {
            r -= p[i];
            if (r <= 0) return idx[i];
        }
        return idx[keep - 1];
    }
}
