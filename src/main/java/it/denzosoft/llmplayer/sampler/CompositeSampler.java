package it.denzosoft.llmplayer.sampler;

import java.util.*;

/**
 * Composite sampler pipeline:
 *   DRY penalty -> repetition penalty -> temperature -> top-K -> softmax(K) -> min-P -> top-P -> (mirostat|multinomial).
 *
 * <p>Optimized: softmax only on top-K candidates (not full vocab), quickselect for top-K.
 * Modern samplers (min-P, DRY, mirostat) are disabled unless explicitly set in SamplerConfig.
 */
public class CompositeSampler implements Sampler {

    private final SamplerConfig config;
    private final Random random;
    private final List<Integer> recentTokens;
    private final int maxRecentTokens;

    // Mirostat v2 running state: current surprise threshold
    private float mirostatMu;

    public CompositeSampler(SamplerConfig config) {
        this.config = config;
        this.random = new Random(config.seed());
        this.recentTokens = new ArrayList<>();
        // Keep enough tokens for DRY lookback when enabled; DRY range can be large.
        this.maxRecentTokens = Math.max(64, config.dryRange());
        // Initialize mirostat mu to 2 * tau (standard starting value)
        this.mirostatMu = 2.0f * config.mirostatTau();
    }

    @Override
    public int sample(float[] logits) {
        int vocabSize = logits.length;

        // 0. Copy logits so we don't mutate caller's buffer
        float[] probs = Arrays.copyOf(logits, vocabSize);

        // 1. DRY penalty (before repetition penalty, applied to logits)
        if (config.dryMultiplier() > 0f) {
            applyDryPenalty(probs, vocabSize);
        }

        // 2. Repetition penalty (in-place on logits copy)
        if (config.repetitionPenalty() != 1.0f) {
            for (int tokenId : recentTokens) {
                if (tokenId >= 0 && tokenId < vocabSize) {
                    if (probs[tokenId] > 0) {
                        probs[tokenId] /= config.repetitionPenalty();
                    } else {
                        probs[tokenId] *= config.repetitionPenalty();
                    }
                }
            }
        }

        // 3. Greedy (temperature == 0)
        if (config.temperature() == 0.0f) {
            int bestToken = argmax(probs, vocabSize);
            addRecentToken(bestToken);
            return bestToken;
        }

        // 4. Temperature scaling
        float invTemp = 1.0f / config.temperature();
        for (int i = 0; i < vocabSize; i++) {
            probs[i] *= invTemp;
        }

        // 5. Top-K: find K largest logits, then softmax ONLY those K values
        int k = config.topK();
        if (k <= 0 || k >= vocabSize) k = vocabSize;

        if (k < vocabSize) {
            float threshold = quickselect(probs, vocabSize, k);
            for (int i = 0; i < vocabSize; i++) {
                if (probs[i] < threshold) probs[i] = Float.NEGATIVE_INFINITY;
            }
        }

        // 6. Softmax
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < vocabSize; i++) {
            if (probs[i] > max) max = probs[i];
        }
        float sum = 0f;
        for (int i = 0; i < vocabSize; i++) {
            if (probs[i] > Float.NEGATIVE_INFINITY) {
                probs[i] = (float) Math.exp(probs[i] - max);
                sum += probs[i];
            } else {
                probs[i] = 0f;
            }
        }
        if (sum > 0f) {
            float invSum = 1.0f / sum;
            for (int i = 0; i < vocabSize; i++) probs[i] *= invSum;
        }

        // 7. min-P: keep tokens with prob >= minP * max_prob (applied BEFORE top-P)
        //    See llama.cpp llama_sampler_init_min_p — this is the modern default in many clients.
        if (config.minP() > 0f) {
            applyMinP(probs, vocabSize, config.minP());
        }

        // 8. Top-P (nucleus)
        if (config.topP() < 1.0f) {
            applyTopP(probs, vocabSize, config.topP());
        }

        // 9. Final selection: mirostat v2 (if enabled) or standard multinomial
        int sampled;
        if (config.mirostatMode() == 2) {
            sampled = sampleMirostatV2(probs, vocabSize);
        } else {
            sampled = multinomialSample(probs, vocabSize);
        }
        addRecentToken(sampled);
        return sampled;
    }

    /**
     * min-P filter: keep only tokens whose probability is at least {@code minP * max_prob}.
     * Zeros out below-threshold tokens and renormalizes.
     */
    private static void applyMinP(float[] probs, int size, float minP) {
        float max = 0f;
        for (int i = 0; i < size; i++) {
            if (probs[i] > max) max = probs[i];
        }
        if (max <= 0f) return;
        float threshold = minP * max;
        float sum = 0f;
        for (int i = 0; i < size; i++) {
            if (probs[i] < threshold) probs[i] = 0f;
            sum += probs[i];
        }
        if (sum > 0f) {
            float invSum = 1.0f / sum;
            for (int i = 0; i < size; i++) probs[i] *= invSum;
        }
    }

    /**
     * DRY (Don't Repeat Yourself) penalty: for every token t in the vocabulary, compute the
     * longest suffix of recentTokens that, if extended by t, would match an earlier occurrence
     * in recentTokens. If the match length exceeds {@code allowedLength}, apply an exponential
     * penalty {@code multiplier * base^(match_length - allowedLength)} to the logit of t.
     *
     * <p>This is a simplified version of the DRY sampler (see llama.cpp sampling.cpp). It uses
     * O(range * vocab) in the worst case; in practice tiny because only a few candidate tokens
     * have matches.
     */
    private void applyDryPenalty(float[] logits, int vocabSize) {
        int n = recentTokens.size();
        if (n < 2) return;
        int range = Math.min(n, config.dryRange());
        int start = n - range;

        // The "current suffix" we're trying to extend is the last token(s) of recentTokens.
        // For each earlier position p where recentTokens[p] == recentTokens[n-1], compute the
        // length of the suffix match: how far backward (p-1,n-2), (p-2,n-3)... the tokens match.
        // Then the token that would be "next" after position p is recentTokens[p+1] — that's
        // our penalty target.
        int lastToken = recentTokens.get(n - 1);
        float multiplier = config.dryMultiplier();
        float base = config.dryBase();
        int allowedLength = config.dryAllowedLength();

        // Track the longest match ending at each earlier position
        for (int p = start; p < n - 1; p++) {
            if (recentTokens.get(p) != lastToken) continue;

            // Count match length by walking backward: (p, n-1) match, so start with 1
            int matchLen = 1;
            int pi = p - 1;
            int ni = n - 2;
            while (pi >= start && ni > p && recentTokens.get(pi).intValue() == recentTokens.get(ni).intValue()) {
                matchLen++;
                pi--;
                ni--;
            }

            // The token that would extend this repetition is recentTokens[p+1]
            int nextToken = recentTokens.get(p + 1);
            if (nextToken < 0 || nextToken >= vocabSize) continue;

            if (matchLen >= allowedLength) {
                // Exponential penalty: multiplier * base^(matchLen - allowedLength)
                float penalty = multiplier * (float) Math.pow(base, matchLen - allowedLength);
                logits[nextToken] -= penalty;
            }
        }
    }

    /**
     * Mirostat v2 sampling: maintains a running surprise threshold {@code mu}, truncates the
     * candidate set to tokens with surprise &lt;= mu, samples, then updates mu based on the
     * observed surprise. Mathematically independent of vocab size — targets a specific entropy.
     *
     * <p>See llama.cpp llama_sampler_init_mirostat_v2. Assumes {@code probs} is already a
     * normalized distribution (after top-K/top-P/min-P filtering).
     */
    private int sampleMirostatV2(float[] probs, int vocabSize) {
        // Collect non-zero candidates with their indices, sort descending by probability
        int nonZero = 0;
        for (int i = 0; i < vocabSize; i++) {
            if (probs[i] > 0f) nonZero++;
        }
        if (nonZero == 0) return multinomialSample(probs, vocabSize);

        int[] indices = new int[nonZero];
        int idx = 0;
        for (int i = 0; i < vocabSize; i++) {
            if (probs[i] > 0f) indices[idx++] = i;
        }
        sortIndicesByProb(indices, probs, nonZero);

        // Truncate to tokens with surprise = -log2(p) <= mu
        // (equivalently: p >= 2^-mu)
        float pThreshold = (float) Math.pow(2.0, -mirostatMu);
        int keepCount = 0;
        for (int i = 0; i < nonZero; i++) {
            if (probs[indices[i]] >= pThreshold) keepCount++;
            else break;
        }
        if (keepCount == 0) keepCount = 1; // always keep at least the best

        // Renormalize the kept subset
        float sum = 0f;
        for (int i = 0; i < keepCount; i++) sum += probs[indices[i]];
        if (sum <= 0f) return indices[0];

        // Sample one token from the renormalized subset
        float r = random.nextFloat() * sum;
        float cum = 0f;
        int sampled = indices[0];
        for (int i = 0; i < keepCount; i++) {
            cum += probs[indices[i]];
            if (r < cum) { sampled = indices[i]; break; }
        }

        // Update mu: observed surprise - tau, scaled by eta
        float observedProb = probs[sampled] / sum;
        float observedSurprise = -(float) (Math.log(observedProb) / Math.log(2.0));
        float error = observedSurprise - config.mirostatTau();
        mirostatMu -= config.mirostatEta() * error;

        return sampled;
    }

    private void addRecentToken(int token) {
        recentTokens.add(token);
        if (recentTokens.size() > maxRecentTokens) {
            recentTokens.remove(0);
        }
    }

    private static int argmax(float[] arr, int size) {
        int best = 0;
        float bestVal = arr[0];
        for (int i = 1; i < size; i++) {
            if (arr[i] > bestVal) {
                bestVal = arr[i];
                best = i;
            }
        }
        return best;
    }

    /**
     * Quickselect: find the k-th largest value in O(n) average time.
     * Returns the threshold value such that exactly k elements are >= threshold.
     */
    private static float quickselect(float[] arr, int size, int k) {
        // Copy to avoid modifying the original during partitioning
        float[] work = new float[size];
        System.arraycopy(arr, 0, work, 0, size);
        int targetIdx = size - k; // k-th largest = (size-k)-th smallest
        return select(work, 0, size - 1, targetIdx);
    }

    private static float select(float[] arr, int left, int right, int k) {
        while (left < right) {
            // Median-of-3 pivot
            int mid = left + (right - left) / 2;
            if (arr[left] > arr[mid]) swap(arr, left, mid);
            if (arr[left] > arr[right]) swap(arr, left, right);
            if (arr[mid] > arr[right]) swap(arr, mid, right);
            float pivot = arr[mid];
            swap(arr, mid, right - 1);

            int i = left, j = right - 1;
            while (true) {
                while (arr[++i] < pivot) {}
                while (arr[--j] > pivot) {}
                if (i >= j) break;
                swap(arr, i, j);
            }
            swap(arr, i, right - 1);

            if (k == i) return arr[i];
            else if (k < i) right = i - 1;
            else left = i + 1;
        }
        return arr[left];
    }

    private static void swap(float[] arr, int i, int j) {
        float tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }

    /**
     * Top-P: zero out tokens beyond nucleus probability mass.
     * Uses primitive arrays instead of Integer[] boxing + HashSet.
     */
    private static void applyTopP(float[] probs, int size, float topP) {
        // Count non-zero entries to avoid sorting zeros
        int nonZero = 0;
        for (int i = 0; i < size; i++) {
            if (probs[i] > 0f) nonZero++;
        }
        if (nonZero <= 1) return;

        // Collect non-zero indices and sort by probability descending
        int[] indices = new int[nonZero];
        int idx = 0;
        for (int i = 0; i < size; i++) {
            if (probs[i] > 0f) indices[idx++] = i;
        }

        // Sort indices by descending probability using insertion sort for small arrays,
        // or Arrays.sort with primitive-friendly approach
        sortIndicesByProb(indices, probs, nonZero);

        // Find cutoff
        float cumSum = 0f;
        int cutoff = nonZero;
        for (int i = 0; i < nonZero; i++) {
            cumSum += probs[indices[i]];
            if (cumSum > topP) {
                cutoff = i + 1;
                break;
            }
        }

        // Zero out tokens beyond cutoff using boolean mask
        boolean[] keep = new boolean[size];
        for (int i = 0; i < cutoff; i++) {
            keep[indices[i]] = true;
        }
        for (int i = 0; i < size; i++) {
            if (!keep[i]) probs[i] = 0f;
        }

        // Re-normalize
        float sum = 0f;
        for (int i = 0; i < size; i++) sum += probs[i];
        if (sum > 0f) {
            float invSum = 1.0f / sum;
            for (int i = 0; i < size; i++) probs[i] *= invSum;
        }
    }

    private static void sortIndicesByProb(int[] indices, float[] probs, int n) {
        // For small n (after top-K), insertion sort is faster than Arrays.sort overhead
        if (n <= 64) {
            for (int i = 1; i < n; i++) {
                int key = indices[i];
                float keyProb = probs[key];
                int j = i - 1;
                while (j >= 0 && probs[indices[j]] < keyProb) {
                    indices[j + 1] = indices[j];
                    j--;
                }
                indices[j + 1] = key;
            }
        } else {
            // Convert to Integer[] only for large n (rare with top-K)
            Integer[] boxed = new Integer[n];
            for (int i = 0; i < n; i++) boxed[i] = indices[i];
            Arrays.sort(boxed, (a, b) -> Float.compare(probs[b], probs[a]));
            for (int i = 0; i < n; i++) indices[i] = boxed[i];
        }
    }

    private int multinomialSample(float[] probs, int size) {
        float r = random.nextFloat();
        float cumSum = 0f;
        for (int i = 0; i < size; i++) {
            cumSum += probs[i];
            if (r < cumSum) return i;
        }
        return size - 1;
    }

    public void reset() {
        recentTokens.clear();
    }
}
