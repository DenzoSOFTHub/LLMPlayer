package it.denzosoft.llmplayer.sampler;

import java.util.*;

/**
 * Composite sampler pipeline: repetition penalty -> temperature -> top-K -> softmax(K) -> top-P -> sample.
 * Optimized: softmax only on top-K candidates (not full vocab), quickselect for top-K.
 */
public class CompositeSampler implements Sampler {

    private final SamplerConfig config;
    private final Random random;
    private final List<Integer> recentTokens;
    private final int maxRecentTokens;

    public CompositeSampler(SamplerConfig config) {
        this.config = config;
        this.random = new Random(config.seed());
        this.recentTokens = new ArrayList<>();
        this.maxRecentTokens = 64;
    }

    @Override
    public int sample(float[] logits) {
        int vocabSize = logits.length;

        // 1. Repetition penalty (applied in-place to logits copy)
        float[] probs = Arrays.copyOf(logits, vocabSize);
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

        // 2. Greedy (temperature == 0)
        if (config.temperature() == 0.0f) {
            int bestToken = argmax(probs, vocabSize);
            addRecentToken(bestToken);
            return bestToken;
        }

        // 3. Temperature scaling
        float invTemp = 1.0f / config.temperature();
        for (int i = 0; i < vocabSize; i++) {
            probs[i] *= invTemp;
        }

        // 4. Top-K: find K largest logits, then softmax ONLY those K values
        int k = config.topK();
        if (k <= 0 || k >= vocabSize) k = vocabSize;

        if (k < vocabSize) {
            // Find k-th largest value using quickselect (O(n) avg vs O(n log n) sort)
            float threshold = quickselect(probs, vocabSize, k);
            // Zero out below threshold and collect top-K indices
            for (int i = 0; i < vocabSize; i++) {
                if (probs[i] < threshold) probs[i] = Float.NEGATIVE_INFINITY;
            }
        }

        // 5. Softmax (only non-neg-inf values contribute, but Math.exp(-inf) = 0 so this is safe)
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

        // 6. Top-P (nucleus) filtering
        if (config.topP() < 1.0f) {
            applyTopP(probs, vocabSize, config.topP());
        }

        // 7. Multinomial sampling
        int sampled = multinomialSample(probs, vocabSize);
        addRecentToken(sampled);
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
