package it.denzosoft.llmplayer.sampler;

import java.util.*;

/**
 * Composite sampler pipeline: repetition penalty -> temperature -> softmax -> top-K -> top-P -> sample.
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

        // Work on a copy to preserve original logits for evaluation
        float[] probs = Arrays.copyOf(logits, vocabSize);

        // 1. Repetition penalty
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
        for (int i = 0; i < vocabSize; i++) {
            probs[i] /= config.temperature();
        }

        // 4. Softmax
        softmax(probs, vocabSize);

        // 5. Top-K filtering
        if (config.topK() > 0 && config.topK() < vocabSize) {
            applyTopK(probs, vocabSize, config.topK());
        }

        // 6. Top-P (nucleus) filtering
        if (config.topP() < 1.0f) {
            applyTopP(probs, vocabSize, config.topP());
        }

        // 7. Re-normalize after filtering
        float sum = 0f;
        for (int i = 0; i < vocabSize; i++) sum += probs[i];
        if (sum > 0) {
            for (int i = 0; i < vocabSize; i++) probs[i] /= sum;
        }

        // 8. Multinomial sampling
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

    private static void softmax(float[] arr, int size) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < size; i++) {
            if (arr[i] > max) max = arr[i];
        }
        float sum = 0f;
        for (int i = 0; i < size; i++) {
            arr[i] = (float) Math.exp(arr[i] - max);
            sum += arr[i];
        }
        for (int i = 0; i < size; i++) {
            arr[i] /= sum;
        }
    }

    private static void applyTopK(float[] probs, int size, int k) {
        // Find k-th largest value using partial sort
        float[] sorted = Arrays.copyOf(probs, size);
        Arrays.sort(sorted);
        float threshold = sorted[size - k];

        for (int i = 0; i < size; i++) {
            if (probs[i] < threshold) {
                probs[i] = 0f;
            }
        }
    }

    private static void applyTopP(float[] probs, int size, float topP) {
        // Create index array sorted by probability descending
        Integer[] indices = new Integer[size];
        for (int i = 0; i < size; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> Float.compare(probs[b], probs[a]));

        float cumSum = 0f;
        int cutoff = size;
        for (int i = 0; i < size; i++) {
            cumSum += probs[indices[i]];
            if (cumSum > topP) {
                cutoff = i + 1;
                break;
            }
        }

        // Zero out tokens beyond cutoff
        Set<Integer> keep = new HashSet<>();
        for (int i = 0; i < cutoff; i++) {
            keep.add(indices[i]);
        }
        for (int i = 0; i < size; i++) {
            if (!keep.contains(i)) {
                probs[i] = 0f;
            }
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
