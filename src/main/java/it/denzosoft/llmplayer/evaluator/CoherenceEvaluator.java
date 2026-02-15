package it.denzosoft.llmplayer.evaluator;

import java.util.*;

/**
 * Evaluates coherence by checking for n-gram repetitions, entropy, and degenerate patterns.
 */
public class CoherenceEvaluator implements ResponseEvaluator {

    @Override
    public EvaluationResult evaluate(EvaluationContext context) {
        if (context.tokens() == null || context.tokens().length < 4) {
            return new EvaluationResult("Coherence", 0.5f, "N/A", "Too few tokens");
        }

        int[] tokens = context.tokens();
        int len = tokens.length;

        // Check n-gram repetitions (2-gram, 3-gram, 4-gram)
        double rep2 = ngramRepetitionRate(tokens, 2);
        double rep3 = ngramRepetitionRate(tokens, 3);
        double rep4 = ngramRepetitionRate(tokens, 4);

        // Check consecutive token repetitions
        int consecutiveReps = 0;
        for (int i = 1; i < len; i++) {
            if (tokens[i] == tokens[i - 1]) consecutiveReps++;
        }
        double consecutiveRate = (double) consecutiveReps / (len - 1);

        // Shannon entropy of token distribution
        Map<Integer, Integer> freq = new HashMap<>();
        for (int t : tokens) freq.merge(t, 1, Integer::sum);
        double entropy = 0;
        for (int count : freq.values()) {
            double p = (double) count / len;
            entropy -= p * Math.log(p);
        }
        double maxEntropy = Math.log(Math.min(len, freq.size()));
        double normalizedEntropy = maxEntropy > 0 ? entropy / maxEntropy : 0;

        // Combine scores
        float repScore = (float) (1.0 - (rep2 * 0.2 + rep3 * 0.3 + rep4 * 0.5));
        float consScore = (float) (1.0 - consecutiveRate * 2);
        float entropyScore = (float) normalizedEntropy;

        float score = Math.max(0, Math.min(1, repScore * 0.4f + consScore * 0.3f + entropyScore * 0.3f));

        return new EvaluationResult("Coherence",
            score,
            EvaluationResult.labelFromScore(score),
            String.format("rep2=%.2f, rep3=%.2f, rep4=%.2f, consec=%.2f, entropy=%.2f",
                rep2, rep3, rep4, consecutiveRate, normalizedEntropy));
    }

    private double ngramRepetitionRate(int[] tokens, int n) {
        if (tokens.length < n) return 0;
        Set<Long> seen = new HashSet<>();
        int total = 0;
        int repeated = 0;
        for (int i = 0; i <= tokens.length - n; i++) {
            long hash = ngramHash(tokens, i, n);
            total++;
            if (!seen.add(hash)) repeated++;
        }
        return total > 0 ? (double) repeated / total : 0;
    }

    private long ngramHash(int[] tokens, int start, int n) {
        long hash = 0;
        for (int i = 0; i < n; i++) {
            hash = hash * 31 + tokens[start + i];
        }
        return hash;
    }
}
