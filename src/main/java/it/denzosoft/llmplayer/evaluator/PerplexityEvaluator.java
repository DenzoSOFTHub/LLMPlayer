package it.denzosoft.llmplayer.evaluator;

/**
 * Evaluates response quality based on perplexity (average negative log-likelihood).
 * Lower perplexity = higher confidence in token predictions.
 */
public class PerplexityEvaluator implements ResponseEvaluator {

    @Override
    public EvaluationResult evaluate(EvaluationContext context) {
        if (context.logitsHistory() == null || context.logitsHistory().isEmpty() || context.tokens() == null) {
            return new EvaluationResult("Perplexity", 0.5f, "N/A", "No logits available");
        }

        double sumLogProb = 0;
        int count = 0;

        for (int i = 0; i < context.logitsHistory().size() && i < context.tokens().length; i++) {
            float[] logits = context.logitsHistory().get(i);
            int actualToken = context.tokens()[i];
            if (actualToken < 0 || actualToken >= logits.length) continue;

            // Softmax to get probability
            float maxLogit = Float.NEGATIVE_INFINITY;
            for (float l : logits) if (l > maxLogit) maxLogit = l;

            double sumExp = 0;
            for (float l : logits) sumExp += Math.exp(l - maxLogit);

            double logProb = (logits[actualToken] - maxLogit) - Math.log(sumExp);
            sumLogProb += logProb;
            count++;
        }

        if (count == 0) {
            return new EvaluationResult("Perplexity", 0.5f, "N/A", "No tokens evaluated");
        }

        double avgNegLogProb = -sumLogProb / count;
        double perplexity = Math.exp(avgNegLogProb);

        // Normalize score: PPL < 5 is excellent, PPL > 100 is bad
        float score = (float) Math.max(0, Math.min(1, 1.0 - Math.log(perplexity) / Math.log(100)));

        return new EvaluationResult("Perplexity",
            score,
            EvaluationResult.labelFromScore(score),
            String.format("PPL=%.2f, avg_nll=%.4f, tokens=%d", perplexity, avgNegLogProb, count));
    }
}
