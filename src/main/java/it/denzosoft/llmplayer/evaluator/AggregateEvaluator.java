package it.denzosoft.llmplayer.evaluator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Aggregates multiple evaluators into a single composite evaluation.
 */
public class AggregateEvaluator implements ResponseEvaluator {

    private final List<ResponseEvaluator> evaluators;
    private final float[] weights;

    public AggregateEvaluator(List<ResponseEvaluator> evaluators, float[] weights) {
        this.evaluators = evaluators;
        this.weights = weights;
    }

    public static AggregateEvaluator createDefault(int eosTokenId, int maxTokens) {
        List<ResponseEvaluator> evals = Arrays.asList(
            new PerplexityEvaluator(),
            new CoherenceEvaluator(),
            new LengthEvaluator(eosTokenId, maxTokens)
        );
        return new AggregateEvaluator(evals, new float[]{0.4f, 0.35f, 0.25f});
    }

    @Override
    public EvaluationResult evaluate(EvaluationContext context) {
        List<EvaluationResult> results = new ArrayList<>();
        float weightedSum = 0;
        float totalWeight = 0;

        for (int i = 0; i < evaluators.size(); i++) {
            EvaluationResult result = evaluators.get(i).evaluate(context);
            results.add(result);
            float w = i < weights.length ? weights[i] : 1.0f;
            weightedSum += result.score() * w;
            totalWeight += w;
        }

        float finalScore = totalWeight > 0 ? weightedSum / totalWeight : 0;

        StringBuilder details = new StringBuilder();
        for (EvaluationResult r : results) {
            if (details.length() > 0) details.append("; ");
            details.append(r.name()).append("=").append(String.format("%.2f", r.score()));
        }

        return new EvaluationResult("Aggregate",
            finalScore,
            EvaluationResult.labelFromScore(finalScore),
            details.toString());
    }

    public List<EvaluationResult> evaluateAll(EvaluationContext context) {
        List<EvaluationResult> results = new ArrayList<>();
        for (ResponseEvaluator evaluator : evaluators) {
            results.add(evaluator.evaluate(context));
        }
        results.add(evaluate(context));
        return results;
    }
}
