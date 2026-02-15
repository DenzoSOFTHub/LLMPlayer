package it.denzosoft.llmplayer.evaluator;

/**
 * Evaluates whether the response ended naturally (EOS) vs truncation or being too short.
 */
public class LengthEvaluator implements ResponseEvaluator {

    private final int eosTokenId;
    private final int maxTokens;

    public LengthEvaluator(int eosTokenId, int maxTokens) {
        this.eosTokenId = eosTokenId;
        this.maxTokens = maxTokens;
    }

    @Override
    public EvaluationResult evaluate(EvaluationContext context) {
        if (context.tokens() == null || context.tokens().length == 0) {
            return new EvaluationResult("Length", 0.0f, "BAD", "No tokens generated");
        }

        int len = context.tokens().length;
        boolean naturalEnd = context.eosReached();
        boolean truncated = len >= maxTokens && !naturalEnd;
        boolean tooShort = len < 3;

        float score;
        String detail;

        if (naturalEnd) {
            score = 1.0f;
            detail = "Natural EOS at token " + len;
        } else if (truncated) {
            score = 0.5f;
            detail = "Truncated at max_tokens=" + maxTokens;
        } else if (tooShort) {
            score = 0.3f;
            detail = "Too short: only " + len + " tokens";
        } else {
            score = 0.7f;
            detail = "Generated " + len + " tokens (no EOS)";
        }

        return new EvaluationResult("Length", score, EvaluationResult.labelFromScore(score), detail);
    }
}
