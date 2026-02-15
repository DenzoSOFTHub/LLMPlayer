package it.denzosoft.llmplayer.api;

import it.denzosoft.llmplayer.evaluator.EvaluationResult;

import java.util.List;

public final class GenerationResponse {
    private final String text;
    private final int tokenCount;
    private final int promptTokenCount;
    private final double tokensPerSecond;
    private final long timeMs;
    private final List<EvaluationResult> evaluation;

    public GenerationResponse(String text, int tokenCount, int promptTokenCount,
                               double tokensPerSecond, long timeMs, List<EvaluationResult> evaluation) {
        this.text = text;
        this.tokenCount = tokenCount;
        this.promptTokenCount = promptTokenCount;
        this.tokensPerSecond = tokensPerSecond;
        this.timeMs = timeMs;
        this.evaluation = evaluation;
    }

    public String text() { return text; }
    public int tokenCount() { return tokenCount; }
    public int promptTokenCount() { return promptTokenCount; }
    public double tokensPerSecond() { return tokensPerSecond; }
    public long timeMs() { return timeMs; }
    public List<EvaluationResult> evaluation() { return evaluation; }

    @Override
    public String toString() {
        return String.format("Generated %d tokens in %dms (%.1f tok/s)\n%s",
            tokenCount, timeMs, tokensPerSecond, text);
    }
}
