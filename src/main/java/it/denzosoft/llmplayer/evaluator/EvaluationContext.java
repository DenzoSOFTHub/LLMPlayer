package it.denzosoft.llmplayer.evaluator;

import java.util.List;

public final class EvaluationContext {
    private final String prompt;
    private final String response;
    private final int[] tokens;
    private final List<float[]> logitsHistory;
    private final long timeMs;
    private final boolean eosReached;

    public EvaluationContext(String prompt, String response, int[] tokens,
                             List<float[]> logitsHistory, long timeMs, boolean eosReached) {
        this.prompt = prompt;
        this.response = response;
        this.tokens = tokens;
        this.logitsHistory = logitsHistory;
        this.timeMs = timeMs;
        this.eosReached = eosReached;
    }

    // Backwards-compatible constructor
    public EvaluationContext(String prompt, String response, int[] tokens,
                             List<float[]> logitsHistory, long timeMs) {
        this(prompt, response, tokens, logitsHistory, timeMs, false);
    }

    public String prompt() { return prompt; }
    public String response() { return response; }
    public int[] tokens() { return tokens; }
    public List<float[]> logitsHistory() { return logitsHistory; }
    public long timeMs() { return timeMs; }
    public boolean eosReached() { return eosReached; }
}
