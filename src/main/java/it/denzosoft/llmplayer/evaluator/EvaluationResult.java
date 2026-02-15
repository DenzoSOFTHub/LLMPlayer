package it.denzosoft.llmplayer.evaluator;

public final class EvaluationResult {
    private final String name;
    private final float score;
    private final String qualityLabel;
    private final String details;

    public EvaluationResult(String name, float score, String qualityLabel, String details) {
        this.name = name;
        this.score = score;
        this.qualityLabel = qualityLabel;
        this.details = details;
    }

    public String name() { return name; }
    public float score() { return score; }
    public String qualityLabel() { return qualityLabel; }
    public String details() { return details; }

    public static String labelFromScore(float score) {
        if (score >= 0.8f) return "EXCELLENT";
        if (score >= 0.6f) return "GOOD";
        if (score >= 0.4f) return "FAIR";
        if (score >= 0.2f) return "POOR";
        return "BAD";
    }

    @Override
    public String toString() {
        return String.format("%s: %.2f (%s) - %s", name, score, qualityLabel, details);
    }
}
