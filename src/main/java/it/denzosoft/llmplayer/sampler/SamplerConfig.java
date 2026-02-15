package it.denzosoft.llmplayer.sampler;

public final class SamplerConfig {

    public static final SamplerConfig DEFAULT = new SamplerConfig(0.7f, 40, 0.9f, 1.1f, System.nanoTime());

    private final float temperature;
    private final int topK;
    private final float topP;
    private final float repetitionPenalty;
    private final long seed;

    public SamplerConfig(float temperature, int topK, float topP, float repetitionPenalty, long seed) {
        this.temperature = temperature;
        this.topK = topK;
        this.topP = topP;
        this.repetitionPenalty = repetitionPenalty;
        this.seed = seed;
    }

    public float temperature() { return temperature; }
    public int topK() { return topK; }
    public float topP() { return topP; }
    public float repetitionPenalty() { return repetitionPenalty; }
    public long seed() { return seed; }

    public static SamplerConfig greedy() {
        return new SamplerConfig(0.0f, 1, 1.0f, 1.0f, 0);
    }

    public static Builder builder() { return new Builder(); }

    public static class Builder {
        private float temperature = 0.7f;
        private int topK = 40;
        private float topP = 0.9f;
        private float repetitionPenalty = 1.1f;
        private long seed = System.nanoTime();

        public Builder temperature(float t) { this.temperature = t; return this; }
        public Builder topK(int k) { this.topK = k; return this; }
        public Builder topP(float p) { this.topP = p; return this; }
        public Builder repetitionPenalty(float rp) { this.repetitionPenalty = rp; return this; }
        public Builder seed(long s) { this.seed = s; return this; }
        public SamplerConfig build() { return new SamplerConfig(temperature, topK, topP, repetitionPenalty, seed); }
    }
}
