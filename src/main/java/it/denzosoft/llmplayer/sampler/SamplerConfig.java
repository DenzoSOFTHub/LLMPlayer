package it.denzosoft.llmplayer.sampler;

/**
 * Sampler configuration.
 *
 * <p>Pipeline order (when enabled):
 * DRY penalty → repetition penalty → temperature → top-K → softmax → min-P → top-P → mirostat/multinomial.
 *
 * <p>Defaults follow the conservative set that matches earlier versions (temp=0.7, top-K=40,
 * top-P=0.9, rep-penalty=1.1). All new samplers (min-P, mirostat, DRY) are OFF by default
 * and must be enabled explicitly.
 */
public final class SamplerConfig {

    public static final SamplerConfig DEFAULT = new SamplerConfig(
        0.7f, 40, 0.9f, 1.1f, System.nanoTime(),
        0f,           // minP disabled
        0,            // mirostat disabled
        5.0f, 0.1f,   // mirostat defaults (tau, eta)
        0f, 1.75f, 2, 1024 // DRY disabled (multiplier=0)
    );

    private final float temperature;
    private final int topK;
    private final float topP;
    private final float repetitionPenalty;
    private final long seed;

    // min-P: keep tokens whose prob >= minP * max_prob. 0 = disabled.
    private final float minP;

    // Mirostat: 0 = disabled, 1 = mirostat v1, 2 = mirostat v2.
    // tau = target entropy (usually ~5), eta = learning rate (~0.1).
    private final int mirostatMode;
    private final float mirostatTau;
    private final float mirostatEta;

    // DRY (Don't Repeat Yourself): n-gram repetition penalty.
    // multiplier = 0 disables DRY. base controls exponential growth.
    // allowedLength = min match length before penalty kicks in. range = lookback window.
    private final float dryMultiplier;
    private final float dryBase;
    private final int dryAllowedLength;
    private final int dryRange;

    public SamplerConfig(float temperature, int topK, float topP, float repetitionPenalty, long seed) {
        this(temperature, topK, topP, repetitionPenalty, seed,
             0f, 0, 5.0f, 0.1f, 0f, 1.75f, 2, 1024);
    }

    public SamplerConfig(float temperature, int topK, float topP, float repetitionPenalty, long seed,
                         float minP, int mirostatMode, float mirostatTau, float mirostatEta,
                         float dryMultiplier, float dryBase, int dryAllowedLength, int dryRange) {
        this.temperature = temperature;
        this.topK = topK;
        this.topP = topP;
        this.repetitionPenalty = repetitionPenalty;
        this.seed = seed;
        this.minP = minP;
        this.mirostatMode = mirostatMode;
        this.mirostatTau = mirostatTau;
        this.mirostatEta = mirostatEta;
        this.dryMultiplier = dryMultiplier;
        this.dryBase = dryBase;
        this.dryAllowedLength = dryAllowedLength;
        this.dryRange = dryRange;
    }

    public float temperature() { return temperature; }
    public int topK() { return topK; }
    public float topP() { return topP; }
    public float repetitionPenalty() { return repetitionPenalty; }
    public long seed() { return seed; }
    public float minP() { return minP; }
    public int mirostatMode() { return mirostatMode; }
    public float mirostatTau() { return mirostatTau; }
    public float mirostatEta() { return mirostatEta; }
    public float dryMultiplier() { return dryMultiplier; }
    public float dryBase() { return dryBase; }
    public int dryAllowedLength() { return dryAllowedLength; }
    public int dryRange() { return dryRange; }

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
        private float minP = 0f;
        private int mirostatMode = 0;
        private float mirostatTau = 5.0f;
        private float mirostatEta = 0.1f;
        private float dryMultiplier = 0f;
        private float dryBase = 1.75f;
        private int dryAllowedLength = 2;
        private int dryRange = 1024;

        public Builder temperature(float t) { this.temperature = t; return this; }
        public Builder topK(int k) { this.topK = k; return this; }
        public Builder topP(float p) { this.topP = p; return this; }
        public Builder repetitionPenalty(float rp) { this.repetitionPenalty = rp; return this; }
        public Builder seed(long s) { this.seed = s; return this; }
        public Builder minP(float m) { this.minP = m; return this; }
        public Builder mirostat(int mode, float tau, float eta) {
            this.mirostatMode = mode; this.mirostatTau = tau; this.mirostatEta = eta; return this;
        }
        public Builder dry(float mult, float base, int allowed, int range) {
            this.dryMultiplier = mult; this.dryBase = base;
            this.dryAllowedLength = allowed; this.dryRange = range; return this;
        }
        public SamplerConfig build() {
            return new SamplerConfig(temperature, topK, topP, repetitionPenalty, seed,
                minP, mirostatMode, mirostatTau, mirostatEta,
                dryMultiplier, dryBase, dryAllowedLength, dryRange);
        }
    }
}
