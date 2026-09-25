package it.denzosoft.llmplayer.inference;

/**
 * Optional reduction of the MoE top-K, as an explicit speed-for-quality knob.
 *
 * Expert work scales linearly with the number of routed experts: on Qwen3-Coder-30B the routed
 * experts are 1.81 G of the 3.04 G parameters touched per token, so dropping two of the eight
 * selected removes a quarter of the largest phase in the profile. What it costs is quality, because
 * the discarded experts are the lowest-weighted ones the router chose — measurable directly as
 * perplexity, which is why this is a knob and not a default.
 *
 * The reduction is applied at selection time rather than by discarding afterwards, so the paths that
 * renormalise the routing weights (Qwen3 MoE, and DeepSeek-V3-style sigmoid gating) renormalise over
 * the reduced set and the gate mass still sums to one. It also shrinks the work upstream: fewer
 * experts are read from disk, so the SSD-streaming cache sees proportionally fewer misses.
 *
 * Set with {@code --expert-top-k N} or {@code -Dmoe.top.k=N}. Values at or above the model's own
 * top-K are ignored, so it can only ever reduce.
 *
 * This is the same class of user-facing lossy trade already offered for the KV cache by
 * {@code -Dkv.q8} and {@code -Dkv.q4}, and it should be documented with its measured PPL cost in the
 * same way. See {@code docs/optimization/per-token-latency-analysis.md}.
 */
public final class MoERouting {

    private static final int OVERRIDE = resolveOverride();
    private static volatile boolean announced;

    private MoERouting() {}

    private static int resolveOverride() {
        String v = System.getProperty("moe.top.k");
        if (v == null) return -1;
        try {
            int n = Integer.parseInt(v.trim());
            return n > 0 ? n : -1;
        } catch (NumberFormatException e) {
            return -1;
        }
    }

    /**
     * The number of experts to actually route to, given what the model specifies. Returns
     * {@code configured} unchanged unless a smaller override was requested.
     */
    public static int effectiveTopK(int configured) {
        if (OVERRIDE <= 0 || OVERRIDE >= configured) return configured;
        if (!announced) {
            announced = true;
            System.out.println("  MoE routing: top-" + OVERRIDE + " instead of top-" + configured
                + " (-Dmoe.top.k / --expert-top-k) — expert work reduced to "
                + Math.round(100.0 * OVERRIDE / configured) + "%, quality cost shows up as PPL");
        }
        return OVERRIDE;
    }
}
