package it.denzosoft.llmplayer.inference;

/**
 * MoE routing-frequency instrumentation ({@code -Dmoe.routing.stats=true}), shared by the MoE
 * engines: counts how often each expert is selected per layer and prints, at exit, how much of
 * the routing the top-M experts of a layer capture. That concentration decides whether a
 * hot-expert cache (GPU or SSD) can pay off. Additive only — no effect on the forward pass.
 */
final class RoutingStats {

    static final boolean ENABLED = "true".equals(System.getProperty("moe.routing.stats", "false"));

    private final long[][] hits;
    private final int experts, topK;
    private long decisions;

    /** Returns null when the instrumentation is off. */
    static RoutingStats create(int layers, int experts, int topK) {
        if (!ENABLED || experts <= 0) return null;
        RoutingStats r = new RoutingStats(layers, experts, topK);
        Runtime.getRuntime().addShutdownHook(new Thread(r::print));
        System.err.println("MoE routing stats: enabled (-Dmoe.routing.stats) — summary printed at exit");
        return r;
    }

    private RoutingStats(int layers, int experts, int topK) {
        this.hits = new long[layers][experts];
        this.experts = experts;
        this.topK = topK;
    }

    synchronized void count(int layer, int[] selected, int k) {
        long[] h = hits[layer];
        for (int i = 0; i < k; i++) if (selected[i] >= 0) h[selected[i]]++;
        decisions += k;
    }

    synchronized void print() {
        if (decisions == 0) return;
        int E = experts, K = topK;
        System.err.println("\n=== MoE routing stats (" + decisions + " selections, " + E + " experts, top-" + K + ") ===");
        int[] ms = {K, 2 * K, 4 * K, 8 * K, Math.max(1, E / 4), Math.max(1, E / 2)};
        double[] cov = new double[ms.length];
        int moeLayers = 0;
        for (long[] layer : hits) {
            long[] h = layer.clone();
            long tot = 0;
            for (long x : h) tot += x;
            if (tot == 0) continue;
            moeLayers++;
            java.util.Arrays.sort(h);
            for (int mi = 0; mi < ms.length; mi++) {
                int m = Math.min(ms[mi], E);
                long top = 0;
                for (int i = E - m; i < E; i++) top += h[i];
                cov[mi] += (double) top / tot;
            }
        }
        if (moeLayers == 0) return;
        System.err.printf("  %d MoE layers. Avg fraction of routing captured by the top-M experts per layer:%n", moeLayers);
        for (int mi = 0; mi < ms.length; mi++) {
            int m = Math.min(ms[mi], E);
            System.err.printf("    top-%-4d (%2.0f%% of experts): %5.1f%% of routing%n", m, 100.0 * m / E, 100.0 * cov[mi] / moeLayers);
        }
        System.err.println("  Interpretation: a hot-expert cache helps when a small top-M captures most"
            + " routing (concentrated); it is wasted memory when routing is near-uniform (top-M ~ M/E).");
    }
}
