package it.denzosoft.llmplayer.gpu;

/**
 * Process-wide signals between the inference engines (base code) and GPU-side helpers that live
 * in the java21 source root, chiefly the CUDA clock keeper
 * ({@code -Dcuda.clockkeeper}, see {@code docs/optimization/gpu-slower-than-cpu.md}, fix F1).
 *
 * <p>Two kinds of signal:
 * <ul>
 *   <li><b>Generation scope.</b> {@link LLMEngineHooks#generationStarted()} /
 *       {@link LLMEngineHooks#generationEnded()} bracket every generation, so a helper runs only
 *       while tokens are being produced — never during model loading, which a keeper stalled — and
 *       {@link #gpuPathDisabled()} tells it the GPU path of the model was dropped (a CPU fallback
 *       should not keep the GPU busy).</li>
 *   <li><b>Per-token phase durations.</b> The MoE engines publish, for every decoded token, the
 *       time spent in the GPU-resident attention phase and in the CPU expert phase. The timers are
 *       always on (a few {@code nanoTime} calls per layer), independent of {@code cpu.profile}, so
 *       the keeper's adaptive duty cycle has a signal in normal runs.</li>
 * </ul>
 *
 * <p>Java 8 compatible; no allocation on the per-token path.
 */
public final class GpuActivity {

    private GpuActivity() {}

    /** Receiver of the generation-scope signals (implemented in java21, e.g. the clock keeper). */
    public interface Listener {
        void generationStarted();
        void generationEnded();
        void gpuPathDisabled();
        /** One-line summary for the CLI stats block (duty cycle, P-states), or null. */
        String report();
        /** Whether a clock keeper is installed (it may be switched off). */
        boolean hasKeeper();
        /** Switch the installed keeper on or off (placement calibrator). */
        void setKeeperEnabled(boolean on);
        boolean isKeeperEnabled();
    }

    /** The listener's report, or null. */
    public static String report() {
        Listener l = listener;
        if (l == null) return null;
        try { return l.report(); } catch (Throwable e) { return null; }
    }

    private static volatile Listener listener;

    /** Install (or, with null, remove) the listener. One per process: the GPU context is global. */
    public static void setListener(Listener l) { listener = l; }

    public static Listener listener() { return listener; }

    /** Entry points for {@code LLMEngine}. */
    public static final class LLMEngineHooks {
        private LLMEngineHooks() {}

        public static void generationStarted() {
            Listener l = listener;
            if (l != null) {
                try { l.generationStarted(); } catch (Throwable ignored) { }
            }
        }

        public static void generationEnded() {
            Listener l = listener;
            if (l != null) {
                try { l.generationEnded(); } catch (Throwable ignored) { }
            }
        }
    }

    /** Called by an engine when its GPU-resident pass failed and was dropped. */
    public static void gpuPathDisabled() {
        Listener l = listener;
        if (l != null) {
            try { l.gpuPathDisabled(); } catch (Throwable ignored) { }
        }
    }

    // ---- per-token phase durations ----

    private static volatile long tokenSeq;
    private static volatile long lastGpuPhaseNs;
    private static volatile long lastCpuPhaseNs;
    private static volatile long lastTokenNs;

    /**
     * Publish the phase durations of one decoded token.
     *
     * @param gpuPhaseNs time in the GPU-resident attention phase (upload, launches, download)
     * @param cpuPhaseNs time in the CPU expert phase (including any wait for GPU-resident experts)
     * @param tokenNs    wall time of the whole token
     */
    public static void publishToken(long gpuPhaseNs, long cpuPhaseNs, long tokenNs) {
        lastGpuPhaseNs = gpuPhaseNs;
        lastCpuPhaseNs = cpuPhaseNs;
        lastTokenNs = tokenNs;
        tokenSeq = tokenSeq + 1; // single writer (the generating thread)
    }

    /** Number of tokens published so far (monotonic). */
    public static long tokenSeq() { return tokenSeq; }

    public static long lastGpuPhaseNs() { return lastGpuPhaseNs; }

    public static long lastCpuPhaseNs() { return lastCpuPhaseNs; }

    public static long lastTokenNs() { return lastTokenNs; }

    private static volatile String vramReport;

    /** Summary of the VRAM allocation guard (java21 {@code VramGuard}), published after each probe. */
    public static void setVramReport(String report) { vramReport = report; }

    /** The VRAM guard summary, or null when no allocation has been probed. */
    public static String vramReport() { return vramReport; }
}
