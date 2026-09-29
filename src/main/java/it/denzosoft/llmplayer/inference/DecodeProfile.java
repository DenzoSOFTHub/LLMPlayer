package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.gpu.GpuActivity;

/**
 * Per-token phase timing for the MoE engines (Qwen3-MoE family, DeepSeek2, Qwen3.5).
 *
 * <p>Two layers of timing, both allocation-free:
 * <ul>
 *   <li><b>Always on</b>: the GPU-resident attention phase and the expert phase of every token
 *       ({@link #attn} / {@link #moe}), published per decoded token through
 *       {@link GpuActivity#publishToken}. The CUDA clock keeper's adaptive duty cycle reads them,
 *       so they cannot depend on {@code -Dcpu.profile}.</li>
 *   <li><b>{@code -Dcpu.profile=true}</b>: every named phase is accumulated, and every ten decoded
 *       tokens one line is printed with the cumulative per-token averages (the historical format,
 *       which the measurement scripts parse), followed by the averages of the last ten tokens, the
 *       split of the expert phase (routing, GPU launch, CPU experts, GPU wait, the rest), a CPU
 *       speed probe and any extra text the engine supplies (the expert-cache hit rate).</li>
 * </ul>
 *
 * <p>Prefill: {@link #startGeneration()} resets everything at the top of a prefill, and the first
 * {@link #endToken} after it (the projection that closes the prompt) drops the prompt's layer time,
 * so every average is per decoded token even when the prompt runs token by token (GPU mode). Before
 * this, a GPU-mode run folded its whole prompt into the decode averages while a CPU-only run, whose
 * batched prefill never touched the counters, did not — which made MiniMax-M2 look 35% slower on
 * the GPU when it was 18-31% faster (docs/optimization/gpu-slower-than-cpu.md, section 2.1).
 */
public final class DecodeProfile {

    /** Detailed profiling ({@code -Dcpu.profile=true}). */
    public final boolean detailed = "true".equals(System.getProperty("cpu.profile"));

    private final String tag;
    private final String[] names;
    private final int attnPhase, moePhase, outputPhase;
    private final long[] sums, marks;
    private int tokens, markTokens;
    private boolean prefillOpen = true;

    // moe split: route, launch, cpu, wait (detailed only)
    private final long[] split = new long[4], splitMark = new long[4];
    private boolean splitSeen;

    // always-on per-token accumulators
    private long tokAttnNs, tokMoeNs, lastTokenEnd;

    /**
     * @param tag         engine tag printed in the profile line
     * @param names       phase names in print order (e.g. {@code "attn(GQA)"}, {@code "moe_ffn"})
     * @param attnPhase   index of the GPU-resident attention phase in {@code names}
     * @param moePhase    index of the expert phase
     * @param outputPhase index of the output projection
     */
    public DecodeProfile(String tag, String[] names, int attnPhase, int moePhase, int outputPhase) {
        this.tag = tag;
        this.names = names;
        this.attnPhase = attnPhase;
        this.moePhase = moePhase;
        this.outputPhase = outputPhase;
        this.sums = new long[names.length];
        this.marks = new long[names.length];
    }

    /** Reset every counter; call at the top of a prefill (every generation starts with one). */
    public void startGeneration() {
        java.util.Arrays.fill(sums, 0);
        java.util.Arrays.fill(marks, 0);
        java.util.Arrays.fill(split, 0);
        java.util.Arrays.fill(splitMark, 0);
        tokens = markTokens = 0;
        tokAttnNs = tokMoeNs = 0;
        prefillOpen = true;
        splitSeen = false;
        lastTokenEnd = 0;
    }

    /** Add time to a named phase (detailed profiling only; the attention and expert phases are always counted). */
    public void add(int phase, long ns) {
        if (phase == attnPhase) tokAttnNs += ns;
        else if (phase == moePhase) tokMoeNs += ns;
        if (detailed) sums[phase] += ns;
    }

    /** Always-on attention-phase time. */
    public void attn(long ns) { add(attnPhase, ns); }

    /** Always-on expert-phase time. */
    public void moe(long ns) { add(moePhase, ns); }

    /** Expert-phase time accumulated in the current token so far (always on). */
    public long tokenMoeNs() { return tokMoeNs; }

    /**
     * Split of the expert phase (detailed only): routing, queueing the GPU-resident experts, the
     * CPU experts and the wait for the GPU-resident ones. The rest of the phase (weighted sum,
     * shared expert) is printed as the remainder.
     */
    public void moeSplit(long routeNs, long launchNs, long cpuNs, long waitNs) {
        split[0] += routeNs;
        split[1] += launchNs;
        split[2] += cpuNs;
        split[3] += waitNs;
        splitSeen = true;
    }

    /**
     * Close one token at the output projection.
     *
     * @param outputNs time of the output projection (counted only when detailed)
     * @param extra    text appended to the printed line (e.g. expert-cache statistics), queried
     *                 only when a line is printed; may be null
     */
    public void endToken(long outputNs, java.util.function.Supplier<String> extra) {
        long now = System.nanoTime();
        if (prefillOpen) {
            // The projection that closes the prompt: drop the prompt's layer time.
            prefillOpen = false;
            java.util.Arrays.fill(sums, 0);
            java.util.Arrays.fill(split, 0);
            splitSeen = false;
            tokAttnNs = tokMoeNs = 0;
            lastTokenEnd = now;
            return;
        }
        GpuActivity.publishToken(tokAttnNs, tokMoeNs, lastTokenEnd == 0 ? 0 : now - lastTokenEnd);
        tokAttnNs = tokMoeNs = 0;
        lastTokenEnd = now;
        if (!detailed) return;
        sums[outputPhase] += outputNs;
        tokens++;
        if (tokens % 10 == 0) print(extra == null ? null : extra.get());
    }

    private void print(String extra) {
        int n = tokens;
        double ms = 1e6;
        StringBuilder sb = new StringBuilder(256);
        sb.append("[cpu-profile ").append(tag).append("] ").append(n).append(" tokens, per-token avg (ms): ");
        long total = 0;
        for (int i = 0; i < names.length; i++) {
            if (i > 0) sb.append(' ');
            sb.append(names[i]).append('=').append(fmt(sums[i] / ms / n));
            total += sums[i];
        }
        sb.append(" | total=").append(fmt(total / ms / n));
        int w = n - markTokens;
        if (w > 0 && markTokens > 0) {
            sb.append(" | last ").append(w).append(':');
            long wt = 0;
            for (int i = 0; i < names.length; i++) {
                long d = sums[i] - marks[i];
                wt += d;
                if (i == attnPhase || i == moePhase || i == outputPhase) {
                    sb.append(' ').append(shortName(names[i])).append('=').append(fmt(d / ms / w));
                }
            }
            sb.append(" sum=").append(fmt(wt / ms / w));
        }
        if (splitSeen) {
            long moeSum = sums[moePhase];
            long rest = moeSum - split[0] - split[1] - split[2] - split[3];
            sb.append(" | moe: route=").append(fmt(split[0] / ms / n))
              .append(" gpu_launch=").append(fmt(split[1] / ms / n))
              .append(" cpu_experts=").append(fmt(split[2] / ms / n))
              .append(" gpu_wait=").append(fmt(split[3] / ms / n))
              .append(" rest=").append(fmt(rest / ms / n));
        }
        sb.append(" | cpu_probe=").append(cpuProbeUs()).append("us");
        if (extra != null) sb.append(" | ").append(extra);
        System.out.println(sb);
        System.arraycopy(sums, 0, marks, 0, sums.length);
        System.arraycopy(split, 0, splitMark, 0, split.length);
        markTokens = n;
    }

    private static String shortName(String name) {
        int p = name.indexOf('(');
        return p > 0 ? name.substring(0, p) : name;
    }

    private static String fmt(double v) {
        return String.format(java.util.Locale.ROOT, "%.1f", v);
    }

    // ---- CPU speed probe ----

    private static volatile int probeSink;

    /**
     * Time of a fixed dependent integer chain on the calling thread, in microseconds: the only
     * in-VM proxy for the CPU clock under WSL2, where /proc/cpuinfo reports a constant frequency
     * and cpufreq is absent. Compare it across the lines of one run; about 1 ms per call.
     */
    public static long cpuProbeUs() {
        long t0 = System.nanoTime();
        int x = probeSink | 1;
        for (int i = 0; i < 400_000; i++) {
            x = x * 1103515245 + 12345;
            x ^= x >>> 13;
        }
        probeSink = x;
        return (System.nanoTime() - t0) / 1000;
    }
}
