package it.denzosoft.llmplayer.api;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Properties;

/**
 * In-process placement calibration (docs/optimization/gpu-slower-than-cpu.md, fix F8).
 *
 * <p>The placement rule decides from sizes alone (a MoE model takes the MoE-optimized placement
 * when its non-expert tensors fit), and the previous {@code --auto-tune} reloaded the model once per
 * candidate (minutes for a 20 GB model) to compare only the heuristic placement with CPU-only. This
 * calibrator measures on the loaded model instead, by parking and restoring its GPU parts:
 * <ol>
 *   <li>stage A: GPU attention against a true CPU candidate (attention pass parked and every GPU
 *       tensor on its CPU twin), expert cache off;</li>
 *   <li>stage B: the expert cache on or off, with the winner of A;</li>
 *   <li>the GPU clock keeper on or off, when one is installed and the GPU is used;</li>
 *   <li>the matmul thread count: the configured count against four more, when the machine has
 *       more logical CPUs.</li>
 * </ol>
 * Each candidate decodes a fixed 8-token sequence on a fresh small state (the first token
 * discarded), three rounds interleaved A-B-A-B-A-B, scored by its best round; an alternative
 * replaces the default only when it wins by more than 8%.
 *
 * <p>Two corrections make the score closer to real use. A position-0 decode under-weights
 * attention, whose cost grows with the context: when a first probe decodes faster than 200 ms per
 * token (so a prefix is cheap), every candidate decodes after a 256-token prefix instead
 * ({@code -Dplacement.context=N} sets the position, 0 turns it off). And the prompt matters too:
 * each candidate's prefill of that prefix (32 tokens at least when the workload counts it) is
 * timed, and {@code -Dplacement.workload=decode|balanced|prompt} (or a number) weights it into the
 * score as 0, 1 or 8 prompt tokens per generated token; the default {@code decode} only prints it.
 * Identical configurations on
 * a laptop differ by up to 2x from minute to minute, which is why the rounds are interleaved and
 * scored best-of-N. The verdict is stored under {@code ~/.cache/llmplayer/placement}, keyed by the
 * model file, its size, the GPU, the context length and the version, and applied at the next load
 * of the same model; {@code -Dplacement.calibrate=false} ignores it, {@code force} re-measures.
 */
public final class PlacementCalibrator {

    static final String VERSION = "2";
    private static final double MARGIN = 0.08;
    private static final int ROUNDS = 3, WARM = 1, TOKENS = 7;

    /** The configuration axes the calibrator can switch. */
    public static final class Config {
        boolean gpuAttention = true, gpuMatmul = true, cache = true, keeper = true;
        int threads;

        Config copy() {
            Config c = new Config();
            c.gpuAttention = gpuAttention; c.gpuMatmul = gpuMatmul; c.cache = cache; c.keeper = keeper;
            c.threads = threads;
            return c;
        }

        @Override
        public String toString() {
            return "attention=" + (gpuAttention ? "gpu" : "cpu") + " matmul=" + (gpuMatmul ? "gpu" : "cpu")
                + " cache=" + (cache ? "on" : "off") + " keeper=" + (keeper ? "on" : "off") + " threads=" + threads;
        }
    }

    /** One timed candidate run: ms per decoded token, and ms per prefilled token (NaN without a prefix). */
    static final class Sample {
        final double decodeMs, prefillMs;
        Sample(double decodeMs, double prefillMs) { this.decodeMs = decodeMs; this.prefillMs = prefillMs; }
    }

    /** What the engine offers the calibrator. */
    interface Target {
        boolean hasGpuAttention();
        boolean hasGpuExpertCache();
        boolean hasKeeper();
        int configuredThreads();
        int logicalCpus();
        int maxContext();
        void apply(Config c);
        /**
         * On a fresh state, prefill {@code prefix} (batched, timed), then decode {@code tokens} from
         * position {@code prefix.length}; decode time is the mean after the first {@code warm} tokens.
         */
        Sample run(int[] tokens, int warm, int[] prefix);
    }

    /** Prompt tokens weighed per generated token ({@code -Dplacement.workload}). */
    static double workloadWeight() {
        String w = System.getProperty("placement.workload", "decode").trim();
        if ("decode".equalsIgnoreCase(w)) return 0;
        if ("balanced".equalsIgnoreCase(w)) return 1;
        if ("prompt".equalsIgnoreCase(w)) return 8;
        try { return Math.max(0, Double.parseDouble(w)); } catch (NumberFormatException e) { return 0; }
    }

    private final Target target;
    private final StringBuilder log = new StringBuilder();
    private final double weight = workloadWeight();
    private int[] prefix = new int[0];

    PlacementCalibrator(Target target) {
        this.target = target;
    }

    /** Run every stage; returns the chosen configuration (already applied). */
    Config calibrate(int[] tokens) {
        Config best = new Config();
        best.keeper = target.hasKeeper();
        best.threads = target.configuredThreads();
        int[] seq = tokens.length >= TOKENS ? java.util.Arrays.copyOf(tokens, TOKENS) : pad(tokens);
        // Decode position (F8 step 8) and prefill length (step 7)
        int pos = Integer.getInteger("placement.context", -1);
        if (pos < 0) {
            target.apply(best);
            double quick = target.run(seq, WARM, new int[0]).decodeMs; // also warms the JIT
            pos = quick < 200 ? 256 : 0;
        }
        pos = Math.max(0, Math.min(pos, target.maxContext() - TOKENS - 8));
        int p = Math.max(pos, weight > 0 ? 32 : 0);
        prefix = filler(tokens, p);
        String where = String.format(Locale.ROOT, "  decode timed at position %d, prefill over %d tokens, workload weight %.0f",
            p, p, weight);
        System.out.println(where);
        log.append(where).append('\n');
        if (target.hasGpuAttention()) {
            Config gpu = best.copy(); gpu.cache = false;
            Config cpu = best.copy(); cpu.cache = false; cpu.gpuAttention = false; cpu.gpuMatmul = false; cpu.keeper = false;
            best = decide("A (attention)", gpu, cpu, seq, best);
            best.cache = true; // stage B decides the cache
        }
        if (target.hasGpuExpertCache()) {
            Config on = best.copy(); on.cache = true;
            Config off = best.copy(); off.cache = false;
            best = decide("B (expert cache)", on, off, seq, best);
        } else {
            best.cache = false;
        }
        if (target.hasKeeper() && (best.gpuAttention || best.cache)) {
            Config on = best.copy(); on.keeper = true;
            Config off = best.copy(); off.keeper = false;
            best = decide("keeper", on, off, seq, best);
        }
        int more = Math.min(target.logicalCpus(), best.threads + 4);
        if (more > best.threads) {
            Config base = best.copy();
            Config alt = best.copy(); alt.threads = more;
            best = decide("threads", base, alt, seq, best);
        }
        target.apply(best);
        log.append("  chosen: ").append(best).append('\n');
        return best;
    }

    /**
     * Measure {@code def} and {@code alt} interleaved; keep {@code def} unless {@code alt} is faster
     * by more than the margin. {@code current} carries the axes this stage does not decide.
     */
    private Config decide(String stage, Config def, Config alt, int[] seq, Config current) {
        double bestDef = Double.MAX_VALUE, bestAlt = Double.MAX_VALUE;
        double decDef = Double.NaN, decAlt = Double.NaN, preDef = Double.NaN, preAlt = Double.NaN;
        for (int r = 0; r < ROUNDS; r++) {
            target.apply(def);
            Sample a = target.run(seq, WARM, prefix);
            double sa = score(a);
            if (sa < bestDef) { bestDef = sa; decDef = a.decodeMs; preDef = a.prefillMs; }
            target.apply(alt);
            Sample b = target.run(seq, WARM, prefix);
            double sb = score(b);
            if (sb < bestAlt) { bestAlt = sb; decAlt = b.decodeMs; preAlt = b.prefillMs; }
        }
        boolean altWins = bestAlt < bestDef * (1 - MARGIN);
        String line = String.format(Locale.ROOT, "  stage %s: %s %.1f ms/token%s vs %s %.1f ms/token%s -> %s",
            stage, describe(def, alt), decDef, prefillNote(preDef), describe(alt, def), decAlt, prefillNote(preAlt),
            altWins ? "alternative" : "default");
        System.out.println(line);
        log.append(line).append('\n');
        return (altWins ? alt : def).copy();
    }

    /** The axes where {@code c} differs from {@code other}. */
    private static String describe(Config c, Config other) {
        List<String> d = new ArrayList<>();
        if (c.gpuAttention != other.gpuAttention) d.add("attention " + (c.gpuAttention ? "GPU" : "CPU"));
        if (c.cache != other.cache) d.add("cache " + (c.cache ? "on" : "off"));
        if (c.keeper != other.keeper) d.add("keeper " + (c.keeper ? "on" : "off"));
        if (c.threads != other.threads) d.add(c.threads + " threads");
        return d.isEmpty() ? "same" : String.join(", ", d);
    }

    private double score(Sample s) {
        return s.decodeMs + (weight > 0 && !Double.isNaN(s.prefillMs) ? weight * s.prefillMs : 0);
    }

    private static String prefillNote(double prefillMs) {
        return Double.isNaN(prefillMs) ? "" : String.format(Locale.ROOT, " (prefill %.1f ms/token)", prefillMs);
    }

    /** {@code n} prompt tokens for the prefix: the calibration text repeated. */
    private static int[] filler(int[] tokens, int n) {
        int[] out = new int[n];
        for (int i = 0; i < n; i++) out[i] = tokens.length == 0 ? 0 : tokens[i % tokens.length];
        return out;
    }

    String log() { return log.toString(); }

    private static int[] pad(int[] tokens) {
        int[] out = new int[TOKENS];
        for (int i = 0; i < TOKENS; i++) out[i] = tokens.length == 0 ? 0 : tokens[i % tokens.length];
        return out;
    }

    // ---- persistence ----

    static Path verdictFile(String key) {
        String dir = System.getProperty("placement.dir",
            System.getProperty("user.home", ".") + "/.cache/llmplayer/placement");
        return Paths.get(dir, key.replaceAll("[^A-Za-z0-9._-]", "_") + ".properties");
    }

    static void save(Path file, Config c, String log) {
        Properties p = new Properties();
        p.setProperty("gpuAttention", String.valueOf(c.gpuAttention));
        p.setProperty("gpuMatmul", String.valueOf(c.gpuMatmul));
        p.setProperty("cache", String.valueOf(c.cache));
        p.setProperty("keeper", String.valueOf(c.keeper));
        p.setProperty("threads", String.valueOf(c.threads));
        try {
            Files.createDirectories(file.getParent());
            Path tmp = Files.createTempFile(file.getParent(), "placement", ".tmp");
            try (OutputStream out = Files.newOutputStream(tmp)) {
                p.store(out, "LLMPlayer placement calibration\n" + log.replace('\n', ' '));
            }
            Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException | RuntimeException ignored) {
            // no persistence: the next load uses the heuristic placement again
        }
    }

    static Config load(Path file) {
        if (!Files.isRegularFile(file)) return null;
        Properties p = new Properties();
        try (InputStream in = Files.newInputStream(file)) {
            p.load(in);
        } catch (IOException e) {
            return null;
        }
        Config c = new Config();
        c.gpuAttention = Boolean.parseBoolean(p.getProperty("gpuAttention", "true"));
        c.gpuMatmul = Boolean.parseBoolean(p.getProperty("gpuMatmul", "true"));
        c.cache = Boolean.parseBoolean(p.getProperty("cache", "true"));
        c.keeper = Boolean.parseBoolean(p.getProperty("keeper", "true"));
        try { c.threads = Integer.parseInt(p.getProperty("threads", "0")); } catch (NumberFormatException e) { c.threads = 0; }
        return c;
    }
}
