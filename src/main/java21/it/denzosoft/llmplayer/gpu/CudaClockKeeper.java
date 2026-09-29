package it.denzosoft.llmplayer.gpu;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.locks.LockSupport;

/**
 * GPU clock keeper and generation-window P-state sampler (docs/optimization/gpu-slower-than-cpu.md,
 * cause C2 and fix F1).
 *
 * <p><b>Why.</b> Under the MoE-optimized placement a decoded token alternates short GPU bursts (the
 * attention half of each layer, 0.1-4 ms) with long CPU phases (the routed experts, 2-27 ms per
 * layer). The driver's governor reads that duty cycle as idle and keeps the GPU in P8/P5
 * (210-510 MHz SM, 405-810 MHz memory), so every GPU phase runs several times below spec. A light
 * kernel on a side stream that keeps the GPU from looking idle took Qwen3-Coder-30B from 291 to
 * 191 ms per token in an interleaved pair.
 *
 * <p><b>What it does</b> ({@code -Dcuda.clockkeeper=true|adaptive|<busy-us>}, opt-in): while a
 * generation is in progress (the {@link GpuActivity} hooks in {@code LLMEngine.generate}, plus a
 * short linger so multi-turn requests do not pay the clock ramp again), a daemon thread repeatedly
 * launches a tiny spin kernel (default 4 blocks of 64 threads: it never fills the SMs) on a stream
 * of the <em>least</em> priority — the work stream is created at the greatest priority, so the
 * block scheduler always dispatches the model's pending blocks first — for {@code busy} µs, then
 * sleeps {@code sleep} µs. It waits for each burst by polling an event and parking, not by
 * {@code cuStreamSynchronize}, which would spin a CPU core under the default scheduling mode.
 * <ul>
 *   <li>The burst length is recalibrated continuously from event timing (the minimum of the last
 *       eight bursts), because the kernel's duration scales with the SM clock the keeper itself
 *       changes: a one-off calibration at start-up made bursts 5-7x too long once the clock fell.</li>
 *   <li>Adaptive duty (default): the duty cycle is chosen among off, 1/8, 1/4 and 1/2 by the
 *       mean wall time per decoded token (see {@link #adapt()}), because on a laptop the GPU power
 *       the keeper adds is taken from the CPU's budget.</li>
 *   <li>Never during model loading (a keeper started with the context stalled preloads for
 *       minutes), and it stops when the engine drops its GPU path.</li>
 * </ul>
 * The memory-fill variant ({@code -Dcuda.clockkeeper.mb}) raises the memory clock too but competes
 * for the bandwidth the resident experts need; it measured 4-5x slower and stays an experiment.
 *
 * <p>With {@code -Dgpu.pstate.sample=true} and no keeper, only the sampler runs: the CLI stats block
 * then reports the P-state histogram of every generation, for comparing runs with and without it.
 *
 * <p>Every CUDA resource (stream, kernel module, buffers, events) is created in the constructor on
 * the installing thread, before any generation and therefore before any graph capture.
 */
final class CudaClockKeeper implements GpuActivity.Listener {

    private static final long LINGER_NS = Long.getLong("cuda.clockkeeper.linger.ms", 3000L) * 1_000_000L;
    private static final boolean VERBOSE = "true".equals(System.getProperty("cuda.clockkeeper.verbose"));
    private static final long SAMPLE_NS = 100_000_000L;

    private final CudaContext ctx;
    private final boolean keep;       // a spin keeper is installed (false: sampling only)
    private volatile boolean keeperOn; // the installed keeper is switched on
    private final boolean adaptive;
    private final int memMb;
    private volatile int busyUs, sleepUs;

    private MemorySegment side, fn, evStart, evEnd;
    private KernelParams params;
    private int grid, block, launchesPerBurst = 1;
    private double itersPerUs;         // spin iterations per microsecond at the current clock
    private final double[] recent = new double[8];
    private int recentN;

    private final Object lock = new Object();
    private int active;                // generations in progress
    private volatile long lingerUntil; // keep running until this nanoTime after the last generation
    private volatile boolean gpuPathLost;
    private volatile boolean running = true;
    private final Thread thread;

    // adaptive controller state (winAttn accumulates token wall time)
    private long lastSeq = -1, winAttn;
    private int winN;
    private int adjustments;

    // sampler
    private final NvmlSampler nvml;
    private final int[] sample = new int[4];
    private final long[] pstateHist = new long[17];
    private long samples, smSum, memSum, powerSum;
    private long lastSampleNs, bursts;

    CudaClockKeeper(CudaContext ctx) {
        this.ctx = ctx;
        String prop = System.getProperty("cuda.clockkeeper");
        this.keep = prop != null && !"false".equals(prop);
        this.keeperOn = keep;
        int busy = 1000;
        if (keep && !"true".equals(prop) && !"adaptive".equals(prop)) {
            try { busy = Integer.parseInt(prop); } catch (NumberFormatException ignored) { }
        }
        this.busyUs = Math.max(50, busy);
        this.sleepUs = Math.max(0, Integer.getInteger("cuda.clockkeeper.sleep", 1000));
        this.adaptive = keep && !"false".equals(System.getProperty("cuda.clockkeeper.adaptive", "true"));
        if (adaptive) setLevel(2);
        this.memMb = Integer.getInteger("cuda.clockkeeper.mb", 0);
        this.nvml = NvmlSampler.open(ctx.getDeviceInfo().index());
        if (keep) initKernel();
        thread = new Thread(this::loop, "cuda-clock-keeper");
        thread.setDaemon(true);
        thread.start();
        if (keep) {
            System.err.println("CUDA clock keeper: installed (" + (adaptive ? "adaptive, start " : "fixed ")
                + busyUs + " us busy / " + sleepUs + " us sleep, "
                + (memMb > 0 ? memMb + " MB fill" : grid + "x" + block + " spin")
                + ", stream priority " + ctx.leastStreamPriority() + " below work stream " + ctx.workStreamPriority()
                + String.format(", %.0f iters/us", itersPerUs) + (nvml != null ? ", NVML sampling" : "") + ")");
        }
    }

    private void initKernel() {
        side = ctx.createLowPriorityStream();
        params = new KernelParams(ctx.getArena(), 2);
        if (memMb > 0) {
            fn = ctx.compileKernel("kernels/cuda/fill_zero.cu", "fill_zero");
            int n = memMb * 1024 * 1024 / 4;
            params.setLong(0, ctx.allocBuffer((long) n * 4)).setInt(1, n);
            block = 256;
            grid = (n + block - 1) / block;
        } else {
            fn = ctx.compileKernel("kernels/cuda/spin.cu", "gpu_spin");
            params.setLong(0, ctx.allocBuffer(4096)).setInt(1, 2000);
            grid = Integer.getInteger("cuda.clockkeeper.blocks", 4);
            block = Integer.getInteger("cuda.clockkeeper.threads", 64);
        }
        evStart = ctx.createEvent(true);
        evEnd = ctx.createEvent(true);
        // Initial calibration at whatever clock the GPU has now; refined after every burst.
        double us = timedBurst(2000, 1);
        if (memMb > 0) {
            launchesPerBurst = (int) Math.max(1, busyUs / Math.max(1.0, us));
        } else {
            itersPerUs = 2000 / Math.max(1.0, us);
        }
    }

    /** One burst of {@code launches} launches ({@code iters} spin iterations each); its GPU time in µs. */
    private double timedBurst(int iters, int launches) {
        if (memMb == 0) params.setInt(1, iters);
        ctx.recordEvent(evStart, side);
        for (int i = 0; i < launches; i++) ctx.launchKernelOnStream(fn, grid, block, 0, params.ptrs(), side);
        ctx.recordEvent(evEnd, side);
        ctx.waitEventParked(evEnd, 50_000L);
        return ctx.elapsedMs(evStart, evEnd) * 1000.0;
    }

    private boolean isActive() {
        if (gpuPathLost) return false;
        return active > 0 || System.nanoTime() < lingerUntil;
    }

    private void loop() {
        try {
            ctx.ensureCurrent();
            while (running) {
                if (!isActive()) {
                    LockSupport.parkNanos(this, 50_000_000L);
                    continue;
                }
                long now = System.nanoTime();
                if (nvml != null && now - lastSampleNs >= SAMPLE_NS) {
                    lastSampleNs = now;
                    takeSample();
                }
                if (!keep || !keeperOn) {
                    LockSupport.parkNanos(this, SAMPLE_NS);
                    continue;
                }
                if (adaptive && levelOff) {
                    adapt(); // keep scoring while switched off
                    LockSupport.parkNanos(this, 2_000_000L);
                    continue;
                }
                burst();
                if (adaptive) adapt();
                int s = sleepUs;
                if (s > 0) LockSupport.parkNanos(this, s * 1000L);
            }
        } catch (Throwable t) {
            System.err.println("CUDA clock keeper stopped: " + t);
        }
    }

    private void burst() {
        int b = busyUs;
        if (memMb > 0) {
            timedBurst(0, launchesPerBurst);
        } else {
            int iters = (int) Math.max(100, Math.min(Integer.MAX_VALUE / 2, itersPerUs * b));
            double us = timedBurst(iters, 1);
            if (us > 0) {
                recent[recentN++ % recent.length] = iters / us;
                if (recentN >= recent.length) {
                    // Fastest recent rate = the least disturbed burst (a keeper burst queued behind
                    // the model's kernels measures long and would otherwise shrink the next one).
                    double best = 0;
                    for (double r : recent) best = Math.max(best, r);
                    itersPerUs = best;
                    recentN = 0;
                }
            }
        }
        bursts++;
    }

    /**
     * Adaptive duty (default): an epsilon-greedy search over a few duty levels — off, 1/8, 1/4,
     * 1/2 of the time busy — scored by the mean wall time per decoded token that the engines
     * publish. Minimising the attention phase alone is the wrong goal on a laptop: CPU and GPU
     * share the power budget, and a keeper holding the GPU at P0 (17-23 W) slowed the CPU by 15-30%
     * (the in-process CPU probe), so the routed experts on the CPU lost what the GPU phases gained.
     * Each level keeps an exponential average of its token time; the controller runs the best
     * level and every fourth window tries a neighbouring one.
     */
    private void adapt() {
        long seq = GpuActivity.tokenSeq();
        if (seq == lastSeq) return;
        if (lastSeq >= 0) {
            long tok = GpuActivity.lastTokenNs();
            if (tok > 0) { winAttn += tok; winN++; }
        }
        lastSeq = seq;
        if (winN < WINDOW) return;
        double mean = winAttn / (double) winN;
        winAttn = 0;
        winN = 0;
        levelEma[level] = levelEma[level] <= 0 ? mean : 0.6 * levelEma[level] + 0.4 * mean;
        windows++;
        int best = level;
        for (int i = 0; i < LEVELS.length; i++) {
            if (levelEma[i] > 0 && levelEma[i] < levelEma[best]) best = i;
        }
        int next = best;
        // explore: untried neighbours first, then a neighbour every fourth window
        if (windows % 4 == 0 || levelEma[Math.max(0, best - 1)] <= 0 || levelEma[Math.min(LEVELS.length - 1, best + 1)] <= 0) {
            int up = Math.min(LEVELS.length - 1, best + 1), down = Math.max(0, best - 1);
            if (levelEma[down] <= 0 && down != best) next = down;
            else if (levelEma[up] <= 0 && up != best) next = up;
            else if (windows % 4 == 0) next = (windows / 4) % 2 == 0 ? up : down;
        }
        if (next != level) {
            adjustments++;
            if (VERBOSE) {
                System.err.printf("[clock-keeper] level %s: %.1f ms/token (best %s) -> %s%n",
                    LEVEL_NAMES[level], mean / 1e6, LEVEL_NAMES[best], LEVEL_NAMES[next]);
            }
            setLevel(next);
        }
    }

    private static final int WINDOW = 6;
    private static final double[] LEVELS = {0.0, 0.125, 0.25, 0.5};
    private static final String[] LEVEL_NAMES = {"off", "1/8", "1/4", "1/2"};
    private final double[] levelEma = new double[LEVELS.length];
    private int level = 2, windows;

    private void setLevel(int l) {
        level = l;
        double d = LEVELS[l];
        if (d <= 0) { levelOff = true; return; }
        levelOff = false;
        busyUs = BASE_BUSY;
        sleepUs = (int) Math.round(BASE_BUSY * (1 - d) / d);
    }

    private static final int BASE_BUSY = 1000;
    private volatile boolean levelOff;

    private void takeSample() {
        if (!nvml.sample(sample)) return;
        int ps = sample[0];
        pstateHist[ps >= 0 && ps < 16 ? ps : 16]++;
        samples++;
        smSum += Math.max(0, sample[1]);
        memSum += Math.max(0, sample[2]);
        powerSum += Math.max(0, sample[3]);
    }

    @Override
    public void generationStarted() {
        synchronized (lock) {
            if (active == 0 && System.nanoTime() >= lingerUntil) {
                // a new generation window: fresh histogram
                java.util.Arrays.fill(pstateHist, 0);
                samples = smSum = memSum = powerSum = 0;
            }
            active++;
            lingerUntil = 0;
        }
        LockSupport.unpark(thread);
    }

    @Override
    public void generationEnded() {
        synchronized (lock) {
            if (active > 0) active--;
            if (active == 0) lingerUntil = System.nanoTime() + LINGER_NS;
        }
    }

    @Override
    public boolean hasKeeper() { return keep; }

    @Override
    public void setKeeperEnabled(boolean on) { keeperOn = on && keep; }

    @Override
    public boolean isKeeperEnabled() { return keeperOn; }

    @Override
    public void gpuPathDisabled() {
        gpuPathLost = true;
    }

    @Override
    public String report() {
        StringBuilder sb = new StringBuilder();
        if (keep) {
            int b = busyUs, s = sleepUs;
            if (adaptive) {
                StringBuilder lv = new StringBuilder();
                for (int i = 0; i < LEVELS.length; i++) {
                    if (levelEma[i] > 0) lv.append(lv.length() == 0 ? "" : ", ").append(LEVEL_NAMES[i]).append(' ')
                        .append(String.format("%.0f", levelEma[i] / 1e6)).append(" ms");
                }
                sb.append(String.format("GPU clock keeper: adaptive, level %s (%s), %d bursts, %d switches",
                    LEVEL_NAMES[level], lv, bursts, adjustments));
            } else {
                sb.append(String.format("GPU clock keeper: fixed, duty %d/%d us (%.0f%%), %d bursts",
                    b, s, 100.0 * b / Math.max(1, b + s), bursts));
            }
        }
        long n = samples;
        if (n > 0) {
            if (sb.length() > 0) sb.append("; ");
            sb.append("GPU during generation (").append(n).append(" samples @100ms): ");
            boolean first = true;
            for (int i = 0; i < 17; i++) {
                if (pstateHist[i] == 0) continue;
                if (!first) sb.append(' ');
                first = false;
                sb.append(i < 16 ? "P" + i : "P?").append('=').append(Math.round(100.0 * pstateHist[i] / n)).append('%');
            }
            sb.append(String.format(", mean SM %d MHz, mem %d MHz, %.1f W", smSum / n, memSum / n, powerSum / n / 1000.0));
        }
        return sb.length() == 0 ? null : sb.toString();
    }

    void shutdown() {
        running = false;
        LockSupport.unpark(thread);
        try { thread.join(500); } catch (InterruptedException ignored) { Thread.currentThread().interrupt(); }
    }

    /** Whether the context holds a keeper that launches kernels (vs. only sampling). */
    boolean keeps() { return keep; }
}
