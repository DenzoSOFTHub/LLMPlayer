package it.denzosoft.llmplayer.gpu;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Locale;

/**
 * Allocation verification and bandwidth canary (docs/optimization/gpu-slower-than-cpu.md, fix F7
 * steps 4 and 5).
 *
 * <p>Under WSL2 the WDDM driver lets {@code cuMemAlloc} succeed beyond the physical VRAM. Measured
 * on an RTX 4050 Laptop GPU (6 GB): 8 GB of 256 MiB allocations all succeeded; a device-to-device
 * copy inside the newest block ran at about 146 GB/s up to about 5.6 GB allocated, then fell to 24
 * and 9 GB/s, because the driver had placed the block in shared system memory. {@code cuMemGetInfo}
 * reported 0 free bytes about 500 MiB before that point: its figure is conservative, but it
 * saturates at 0 and cannot tell where the fast memory ends. A weight matrix or an expert-cache
 * chunk placed in shared memory runs ten times slower than on the CPU, silently.
 *
 * <p>The guard therefore verifies by measurement. An allocation of at least 1 MiB made while the
 * free figure is below the reserve ({@code -Dcuda.vram.reserve.mb}, default 256) is probed: a kernel
 * reads it with cache-volatile loads ({@code __ldcv}), timed against the same kernel over a
 * reference buffer allocated when the context was created (so in VRAM by construction),
 * interleaved, after a short warm-up that raises the clocks. A cache-volatile load re-fetches a
 * system-memory line over PCIe on every read, while a VRAM line may be served from the L2 cache;
 * so a block in shared memory can never look fast, whatever the L2 size (a plain device-to-device
 * copy of an 8 MiB block is served from the 24 MB L2 of the RTX 4050 and measured 840 GB/s). The
 * allocation is rejected when it reads below a third of the reference. A probe whose reference
 * stays below 60 GB/s after the warm-up is inconclusive and accepts. Allocations
 * made while plenty of memory is free are never probed, so the check costs one {@code cuMemGetInfo}
 * per large allocation in the common case. {@code -Dcuda.vram.verify=false} disables it.
 *
 * <p>Callers decide what a rejection means: a per-tensor weight becomes its CPU twin, a MoE
 * attention layer stays on the CPU, the expert cache stops growing.
 */
public final class VramGuard {

    private static final long MIN_CHECKED = 1L << 20;
    private static final long PROBE_BYTES = 8L << 20;
    private static final int PROBE_GRID = 128, PROBE_BLOCK = 256;
    private static final String PROBE_SRC =
        "extern \"C\" __global__ void vram_probe(const uint4* __restrict__ src, unsigned int* __restrict__ out,\n"
      + "                                        const long long n16) {\n"
      + "    unsigned int acc = 0;\n"
      + "    const long long stride = (long long) gridDim.x * blockDim.x;\n"
      + "    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) {\n"
      + "        uint4 v = __ldcv(src + i);\n"
      + "        acc += v.x ^ v.y ^ v.z ^ v.w;\n"
      + "    }\n"
      + "    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;\n"
      + "}\n";
    private static final long RESERVE = Long.getLong("cuda.vram.reserve.mb", 256L) << 20;
    private static final double MIN_RATIO = 1.0 / 3;
    private static final double MIN_TRUSTED_GBS = 60;
    private static final long WARM_NS = 300_000_000L;

    /** Thrown by a checked allocation that landed in shared system memory (it has been freed). */
    public static final class VramExhaustedException extends RuntimeException {
        public VramExhaustedException(String message) { super(message); }
    }

    private final CudaContext ctx;
    private final long ref, scratch;
    private final MemorySegment ev0, ev1, probeFunc;
    private final Arena arena = Arena.ofShared();
    private final KernelParams probePB = new KernelParams(arena, 3);
    private int probes, rejected, inconclusive;
    private long lastProbeNs;
    private double lastRatio = Double.NaN, lastRefGbs = Double.NaN;

    private VramGuard(CudaContext ctx, long ref, long scratch) {
        this.ctx = ctx;
        this.ref = ref;
        this.scratch = scratch;
        this.ev0 = ctx.createEvent(true);
        this.ev1 = ctx.createEvent(true);
        this.probeFunc = ctx.compileKernelSource("vram_probe.cu", PROBE_SRC, "vram_probe");
    }

    /** Allocate the reference buffers; null when disabled or when that fails. */
    static VramGuard create(CudaContext ctx) {
        if ("false".equals(System.getProperty("cuda.vram.verify", "true"))) return null;
        long a = 0, b = 0;
        try {
            a = ctx.allocBuffer(PROBE_BYTES);
            b = ctx.allocBuffer((long) PROBE_GRID * PROBE_BLOCK * 4);
            ctx.fillBufferZero(a, PROBE_BYTES);
            ctx.finish();
            return new VramGuard(ctx, a, b);
        } catch (RuntimeException e) {
            if (a != 0) ctx.freeBuffer(a);
            if (b != 0) ctx.freeBuffer(b);
            return null;
        }
    }

    /**
     * Whether the allocation {@code [ptr, ptr + bytes)} is usable. Its contents must be written
     * (or zero-filled) before the call, so that the driver has committed its pages.
     */
    public synchronized boolean verify(long ptr, long bytes, String what) {
        if (bytes < MIN_CHECKED) return true;
        long free = ctx.getMemoryInfo()[0];
        if (free >= RESERVE) return true;
        double r = probe(ptr, bytes);
        if (Double.isNaN(r)) {
            inconclusive++;
            publish();
            return true;
        }
        if (r >= MIN_RATIO) {
            publish();
            return true;
        }
        rejected++;
        System.out.println(String.format(Locale.ROOT,
            "  VRAM guard: %s (%d MiB) landed in shared system memory (%.0f%% of the reference read rate, %.0f GB/s) - released",
            what, bytes >> 20, 100 * r, lastRefGbs));
        publish();
        return false;
    }

    /**
     * Allocate {@code bytes}, zero-fill and verify; frees the buffer and throws
     * {@link VramExhaustedException} when it landed in shared memory.
     */
    public long allocChecked(long bytes, String what) {
        long p = ctx.allocBuffer(bytes);
        if (bytes < MIN_CHECKED) return p;
        ctx.fillBufferZero(p, bytes);
        if (verify(p, bytes, what)) return p;
        ctx.freeBuffer(p);
        throw new VramExhaustedException(what + ": " + (bytes >> 20) + " MiB would live in shared system memory");
    }

    /**
     * Read bandwidth of {@code ptr} relative to the reference buffer under the same clock, or NaN
     * when the reference never left the idle clock.
     */
    public synchronized double probe(long ptr, long bytes) {
        long n = Math.min(PROBE_BYTES, bytes);
        probes++;
        // Warm-up: raise the memory clock until two consecutive reference rounds agree.
        // A probe right after another one finds the clock already raised: two rounds suffice.
        double refBest = 0, prev = 0;
        long t0 = System.nanoTime();
        long warm = t0 - lastProbeNs < 200_000_000L ? 0 : WARM_NS;
        while (System.nanoTime() - t0 < warm) {
            double g = gbs(ref, n);
            refBest = Math.max(refBest, g);
            if (prev > 0 && Math.abs(g - prev) < 0.05 * g && System.nanoTime() - t0 > 30_000_000L
                    && g >= MIN_TRUSTED_GBS) break;
            prev = g;
        }
        double cand = 0;
        for (int i = 0; i < 3; i++) {
            cand = Math.max(cand, gbs(ptr, n));
            refBest = Math.max(refBest, gbs(ref, n));
        }
        lastRefGbs = refBest;
        lastProbeNs = System.nanoTime();
        if (refBest < MIN_TRUSTED_GBS) {
            lastRatio = Double.NaN;
            return Double.NaN;
        }
        lastRatio = cand / refBest;
        return lastRatio;
    }

    private double gbs(long src, long n) {
        MemorySegment s = ctx.getStream();
        long n16 = Math.max(1, n / 16);
        probePB.setLong(0, src).setLong(1, scratch).setLong(2, n16);
        ctx.recordEvent(ev0, s);
        int err = CudaBindings.launchKernel(probeFunc, PROBE_GRID, 1, 1, PROBE_BLOCK, 1, 1, 0, s,
            probePB.ptrs(), MemorySegment.NULL);
        if (err != CudaBindings.CUDA_SUCCESS) throw new RuntimeException("VRAM probe launch error: " + err);
        ctx.recordEvent(ev1, s);
        ctx.finish();
        double ms = Math.max(1e-3, ctx.elapsedMs(ev0, ev1));
        return n16 * 16.0 / (ms * 1e6);
    }

    private void publish() {
        GpuActivity.setVramReport(report());
    }

    /** One-line summary for the stats block and the metrics. */
    public synchronized String report() {
        return String.format(Locale.ROOT, "%d probes, %d rejected, %d inconclusive%s", probes, rejected, inconclusive,
            Double.isNaN(lastRatio) ? "" : String.format(Locale.ROOT, ", last %.0f%% of %.0f GB/s", 100 * lastRatio, lastRefGbs));
    }

    void close() {
        try { ctx.freeBuffer(ref); } catch (Exception ignored) { }
        try { ctx.freeBuffer(scratch); } catch (Exception ignored) { }
        ctx.destroyEvent(ev0);
        ctx.destroyEvent(ev1);
        arena.close();
    }
}
