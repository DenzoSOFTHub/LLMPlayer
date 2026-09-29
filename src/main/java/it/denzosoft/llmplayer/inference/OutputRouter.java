package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.tensor.FloatTensor;

/**
 * Runs a large GPU-resident projection (the output projection of the MoE engines) on whichever
 * device is faster at the moment: the GPU per-tensor path or the tensor's SIMD CPU twin.
 *
 * <p>Why a runtime decision: under the MoE-optimized placement the GPU spends the decode in an
 * idle P-state, and MiniMax-M2's 422 MB Q5_K output projection took 59-75 ms on the GPU against
 * 30-34 ms on the CPU, while Qwen3-Coder's took 9-14 ms on the GPU against 27-42 on the CPU — and
 * a GPU clock keeper flips the first case again. A load-time decision would measure the GPU at the
 * clock it has right after the upload. So the router times the GPU path on every call, samples the
 * CPU twin on two early decode tokens (after the matmul pool is on, so it is measured on the pool
 * it will use) and then once every {@link #REPROBE} tokens, and switches when the other device is
 * faster by more than 20%.
 *
 * <p>The two paths are the same kernel math with a different summation order, so switching changes
 * the logits in the last bits. Disable with {@code -Doutput.router=false}.
 */
final class OutputRouter {

    private static final boolean ENABLED = !"false".equals(System.getProperty("output.router", "true"));
    private static final boolean VERBOSE = "true".equals(System.getProperty("output.router.verbose"));
    private static final int REPROBE = 64;
    private static final double MARGIN = 0.8;

    private final FloatTensor w;
    private final String name;
    private final boolean eligible;
    private double gpuMs = -1, cpuMs = -1;
    private boolean useCpu;
    private long calls;

    OutputRouter(FloatTensor w, String name) {
        this.w = w;
        this.name = name;
        this.eligible = ENABLED && w != null && w.isGpuResident();
    }

    void matmul(float[] in, float[] out, int rows, int cols) {
        if (!eligible) {
            w.matmulParallel(in, out, rows, cols);
            return;
        }
        long c = ++calls;
        boolean cpu;
        if (c == 6 || c == 7) cpu = true;                        // first CPU samples (warm JIT by then)
        else if (c > 7 && c % REPROBE == 0) cpu = !useCpu;       // re-probe the path not in use
        else cpu = useCpu;
        long t0 = System.nanoTime();
        if (cpu) w.matmulParallelCpu(in, out, rows, cols);
        else w.matmulParallel(in, out, rows, cols);
        double ms = (System.nanoTime() - t0) / 1e6;
        if (cpu) cpuMs = cpuMs < 0 ? ms : (c == 7 ? Math.min(cpuMs, ms) : 0.5 * cpuMs + 0.5 * ms);
        else gpuMs = gpuMs < 0 ? ms : 0.8 * gpuMs + 0.2 * ms;
        if (c >= 7 && gpuMs > 0 && cpuMs > 0) {
            boolean before = useCpu;
            if (!useCpu && cpuMs < MARGIN * gpuMs) useCpu = true;
            else if (useCpu && gpuMs < MARGIN * cpuMs) useCpu = false;
            if (before != useCpu && (VERBOSE || c <= 7)) {
                System.err.printf("  %s projection on the %s (CPU %.1f ms, GPU %.1f ms)%n",
                    name, useCpu ? "CPU twin" : "GPU", cpuMs, gpuMs);
            }
        }
    }
}
