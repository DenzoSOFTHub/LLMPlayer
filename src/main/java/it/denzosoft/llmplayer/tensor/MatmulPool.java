package it.denzosoft.llmplayer.tensor;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;

/**
 * Persistent worker pool for the CPU matmul hot path, with dynamic (work-stealing style) row
 * scheduling. Java 8 compatible: platform threads, an atomic chunk counter and
 * {@link LockSupport} parking — no virtual threads, no ForkJoinPool.
 *
 * <h2>Why it exists</h2>
 * The previous dispatchers split every matmul into exactly {@code availableProcessors()} static
 * chunks and submitted each as a new virtual-thread task. That has three costs that this class
 * removes:
 * <ul>
 *   <li><b>Static partitioning.</b> A matmul finishes when its slowest chunk does. On a hybrid CPU
 *       (P-cores + E-cores), in a VM whose vCPUs are time-shared, or on a machine where another
 *       process holds a core, one slow chunk stalls every other thread. Here the rows are cut into
 *       several chunks per thread and claimed from a shared counter, so fast cores simply take
 *       more of them.</li>
 *   <li><b>Per-call task creation.</b> A decode step issues roughly 4 matmuls per layer plus the
 *       output projection; each used to allocate futures and start virtual threads. Workers here
 *       live for the whole process and spin briefly between dispatches before parking.</li>
 *   <li><b>Ignored thread count.</b> The virtual-thread path always used
 *       {@code availableProcessors()}, so {@code --threads} (and the physical-core default) had no
 *       effect on it. The pool size comes from {@code -Dmatmul.threads}, which the CLI sets from
 *       {@code --threads}.</li>
 * </ul>
 *
 * <h2>Concurrency contract</h2>
 * One dispatch runs at a time. A call made while the pool is busy (a second generation thread, or
 * a matmul nested inside another parallel region) runs inline on the calling thread, so the pool
 * can never deadlock and callers never need to know whether they are nested.
 *
 * <p>Disable with {@code -Dmatmul.pool=false} to return to the previous dispatchers.
 */
public final class MatmulPool {

    /** Work item: process units {@code [from, to)} of the current job. */
    public interface RangeTask {
        void run(int from, int to);
    }

    /** Target number of chunks each thread gets per dispatch; more gives better balance, fewer less overhead. */
    private static final int CHUNKS_PER_THREAD = Integer.getInteger("matmul.chunks.per.thread", 4);
    /** Spin iterations a worker waits for the next dispatch before parking (a decode step dispatches back to back). */
    private static final int SPIN_LIMIT = Integer.getInteger("matmul.spin", 20000);

    /** {@code Thread.onSpinWait} (Java 9+), resolved reflectively so this class stays Java 8 compatible. */
    private static final java.lang.invoke.MethodHandle ON_SPIN_WAIT = resolveSpinHint();

    private static volatile MatmulPool instance;

    private final int threads;
    private final Thread[] workers;
    private final AtomicBoolean busy = new AtomicBoolean(false);

    /**
     * The current job, published through this volatile field. Each job carries its own counters,
     * so a worker that wakes late and picks up an already-finished job finds no chunk left to
     * claim instead of mixing fields from two jobs.
     */
    private volatile Job job;

    private static final class Job {
        final RangeTask task;
        final int units;
        final int chunkSize;
        final int chunkCount;
        final AtomicInteger nextChunk = new AtomicInteger();
        final AtomicInteger remainingChunks;
        volatile Throwable failure;

        Job(RangeTask task, int units, int chunkSize, int chunkCount) {
            this.task = task;
            this.units = units;
            this.chunkSize = chunkSize;
            this.chunkCount = chunkCount;
            this.remainingChunks = new AtomicInteger(chunkCount);
        }

        void runChunks() {
            int c;
            while ((c = nextChunk.getAndIncrement()) < chunkCount) {
                int from = c * chunkSize;
                try {
                    task.run(from, Math.min(from + chunkSize, units));
                } catch (Throwable e) {
                    failure = e;
                }
                remainingChunks.decrementAndGet();
            }
        }
    }

    private MatmulPool(int threads) {
        this.threads = threads;
        this.workers = new Thread[threads - 1];
        for (int i = 0; i < workers.length; i++) {
            Thread t = new Thread(this::workerLoop, "matmul-" + (i + 1));
            t.setDaemon(true);
            workers[i] = t;
            t.start();
        }
    }

    /**
     * The shared pool, or {@code null} when disabled. With one configured thread the pool has no
     * workers and every dispatch runs on the caller, which keeps {@code --threads 1} honest.
     */
    public static MatmulPool get() {
        MatmulPool p = instance;
        if (p != null) return p;
        synchronized (MatmulPool.class) {
            if (instance == null && enabled()) {
                instance = new MatmulPool(Math.max(1, configuredThreads()));
            }
            return instance;
        }
    }

    private static volatile boolean disabled = !"true".equals(System.getProperty("matmul.pool", "true"));

    /** Whether the pool may be used at all. */
    public static boolean enabled() { return !disabled; }

    /**
     * Stop routing matmuls through the pool. Called when a GPU backend is initialised, together
     * with {@link FloatTensor#disableVirtualThreadMatmul()}, so GPU runs keep the dispatch they
     * were validated with. {@code -Dmatmul.pool=force} keeps the pool on regardless.
     */
    public static void disable() {
        if (!"force".equals(System.getProperty("matmul.pool"))) disabled = true;
    }

    /** Thread count: {@code -Dmatmul.threads}, else the ForkJoin parallelism set by the CLI, else all logical CPUs. */
    public static int configuredThreads() {
        Integer n = Integer.getInteger("matmul.threads");
        if (n == null) n = Integer.getInteger("java.util.concurrent.ForkJoinPool.common.parallelism");
        if (n == null || n <= 0) n = Runtime.getRuntime().availableProcessors();
        return n;
    }

    public int threads() { return threads; }

    /**
     * Parallel loop over {@code [0, n)} on the shared pool (one index per chunk), falling back to
     * a parallel stream when the pool is disabled. Used for the per-head attention and per-expert
     * MoE loops so they share the matmul workers instead of competing with them from the
     * ForkJoin common pool.
     */
    public static void forEach(int n, java.util.function.IntConsumer body) {
        MatmulPool pool = enabled() ? get() : null;
        if (pool == null) {
            java.util.stream.IntStream.range(0, n).parallel().forEach(body);
            return;
        }
        pool.parallelFor(n, 1, (from, to) -> {
            for (int i = from; i < to; i++) body.accept(i);
        });
    }

    /**
     * Runs {@code task} over {@code [0, units)} split into chunks of at least {@code minChunk} units,
     * using every worker plus the calling thread, and returns once all units are done.
     */
    public void parallelFor(int units, int minChunk, RangeTask task) {
        if (units <= 0) return;
        int target = threads * CHUNKS_PER_THREAD;
        int size = Math.max(Math.max(1, minChunk), (units + target - 1) / target);
        int count = (units + size - 1) / size;
        if (count <= 1 || workers.length == 0 || !busy.compareAndSet(false, true)) {
            task.run(0, units);
            return;
        }
        try {
            Job j = new Job(task, units, size, count);
            job = j;                    // volatile write publishes the job
            for (Thread w : workers) LockSupport.unpark(w);

            j.runChunks();
            // Wait for chunks still being processed by workers.
            int spins = 0;
            while (j.remainingChunks.get() > 0) {
                if (++spins < SPIN_LIMIT) {
                    spinHint();
                } else {
                    Thread.yield();
                }
            }
            Throwable f = j.failure;
            if (f != null) {
                if (f instanceof RuntimeException) throw (RuntimeException) f;
                if (f instanceof Error) throw (Error) f;
                throw new RuntimeException(f);
            }
        } finally {
            busy.set(false);
        }
    }

    private static java.lang.invoke.MethodHandle resolveSpinHint() {
        try {
            return java.lang.invoke.MethodHandles.lookup().findStatic(Thread.class, "onSpinWait",
                java.lang.invoke.MethodType.methodType(void.class));
        } catch (Throwable e) {
            return null; // Java 8: plain busy loop
        }
    }

    private static void spinHint() {
        if (ON_SPIN_WAIT != null) {
            try {
                ON_SPIN_WAIT.invokeExact();
            } catch (Throwable ignore) {
                // onSpinWait cannot throw
            }
        }
    }

    private void workerLoop() {
        Job seen = null;
        while (true) {
            int spins = 0;
            Job j;
            while ((j = job) == seen) {
                if (++spins < SPIN_LIMIT) {
                    spinHint();
                } else {
                    LockSupport.park(this);
                    spins = 0;
                }
            }
            seen = j;
            j.runChunks();
        }
    }
}
