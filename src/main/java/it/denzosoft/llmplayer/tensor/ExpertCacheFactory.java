package it.denzosoft.llmplayer.tensor;

import java.nio.file.Path;

/**
 * Reflective loader for the java21 {@link ExpertCache} implementation, following the same pattern as
 * {@code VectorOpsFactory} and {@code TensorDataFactory}: base code never imports the java21 class,
 * so a Java 8 build simply gets no cache and keeps the mmap path.
 *
 * The cache is for MoE models streamed from SSD. It is created only when the model was loaded
 * lazily because it exceeds RAM ({@code mmap.advise=random}), unless forced.
 */
public final class ExpertCacheFactory {

    private ExpertCacheFactory() {}

    /**
     * Total cache budget in bytes, from {@code -Dmoe.expert.cache.mb} or a share of physical RAM.
     *
     * The budget is not free: these slots are off-heap and come out of the same RAM the kernel uses
     * for the page cache. Measured on the reference box (7.8 GB RAM, 2 GB heap, Qwen3-Coder-30B), a
     * 3 GB budget bought the best hit rate of the sweep — 57.3 % against 48.4 % at 2 GB — and still
     * ran four times slower end to end, because heap plus cache left the page cache nothing to work
     * with. More cache is not monotonically better; see docs/optimization/ssd-streaming-cache.md.
     */
    private static long resolveBudgetBytes() {
        String explicit = System.getProperty("moe.expert.cache.mb");
        if (explicit != null) {
            try {
                long bytes = Math.max(0L, Long.parseLong(explicit.trim())) * 1024L * 1024L;
                warnIfOversized(bytes);
                return bytes;
            } catch (NumberFormatException ignore) {
                // fall through to the default
            }
        }
        // Default: a quarter of physical RAM, capped at 4 GB. The cache displaces page cache rather
        // than competing with it — the pages it holds are ones the kernel was thrashing anyway.
        long ram = physicalMemoryBytes();
        if (ram <= 0) return 0L;
        return Math.min(ram / 4, 4L * 1024 * 1024 * 1024);
    }

    /**
     * An explicit budget is honoured, but flag the case where it plus the JVM heap leaves the page
     * cache too little to work with — that configuration measures slower despite a better hit rate.
     */
    private static void warnIfOversized(long budgetBytes) {
        long ram = physicalMemoryBytes();
        if (ram <= 0) return;
        long heap = Runtime.getRuntime().maxMemory();
        if (budgetBytes + heap > (long) (ram * 0.6)) {
            System.out.println("  Expert RAM cache: WARNING — " + (budgetBytes / (1024 * 1024))
                + " MB cache plus " + (heap / (1024 * 1024)) + " MB heap is a large share of "
                + (ram / (1024 * 1024)) + " MB RAM. This starves the page cache and usually measures "
                + "slower despite a higher hit rate. Consider a smaller --expert-cache-size.");
        }
    }

    private static long physicalMemoryBytes() {
        try {
            Object os = java.lang.management.ManagementFactory.getOperatingSystemMXBean();
            java.lang.reflect.Method m = Class.forName("com.sun.management.OperatingSystemMXBean")
                .getMethod("getTotalPhysicalMemorySize");
            return (Long) m.invoke(os);
        } catch (Throwable ignore) {
            return -1L;
        }
    }

    private static boolean enabled() {
        String v = System.getProperty("moe.expert.cache", "auto");
        if ("true".equals(v)) return true;
        if ("false".equals(v)) return false;
        // auto: only for the lazy >RAM MoE load, where the experts actually live on disk.
        return "random".equals(System.getProperty("mmap.advise", "none"));
    }

    /**
     * Build a cache for the given model file, or null when it is disabled, unavailable (Java 8), or
     * too small to be useful.
     *
     * @param maxBytesPerSlice byte size of the largest single expert slice across all MoE layers,
     *                         summed over the three projections
     */
    public static ExpertCache createIfEnabled(Path modelPath, TensorData mappedFile,
                                              long maxBytesPerSlice) {
        if (!enabled() || modelPath == null || mappedFile == null || maxBytesPerSlice <= 0) return null;
        long budget = resolveBudgetBytes();
        int slots = (int) Math.min(budget / maxBytesPerSlice, Integer.MAX_VALUE);
        if (slots < 8) {
            if (budget > 0) {
                System.out.println("  Expert RAM cache: budget " + (budget / (1024 * 1024))
                    + " MB fits only " + slots + " experts — not enabled (raise -Dmoe.expert.cache.mb)");
            }
            return null;
        }
        try {
            Class<?> impl = Class.forName("it.denzosoft.llmplayer.tensor.MappedExpertCache");
            return (ExpertCache) impl
                .getConstructor(Path.class, TensorData.class, int.class, long.class)
                .newInstance(modelPath, mappedFile, slots, maxBytesPerSlice);
        } catch (ClassNotFoundException e) {
            return null; // java21 classes not available
        } catch (Throwable t) {
            Throwable c = (t instanceof java.lang.reflect.InvocationTargetException && t.getCause() != null)
                ? t.getCause() : t;
            System.out.println("  Expert RAM cache: unavailable (" + c + ")");
            return null;
        }
    }
}
