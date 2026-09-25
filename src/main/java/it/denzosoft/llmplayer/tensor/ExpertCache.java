package it.denzosoft.llmplayer.tensor;

/**
 * RAM cache of MoE routed-expert weight slices, filled by explicit positional reads from the model
 * file (SSD streaming, lever L1 in {@code docs/optimization/ssd-streaming-cache.md}).
 *
 * The point is read granularity. When a model exceeds RAM, {@code LLMEngine.load} skips the preload
 * and the routed experts stay on disk, faulted in 4 KB at a time. Measured on the reference box,
 * that costs ~15.9 MB/s, while reading the same bytes at expert granularity reaches ~116-480 MB/s.
 * An mmap hint cannot close that gap — {@code MADV_WILLNEED} over a 2.65 MB range leaves only 32 of
 * 647 pages resident — so this cache owns its memory and reads into it explicitly.
 *
 * Usage per MoE layer, from the inference engine:
 * <ol>
 *   <li>{@link #prepare} once, right after the router picks top-K — resolves slots and reads any
 *       misses, in parallel.</li>
 *   <li>{@link #tensorFor} per expert and projection inside the compute loop — a pure lookup that
 *       returns a {@link FloatTensor} over the cached slice, whose element offsets are
 *       slice-relative (row {@code r} starts at {@code r * inDim}, with no per-expert base).</li>
 * </ol>
 *
 * Implementations live in {@code src/main/java21} and are loaded reflectively by
 * {@link ExpertCacheFactory}, so the Java 8 build degrades to the mmap path.
 */
public interface ExpertCache {

    int PROJ_GATE = 0;
    int PROJ_UP = 1;
    int PROJ_DOWN = 2;

    /**
     * Make the given experts of one layer resident, reading any that are missing. Called once per
     * MoE layer before the expert compute loop.
     *
     * @param selectedExperts top-K expert indices; negative entries are skipped
     * @param usedCount       number of valid entries in {@code selectedExperts}
     * @return true if every requested expert is now resident, so {@link #tensorFor} will answer for
     *         all of them; false if the caller should fall back to the mmap path for this layer
     */
    boolean prepare(int layer, int[] selectedExperts, int usedCount,
                    FloatTensor gateExps, FloatTensor upExps, FloatTensor downExps,
                    long elementsPerSlice);

    /**
     * The cached slice for one expert and projection, or null when it is not resident. Only valid
     * for experts passed to the most recent {@link #prepare} for the same layer. A slice obtained
     * after a prepare() stays valid (its slot is not refilled) through the next prepare() call, so a
     * caller may compute one group of experts while preparing the next; it must not call this
     * method concurrently with prepare().
     */
    FloatTensor tensorFor(int layer, int expert, int projection);

    /** Human-readable hit rate and I/O totals, for the shutdown summary. */
    String stats();

    // --- Counters, surfaced through LLMPlayerMetrics (JMX and /api/metrics) ---

    /** Expert lookups served from RAM. */
    long hits();

    /** Expert lookups that required a read from the model file. */
    long misses();

    /** Total bytes read from the model file to fill slots. */
    long bytesRead();

    /** Wall-clock nanoseconds spent inside those reads. */
    long readNanos();

    /** Number of expert slots allocated. */
    int slotCount();

    /** Bytes of one slot, i.e. the largest expert across all MoE layers. */
    long slotBytes();

    /** Release the off-heap slots and the file handle. */
    void close();
}
