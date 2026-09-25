package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.GGMLType;
import it.denzosoft.llmplayer.tensor.MatmulPool;
import it.denzosoft.llmplayer.tensor.TensorFactory;

/**
 * Per-expert views over the 3D routed-expert tensors of a MoE model, for batched prefill on the CPU.
 *
 * <p>A view is a standalone tensor over one expert's slice of {@code ffn_{gate,up,down}_exps}
 * ({@link TensorFactory#create} over {@code TensorData.slice}, no copy), so its rows start at 0 —
 * the same shape as a slice returned by the SSD-streaming {@code ExpertCache}. Batched prefill runs
 * each expert once per chunk over all the tokens routed to it with {@code matmulRowsBatch} on its
 * view (or on its cached slice), instead of once per (token, slot).
 *
 * <p>One-token decode keeps the per-row {@code dot} loop over the 3D tensor or the cached slice.
 * Running decode through {@code matmulRows} on these views was measured slower on Qwen3-Coder-30B
 * streamed from SSD (0.3-0.7 against 0.8-1.1 tok/s), so it was not kept.
 *
 * <p>Views are created on first use and kept; {@link #ensure} must run on one thread, outside the
 * parallel expert loop, because TensorFactory's GPU toggle is a static. {@code -Dmoe.expert.rows=false}
 * disables the batched MoE prefill (the layer-outer prefill with one-token matmuls is used instead).
 */
final class ExpertViews {

    private static final boolean ENABLED =
        !"false".equalsIgnoreCase(System.getProperty("moe.expert.rows", "true"));

    /** [layer][projection][expert], projections in {@code ExpertCache.PROJ_*} order. */
    private final FloatTensor[][][] views;
    private final int expertCount;

    ExpertViews(int layers, int expertCount) {
        this.views = new FloatTensor[layers][3][];
        this.expertCount = expertCount;
    }

    /** Whether batched MoE prefill may use the views: on by default, CPU matmul pool only. */
    static boolean active() {
        return ENABLED && MatmulPool.enabled();
    }

    /** Create the missing views of the given experts of one layer (negative ids are skipped). */
    void ensure(int layer, FloatTensor gate, FloatTensor up, FloatTensor down, int[] experts, int count,
                long elementsPerSlice) {
        FloatTensor[] projections = { gate, up, down };
        for (int p = 0; p < 3; p++) {
            if (views[layer][p] == null) views[layer][p] = new FloatTensor[expertCount];
            FloatTensor[] row = views[layer][p];
            FloatTensor w = projections[p];
            GGMLType type = w.type();
            long bytes = elementsPerSlice / type.getBlockSize() * type.getTypeSize();
            for (int i = 0; i < count; i++) {
                int e = experts[i];
                if (e < 0 || row[e] != null) continue;
                Object savedGpu = TensorFactory.getGpuBufferManager();
                try {
                    TensorFactory.setGpuBufferManager(null); // CPU path: want CPU/SIMD tensors
                    row[e] = TensorFactory.create(type, w.data().slice(e * bytes, bytes), elementsPerSlice);
                } finally {
                    TensorFactory.setGpuBufferManager(savedGpu);
                }
            }
        }
    }

    /** The view of one expert and projection, or null if {@link #ensure} has not created it. */
    FloatTensor get(int layer, int projection, int expert) {
        FloatTensor[] row = views[layer][projection];
        return row != null ? row[expert] : null;
    }

    // ==================== Running the experts of one layer ====================

    /** Experts prepared per SSD-cache call during batched prefill (a group must fit the cache at once). */
    private static final int CACHE_GROUP = Integer.getInteger("moe.prefill.cache.group", 16);

    /** Read the next group of experts from disk while the current one is computed. */
    private static final boolean OVERLAP =
        !"false".equalsIgnoreCase(System.getProperty("moe.prefill.overlap", "true"));

    /**
     * Prefill chunk when the routed experts stream from disk through the SSD cache. Each chunk reads
     * every expert it needs at most once per layer, so the bytes read fall in proportion to the
     * number of chunks: on Qwen3-Coder-30B with a 290-token prompt, 47.4 GB at 64 tokens per chunk,
     * 31.8 GB at 128 and 16.2 GB at 320. Models that fit RAM keep {@code -Dprefill.batch} (64),
     * which is sized for the CPU caches of the multi-token kernels rather than for disk.
     */
    private static final int STREAM_CHUNK = Integer.getInteger("moe.prefill.stream.batch", 256);

    /** Tokens per batched-prefill chunk: {@link #STREAM_CHUNK} with an SSD cache, else {@code defaultChunk}. */
    static int prefillChunk(it.denzosoft.llmplayer.tensor.ExpertCache cache, int defaultChunk) {
        return Math.max(1, cache != null ? STREAM_CHUNK : defaultChunk);
    }

    /** Single daemon thread that runs {@code ExpertCache.prepare} for the next group. */
    private static volatile java.util.concurrent.ExecutorService io;

    /** The work for one routed expert, given its three projection tensors (rows start at 0). */
    interface ExpertTask {
        void run(int expert, FloatTensor gate, FloatTensor up, FloatTensor down);
    }

    /**
     * Run {@code task} for each of the {@code nUsed} experts in {@code used}, in parallel on the
     * matmul pool, with each expert's projections taken from the SSD cache when it is resident and
     * from its view otherwise.
     *
     * <p>Without a cache all experts run as one parallel batch. With a cache they run in groups of
     * {@code -Dmoe.prefill.cache.group} (16) experts, each group prepared (its misses read from disk)
     * before it is computed. The next group's prepare runs on a background thread while the current
     * group is computed, so disk and CPU overlap; the cache keeps the previous prepare's slots
     * resident for exactly this. The first group of a layer cannot be read ahead, because a layer's
     * experts are known only after its router has run. {@code -Dmoe.prefill.overlap=false} prepares
     * every group synchronously.
     */
    void forEachExpert(it.denzosoft.llmplayer.tensor.ExpertCache cache, int layer, int[] used, int nUsed,
                       FloatTensor gate, FloatTensor up, FloatTensor down, long elementsPerSlice,
                       ExpertTask task) {
        ensure(layer, gate, up, down, used, nUsed, elementsPerSlice);
        int group = cache != null ? Math.max(1, CACHE_GROUP) : Math.max(1, nUsed);
        FloatTensor[][] current = prepareGroup(cache, layer, used, 0, Math.min(group, nUsed),
            gate, up, down, elementsPerSlice);
        for (int g0 = 0; g0 < nUsed; g0 += group) {
            int g = Math.min(group, nUsed - g0);
            int n0 = g0 + group;
            java.util.concurrent.Future<FloatTensor[][]> next = null;
            if (n0 < nUsed && cache != null && OVERLAP) {
                final int nextCount = Math.min(group, nUsed - n0);
                next = ioThread().submit(() ->
                    prepareGroup(cache, layer, used, n0, nextCount, gate, up, down, elementsPerSlice));
            }
            final FloatTensor[][] t = current;
            final int base = g0;
            MatmulPool.forEach(g, i -> task.run(used[base + i], t[i][0], t[i][1], t[i][2]));
            if (n0 < nUsed) {
                current = next != null ? await(next)
                    : prepareGroup(cache, layer, used, n0, Math.min(group, nUsed - n0), gate, up, down, elementsPerSlice);
            }
        }
    }

    /**
     * Prepare {@code count} experts starting at {@code used[from]} in the cache (falling back to the
     * mmap read-ahead hint) and resolve their three projection tensors: the cached slice when
     * resident, the view otherwise.
     */
    private FloatTensor[][] prepareGroup(it.denzosoft.llmplayer.tensor.ExpertCache cache, int layer,
                                         int[] used, int from, int count, FloatTensor gate, FloatTensor up,
                                         FloatTensor down, long elementsPerSlice) {
        int[] ids = java.util.Arrays.copyOfRange(used, from, from + count);
        boolean cached = cache != null && cache.prepare(layer, ids, count, gate, up, down, elementsPerSlice);
        if (!cached) ExpertPrefetch.willNeed(gate, up, down, ids, count, elementsPerSlice);
        FloatTensor[][] t = new FloatTensor[count][3];
        for (int i = 0; i < count; i++) {
            for (int p = 0; p < 3; p++) {
                FloatTensor c = cached ? cache.tensorFor(layer, ids[i], p) : null;
                t[i][p] = c != null ? c : get(layer, p, ids[i]);
            }
        }
        return t;
    }

    private static FloatTensor[][] await(java.util.concurrent.Future<FloatTensor[][]> f) {
        try {
            return f.get();
        } catch (java.util.concurrent.ExecutionException e) {
            Throwable c = e.getCause();
            if (c instanceof RuntimeException) throw (RuntimeException) c;
            throw new RuntimeException(c);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        }
    }

    private static java.util.concurrent.ExecutorService ioThread() {
        java.util.concurrent.ExecutorService e = io;
        if (e == null) {
            synchronized (ExpertViews.class) {
                e = io;
                if (e == null) {
                    e = java.util.concurrent.Executors.newSingleThreadExecutor(r -> {
                        Thread th = new Thread(r, "moe-expert-io");
                        th.setDaemon(true);
                        return th;
                    });
                    io = e;
                }
            }
        }
        return e;
    }

    /**
     * Group the {@code slots} (token, slot) pairs of a chunk by expert with a counting sort:
     * afterwards the pairs routed to expert {@code e} are {@code grouped[start[e] .. start[e+1])},
     * in ascending pair order, and {@code used[0 .. return value)} lists the distinct experts in
     * ascending id order. Pairs with a negative expert are left out.
     */
    static int groupByExpert(int[] sel, int slots, int expertCount, int[] start, int[] grouped, int[] used) {
        java.util.Arrays.fill(start, 0, expertCount + 1, 0);
        for (int s = 0; s < slots; s++) {
            if (sel[s] >= 0) start[sel[s] + 1]++;
        }
        int nUsed = 0;
        for (int e = 0; e < expertCount; e++) {
            if (start[e + 1] > 0) used[nUsed++] = e;
            start[e + 1] += start[e];
        }
        int[] fill = java.util.Arrays.copyOf(start, expertCount);
        for (int s = 0; s < slots; s++) {
            if (sel[s] >= 0) grouped[fill[sel[s]]++] = s;
        }
        return nUsed;
    }
}
