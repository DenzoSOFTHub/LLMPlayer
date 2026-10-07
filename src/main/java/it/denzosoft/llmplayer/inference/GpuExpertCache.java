package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.GGMLType;

/**
 * GPU LRU cache of MoE expert slices (ExpertGpuCache in the java21 source root), called through
 * this interface so the per-layer hot path has no reflection. {@link #create} sizes and builds it.
 */
public interface GpuExpertCache {

    /** Routed experts of one layer on the GPU; writes each expert's output to outPerExpert[k]. */
    void computeExperts(FloatTensor gateExps, FloatTensor upExps, FloatTensor downExps,
                        float[] input, int[] selectedExperts, float[] selectedWeights,
                        int expertUsedCount, int layer, int dim, int expertFfnDim,
                        float[][] gatePerExpert, float[][] upPerExpert, float[][] outPerExpert,
                        boolean useSwigluOai,
                        FloatTensor gateExpsBias, FloatTensor upExpsBias, FloatTensor downExpsBias);

    /**
     * Hybrid step 1: queue, asynchronously, the selected experts that are resident on the GPU and
     * return a bitmask of their slots {@code k}. The caller computes the other slots on the CPU
     * meanwhile, then calls {@link #finishResident}. A missing expert is never uploaded here.
     */
    int launchResident(FloatTensor gateExps, FloatTensor upExps, FloatTensor downExps, float[] input,
                       int[] selectedExperts, int expertUsedCount, int layer, int dim, int expertFfnDim,
                       boolean useSwigluOai,
                       FloatTensor gateExpsBias, FloatTensor upExpsBias, FloatTensor downExpsBias);

    /**
     * Hybrid step 2: wait for the experts of {@code mask}, write their outputs to
     * {@code outPerExpert[k]}, then promote at most one frequently routed missing expert.
     */
    void finishResident(int mask, float[][] outPerExpert);

    /** Whether the engines use the hybrid split ({@code -Dmoe.expert.cache.hybrid}, default on). */
    static boolean hybrid() {
        return !"false".equals(System.getProperty("moe.expert.cache.hybrid", "true"));
    }

    String getStats();

    /** Routed selections served by resident experts. */
    default long hits() { return 0; }

    /** Routed selections computed on the CPU (or uploaded, on the non-hybrid path). */
    default long misses() { return 0; }

    /** Experts (layer, expert pairs) currently resident. */
    default int residentExperts() { return 0; }

    /** Experts the cache can hold. */
    default int capacityExperts() { return 0; }

    /** Device memory the cache allocated for expert weights, in bytes. */
    default long deviceBytes() { return 0; }

    /** Free the device memory (idempotent); writes the routing profile when one is set. */
    default void close() { }

    /**
     * The engine starts decoding position {@code position} (called per layer; only a change
     * counts): the per-token promotion budget restarts.
     */
    default void noteToken(int position) { }

    /**
     * Batched prefill: queue on the GPU every resident expert among {@code used[0..nUsed)} over all
     * the chunk's tokens routed to it (slot {@code s} is token {@code s / k}; the slots of expert
     * {@code e} are {@code groupedSlots[groupStart[e] .. groupStart[e+1])}), set {@code onGpu[e]}
     * for them and return how many were queued. The caller computes the others on the CPU
     * meanwhile, then calls {@link #finishResidentBatch}. Returns 0 when not supported.
     */
    default int launchResidentBatch(int layer, FloatTensor gateExps, FloatTensor upExps, FloatTensor downExps,
                                    float[][] xn, int nTokens, int[] used, int nUsed, int[] groupStart,
                                    int[] groupedSlots, int k, boolean[] onGpu, boolean useSwigluOai) {
        return 0;
    }

    /** Wait for the experts of the last {@link #launchResidentBatch} and write their outputs to {@code out[slot]}. */
    default void finishResidentBatch(float[][] out) { }

    /** Count a routing decision the cache did not see through {@link #launchResident} (batched prefill). */
    default void noteRouting(int layer, int[] selectedExperts, int count) { }

    /** The expert tensors ([layer][gate, up, down]) the cache promotes from. */
    default void bindExperts(FloatTensor[][] experts) { }

    /**
     * Routing profile of this model: counts from an earlier run are loaded (and the most routed
     * experts uploaded) now, and the counts are written back at {@link #close()}.
     */
    default void setRoutingProfile(java.nio.file.Path file) { }

    /** Whether ExpertGpuCache has an FP32-input matmul kernel for this expert quant type. */
    static boolean kernelExists(GGMLType t) {
        switch (t) {
            case MXFP4: case Q4_K: case Q5_K: case Q6_K: case Q3_K: case Q8_0: case Q4_0:
            case Q5_0: case Q5_1: case IQ4_NL: case IQ4_XS: case F16:
            case IQ1_M: case IQ2_XXS: case Q2_K: return true;
            default: return false;
        }
    }

    /**
     * Build a cache for the given expert tensors ([layer][gate, up, down]; null rows skipped) within
     * {@code maxCacheBytes} of device memory, or return null (unsupported type, too little VRAM,
     * java21 missing). A cache unit holds one expert's three slices; each slot is sized for the
     * largest slice of its projection over all layers (not the largest of all three: mixed-quant
     * GGUFs ship a larger type for {@code ffn_down_exps} in some layers), and units are packed into
     * a few large allocations so the driver's 2 MiB page rounding is paid per chunk, not per slot.
     * K-quant experts can be excluded with {@code -Dmoe.expert.cache.experimental=false} (MXFP4
     * always allowed).
     */
    static GpuExpertCache create(Object cudaContext, long maxCacheBytes, FloatTensor[][] experts,
                                 int expertFfnDim, int dim, int expertCount) {
        boolean kQuant = !"false".equals(System.getProperty("moe.expert.cache.experimental", "true"));
        long elementsPerSlice = (long) expertFfnDim * dim;
        long[] proj = new long[3];
        GGMLType first = null;
        // Per-layer slot sizes (F2 step 4): a Q4_K_M model ships some layers' ffn_down_exps as Q6_K
        // and others as Q4_K, so one slot size for every layer (the largest) wastes the difference
        // on the smaller layers. Each distinct gate/up/down triple becomes a size class.
        java.util.List<long[]> classes = new java.util.ArrayList<>();
        int[] layerClass = new int[experts.length];
        java.util.Arrays.fill(layerClass, -1);
        for (int l = 0; l < experts.length; l++) {
            FloatTensor[] layer = experts[l];
            if (layer == null) continue;
            long[] triple = new long[3];
            for (int p = 0; p < layer.length && p < 3; p++) {
                FloatTensor t = layer[p];
                if (t == null) continue;
                GGMLType ty = t.type();
                if (first == null) first = ty;
                if (!kernelExists(ty) || (!kQuant && ty != GGMLType.MXFP4)) {
                    System.out.println("  Expert GPU cache: experts are " + ty + " — using CPU expert path");
                    return null;
                }
                triple[p] = ((elementsPerSlice / ty.getBlockSize()) * ty.getTypeSize() + 255) & ~255L; // 256-byte aligned
                proj[p] = Math.max(proj[p], triple[p]);
            }
            int c = -1;
            for (int i = 0; i < classes.size(); i++) if (java.util.Arrays.equals(classes.get(i), triple)) c = i;
            if (c < 0) { classes.add(triple); c = classes.size() - 1; }
            layerClass[l] = c;
        }
        if (proj[0] == 0 || proj[1] == 0 || proj[2] == 0) return null;
        if ("false".equals(System.getProperty("moe.expert.gpu.classes", "true")) || classes.size() > 8) {
            // one class sized for the largest slice of every projection, as before
            classes.clear();
            classes.add(proj.clone());
            for (int l = 0; l < layerClass.length; l++) if (layerClass[l] >= 0) layerClass[l] = 0;
        }
        long unitBytes = proj[0] + proj[1] + proj[2];
        if (maxCacheBytes / unitBytes < 4) {
            System.out.println("  Expert GPU cache: not enough VRAM (" + (maxCacheBytes / unitBytes) + " experts)");
            return null;
        }
        try {
            Class<?> cacheClass = Class.forName("it.denzosoft.llmplayer.gpu.ExpertGpuCache");
            Class<?> ctxClass = Class.forName("it.denzosoft.llmplayer.gpu.CudaContext");
            java.util.EnumSet<GGMLType> types = java.util.EnumSet.noneOf(GGMLType.class);
            for (FloatTensor[] layer : experts) {
                if (layer != null) for (FloatTensor t : layer) if (t != null) types.add(t.type());
            }
            GpuExpertCache cache = (GpuExpertCache) cacheClass.getConstructor(ctxClass, long.class, long[].class,
                    int.class, int.class, int.class, int.class, GGMLType[].class, long[][].class, int[].class)
                .newInstance(cudaContext, maxCacheBytes, proj, experts.length, expertCount, dim, expertFfnDim,
                    types.toArray(new GGMLType[0]), classes.toArray(new long[0][]), layerClass);
            cache.bindExperts(experts);
            if (cache.capacityExperts() < 4) {
                cache.close();
                System.out.println("  Expert GPU cache: allocation failed");
                return null;
            }
            System.out.println("  Expert GPU cache: " + first + " experts, " + cache.capacityExperts()
                + " resident at most — hot-expert cache over " + expertCount + " experts per layer");
            return cache;
        } catch (Throwable e) {
            System.out.println("  Expert GPU cache init failed: " + GpuFailureException.describe(e));
            return null;
        }
    }
}
