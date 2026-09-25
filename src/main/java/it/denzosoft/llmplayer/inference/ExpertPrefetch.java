package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.GGMLType;

/**
 * Expert-granular read-ahead for MoE models streamed from SSD (larger than physical RAM).
 *
 * When {@code LLMEngine.load} skips the preload for a model above 85 % of RAM, the routed experts
 * stay on disk and are faulted in on demand. That path sets MADV_RANDOM over the whole mapping,
 * which switches kernel read-ahead off — correct for the access pattern *between* experts, wrong
 * for the pattern *inside* one: a routed expert is a single contiguous multi-megabyte slice.
 * Without read-ahead, faulting one costs hundreds of independent page faults.
 *
 * Read bandwidth against request size on the reference box (vboxsf volume, random offsets,
 * in-process pread) shows how much this costs:
 *
 * <pre>
 *   4 KB granularity (page fault)   15.9 MB/s
 *   2.5 MB granularity (one expert) 481.7 MB/s     ~30x
 * </pre>
 *
 * So immediately after the router picks top-K, this issues one MADV_WILLNEED per selected expert
 * slice. The hint is asynchronous, so queueing all of them before the expert loop lets the kernel
 * run the reads concurrently and overlaps them with compute instead of serialising fault by fault.
 *
 * <b>Measured effect: small, and capped by the filesystem.</b> On Qwen3-Coder-30B (18.6 GB against
 * 7.8 GB of RAM) this is worth about 1.15x — 93.9 s vs 107.9 s for five tokens — which is below the
 * noise threshold of this box. A mincore() probe explains why: MADV_WILLNEED over a 2.65 MB expert
 * range returns success but leaves only 32 of 647 pages resident after 1.5 s, because vboxsf caps
 * read-ahead near 128 KB. An explicit read of the same range takes 21.7 ms. Reaching the 30x above
 * therefore requires an explicit read path rather than an mmap hint — see the L1 design in
 * {@code docs/optimization/ssd-streaming-cache.md}. Kept enabled because it is never negative and
 * should do considerably better on storage whose read-ahead honours the hint.
 *
 * Controlled by {@code -Dmoe.expert.willneed}: {@code auto} (default) enables it exactly when the
 * MoE lazy >RAM path is active ({@code mmap.advise=random}), {@code true} / {@code false} force it.
 * It is a hint only — it never changes results, and a platform without madvise silently ignores it.
 */
public final class ExpertPrefetch {

    /**
     * Resolved once, on first use. That happens during the first MoE forward pass, which is well
     * after {@code LLMEngine.load} has set {@code mmap.advise}, so the auto mode sees the final
     * value.
     */
    private static final boolean ENABLED = resolveEnabled();

    private ExpertPrefetch() {}

    private static boolean resolveEnabled() {
        String v = System.getProperty("moe.expert.willneed", "auto");
        if ("true".equals(v)) return true;
        if ("false".equals(v)) return false;
        // auto: only for the lazy >RAM MoE load, where the experts actually live on disk.
        return "random".equals(System.getProperty("mmap.advise", "none"));
    }

    public static boolean isEnabled() {
        return ENABLED;
    }

    /**
     * Queue a read-ahead for the gate/up/down slices of every selected expert.
     *
     * @param elementsPerSlice element count of one expert's 2D slice, i.e. {@code expertFfnDim * dim}
     *                         (the same for all three projections)
     */
    public static void willNeed(FloatTensor gate, FloatTensor up, FloatTensor down,
                                int[] selectedExperts, int usedCount, long elementsPerSlice) {
        if (!ENABLED || elementsPerSlice <= 0) return;
        for (int k = 0; k < usedCount; k++) {
            int expert = selectedExperts[k];
            if (expert < 0) continue; // backfilled slot from a NaN router row
            adviseSlice(gate, expert, elementsPerSlice);
            adviseSlice(up, expert, elementsPerSlice);
            adviseSlice(down, expert, elementsPerSlice);
        }
    }

    /**
     * Byte range of one expert inside a 3D {@code ffn_*_exps} tensor. Mirrors the offset arithmetic
     * of {@code MoEFFN.expertMatmul} and {@code ExpertGpuCache.uploadExpertSlice}.
     */
    private static void adviseSlice(FloatTensor tensor, int expert, long elementsPerSlice) {
        if (tensor == null) return;
        GGMLType type = tensor.type();
        if (type == null) return;
        int blockSize = type.getBlockSize();
        if (blockSize <= 0) return;
        long blocksPerSlice = elementsPerSlice / blockSize;
        if (blocksPerSlice <= 0) return;
        long blockBytes = type.getTypeSize();
        long byteOffset = (expert * elementsPerSlice / blockSize) * blockBytes;
        tensor.data().adviseWillNeed(byteOffset, blocksPerSlice * blockBytes);
    }
}
