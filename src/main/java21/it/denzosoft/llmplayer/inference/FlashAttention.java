package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.gpu.CudaBindings;
import it.denzosoft.llmplayer.gpu.CudaBufferManager;
import it.denzosoft.llmplayer.gpu.CudaContext;
import it.denzosoft.llmplayer.gpu.KernelParams;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Single-token GPU attention shared by the GPU-resident forward passes.
 *
 * <p>Default path: flash-decoding ({@code attention_flash.cu}) — the sequence is split across
 * {@link #nSplits} blocks per query head, each warp streams one K/V row at a time with coalesced
 * loads and an online softmax, and a second kernel merges the partial results. Shared memory is
 * fixed (at most {@code 8 * headSize * 4} bytes), so there is no context-length ceiling and the
 * launch configuration is the same for every position, which is what CUDA graph capture needs.
 *
 * <p>Legacy path ({@code -Dcuda.attn.flash=false}, or headSize above 512): the original
 * {@code attention_full} kernel, one block per head with all scores in shared memory. Its dynamic
 * shared memory is raised above the 48 KB default with {@code cuFuncSetAttribute} when the device
 * allows it; {@link #maxLegacySeqLen()} reports the resulting ceiling.
 *
 * <p>Zero allocation per launch: the parameter blocks are pre-built and only slot values change.
 */
final class FlashAttention {

    static final boolean FLASH_ENABLED = !"false".equals(System.getProperty("cuda.attn.flash", "true"));
    private static final int WARPS = 8;       // FA_WARPS in attention_flash.cu
    private static final int MAX_HEAD = 576;  // 32 * 18: the latent MLA head (kvLoraRank 512 + rope 64)

    private final CudaContext ctx;
    private final CudaBufferManager bufferManager;
    private final boolean flash;
    private final boolean fp16;
    private final int headCount;
    private final int maxSplits;
    private final int smCount;
    private final long tokenParamsPtr;
    private final int maxSeqLen;

    // attention_flash_g<G>_j<NJ>[_f16], compiled on first use: [G][NJ]
    private final MemorySegment[][] splitFuncs = new MemorySegment[9][19];
    private final MemorySegment legacyFunc;
    private final KernelParams splitPB;
    private final KernelParams batchPB;
    private final int maxBatchTokens;
    private final long tickets;   // per head-group completion counters (last block merges)
    private final KernelParams legacyPB;
    private final long partialAcc;
    private final long partialML;
    private final int legacyMaxShared;

    /**
     * @param headCount   query heads (fixed for the model)
     * @param maxHeadSize largest head size any layer uses
     * @param fp16        K/V cache stored as FP16
     * @param tokenParams device pointer to {position, seqLen}
     */
    FlashAttention(CudaContext ctx, CudaBufferManager bufferManager, Arena arena, int headCount, int maxHeadSize,
                   int maxSeqLen, boolean fp16, long tokenParams) {
        this(ctx, bufferManager, arena, headCount, maxHeadSize, maxSeqLen, fp16, tokenParams, 1);
    }

    /**
     * @param maxBatchTokens largest token count of a {@link #launchBatch} call (sizes the
     *                       per-token partial buffers); 1 when only single-token decode is used
     */
    FlashAttention(CudaContext ctx, CudaBufferManager bufferManager, Arena arena, int headCount, int maxHeadSize,
                   int maxSeqLen, boolean fp16, long tokenParams, int maxBatchTokens) {
        this.ctx = ctx;
        this.bufferManager = bufferManager;
        this.headCount = headCount;
        this.fp16 = fp16;
        this.tokenParamsPtr = tokenParams;
        this.maxSeqLen = maxSeqLen;
        this.flash = FLASH_ENABLED && maxHeadSize <= MAX_HEAD;

        this.smCount = Math.max(1, ctx.getDeviceInfo().computeUnits());
        if (flash) {
            this.maxSplits = Math.max(1, Math.min(MAX_SPLITS, (maxSeqLen + 15) / 16));
            // Decode uses up to maxSplits slices of one token; a batched launch uses one slice per
            // token (the tokens themselves provide the parallelism).
            this.maxBatchTokens = Math.max(1, maxBatchTokens);
            long slices = Math.max(maxSplits, this.maxBatchTokens);
            partialAcc = bufferManager.createBuffer((long) headCount * slices * maxHeadSize * Float.BYTES);
            partialML = bufferManager.createBuffer((long) headCount * slices * 2 * Float.BYTES);
            tickets = bufferManager.createBuffer((long) headCount * this.maxBatchTokens * Integer.BYTES);
            ctx.fillBufferZero(tickets, (long) headCount * this.maxBatchTokens * Integer.BYTES);
            splitPB = new KernelParams(arena, 19);
            splitPB.setLong(1, tickets).setLong(2, partialAcc).setLong(3, partialML)
                   .setInt(7, headCount).setLong(11, tokenParams).setInt(16, 0).setInt(17, 0).setLong(18, 0L);
            batchPB = new KernelParams(arena, 19);
            batchPB.setLong(1, tickets).setLong(2, partialAcc).setLong(3, partialML).setInt(7, headCount)
                   .setLong(18, 0L);
            legacyFunc = null;
            legacyPB = null;
            legacyMaxShared = 0;
        } else {
            this.maxSplits = 0;
            this.maxBatchTokens = 0;
            splitPB = null;
            batchPB = null;
            tickets = 0;
            partialAcc = 0;
            partialML = 0;
            legacyFunc = fp16
                ? ctx.compileKernel("kernels/cuda/attention_f16.cu", "attention_full_f16")
                : ctx.compileKernel("kernels/cuda/attention.cu", "attention_full");
            int want = (maxSeqLen + 32) * Float.BYTES;
            int optin = ctx.getMaxSharedMemOptin();
            int granted = Math.min(want, optin);
            if (granted > 48 * 1024 && !ctx.setMaxDynamicSharedMem(legacyFunc, granted)) granted = 48 * 1024;
            legacyMaxShared = Math.max(granted, Math.min(want, 48 * 1024));
            legacyPB = new KernelParams(arena, 10);
            legacyPB.setInt(4, headCount).setLong(8, tokenParams);
        }
    }

    /**
     * Compile the split kernel {@link #launch} will use for this head layout now, instead of at
     * the first launch (module loads allocate device memory, and the first launch of a pass may
     * come after the expert cache has taken the free VRAM).
     */
    void precompile(int headCountKV, int headSize) {
        if (!flash) return;
        int nj = njFor(headSize);
        splitFunc(groupSize(headCount / headCountKV, nj), nj);
    }

    /** True when the flash-decoding kernels are in use. */
    boolean isFlash() { return flash; }

    /**
     * Longest context the kernels can attend over. Unlimited for flash; for the legacy kernel it
     * is bounded by the shared memory the device granted.
     */
    int maxLegacySeqLen() {
        return flash ? Integer.MAX_VALUE : legacyMaxShared / Float.BYTES - 32;
    }

    /** Whether a CUDA graph (fixed launch configuration for every position) can use this. */
    boolean graphCompatible() {
        return flash || (maxSeqLen + 32) * Float.BYTES <= legacyMaxShared;
    }

    /**
     * Attention for the current token (position and seqLen read on the device from tokenParams).
     *
     * @param position    host-side position; only used to size the legacy kernel's shared memory
     *                    outside graph capture (pass -1 when capturing a graph)
     * @param scale       score scale, normally 1/sqrt(headSize)
     * @param softcap     attention-logit soft-cap (Gemma 2), 0 for none — flash path only
     */
    void launch(MemorySegment stream, long out, long q, long kCache, long vCache,
                int headCountKV, int headSize, int kvDim, int slidingWindow,
                float scale, float softcap, int position) {
        launch(stream, out, q, kCache, vCache, headCountKV, headSize, kvDim, slidingWindow, scale, softcap, position, 0L);
    }

    /**
     * As above, with optional attention sinks: {@code sinks} is a device pointer to one logit per
     * query head that joins the softmax denominator only (GPT-OSS), 0 for none. Flash path only.
     */
    void launch(MemorySegment stream, long out, long q, long kCache, long vCache,
                int headCountKV, int headSize, int kvDim, int slidingWindow,
                float scale, float softcap, int position, long sinks) {
        if (flash) {
            splitPB.setLong(18, sinks);
            int njK = njFor(headSize);
            int g = groupSize(headCount / headCountKV, njK);
            int groups = headCount / g;
            // Enough blocks to fill the GPU several times over (4 per SM).
            int nSplits = Math.max(1, Math.min(maxSplits, (4 * smCount + groups - 1) / groups));
            if (SPLITS_OVERRIDE > 0) nSplits = Math.min(maxSplits, SPLITS_OVERRIDE);
            splitPB.setLong(0, out).setLong(4, q).setLong(5, kCache).setLong(6, vCache)
                   .setInt(8, headCountKV).setInt(9, headSize).setInt(10, kvDim)
                   .setInt(12, slidingWindow).setInt(13, nSplits).setFloat(14, scale).setFloat(15, softcap);
            check(CudaBindings.launchKernel(splitFunc(g, njK), groups, nSplits, 1, WARPS * 32, 1, 1,
                WARPS * g * headSize * Float.BYTES, stream, splitPB.ptrs(), MemorySegment.NULL));
        } else {
            if (softcap > 0f || sinks != 0) throw new IllegalStateException("attention soft-cap / sinks need the flash kernel");
            int seq = position < 0 ? maxSeqLen : position + 1;
            int shared = (seq + 32) * Float.BYTES;
            if (shared > legacyMaxShared) {
                throw new IllegalStateException("context " + seq + " exceeds the legacy attention kernel limit ("
                    + maxLegacySeqLen() + " tokens); enable -Dcuda.attn.flash=true");
            }
            // The legacy kernel hard-codes 1/sqrt(headSize); callers with another scale pre-scale Q.
            legacyPB.setLong(0, out).setLong(1, q).setLong(2, kCache).setLong(3, vCache)
                    .setInt(5, headCountKV).setInt(6, headSize).setInt(7, kvDim).setInt(9, slidingWindow);
            check(CudaBindings.launchKernel(legacyFunc, headCount, 1, 1, 256, 1, 1,
                shared, stream, legacyPB.ptrs(), MemorySegment.NULL));
        }
    }

    private static final int MAX_SPLITS = 32;
    private static final int SPLITS_OVERRIDE = Integer.getInteger("cuda.attn.splits", 0);
    private static final int[] GROUP_SIZES = {8, 7, 6, 4, 3, 2, 1};

    /**
     * Query heads per block: the largest instantiated G that divides the GQA ratio and keeps
     * G * NJ <= 32 (so q and the accumulators fit in registers). Every head of a block shares
     * one KV head, so each K/V row is read once for all G heads.
     */
    private static int groupSize(int kvMul, int nj) {
        for (int g : GROUP_SIZES) {
            if (kvMul % g == 0 && g * nj <= 32 && instantiated(g, nj)) return g;
        }
        return 1;
    }

    /** Registers per lane for a head size: the instantiated NJ that covers it. */
    private static int njFor(int headSize) {
        int nj = (headSize + 31) / 32;
        return nj <= 2 ? 2 : nj <= 3 ? 3 : nj <= 4 ? 4 : nj <= 8 ? 8 : nj <= 16 ? 16 : 18;
    }

    private static boolean instantiated(int g, int nj) {
        if (nj == 18) return g == 1;
        switch (g) {
            case 1: case 2: return nj <= 16;
            case 3: case 4: return nj <= 8;
            default: return nj <= 4; // 6, 7, 8
        }
    }

    private MemorySegment splitFunc(int g, int nj) {
        MemorySegment f = splitFuncs[g][nj];
        if (f == null) {
            f = ctx.compileKernel("kernels/cuda/attention_flash.cu",
                "attention_flash_g" + g + "_j" + nj + (fp16 ? "_f16" : ""),
                "#define FA_G " + g + "\n#define FA_NJ " + nj + "\n");
            splitFuncs[g][nj] = f;
        }
        return f;
    }

    /**
     * Attention for {@code n} consecutive tokens in one launch (batched prefill): token t reads its
     * position from {@code tokenParams[2t]}, its query at {@code q + t*qStride} floats and writes
     * {@code out + t*qStride}. Every token's K/V must already be in the cache. Flash path only.
     */
    void launchBatch(MemorySegment stream, long out, long q, long kCache, long vCache,
                     int headCountKV, int headSize, int kvDim, int slidingWindow,
                     float scale, float softcap, int n, long tokenParams, int qStride) {
        launchBatch(stream, out, q, kCache, vCache, headCountKV, headSize, kvDim, slidingWindow, scale, softcap,
            n, tokenParams, qStride, 0L);
    }

    /** As above, with optional attention sinks (GPT-OSS; device pointer, 0 for none). */
    void launchBatch(MemorySegment stream, long out, long q, long kCache, long vCache,
                     int headCountKV, int headSize, int kvDim, int slidingWindow,
                     float scale, float softcap, int n, long tokenParams, int qStride, long sinks) {
        if (!flash) throw new IllegalStateException("batched attention needs the flash kernel");
        batchPB.setLong(18, sinks);
        if (n > maxBatchTokens) throw new IllegalArgumentException("batch " + n + " > " + maxBatchTokens);
        int njK = njFor(headSize);
        int g = groupSize(headCount / headCountKV, njK);
        int groups = headCount / g;
        batchPB.setLong(0, out).setLong(4, q).setLong(5, kCache).setLong(6, vCache)
               .setInt(8, headCountKV).setInt(9, headSize).setInt(10, kvDim).setLong(11, tokenParams)
               .setInt(12, slidingWindow).setInt(13, 1).setFloat(14, scale).setFloat(15, softcap)
               .setInt(16, qStride).setInt(17, qStride);
        check(CudaBindings.launchKernel(splitFunc(g, njK), groups, 1, n, WARPS * 32, 1, 1,
            WARPS * g * headSize * Float.BYTES, stream, batchPB.ptrs(), MemorySegment.NULL));
    }

    /** Legacy kernel only honours 1/sqrt(headSize): callers that need another scale must pre-scale Q. */
    boolean supportsCustomScale() { return flash; }


    private static void check(int err) {
        if (err != CudaBindings.CUDA_SUCCESS) throw new RuntimeException("CUDA error in attention launch: " + err);
    }

    void close() {
        if (partialAcc != 0) try { ctx.freeBuffer(partialAcc); } catch (Exception ignored) {}
        if (partialML != 0) try { ctx.freeBuffer(partialML); } catch (Exception ignored) {}
        if (tickets != 0) try { ctx.freeBuffer(tickets); } catch (Exception ignored) {}
    }
}
