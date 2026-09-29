package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.gpu.CudaBindings;
import it.denzosoft.llmplayer.gpu.CudaBufferManager;
import it.denzosoft.llmplayer.gpu.CudaContext;
import it.denzosoft.llmplayer.gpu.Dp4aMatmul;
import it.denzosoft.llmplayer.gpu.KernelParams;
import it.denzosoft.llmplayer.model.DeepSeek2LayerWeights;
import it.denzosoft.llmplayer.model.DeepSeek2Weights;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.tensor.CudaFloatTensor;
import it.denzosoft.llmplayer.tensor.FloatTensor;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * GPU-resident Multi-head Latent Attention (DeepSeek-V2 layout: direct or Q-LoRA query, combined
 * {@code wkv_b}) for {@link DeepSeek2InferenceEngine}; the MoE FFN stays with the engine.
 *
 * <p>Per layer, with one upload (residual + position) and one download (residual + FFN-normed
 * input): attn RMSNorm → Q (or Q-LoRA A → norm → B) → wkv_a → latent RMSNorm → wkv_b → per-head
 * K = [K_nope | RoPE(k_rope)] and V (padded to keyLength so the flash kernel can use one head size)
 * → RoPE on each head's Q rope part → KV write → flash attention (scale mscale² / sqrt(keyLength),
 * as MLAAttention) → drop the V padding → Wo into the residual → FFN RMSNorm. The expanded K/V cache
 * lives on the device, like the CPU path's.
 *
 * <p>The separate {@code wk_b}/{@code wv_b} variant (GLM-4.7-Flash, DeepSeek-V3) runs in the latent
 * space by default ({@link ModelConfig#mlaLatentCache()}, as llama.cpp does): the cache holds one
 * {@code [latent | k_rope]} row of kvLoraRank + ropeDim floats per token, shared by every head, so
 * attention is multi-query with a single KV head of that size used as both K and V. Per head the
 * query is absorbed first ({@code q_lat = wk_b[h] · q_nope}, FP16 per-head matvec) and the output
 * decompressed after ({@code out = wv_b[h] · o_lat}). For GLM-4.7-Flash at 2K context this is
 * 222 MB of KV instead of 3.9 GB, which on a 6 GB card is the difference between fitting and
 * paging. With {@code -Dmla.latent=false} it expands K/V instead: {@code wk_b} is applied
 * transposed from an FP16 copy [heads·keyNope × kvLoraRank], {@code wv_b} as is.
 */
public final class MlaAttentionCudaPass implements GpuAttentionPass {

    private final CudaContext ctx;
    private final Arena arena;
    private final MemorySegment stream;
    private final DeepSeek2Weights weights;
    private final int dim, headCount, keyLength, valueLength, kvLoraRank, qLoraRank, ropeDim, keyNope;
    private final int kDim, blockSize, halfRope, ropeType;
    private final float normEps, attnScale;
    private final boolean[] onGpu;

    private final long gpuBlock, gpuTokenParams, gpuX, gpuXbOut;
    private final long gpuXn, gpuQ, gpuQc, gpuKvc, gpuLat, gpuKvd, gpuK, gpuV, gpuAtt, gpuAttC;
    private final MemorySegment hostBlock;
    private final long[] gpuAttnNorm, gpuFfnNorm, gpuKvANorm, gpuQANorm, gpuKeyCache, gpuValueCache;
    private final long gpuCos, gpuSin;
    private final MemorySegment rmsnormFunc, ropeFunc, kvUpdateFunc, splitFunc, compactFunc;
    private final KernelParams normPB, ropePB, kvPB, splitPB, compactPB;
    // separate K_b / V_b variant
    private final long[] gpuKbT;       // per layer: transposed wk_b as FP16 [heads*keyNope][kvLoraRank]
    private final long gpuKnope, gpuVsrc;
    // latent variant
    private final boolean latent;
    private final int entry;               // kvLoraRank + ropeDim
    private final long[] gpuKbH, gpuVbH;   // per layer: wk_b / wv_b as FP16, layout as in the GGUF
    private final long gpuQLat, gpuOLat;   // [heads][entry]
    private MemorySegment headsMatmulFunc, qRopeCopyFunc;
    private final KernelParams headsPB, qRopeCopyPB;
    // shared expert on the FFN-normed input, downloaded with the residual (one sync per layer)
    private final boolean[] sharedOnGpu;
    private final int sharedFfn;
    private final long gpuSh, gpuShG, gpuShU;
    private final Dp4aMatmul mmXb, mmSh;
    private final MemorySegment siluMulFunc;
    private final KernelParams siluPB;
    private int sharedLayer = -1;
    private MemorySegment f16MatmulFunc, assembleFunc;
    private final KernelParams f16PB, assemblePB;
    private final int normShared;
    private final Dp4aMatmul mmX, mmQc, mmLat, mmAtt;
    private final FlashAttention flash;
    private final boolean fp16Kv = "true".equals(System.getProperty("cuda.kv.fp16", "false"));

    // Per-layer CUDA graphs, as in MoeAttentionCudaPass (F3): one for the attention launches and
    // one for the shared expert, captured on a layer's second call.
    private final MemorySegment[] graphExec, sharedExec;
    private final boolean[] ranOnce, sharedRanOnce;
    private boolean graphsBroken;

    // Batched prefill (F6): chunk buffers allocated in the constructor. The expanded layout
    // (DeepSeek-V2-Lite's combined wkv_b, or -Dmla.latent=false) needs cuBLAS.
    private final int maxBatch;
    private long bX, bXb, bXn, bQ, bQc, bKvc, bQLat, bOLat, bAttC, bTP;
    private long bLat, bKvd, bKnope, bVsrc, bK, bV, bAtt;   // expanded layout
    private MemorySegment bHost;
    private MemorySegment rmsBatchFunc, ropeBatchFunc, kvBatchFunc;
    private KernelParams bNormPB, bRopePB, bKvPB, bRowPB, bHeadsPB, bCopyPB;
    private GemmF16 gemm; // batched projections through cuBLAS; null: per-token dp4a loop

    public MlaAttentionCudaPass(ModelConfig config, DeepSeek2Weights weights, CudaBufferManager bm,
                                RoPE rope, int maxSeqLen) {
        this.ctx = bm.getCudaContext();
        this.arena = Arena.ofShared();
        this.stream = ctx.getStream();
        this.weights = weights;
        this.dim = config.embeddingLength();
        this.headCount = config.headCount();
        this.keyLength = config.keyLength();
        this.valueLength = config.valueLength();
        this.kvLoraRank = config.kvLoraRank();
        this.qLoraRank = config.qLoraRank();
        this.ropeDim = config.ropeDimensionCount();
        this.keyNope = keyLength - ropeDim;
        this.kDim = headCount * keyLength;
        this.normEps = config.normEps();
        this.halfRope = rope.getRopeDimCount() / 2;
        this.ropeType = rope.getRopeType();
        float mscale = rope.getMscale();
        this.attnScale = mscale * mscale / (float) Math.sqrt(keyLength);
        this.blockSize = (int) Math.min(256, ctx.getDeviceInfo().maxWorkGroupSize());
        int blocks = config.blockCount();
        long fb = Float.BYTES;

        // [position, seqLen, pad | x | xbOut | shared expert output]
        gpuBlock = bm.createBuffer(16 + 3L * dim * fb);
        gpuTokenParams = gpuBlock;
        gpuX = gpuBlock + 16;
        gpuXbOut = gpuX + dim * fb;
        gpuSh = gpuXbOut + dim * fb;
        hostBlock = ctx.allocPinnedHost(16 + 3L * dim * fb); // page-locked: true DMA both ways
        gpuXn = bm.createBuffer(dim * fb);
        gpuQ = bm.createBuffer((long) kDim * fb);
        gpuQc = bm.createBuffer((long) Math.max(1, qLoraRank) * fb);
        gpuKvc = bm.createBuffer((long) (kvLoraRank + ropeDim) * fb);
        gpuLat = bm.createBuffer((long) kvLoraRank * fb);
        gpuKvd = bm.createBuffer((long) headCount * (keyNope + valueLength) * fb);
        gpuK = bm.createBuffer((long) kDim * fb);
        gpuV = bm.createBuffer((long) kDim * fb);
        gpuAtt = bm.createBuffer((long) kDim * fb);
        gpuAttC = bm.createBuffer((long) headCount * valueLength * fb);
        gpuKnope = bm.createBuffer((long) headCount * keyNope * fb);
        gpuVsrc = bm.createBuffer((long) headCount * valueLength * fb);
        gpuKbT = new long[config.blockCount()];
        entry = kvLoraRank + ropeDim;
        boolean allSeparate = blocks > 0;
        for (DeepSeek2LayerWeights lw : weights.layers()) allSeparate &= lw.hasSeparateKVB();
        latent = config.mlaLatentCache() && allSeparate;
        gpuKbH = new long[blocks];
        gpuVbH = new long[blocks];
        gpuQLat = latent ? bm.createBuffer((long) headCount * entry * fb) : 0;
        gpuOLat = latent ? bm.createBuffer((long) headCount * entry * fb) : 0;

        onGpu = new boolean[blocks];
        gpuAttnNorm = new long[blocks]; gpuFfnNorm = new long[blocks];
        gpuKvANorm = new long[blocks]; gpuQANorm = new long[blocks];
        gpuKeyCache = new long[blocks]; gpuValueCache = new long[blocks];
        long kvBytes = (long) maxSeqLen * (latent ? entry : kDim) * (fp16Kv ? 2L : fb);
        for (int i = 0; i < blocks; i++) {
            DeepSeek2LayerWeights lw = weights.layers()[i];
            onGpu[i] = layerSupported(lw);
            if (!onGpu[i]) continue;
            try {
            for (FloatTensor t : new FloatTensor[]{lw.wq(), lw.wqA(), lw.wqB(), lw.wkvA(), lw.wkvB(), lw.wo()}) {
                if (t != null) ((CudaFloatTensor) t).getGpuWeights();
            }
            gpuAttnNorm[i] = upload(bm, lw.attnNorm(), dim);
            gpuFfnNorm[i] = upload(bm, lw.ffnNorm(), dim);
            gpuKvANorm[i] = upload(bm, lw.kvANorm(), kvLoraRank);
            if (lw.hasQLoRA()) gpuQANorm[i] = upload(bm, lw.wqANorm(), qLoraRank);
            if (latent) {
                gpuKbH[i] = uploadF16(bm, lw.wkB(), (long) headCount * kvLoraRank * keyNope);
                gpuVbH[i] = uploadF16(bm, lw.wvB(), (long) headCount * valueLength * kvLoraRank);
            } else if (lw.hasSeparateKVB()) {
                ((CudaFloatTensor) lw.wvB()).getGpuWeights();
                gpuKbT[i] = uploadTransposedKb(bm, lw.wkB());
            }
            gpuKeyCache[i] = ctx.allocBufferChecked(kvBytes, "MLA KV");
            ctx.fillBufferZero(gpuKeyCache[i], kvBytes);
            if (latent) {
                gpuValueCache[i] = gpuKeyCache[i]; // V is the latent part of the same row
            } else {
                gpuValueCache[i] = ctx.allocBufferChecked(kvBytes, "MLA KV");
                ctx.fillBufferZero(gpuValueCache[i], kvBytes);
            }
            } catch (it.denzosoft.llmplayer.gpu.VramGuard.VramExhaustedException e) {
                // Out of real VRAM (F7): this layer and the following ones keep CPU attention.
                if (gpuKeyCache[i] != 0) { ctx.freeBuffer(gpuKeyCache[i]); gpuKeyCache[i] = 0; }
                gpuValueCache[i] = 0;
                for (int j = i; j < blocks; j++) onGpu[j] = false;
                System.out.println("MLA CUDA attention: layers " + i + "-" + (blocks - 1)
                    + " stay on the CPU (" + e.getMessage() + ")");
                break;
            }
        }
        gpuCos = uploadArray(bm, rope.getCosTable());
        gpuSin = uploadArray(bm, rope.getSinTable());

        rmsnormFunc = ctx.compileKernel("kernels/cuda/rmsnorm.cu", "rmsnorm_fused");
        ropeFunc = ctx.compileKernel("kernels/cuda/rope.cu", "rope_apply");
        kvUpdateFunc = fp16Kv
            ? ctx.compileKernel("kernels/cuda/attention_f16.cu", "kv_cache_update_f16")
            : ctx.compileKernel("kernels/cuda/attention.cu", "kv_cache_update");
        splitFunc = ctx.compileKernel("kernels/cuda/batch_ops.cu", "mla_split_kv");
        compactFunc = ctx.compileKernel("kernels/cuda/batch_ops.cu", "mla_compact");
        normPB = new KernelParams(arena, 5);
        normPB.setFloat(4, normEps);
        ropePB = new KernelParams(arena, 8);
        ropePB.setLong(1, gpuCos).setLong(2, gpuSin).setInt(5, halfRope).setLong(6, gpuTokenParams).setInt(7, ropeType);
        kvPB = new KernelParams(arena, 6);
        if (latent) kvPB.setLong(2, gpuKvc).setLong(3, gpuKvc).setInt(4, entry).setLong(5, gpuTokenParams);
        else kvPB.setLong(2, gpuK).setLong(3, gpuV).setInt(4, kDim).setLong(5, gpuTokenParams);
        splitPB = new KernelParams(arena, 8);
        splitPB.setLong(0, gpuKvd).setLong(1, gpuKvc + (long) kvLoraRank * fb).setLong(2, gpuK).setLong(3, gpuV)
               .setInt(4, headCount).setInt(5, keyNope).setInt(6, ropeDim).setInt(7, valueLength);
        compactPB = new KernelParams(arena, 5);
        compactPB.setLong(0, gpuAtt).setLong(1, gpuAttC).setInt(2, headCount).setInt(3, keyLength).setInt(4, valueLength);
        normShared = (blockSize / 32 + 1) * Float.BYTES;
        f16PB = new KernelParams(arena, 6);
        assemblePB = new KernelParams(arena, 9);
        headsPB = new KernelParams(arena, 7);
        qRopeCopyPB = new KernelParams(arena, 8);
        if (latent) {
            headsMatmulFunc = ctx.compileKernel("kernels/cuda/batch_ops.cu", "matmul_f16_heads");
            qRopeCopyFunc = ctx.compileKernel("kernels/cuda/batch_ops.cu", "mla_q_rope_copy");
            qRopeCopyPB.setLong(0, gpuQ).setLong(1, gpuQLat).setInt(2, headCount).setInt(3, keyLength)
                       .setInt(4, keyNope).setInt(5, entry).setInt(6, kvLoraRank).setInt(7, ropeDim);
        }
        boolean anySeparate = false;
        for (long p : gpuKbT) if (p != 0) anySeparate = true;
        if (anySeparate) {
            f16MatmulFunc = ctx.compileKernel("kernels/cuda/matmul_f16.cu", "matmul_f16");
            assembleFunc = ctx.compileKernel("kernels/cuda/batch_ops.cu", "mla_assemble_kv");
            assemblePB.setLong(0, gpuKnope).setLong(1, gpuVsrc).setLong(2, gpuKvc + (long) kvLoraRank * fb)
                      .setLong(3, gpuK).setLong(4, gpuV).setInt(5, headCount).setInt(6, keyNope)
                      .setInt(7, ropeDim).setInt(8, valueLength);
        }

        sharedFfn = config.expertSharedCount() * config.expertFfnLength();
        sharedOnGpu = new boolean[blocks];
        boolean anyShared = false;
        if (sharedFfn > 0 && !"false".equals(System.getProperty("mla.shared.gpu", "true"))) {
            for (int i = config.leadingDenseBlockCount(); i < blocks; i++) {
                DeepSeek2LayerWeights lw = weights.layers()[i];
                sharedOnGpu[i] = onGpu[i] && lw.ffnGateShexp() instanceof CudaFloatTensor
                    && lw.ffnUpShexp() instanceof CudaFloatTensor && lw.ffnDownShexp() instanceof CudaFloatTensor;
                if (sharedOnGpu[i]) {
                    // Upload now: lazily, on the first forward, they would land after the expert
                    // cache has taken the free VRAM (in shared memory under WDDM)
                    ((CudaFloatTensor) lw.ffnGateShexp()).getGpuWeights();
                    ((CudaFloatTensor) lw.ffnUpShexp()).getGpuWeights();
                    ((CudaFloatTensor) lw.ffnDownShexp()).getGpuWeights();
                }
                anyShared |= sharedOnGpu[i];
            }
        }
        gpuShG = anyShared ? bm.createBuffer((long) sharedFfn * fb) : 0;
        gpuShU = anyShared ? bm.createBuffer((long) sharedFfn * fb) : 0;
        siluMulFunc = anyShared ? ctx.compileKernel("kernels/cuda/silu_mul.cu", "silu_mul") : null;
        siluPB = new KernelParams(arena, 3);
        siluPB.setLong(0, gpuShG).setLong(1, gpuShU).setInt(2, sharedFfn);
        mmXb = new Dp4aMatmul(ctx, bm, arena, dim);
        mmSh = new Dp4aMatmul(ctx, bm, arena, Math.max(32, sharedFfn));

        mmX = new Dp4aMatmul(ctx, bm, arena, dim);
        mmQc = new Dp4aMatmul(ctx, bm, arena, Math.max(32, qLoraRank));
        mmLat = new Dp4aMatmul(ctx, bm, arena, kvLoraRank);
        mmAtt = new Dp4aMatmul(ctx, bm, arena, headCount * valueLength);
        int nb = latent || GemmF16.available() ? MoeAttentionCudaPass.BATCH : 0;
        if (nb > 0) {
            try {
                long f = Float.BYTES;
                bX = bm.createBuffer(2L * nb * dim * f);
                bXb = bX + (long) nb * dim * f;
                bXn = bm.createBuffer((long) nb * dim * f);
                bQ = bm.createBuffer((long) nb * kDim * f);
                bQc = bm.createBuffer((long) nb * Math.max(1, qLoraRank) * f);
                bKvc = bm.createBuffer((long) nb * entry * f);
                if (latent) {
                    bQLat = bm.createBuffer((long) nb * headCount * entry * f);
                    bOLat = bm.createBuffer((long) nb * headCount * entry * f);
                } else {
                    bLat = bm.createBuffer((long) nb * kvLoraRank * f);
                    bKvd = bm.createBuffer((long) nb * headCount * (keyNope + valueLength) * f);
                    bKnope = bm.createBuffer((long) nb * headCount * keyNope * f);
                    bVsrc = bm.createBuffer((long) nb * headCount * valueLength * f);
                    bK = bm.createBuffer((long) nb * kDim * f);
                    bV = bm.createBuffer((long) nb * kDim * f);
                    bAtt = bm.createBuffer((long) nb * kDim * f);
                }
                bAttC = bm.createBuffer((long) nb * headCount * valueLength * f);
                bTP = bm.createBuffer((long) nb * 8);
                bHost = ctx.allocPinnedHost(2L * nb * dim * f + (long) nb * 8);
                String bo = "kernels/cuda/batch_ops.cu";
                rmsBatchFunc = ctx.compileKernel(bo, "rmsnorm_batch");
                ropeBatchFunc = ctx.compileKernel(bo, "rope_apply_batch");
                kvBatchFunc = ctx.compileKernel(bo, fp16Kv ? "kv_cache_update_batch_f16" : "kv_cache_update_batch");
                bNormPB = new KernelParams(arena, 5);
                bRopePB = new KernelParams(arena, 9);
                bKvPB = new KernelParams(arena, 6);
                bRowPB = new KernelParams(arena, 5);
                bHeadsPB = new KernelParams(arena, 7);
                bCopyPB = new KernelParams(arena, 8);
            } catch (RuntimeException e) {
                System.err.println("MLA GPU attention: batched prefill unavailable — " + e.getMessage());
                nb = 0;
            }
            if (nb > 0 && GemmF16.available()) {
                try {
                    java.util.List<CudaFloatTensor> ws = new java.util.ArrayList<>();
                    for (int i = 0; i < blocks; i++) {
                        if (!onGpu[i]) continue;
                        DeepSeek2LayerWeights lw = weights.layers()[i];
                        for (FloatTensor t : new FloatTensor[]{lw.wq(), lw.wqA(), lw.wqB(), lw.wkvA(), lw.wo()}) {
                            if (t != null) ws.add((CudaFloatTensor) t);
                        }
                        if (!latent) ws.add((CudaFloatTensor) (lw.hasSeparateKVB() ? lw.wvB() : lw.wkvB()));
                    }
                    int maxCols = Math.max(Math.max(dim, qLoraRank), Math.max(headCount * valueLength, kvLoraRank));
                    gemm = new GemmF16(ctx, bm, arena, nb, maxCols, 16L << 20, ws);
                } catch (RuntimeException e) {
                    gemm = null;
                }
            }
            if (!latent && gemm == null) nb = 0; // the expanded batch has no per-token projection loop
        }
        maxBatch = nb;
        flash = new FlashAttention(ctx, bm, arena, headCount, latent ? entry : keyLength, maxSeqLen, fp16Kv, gpuTokenParams,
            Math.max(1, nb));
        if (latent) flash.precompile(1, entry); else flash.precompile(headCount, keyLength);

        graphExec = new MemorySegment[blocks];
        sharedExec = new MemorySegment[blocks];
        ranOnce = new boolean[blocks];
        sharedRanOnce = new boolean[blocks];
        graphsBroken = !MoeAttentionCudaPass.LAYER_GRAPHS || !ctx.isGraphApiAvailable() || !flash.graphCompatible();

        int n = 0;
        for (boolean b : onGpu) if (b) n++;
        System.err.println("MLA GPU attention: " + n + "/" + blocks + " layers GPU-resident"
            + (latent ? ", latent KV" : "") + (fp16Kv ? " (FP16 KV)" : "")
            + (graphsBroken ? "" : ", per-layer CUDA graphs"));
    }

    private static boolean layerSupported(DeepSeek2LayerWeights lw) {
        if (lw.hasSeparateKVB() && !(lw.wvB() instanceof CudaFloatTensor && lw.wkB() != null)) return false;
        boolean q = lw.hasQLoRA()
            ? lw.wqA() instanceof CudaFloatTensor && lw.wqB() instanceof CudaFloatTensor
            : lw.wq() instanceof CudaFloatTensor;
        return q && lw.wkvA() instanceof CudaFloatTensor
            && (lw.hasSeparateKVB() || lw.wkvB() instanceof CudaFloatTensor)
            && lw.wo() instanceof CudaFloatTensor;
    }

    public static boolean isSupported(ModelConfig config, DeepSeek2Weights weights) {
        if (!FlashAttention.FLASH_ENABLED || config.keyLength() > 512) return false;
        if (config.mlaLatentCache() && config.kvLoraRank() + config.ropeDimensionCount() > 576) return false;
        return weights.layers().length > 0 && layerSupported(weights.layers()[0]);
    }

    @Override
    public boolean isLayerOnGpu(int layer) { return onGpu[layer]; }

    @Override
    public void attentionLayer(int layer, float[] x, float[] xbOut, int position) {
        long fb = Float.BYTES;
        hostBlock.set(ValueLayout.JAVA_INT, 0, position);
        hostBlock.set(ValueLayout.JAVA_INT, 4, position + 1);
        MemorySegment.copy(x, 0, hostBlock, ValueLayout.JAVA_FLOAT, 16, dim);
        ctx.writeBufferAsync(gpuBlock, hostBlock, 16 + (long) dim * fb);

        MemorySegment g = graphExec[layer];
        if (g != null) {
            ctx.launchGraph(g);
        } else if (!graphsBroken && ranOnce[layer] && (g = capture(layer, false)) != null) {
            graphExec[layer] = g;
            ctx.launchGraph(g);
        } else {
            attentionLaunches(layer);
            ranOnce[layer] = true;
        }

        MemorySegment out = hostBlock.asSlice(16, 2L * dim * fb);
        ctx.readBuffer(gpuX, out, 2L * dim * fb);
        MemorySegment.copy(out, ValueLayout.JAVA_FLOAT, 0, x, 0, dim);
        MemorySegment.copy(out, ValueLayout.JAVA_FLOAT, (long) dim * fb, xbOut, 0, dim);

        // Shared expert (SwiGLU) on the FFN-normed input, queued after the download so it runs on
        // the GPU while the CPU routes and computes the routed experts; takeSharedExpert reads it.
        boolean shared = sharedOnGpu[layer];
        if (shared) {
            MemorySegment sg = sharedExec[layer];
            if (sg != null) {
                ctx.launchGraph(sg);
            } else if (!graphsBroken && sharedRanOnce[layer] && (sg = capture(layer, true)) != null) {
                sharedExec[layer] = sg;
                ctx.launchGraph(sg);
            } else {
                sharedLaunches(layer);
                sharedRanOnce[layer] = true;
            }
        }
        sharedLayer = shared ? layer : -1;
    }

    /** Capture the attention (or shared-expert) launches of {@code layer}; null and per-launch for good on failure. */
    private MemorySegment capture(int layer, boolean sharedPart) {
        try {
            ctx.beginCapture();
            try {
                if (sharedPart) sharedLaunches(layer); else attentionLaunches(layer);
            } catch (RuntimeException e) {
                try { ctx.endCapture(); } catch (RuntimeException ignored) { }
                throw e;
            }
            MemorySegment graph = ctx.endCapture();
            MemorySegment exec = ctx.instantiateGraph(graph);
            ctx.destroyGraph(graph);
            return exec;
        } catch (RuntimeException e) {
            graphsBroken = true;
            System.err.println("MLA GPU attention: per-layer graph capture failed (" + e.getMessage()
                + ") — running per launch");
            return null;
        }
    }

    private void sharedLaunches(int layer) {
        DeepSeek2LayerWeights lw = weights.layers()[layer];
        mmXb.invalidate();
        mmXb.matmul((CudaFloatTensor) lw.ffnGateShexp(), 0, gpuXbOut, gpuShG, sharedFfn, dim, false);
        mmXb.matmul((CudaFloatTensor) lw.ffnUpShexp(), 0, gpuXbOut, gpuShU, sharedFfn, dim, false);
        launch(siluMulFunc, (sharedFfn + blockSize - 1) / blockSize, blockSize, 0, siluPB);
        mmSh.invalidate();
        mmSh.matmul((CudaFloatTensor) lw.ffnDownShexp(), 0, gpuShG, gpuSh, dim, sharedFfn, false);
    }

    /** The attention launches of {@code layer}, from the uploaded x to x and xb on the device. */
    private void attentionLaunches(int layer) {
        DeepSeek2LayerWeights lw = weights.layers()[layer];
        long fb = Float.BYTES;
        rmsnorm(gpuXn, gpuX, gpuAttnNorm[layer], dim);
        mmX.invalidate();
        if (lw.hasQLoRA()) {
            mmX.matmul((CudaFloatTensor) lw.wqA(), 0, gpuXn, gpuQc, qLoraRank, dim, false);
            rmsnorm(gpuQc, gpuQc, gpuQANorm[layer], qLoraRank);
            mmQc.invalidate();
            mmQc.matmul((CudaFloatTensor) lw.wqB(), 0, gpuQc, gpuQ, kDim, qLoraRank, false);
        } else {
            mmX.matmul((CudaFloatTensor) lw.wq(), 0, gpuXn, gpuQ, kDim, dim, false);
        }
        mmX.matmul((CudaFloatTensor) lw.wkvA(), 0, gpuXn, gpuKvc, kvLoraRank + ropeDim, dim, false);
        if (latent) {
            latentAttention(layer);
        } else {
        rmsnorm(gpuLat, gpuKvc, gpuKvANorm[layer], kvLoraRank);
        mmLat.invalidate();
        if (!lw.hasSeparateKVB()) {
            mmLat.matmul((CudaFloatTensor) lw.wkvB(), 0, gpuLat, gpuKvd, headCount * (keyNope + valueLength), kvLoraRank, false);
        } else {
            // K_nope = wk_b^T · latent (pre-transposed FP16), V = wv_b · latent
            int rows = headCount * keyNope;
            f16PB.setLong(0, gpuKbT[layer]).setLong(1, gpuLat).setLong(2, gpuKnope)
                 .setInt(3, rows).setInt(4, kvLoraRank).setInt(5, 0);
            launch(f16MatmulFunc, (rows + 7) / 8, 256, 0, f16PB);
            mmLat.matmul((CudaFloatTensor) lw.wvB(), 0, gpuLat, gpuVsrc, headCount * valueLength, kvLoraRank, false);
        }
        // RoPE on the shared k_rope (one "head" of ropeDim) and on every head's Q rope part
        ropePB.setLong(0, gpuKvc + (long) kvLoraRank * fb).setInt(3, 1).setInt(4, ropeDim);
        launch(ropeFunc, (halfRope + blockSize - 1) / blockSize, blockSize, 0, ropePB);
        ropePB.setLong(0, gpuQ + (long) keyNope * fb).setInt(3, headCount).setInt(4, keyLength);
        launch(ropeFunc, (headCount * halfRope + blockSize - 1) / blockSize, blockSize, 0, ropePB);
        launch(lw.hasSeparateKVB() ? assembleFunc : splitFunc, (kDim + blockSize - 1) / blockSize, blockSize, 0,
            lw.hasSeparateKVB() ? assemblePB : splitPB);
        kvPB.setLong(0, gpuKeyCache[layer]).setLong(1, gpuValueCache[layer]);
        launch(kvUpdateFunc, (kDim + blockSize - 1) / blockSize, blockSize, 0, kvPB);
        flash.launch(stream, gpuAtt, gpuQ, gpuKeyCache[layer], gpuValueCache[layer],
            headCount, keyLength, kDim, 0, attnScale, 0f, position());
        launch(compactFunc, (headCount * valueLength + blockSize - 1) / blockSize, blockSize, 0, compactPB);
        }
        mmAtt.invalidate();
        mmAtt.matmul((CudaFloatTensor) lw.wo(), 0, gpuAttC, gpuX, dim, headCount * valueLength, true);
        rmsnorm(gpuXbOut, gpuX, gpuFfnNorm[layer], dim);
    }

    /**
     * Latent attention for one layer, from {@code gpuQ} (heads × [q_nope | q_rope]) and
     * {@code gpuKvc} ([latent | k_rope], raw) to the per-head outputs in {@code gpuAttC}.
     */
    private void latentAttention(int layer) {
        long fb = Float.BYTES;
        rmsnorm(gpuKvc, gpuKvc, gpuKvANorm[layer], kvLoraRank); // in place: gpuKvc = [latent | k_rope]
        ropePB.setLong(0, gpuKvc + (long) kvLoraRank * fb).setInt(3, 1).setInt(4, ropeDim);
        launch(ropeFunc, (halfRope + blockSize - 1) / blockSize, blockSize, 0, ropePB);
        ropePB.setLong(0, gpuQ + (long) keyNope * fb).setInt(3, headCount).setInt(4, keyLength);
        launch(ropeFunc, (headCount * halfRope + blockSize - 1) / blockSize, blockSize, 0, ropePB);
        kvPB.setLong(0, gpuKeyCache[layer]).setLong(1, gpuValueCache[layer]);
        launch(kvUpdateFunc, (entry + blockSize - 1) / blockSize, blockSize, 0, kvPB);
        // q_lat[h] = wk_b[h] · q_nope[h], then the rotated q_rope after it
        headsPB.setLong(0, gpuKbH[layer]).setLong(1, gpuQ).setLong(2, gpuQLat)
               .setInt(3, kvLoraRank).setInt(4, keyNope).setInt(5, keyLength).setInt(6, entry);
        launch2D(headsMatmulFunc, (kvLoraRank + 7) / 8, headCount, headsPB);
        launch(qRopeCopyFunc, (headCount * ropeDim + blockSize - 1) / blockSize, blockSize, 0, qRopeCopyPB);
        flash.launch(stream, gpuOLat, gpuQLat, gpuKeyCache[layer], gpuValueCache[layer],
            1, entry, entry, 0, attnScale, 0f, position());
        // out[h] = wv_b[h] · o_lat[h] (the first kvLoraRank of each head's output)
        headsPB.setLong(0, gpuVbH[layer]).setLong(1, gpuOLat).setLong(2, gpuAttC)
               .setInt(3, valueLength).setInt(4, kvLoraRank).setInt(5, entry).setInt(6, valueLength);
        launch2D(headsMatmulFunc, (valueLength + 7) / 8, headCount, headsPB);
    }

    @Override
    public int maxBatchTokens() { return maxBatch; }

    /**
     * Batched prefill of one layer (latent layout): the norms, RoPE, KV writes and attention run as
     * one launch each over the chunk (one causal flash launch over the latent cache); the
     * projections and the per-head absorb / decompress run per token with the decode kernels.
     */
    @Override
    public void attentionLayerBatch(int layer, float[][] x, float[][] xbOut, int basePos, int n) {
        if (n < 1 || n > maxBatch) throw new IllegalArgumentException("batch " + n + " > " + maxBatch);
        if (!latent) {
            expandedLayerBatch(layer, x, xbOut, basePos, n);
            return;
        }
        DeepSeek2LayerWeights lw = weights.layers()[layer];
        long f = Float.BYTES;
        long xBytes = (long) n * dim * f;
        long tpOff = 2L * maxBatch * dim * f;
        for (int t = 0; t < n; t++) {
            MemorySegment.copy(x[t], 0, bHost, ValueLayout.JAVA_FLOAT, (long) t * dim * f, dim);
            bHost.set(ValueLayout.JAVA_INT, tpOff + t * 8L, basePos + t);
            bHost.set(ValueLayout.JAVA_INT, tpOff + t * 8L + 4, basePos + t + 1);
        }
        ctx.writeBufferAsync(bX, bHost, xBytes);
        ctx.writeBufferAsync(bTP, bHost.asSlice(tpOff, n * 8L), n * 8L);
        mmX.invalidate(); mmQc.invalidate(); mmAtt.invalidate();

        rmsnormB(bXn, bX, gpuAttnNorm[layer], dim, n);
        if (gemm != null) {
            gemm.toF16(bXn, n * dim);
            if (lw.hasQLoRA()) {
                gemm.gemm((CudaFloatTensor) lw.wqA(), qLoraRank, dim, bQc, qLoraRank, n, false);
            } else {
                gemm.gemm((CudaFloatTensor) lw.wq(), kDim, dim, bQ, kDim, n, false);
            }
            gemm.gemm((CudaFloatTensor) lw.wkvA(), kvLoraRank + ropeDim, dim, bKvc, entry, n, false);
            for (int t = 0; t < n; t++) rowNorm(bKvc + (long) t * entry * f, gpuKvANorm[layer], kvLoraRank);
            if (lw.hasQLoRA()) {
                for (int t = 0; t < n; t++) rowNorm(bQc + (long) t * qLoraRank * f, gpuQANorm[layer], qLoraRank);
                gemm.toF16(bQc, n * qLoraRank);
                gemm.gemm((CudaFloatTensor) lw.wqB(), kDim, qLoraRank, bQ, kDim, n, false);
            }
        } else for (int t = 0; t < n; t++) {
            long in = bXn + (long) t * dim * f, q = bQ + (long) t * kDim * f;
            if (lw.hasQLoRA()) {
                long qc = bQc + (long) t * qLoraRank * f;
                mmX.matmul((CudaFloatTensor) lw.wqA(), 0, in, qc, qLoraRank, dim, false);
                rowNorm(qc, gpuQANorm[layer], qLoraRank);
                mmQc.matmul((CudaFloatTensor) lw.wqB(), 0, qc, q, kDim, qLoraRank, false);
            } else {
                mmX.matmul((CudaFloatTensor) lw.wq(), 0, in, q, kDim, dim, false);
            }
            long kvc = bKvc + (long) t * entry * f;
            mmX.matmul((CudaFloatTensor) lw.wkvA(), 0, in, kvc, kvLoraRank + ropeDim, dim, false);
            rowNorm(kvc, gpuKvANorm[layer], kvLoraRank); // the latent part of the row, in place
        }
        // RoPE on each token's shared k_rope and on every head's q rope part
        ropeB(bKvc + (long) kvLoraRank * f, 1, ropeDim, entry, n);
        ropeB(bQ + (long) keyNope * f, headCount, keyLength, kDim, n);
        bKvPB.setLong(0, gpuKeyCache[layer]).setLong(1, gpuValueCache[layer]).setLong(2, bKvc).setLong(3, bKvc)
             .setInt(4, entry).setLong(5, bTP);
        launchG(kvBatchFunc, (entry + blockSize - 1) / blockSize, n, blockSize, 0, bKvPB);
        for (int t = 0; t < n; t++) {
            long q = bQ + (long) t * kDim * f, qLat = bQLat + (long) t * headCount * entry * f;
            bHeadsPB.setLong(0, gpuKbH[layer]).setLong(1, q).setLong(2, qLat)
                    .setInt(3, kvLoraRank).setInt(4, keyNope).setInt(5, keyLength).setInt(6, entry);
            launchG(headsMatmulFunc, (kvLoraRank + 7) / 8, headCount, 256, 0, bHeadsPB);
            bCopyPB.setLong(0, q).setLong(1, qLat).setInt(2, headCount).setInt(3, keyLength)
                   .setInt(4, keyNope).setInt(5, entry).setInt(6, kvLoraRank).setInt(7, ropeDim);
            launchG(qRopeCopyFunc, (headCount * ropeDim + blockSize - 1) / blockSize, 1, blockSize, 0, bCopyPB);
        }
        flash.launchBatch(stream, bOLat, bQLat, gpuKeyCache[layer], gpuValueCache[layer], 1, entry, entry, 0,
            attnScale, 0f, n, bTP, headCount * entry);
        for (int t = 0; t < n; t++) {
            long oLat = bOLat + (long) t * headCount * entry * f, attC = bAttC + (long) t * headCount * valueLength * f;
            bHeadsPB.setLong(0, gpuVbH[layer]).setLong(1, oLat).setLong(2, attC)
                    .setInt(3, valueLength).setInt(4, kvLoraRank).setInt(5, entry).setInt(6, valueLength);
            launchG(headsMatmulFunc, (valueLength + 7) / 8, headCount, 256, 0, bHeadsPB);
            if (gemm == null) {
                mmAtt.matmul((CudaFloatTensor) lw.wo(), 0, attC, bX + (long) t * dim * f, dim, headCount * valueLength, true);
            }
        }
        if (gemm != null) {
            gemm.toF16(bAttC, n * headCount * valueLength);
            gemm.gemm((CudaFloatTensor) lw.wo(), dim, headCount * valueLength, bX, dim, n, true);
        }
        rmsnormB(bXb, bX, gpuFfnNorm[layer], dim, n);

        ctx.readBufferAsync(bX, bHost, xBytes);
        ctx.readBufferAsync(bXb, bHost.asSlice(xBytes, xBytes), xBytes);
        ctx.finish();
        for (int t = 0; t < n; t++) {
            MemorySegment.copy(bHost, ValueLayout.JAVA_FLOAT, (long) t * dim * f, x[t], 0, dim);
            MemorySegment.copy(bHost, ValueLayout.JAVA_FLOAT, xBytes + (long) t * dim * f, xbOut[t], 0, dim);
        }
        mmX.invalidate(); mmQc.invalidate(); mmAtt.invalidate();
        sharedLayer = -1; // the batched prefill computes the shared expert on the CPU
    }

    /**
     * Batched prefill of one layer in the expanded layout (per-head K/V in the cache; DeepSeek-V2-Lite
     * with its combined {@code wkv_b}, or {@code -Dmla.latent=false}): the projections, {@code wkv_b}
     * (or {@code wv_b}) and {@code Wo} as GEMMs over the chunk, the norms, RoPE, KV writes, the
     * attention (one causal flash launch) and the V compaction over the whole chunk; the latent row
     * norm, the per-head K/V assembly (the shared k_rope differs per token) and the transposed
     * {@code wk_b} product (FP16, separate layout) per token.
     */
    private void expandedLayerBatch(int layer, float[][] x, float[][] xbOut, int basePos, int n) {
        DeepSeek2LayerWeights lw = weights.layers()[layer];
        long f = Float.BYTES;
        long xBytes = (long) n * dim * f;
        long tpOff = 2L * maxBatch * dim * f;
        for (int t = 0; t < n; t++) {
            MemorySegment.copy(x[t], 0, bHost, ValueLayout.JAVA_FLOAT, (long) t * dim * f, dim);
            bHost.set(ValueLayout.JAVA_INT, tpOff + t * 8L, basePos + t);
            bHost.set(ValueLayout.JAVA_INT, tpOff + t * 8L + 4, basePos + t + 1);
        }
        ctx.writeBufferAsync(bX, bHost, xBytes);
        ctx.writeBufferAsync(bTP, bHost.asSlice(tpOff, n * 8L), n * 8L);
        mmX.invalidate(); mmQc.invalidate(); mmLat.invalidate(); mmAtt.invalidate();

        rmsnormB(bXn, bX, gpuAttnNorm[layer], dim, n);
        gemm.toF16(bXn, n * dim);
        if (lw.hasQLoRA()) {
            gemm.gemm((CudaFloatTensor) lw.wqA(), qLoraRank, dim, bQc, qLoraRank, n, false);
        } else {
            gemm.gemm((CudaFloatTensor) lw.wq(), kDim, dim, bQ, kDim, n, false);
        }
        gemm.gemm((CudaFloatTensor) lw.wkvA(), kvLoraRank + ropeDim, dim, bKvc, entry, n, false);
        if (lw.hasQLoRA()) {
            for (int t = 0; t < n; t++) rowNorm(bQc + (long) t * qLoraRank * f, gpuQANorm[layer], qLoraRank);
            gemm.toF16(bQc, n * qLoraRank);
            gemm.gemm((CudaFloatTensor) lw.wqB(), kDim, qLoraRank, bQ, kDim, n, false);
        }
        // latent = RMSNorm(kvc[0..kvLoraRank)), out of place (the k_rope part stays in bKvc)
        for (int t = 0; t < n; t++) {
            bRowPB.setLong(0, bLat + (long) t * kvLoraRank * f).setLong(1, bKvc + (long) t * entry * f)
                  .setLong(2, gpuKvANorm[layer]).setInt(3, kvLoraRank).setFloat(4, normEps);
            launchG(rmsnormFunc, 1, 1, blockSize, normShared, bRowPB);
        }
        gemm.toF16(bLat, n * kvLoraRank);
        int kvdRow = headCount * (keyNope + valueLength);
        if (!lw.hasSeparateKVB()) {
            gemm.gemm((CudaFloatTensor) lw.wkvB(), kvdRow, kvLoraRank, bKvd, kvdRow, n, false);
        } else {
            gemm.gemm((CudaFloatTensor) lw.wvB(), headCount * valueLength, kvLoraRank, bVsrc, headCount * valueLength, n, false);
            int rows = headCount * keyNope;
            for (int t = 0; t < n; t++) {
                f16PB.setLong(0, gpuKbT[layer]).setLong(1, bLat + (long) t * kvLoraRank * f)
                     .setLong(2, bKnope + (long) t * rows * f).setInt(3, rows).setInt(4, kvLoraRank).setInt(5, 0);
                launch(f16MatmulFunc, (rows + 7) / 8, 256, 0, f16PB);
            }
        }
        // RoPE on each token's shared k_rope and on every head's q rope part
        ropeB(bKvc + (long) kvLoraRank * f, 1, ropeDim, entry, n);
        ropeB(bQ + (long) keyNope * f, headCount, keyLength, kDim, n);
        // Per-head K = [k_nope | k_rope], V padded to keyLength
        try {
            for (int t = 0; t < n; t++) {
                long kRope = bKvc + ((long) t * entry + kvLoraRank) * f;
                long k = bK + (long) t * kDim * f, v = bV + (long) t * kDim * f;
                if (lw.hasSeparateKVB()) {
                    assemblePB.setLong(0, bKnope + (long) t * headCount * keyNope * f)
                              .setLong(1, bVsrc + (long) t * headCount * valueLength * f)
                              .setLong(2, kRope).setLong(3, k).setLong(4, v);
                    launch(assembleFunc, (kDim + blockSize - 1) / blockSize, blockSize, 0, assemblePB);
                } else {
                    splitPB.setLong(0, bKvd + (long) t * kvdRow * f).setLong(1, kRope).setLong(2, k).setLong(3, v);
                    launch(splitFunc, (kDim + blockSize - 1) / blockSize, blockSize, 0, splitPB);
                }
            }
        } finally {
            // the decode launches (and their graphs, captured later) read these blocks as built
            assemblePB.setLong(0, gpuKnope).setLong(1, gpuVsrc).setLong(2, gpuKvc + (long) kvLoraRank * f)
                      .setLong(3, gpuK).setLong(4, gpuV);
            splitPB.setLong(0, gpuKvd).setLong(1, gpuKvc + (long) kvLoraRank * f).setLong(2, gpuK).setLong(3, gpuV);
        }
        bKvPB.setLong(0, gpuKeyCache[layer]).setLong(1, gpuValueCache[layer]).setLong(2, bK).setLong(3, bV)
             .setInt(4, kDim).setLong(5, bTP);
        launchG(kvBatchFunc, (kDim + blockSize - 1) / blockSize, n, blockSize, 0, bKvPB);
        flash.launchBatch(stream, bAtt, bQ, gpuKeyCache[layer], gpuValueCache[layer], headCount, keyLength, kDim, 0,
            attnScale, 0f, n, bTP, kDim);
        // Drop the V padding: the chunk's n tokens are n * headCount heads
        try {
            compactPB.setLong(0, bAtt).setLong(1, bAttC).setInt(2, n * headCount);
            launch(compactFunc, (n * headCount * valueLength + blockSize - 1) / blockSize, blockSize, 0, compactPB);
        } finally {
            compactPB.setLong(0, gpuAtt).setLong(1, gpuAttC).setInt(2, headCount);
        }
        gemm.toF16(bAttC, n * headCount * valueLength);
        gemm.gemm((CudaFloatTensor) lw.wo(), dim, headCount * valueLength, bX, dim, n, true);
        rmsnormB(bXb, bX, gpuFfnNorm[layer], dim, n);

        ctx.readBufferAsync(bX, bHost, xBytes);
        ctx.readBufferAsync(bXb, bHost.asSlice(xBytes, xBytes), xBytes);
        ctx.finish();
        for (int t = 0; t < n; t++) {
            MemorySegment.copy(bHost, ValueLayout.JAVA_FLOAT, (long) t * dim * f, x[t], 0, dim);
            MemorySegment.copy(bHost, ValueLayout.JAVA_FLOAT, xBytes + (long) t * dim * f, xbOut[t], 0, dim);
        }
        mmX.invalidate(); mmQc.invalidate(); mmLat.invalidate(); mmAtt.invalidate();
        sharedLayer = -1; // the batched prefill computes the shared expert on the CPU
    }

    private void rmsnormB(long out, long in, long w, int size, int n) {
        bNormPB.setLong(0, out).setLong(1, in).setLong(2, w).setInt(3, size).setFloat(4, normEps);
        launchG(rmsBatchFunc, n, 1, blockSize, normShared, bNormPB);
    }

    /** RMSNorm of one row in place (the first {@code size} floats at {@code row}). */
    private void rowNorm(long row, long w, int size) {
        bRowPB.setLong(0, row).setLong(1, row).setLong(2, w).setInt(3, size).setFloat(4, normEps);
        launchG(rmsnormFunc, 1, 1, blockSize, normShared, bRowPB);
    }

    private void ropeB(long vec, int heads, int headStride, int rowStride, int n) {
        bRopePB.setLong(0, vec).setLong(1, gpuCos).setLong(2, gpuSin).setInt(3, heads).setInt(4, headStride)
               .setInt(5, halfRope).setLong(6, bTP).setInt(7, ropeType).setInt(8, rowStride);
        launchG(ropeBatchFunc, (heads * halfRope + blockSize - 1) / blockSize, n, blockSize, 0, bRopePB);
    }

    private void launchG(MemorySegment fn, int gridX, int gridY, int block, int shared, KernelParams p) {
        int err = CudaBindings.launchKernel(fn, gridX, gridY, 1, block, 1, 1, shared, stream, p.ptrs(), MemorySegment.NULL);
        if (err != CudaBindings.CUDA_SUCCESS) throw new RuntimeException("MLA attention CUDA error: " + err);
    }

    /**
     * Download the shared expert queued by the last {@link #attentionLayer} of {@code layer} (its
     * own synchronisation: call it after the routed experts, so the GPU computed it meanwhile).
     */
    @Override
    public boolean takeSharedExpert(int layer, float[] out) {
        if (sharedLayer != layer) return false;
        sharedLayer = -1;
        MemorySegment sh = hostBlock.asSlice(16 + 2L * dim * Float.BYTES, (long) dim * Float.BYTES);
        ctx.readBuffer(gpuSh, sh, (long) dim * Float.BYTES);
        MemorySegment.copy(sh, ValueLayout.JAVA_FLOAT, 0, out, 0, dim);
        return true;
    }

    private int position() { return hostBlock.get(ValueLayout.JAVA_INT, 0); }

    private void launch2D(MemorySegment fn, int gridX, int gridY, KernelParams p) {
        int err = CudaBindings.launchKernel(fn, gridX, gridY, 1, 256, 1, 1, 0, stream, p.ptrs(), MemorySegment.NULL);
        if (err != CudaBindings.CUDA_SUCCESS) throw new RuntimeException("MLA attention CUDA error: " + err);
    }

    /** A whole tensor as FP16, in its own element order (n elements). */
    private long uploadF16(CudaBufferManager bm, FloatTensor t, long n) {
        long bytes = n * 2;
        long ptr = bm.createBuffer(bytes);
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment host = temp.allocate(bytes, 16);
            float[] chunk = new float[4096];
            for (long off = 0; off < n; off += chunk.length) {
                int len = (int) Math.min(chunk.length, n - off);
                t.dequantize(chunk, 0, off, len);
                for (int i = 0; i < len; i++) host.set(ValueLayout.JAVA_SHORT, (off + i) * 2, Float.floatToFloat16(chunk[i]));
            }
            ctx.writeBuffer(ptr, host, bytes);
        }
        return ptr;
    }

    private void rmsnorm(long out, long in, long w, int size) {
        normPB.setLong(0, out).setLong(1, in).setLong(2, w).setInt(3, size);
        launch(rmsnormFunc, 1, blockSize, normShared, normPB);
    }

    private void launch(MemorySegment fn, int grid, int block, int shared, KernelParams p) {
        int err = CudaBindings.launchKernel(fn, grid, 1, 1, block, 1, 1, shared, stream, p.ptrs(), MemorySegment.NULL);
        if (err != CudaBindings.CUDA_SUCCESS) throw new RuntimeException("MLA attention CUDA error: " + err);
    }

    /** wk_b [heads][kvLoraRank][keyNope] → FP16 [heads*keyNope][kvLoraRank] (row = output element). */
    private long uploadTransposedKb(CudaBufferManager bm, FloatTensor wkB) {
        int rows = headCount * keyNope;
        long bytes = (long) rows * kvLoraRank * 2;
        long ptr = bm.createBuffer(bytes);
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment host = temp.allocate(bytes, 16);
            float[] row = new float[keyNope];
            for (int h = 0; h < headCount; h++) {
                for (int r = 0; r < kvLoraRank; r++) {
                    wkB.dequantize(row, 0, ((long) h * kvLoraRank + r) * keyNope, keyNope);
                    for (int c = 0; c < keyNope; c++) {
                        host.set(ValueLayout.JAVA_SHORT, (((long) (h * keyNope + c)) * kvLoraRank + r) * 2,
                            Float.floatToFloat16(row[c]));
                    }
                }
            }
            ctx.writeBuffer(ptr, host, bytes);
        }
        return ptr;
    }

    private long upload(CudaBufferManager bm, FloatTensor t, int n) {
        float[] w = new float[n];
        for (int i = 0; i < n; i++) w[i] = t.getFloat(i);
        return uploadArray(bm, w);
    }

    private long uploadArray(CudaBufferManager bm, float[] data) {
        long bytes = (long) data.length * Float.BYTES;
        long ptr = bm.createBuffer(bytes);
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment host = temp.allocate(ValueLayout.JAVA_FLOAT, data.length);
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_FLOAT, 0, data.length);
            ctx.writeBuffer(ptr, host, bytes);
        }
        return ptr;
    }

    @Override
    public void close() {
        for (MemorySegment[] execs : new MemorySegment[][] { graphExec, sharedExec }) {
            for (int i = 0; i < execs.length; i++) {
                if (execs[i] != null) { ctx.destroyGraphExec(execs[i]); execs[i] = null; }
            }
        }
        flash.close();
        try { ctx.freePinnedHost(hostBlock); } catch (RuntimeException ignored) { }
        if (bHost != null) try { ctx.freePinnedHost(bHost); } catch (RuntimeException ignored) { }
        if (gemm != null) gemm.close();
        arena.close();
    }
}
