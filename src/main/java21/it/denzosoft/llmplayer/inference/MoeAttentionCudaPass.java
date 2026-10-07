package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.gpu.CudaBindings;
import it.denzosoft.llmplayer.gpu.CudaBufferManager;
import it.denzosoft.llmplayer.gpu.CudaContext;
import it.denzosoft.llmplayer.gpu.Dp4aMatmul;
import it.denzosoft.llmplayer.gpu.KernelParams;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.model.Qwen3MoELayerWeights;
import it.denzosoft.llmplayer.model.Qwen3MoEWeights;
import it.denzosoft.llmplayer.tensor.CudaFloatTensor;
import it.denzosoft.llmplayer.tensor.FloatTensor;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * GPU-resident attention half of a Qwen3-MoE-family layer (Qwen3-MoE, Llama 4 MoE, GLM4 MoE);
 * the router and the experts stay with the CPU engine.
 *
 * <p>Under MoE-optimized placement every attention weight is GPU-resident but, without this pass,
 * each projection still ran through the per-tensor path — a synchronous upload, launch, sync and
 * download per matmul, about five round trips per layer — with the attention itself (and its KV
 * cache) on the CPU. Here one call runs the whole attention half on the device:
 * attn RMSNorm → Q/K/V (dp4a, input quantized once) → biases → per-head QK-norm → RoPE → KV cache
 * update → flash attention → Wo accumulated into the residual → FFN RMSNorm; one upload (residual +
 * position, 16 + 4·dim bytes) and one download (residual + FFN-normed input, 8·dim bytes).
 *
 * <p>The KV cache of the GPU layers lives on the device (the engine must treat the pass as the
 * owner of that history, see {@link GpuFailureException}). Layers whose attention weights are not
 * GPU-resident (a first-N-layers offload) keep running on the CPU with the CPU KV cache.
 */
public final class MoeAttentionCudaPass implements GpuAttentionPass {

    private final CudaContext ctx;
    private final Arena arena;
    private final MemorySegment stream;
    private final int dim, qDim, kvDim, headCount, headCountKV, headSize, halfRope, ropeType;
    private final float normEps, attnScale;
    private final int noRopeLayerInterval;
    private final int[] slidingWindowPerLayer;
    private final boolean[] onGpu;
    private final Qwen3MoEWeights weights;

    // [tokenParams (16 bytes) | x (dim) | xb (dim)] — one upload of params + x, one download of x + xb
    private final long gpuBlock, gpuTokenParams, gpuX, gpuXbOut;
    private final long gpuXn, gpuQ, gpuK, gpuV, gpuAttnOut;
    private final MemorySegment hostBlock;
    private final long[] gpuAttnNorm, gpuFfnNorm, gpuQNorm, gpuKNorm;
    private final boolean[] fullQkNorm;
    private final long[] gpuQBias, gpuKBias, gpuVBias, gpuWoBias;
    private final long[] gpuKeyCache, gpuValueCache;
    private final long[] gpuSinks;     // GPT-OSS attention sinks per layer (0 = none)
    private final long gpuCos, gpuSin;

    private final MemorySegment rmsnormFunc, perHeadFunc, ropeFunc, kvUpdateFunc, accumFunc;
    private final KernelParams normPB, perHeadPB, ropePB, kvPB, accumPB;
    private final int normShared, perHeadBlock, perHeadShared, blockSize;
    private final Dp4aMatmul mmIn;   // Q/K/V: input gpuXn
    private final Dp4aMatmul mmOut;  // Wo: input gpuAttnOut
    private final FlashAttention flash;
    private final boolean fp16Kv = "true".equals(System.getProperty("cuda.kv.fp16", "false"));

    // Per-layer CUDA graphs (docs/optimization/gpu-slower-than-cpu.md, F3): every launch between
    // the upload and the download has a fixed configuration (positions are read on the device
    // from tokenParams, constant weight pointers), so from the second call of a layer its 14-18
    // launches replay as one cuGraphLaunch. The first call runs uncaptured: it compiles the flash
    // split kernel and the FP32 fallback kernels lazily, and nothing may allocate or load a module
    // inside a capture. Disable with -Dcuda.moe.graph=false (or -Dcuda.nograph=true).
    static final boolean LAYER_GRAPHS = !"false".equals(System.getProperty("cuda.moe.graph", "true"))
        && !"true".equals(System.getProperty("cuda.nograph", "false"));

    // Batched prefill (F6): chunk buffers allocated in the constructor, before the expert cache
    // takes the free VRAM. -Dprefill.gpu.batched=false keeps the per-token prefill.
    static final int BATCH = "false".equals(System.getProperty("prefill.gpu.batched", "true")) ? 0
        : Math.max(1, Integer.getInteger("prefill.batch", 64));
    private final int maxBatch;
    private long bXXb, bX, bXb, bXn, bQ, bK, bV, bAtt, bTP;
    private MemorySegment bHost;
    private MemorySegment rmsBatchFunc, ropeBatchFunc, kvBatchFunc, biasBatchFunc;
    private KernelParams bNormPB, bRopePB, bKvPB, bBiasPB, bHeadPB;
    private GemmF16 gemm; // batched projections through cuBLAS; null: per-token dp4a loop

    // Shared expert on the GPU (F9 step 3; GLM-4.5 / Llama 4 class models): queued after the
    // attention download, so it runs while the CPU computes the routed experts, and downloaded by
    // takeSharedExpert after them, as MlaAttentionCudaPass does. Validated with teacher-forced logits
    // on a tiny GLM4-MoE model (relative difference 1.8e-7 against the CPU shared expert); no
    // full-size model of the family was available to time it. -Dmoe.attn.shared=false disables it.
    private static final boolean SHARED = !"false".equals(System.getProperty("moe.attn.shared", "true"));
    private boolean[] sharedOnGpu;
    private int sharedFfn;
    private long gpuSh, gpuShG, gpuShU;
    private Dp4aMatmul mmXb, mmSh;
    private MemorySegment siluMulFunc;
    private KernelParams siluPB;
    private int sharedLayer = -1;
    private MemorySegment[] sharedExec;
    private boolean[] sharedRanOnce;
    private final MemorySegment[] graphExec;
    private final boolean[] ranOnce;
    private boolean graphsBroken;

    /**
     * @param rope                the engine's RoPE (same tables as the CPU path)
     * @param slidingWindowPerLayer per-layer window, 0 = full attention
     */
    public MoeAttentionCudaPass(ModelConfig config, Qwen3MoEWeights weights, CudaBufferManager bm,
                                RoPE rope, int maxSeqLen, int[] slidingWindowPerLayer) {
        this.ctx = bm.getCudaContext();
        this.arena = Arena.ofShared();
        this.stream = ctx.getStream();
        this.weights = weights;
        this.dim = config.embeddingLength();
        this.headCount = config.headCount();
        this.headCountKV = config.headCountKV();
        this.headSize = config.headSize();
        this.qDim = headCount * headSize;
        this.kvDim = config.kvDim();
        this.normEps = config.normEps();
        this.halfRope = rope.getRopeDimCount() / 2;
        this.ropeType = rope.getRopeType();
        float mscale = rope.getMscale();
        this.attnScale = mscale * mscale / (float) Math.sqrt(headSize); // as Qwen3MoEInferenceEngine
        this.noRopeLayerInterval = config.noRopeLayerInterval();
        this.slidingWindowPerLayer = slidingWindowPerLayer;
        this.blockSize = (int) Math.min(256, ctx.getDeviceInfo().maxWorkGroupSize());
        int blocks = config.blockCount();
        long fb = Float.BYTES;

        gpuBlock = bm.createBuffer(16 + 3L * dim * fb); // [params | x | xb | shared expert out]
        gpuTokenParams = gpuBlock;
        gpuX = gpuBlock + 16;
        gpuXbOut = gpuX + dim * fb;
        // Page-locked, so the per-layer upload and download are true DMA transfers (a pageable
        // source is staged by the driver on the calling thread)
        hostBlock = ctx.allocPinnedHost(16 + 3L * dim * fb);
        gpuXn = bm.createBuffer(dim * fb);
        gpuQ = bm.createBuffer(qDim * fb);
        gpuK = bm.createBuffer(kvDim * fb);
        gpuV = bm.createBuffer(kvDim * fb);
        gpuAttnOut = bm.createBuffer(qDim * fb);

        onGpu = new boolean[blocks];
        gpuAttnNorm = new long[blocks]; gpuFfnNorm = new long[blocks];
        gpuQNorm = new long[blocks]; gpuKNorm = new long[blocks];
        fullQkNorm = new boolean[blocks];
        gpuQBias = new long[blocks]; gpuKBias = new long[blocks]; gpuVBias = new long[blocks]; gpuWoBias = new long[blocks];
        gpuKeyCache = new long[blocks]; gpuValueCache = new long[blocks];
        gpuSinks = new long[blocks];
        long kvBytes = (long) maxSeqLen * kvDim * (fp16Kv ? 2L : fb);
        for (int i = 0; i < blocks; i++) {
            Qwen3MoELayerWeights lw = weights.layers()[i];
            onGpu[i] = layerSupported(lw);
            if (!onGpu[i]) continue;
            try {
            // Weights upload now, not lazily inside the first forward.
            ((CudaFloatTensor) lw.wq()).getGpuWeights(); ((CudaFloatTensor) lw.wk()).getGpuWeights();
            ((CudaFloatTensor) lw.wv()).getGpuWeights(); ((CudaFloatTensor) lw.wo()).getGpuWeights();
            gpuAttnNorm[i] = upload(bm, lw.attnNorm(), dim);
            gpuFfnNorm[i] = upload(bm, lw.ffnNorm(), dim);
            if (lw.qNorm() != null) {
                // Per-head weights [headSize], or one norm over the whole projection (MiniMax-M2)
                fullQkNorm[i] = lw.qNorm().size() > headSize;
                gpuQNorm[i] = upload(bm, lw.qNorm(), fullQkNorm[i] ? (int) lw.qNorm().size() : headSize);
                gpuKNorm[i] = upload(bm, lw.kNorm(), fullQkNorm[i] ? (int) lw.kNorm().size() : headSize);
            }
            if (lw.wqBias() != null) gpuQBias[i] = upload(bm, lw.wqBias(), qDim);
            if (lw.wkBias() != null) gpuKBias[i] = upload(bm, lw.wkBias(), kvDim);
            if (lw.wvBias() != null) gpuVBias[i] = upload(bm, lw.wvBias(), kvDim);
            if (lw.woBias() != null) gpuWoBias[i] = upload(bm, lw.woBias(), dim);
            if (lw.attnSinks() != null) gpuSinks[i] = upload(bm, lw.attnSinks(), headCount);
            gpuKeyCache[i] = ctx.allocBufferChecked(kvBytes, "attention KV");
            gpuValueCache[i] = ctx.allocBufferChecked(kvBytes, "attention KV");
            ctx.fillBufferZero(gpuKeyCache[i], kvBytes);
            ctx.fillBufferZero(gpuValueCache[i], kvBytes);
            } catch (it.denzosoft.llmplayer.gpu.VramGuard.VramExhaustedException e) {
                // Out of real VRAM (F7): this layer and the following ones keep CPU attention.
                if (gpuKeyCache[i] != 0) { ctx.freeBuffer(gpuKeyCache[i]); gpuKeyCache[i] = 0; }
                for (int j = i; j < blocks; j++) onGpu[j] = false;
                System.out.println("MoE CUDA attention: layers " + i + "-" + (blocks - 1)
                    + " stay on the CPU (" + e.getMessage() + ")");
                break;
            }
        }
        gpuCos = uploadArray(bm, rope.getCosTable());
        gpuSin = uploadArray(bm, rope.getSinTable());

        rmsnormFunc = ctx.compileKernel("kernels/cuda/rmsnorm.cu", "rmsnorm_fused");
        perHeadFunc = ctx.compileKernel("kernels/cuda/rmsnorm_per_head.cu", "rmsnorm_per_head");
        ropeFunc = ctx.compileKernel("kernels/cuda/rope.cu", "rope_apply");
        kvUpdateFunc = fp16Kv
            ? ctx.compileKernel("kernels/cuda/attention_f16.cu", "kv_cache_update_f16")
            : ctx.compileKernel("kernels/cuda/attention.cu", "kv_cache_update");
        accumFunc = ctx.compileKernel("kernels/cuda/accumulate.cu", "accumulate");

        normPB = new KernelParams(arena, 5);
        normPB.setInt(3, dim).setFloat(4, normEps);
        perHeadPB = new KernelParams(arena, 4);
        perHeadPB.setInt(2, headSize).setFloat(3, normEps);
        ropePB = new KernelParams(arena, 8);
        ropePB.setLong(1, gpuCos).setLong(2, gpuSin).setInt(4, headSize).setInt(5, halfRope)
              .setLong(6, gpuTokenParams).setInt(7, ropeType);
        kvPB = new KernelParams(arena, 6);
        kvPB.setLong(2, gpuK).setLong(3, gpuV).setInt(4, kvDim).setLong(5, gpuTokenParams);
        accumPB = new KernelParams(arena, 3);
        normShared = (blockSize / 32 + 1) * Float.BYTES;
        perHeadBlock = Math.min(Math.max(32, ((headSize + 31) / 32) * 32), blockSize);
        perHeadShared = (perHeadBlock / 32 + 1) * Float.BYTES;

        mmIn = new Dp4aMatmul(ctx, bm, arena, dim);
        mmOut = new Dp4aMatmul(ctx, bm, arena, qDim);
        int nb = BATCH;
        if (nb > 0) {
            try {
                long f = Float.BYTES;
                bXXb = bm.createBuffer(2L * nb * dim * f);
                bX = bXXb;
                bXb = bXXb + (long) nb * dim * f;
                bXn = bm.createBuffer((long) nb * dim * f);
                bQ = bm.createBuffer((long) nb * qDim * f);
                bK = bm.createBuffer((long) nb * kvDim * f);
                bV = bm.createBuffer((long) nb * kvDim * f);
                bAtt = bm.createBuffer((long) nb * qDim * f);
                bTP = bm.createBuffer((long) nb * 8);
                bHost = ctx.allocPinnedHost(2L * nb * dim * f + (long) nb * 8);
                String bo = "kernels/cuda/batch_ops.cu";
                rmsBatchFunc = ctx.compileKernel(bo, "rmsnorm_batch");
                ropeBatchFunc = ctx.compileKernel(bo, "rope_apply_batch");
                kvBatchFunc = ctx.compileKernel(bo, fp16Kv ? "kv_cache_update_batch_f16" : "kv_cache_update_batch");
                biasBatchFunc = ctx.compileKernel(bo, "add_bias_batch");
                bNormPB = new KernelParams(arena, 5);
                bRopePB = new KernelParams(arena, 9);
                bKvPB = new KernelParams(arena, 6);
                bBiasPB = new KernelParams(arena, 4);
                bHeadPB = new KernelParams(arena, 4);
            } catch (RuntimeException e) {
                System.err.println("MoE GPU attention: batched prefill unavailable — " + e.getMessage());
                nb = 0;
            }
            if (nb > 0 && GemmF16.available()) {
                try {
                    java.util.List<CudaFloatTensor> ws = new java.util.ArrayList<>();
                    for (int i = 0; i < blocks; i++) {
                        if (!onGpu[i]) continue;
                        Qwen3MoELayerWeights lw = weights.layers()[i];
                        ws.add((CudaFloatTensor) lw.wq()); ws.add((CudaFloatTensor) lw.wk());
                        ws.add((CudaFloatTensor) lw.wv()); ws.add((CudaFloatTensor) lw.wo());
                    }
                    gemm = new GemmF16(ctx, bm, arena, nb, Math.max(dim, qDim), 16L << 20, ws);
                } catch (RuntimeException e) {
                    gemm = null; // per-token dp4a projections
                }
            }
        }
        maxBatch = nb;
        flash = new FlashAttention(ctx, bm, arena, headCount, headSize, maxSeqLen, fp16Kv, gpuTokenParams, Math.max(1, nb));
        flash.precompile(headCountKV, headSize);

        graphExec = new MemorySegment[blocks];
        ranOnce = new boolean[blocks];
        graphsBroken = !LAYER_GRAPHS || !ctx.isGraphApiAvailable() || !flash.graphCompatible();

        sharedOnGpu = new boolean[blocks];
        sharedExec = new MemorySegment[blocks];
        sharedRanOnce = new boolean[blocks];
        gpuSh = gpuXbOut + dim * fb;
        sharedFfn = config.expertSharedCount() * config.expertFfnLength();
        boolean anyShared = false;
        if (SHARED && sharedFfn > 0) {
            for (int i = config.leadingDenseBlockCount(); i < blocks; i++) {
                Qwen3MoELayerWeights lw = weights.layers()[i];
                sharedOnGpu[i] = onGpu[i] && lw.ffnGateShexp() instanceof CudaFloatTensor
                    && lw.ffnUpShexp() instanceof CudaFloatTensor && lw.ffnDownShexp() instanceof CudaFloatTensor;
                if (sharedOnGpu[i]) {
                    ((CudaFloatTensor) lw.ffnGateShexp()).getGpuWeights();
                    ((CudaFloatTensor) lw.ffnUpShexp()).getGpuWeights();
                    ((CudaFloatTensor) lw.ffnDownShexp()).getGpuWeights();
                    anyShared = true;
                }
            }
        }
        if (anyShared) {
            gpuShG = bm.createBuffer((long) sharedFfn * fb);
            gpuShU = bm.createBuffer((long) sharedFfn * fb);
            siluMulFunc = ctx.compileKernel("kernels/cuda/silu_mul.cu", "silu_mul");
            siluPB = new KernelParams(arena, 3);
            siluPB.setLong(0, gpuShG).setLong(1, gpuShU).setInt(2, sharedFfn);
            mmXb = new Dp4aMatmul(ctx, bm, arena, dim);
            mmSh = new Dp4aMatmul(ctx, bm, arena, Math.max(32, sharedFfn));
        }

        int n = 0;
        for (boolean b : onGpu) if (b) n++;
        System.err.println("MoE GPU attention: " + n + "/" + blocks + " layers GPU-resident"
            + (fp16Kv ? " (FP16 KV)" : "") + (graphsBroken ? "" : ", per-layer CUDA graphs"));
    }

    private static boolean layerSupported(Qwen3MoELayerWeights lw) {
        return lw.wq() instanceof CudaFloatTensor && lw.wk() instanceof CudaFloatTensor
            && lw.wv() instanceof CudaFloatTensor && lw.wo() instanceof CudaFloatTensor;
    }

    /**
     * Whether the model can use this pass: at least layer 0 has GPU-resident attention weights, head
     * size within the flash kernel's limit, and not Qwen-VL multi-axis RoPE. GPT-OSS attention sinks
     * are passed to the flash kernel.
     */
    public static boolean isSupported(ModelConfig config, Qwen3MoEWeights weights) {
        if (config.requiresCpuLayerPath() || config.ropeSections() != null) return false;
        if (config.headSize() > 512 || !FlashAttention.FLASH_ENABLED) return false;
        if (weights.layers().length == 0) return false;
        return layerSupported(weights.layers()[0]);
    }

    @Override
    public boolean isLayerOnGpu(int layer) {
        return onGpu[layer];
    }

    @Override
    public void attentionLayer(int layer, float[] x, float[] xbOut, int position) {
        // upload [position, seqLen, pad, pad | x]; the trailing readBuffer synchronises the stream,
        // so the pinned block is free again when the next call writes it
        hostBlock.set(ValueLayout.JAVA_INT, 0, position);
        hostBlock.set(ValueLayout.JAVA_INT, 4, position + 1);
        MemorySegment.copy(x, 0, hostBlock, ValueLayout.JAVA_FLOAT, 16, dim);
        ctx.writeBufferAsync(gpuBlock, hostBlock, 16 + (long) dim * Float.BYTES);

        MemorySegment g = graphExec[layer];
        if (g != null) {
            ctx.launchGraph(g);
        } else if (!graphsBroken && ranOnce[layer] && (g = capture(layer, position)) != null) {
            graphExec[layer] = g;
            ctx.launchGraph(g);
        } else {
            launches(layer, position);
            ranOnce[layer] = true;
        }

        // download [x | xb] (waits for every launch above)
        MemorySegment out = hostBlock.asSlice(16, 2L * dim * Float.BYTES);
        float[] xIn = CHECK ? x.clone() : null;
        ctx.readBuffer(gpuX, out, 2L * dim * Float.BYTES);
        MemorySegment.copy(out, ValueLayout.JAVA_FLOAT, 0, x, 0, dim);
        MemorySegment.copy(out, ValueLayout.JAVA_FLOAT, (long) dim * Float.BYTES, xbOut, 0, dim);
        if (CHECK) check(layer, position, xIn, x, xbOut);

        // Shared expert on the FFN-normed input, queued now, downloaded in takeSharedExpert
        boolean shared = sharedOnGpu[layer];
        if (shared) {
            MemorySegment sg = sharedExec[layer];
            if (sg != null) {
                ctx.launchGraph(sg);
            } else if (!graphsBroken && sharedRanOnce[layer] && (sg = captureShared(layer)) != null) {
                sharedExec[layer] = sg;
                ctx.launchGraph(sg);
            } else {
                sharedLaunches(layer);
                sharedRanOnce[layer] = true;
            }
        }
        sharedLayer = shared ? layer : -1;
    }

    private void sharedLaunches(int layer) {
        Qwen3MoELayerWeights lw = weights.layers()[layer];
        mmXb.invalidate();
        mmXb.matmul((CudaFloatTensor) lw.ffnGateShexp(), 0, gpuXbOut, gpuShG, sharedFfn, dim, false);
        mmXb.matmul((CudaFloatTensor) lw.ffnUpShexp(), 0, gpuXbOut, gpuShU, sharedFfn, dim, false);
        launch(siluMulFunc, (sharedFfn + blockSize - 1) / blockSize, blockSize, 0, siluPB);
        mmSh.invalidate();
        mmSh.matmul((CudaFloatTensor) lw.ffnDownShexp(), 0, gpuShG, gpuSh, dim, sharedFfn, false);
    }

    private MemorySegment captureShared(int layer) {
        try {
            ctx.beginCapture();
            try {
                sharedLaunches(layer);
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
            return null;
        }
    }

    /** Download the shared expert queued by the last {@link #attentionLayer} (see GpuAttentionPass). */
    @Override
    public boolean takeSharedExpert(int layer, float[] out) {
        if (sharedLayer != layer) return false;
        sharedLayer = -1;
        MemorySegment sh = hostBlock.asSlice(16 + 2L * dim * Float.BYTES, (long) dim * Float.BYTES);
        ctx.readBuffer(gpuSh, sh, (long) dim * Float.BYTES);
        MemorySegment.copy(sh, ValueLayout.JAVA_FLOAT, 0, out, 0, dim);
        return true;
    }

    /**
     * Diagnostic ({@code -Dcuda.moe.check=true}): every layer call is run a second time per launch
     * on the same input and position and compared; a graph replay must match bit for bit.
     */
    private static final boolean CHECK = "true".equals(System.getProperty("cuda.moe.check"));
    private long checks, graphChecks;

    /** Diagnostic: run the layer again on the same input (same position) and compare. */
    private void check(int layer, int position, float[] xIn, float[] x, float[] xb) {
        float[] x2 = new float[dim], xb2 = new float[dim];
        hostBlock.set(ValueLayout.JAVA_INT, 0, position);
        hostBlock.set(ValueLayout.JAVA_INT, 4, position + 1);
        MemorySegment.copy(xIn, 0, hostBlock, ValueLayout.JAVA_FLOAT, 16, dim);
        ctx.writeBufferAsync(gpuBlock, hostBlock, 16 + (long) dim * Float.BYTES);
        launches(layer, position);
        MemorySegment out = hostBlock.asSlice(16, 2L * dim * Float.BYTES);
        ctx.readBuffer(gpuX, out, 2L * dim * Float.BYTES);
        MemorySegment.copy(out, ValueLayout.JAVA_FLOAT, 0, x2, 0, dim);
        MemorySegment.copy(out, ValueLayout.JAVA_FLOAT, (long) dim * Float.BYTES, xb2, 0, dim);
        double dx = 0, dxb = 0;
        for (int i = 0; i < dim; i++) { dx = Math.max(dx, Math.abs(x[i] - x2[i])); dxb = Math.max(dxb, Math.abs(xb[i] - xb2[i])); }
        checks++;
        if (graphExec[layer] != null) graphChecks++;
        if (dx > 0 || dxb > 0) System.err.printf("[moe-check] layer %d pos %d (%s): per-launch rerun differs max|dx|=%.3g max|dxb|=%.3g%n",
            layer, position, graphExec[layer] != null ? "graph" : "launches", dx, dxb);
        if (checks % 480 == 0) System.err.println("[moe-check] " + checks + " layer calls re-run per launch (" + graphChecks + " of them replayed a graph): no difference unless reported above");
    }

    /** Capture {@code layer}'s launches into an executable graph; null (and per-launch for good) on failure. */
    private MemorySegment capture(int layer, int position) {
        try {
            ctx.beginCapture();
            try {
                launches(layer, position);
            } catch (RuntimeException e) {
                try { ctx.endCapture(); } catch (RuntimeException ignored) { } // clear the invalidated capture
                throw e;
            }
            MemorySegment graph = ctx.endCapture();
            MemorySegment exec = ctx.instantiateGraph(graph);
            ctx.destroyGraph(graph);
            return exec;
        } catch (RuntimeException e) {
            graphsBroken = true;
            System.err.println("MoE GPU attention: per-layer graph capture failed (" + e.getMessage()
                + ") — running per launch");
            return null;
        }
    }

    /** Every launch of {@code layer}'s attention half, from the uploaded x to x and xb on the device. */
    private void launches(int layer, int position) {
        Qwen3MoELayerWeights lw = weights.layers()[layer];
        // attn RMSNorm: gpuXn = norm(gpuX)
        rmsnorm(gpuXn, gpuX, gpuAttnNorm[layer]);
        mmIn.invalidate();
        mmIn.matmul((CudaFloatTensor) lw.wq(), 0, gpuXn, gpuQ, qDim, dim, false);
        mmIn.matmul((CudaFloatTensor) lw.wk(), 0, gpuXn, gpuK, kvDim, dim, false);
        mmIn.matmul((CudaFloatTensor) lw.wv(), 0, gpuXn, gpuV, kvDim, dim, false);
        if (gpuQBias[layer] != 0) accumulate(gpuQ, gpuQBias[layer], qDim);
        if (gpuKBias[layer] != 0) accumulate(gpuK, gpuKBias[layer], kvDim);
        if (gpuVBias[layer] != 0) accumulate(gpuV, gpuVBias[layer], kvDim);
        if (gpuQNorm[layer] != 0 && fullQkNorm[layer]) {
            // One "head" spanning the whole projection
            perHeadPB.setLong(0, gpuQ).setLong(1, gpuQNorm[layer]).setInt(2, headCount * headSize);
            launch(perHeadFunc, 1, blockSize, (blockSize / 32 + 1) * Float.BYTES, perHeadPB);
            perHeadPB.setLong(0, gpuK).setLong(1, gpuKNorm[layer]).setInt(2, headCountKV * headSize);
            launch(perHeadFunc, 1, blockSize, (blockSize / 32 + 1) * Float.BYTES, perHeadPB);
            perHeadPB.setInt(2, headSize);
        } else if (gpuQNorm[layer] != 0) {
            perHeadPB.setLong(0, gpuQ).setLong(1, gpuQNorm[layer]);
            launch(perHeadFunc, headCount, perHeadBlock, perHeadShared, perHeadPB);
            perHeadPB.setLong(0, gpuK).setLong(1, gpuKNorm[layer]);
            launch(perHeadFunc, headCountKV, perHeadBlock, perHeadShared, perHeadPB);
        }
        // halfRope == 0: no rotated dimension (a zero-sized grid is an invalid launch)
        if (halfRope > 0 && (noRopeLayerInterval == 0 || (layer % noRopeLayerInterval) != (noRopeLayerInterval - 1))) {
            ropePB.setLong(0, gpuQ).setInt(3, headCount);
            launch(ropeFunc, (headCount * halfRope + blockSize - 1) / blockSize, blockSize, 0, ropePB);
            ropePB.setLong(0, gpuK).setInt(3, headCountKV);
            launch(ropeFunc, (headCountKV * halfRope + blockSize - 1) / blockSize, blockSize, 0, ropePB);
        }
        kvPB.setLong(0, gpuKeyCache[layer]).setLong(1, gpuValueCache[layer]);
        launch(kvUpdateFunc, (kvDim + blockSize - 1) / blockSize, blockSize, 0, kvPB);
        flash.launch(stream, gpuAttnOut, gpuQ, gpuKeyCache[layer], gpuValueCache[layer],
            headCountKV, headSize, kvDim, slidingWindowPerLayer[layer], attnScale, 0f, position, gpuSinks[layer]);
        mmOut.invalidate();
        // residual: gpuX += Wo · attnOut (+ bias)
        mmOut.matmul((CudaFloatTensor) lw.wo(), 0, gpuAttnOut, gpuX, dim, qDim, true);
        if (gpuWoBias[layer] != 0) accumulate(gpuX, gpuWoBias[layer], dim);
        // FFN RMSNorm: gpuXbOut = norm(gpuX)
        rmsnorm(gpuXbOut, gpuX, gpuFfnNorm[layer]);
    }

    @Override
    public int maxBatchTokens() { return maxBatch; }

    /**
     * Batched prefill of one layer for {@code n} tokens (see {@link GpuAttentionPass}). The norms,
     * QK-norm, RoPE, KV writes and attention run as one launch each over the chunk (one causal flash
     * launch); the projections run per token with the decode kernels (dp4a, input quantized once per
     * token), so they match the per-token path's arithmetic.
     */
    @Override
    public void attentionLayerBatch(int layer, float[][] x, float[][] xbOut, int basePos, int n) {
        if (n < 1 || n > maxBatch) throw new IllegalArgumentException("batch " + n + " > " + maxBatch);
        sharedLayer = -1; // the batched prefill computes the shared expert on the CPU
        Qwen3MoELayerWeights lw = weights.layers()[layer];
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

        rmsnormB(bXn, bX, gpuAttnNorm[layer], dim, n);
        mmIn.invalidate(); // the chunk buffers were rewritten: never reuse a cached quantization
        mmOut.invalidate();
        if (gemm != null) {
            gemm.toF16(bXn, n * dim);
            gemm.gemm((CudaFloatTensor) lw.wq(), qDim, dim, bQ, qDim, n, false);
            gemm.gemm((CudaFloatTensor) lw.wk(), kvDim, dim, bK, kvDim, n, false);
            gemm.gemm((CudaFloatTensor) lw.wv(), kvDim, dim, bV, kvDim, n, false);
        } else {
            for (int t = 0; t < n; t++) {
                long in = bXn + (long) t * dim * f;
                mmIn.matmul((CudaFloatTensor) lw.wq(), 0, in, bQ + (long) t * qDim * f, qDim, dim, false);
                mmIn.matmul((CudaFloatTensor) lw.wk(), 0, in, bK + (long) t * kvDim * f, kvDim, dim, false);
                mmIn.matmul((CudaFloatTensor) lw.wv(), 0, in, bV + (long) t * kvDim * f, kvDim, dim, false);
            }
        }
        mmIn.invalidate();
        if (gpuQBias[layer] != 0) biasB(bQ, gpuQBias[layer], qDim, n);
        if (gpuKBias[layer] != 0) biasB(bK, gpuKBias[layer], kvDim, n);
        if (gpuVBias[layer] != 0) biasB(bV, gpuVBias[layer], kvDim, n);
        if (gpuQNorm[layer] != 0) {
            if (fullQkNorm[layer]) {
                headNormB(bQ, gpuQNorm[layer], qDim, n);
                headNormB(bK, gpuKNorm[layer], kvDim, n);
            } else {
                headNormB(bQ, gpuQNorm[layer], headSize, n * headCount);
                headNormB(bK, gpuKNorm[layer], headSize, n * headCountKV);
            }
        }
        if (halfRope > 0 && (noRopeLayerInterval == 0 || (layer % noRopeLayerInterval) != (noRopeLayerInterval - 1))) {
            ropeB(bQ, headCount, qDim, n);
            ropeB(bK, headCountKV, kvDim, n);
        }
        bKvPB.setLong(0, gpuKeyCache[layer]).setLong(1, gpuValueCache[layer]).setLong(2, bK).setLong(3, bV)
             .setInt(4, kvDim).setLong(5, bTP);
        launch2(kvBatchFunc, (kvDim + blockSize - 1) / blockSize, n, blockSize, 0, bKvPB);
        flash.launchBatch(stream, bAtt, bQ, gpuKeyCache[layer], gpuValueCache[layer], headCountKV, headSize, kvDim,
            slidingWindowPerLayer[layer], attnScale, 0f, n, bTP, qDim, gpuSinks[layer]);
        if (gemm != null) {
            gemm.toF16(bAtt, n * qDim);
            gemm.gemm((CudaFloatTensor) lw.wo(), dim, qDim, bX, dim, n, true);
        } else {
            for (int t = 0; t < n; t++) {
                mmOut.matmul((CudaFloatTensor) lw.wo(), 0, bAtt + (long) t * qDim * f, bX + (long) t * dim * f, dim, qDim, true);
            }
        }
        mmOut.invalidate();
        if (gpuWoBias[layer] != 0) biasB(bX, gpuWoBias[layer], dim, n);
        rmsnormB(bXb, bX, gpuFfnNorm[layer], dim, n);

        ctx.readBufferAsync(bX, bHost, xBytes);
        ctx.readBufferAsync(bXb, bHost.asSlice(xBytes, xBytes), xBytes);
        ctx.finish();
        for (int t = 0; t < n; t++) {
            MemorySegment.copy(bHost, ValueLayout.JAVA_FLOAT, (long) t * dim * f, x[t], 0, dim);
            MemorySegment.copy(bHost, ValueLayout.JAVA_FLOAT, xBytes + (long) t * dim * f, xbOut[t], 0, dim);
        }
        mmIn.invalidate();
        mmOut.invalidate();
    }

    private void rmsnormB(long out, long in, long w, int size, int n) {
        bNormPB.setLong(0, out).setLong(1, in).setLong(2, w).setInt(3, size).setFloat(4, normEps);
        launch2(rmsBatchFunc, n, 1, blockSize, normShared, bNormPB);
    }

    /** RMSNorm over {@code rows} consecutive rows of {@code size} with shared weights (per-head or whole-projection QK-norm). */
    private void headNormB(long vec, long w, int size, int rows) {
        bHeadPB.setLong(0, vec).setLong(1, w).setInt(2, size).setFloat(3, normEps);
        int block = size <= perHeadBlock ? perHeadBlock : blockSize;
        launch2(perHeadFunc, rows, 1, block, (block / 32 + 1) * Float.BYTES, bHeadPB);
    }

    private void ropeB(long vec, int heads, int stride, int n) {
        bRopePB.setLong(0, vec).setLong(1, gpuCos).setLong(2, gpuSin).setInt(3, heads).setInt(4, headSize)
               .setInt(5, halfRope).setLong(6, bTP).setInt(7, ropeType).setInt(8, stride);
        launch2(ropeBatchFunc, (heads * halfRope + blockSize - 1) / blockSize, n, blockSize, 0, bRopePB);
    }

    private void biasB(long y, long bias, int size, int n) {
        bBiasPB.setLong(0, y).setLong(1, bias).setInt(2, size).setInt(3, size * n);
        launch2(biasBatchFunc, (size * n + blockSize - 1) / blockSize, 1, blockSize, 0, bBiasPB);
    }

    private void launch2(MemorySegment fn, int gridX, int gridY, int block, int shared, KernelParams p) {
        int err = CudaBindings.launchKernel(fn, gridX, gridY, 1, block, 1, 1, shared, stream, p.ptrs(), MemorySegment.NULL);
        if (err != CudaBindings.CUDA_SUCCESS) throw new RuntimeException("MoE attention CUDA error: " + err);
    }

    private void rmsnorm(long out, long in, long w) {
        normPB.setLong(0, out).setLong(1, in).setLong(2, w);
        launch(rmsnormFunc, 1, blockSize, normShared, normPB);
    }

    private void accumulate(long y, long x, int n) {
        accumPB.setLong(0, y).setLong(1, x).setInt(2, n);
        launch(accumFunc, (n + blockSize - 1) / blockSize, blockSize, 0, accumPB);
    }

    private void launch(MemorySegment fn, int grid, int block, int shared, KernelParams p) {
        int err = CudaBindings.launchKernel(fn, grid, 1, 1, block, 1, 1, shared, stream, p.ptrs(), MemorySegment.NULL);
        if (err != CudaBindings.CUDA_SUCCESS) throw new RuntimeException("MoE attention CUDA error: " + err);
    }

    private long upload(CudaBufferManager bm, FloatTensor t, int n) {
        float[] w = new float[n];
        for (int i = 0; i < n; i++) w[i] = t.getFloat(i);
        return uploadArray(bm, w);
    }

    private long uploadArray(CudaBufferManager bm, float[] data) {
        long bytes = (long) data.length * Float.BYTES;
        // An empty table (no rotated dimensions, e.g. a rope dimension of 1) still needs a valid
        // pointer: cuMemAlloc rejects 0 bytes with CUDA_ERROR_INVALID_VALUE
        long ptr = bm.createBuffer(Math.max(bytes, Float.BYTES));
        if (bytes == 0) return ptr;
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment host = temp.allocate(ValueLayout.JAVA_FLOAT, data.length);
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_FLOAT, 0, data.length);
            ctx.writeBuffer(ptr, host, bytes);
        }
        return ptr;
    }

    @Override
    public void close() {
        for (int i = 0; i < graphExec.length; i++) {
            if (graphExec[i] != null) { ctx.destroyGraphExec(graphExec[i]); graphExec[i] = null; }
            if (sharedExec[i] != null) { ctx.destroyGraphExec(sharedExec[i]); sharedExec[i] = null; }
        }
        flash.close();
        try { ctx.freePinnedHost(hostBlock); } catch (RuntimeException ignored) { }
        if (bHost != null) try { ctx.freePinnedHost(bHost); } catch (RuntimeException ignored) { }
        if (gemm != null) gemm.close();
        arena.close();
    }
}
