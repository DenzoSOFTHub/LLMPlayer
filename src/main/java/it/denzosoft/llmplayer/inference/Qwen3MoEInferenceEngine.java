package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.ModelArchitecture;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.model.Qwen3MoELayerWeights;
import it.denzosoft.llmplayer.model.Qwen3MoEWeights;
import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Inference engine for Qwen3 MoE architecture.
 *
 * Combines standard GQA attention (with per-head QK normalization, like Qwen3 dense)
 * with Mixture-of-Experts FFN for MoE layers and dense SwiGLU FFN for leading dense blocks.
 *
 * Forward pass per layer:
 * 1. RMSNorm -> GQA Attention (with QK-norm + RoPE) -> Residual
 * 2. RMSNorm -> (Dense SwiGLU FFN or MoE FFN) -> Residual
 */
public class Qwen3MoEInferenceEngine {

    private final ModelConfig config;
    private final Qwen3MoEWeights weights;
    private final RoPE rope;
    private final float[][] cachedAttnNorm;
    private final float[][] cachedFfnNorm;
    private final float[][] cachedQNorm;
    private final float[][] cachedKNorm;
    private final float[] outputNormCache;
    private final int maxSeqLen;
    private final boolean isGptOss;
    // GLM4-MoE: sigmoid gating with the exp_probs_b selection bias (llama.cpp build_moe_ffn)
    private final boolean sigmoidRouting;
    private final int slidingWindow; // ISWA: 0 = disabled, >0 = window size for SWA layers
    private final int noRopeLayerInterval; // Llama4 iRoPE: 0 = all layers use RoPE

    // Cached attention sinks per layer (GPT-OSS): float[blockCount][headCount]
    private final float[][] cachedAttnSinks;

    // Expert GPU cache (loaded via reflection from java21, null if unavailable)
    private GpuExpertCache expertGpuCache;
    private int currentLayer; // tracks current layer for GPU cache keying
    private int currentPosition;

    // SSD-streaming expert RAM cache (models larger than RAM), or null when the model is resident.
    private it.denzosoft.llmplayer.tensor.ExpertCache expertCache;
    /** Per-expert views for the row-kernel expert path and batched prefill (see {@link ExpertViews}). */
    private final ExpertViews expertViews;
    private boolean cacheLayerReady;

    // Phase timing: the attention and expert phases are always timed (the CUDA clock keeper reads
    // them through GpuActivity); the rest, and the printed profile, only with -Dcpu.profile=true.
    private static final int P_ATTN_NORM = 0, P_ATTN = 1, P_FFN_NORM = 2, P_DENSE = 3, P_MOE = 4,
        P_RESIDUAL = 5, P_OUTPUT = 6;
    private final DecodeProfile prof = new DecodeProfile("Qwen3MoE", new String[] {
        "attn_norm", "attn(GQA)", "ffn_norm", "dense_ffn", "moe_ffn", "residual", "output" }, P_ATTN, P_MOE, P_OUTPUT);
    private final boolean cpuProfile = prof.detailed;
    private final java.util.function.Supplier<String> cacheStatsSupplier = this::getExpertCacheStats;
    private OutputRouter outputRouter;

    // Phase 2.2a: MoE routing-frequency instrumentation (opt-in, -Dmoe.routing.stats=true).
    private final boolean routingStats = "true".equals(System.getProperty("moe.routing.stats", "false"));
    private long[][] expertHits;     // [layer][expert] selection counts
    private long routingDecisions;   // total expert selections counted

    // Phase 2.2b diagnostic: compare the GPU expert-cache output to the CPU path on the first MoE call.
    private final boolean debugCache = "true".equals(System.getProperty("moe.cache.debug", "false"));
    private boolean debugCacheDone;

    public Qwen3MoEInferenceEngine(ModelConfig config, Qwen3MoEWeights weights, int maxSeqLen,
                                    float[] ropeFreqFactors) {
        this.config = config;
        this.weights = weights;
        this.maxSeqLen = maxSeqLen;
        this.isGptOss = config.architecture() == ModelArchitecture.GPT_OSS;
        this.sigmoidRouting = (config.architecture() == ModelArchitecture.GLM4
            || config.architecture() == ModelArchitecture.MINIMAX_M2) && config.expertGatingFunc() == 2;
        this.slidingWindow = config.slidingWindow();
        this.noRopeLayerInterval = config.noRopeLayerInterval();
        this.expertViews = new ExpertViews(config.blockCount(), Math.max(1, config.expertCount()));

        if (routingStats && config.expertCount() > 0) {
            expertHits = new long[config.blockCount()][config.expertCount()];
            Runtime.getRuntime().addShutdownHook(new Thread(this::printRoutingStats));
            System.err.println("MoE routing stats: enabled (-Dmoe.routing.stats) — summary printed at exit");
        }

        int ropeDimCount = config.ropeDimensionCount();
        RoPE.YarnParams yarnParams = null;
        if (config.ropeScalingFactor() > 1.0f) {
            yarnParams = new RoPE.YarnParams(
                config.ropeScalingFactor(), config.ropeOrigContextLength(),
                config.yarnLogMultiplier());
        }
        this.rope = new RoPE(config.headSize(), ropeDimCount, maxSeqLen, config.ropeFreqBase(),
            config.ropeType(), ropeFreqFactors, yarnParams);

        int dim = config.embeddingLength();
        int headSize = config.headSize();
        int blockCount = config.blockCount();

        // Pre-cache norm weights
        this.cachedAttnNorm = new float[blockCount][];
        this.cachedFfnNorm = new float[blockCount][];
        this.cachedQNorm = new float[blockCount][];
        this.cachedKNorm = new float[blockCount][];
        for (int i = 0; i < blockCount; i++) {
            cachedAttnNorm[i] = RMSNorm.cacheWeights(weights.layers()[i].attnNorm(), dim);
            cachedFfnNorm[i] = RMSNorm.cacheWeights(weights.layers()[i].ffnNorm(), dim);
            if (weights.layers()[i].qNorm() != null) {
                // Per-head norm ([headSize] weights), or over the whole projection when the weight
                // spans every head (MiniMax-M2: attn_q_norm [heads*headSize], as OLMo-2)
                FloatTensor qn = weights.layers()[i].qNorm(), kn = weights.layers()[i].kNorm();
                cachedQNorm[i] = RMSNorm.cacheWeights(qn, qn.size() > headSize ? (int) qn.size() : headSize);
                cachedKNorm[i] = RMSNorm.cacheWeights(kn, kn.size() > headSize ? (int) kn.size() : headSize);
            }
        }

        // Cache attention sinks (GPT-OSS)
        this.cachedAttnSinks = new float[blockCount][];
        for (int i = 0; i < blockCount; i++) {
            if (weights.layers()[i].attnSinks() != null) {
                int headCount = config.headCount();
                cachedAttnSinks[i] = new float[headCount];
                for (int h = 0; h < headCount; h++) {
                    cachedAttnSinks[i][h] = weights.layers()[i].attnSinks().getFloat(h);
                }
            }
        }

        this.outputRouter = new OutputRouter(weights.output(), "Output");
        this.outputNormCache = new float[dim];
        for (int i = 0; i < dim; i++) {
            outputNormCache[i] = weights.outputNorm().getFloat(i);
        }
    }

    /**
     * Initialize expert GPU cache for accelerated MoE FFN computation.
     * Called from LLMEngine when CUDA is active and the model is MoE.
     * @param cudaContext the CudaContext object (from java21)
     * @param maxCacheBytes maximum VRAM to use for expert caching
     */
    public void initExpertGpuCache(Object cudaContext, long maxCacheBytes) {
        // Every expert quant type with an FP32-input kernel is eligible. The K-quant paths used to
        // be opt-in because they produced wrong output: the cache took the gate tensor's geometry
        // and kernel for up and down too, and Q4_K_M ships ffn_down_exps as Q6_K in many layers.
        // With per-projection geometry the output matches the CPU expert path token for token
        // (Qwen3-Coder-30B Q4_K_M), at ~1.1-2x the decode speed.
        FloatTensor[][] ex = new FloatTensor[weights.layers().length][];
        for (int i = 0; i < ex.length; i++) {
            Qwen3MoELayerWeights lw = weights.layers()[i];
            if (lw.ffnGateExps() != null) ex[i] = new FloatTensor[] { lw.ffnGateExps(), lw.ffnUpExps(), lw.ffnDownExps() };
        }
        expertGpuCache = GpuExpertCache.create(cudaContext, maxCacheBytes, ex, config.expertFfnLength(),
            config.embeddingLength(), config.expertCount());
    }

    /** Attach the SSD-streaming expert cache (models larger than RAM). */
    public void setExpertCache(it.denzosoft.llmplayer.tensor.ExpertCache cache) {
        this.expertCache = cache;
    }

    /** The GPU expert cache, or null (for metrics and close). */
    public GpuExpertCache getGpuExpertCache() { return expertGpuCache != null ? expertGpuCache : parkedCache; }

    /**
     * Get expert GPU cache statistics, or null if cache not active.
     */
    public String getExpertCacheStats() {
        if (expertGpuCache == null) return null;
        try {
            return expertGpuCache.getStats();
        } catch (Exception e) {
            return null;
        }
    }

    // GPU-resident attention (MoE-optimized placement); null when unavailable or disabled
    private volatile GpuAttentionPass gpuAttention;

    /**
     * Run the attention half of every layer with GPU-resident attention weights on the device
     * (MoeAttentionCudaPass): one upload and one download per layer instead of a synchronous round
     * trip per projection, and the attention itself (with its KV cache) on the GPU. The experts
     * stay on the CPU. Disable with {@code -Dmoe.gpu.attention=false}.
     */
    public void tryInitGpuAttention(Object bufferManager) {
        if ("false".equals(System.getProperty("moe.gpu.attention", "true"))) return;
        try {
            Class<?> cls = Class.forName("it.denzosoft.llmplayer.inference.MoeAttentionCudaPass");
            java.lang.reflect.Method isSup = cls.getMethod("isSupported", ModelConfig.class, Qwen3MoEWeights.class);
            if (!(Boolean) isSup.invoke(null, config, weights)) return;
            int blocks = config.blockCount();
            int[] sw = new int[blocks];
            for (int i = 0; i < blocks; i++) sw[i] = (slidingWindow > 0 && !isSwaGlobalLayer(i)) ? slidingWindow : 0;
            gpuAttention = (GpuAttentionPass) cls.getConstructor(ModelConfig.class, Qwen3MoEWeights.class,
                    bufferManager.getClass(), RoPE.class, int.class, int[].class)
                .newInstance(config, weights, bufferManager, rope, maxSeqLen, sw);
        } catch (Throwable e) {
            System.err.println("MoE GPU attention: unavailable — " + GpuFailureException.describe(e));
            gpuAttention = null;
        }
    }

    /**
     * Upload (and compile the kernels of) every GPU tensor the CPU-side code reaches through the
     * per-tensor path — shared experts, dense leading layers — by running one matmul on each. Done
     * before the expert cache is sized: lazily, on the first forward, they would be allocated after
     * the cache has taken the free VRAM, which under WDDM means shared system memory.
     */
    public void warmGpuTensors() {
        int dim = config.embeddingLength();
        int sharedFfn = config.expertSharedCount() * config.expertFfnLength();
        int ffn = config.intermediateSize();
        for (int i = 0; i < weights.layers().length; i++) {
            Qwen3MoELayerWeights lw = weights.layers()[i];
            warmGpu(lw.ffnGateShexp(), sharedFfn, dim);
            warmGpu(lw.ffnUpShexp(), sharedFfn, dim);
            warmGpu(lw.ffnDownShexp(), dim, sharedFfn);
            if (i < config.leadingDenseBlockCount()) {
                warmGpu(lw.wGate(), ffn, dim);
                warmGpu(lw.wUp(), ffn, dim);
                warmGpu(lw.wDown(), dim, ffn);
            }
        }
    }

    private static void warmGpu(FloatTensor t, int rows, int cols) {
        if (t == null || !t.isGpuResident() || rows <= 0 || cols <= 0) return;
        try {
            t.matmulParallel(new float[cols], new float[rows], rows, cols);
        } catch (RuntimeException ignored) {
            // the tensor falls back to its CPU twin on its own
        }
    }

    private GpuAttentionPass parkedAttention;
    private GpuExpertCache parkedCache;

    /**
     * Park (or restore) the GPU attention pass without closing it, for the placement calibrator:
     * while parked the attention runs on the CPU path. A sequence must restart at position 0.
     */
    public synchronized void setGpuAttentionParked(boolean park) {
        if (park && gpuAttention != null) { parkedAttention = gpuAttention; gpuAttention = null; }
        else if (!park && parkedAttention != null) { gpuAttention = parkedAttention; parkedAttention = null; }
    }

    public boolean hasParkedAttention() { return parkedAttention != null; }

    /** Park (or restore) the GPU expert cache (placement calibrator). */
    public synchronized void setExpertGpuCacheParked(boolean park) {
        if (park && expertGpuCache != null) { parkedCache = expertGpuCache; expertGpuCache = null; }
        else if (!park && parkedCache != null) { expertGpuCache = parkedCache; parkedCache = null; }
    }

    /** True when a GPU-resident attention pass (which owns the device KV cache) is active. */
    public boolean hasGpuForwardPass() {
        return gpuAttention != null;
    }

    private void gpuAttentionFailed(RuntimeException e, int position, int layer) {
        GpuAttentionPass failed = gpuAttention;
        gpuAttention = null;
        System.err.println("MoE GPU attention failed, disabling it — " + GpuFailureException.describe(e));
        try { failed.close(); } catch (Exception ignored) { }
        it.denzosoft.llmplayer.gpu.GpuActivity.gpuPathDisabled();
        // The device held the KV cache of every earlier (position, layer) it ran — including the
        // earlier layers at this position, and in layer-outer prefill the earlier tokens of this
        // layer. Only the very first GPU call of a sequence (layer 0, position 0) can move to the CPU.
        if (position > 0 || layer > 0) {
            throw new GpuFailureException("MoE GPU attention failed at layer " + layer + ", position " + position, e);
        }
    }

    public Qwen3MoEState createState(int maxSeqLen) {
        return new Qwen3MoEState(config, maxSeqLen);
    }

    // Mirrors Attention.isGlobalLayer for the MoE-routed architectures.
    // Returns true when this layer should use full attention (no sliding window).
    private boolean isSwaGlobalLayer(int layer) {
        ModelArchitecture arch = config.architecture();
        if (arch == ModelArchitecture.GPT_OSS) {
            // GPT-OSS: alternating, even = global, odd = local (SWA).
            return layer % 2 == 0;
        }
        // QWEN3MOE / LLAMA4 MoE / GLM4 MoE: no SWA pattern today — if slidingWindow > 0 is set
        // in GGUF metadata, treat every layer as windowed (same as the pre-refactor default).
        return false;
    }

    public float[] forward(Qwen3MoEState state, int token, int position) {
        return forwardInternal(state, token, position, true);
    }

    public void forwardNoOutput(Qwen3MoEState state, int token, int position) {
        forwardInternal(state, token, position, false);
    }

    private float[] forwardInternal(Qwen3MoEState state, int token, int position, boolean computeLogits) {
        embedToken(state, token);
        for (int layer = 0; layer < config.blockCount(); layer++) {
            forwardLayer(state, layer, position);
        }
        if (!computeLogits) return null;
        return outputProjection(state);
    }

    /** Tokens per layer-outer prefill chunk; see {@link #forwardPrefill}. */
    private static final int PREFILL_BATCH = Integer.getInteger("prefill.batch", 64);
    private static final boolean PREFILL_BATCHED =
        !"false".equals(System.getProperty("prefill.batched", "true"));

    /**
     * Prefill positions {@code fromPos..toPos-1}, driving the layers in the outer loop and the
     * tokens in the inner loop, and return the logits for the last position.
     *
     * This computes exactly the same values in the same per-(layer, token) order as calling
     * {@code forward} once per token — token *t* at layer *L* still reads its own residual stream
     * from layer *L-1* and attends over KV[L][0..t], which are written by the tokens processed
     * before it in the same layer. Only the loop nesting changes, so output is bit-identical.
     *
     * What changes is the working set. Token-outer order walks all 48 layers for one token before
     * moving on, so keeping an expert resident across tokens requires the cache to hold
     * layers x top-K slots at once; on a model streamed from SSD most of them are evicted before the
     * next token needs them and get re-read. Layer-outer order collapses the working set to the
     * union of experts selected by the tokens of one chunk at one layer, which the existing cache
     * absorbs easily — so each expert is read roughly once per layer instead of once per
     * (layer, token). This is the "batch-union" effect that makes batched prefill fast, obtained by
     * reordering alone, with no batched matmul.
     *
     * Chunked at {@link #PREFILL_BATCH} tokens to bound both the residual-stream buffer and the
     * per-layer expert union. Disable with {@code -Dprefill.batched=false}.
     */
    // F4: whether any layer weight is GPU-resident (scanned once). The batched CPU prefill used to
    // be gated on the matmul pool, which GPU init switched off; the pool now stays on.
    private volatile Boolean layersGpuResident;

    private boolean layersGpuResident() {
        Boolean b = layersGpuResident;
        if (b == null) layersGpuResident = b = it.denzosoft.llmplayer.tensor.FloatTensor.anyGpuResident(weights.layers());
        // GPU matmuls switched off (the placement calibrator's CPU candidate): the tensors run
        // on their CPU twins, so the batched prefill applies as in a CPU-only run
        return b && it.denzosoft.llmplayer.tensor.FloatTensor.gpuMatmulEnabled();
    }

    public float[] forwardPrefill(Qwen3MoEState state, int[] tokens, int fromPos, int toPos) {
        int count = toPos - fromPos;
        if (count <= 0) return null;
        prof.startGeneration(); // a second generation in the same process starts clean too
        if (!PREFILL_BATCHED || count == 1) {
            float[] logits = null;
            for (int i = fromPos; i < toPos; i++) {
                if (i < toPos - 1) forwardInternal(state, tokens[i], i, false);
                else logits = forwardInternal(state, tokens[i], i, true);
            }
            return logits;
        }
        // Batched prefill: the CPU layers attend over the CPU KV cache; the GPU-resident attention
        // layers run through the pass's batched attention (their KV lives on the device), and the
        // routed experts run batched on the CPU (the GPU expert cache only counts the routing).
        // A pass without batched attention keeps the layer-outer per-token path below.
        GpuAttentionPass gpu = gpuAttention;
        // Without an attention pass, a first-N placement (explicit --gpu-layers) keeps its GPU
        // layers on the per-token path (F4 residency gate).
        if (ExpertViews.active() && (gpu == null ? !layersGpuResident() : gpu.maxBatchTokens() > 0)) {
            return prefillBatched(state, tokens, fromPos, toPos);
        }
        // The per-token path below already runs the routed experts through the dot kernels decode
        // uses, so C2 code is in place before the first decoded token; this call only moves the
        // compilation ahead of the prompt. Measured like for like it is worth at most 5-8% of the
        // first window (docs/optimization/gpu-slower-than-cpu.md, section 6.2); an earlier
        // "3111 vs 1414 ms/token" comparison here was the profiler folding the prompt into the
        // decode averages.
        warmDecodeKernels();

        int dim = config.embeddingLength();
        int blockCount = config.blockCount();
        long tStart = System.nanoTime();
        for (int base = fromPos; base < toPos; base += PREFILL_BATCH) {
            int n = Math.min(PREFILL_BATCH, toPos - base);
            float[][] xs = new float[n][];
            for (int t = 0; t < n; t++) {
                embedToken(state, tokens[base + t]);
                xs[t] = state.x.clone();
            }
            for (int layer = 0; layer < blockCount; layer++) {
                for (int t = 0; t < n; t++) {
                    System.arraycopy(xs[t], 0, state.x, 0, dim);
                    forwardLayer(state, layer, base + t);
                    System.arraycopy(state.x, 0, xs[t], 0, dim);
                }
            }
            // Carry the last token of the final chunk into the output projection.
            System.arraycopy(xs[n - 1], 0, state.x, 0, dim);
        }
        if (cpuProfile) {
            System.out.printf("[prefill-profile Qwen3MoE] %d tokens per token (layer-outer): %.0f ms (%.1f ms per token)%n",
                count, (System.nanoTime() - tStart) / 1e6, (System.nanoTime() - tStart) / 1e6 / count);
        }
        return outputProjection(state);
    }

    // ==================== Batched prefill ====================

    // Indices into Qwen3MoEState.prefillBuffers
    private static final int B_X = 0, B_XN = 1, B_XB = 2, B_Q = 3, B_K = 4, B_V = 5, B_ATT = 6,
        B_HB = 7, B_HB2 = 8, B_SHOUT = 9, B_EGATE = 10, B_EUP = 11, B_EOUT = 12;

    /**
     * Layer-outer prefill with multi-token matmuls, on the CPU path. Per chunk and layer: the
     * attention projections (Q, K, V, O) and the shared-expert and dense FFNs run as batched
     * matmuls over the chunk; {@link #attentionCore} runs token by token in position order; and
     * each routed expert runs once per projection over all the chunk's tokens routed to it
     * ({@code matmulRowsBatch} on its per-expert view), so an expert's weights are read and
     * dequantised once per chunk instead of once per (token, slot). Routing, the per-token sum of
     * the expert outputs in slot order and the residuals are unchanged; only the summation order
     * inside the matmul kernels differs from the one-token path.
     */
    private float[] prefillBatched(Qwen3MoEState state, int[] tokens, int fromPos, int toPos) {
        warmDecodeKernels();
        int dim = config.embeddingLength();
        int cap = ExpertViews.prefillChunk(expertCache, PREFILL_BATCH);
        GpuAttentionPass gpu0 = gpuAttention;
        if (gpu0 != null) cap = Math.min(cap, gpu0.maxBatchTokens());
        float[][][] b = prefillBuffers(state, cap);
        long attnNs = 0, ffnNs = 0, t0;
        for (int base = fromPos; base < toPos; base += cap) {
            int n = Math.min(cap, toPos - base);
            for (int t = 0; t < n; t++) {
                embedToken(state, tokens[base + t]);
                System.arraycopy(state.x, 0, b[B_X][t], 0, dim);
            }
            for (int layer = 0; layer < config.blockCount(); layer++) {
                t0 = System.nanoTime();
                boolean normed = attentionBatch(state, b, layer, base, n);
                long t1 = System.nanoTime();
                ffnBatch(state, b, layer, n, normed);
                attnNs += t1 - t0;
                ffnNs += System.nanoTime() - t1;
            }
            if (base + n == toPos) System.arraycopy(b[B_X][n - 1], 0, state.x, 0, dim);
        }
        if (cpuProfile) {
            int total = toPos - fromPos;
            System.out.printf("[prefill-profile Qwen3MoE] %d tokens batched (chunk %d%s): attention %.0f ms, ffn %.0f ms (%.1f + %.1f ms per token)%n",
                total, cap, gpuAttention != null ? ", GPU attention" : "", attnNs / 1e6, ffnNs / 1e6,
                attnNs / 1e6 / total, ffnNs / 1e6 / total);
        }
        return outputProjection(state);
    }

    private float[][][] prefillBuffers(Qwen3MoEState state, int cap) {
        if (state.prefillBuffers == null || state.prefillBuffers[B_X].length < cap) {
            int dim = config.embeddingLength();
            int qDim = config.headCount() * config.headSize();
            int kvDim = config.kvDim();
            int efd = config.expertFfnLength();
            int ffn = Math.max(config.intermediateSize(), config.expertSharedCount() * efd);
            int slots = cap * Math.max(1, config.expertUsedCount());
            state.prefillBuffers = new float[][][] {
                new float[cap][dim], new float[cap][dim], new float[cap][dim],
                new float[cap][qDim], new float[cap][kvDim], new float[cap][kvDim], new float[cap][qDim],
                new float[cap][ffn], new float[cap][ffn], new float[cap][dim],
                new float[slots][efd], new float[slots][efd], new float[slots][dim]
            };
            int experts = Math.max(1, config.expertCount());
            state.prefillExperts = new int[slots];
            state.prefillWeights = new float[slots];
            state.prefillGroupStart = new int[experts + 1];
            state.prefillGroupSlots = new int[slots];
            state.prefillUsed = new int[experts];
            state.prefillCpuUsed = new int[experts];
            state.prefillOnGpu = new boolean[experts];
        }
        return state.prefillBuffers;
    }

    /** See {@link FloatTensor#warmUpRows}: batched prefill skips the kernels decode will use. */
    private void warmDecodeKernels() {
        int dim = config.embeddingLength();
        int qDim = config.headCount() * config.headSize();
        int kvDim = config.kvDim();
        int efd = config.expertFfnLength();
        int ffn = config.intermediateSize();
        int sharedFfn = config.expertSharedCount() * efd;
        for (int layer = 0; layer < config.blockCount(); layer++) {
            Qwen3MoELayerWeights lw = weights.layers()[layer];
            FloatTensor.warmUpRows(lw.wq(), qDim, dim);
            FloatTensor.warmUpRows(lw.wk(), kvDim, dim);
            FloatTensor.warmUpRows(lw.wv(), kvDim, dim);
            FloatTensor.warmUpRows(lw.wo(), dim, qDim);
            if (layer < config.leadingDenseBlockCount()) {
                FloatTensor.warmUpRows(lw.wGate(), ffn, dim);
                FloatTensor.warmUpRows(lw.wUp(), ffn, dim);
                FloatTensor.warmUpRows(lw.wDown(), dim, ffn);
                continue;
            }
            if (lw.ffnGateExps() != null) {
                // Decode computes routed experts with one dot per row (expertMatmul)
                FloatTensor.warmUpDot(lw.ffnGateExps(), efd, dim);
                FloatTensor.warmUpDot(lw.ffnUpExps(), efd, dim);
                FloatTensor.warmUpDot(lw.ffnDownExps(), dim, efd);
            }
            if (lw.ffnGateShexp() != null) {
                FloatTensor.warmUpRows(lw.ffnGateShexp(), sharedFfn, dim);
                FloatTensor.warmUpRows(lw.ffnUpShexp(), sharedFfn, dim);
                FloatTensor.warmUpRows(lw.ffnDownShexp(), dim, sharedFfn);
            }
        }
        FloatTensor.warmUpRows(weights.output(), config.vocabSize(), dim);
    }

    /**
     * Multi-token {@link #gqaAttention} plus its residual, for the chunk's tokens at one layer.
     * Returns true when the GPU pass ran it and already wrote the FFN-normed input to {@code b[B_XN]}.
     */
    private boolean attentionBatch(Qwen3MoEState state, float[][][] b, int layer, int basePos, int n) {
        GpuAttentionPass gpu = gpuAttention;
        if (gpu != null && gpu.isLayerOnGpu(layer)) {
            try {
                gpu.attentionLayerBatch(layer, b[B_X], b[B_XN], basePos, n);
                return true;
            } catch (RuntimeException e) {
                // Throws GpuFailureException unless this is the sequence's very first GPU call
                gpuAttentionFailed(e, basePos, layer);
            }
        }
        cpuAttentionBatch(state, b, layer, basePos, n);
        return false;
    }

    private void cpuAttentionBatch(Qwen3MoEState state, float[][][] b, int layer, int basePos, int n) {
        Qwen3MoELayerWeights lw = weights.layers()[layer];
        int dim = config.embeddingLength();
        int qDim = config.headCount() * config.headSize();
        int kvDim = config.kvDim();
        float[][] x = b[B_X], xn = b[B_XN], xb = b[B_XB];
        for (int t = 0; t < n; t++) {
            RMSNorm.apply(xn[t], x[t], cachedAttnNorm[layer], dim, config.normEps());
            Arrays.fill(b[B_Q][t], 0, qDim, 0f);
            Arrays.fill(b[B_K][t], 0, kvDim, 0f);
            Arrays.fill(b[B_V][t], 0, kvDim, 0f);
        }
        FloatTensor.fusedQKVBatchParallel(lw.wq(), lw.wk(), lw.wv(), xn, b[B_Q], b[B_K], b[B_V],
            n, qDim, kvDim, dim);
        for (int t = 0; t < n; t++) {
            System.arraycopy(b[B_Q][t], 0, state.q, 0, qDim);
            System.arraycopy(b[B_K][t], 0, state.k, 0, kvDim);
            System.arraycopy(b[B_V][t], 0, state.v, 0, kvDim);
            attentionCore(state, lw, layer, basePos + t);
            System.arraycopy(state.xb2, 0, b[B_ATT][t], 0, qDim);
            Arrays.fill(xb[t], 0f);
        }
        FloatTensor.matmulBatchParallel(lw.wo(), b[B_ATT], xb, n, dim, qDim);
        for (int t = 0; t < n; t++) {
            if (lw.woBias() != null) addBias(xb[t], lw.woBias(), dim);
            VectorOpsFactory.get().accumulate(x[t], xb[t], dim);
        }
    }

    /**
     * Multi-token FFN half of {@link #forwardLayer}: norm (unless {@code normed}: the GPU pass wrote
     * it), dense or MoE FFN, residual.
     */
    private void ffnBatch(Qwen3MoEState state, float[][][] b, int layer, int n, boolean normed) {
        Qwen3MoELayerWeights lw = weights.layers()[layer];
        int dim = config.embeddingLength();
        float[][] x = b[B_X], xn = b[B_XN], xb = b[B_XB];
        if (!normed) {
            for (int t = 0; t < n; t++) {
                RMSNorm.apply(xn[t], x[t], cachedFfnNorm[layer], dim, config.normEps());
            }
        }
        if (layer < config.leadingDenseBlockCount()) {
            int ffn = config.intermediateSize();
            swigluBatch(lw.wGate(), lw.wUp(), lw.wDown(), b, n, ffn, b[B_XB]);
        } else {
            moeBatch(state, b, lw, layer, n);
        }
        for (int t = 0; t < n; t++) {
            VectorOpsFactory.get().accumulate(x[t], xb[t], dim);
        }
    }

    /** {@code out[t] = down(silu(gate(xn[t])) * up(xn[t]))} over the chunk, as {@link #denseFFN}. */
    private void swigluBatch(FloatTensor gate, FloatTensor up, FloatTensor down, float[][][] b, int n,
                             int ffn, float[][] out) {
        int dim = config.embeddingLength();
        float[][] hb = b[B_HB], hb2 = b[B_HB2];
        for (int t = 0; t < n; t++) {
            Arrays.fill(hb[t], 0, ffn, 0f);
            Arrays.fill(hb2[t], 0, ffn, 0f);
        }
        FloatTensor.fusedGateUpBatchParallel(gate, up, b[B_XN], hb, hb2, n, ffn, dim);
        for (int t = 0; t < n; t++) {
            VectorOpsFactory.get().silu(hb[t], ffn);
            VectorOpsFactory.get().elementwiseMul(hb[t], hb2[t], hb[t], ffn);
            Arrays.fill(out[t], 0, dim, 0f);
        }
        FloatTensor.matmulBatchParallel(down, hb, out, n, dim, ffn);
    }

    /**
     * Multi-token {@link #moeFFN}: route every token, group the (token, slot) pairs by expert, run
     * each expert once over its tokens, then sum each token's expert outputs in slot order and add
     * the shared expert — the same per-token arithmetic as the one-token path.
     */
    private void moeBatch(Qwen3MoEState state, float[][][] b, Qwen3MoELayerWeights lw, int layer, int n) {
        int dim = config.embeddingLength();
        int expertCount = config.expertCount();
        int k = MoERouting.effectiveTopK(config.expertUsedCount());
        int efd = config.expertFfnLength();
        long elementsPerSlice = (long) efd * dim;
        float[][] xn = b[B_XN], xb = b[B_XB];
        int[] sel = state.prefillExperts;
        float[] selW = state.prefillWeights;

        // 1. Route each token (same router code and state buffers as the one-token path)
        currentLayer = layer;
        for (int t = 0; t < n; t++) {
            System.arraycopy(xn[t], 0, state.xbSaved, 0, dim);
            routeExperts(state, lw);
            System.arraycopy(state.selectedExperts, 0, sel, t * k, k);
            System.arraycopy(state.selectedWeights, 0, selW, t * k, k);
            GpuExpertCache gc = expertGpuCache;
            if (gc != null) gc.noteRouting(layer, state.selectedExperts, k); // warm the LFU counts
        }

        // 2. Group the (token, slot) pairs by expert
        int slots = n * k;
        int[] start = state.prefillGroupStart;
        int[] grouped = state.prefillGroupSlots;
        int[] used = state.prefillUsed;
        int nUsed = ExpertViews.groupByExpert(sel, slots, expertCount, start, grouped, used);
        for (int s = 0; s < slots; s++) {
            if (sel[s] < 0) Arrays.fill(b[B_EOUT][s], 0, dim, 0f);
        }

        // 3. Compute the experts: the ones resident in the GPU expert cache on the GPU over all
        // their tokens, queued first; the others on the CPU meanwhile (grouped for the SSD cache,
        // the next group read while one computes)
        GpuExpertCache gc = expertGpuCache;
        int[] cpuUsed = used;
        int nCpu = nUsed;
        int onGpu = 0;
        if (gc != null && lw.ffnGateExpsBias() == null) {
            try {
                onGpu = gc.launchResidentBatch(layer, lw.ffnGateExps(), lw.ffnUpExps(), lw.ffnDownExps(), xn, n,
                    used, nUsed, start, grouped, k, state.prefillOnGpu, isGptOss);
            } catch (RuntimeException e) {
                System.err.println("Expert GPU cache error: " + GpuFailureException.describe(e) + " — using the CPU experts");
                expertGpuCache = null;
                gc = null;
                onGpu = 0;
            }
            if (onGpu > 0) {
                cpuUsed = state.prefillCpuUsed;
                nCpu = 0;
                for (int i = 0; i < nUsed; i++) if (!state.prefillOnGpu[used[i]]) cpuUsed[nCpu++] = used[i];
            }
        }
        expertViews.forEachExpert(expertCache, layer, cpuUsed, nCpu, lw.ffnGateExps(), lw.ffnUpExps(),
            lw.ffnDownExps(), elementsPerSlice,
            (e, gate, up, down) -> expertBatch(b, lw, e, gate, up, down, start, grouped, k, efd, dim));
        if (onGpu > 0) {
            try {
                gc.finishResidentBatch(b[B_EOUT]);
            } catch (RuntimeException e) {
                // the GPU experts' outputs are lost: compute them on the CPU
                System.err.println("Expert GPU cache error: " + GpuFailureException.describe(e) + " — using the CPU experts");
                expertGpuCache = null;
                nCpu = 0;
                for (int i = 0; i < nUsed; i++) if (state.prefillOnGpu[used[i]]) cpuUsed[nCpu++] = used[i];
                expertViews.forEachExpert(expertCache, layer, cpuUsed, nCpu, lw.ffnGateExps(), lw.ffnUpExps(),
                    lw.ffnDownExps(), elementsPerSlice,
                    (e2, gate, up, down) -> expertBatch(b, lw, e2, gate, up, down, start, grouped, k, efd, dim));
            }
        }

        // 4. Per token: weighted sum of its expert outputs in slot order, then the shared expert
        for (int t = 0; t < n; t++) {
            Arrays.fill(xb[t], 0, dim, 0f);
            for (int j = 0; j < k; j++) {
                VectorOpsFactory.get().saxpy(selW[t * k + j], b[B_EOUT][t * k + j], 0, xb[t], 0, dim);
            }
        }
        if (lw.ffnGateShexp() != null) {
            int sharedFfn = config.expertSharedCount() * efd;
            swigluBatch(lw.ffnGateShexp(), lw.ffnUpShexp(), lw.ffnDownShexp(), b, n, sharedFfn, b[B_SHOUT]);
            for (int t = 0; t < n; t++) {
                VectorOpsFactory.get().accumulate(xb[t], b[B_SHOUT][t], dim);
            }
        }
    }

    /** One routed expert over all its (token, slot) pairs of the chunk: gate, up, activation, down. */
    private void expertBatch(float[][][] b, Qwen3MoELayerWeights lw, int e, FloatTensor wGate, FloatTensor wUp,
                             FloatTensor wDown, int[] start, int[] grouped, int k, int efd, int dim) {
        int from = start[e], m = start[e + 1] - from;
        float[][] in = new float[m][], gate = new float[m][], up = new float[m][], out = new float[m][];
        for (int i = 0; i < m; i++) {
            int s = grouped[from + i];
            in[i] = b[B_XN][s / k];
            gate[i] = b[B_EGATE][s];
            up[i] = b[B_EUP][s];
            out[i] = b[B_EOUT][s];
            Arrays.fill(gate[i], 0, efd, 0f);
            Arrays.fill(up[i], 0, efd, 0f);
            Arrays.fill(out[i], 0, dim, 0f);
        }
        wGate.matmulRowsBatch(in, gate, m, 0, efd, dim);
        wUp.matmulRowsBatch(in, up, m, 0, efd, dim);
        for (int i = 0; i < m; i++) {
            if (lw.ffnGateExpsBias() != null) addExpertBias(gate[i], lw.ffnGateExpsBias(), e, efd);
            if (lw.ffnUpExpsBias() != null) addExpertBias(up[i], lw.ffnUpExpsBias(), e, efd);
            if (isGptOss) {
                swigluOai(gate[i], up[i], efd);
            } else {
                VectorOpsFactory.get().silu(gate[i], efd);
                VectorOpsFactory.get().elementwiseMul(gate[i], up[i], gate[i], efd);
            }
        }
        wDown.matmulRowsBatch(gate, out, m, 0, dim, efd);
        if (lw.ffnDownExpsBias() != null) {
            for (int i = 0; i < m; i++) addExpertBias(out[i], lw.ffnDownExpsBias(), e, dim);
        }
    }

    /** Load a token's embedding into the residual stream. */
    private void embedToken(Qwen3MoEState state, int token) {
        int dim = config.embeddingLength();
        for (int i = 0; i < dim; i++) {
            state.x[i] = weights.tokenEmbedding().getFloat((long) token * dim + i);
        }
    }

    /**
     * One transformer block, reading and writing {@code state.x}. Split out of {@code forwardInternal}
     * so prefill can drive the layers in the outer loop — see {@link #forwardPrefill}.
     */
    private void forwardLayer(Qwen3MoEState state, int layer, int position) {
        int dim = config.embeddingLength();
        int leadingDenseCount = config.leadingDenseBlockCount();
        final boolean d = cpuProfile;
        Qwen3MoELayerWeights layerWeights = weights.layers()[layer];
        long t0 = System.nanoTime(), t1;

        GpuAttentionPass gpu = gpuAttention;
        if (gpu != null && gpu.isLayerOnGpu(layer)) {
            // Whole attention half on the GPU: x += Attn(norm(x)), xb = ffnNorm(x)
            try {
                gpu.attentionLayer(layer, state.x, state.xb, position);
            } catch (RuntimeException e) {
                gpuAttentionFailed(e, position, layer);
                forwardLayer(state, layer, position); // first GPU call of a sequence: redo on the CPU
                return;
            }
            t1 = System.nanoTime(); prof.add(P_ATTN, t1 - t0); t0 = t1;
        } else {
            RMSNorm.apply(state.xb, state.x, cachedAttnNorm[layer], dim, config.normEps());
            if (d) { t1 = System.nanoTime(); prof.add(P_ATTN_NORM, t1 - t0); t0 = t1; }

            gqaAttention(state, layerWeights, layer, position);
            t1 = System.nanoTime(); prof.add(P_ATTN, t1 - t0); t0 = t1;

            VectorOpsFactory.get().accumulate(state.x, state.xb, dim);
            if (d) { t1 = System.nanoTime(); prof.add(P_RESIDUAL, t1 - t0); t0 = t1; }

            RMSNorm.apply(state.xb, state.x, cachedFfnNorm[layer], dim, config.normEps());
            if (d) { t1 = System.nanoTime(); prof.add(P_FFN_NORM, t1 - t0); t0 = t1; }
        }

        if (layer < leadingDenseCount) {
            denseFFN(state, layerWeights);
            if (d) { t1 = System.nanoTime(); prof.add(P_DENSE, t1 - t0); t0 = t1; }
        } else {
            System.arraycopy(state.xb, 0, state.xbSaved, 0, dim);
            currentLayer = layer;
            GpuExpertCache gc = expertGpuCache;
            if (gc != null) gc.noteToken(position);
            currentPosition = position;
            moeFFN(state, layerWeights);
            t1 = System.nanoTime(); prof.add(P_MOE, t1 - t0); t0 = t1;
        }

        VectorOpsFactory.get().accumulate(state.x, state.xb, dim);
        if (d) prof.add(P_RESIDUAL, System.nanoTime() - t0);
    }

    /** Final norm + logit projection over the current residual stream. */
    private float[] outputProjection(Qwen3MoEState state) {
        int dim = config.embeddingLength();
        long t0 = System.nanoTime();
        RMSNorm.apply(state.xb, state.x, outputNormCache, dim, config.normEps());
        int vocabSize = config.vocabSize();
        Arrays.fill(state.logits, 0);
        outputRouter.matmul(state.xb, state.logits, vocabSize, dim);
        // The first projection after a prefill closes the prompt; DecodeProfile drops the prompt's
        // layer time there, so every average is per decoded token.
        prof.endToken(System.nanoTime() - t0, cacheStatsSupplier);
        return state.logits;
    }

    /**
     * Standard GQA Attention with per-head QK normalization and RoPE.
     * Same as standard Attention class but inlined for Qwen3 MoE state.
     */
    private void gqaAttention(Qwen3MoEState state, Qwen3MoELayerWeights weights, int layer, int position) {
        int dim = config.embeddingLength();
        int kvDim = config.kvDim();
        int qDim = config.headCount() * config.headSize(); // may differ from dim (e.g., Qwen3-Coder-30B)

        // Project Q, K, V
        Arrays.fill(state.q, 0, qDim, 0f);
        Arrays.fill(state.k, 0, kvDim, 0f);
        Arrays.fill(state.v, 0, kvDim, 0f);

        weights.wq().matmulParallel(state.xb, state.q, qDim, dim);
        weights.wk().matmulParallel(state.xb, state.k, kvDim, dim);
        weights.wv().matmulParallel(state.xb, state.v, kvDim, dim);

        attentionCore(state, weights, layer, position);

        // Output projection: qDim -> dim
        Arrays.fill(state.xb, 0);
        weights.wo().matmulParallel(state.xb2, state.xb, dim, qDim);
        if (weights.woBias() != null) addBias(state.xb, weights.woBias(), dim);
    }

    /**
     * The order-dependent part of attention for one token: Q/K/V biases, QK-norm, RoPE, KV store
     * and attention over the cache. Reads the raw projections from {@code state.q/k/v} and writes
     * the attention output, before Wo, to {@code state.xb2}. Shared by the one-token path and
     * batched prefill, which feeds it one token at a time in position order.
     */
    private void attentionCore(Qwen3MoEState state, Qwen3MoELayerWeights weights, int layer, int position) {
        int headCount = config.headCount();
        int headCountKV = config.headCountKV();
        int headSize = config.headSize();
        int kvDim = config.kvDim();
        int qDim = headCount * headSize;
        int kvMul = headCount / headCountKV;

        // Apply attention biases (GPT-OSS)
        if (weights.wqBias() != null) addBias(state.q, weights.wqBias(), qDim);
        if (weights.wkBias() != null) addBias(state.k, weights.wkBias(), kvDim);
        if (weights.wvBias() != null) addBias(state.v, weights.wvBias(), kvDim);

        // Apply per-head QK-norm (Qwen3)
        if (cachedQNorm[layer] != null) {
            applyPerHeadNorm(state.q, cachedQNorm[layer], headCount, headSize, config.normEps());
            applyPerHeadNorm(state.k, cachedKNorm[layer], headCountKV, headSize, config.normEps());
        }

        // Apply RoPE (skip for NoPE layers in Llama4 iRoPE: every Nth layer)
        if (noRopeLayerInterval == 0 || (layer % noRopeLayerInterval) != (noRopeLayerInterval - 1)) {
            rope.applyAllHeads(state.q, headCount, position);
            rope.applyAllHeads(state.k, headCountKV, position);
        }

        // Store K and V in cache (quantized transparently if KV cache is in Q8 mode)
        state.kvCache.storeK(layer, position, state.k, kvDim);
        state.kvCache.storeV(layer, position, state.v, kvDim);

        // Attention computation - parallel over heads
        // Include YaRN mscale^2 for attention magnitude correction
        float mscale = rope.getMscale();
        final float scaleFactor = mscale * mscale / (float) Math.sqrt(headSize);

        // ISWA: dispatch matches Attention.isGlobalLayer so each MoE-routed arch gets the right pattern.
        // GPT-OSS routes here for the MoE variant — its convention is even=global, odd=local.
        // Other archs through this engine (QWEN3MOE, LLAMA4 MoE, GLM4 MoE) don't ship SWA today,
        // but the dispatch is centralized for future-proofing.
        final int startPos;
        if (slidingWindow > 0 && !isSwaGlobalLayer(layer)) {
            startPos = Math.max(0, position - slidingWindow + 1);
        } else {
            startPos = 0;
        }
        final int attLen = position - startPos + 1;
        final int positionFinal = position;

        Arrays.fill(state.xb2, 0, qDim, 0f);

        // Capture for use in lambda
        final float[] sinks = cachedAttnSinks[layer];
        final KVCache kv = state.kvCache;
        final int layerFinal = layer;

        it.denzosoft.llmplayer.tensor.MatmulPool.forEach(headCount, new java.util.function.IntConsumer() {
            @Override
            public void accept(int h) {
                int kvHead = h / kvMul;
                int kvHeadOff = kvHead * headSize;
                int qOffset = h * headSize;
                int attOffset = h * attLen;
                for (int t = startPos; t <= positionFinal; t++) {
                    float score = kv.dotK(layerFinal, t, kvHeadOff, headSize, state.q, qOffset);
                    state.att[attOffset + (t - startPos)] = score * scaleFactor;
                }

                if (sinks != null) {
                    softmaxWithSink(state.att, attOffset, attLen, sinks[h]);
                } else {
                    VectorOpsFactory.get().softmax(state.att, attOffset, attLen);
                }

                int outOffset = h * headSize;
                for (int t = startPos; t <= positionFinal; t++) {
                    float a = state.att[attOffset + (t - startPos)];
                    kv.saxpyV(layerFinal, t, kvHeadOff, headSize, a, state.xb2, outOffset);
                }
            }
        });
    }

    /**
     * Dense SwiGLU FFN for leading dense blocks.
     */
    private void denseFFN(Qwen3MoEState state, Qwen3MoELayerWeights weights) {
        int dim = config.embeddingLength();
        int ffnDim = config.intermediateSize();

        Arrays.fill(state.hb, 0, ffnDim, 0f);
        weights.wGate().matmulParallel(state.xb, state.hb, ffnDim, dim);

        Arrays.fill(state.hb2, 0, ffnDim, 0f);
        weights.wUp().matmulParallel(state.xb, state.hb2, ffnDim, dim);

        VectorOpsFactory.get().silu(state.hb, ffnDim);
        VectorOpsFactory.get().elementwiseMul(state.hb, state.hb2, state.hb, ffnDim);

        Arrays.fill(state.xb, 0);
        weights.wDown().matmulParallel(state.hb, state.xb, dim, ffnDim);
    }

    /**
     * Mixture-of-Experts FFN with shared expert.
     */
    private void moeFFN(Qwen3MoEState state, Qwen3MoELayerWeights weights) {
        int dim = config.embeddingLength();
        int expertUsedCount = MoERouting.effectiveTopK(config.expertUsedCount());
        int expertFfnDim = config.expertFfnLength();
        int sharedFfnDim = config.expertSharedCount() * expertFfnDim;

        final boolean d = cpuProfile;
        long tRoute = d ? System.nanoTime() : 0;
        // 1. Router: expert logits and top-K selection
        routeExperts(state, weights);
        if (d) tRoute = System.nanoTime() - tRoute;

        // SSD streaming: the top-K experts for this layer are now known. Prefer L1 — read the whole
        // slices into the RAM cache with explicit positional reads — and fall back to the L0
        // read-ahead hint when there is no cache. Both are no-ops when the model fits RAM. Skipped
        // entirely when the GPU expert cache owns this path, which reads from the mapping itself.
        boolean gpuOwnsExperts = expertGpuCache != null && !GpuExpertCache.hybrid();
        cacheLayerReady = !gpuOwnsExperts && expertCache != null
            && expertCache.prepare(currentLayer, state.selectedExperts, expertUsedCount,
                weights.ffnGateExps(), weights.ffnUpExps(), weights.ffnDownExps(),
                (long) expertFfnDim * dim);
        if (!cacheLayerReady && !gpuOwnsExperts) {
            ExpertPrefetch.willNeed(weights.ffnGateExps(), weights.ffnUpExps(), weights.ffnDownExps(),
                state.selectedExperts, expertUsedCount, (long) expertFfnDim * dim);
        }

        // 2. Compute routed expert outputs
        Arrays.fill(state.xb, 0);

        // Capture for lambda
        final boolean useSwigluOai = isGptOss;

        if (expertGpuCache != null && GpuExpertCache.hybrid()) {
            // Hybrid: resident experts on the GPU, the others on the CPU at the same time
            GpuExpertCache cache = expertGpuCache;
            int mask = 0;
            try {
                long ta = d ? System.nanoTime() : 0;
                mask = cache.launchResident(weights.ffnGateExps(), weights.ffnUpExps(), weights.ffnDownExps(),
                    state.xbSaved, state.selectedExperts, expertUsedCount, currentLayer, dim, expertFfnDim,
                    useSwigluOai, weights.ffnGateExpsBias(), weights.ffnUpExpsBias(), weights.ffnDownExpsBias());
                long tb = d ? System.nanoTime() : 0;
                cpuExpertCompute(state, weights, expertUsedCount, expertFfnDim, dim, useSwigluOai, mask);
                long tc = d ? System.nanoTime() : 0;
                cache.finishResident(mask, state.expertOutPerExpert);
                if (d) prof.moeSplit(tRoute, tb - ta, tc - tb, System.nanoTime() - tc);
            } catch (RuntimeException e) {
                System.err.println("Expert GPU cache error: " + GpuFailureException.describe(e) + " — using the CPU experts");
                expertGpuCache = null;
                cpuExpertCompute(state, weights, expertUsedCount, expertFfnDim, dim, useSwigluOai);
            }
        } else if (expertGpuCache != null) {
            // GPU-accelerated path: batch all experts on GPU with LRU caching
            try {
                // Zero per-expert buffers
                for (int k = 0; k < expertUsedCount; k++) {
                    Arrays.fill(state.moeHbPerExpert[k], 0, expertFfnDim, 0f);
                    Arrays.fill(state.moeHb2PerExpert[k], 0, expertFfnDim, 0f);
                    Arrays.fill(state.expertOutPerExpert[k], 0, dim, 0f);
                }
                expertGpuCache.computeExperts(
                    weights.ffnGateExps(), weights.ffnUpExps(), weights.ffnDownExps(),
                    state.xbSaved, state.selectedExperts, state.selectedWeights,
                    expertUsedCount, currentLayer, dim, expertFfnDim,
                    state.moeHbPerExpert, state.moeHb2PerExpert, state.expertOutPerExpert,
                    useSwigluOai,
                    weights.ffnGateExpsBias(), weights.ffnUpExpsBias(), weights.ffnDownExpsBias());

                if (debugCache && !debugCacheDone) {
                    debugCacheDone = true;
                    float[][] gpuOut = new float[expertUsedCount][];
                    for (int k = 0; k < expertUsedCount; k++) gpuOut[k] = state.expertOutPerExpert[k].clone();
                    // CPU recompute (overwrites expertOutPerExpert) for comparison
                    cpuExpertCompute(state, weights, expertUsedCount, expertFfnDim, dim, useSwigluOai);
                    for (int k = 0; k < expertUsedCount; k++) {
                        double maxd = 0; int e = state.selectedExperts[k];
                        for (int i = 0; i < dim; i++) maxd = Math.max(maxd, Math.abs(gpuOut[k][i] - state.expertOutPerExpert[k][i]));
                        System.err.printf("  [cache.debug] layer %d k=%d expert %d: max|GPU-CPU|=%.4f  GPU[0..2]=%.3f,%.3f,%.3f  CPU[0..2]=%.3f,%.3f,%.3f%n",
                            currentLayer, k, e, maxd, gpuOut[k][0], gpuOut[k][1], gpuOut[k][2],
                            state.expertOutPerExpert[k][0], state.expertOutPerExpert[k][1], state.expertOutPerExpert[k][2]);
                    }
                }
            } catch (Throwable e) {
                // Fallback to CPU on error, disable cache. Print the real cause (reflection wraps it).
                Throwable c = (e instanceof java.lang.reflect.InvocationTargetException && e.getCause() != null) ? e.getCause() : e;
                System.err.println("Expert GPU cache error: " + c + " — falling back to CPU");
                if ("true".equals(System.getProperty("cuda.debug", "false"))) c.printStackTrace();
                expertGpuCache = null;
                cpuExpertCompute(state, weights, expertUsedCount, expertFfnDim, dim, useSwigluOai);
            }
        } else {
            // CPU parallel path
            long ta = d ? System.nanoTime() : 0;
            cpuExpertCompute(state, weights, expertUsedCount, expertFfnDim, dim, useSwigluOai);
            if (d) prof.moeSplit(tRoute, 0, System.nanoTime() - ta, 0);
        }

        // Sequential accumulation of weighted expert outputs
        for (int k = 0; k < expertUsedCount; k++) {
            VectorOpsFactory.get().saxpy(state.selectedWeights[k], state.expertOutPerExpert[k], 0, state.xb, 0, dim);
        }

        // 3. Shared expert: computed by the GPU attention pass when it holds it (-Dmoe.attn.shared=false disables)
        GpuAttentionPass gpuSh = gpuAttention;
        boolean sharedDone = false;
        if (weights.ffnGateShexp() != null && gpuSh != null && gpuSh.isLayerOnGpu(currentLayer)) {
            try {
                sharedDone = gpuSh.takeSharedExpert(currentLayer, state.expertOut);
            } catch (RuntimeException e) {
                gpuAttentionFailed(e, currentPosition, currentLayer); // throws past the first GPU call
            }
            if (sharedDone) VectorOpsFactory.get().accumulate(state.xb, state.expertOut, dim);
        }
        if (weights.ffnGateShexp() != null && !sharedDone) {
            float[] shGate = state.sharedHb;
            float[] shUp = state.sharedHb2;
            Arrays.fill(shGate, 0, sharedFfnDim, 0f);
            Arrays.fill(shUp, 0, sharedFfnDim, 0f);

            weights.ffnGateShexp().matmulParallel(state.xbSaved, shGate, sharedFfnDim, dim);
            weights.ffnUpShexp().matmulParallel(state.xbSaved, shUp, sharedFfnDim, dim);

            VectorOpsFactory.get().silu(shGate, sharedFfnDim);
            VectorOpsFactory.get().elementwiseMul(shGate, shUp, shGate, sharedFfnDim);

            float[] sharedOut = state.expertOut;
            Arrays.fill(sharedOut, 0, dim, 0f);
            weights.ffnDownShexp().matmulParallel(shGate, sharedOut, dim, sharedFfnDim);

            VectorOpsFactory.get().accumulate(state.xb, sharedOut, dim);
        }
    }

    /**
     * Router for one token: logits from {@code state.xbSaved}, top-K selection and weight
     * normalisation into {@code state.selectedExperts} / {@code state.selectedWeights}.
     */
    private void routeExperts(Qwen3MoEState state, Qwen3MoELayerWeights weights) {
        int dim = config.embeddingLength();
        int expertCount = config.expertCount();
        int expertUsedCount = MoERouting.effectiveTopK(config.expertUsedCount());

        Arrays.fill(state.routerLogits, 0, expertCount, 0f);
        weights.ffnGateInp().matmul(state.xbSaved, state.routerLogits, expertCount, dim);
        if (weights.ffnGateInpBias() != null) addBias(state.routerLogits, weights.ffnGateInpBias(), expertCount);

        if (sigmoidRouting) {
            // llama.cpp build_moe_ffn with SIGMOID gating (GLM4-MoE): probs = sigmoid(logits); top-K
            // on probs + exp_probs_b; the selected UNBIASED probs, sum-normalised when
            // expert_weights_norm (clamped at the F16 epsilon), times expert_weights_scale.
            float[] probs = state.routerLogits;
            for (int e = 0; e < expertCount; e++) probs[e] = 1.0f / (1.0f + (float) Math.exp(-probs[e]));
            float[] sel = state.selectionScores;
            FloatTensor bias = weights.expProbsBias();
            for (int e = 0; e < expertCount; e++) sel[e] = probs[e] + (bias != null ? bias.getFloat(e) : 0f);
            selectTopK(sel, expertCount, expertUsedCount, state.selectedExperts, state.selectedWeights);
            float sum = 0f;
            for (int k = 0; k < expertUsedCount; k++) {
                int e = state.selectedExperts[k];
                state.selectedWeights[k] = e >= 0 ? probs[e] : 0f;
                sum += state.selectedWeights[k];
            }
            float mul = config.expertWeightsNorm() ? 1f / Math.max(sum, 6.103515625e-5f) : 1f;
            float scale = config.expertWeightsScale();
            if (scale != 0f) mul *= scale;
            for (int k = 0; k < expertUsedCount; k++) state.selectedWeights[k] *= mul;
        } else if (isGptOss) {
            // SOFTMAX_WEIGHT routing: select top-K by raw logits, then softmax over selected
            selectTopK(state.routerLogits, expertCount, expertUsedCount,
                state.selectedExperts, state.selectedWeights);

            // Softmax only over the selected experts' raw logits
            float maxW = Float.NEGATIVE_INFINITY;
            for (int k = 0; k < expertUsedCount; k++) maxW = Math.max(maxW, state.selectedWeights[k]);
            float sum = 0f;
            for (int k = 0; k < expertUsedCount; k++) {
                state.selectedWeights[k] = (float) Math.exp(state.selectedWeights[k] - maxW);
                sum += state.selectedWeights[k];
            }
            if (sum > 0f) {
                for (int k = 0; k < expertUsedCount; k++) {
                    state.selectedWeights[k] /= sum;
                }
            }
        } else {
            // Standard Qwen3 MoE: softmax over all experts first, then top-K + renormalize
            VectorOpsFactory.get().softmax(state.routerLogits, 0, expertCount);

            selectTopK(state.routerLogits, expertCount, expertUsedCount,
                state.selectedExperts, state.selectedWeights);

            float weightSum = 0f;
            for (int k = 0; k < expertUsedCount; k++) {
                weightSum += state.selectedWeights[k];
            }
            // E18: clamp to smallest F16 normal (6.103515625e-5) to guard against NaN when
            // the routing distribution has collapsed to near-zero — matches llama.cpp
            // ggml_clamp in build_moe_ffn (llama-graph.cpp:1325).
            if (weightSum > 6.103515625e-5f) {
                for (int k = 0; k < expertUsedCount; k++) {
                    state.selectedWeights[k] /= weightSum;
                }
            }
        }

        // Phase 2.2a: routing-frequency instrumentation (opt-in, -Dmoe.routing.stats=true). Counts
        // how often each expert is selected per layer, to measure whether routing is concentrated
        // enough to justify a hot-expert GPU cache. Additive only — no effect on the forward pass.
        if (routingStats) {
            long[] hits = expertHits[currentLayer];
            for (int k = 0; k < expertUsedCount; k++) hits[state.selectedExperts[k]]++;
            routingDecisions += expertUsedCount;
        }
    }

    /** Minimum rows per parallel chunk of the routed-expert loops. */
    private static final int EXPERT_ROW_CHUNK = 16;

    /**
     * CPU routed experts. The gate/up rows of all K experts form one parallel range and the down
     * rows a second one, so every core works on each projection (a loop over the K experts alone
     * keeps only K threads busy).
     */
    private void cpuExpertCompute(Qwen3MoEState state, Qwen3MoELayerWeights weights,
                                   int expertUsedCount, int expertFfnDim, int dim,
                                   boolean useSwigluOai) {
        cpuExpertCompute(state, weights, expertUsedCount, expertFfnDim, dim, useSwigluOai, 0);
    }

    /** As above, skipping the slots in {@code gpuMask} (computed by the GPU expert cache). */
    private void cpuExpertCompute(Qwen3MoEState state, Qwen3MoELayerWeights weights,
                                   int expertUsedCount, int expertFfnDim, int dim,
                                   boolean useSwigluOai, int gpuMask) {
        if (gpuMask == (1 << expertUsedCount) - 1) return;
        final int[] sel = state.selectedExperts;
        final float[][] gate = state.moeHbPerExpert, up = state.moeHb2PerExpert, out = state.expertOutPerExpert;
        final int efd = expertFfnDim;
        // F11: the parallel ranges cover only the CPU slots, so no chunk lands on a slot the GPU
        // computes (those finished instantly and left their worker idle). Bit-identical: each row
        // is one independent dot. An unfilled routing slot (selectTopK leaves -1 when the router
        // logits contain NaN) contributes nothing; its output is zeroed here, outside the loops.
        final int[] slots = new int[expertUsedCount];
        int nCpu = 0;
        for (int k = 0; k < expertUsedCount; k++) {
            if ((gpuMask & (1 << k)) != 0) continue;
            if (sel[k] < 0) { Arrays.fill(out[k], 0, dim, 0f); continue; }
            slots[nCpu++] = k;
        }
        if (nCpu == 0) return;
        final int m = nCpu;
        it.denzosoft.llmplayer.tensor.MatmulPool.forRange(m * efd, EXPERT_ROW_CHUNK, (from, to) -> {
            for (int u = from; u < to; ) {
                int j = u / efd, slot = slots[j], r0 = u - j * efd, r1 = Math.min(efd, r0 + (to - u));
                int e = sel[slot];
                expertMatmul(weights.ffnGateExps(), state.xbSaved, gate[slot], e, dim, efd, r0, r1,
                    it.denzosoft.llmplayer.tensor.ExpertCache.PROJ_GATE);
                expertMatmul(weights.ffnUpExps(), state.xbSaved, up[slot], e, dim, efd, r0, r1,
                    it.denzosoft.llmplayer.tensor.ExpertCache.PROJ_UP);
                u += r1 - r0;
            }
        });
        for (int k = 0; k < expertUsedCount; k++) {
            int e = sel[k];
            if (e < 0 || (gpuMask & (1 << k)) != 0) continue;
            if (weights.ffnGateExpsBias() != null) addExpertBias(gate[k], weights.ffnGateExpsBias(), e, efd);
            if (weights.ffnUpExpsBias() != null) addExpertBias(up[k], weights.ffnUpExpsBias(), e, efd);
            if (useSwigluOai) {
                swigluOai(gate[k], up[k], efd);
            } else {
                VectorOpsFactory.get().silu(gate[k], efd);
                VectorOpsFactory.get().elementwiseMul(gate[k], up[k], gate[k], efd);
            }
        }
        it.denzosoft.llmplayer.tensor.MatmulPool.forRange(m * dim, EXPERT_ROW_CHUNK, (from, to) -> {
            for (int u = from; u < to; ) {
                int j = u / dim, slot = slots[j], r0 = u - j * dim, r1 = Math.min(dim, r0 + (to - u));
                expertMatmul(weights.ffnDownExps(), gate[slot], out[slot], sel[slot], efd, dim, r0, r1,
                    it.denzosoft.llmplayer.tensor.ExpertCache.PROJ_DOWN);
                u += r1 - r0;
            }
        });
        if (weights.ffnDownExpsBias() != null) {
            for (int k = 0; k < expertUsedCount; k++) {
                if (sel[k] >= 0 && (gpuMask & (1 << k)) == 0) addExpertBias(out[k], weights.ffnDownExpsBias(), sel[k], dim);
            }
        }
    }

    /**
     * Rows {@code [r0, r1)} of one expert slice of a 3D tensor: {@code output[row] = W[row]·input}.
     */
    private void expertMatmul(FloatTensor weights3D, float[] input, float[] output,
                              int expert, int inDim, int outDim, int r0, int r1, int projection) {
        // When the slice is cached, it is a standalone tensor holding just this expert, so the rows
        // start at 0 instead of the expert's base offset inside the 3D tensor.
        if (cacheLayerReady) {
            FloatTensor cached = expertCache.tensorFor(currentLayer, expert, projection);
            if (cached != null) {
                for (int row = r0; row < r1; row++) {
                    output[row] = cached.dot((long) row * inDim, input, 0, inDim);
                }
                return;
            }
        }
        long expertOffset = (long) expert * outDim * inDim;
        for (int row = r0; row < r1; row++) {
            output[row] = weights3D.dot(expertOffset + (long) row * inDim, input, 0, inDim);
        }
    }

    /**
     * Select top-K indices and values from logits.
     */
    /** Phase 2.2a: print the expert-routing concentration at exit (decides if a hot-expert cache helps). */
    private void printRoutingStats() {
        if (expertHits == null || routingDecisions == 0) return;
        int L = expertHits.length, E = config.expertCount(), K = config.expertUsedCount();
        System.err.println("\n=== MoE routing stats (" + routingDecisions + " selections, " + E
            + " experts, top-" + K + ") ===");
        int[] Ms = {K, 2 * K, 4 * K, 8 * K, Math.max(1, E / 4), Math.max(1, E / 2)};
        double[] covSum = new double[Ms.length];
        int moeLayers = 0;
        for (int l = 0; l < L; l++) {
            long[] h = expertHits[l].clone();
            long tot = 0; for (long x : h) tot += x;
            if (tot == 0) continue;
            moeLayers++;
            java.util.Arrays.sort(h); // ascending; hottest are at the end
            for (int mi = 0; mi < Ms.length; mi++) {
                int m = Math.min(Ms[mi], E);
                long top = 0; for (int i = E - m; i < E; i++) top += h[i];
                covSum[mi] += (double) top / tot;
            }
        }
        if (moeLayers == 0) return;
        System.err.printf("  %d MoE layers. Avg fraction of routing captured by the top-M experts per layer:%n", moeLayers);
        for (int mi = 0; mi < Ms.length; mi++) {
            int m = Math.min(Ms[mi], E);
            System.err.printf("    top-%-4d (%2.0f%% of experts): %5.1f%% of routing%n",
                m, 100.0 * m / E, 100.0 * covSum[mi] / moeLayers);
        }
        System.err.println("  Interpretation: a hot-expert GPU cache helps when a small top-M captures most"
            + " routing (concentrated); it is wasted VRAM when routing is near-uniform (top-M ~ M/E).");
    }

    private static void selectTopK(float[] logits, int n, int k,
                                    int[] outIndices, float[] outValues) {
        Arrays.fill(outIndices, 0, k, -1);
        Arrays.fill(outValues, 0, k, Float.NEGATIVE_INFINITY);

        // Track min position persistently — only rescan when replaced
        int minPos = 0;
        float minVal = Float.NEGATIVE_INFINITY;

        for (int i = 0; i < n; i++) {
            if (logits[i] > minVal) {
                outValues[minPos] = logits[i];
                outIndices[minPos] = i;
                // Rescan for new minimum
                minPos = 0;
                minVal = outValues[0];
                for (int j = 1; j < k; j++) {
                    if (outValues[j] < minVal) {
                        minPos = j;
                        minVal = outValues[j];
                    }
                }
            }
        }
        // Robustness: a slot stays unfilled only when fewer than k logits beat -inf, i.e. the router
        // logits contained NaN (NaN > x is always false). Fill any such slot with a valid expert at a
        // negligible weight so downstream never reads expert -1 nor sums a NEGATIVE_INFINITY weight.
        for (int j = 0; j < k; j++) {
            if (outIndices[j] < 0) { outIndices[j] = 0; outValues[j] = -1e30f; }
        }
    }

    /**
     * GPT-OSS custom SwiGLU activation: alpha=1.702, limit=7.0, (up + 1).
     * output[i] = clamp(gate[i], max=7) * sigmoid(1.702 * clamp(gate[i], max=7)) * (clamp(up[i], -7, 7) + 1)
     * Result is stored back in gate[].
     */
    private static void swigluOai(float[] gate, float[] up, int size) {
        for (int i = 0; i < size; i++) {
            float x = Math.min(gate[i], 7.0f);
            float y = Math.max(-7.0f, Math.min(up[i], 7.0f));
            float glu = x / (1.0f + (float) Math.exp(-1.702f * x));
            gate[i] = glu * (y + 1.0f);
        }
    }

    /**
     * Softmax with attention sink: includes exp(sinkValue) in the denominator
     * but doesn't produce an attention weight for it (probability is discarded).
     */
    private static void softmaxWithSink(float[] x, int offset, int size, float sinkValue) {
        float max = sinkValue;
        for (int i = 0; i < size; i++) {
            max = Math.max(max, x[offset + i]);
        }
        float sum = (float) Math.exp(sinkValue - max); // sink's contribution to denominator
        for (int i = 0; i < size; i++) {
            x[offset + i] = (float) Math.exp(x[offset + i] - max);
            sum += x[offset + i];
        }
        float invSum = 1.0f / sum;
        for (int i = 0; i < size; i++) {
            x[offset + i] *= invSum;
        }
    }

    /** Add 1D bias vector to output array. */
    private static void addBias(float[] output, FloatTensor bias, int size) {
        for (int i = 0; i < size; i++) {
            output[i] += bias.getFloat(i);
        }
    }

    /** Add per-expert bias from a 2D bias tensor [size, expertCount]. */
    private static void addExpertBias(float[] output, FloatTensor bias2D, int expert, int size) {
        long offset = (long) expert * size;
        for (int i = 0; i < size; i++) {
            output[i] += bias2D.getFloat(offset + i);
        }
    }

    /**
     * Apply RMSNorm per-head.
     */
    private static void applyPerHeadNorm(float[] vec, float[] normWeights, int nHeads, int headSize, float eps) {
        if (normWeights.length == nHeads * headSize && nHeads > 1) {
            RMSNorm.apply(vec, vec, normWeights, nHeads * headSize, eps); // whole-projection norm
            return;
        }
        for (int h = 0; h < nHeads; h++) {
            int offset = h * headSize;
            float ss = 0f;
            for (int i = 0; i < headSize; i++) {
                ss += vec[offset + i] * vec[offset + i];
            }
            ss = 1.0f / (float) Math.sqrt(ss / headSize + eps);
            for (int i = 0; i < headSize; i++) {
                vec[offset + i] = vec[offset + i] * ss * normWeights[i];
            }
        }
    }

    public float[] prefill(Qwen3MoEState state, int[] tokens) {
        long t0 = System.currentTimeMillis();
        for (int i = 0; i < tokens.length - 1; i++) {
            forwardNoOutput(state, tokens[i], i);
            if (tokens.length > 10) {
                long elapsed = System.currentTimeMillis() - t0;
                System.out.printf("[prefill] token %d/%d (%.1fs)%n", i + 1, tokens.length, elapsed / 1000.0);
            }
        }
        float[] logits = forward(state, tokens[tokens.length - 1], tokens.length - 1);
        long total = System.currentTimeMillis() - t0;
        System.out.printf("[prefill] done: %d tokens in %.1fs%n", tokens.length, total / 1000.0);
        return logits;
    }

    public ModelConfig getConfig() { return config; }
}
