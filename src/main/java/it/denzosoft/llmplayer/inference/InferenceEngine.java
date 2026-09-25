package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.ModelArchitecture;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.model.ModelWeights;
import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

import java.util.Arrays;

/**
 * Orchestrates the complete forward pass of the transformer model.
 *
 * Forward pass pipeline:
 * 1. Token embedding lookup
 * 2. For each layer: RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual
 * 3. Final RMSNorm
 * 4. Output projection -> logits
 */
public class InferenceEngine {

    private final ModelConfig config;
    private final ModelWeights weights;
    private final TransformerBlock block;
    private final Attention attention;
    private final float[] normWeightCache;
    private final int maxSeqLen;
    private final float finalLogitSoftCap;
    private final float logitScale;
    private final float embeddingScale;
    private final boolean useLayerNorm; // Command-R: centered LayerNorm instead of RMSNorm

    // GPU-resident forward pass (null if not available or not supported)
    private AutoCloseable gpuForwardPass;
    private boolean gpuChainEnabled;

    // Next-layer mmap prefetcher for lazy (>RAM) loads; null when not applicable
    private LayerPrefetcher layerPrefetcher;

    // Cached reflection Method handles for GPU hot path (avoids per-token getMethod lookups)
    private java.lang.reflect.Method cachedUploadX;
    private java.lang.reflect.Method cachedUpdateTokenParams;
    private java.lang.reflect.Method cachedUploadXAndUpdateParams;
    private java.lang.reflect.Method cachedForwardGraph;
    private java.lang.reflect.Method cachedForwardGraphArgmax;
    private java.lang.reflect.Method cachedForwardLayer;
    private java.lang.reflect.Method cachedForwardFinalLogits;
    private java.lang.reflect.Method cachedForwardFinalArgmax;
    private java.lang.reflect.Method cachedDownloadX;
    private int gpuLayerCount;  // number of layers on GPU (for partial offload)

    public InferenceEngine(ModelConfig config, ModelWeights weights, int maxSeqLen) {
        this(config, weights, maxSeqLen, null);
    }

    public InferenceEngine(ModelConfig config, ModelWeights weights, int maxSeqLen, float[] ropeFreqFactors) {
        this.config = config;
        this.weights = weights;
        this.maxSeqLen = maxSeqLen;

        int ropeDimCount = config.ropeDimensionCount();
        ModelArchitecture arch = config.architecture();
        // Linear RoPE scaling factor (Gemma 3 4B has rope.scaling.type=linear, factor=8.0).
        // For yarn-scaled models, ropeScalingFactor is consumed via YarnParams instead.
        float linearFreqScale = (config.ropeScalingFactor() > 1.0f && config.yarnLogMultiplier() == 0)
            ? config.ropeScalingFactor() : 1.0f;
        RoPE rope = new RoPE(config.headSize(), ropeDimCount, maxSeqLen, config.ropeFreqBase(),
            config.ropeType(), ropeFreqFactors, null, linearFreqScale);
        // Gemma 3: local (sliding window) layers use theta=10000, global layers use the main theta.
        // Linear scale is NOT applied to the SWA local rope per llama.cpp behavior — short SWA
        // doesn't need rope scaling.
        RoPE ropeLocal = null;
        if (arch == ModelArchitecture.GEMMA3 && config.slidingWindow() > 0) {
            ropeLocal = new RoPE(config.headSize(), ropeDimCount, maxSeqLen, 10000f,
                config.ropeType(), ropeFreqFactors);
        } else if (arch == ModelArchitecture.GEMMA4 && config.slidingWindow() > 0) {
            // Gemma 4: both SWA and full use same freq factors (proportional RoPE)
            // Only theta differs: SWA=ropeFreqBaseSwa (10K), full=main theta (1M)
            ropeLocal = new RoPE(config.headSize(), ropeDimCount, maxSeqLen,
                config.ropeFreqBaseSwa(), config.ropeType(), ropeFreqFactors);
        } else if (arch == ModelArchitecture.SPARK2_5) {
            // Spark2.5: SWA layers rotate rope.dimension_count_swa dims (all of them) at
            // theta_swa (10K); full layers rotate rope.dimension_count dims at the main theta.
            ropeLocal = new RoPE(config.headSize(), config.ropeDimCountSwa(), maxSeqLen,
                config.ropeFreqBaseSwa(), config.ropeType(), null);
        }
        this.attention = new Attention(config, rope, ropeLocal);
        attention.initNormCachesIfNeeded(weights.layers());
        SwiGLUFFN ffn = new SwiGLUFFN(config);
        this.block = new TransformerBlock(config, attention, ffn, weights.layers());

        this.finalLogitSoftCap = config.finalLogitSoftCap();
        this.logitScale = config.logitScale();

        // Embedding scaling: Gemma uses sqrt(dim), Granite uses explicit metadata value
        if (config.embeddingScale() > 0f) {
            this.embeddingScale = config.embeddingScale(); // Granite: 12.0
        } else if (arch == ModelArchitecture.GEMMA2 || arch == ModelArchitecture.GEMMA3) {
            this.embeddingScale = (float) Math.sqrt(config.embeddingLength());
        } else {
            this.embeddingScale = 0f;
        }

        // Pre-cache output norm weights
        int dim = config.embeddingLength();
        this.normWeightCache = RMSNorm.cacheWeights(weights.outputNorm(), dim);

        this.useLayerNorm = config.useLayerNorm();
    }

    /**
     * Try to initialize GPU-resident forward pass via reflection.
     * Called by LLMEngine after construction when GPU is available.
     * The bufferManager must be an instance of GpuBufferManager from java21/.
     */
    public void tryInitGpuForwardPass(Object bufferManager) {
        if (!gpuChainEnabled) return;

        // Try CUDA forward pass first
        if (tryInitForwardPass("it.denzosoft.llmplayer.inference.CudaForwardPass", bufferManager, "CUDA")) {
            return;
        }
        // Fall back to OpenCL forward pass
        tryInitForwardPass("it.denzosoft.llmplayer.inference.GpuForwardPass", bufferManager, "OpenCL");
    }

    private boolean tryInitForwardPass(String className, Object bufferManager, String label) {
        try {
            Class<?> fpClass = Class.forName(className);
            java.lang.reflect.Method isSupported = fpClass.getMethod("isSupported",
                ModelConfig.class, ModelWeights.class);
            boolean supported = (boolean) isSupported.invoke(null, config, weights);
            if (!supported) {
                System.out.println("GPU chain (" + label + "): model not supported (requires pre-norm dense with separate Q/K/V, at least 1 GPU layer)");
                return false;
            }
            // Try 5-param constructor (with Attention + maxSeqLen for GPU attention)
            java.lang.reflect.Constructor<?>[] ctors = fpClass.getConstructors();
            if (ctors.length > 0 && ctors[0].getParameterCount() == 5) {
                gpuForwardPass = (AutoCloseable) ctors[0]
                    .newInstance(config, weights, bufferManager, attention, maxSeqLen);
            } else {
                gpuForwardPass = (AutoCloseable) ctors[0]
                    .newInstance(config, weights, bufferManager);
            }
            // Get GPU layer count for partial offload support
            try {
                java.lang.reflect.Method getGpuLayerCount = fpClass.getMethod("getGpuLayerCount");
                gpuLayerCount = (int) getGpuLayerCount.invoke(gpuForwardPass);
            } catch (NoSuchMethodException ignored) {
                gpuLayerCount = config.blockCount(); // fallback: assume full offload
            }
            if (gpuLayerCount < config.blockCount()) {
                System.out.println("GPU chain: enabled — " + label + " partial offload (" + gpuLayerCount + "/" + config.blockCount() + " layers on GPU)");
            } else {
                System.out.println("GPU chain: enabled — " + label + " GPU-resident forward pass active");
            }
            // Cache Method handles for hot path (avoids per-token getMethod overhead)
            cacheGpuMethods(fpClass);
            return true;
        } catch (ClassNotFoundException e) {
            // Not on classpath — expected on Java 8 or when backend not available
            return false;
        } catch (Exception e) {
            System.out.println("GPU chain (" + label + "): initialization failed — " + e.getMessage());
            if ("true".equals(System.getProperty("cuda.debug", "false"))) e.printStackTrace();
            return false;
        }
    }

    /**
     * Drop the GPU-resident forward pass, if any, so that every layer runs through the CPU block
     * (whose tensors may still be GPU-backed). Needed before prefilling vision embeddings, which only
     * the CPU path supports; must be called before any state is created.
     */
    public void disableGpuForwardPass() {
        if (gpuForwardPass != null) {
            try { gpuForwardPass.close(); } catch (Exception ignored) { }
            gpuForwardPass = null;
            System.out.println("GPU chain: disabled (vision input needs the CPU layer path)");
        }
    }

    /**
     * Enable or disable GPU kernel chaining.
     */
    public void setGpuChainEnabled(boolean enabled) {
        this.gpuChainEnabled = enabled;
    }

    /**
     * Attach a next-layer mmap prefetcher (lazy >RAM loads). The CPU layer loop kicks an async
     * page-in of layer N+1 while computing layer N, overlapping disk I/O with compute.
     */
    public void setLayerPrefetcher(LayerPrefetcher prefetcher) {
        this.layerPrefetcher = prefetcher;
    }

    /**
     * Create a new inference state for this model.
     */
    public InferenceState createState(int maxSeqLen) {
        return new InferenceState(config, maxSeqLen);
    }

    /**
     * Run forward pass for a single token at the given position.
     * Returns logits array (part of InferenceState, do not modify externally).
     */
    public float[] forward(InferenceState state, int token, int position) {
        return forwardInternal(state, token, position, true);
    }

    /**
     * Run forward pass through transformer layers only (no output projection).
     * Used during prefill for all tokens except the last, since logits are only
     * needed for the final token. Saves vocabSize * dim multiply-adds per token.
     */
    public void forwardNoOutput(InferenceState state, int token, int position) {
        forwardInternal(state, token, position, false);
    }

    /** Tokens per batched-prefill chunk; bounds the per-token buffers ({@link PrefillBatch}). */
    private static final int PREFILL_BATCH = Integer.getInteger("prefill.batch", 64);
    private static final boolean PREFILL_BATCHED =
        !"false".equalsIgnoreCase(System.getProperty("prefill.batched", "true"));

    /**
     * Prefill {@code tokens[fromPos..toPos)} at positions {@code fromPos..toPos-1} and return the
     * logits of the last one.
     *
     * <p>When the model allows it (see {@link #canBatchPrefill}) the prompt is processed layer by
     * layer in chunks of {@link #PREFILL_BATCH} tokens, and every projection runs as one
     * multi-token matmul: each weight row is read from memory once per chunk instead of once per
     * token, and the SIMD kernels dequantise each weight block once for several tokens. Attention
     * still runs token by token in position order inside each layer, so causality and the
     * per-token math are unchanged; only the floating-point summation order inside the matmul
     * kernels differs. Disable with {@code -Dprefill.batched=false}.
     */
    public float[] forwardPrefill(InferenceState state, int[] tokens, int fromPos, int toPos) {
        int count = toPos - fromPos;
        if (count <= 0) return null;
        if (count == 1 || !canBatchPrefill()) {
            float[] logits = null;
            for (int i = fromPos; i < toPos; i++) {
                if (i < toPos - 1) forwardNoOutput(state, tokens[i], i);
                else logits = forward(state, tokens[i], i);
            }
            return logits;
        }

        warmDecodeKernels();
        int dim = config.embeddingLength();
        PrefillBatch b = state.prefillBatch;
        if (b == null) {
            b = new PrefillBatch(config, Math.max(1, PREFILL_BATCH));
            state.prefillBatch = b;
        }
        for (int base = fromPos; base < toPos; base += b.capacity) {
            int n = Math.min(b.capacity, toPos - base);
            for (int t = 0; t < n; t++) embed(tokens[base + t], b.x[t]);
            for (int layer = 0; layer < config.blockCount(); layer++) {
                block.forwardBatch(b, state, weights.layers()[layer], layer, base, n);
                if (config.isLoopBoundary(layer)) {
                    for (int t = 0; t < n; t++) loopBoundaryNorm(b.x[t]);
                }
            }
            if (base + n == toPos) System.arraycopy(b.x[n - 1], 0, state.x, 0, dim);
        }
        return finishLogits(state, false);
    }

    /**
     * Prefill precomputed input embeddings (vision tokens) at KV slots {@code kvStart ..
     * kvStart + n - 1}. Each row holds the embedding ({@code dim} floats) followed by
     * {@code deepstackCount} deepstack vectors of the same width, which are added to the residual
     * stream after layers 0 .. deepstackCount-1 (llama.cpp qwen3vl.cpp). Rope positions come from
     * {@code state.mrope}. Logits are not computed; the last token's residual stream is left in
     * {@code state.x}.
     */
    public void prefillEmbeddings(InferenceState state, float[][] embds, int kvStart, int deepstackCount) {
        int dim = config.embeddingLength();
        int n = embds.length;
        int nDs = Math.min(deepstackCount, config.nDeepstackLayers() > 0 ? config.nDeepstackLayers() : deepstackCount);
        if (canBatchPrefill()) {
            warmDecodeKernels();
            PrefillBatch b = state.prefillBatch;
            if (b == null) {
                b = new PrefillBatch(config, Math.max(1, PREFILL_BATCH));
                state.prefillBatch = b;
            }
            for (int base = 0; base < n; base += b.capacity) {
                int m = Math.min(b.capacity, n - base);
                for (int t = 0; t < m; t++) System.arraycopy(embds[base + t], 0, b.x[t], 0, dim);
                runBatchLayers(b, state, embds, base, m, kvStart, nDs);
                // leave the last token's residual stream in state.x (see normedHidden)
                if (base + m == n) System.arraycopy(b.x[m - 1], 0, state.x, 0, dim);
            }
            return;
        }
        for (int t = 0; t < n; t++) {
            System.arraycopy(embds[t], 0, state.x, 0, dim);
            for (int layer = 0; layer < config.blockCount(); layer++) {
                block.forward(state, weights.layers()[layer], layer, kvStart + t);
                if (layer < nDs) VectorOpsFactory.get().saxpy(1f, embds[t], (layer + 1) * dim, state.x, 0, dim);
            }
        }
    }

    private void runBatchLayers(PrefillBatch b, InferenceState state, float[][] embds, int base, int m,
                                int kvStart, int nDs) {
        int dim = config.embeddingLength();
        for (int layer = 0; layer < config.blockCount(); layer++) {
            block.forwardBatch(b, state, weights.layers()[layer], layer, kvStart + base, m);
            if (layer < nDs) {
                for (int t = 0; t < m; t++) {
                    VectorOpsFactory.get().saxpy(1f, embds[base + t], (layer + 1) * dim, b.x[t], 0, dim);
                }
            }
        }
    }

    /**
     * One token from a precomputed input embedding at KV slot {@code pos}; the residual stream is
     * left in {@code state.x} (read it with {@link #normedHidden}). Logits are not computed.
     */
    public void forwardEmbedding(InferenceState state, float[] embd, int pos) {
        System.arraycopy(embd, 0, state.x, 0, config.embeddingLength());
        for (int layer = 0; layer < config.blockCount(); layer++) {
            block.forward(state, weights.layers()[layer], layer, pos);
        }
    }

    /** The final-norm output of the residual stream in {@code state.x} (llama.cpp t_embd). */
    public void normedHidden(InferenceState state, float[] out) {
        int dim = config.embeddingLength();
        if (useLayerNorm) {
            LayerNorm.apply(out, state.x, normWeightCache, dim, config.normEps());
        } else {
            VectorOpsFactory.get().rmsnorm(out, state.x, normWeightCache, dim, config.normEps());
        }
    }

    public ModelWeights getWeights() { return weights; }

    /** See {@link FloatTensor#warmUpRows}: batched prefill skips the kernels decode will use. */
    private void warmDecodeKernels() {
        int dim = config.embeddingLength();
        int qDim = config.headCount() * config.headSize();
        int ffn = config.intermediateSize();
        it.denzosoft.llmplayer.model.TransformerLayerWeights l = weights.layers()[0];
        FloatTensor.warmUpRows(l.wq(), qDim, dim);
        FloatTensor.warmUpRows(l.wqkv(), qDim + 2 * config.kvDim(), dim);
        FloatTensor.warmUpRows(l.wk(), config.kvDim(), dim);
        FloatTensor.warmUpRows(l.wv(), config.kvDim(), dim);
        FloatTensor.warmUpRows(l.wo(), dim, qDim);
        FloatTensor.warmUpRows(l.wGate(), ffn, dim);
        FloatTensor.warmUpRows(l.wUp(), ffn, dim);
        FloatTensor.warmUpRows(l.wDown(), dim, ffn);
        FloatTensor.warmUpRows(weights.output(), config.vocabSize(), dim);
    }

    /**
     * Batched prefill needs the CPU layer loop (no GPU-resident pass, no lazy >RAM prefetcher),
     * the CPU matmul pool (it is disabled once a GPU backend is initialised, so GPU tensors never
     * reach the batched kernels), and a layer shape {@link TransformerBlock#forwardBatch} covers.
     */
    private boolean canBatchPrefill() {
        if (!PREFILL_BATCHED || gpuForwardPass != null || layerPrefetcher != null
                || !it.denzosoft.llmplayer.tensor.MatmulPool.enabled()) {
            return false;
        }
        for (int layer = 0; layer < config.blockCount(); layer++) {
            if (!block.supportsBatch(weights.layers()[layer], layer)) return false;
        }
        return true;
    }

    private float[] forwardInternal(InferenceState state, int token, int position, boolean computeLogits) {
        int dim = config.embeddingLength();
        long t0 = 0;

        // 1. Token embedding lookup
        if (cpuProfile) t0 = System.nanoTime();
        embed(token, state.x);
        if (cpuProfile) profEmbedNs += System.nanoTime() - t0;

        // 2. Forward through all transformer layers
        boolean logitsDone = false;
        if (gpuForwardPass != null) {
            logitsDone = forwardGpu(state, position);
        } else {
            for (int layer = 0; layer < config.blockCount(); layer++) {
                // Lazy (>RAM) load: kick the async page-in of layer N+1 while computing layer N
                if (layerPrefetcher != null) layerPrefetcher.prefetchLayer(layer + 1);
                block.forward(state, weights.layers()[layer], layer, position);
                if (config.isLoopBoundary(layer)) loopBoundaryNorm(state.x);
            }
        }

        if (!computeLogits) {
            return null;
        }
        return finishLogits(state, logitsDone);
    }

    /**
     * Nanbeige looped depth: between two loops over the shared layers the residual stream is
     * replaced by its output-norm (llama.cpp nanbeige.cpp, unless skip_loop_final_norm).
     */
    private void loopBoundaryNorm(float[] x) {
        VectorOpsFactory.get().rmsnorm(x, x, normWeightCache, config.embeddingLength(), config.normEps());
    }

    /** Token embedding into {@code out}, including the Gemma/Granite embedding scale. */
    private void embed(int token, float[] out) {
        int dim = config.embeddingLength();
        for (int i = 0; i < dim; i++) {
            out[i] = weights.tokenEmbedding().getFloat((long) token * dim + i);
        }
        // Gemma: scale embedding by sqrt(dim)
        if (embeddingScale > 0f) {
            for (int i = 0; i < dim; i++) {
                out[i] *= embeddingScale;
            }
        }
    }

    /** Final norm, output projection and logit post-processing for the residual stream in {@code state.x}. */
    private float[] finishLogits(InferenceState state, boolean logitsDone) {
        int dim = config.embeddingLength();
        long t0 = 0;
        int vocabSize = config.vocabSize();

        if (!logitsDone) {
            if (cpuProfile) t0 = System.nanoTime();
            if (useLayerNorm) {
                LayerNorm.apply(state.xb, state.x, normWeightCache, dim, config.normEps());
            } else {
                VectorOpsFactory.get().rmsnorm(state.xb, state.x, normWeightCache, dim, config.normEps());
            }
            if (cpuProfile) { long t1 = System.nanoTime(); profFinalNormNs += t1 - t0; t0 = t1; }

            Arrays.fill(state.logits, 0);
            weights.output().matmulParallel(state.xb, state.logits, vocabSize, dim);
            if (cpuProfile) profOutputNs += System.nanoTime() - t0;
            // Output bias and logit scaling (skipped if the GPU pass already applied them)
            biasAndScaleLogits(state.logits);
        }

        // 6. Logit soft-capping (Gemma2/3): logits = softCap * tanh(logits / softCap)
        softCapLogits(state.logits);

        if (cpuProfile) {
            profEngineTokenCount++;
            if (profEngineTokenCount % 10 == 0) printEngineProfile();
        }

        return state.logits;
    }

    /**
     * Output bias (E12: optional output.bias for Qwen2 variants — see llama.cpp qwen2.cpp:119-121)
     * and logit scaling (Command-R multiplies by logitScale; Granite divides by it, as llama.cpp's
     * 1.0f / f_logit_scale) of freshly projected logits.
     */
    private void biasAndScaleLogits(float[] logits) {
        int vocabSize = config.vocabSize();
        if (weights.outputBias() != null) {
            FloatTensor ob = weights.outputBias();
            for (int i = 0; i < vocabSize; i++) logits[i] += ob.getFloat(i);
        }
        if (logitScale > 0f) {
            float scale = (config.architecture() == ModelArchitecture.GRANITE)
                ? (1.0f / logitScale) : logitScale;
            for (int i = 0; i < vocabSize; i++) {
                logits[i] *= scale;
            }
        }
    }

    /** Logit soft-capping (Gemma2/3): logits = softCap * tanh(logits / softCap). */
    private void softCapLogits(float[] logits) {
        if (finalLogitSoftCap > 0f) {
            int vocabSize = config.vocabSize();
            for (int i = 0; i < vocabSize; i++) {
                logits[i] = finalLogitSoftCap * (float) Math.tanh(logits[i] / finalLogitSoftCap);
            }
        }
    }

    /**
     * Multi-token forward that keeps every position's logits, for speculative verification: feeds
     * {@code tokens[i]} at position {@code startPos + i} and returns {@code [n][vocabSize]} fresh
     * logits, where row {@code i} predicts the token after {@code tokens[i]}.
     *
     * <p>With batched prefill available (see {@link #canBatchPrefill}) the tokens go through the
     * layers as one multi-token pass exactly as in {@link #forwardPrefill}, and the output projection
     * — the largest single weight — is also one multi-token matmul, so verifying K draft tokens reads
     * the weights about once instead of K times. Otherwise it is K one-token forwards.
     */
    public float[][] forwardBatchLogits(InferenceState state, int[] tokens, int startPos) {
        int n = tokens.length;
        int dim = config.embeddingLength();
        int vocabSize = config.vocabSize();
        float[][] out = new float[n][];
        if (n == 1 || !canBatchPrefill()) {
            for (int i = 0; i < n; i++) {
                out[i] = Arrays.copyOf(forward(state, tokens[i], startPos + i), vocabSize);
            }
            return out;
        }

        warmDecodeKernels();
        PrefillBatch b = state.prefillBatch;
        if (b == null) {
            b = new PrefillBatch(config, Math.max(1, PREFILL_BATCH));
            state.prefillBatch = b;
        }
        for (int base = 0; base < n; base += b.capacity) {
            int m = Math.min(b.capacity, n - base);
            for (int t = 0; t < m; t++) embed(tokens[base + t], b.x[t]);
            for (int layer = 0; layer < config.blockCount(); layer++) {
                block.forwardBatch(b, state, weights.layers()[layer], layer, startPos + base, m);
                if (config.isLoopBoundary(layer)) {
                    for (int t = 0; t < m; t++) loopBoundaryNorm(b.x[t]);
                }
            }
            float[][] logits = new float[m][];
            for (int t = 0; t < m; t++) {
                if (useLayerNorm) {
                    LayerNorm.apply(b.xb[t], b.x[t], normWeightCache, dim, config.normEps());
                } else {
                    VectorOpsFactory.get().rmsnorm(b.xb[t], b.x[t], normWeightCache, dim, config.normEps());
                }
                logits[t] = new float[vocabSize];
                out[base + t] = logits[t];
            }
            FloatTensor.matmulBatchParallel(weights.output(), b.xb, logits, m, vocabSize, dim);
            for (int t = 0; t < m; t++) {
                biasAndScaleLogits(logits[t]);
                softCapLogits(logits[t]);
            }
            if (base + m == n) System.arraycopy(b.x[m - 1], 0, state.x, 0, dim);
        }
        return out;
    }

    private final boolean cpuProfile = "true".equals(System.getProperty("cpu.profile"));
    private long profEmbedNs, profFinalNormNs, profOutputNs;
    private int profEngineTokenCount;

    private void printEngineProfile() {
        int n = profEngineTokenCount;
        double ms = 1e6;
        System.out.printf("[cpu-profile Engine] %d tokens, per-token avg (ms): embed=%.2f final_norm=%.2f output_proj=%.2f%n",
            n, profEmbedNs / ms / n, profFinalNormNs / ms / n, profOutputNs / ms / n);
    }

    /**
     * Cache reflection Method handles once at init time.
     * Eliminates per-token getMethod() overhead in the hot path.
     */
    private void cacheGpuMethods(Class<?> fpClass) {
        try {
            cachedUploadX = fpClass.getMethod("uploadX", float[].class);
            cachedForwardLayer = fpClass.getMethod("forwardLayer",
                InferenceState.class,
                it.denzosoft.llmplayer.model.TransformerLayerWeights.class,
                int.class, int.class, Attention.class);
            cachedDownloadX = fpClass.getMethod("downloadX", float[].class);
            try { cachedUpdateTokenParams = fpClass.getMethod("updateTokenParams", int.class); }
            catch (NoSuchMethodException ignored) {}
            try { cachedUploadXAndUpdateParams = fpClass.getMethod("uploadXAndUpdateParams", float[].class, int.class); }
            catch (NoSuchMethodException ignored) {}
            try { cachedForwardGraph = fpClass.getMethod("forwardGraph", float[].class); }
            catch (NoSuchMethodException ignored) {}
            try { cachedForwardGraphArgmax = fpClass.getMethod("forwardGraphArgmax"); }
            catch (NoSuchMethodException ignored) {}
            try { cachedForwardFinalLogits = fpClass.getMethod("forwardFinalLogits", float[].class); }
            catch (NoSuchMethodException ignored) {}
            try { cachedForwardFinalArgmax = fpClass.getMethod("forwardFinalArgmax"); }
            catch (NoSuchMethodException ignored) {}
        } catch (NoSuchMethodException e) {
            throw new RuntimeException("GPU forward pass missing required methods", e);
        }
    }

    /**
     * GPU-resident forward pass through all layers.
     * Uses cached reflection Method handles to minimize per-token overhead.
     * Returns true if logits were computed on GPU (caller skips CPU RMSNorm + matmul).
     */
    private boolean forwardGpu(InferenceState state, int position) {
        try {
            // Combined upload: embedding + token params in single cuMemcpyHtoD
            if (cachedUploadXAndUpdateParams != null) {
                cachedUploadXAndUpdateParams.invoke(gpuForwardPass, state.x, position);
            } else {
                cachedUploadX.invoke(gpuForwardPass, state.x);
                if (cachedUpdateTokenParams != null) {
                    cachedUpdateTokenParams.invoke(gpuForwardPass, position);
                }
            }

            // Try CUDA graph mode first (all GPU layers + output in single API call)
            if (cachedForwardGraph != null && gpuLayerCount == config.blockCount()) {
                boolean done = (boolean) cachedForwardGraph.invoke(gpuForwardPass, state.logits);
                if (done) return true;
            }

            // Per-layer mode: run GPU layers via CudaForwardPass
            for (int layer = 0; layer < gpuLayerCount; layer++) {
                cachedForwardLayer.invoke(gpuForwardPass, state, weights.layers()[layer], layer, position, attention);
            }

            // If all layers are on GPU, try final RMSNorm + output projection on GPU
            if (gpuLayerCount == config.blockCount() && cachedForwardFinalLogits != null) {
                boolean done = (boolean) cachedForwardFinalLogits.invoke(gpuForwardPass, state.logits);
                if (done) return true;
            }

            // Download X from GPU for CPU layers or final steps
            cachedDownloadX.invoke(gpuForwardPass, state.x);

            // Run remaining CPU layers (partial offload)
            for (int layer = gpuLayerCount; layer < config.blockCount(); layer++) {
                block.forward(state, weights.layers()[layer], layer, position);
            }

            return false;
        } catch (Exception e) {
            Throwable cause = e;
            while (cause.getCause() != null) cause = cause.getCause();
            System.err.println("GPU chain: forward failed, permanently disabling GPU forward pass — " + cause);
            cause.printStackTrace(System.err);
            gpuForwardPass = null;  // Prevent repeated GPU failures on subsequent tokens
            for (int layer = 0; layer < config.blockCount(); layer++) {
                block.forward(state, weights.layers()[layer], layer, position);
            }
            return false;
        }
    }

    /**
     * Prefill: process multiple tokens (prompt) and return logits for the last token.
     * Skips the output projection (final RMSNorm + vocabSize matmul) for all tokens
     * except the last, since only the last token's logits are needed for generation.
     * This saves vocabSize * dim multiply-adds per skipped token.
     */
    public float[] prefill(InferenceState state, int[] tokens) {
        // Process all but the last token without computing logits
        for (int i = 0; i < tokens.length - 1; i++) {
            forwardNoOutput(state, tokens[i], i);
        }
        // Only compute logits for the last token
        return forward(state, tokens[tokens.length - 1], tokens.length - 1);
    }

    /**
     * Run forward pass with a limited number of layers (for debugging).
     */
    public float[] forwardLayers(InferenceState state, int token, int position, int numLayers) {
        int dim = config.embeddingLength();
        int vocabSize = config.vocabSize();

        for (int i = 0; i < dim; i++) {
            state.x[i] = weights.tokenEmbedding().getFloat((long) token * dim + i);
        }

        for (int layer = 0; layer < numLayers; layer++) {
            block.forward(state, weights.layers()[layer], layer, position);
        }

        if (useLayerNorm) {
            LayerNorm.apply(state.xb, state.x, normWeightCache, dim, config.normEps());
        } else {
            VectorOpsFactory.get().rmsnorm(state.xb, state.x, normWeightCache, dim, config.normEps());
        }
        Arrays.fill(state.logits, 0);
        weights.output().matmulParallel(state.xb, state.logits, vocabSize, dim);
        // E12: optional output.bias
        if (weights.outputBias() != null) {
            FloatTensor ob = weights.outputBias();
            for (int i = 0; i < vocabSize; i++) state.logits[i] += ob.getFloat(i);
        }

        return state.logits;
    }

    public ModelConfig getConfig() { return config; }
}
