package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.ModelArchitecture;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.model.ModelWeights;
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

    // GPU-resident forward pass (null if not available or not supported)
    private AutoCloseable gpuForwardPass;
    private boolean gpuChainEnabled;

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

    public InferenceEngine(ModelConfig config, ModelWeights weights, int maxSeqLen) {
        this(config, weights, maxSeqLen, null);
    }

    public InferenceEngine(ModelConfig config, ModelWeights weights, int maxSeqLen, float[] ropeFreqFactors) {
        this.config = config;
        this.weights = weights;
        this.maxSeqLen = maxSeqLen;

        int ropeDimCount = config.ropeDimensionCount();
        ModelArchitecture arch = config.architecture();
        RoPE rope = new RoPE(config.headSize(), ropeDimCount, maxSeqLen, config.ropeFreqBase(),
            config.ropeType(), ropeFreqFactors);
        // Gemma 3: local (sliding window) layers use theta=10000, global layers use the main theta
        RoPE ropeLocal = null;
        if (arch == ModelArchitecture.GEMMA3 && config.slidingWindow() > 0) {
            ropeLocal = new RoPE(config.headSize(), ropeDimCount, maxSeqLen, 10000f,
                config.ropeType(), ropeFreqFactors);
        }
        this.attention = new Attention(config, rope, ropeLocal);
        attention.initNormCachesIfNeeded(weights.layers());
        SwiGLUFFN ffn = new SwiGLUFFN(config);
        this.block = new TransformerBlock(config, attention, ffn, weights.layers());

        this.finalLogitSoftCap = config.finalLogitSoftCap();
        this.logitScale = config.logitScale();

        // Gemma models scale embeddings by sqrt(dim)
        if (arch == ModelArchitecture.GEMMA2 || arch == ModelArchitecture.GEMMA3) {
            this.embeddingScale = (float) Math.sqrt(config.embeddingLength());
        } else {
            this.embeddingScale = 0f;
        }

        // Pre-cache output norm weights
        int dim = config.embeddingLength();
        this.normWeightCache = RMSNorm.cacheWeights(weights.outputNorm(), dim);
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
                System.out.println("GPU chain (" + label + "): model not supported (requires pre-norm dense with separate Q/K/V, full GPU offload)");
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
            System.out.println("GPU chain: enabled — " + label + " GPU-resident forward pass active");
            // Cache Method handles for hot path (avoids per-token getMethod overhead)
            cacheGpuMethods(fpClass);
            return true;
        } catch (ClassNotFoundException e) {
            // Not on classpath — expected on Java 8 or when backend not available
            return false;
        } catch (Exception e) {
            System.out.println("GPU chain (" + label + "): initialization failed — " + e.getMessage());
            return false;
        }
    }

    /**
     * Enable or disable GPU kernel chaining.
     */
    public void setGpuChainEnabled(boolean enabled) {
        this.gpuChainEnabled = enabled;
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
        int dim = config.embeddingLength();
        int vocabSize = config.vocabSize();

        // 1. Token embedding lookup
        for (int i = 0; i < dim; i++) {
            state.x[i] = weights.tokenEmbedding().getFloat((long) token * dim + i);
        }

        // Gemma: scale embedding by sqrt(dim)
        if (embeddingScale > 0f) {
            for (int i = 0; i < dim; i++) {
                state.x[i] *= embeddingScale;
            }
        }

        // 2. Forward through all transformer layers
        boolean logitsDone = false;
        if (gpuForwardPass != null) {
            logitsDone = forwardGpu(state, position);
        } else {
            for (int layer = 0; layer < config.blockCount(); layer++) {
                block.forward(state, weights.layers()[layer], layer, position);
            }
        }

        if (!logitsDone) {
            // 3. Final RMSNorm
            VectorOpsFactory.get().rmsnorm(state.xb, state.x, normWeightCache, dim, config.normEps());

            // 4. Output projection: logits = output_weight * xb
            Arrays.fill(state.logits, 0);
            weights.output().matmulParallel(state.xb, state.logits, vocabSize, dim);
        }

        // 5. Logit scaling (Command-R): logits *= logitScale
        if (logitScale > 0f) {
            for (int i = 0; i < vocabSize; i++) {
                state.logits[i] *= logitScale;
            }
        }

        // 6. Logit soft-capping (Gemma2/3): logits = softCap * tanh(logits / softCap)
        if (finalLogitSoftCap > 0f) {
            for (int i = 0; i < vocabSize; i++) {
                state.logits[i] = finalLogitSoftCap * (float) Math.tanh(state.logits[i] / finalLogitSoftCap);
            }
        }

        return state.logits;
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

            // Try CUDA graph mode first (all layers + output in single API call)
            if (cachedForwardGraph != null) {
                boolean done = (boolean) cachedForwardGraph.invoke(gpuForwardPass, state.logits);
                if (done) return true;
            }

            // Fall back to per-layer mode
            for (int layer = 0; layer < config.blockCount(); layer++) {
                cachedForwardLayer.invoke(gpuForwardPass, state, weights.layers()[layer], layer, position, attention);
            }

            // Try final RMSNorm + output projection on GPU
            if (cachedForwardFinalLogits != null) {
                boolean done = (boolean) cachedForwardFinalLogits.invoke(gpuForwardPass, state.logits);
                if (done) return true;
            }

            // Fall back: download X and let CPU handle final steps
            cachedDownloadX.invoke(gpuForwardPass, state.x);
            return false;
        } catch (Exception e) {
            Throwable cause = e;
            while (cause.getCause() != null) cause = cause.getCause();
            System.err.println("GPU chain: forward failed, falling back to CPU — " + cause);
            cause.printStackTrace(System.err);
            for (int layer = 0; layer < config.blockCount(); layer++) {
                block.forward(state, weights.layers()[layer], layer, position);
            }
            return false;
        }
    }

    /**
     * Prefill: process multiple tokens (prompt) and return logits for the last token.
     */
    public float[] prefill(InferenceState state, int[] tokens) {
        float[] logits = null;
        for (int i = 0; i < tokens.length; i++) {
            logits = forward(state, tokens[i], i);
        }
        return logits;
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

        VectorOpsFactory.get().rmsnorm(state.xb, state.x, normWeightCache, dim, config.normEps());
        Arrays.fill(state.logits, 0);
        weights.output().matmulParallel(state.xb, state.logits, vocabSize, dim);

        return state.logits;
    }

    public ModelConfig getConfig() { return config; }
}
