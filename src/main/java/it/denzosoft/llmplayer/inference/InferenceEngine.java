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
        RoPE rope = new RoPE(config.headSize(), ropeDimCount, maxSeqLen, config.ropeFreqBase(),
            config.ropeType(), ropeFreqFactors);
        // Gemma 3: local (sliding window) layers use theta=10000, global layers use the main theta
        RoPE ropeLocal = null;
        if (arch == ModelArchitecture.GEMMA3 && config.slidingWindow() > 0) {
            ropeLocal = new RoPE(config.headSize(), ropeDimCount, maxSeqLen, 10000f,
                config.ropeType(), ropeFreqFactors);
        } else if (arch == ModelArchitecture.GEMMA4 && config.slidingWindow() > 0) {
            // Gemma 4: both SWA and full use same freq factors (proportional RoPE)
            // Only theta differs: SWA=ropeFreqBaseSwa (10K), full=main theta (1M)
            ropeLocal = new RoPE(config.headSize(), ropeDimCount, maxSeqLen,
                config.ropeFreqBaseSwa(), config.ropeType(), ropeFreqFactors);
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

    private float[] forwardInternal(InferenceState state, int token, int position, boolean computeLogits) {
        int dim = config.embeddingLength();

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

        if (!computeLogits) {
            return null;
        }

        int vocabSize = config.vocabSize();

        if (!logitsDone) {
            // 3. Final RMSNorm
            VectorOpsFactory.get().rmsnorm(state.xb, state.x, normWeightCache, dim, config.normEps());

            // 4. Output projection: logits = output_weight * xb
            Arrays.fill(state.logits, 0);
            weights.output().matmulParallel(state.xb, state.logits, vocabSize, dim);
        }

        // 5. Logit scaling (skip if GPU already applied it via CudaForwardPass)
        // Command-R: multiply by logitScale
        // Granite: DIVIDE by logitScale (llama.cpp: 1.0f / f_logit_scale)
        if (logitScale > 0f && !logitsDone) {
            float scale = (config.architecture() == ModelArchitecture.GRANITE)
                ? (1.0f / logitScale) : logitScale;
            for (int i = 0; i < vocabSize; i++) {
                state.logits[i] *= scale;
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

        VectorOpsFactory.get().rmsnorm(state.xb, state.x, normWeightCache, dim, config.normEps());
        Arrays.fill(state.logits, 0);
        weights.output().matmulParallel(state.xb, state.logits, vocabSize, dim);

        return state.logits;
    }

    public ModelConfig getConfig() { return config; }
}
