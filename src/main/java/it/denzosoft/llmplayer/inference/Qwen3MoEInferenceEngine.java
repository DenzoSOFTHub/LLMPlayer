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
    private final int slidingWindow; // ISWA: 0 = disabled, >0 = window size for SWA layers
    private final int noRopeLayerInterval; // Llama4 iRoPE: 0 = all layers use RoPE

    // Cached attention sinks per layer (GPT-OSS): float[blockCount][headCount]
    private final float[][] cachedAttnSinks;

    // Expert GPU cache (loaded via reflection from java21, null if unavailable)
    private Object expertGpuCache;
    private java.lang.reflect.Method computeExpertsMethod;
    private int currentLayer; // tracks current layer for GPU cache keying

    private final boolean cpuProfile = "true".equals(System.getProperty("cpu.profile"));
    private long profAttnNormNs, profAttnNs, profFfnNormNs, profDenseFfnNs, profMoeFfnNs, profResidualNs, profOutputNs;
    private int profTokenCount;

    public Qwen3MoEInferenceEngine(ModelConfig config, Qwen3MoEWeights weights, int maxSeqLen,
                                    float[] ropeFreqFactors) {
        this.config = config;
        this.weights = weights;
        this.maxSeqLen = maxSeqLen;
        this.isGptOss = config.architecture() == ModelArchitecture.GPT_OSS;
        this.slidingWindow = config.slidingWindow();
        this.noRopeLayerInterval = config.noRopeLayerInterval();

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
                cachedQNorm[i] = RMSNorm.cacheWeights(weights.layers()[i].qNorm(), headSize);
                cachedKNorm[i] = RMSNorm.cacheWeights(weights.layers()[i].kNorm(), headSize);
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
        try {
            int expertFfnDim = config.expertFfnLength();
            int dim = config.embeddingLength();
            long elementsPerSlice = (long) expertFfnDim * dim;
            int blockSize = 32;
            int blockBytes = 17; // MXFP4
            long bytesPerSlice = (elementsPerSlice / blockSize) * blockBytes;
            int maxSlots = (int) (maxCacheBytes / bytesPerSlice);
            if (maxSlots < 12) { // need at least 3 projections × 4 experts
                System.out.println("  Expert GPU cache: not enough VRAM (" + maxSlots + " slots)");
                return;
            }

            Class<?> cacheClass = Class.forName("it.denzosoft.llmplayer.gpu.ExpertGpuCache");
            Class<?> ctxClass = Class.forName("it.denzosoft.llmplayer.gpu.CudaContext");
            Object cache = cacheClass.getConstructor(ctxClass, int.class, long.class)
                .newInstance(cudaContext, maxSlots, elementsPerSlice);
            computeExpertsMethod = cacheClass.getMethod("computeExperts",
                FloatTensor.class, FloatTensor.class, FloatTensor.class,
                float[].class, int[].class, float[].class,
                int.class, int.class, int.class, int.class,
                float[][].class, float[][].class, float[][].class,
                boolean.class,
                FloatTensor.class, FloatTensor.class, FloatTensor.class);
            expertGpuCache = cache;
        } catch (ClassNotFoundException e) {
            // java21 classes not available
        } catch (Exception e) {
            System.out.println("  Expert GPU cache init failed: " + e.getMessage());
        }
    }

    /**
     * Get expert GPU cache statistics, or null if cache not active.
     */
    public String getExpertCacheStats() {
        if (expertGpuCache == null) return null;
        try {
            return (String) expertGpuCache.getClass().getMethod("getStats").invoke(expertGpuCache);
        } catch (Exception e) {
            return null;
        }
    }

    public Qwen3MoEState createState(int maxSeqLen) {
        return new Qwen3MoEState(config, maxSeqLen);
    }

    public float[] forward(Qwen3MoEState state, int token, int position) {
        return forwardInternal(state, token, position, true);
    }

    public void forwardNoOutput(Qwen3MoEState state, int token, int position) {
        forwardInternal(state, token, position, false);
    }

    private float[] forwardInternal(Qwen3MoEState state, int token, int position, boolean computeLogits) {
        int dim = config.embeddingLength();
        int leadingDenseCount = config.leadingDenseBlockCount();
        long t0 = 0, t1;

        for (int i = 0; i < dim; i++) {
            state.x[i] = weights.tokenEmbedding().getFloat((long) token * dim + i);
        }

        for (int layer = 0; layer < config.blockCount(); layer++) {
            Qwen3MoELayerWeights layerWeights = weights.layers()[layer];

            if (cpuProfile) t0 = System.nanoTime();
            RMSNorm.apply(state.xb, state.x, cachedAttnNorm[layer], dim, config.normEps());
            if (cpuProfile) { t1 = System.nanoTime(); profAttnNormNs += t1 - t0; t0 = t1; }

            gqaAttention(state, layerWeights, layer, position);
            if (cpuProfile) { t1 = System.nanoTime(); profAttnNs += t1 - t0; t0 = t1; }

            VectorOpsFactory.get().accumulate(state.x, state.xb, dim);
            if (cpuProfile) { t1 = System.nanoTime(); profResidualNs += t1 - t0; t0 = t1; }

            RMSNorm.apply(state.xb, state.x, cachedFfnNorm[layer], dim, config.normEps());
            if (cpuProfile) { t1 = System.nanoTime(); profFfnNormNs += t1 - t0; t0 = t1; }

            if (layer < leadingDenseCount) {
                denseFFN(state, layerWeights);
                if (cpuProfile) { t1 = System.nanoTime(); profDenseFfnNs += t1 - t0; t0 = t1; }
            } else {
                System.arraycopy(state.xb, 0, state.xbSaved, 0, dim);
                currentLayer = layer;
                moeFFN(state, layerWeights);
                if (cpuProfile) { t1 = System.nanoTime(); profMoeFfnNs += t1 - t0; t0 = t1; }
            }

            VectorOpsFactory.get().accumulate(state.x, state.xb, dim);
            if (cpuProfile) { t1 = System.nanoTime(); profResidualNs += t1 - t0; }
        }

        if (!computeLogits) return null;

        if (cpuProfile) t0 = System.nanoTime();
        RMSNorm.apply(state.xb, state.x, outputNormCache, dim, config.normEps());
        int vocabSize = config.vocabSize();
        Arrays.fill(state.logits, 0);
        weights.output().matmulParallel(state.xb, state.logits, vocabSize, dim);
        if (cpuProfile) {
            profOutputNs += System.nanoTime() - t0;
            profTokenCount++;
            if (profTokenCount % 10 == 0) printProfile();
        }

        return state.logits;
    }

    private void printProfile() {
        int n = profTokenCount;
        double ms = 1e6;
        long total = profAttnNormNs + profAttnNs + profFfnNormNs + profDenseFfnNs + profMoeFfnNs + profResidualNs + profOutputNs;
        System.out.printf("[cpu-profile Qwen3MoE] %d tokens, per-token avg (ms): attn_norm=%.1f attn(GQA)=%.1f ffn_norm=%.1f dense_ffn=%.1f moe_ffn=%.1f residual=%.1f output=%.1f | total=%.1f%n",
            n, profAttnNormNs / ms / n, profAttnNs / ms / n, profFfnNormNs / ms / n,
            profDenseFfnNs / ms / n, profMoeFfnNs / ms / n, profResidualNs / ms / n,
            profOutputNs / ms / n, total / ms / n);
    }

    /**
     * Standard GQA Attention with per-head QK normalization and RoPE.
     * Same as standard Attention class but inlined for Qwen3 MoE state.
     */
    private void gqaAttention(Qwen3MoEState state, Qwen3MoELayerWeights weights, int layer, int position) {
        int dim = config.embeddingLength();
        int headCount = config.headCount();
        int headCountKV = config.headCountKV();
        int headSize = config.headSize();
        int kvDim = config.kvDim();
        int qDim = headCount * headSize; // may differ from dim (e.g., Qwen3-Coder-30B)
        int kvMul = headCount / headCountKV;

        // Project Q, K, V
        Arrays.fill(state.q, 0, qDim, 0f);
        Arrays.fill(state.k, 0, kvDim, 0f);
        Arrays.fill(state.v, 0, kvDim, 0f);

        weights.wq().matmulParallel(state.xb, state.q, qDim, dim);
        weights.wk().matmulParallel(state.xb, state.k, kvDim, dim);
        weights.wv().matmulParallel(state.xb, state.v, kvDim, dim);

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

        // ISWA: even layers use sliding window, odd layers use full attention
        // startPos is the first position to attend to
        final int startPos;
        if (slidingWindow > 0 && (layer % 2 == 0)) {
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

        IntStream.range(0, headCount).parallel().forEach(new java.util.function.IntConsumer() {
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

        // Output projection: qDim -> dim
        Arrays.fill(state.xb, 0);
        weights.wo().matmulParallel(state.xb2, state.xb, dim, qDim);
        if (weights.woBias() != null) addBias(state.xb, weights.woBias(), dim);
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
        int expertCount = config.expertCount();
        int expertUsedCount = config.expertUsedCount();
        int expertFfnDim = config.expertFfnLength();
        int sharedFfnDim = config.expertSharedCount() * expertFfnDim;

        // 1. Router: compute expert logits and select top-K
        Arrays.fill(state.routerLogits, 0, expertCount, 0f);
        weights.ffnGateInp().matmul(state.xbSaved, state.routerLogits, expertCount, dim);
        if (weights.ffnGateInpBias() != null) addBias(state.routerLogits, weights.ffnGateInpBias(), expertCount);

        if (isGptOss) {
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

        // 2. Compute routed expert outputs
        Arrays.fill(state.xb, 0);

        // Capture for lambda
        final boolean useSwigluOai = isGptOss;

        if (expertGpuCache != null && computeExpertsMethod != null) {
            // GPU-accelerated path: batch all experts on GPU with LRU caching
            try {
                // Zero per-expert buffers
                for (int k = 0; k < expertUsedCount; k++) {
                    Arrays.fill(state.moeHbPerExpert[k], 0, expertFfnDim, 0f);
                    Arrays.fill(state.moeHb2PerExpert[k], 0, expertFfnDim, 0f);
                    Arrays.fill(state.expertOutPerExpert[k], 0, dim, 0f);
                }
                computeExpertsMethod.invoke(expertGpuCache,
                    weights.ffnGateExps(), weights.ffnUpExps(), weights.ffnDownExps(),
                    state.xbSaved, state.selectedExperts, state.selectedWeights,
                    expertUsedCount, currentLayer, dim, expertFfnDim,
                    state.moeHbPerExpert, state.moeHb2PerExpert, state.expertOutPerExpert,
                    useSwigluOai,
                    weights.ffnGateExpsBias(), weights.ffnUpExpsBias(), weights.ffnDownExpsBias());
            } catch (Exception e) {
                // Fallback to CPU on error, disable cache
                System.err.println("Expert GPU cache error: " + e.getMessage() + " — falling back to CPU");
                expertGpuCache = null;
                computeExpertsMethod = null;
                cpuExpertCompute(state, weights, expertUsedCount, expertFfnDim, dim, useSwigluOai);
            }
        } else {
            // CPU parallel path
            cpuExpertCompute(state, weights, expertUsedCount, expertFfnDim, dim, useSwigluOai);
        }

        // Sequential accumulation of weighted expert outputs
        for (int k = 0; k < expertUsedCount; k++) {
            VectorOpsFactory.get().saxpy(state.selectedWeights[k], state.expertOutPerExpert[k], 0, state.xb, 0, dim);
        }

        // 3. Shared expert
        if (weights.ffnGateShexp() != null) {
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
     * CPU parallel expert computation (original path).
     */
    private void cpuExpertCompute(Qwen3MoEState state, Qwen3MoELayerWeights weights,
                                   int expertUsedCount, int expertFfnDim, int dim,
                                   boolean useSwigluOai) {
        IntStream.range(0, expertUsedCount).parallel().forEach(new java.util.function.IntConsumer() {
            @Override
            public void accept(int k) {
                int e = state.selectedExperts[k];

                float[] gate = state.moeHbPerExpert[k];
                float[] up = state.moeHb2PerExpert[k];
                float[] out = state.expertOutPerExpert[k];

                Arrays.fill(gate, 0, expertFfnDim, 0f);
                Arrays.fill(up, 0, expertFfnDim, 0f);

                expertMatmul(weights.ffnGateExps(), state.xbSaved, gate, e, dim, expertFfnDim);
                expertMatmul(weights.ffnUpExps(), state.xbSaved, up, e, dim, expertFfnDim);

                if (weights.ffnGateExpsBias() != null) addExpertBias(gate, weights.ffnGateExpsBias(), e, expertFfnDim);
                if (weights.ffnUpExpsBias() != null) addExpertBias(up, weights.ffnUpExpsBias(), e, expertFfnDim);

                if (useSwigluOai) {
                    swigluOai(gate, up, expertFfnDim);
                } else {
                    VectorOpsFactory.get().silu(gate, expertFfnDim);
                    VectorOpsFactory.get().elementwiseMul(gate, up, gate, expertFfnDim);
                }

                Arrays.fill(out, 0, dim, 0f);
                expertMatmul(weights.ffnDownExps(), gate, out, e, expertFfnDim, dim);
                if (weights.ffnDownExpsBias() != null) addExpertBias(out, weights.ffnDownExpsBias(), e, dim);
            }
        });
    }

    /**
     * Matrix-vector multiply for a single expert slice from a 3D tensor.
     */
    private static void expertMatmul(FloatTensor weights3D, float[] input, float[] output,
                                      int expert, int inDim, int outDim) {
        long expertOffset = (long) expert * outDim * inDim;
        for (int row = 0; row < outDim; row++) {
            output[row] += weights3D.dot(expertOffset + (long) row * inDim, input, 0, inDim);
        }
    }

    /**
     * Select top-K indices and values from logits.
     */
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
