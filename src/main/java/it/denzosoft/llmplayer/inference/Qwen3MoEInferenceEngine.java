package it.denzosoft.llmplayer.inference;

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

    public Qwen3MoEInferenceEngine(ModelConfig config, Qwen3MoEWeights weights, int maxSeqLen,
                                    float[] ropeFreqFactors) {
        this.config = config;
        this.weights = weights;
        this.maxSeqLen = maxSeqLen;

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

        this.outputNormCache = new float[dim];
        for (int i = 0; i < dim; i++) {
            outputNormCache[i] = weights.outputNorm().getFloat(i);
        }
    }

    public Qwen3MoEState createState(int maxSeqLen) {
        return new Qwen3MoEState(config, maxSeqLen);
    }

    public float[] forward(Qwen3MoEState state, int token, int position) {
        int dim = config.embeddingLength();
        int vocabSize = config.vocabSize();
        int leadingDenseCount = config.leadingDenseBlockCount();

        // 1. Token embedding lookup
        for (int i = 0; i < dim; i++) {
            state.x[i] = weights.tokenEmbedding().getFloat((long) token * dim + i);
        }

        // 2. Forward through all layers
        for (int layer = 0; layer < config.blockCount(); layer++) {
            Qwen3MoELayerWeights layerWeights = weights.layers()[layer];

            // Attention norm
            RMSNorm.apply(state.xb, state.x, cachedAttnNorm[layer], dim, config.normEps());

            // GQA Attention with QK-norm
            gqaAttention(state, layerWeights, layer, position);

            // Residual
            VectorOpsFactory.get().accumulate(state.x, state.xb, dim);

            // FFN norm
            RMSNorm.apply(state.xb, state.x, cachedFfnNorm[layer], dim, config.normEps());

            if (layer < leadingDenseCount) {
                // Dense SwiGLU FFN
                denseFFN(state, layerWeights);
            } else {
                // Save xb before MoE (since xb is reused as output accumulator)
                System.arraycopy(state.xb, 0, state.xbSaved, 0, dim);
                // MoE FFN
                moeFFN(state, layerWeights);
            }

            // Residual
            VectorOpsFactory.get().accumulate(state.x, state.xb, dim);
        }

        // 3. Final RMSNorm
        RMSNorm.apply(state.xb, state.x, outputNormCache, dim, config.normEps());

        // 4. Output projection
        Arrays.fill(state.logits, 0);
        weights.output().matmulParallel(state.xb, state.logits, vocabSize, dim);

        return state.logits;
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

        // Apply per-head QK-norm (Qwen3)
        if (cachedQNorm[layer] != null) {
            applyPerHeadNorm(state.q, cachedQNorm[layer], headCount, headSize, config.normEps());
            applyPerHeadNorm(state.k, cachedKNorm[layer], headCountKV, headSize, config.normEps());
        }

        // Apply RoPE
        rope.applyAllHeads(state.q, headCount, position);
        rope.applyAllHeads(state.k, headCountKV, position);

        // Store K and V in cache
        float[] keyCache = state.kvCache.keyLayer(layer);
        float[] valueCache = state.kvCache.valueLayer(layer);
        System.arraycopy(state.k, 0, keyCache, state.kvCache.offset(position), kvDim);
        System.arraycopy(state.v, 0, valueCache, state.kvCache.offset(position), kvDim);

        // Attention computation - parallel over heads
        float scaleFactor = (float) (1.0 / Math.sqrt(headSize));

        Arrays.fill(state.xb2, 0, qDim, 0f);

        IntStream.range(0, headCount).parallel().forEach(new java.util.function.IntConsumer() {
            @Override
            public void accept(int h) {
                int kvHead = h / kvMul;
                int attOffset = h * (position + 1);
                for (int t = 0; t <= position; t++) {
                    float score = 0f;
                    int qOffset = h * headSize;
                    int kOffset = state.kvCache.offset(t) + kvHead * headSize;
                    for (int i = 0; i < headSize; i++) {
                        score += state.q[qOffset + i] * keyCache[kOffset + i];
                    }
                    state.att[attOffset + t] = score * scaleFactor;
                }

                VectorOpsFactory.get().softmax(state.att, attOffset, position + 1);

                int outOffset = h * headSize;
                for (int t = 0; t <= position; t++) {
                    float a = state.att[attOffset + t];
                    int vOffset = state.kvCache.offset(t) + kvHead * headSize;
                    for (int i = 0; i < headSize; i++) {
                        state.xb2[outOffset + i] += a * valueCache[vOffset + i];
                    }
                }
            }
        });

        // Output projection: qDim -> dim
        Arrays.fill(state.xb, 0);
        weights.wo().matmulParallel(state.xb2, state.xb, dim, qDim);
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

        // Softmax + sigmoid gating: Qwen3 MoE uses softmax then sigmoid
        VectorOpsFactory.get().softmax(state.routerLogits, 0, expertCount);

        // Select top-K experts
        selectTopK(state.routerLogits, expertCount, expertUsedCount,
            state.selectedExperts, state.selectedWeights);

        // Renormalize weights (Qwen3 MoE uses normalized top-K weights)
        float weightSum = 0f;
        for (int k = 0; k < expertUsedCount; k++) {
            weightSum += state.selectedWeights[k];
        }
        if (weightSum > 0f) {
            for (int k = 0; k < expertUsedCount; k++) {
                state.selectedWeights[k] /= weightSum;
            }
        }

        // 2. Compute routed expert outputs - parallel across experts
        Arrays.fill(state.xb, 0);

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

                VectorOpsFactory.get().silu(gate, expertFfnDim);
                VectorOpsFactory.get().elementwiseMul(gate, up, gate, expertFfnDim);

                Arrays.fill(out, 0, dim, 0f);
                expertMatmul(weights.ffnDownExps(), gate, out, e, expertFfnDim, dim);
            }
        });

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

        for (int i = 0; i < n; i++) {
            int minIdx = 0;
            for (int j = 1; j < k; j++) {
                if (outValues[j] < outValues[minIdx]) {
                    minIdx = j;
                }
            }
            if (logits[i] > outValues[minIdx]) {
                outValues[minIdx] = logits[i];
                outIndices[minIdx] = i;
            }
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
        float[] logits = null;
        long t0 = System.currentTimeMillis();
        for (int i = 0; i < tokens.length; i++) {
            logits = forward(state, tokens[i], i);
            if (tokens.length > 10) {
                long elapsed = System.currentTimeMillis() - t0;
                System.out.printf("[prefill] token %d/%d (%.1fs)%n", i + 1, tokens.length, elapsed / 1000.0);
            }
        }
        long total = System.currentTimeMillis() - t0;
        System.out.printf("[prefill] done: %d tokens in %.1fs%n", tokens.length, total / 1000.0);
        return logits;
    }

    public ModelConfig getConfig() { return config; }
}
