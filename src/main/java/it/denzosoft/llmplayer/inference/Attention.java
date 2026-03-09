package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.ModelArchitecture;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.model.TransformerLayerWeights;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Multi-Head / Grouped-Query Attention with RoPE.
 * Supports optional Q/K/V bias (Qwen2) and per-head QK-norm (Qwen3).
 */
public class Attention {

    private final ModelConfig config;
    private final RoPE rope;
    private final RoPE ropeLocal; // for sliding window layers (Gemma 3: theta=10000), null if not used
    private final float attnLogitSoftCap;
    private final int slidingWindow; // 0 = no sliding window
    // Cached per-head QK norm weights (null if not used)
    private float[][] cachedQNorm;
    private float[][] cachedKNorm;

    public Attention(ModelConfig config, RoPE rope) {
        this(config, rope, null);
    }

    public Attention(ModelConfig config, RoPE rope, RoPE ropeLocal) {
        this.config = config;
        this.rope = rope;
        this.ropeLocal = ropeLocal;
        this.attnLogitSoftCap = config.attnLogitSoftCap();
        this.slidingWindow = config.slidingWindow();
    }

    /**
     * Initialize per-head norm caches from the first layer's weights.
     * Called lazily on first forward pass.
     */
    private void initNormCaches(TransformerLayerWeights[] allLayers) {
        int headSize = config.headSize();
        int blockCount = allLayers.length;
        if (allLayers[0].qNorm() != null) {
            cachedQNorm = new float[blockCount][];
            cachedKNorm = new float[blockCount][];
            for (int i = 0; i < blockCount; i++) {
                cachedQNorm[i] = RMSNorm.cacheWeights(allLayers[i].qNorm(), headSize);
                cachedKNorm[i] = RMSNorm.cacheWeights(allLayers[i].kNorm(), headSize);
            }
        }
    }

    public void initNormCachesIfNeeded(TransformerLayerWeights[] allLayers) {
        if (cachedQNorm == null && allLayers[0].qNorm() != null) {
            initNormCaches(allLayers);
        }
    }

    /**
     * Perform attention for a single position.
     * Reads from xb (normalized input), writes to xb2 (output).
     */
    public void forward(InferenceState state, TransformerLayerWeights weights, int layer, int position) {
        int dim = config.embeddingLength();
        int headCount = config.headCount();
        int headCountKV = config.headCountKV();
        int headSize = config.headSize();
        int kvDim = config.kvDim();
        int qDim = headCount * headSize; // may differ from dim (e.g., Mistral3/Devstral)
        int kvMul = headCount / headCountKV; // GQA ratio

        // Project Q, K, V
        Arrays.fill(state.q, 0, qDim, 0f);
        Arrays.fill(state.k, 0, kvDim, 0f);
        Arrays.fill(state.v, 0, kvDim, 0f);

        if (weights.wqkv() != null) {
            // Merged QKV (Phi3/Phi4): single matmul then split
            int qkvDim = qDim + kvDim + kvDim;
            float[] qkvBuf = state.hbPacked; // reuse packed buffer (large enough: 2*ffnDim >= qkvDim)
            Arrays.fill(qkvBuf, 0, qkvDim, 0f);
            weights.wqkv().matmulParallel(state.xb, qkvBuf, qkvDim, dim);
            System.arraycopy(qkvBuf, 0, state.q, 0, qDim);
            System.arraycopy(qkvBuf, qDim, state.k, 0, kvDim);
            System.arraycopy(qkvBuf, qDim + kvDim, state.v, 0, kvDim);
        } else {
            // Separate Q, K, V (standard)
            weights.wq().matmulParallel(state.xb, state.q, qDim, dim);
            weights.wk().matmulParallel(state.xb, state.k, kvDim, dim);
            weights.wv().matmulParallel(state.xb, state.v, kvDim, dim);
        }

        // Apply Q/K/V bias if present (Qwen2)
        if (weights.qBias() != null) {
            addBias(state.q, weights.qBias(), qDim);
        }
        if (weights.kBias() != null) {
            addBias(state.k, weights.kBias(), kvDim);
        }
        if (weights.vBias() != null) {
            addBias(state.v, weights.vBias(), kvDim);
        }

        // Apply per-head QK-norm if present (Qwen3, Gemma3)
        if (cachedQNorm != null) {
            applyPerHeadNorm(state.q, cachedQNorm[layer], headCount, headSize, config.normEps());
            applyPerHeadNorm(state.k, cachedKNorm[layer], headCountKV, headSize, config.normEps());
        }

        // Apply RoPE to Q and K (skip for NoPE layers in Llama4 iRoPE)
        int noRopeInterval = config.noRopeLayerInterval();
        if (noRopeInterval == 0 || (layer % noRopeInterval) != (noRopeInterval - 1)) {
            // Gemma 3: local layers use theta=10000, global use main theta
            RoPE activeRope = (ropeLocal != null && !isGlobalLayer(layer)) ? ropeLocal : rope;
            activeRope.applyAllHeads(state.q, headCount, position);
            activeRope.applyAllHeads(state.k, headCountKV, position);
        }

        // Store K and V in cache
        float[] keyCache = state.kvCache.keyLayer(layer);
        float[] valueCache = state.kvCache.valueLayer(layer);
        System.arraycopy(state.k, 0, keyCache, state.kvCache.offset(position), kvDim);
        System.arraycopy(state.v, 0, valueCache, state.kvCache.offset(position), kvDim);

        // Attention computation - parallel over heads
        float scaleFactor = (float) (1.0 / Math.sqrt(headSize));

        // Sliding window: local layers attend only to last slidingWindow positions
        // Gemma 3: every 6th layer is global (layer % 6 == 5), rest are local
        // GPT-OSS: alternating layers use sliding window
        int startPos = 0;
        if (slidingWindow > 0 && !isGlobalLayer(layer)) {
            startPos = Math.max(0, position - slidingWindow + 1);
        }
        final int attnStartPos = startPos;
        int seqLen = position + 1 - attnStartPos;

        Arrays.fill(state.xb2, 0, qDim, 0f);

        IntStream.range(0, headCount).parallel().forEach(h -> {
            int kvHead = h / kvMul;

            // Compute attention scores: Q_h * K_h^T for attended positions
            int attOffset = h * seqLen;
            for (int t = attnStartPos; t <= position; t++) {
                float score = 0f;
                int qOffset = h * headSize;
                int kOffset = state.kvCache.offset(t) + kvHead * headSize;
                for (int i = 0; i < headSize; i++) {
                    score += state.q[qOffset + i] * keyCache[kOffset + i];
                }
                float s = score * scaleFactor;
                // Attention logit soft-capping (Gemma2/3)
                if (attnLogitSoftCap > 0f) {
                    s = attnLogitSoftCap * (float) Math.tanh(s / attnLogitSoftCap);
                }
                state.att[attOffset + (t - attnStartPos)] = s;
            }

            // Softmax over attention scores
            VectorOpsFactory.get().softmax(state.att, attOffset, seqLen);

            // Weighted sum of values
            int outOffset = h * headSize;
            for (int t = attnStartPos; t <= position; t++) {
                float a = state.att[attOffset + (t - attnStartPos)];
                int vOffset = state.kvCache.offset(t) + kvHead * headSize;
                for (int i = 0; i < headSize; i++) {
                    state.xb2[outOffset + i] += a * valueCache[vOffset + i];
                }
            }
        });

        // Output projection: xb = Wo * xb2 (qDim -> dim)
        Arrays.fill(state.xb, 0);
        weights.wo().matmulParallel(state.xb2, state.xb, dim, qDim);
    }

    /**
     * Determine if a layer uses global (full) attention or local (sliding window).
     * Gemma 2: alternating — odd layers are global, even layers are local (sliding window).
     * Gemma 3: every 6th layer (layer % 6 == 5) is global, rest are local.
     * GPT-OSS: even layers are global, odd layers use sliding window.
     */
    private boolean isGlobalLayer(int layer) {
        // Gemma 2: alternating (even = local/sliding, odd = global/full)
        if (config.architecture() == ModelArchitecture.GEMMA2) {
            return layer % 2 == 1;
        }
        // Gemma 3: 5 local + 1 global, repeating
        if (config.architecture() == ModelArchitecture.GEMMA3) {
            return layer % 6 == 5;
        }
        // GPT-OSS: alternating (even = global, odd = local)
        if (config.architecture() == ModelArchitecture.GPT_OSS) {
            return layer % 2 == 0;
        }
        return true; // default: global (shouldn't reach here with slidingWindow > 0)
    }

    /**
     * Perform attention from pre-projected Q/K/V values (already in state.q/k/v).
     * Does bias, QK-norm, RoPE, KV cache, attention scores, softmax, weighted sum.
     * Result is in state.xb2. Does NOT perform output projection (Wo matmul).
     * Used by GpuForwardPass which does projections and Wo on GPU.
     */
    public void forwardFromProjections(InferenceState state, TransformerLayerWeights weights,
                                        int layer, int position) {
        int headCount = config.headCount();
        int headCountKV = config.headCountKV();
        int headSize = config.headSize();
        int kvDim = config.kvDim();
        int qDim = headCount * headSize;
        int kvMul = headCount / headCountKV;

        // Apply Q/K/V bias if present (Qwen2)
        if (weights.qBias() != null) {
            addBias(state.q, weights.qBias(), qDim);
        }
        if (weights.kBias() != null) {
            addBias(state.k, weights.kBias(), kvDim);
        }
        if (weights.vBias() != null) {
            addBias(state.v, weights.vBias(), kvDim);
        }

        // Apply per-head QK-norm if present (Qwen3)
        if (cachedQNorm != null) {
            applyPerHeadNorm(state.q, cachedQNorm[layer], headCount, headSize, config.normEps());
            applyPerHeadNorm(state.k, cachedKNorm[layer], headCountKV, headSize, config.normEps());
        }

        // Apply RoPE to Q and K (skip for NoPE layers in Llama4 iRoPE)
        int noRopeInterval = config.noRopeLayerInterval();
        if (noRopeInterval == 0 || (layer % noRopeInterval) != (noRopeInterval - 1)) {
            // Gemma 3: local layers use theta=10000, global use main theta
            RoPE activeRope = (ropeLocal != null && !isGlobalLayer(layer)) ? ropeLocal : rope;
            activeRope.applyAllHeads(state.q, headCount, position);
            activeRope.applyAllHeads(state.k, headCountKV, position);
        }

        // Store K and V in cache
        float[] keyCache = state.kvCache.keyLayer(layer);
        float[] valueCache = state.kvCache.valueLayer(layer);
        System.arraycopy(state.k, 0, keyCache, state.kvCache.offset(position), kvDim);
        System.arraycopy(state.v, 0, valueCache, state.kvCache.offset(position), kvDim);

        // Attention computation - parallel over heads
        float scaleFactor = (float) (1.0 / Math.sqrt(headSize));

        int startPos = 0;
        if (slidingWindow > 0 && !isGlobalLayer(layer)) {
            startPos = Math.max(0, position - slidingWindow + 1);
        }
        final int attnStartPos = startPos;
        int seqLen = position + 1 - attnStartPos;

        Arrays.fill(state.xb2, 0, qDim, 0f);

        IntStream.range(0, headCount).parallel().forEach(h -> {
            int kvHead = h / kvMul;

            int attOffset = h * seqLen;
            for (int t = attnStartPos; t <= position; t++) {
                float score = 0f;
                int qOffset = h * headSize;
                int kOffset = state.kvCache.offset(t) + kvHead * headSize;
                for (int i = 0; i < headSize; i++) {
                    score += state.q[qOffset + i] * keyCache[kOffset + i];
                }
                float s = score * scaleFactor;
                if (attnLogitSoftCap > 0f) {
                    s = attnLogitSoftCap * (float) Math.tanh(s / attnLogitSoftCap);
                }
                state.att[attOffset + (t - attnStartPos)] = s;
            }

            VectorOpsFactory.get().softmax(state.att, attOffset, seqLen);

            int outOffset = h * headSize;
            for (int t = attnStartPos; t <= position; t++) {
                float a = state.att[attOffset + (t - attnStartPos)];
                int vOffset = state.kvCache.offset(t) + kvHead * headSize;
                for (int i = 0; i < headSize; i++) {
                    state.xb2[outOffset + i] += a * valueCache[vOffset + i];
                }
            }
        });
        // Result in state.xb2 — caller does output projection (Wo matmul)
    }

    public RoPE getRope() { return rope; }

    /**
     * Add bias from a FloatTensor to a float array: vec[i] += bias.getFloat(i)
     */
    private static void addBias(float[] vec, it.denzosoft.llmplayer.tensor.FloatTensor bias, int size) {
        for (int i = 0; i < size; i++) {
            vec[i] += bias.getFloat(i);
        }
    }

    /**
     * Apply RMSNorm per-head: each head of size headSize is independently normalized.
     * The norm weights are shared across heads (same weights for all heads, size = headSize).
     */
    private static void applyPerHeadNorm(float[] vec, float[] normWeights, int nHeads, int headSize, float eps) {
        for (int h = 0; h < nHeads; h++) {
            int offset = h * headSize;
            // In-place RMSNorm on vec[offset..offset+headSize-1]
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
}
