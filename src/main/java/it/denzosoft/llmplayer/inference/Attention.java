package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.ModelArchitecture;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.model.TransformerLayerWeights;
import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.VectorOps;
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
    // Qwen-VL multi-axis RoPE: section (t/h/w/e) of each rotated pair; null for 1D RoPE
    private final int[] mropeMap;
    // Cached per-head QK norm weights (null if not used)
    private float[][] cachedQNorm;
    private float[][] cachedKNorm;

    // FlashAttention online-softmax mode: single-pass over K/V with running max+sum.
    // Bit-identical (within FP noise) to the legacy 2-pass implementation. OFF by default
    // because CPU benchmarks on Java show a ~6-15% slowdown at short/medium context: the
    // scalar rescale inside the loop dominates the saved second pass, while the 2-pass
    // version's softmax is already SIMD-optimized in VectorOps. Enable opt-in with
    // -Dattn.flash=true. Primary use case: eventual GPU HBM-bound path and long contexts.
    private static final boolean USE_FLASH =
        "true".equalsIgnoreCase(System.getProperty("attn.flash", "false"));

    public Attention(ModelConfig config, RoPE rope) {
        this(config, rope, null);
    }

    public Attention(ModelConfig config, RoPE rope, RoPE ropeLocal) {
        this.config = config;
        this.rope = rope;
        this.ropeLocal = ropeLocal;
        this.attnLogitSoftCap = config.attnLogitSoftCap();
        this.slidingWindow = config.slidingWindow();
        this.mropeMap = config.ropeSections() != null
            ? RoPE.mropeSectionMap(config.ropeSections(), config.ropeSectionsInterleaved(), rope.getRopeDimCount() / 2)
            : null;
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
            // Fused Q+K+V: single parallel dispatch, input stays in L1 cache
            FloatTensor.fusedQKVMatmulParallel(weights.wq(), weights.wk(), weights.wv(),
                state.xb, state.q, state.k, state.v, qDim, kvDim, dim);
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

        normAndRope(state, layer, position);
        // Store K and V in cache (transparently quantizes if KV cache is in Q8 mode)
        state.kvCache.storeK(layer, position, state.k, kvDim);
        state.kvCache.storeV(layer, position, state.v, kvDim);

        // Attention computation - parallel over heads
        float scaleFactor = config.attentionScale() > 0f
            ? config.attentionScale()
            : (float) (1.0 / Math.sqrt(headSize));

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

        final VectorOps vecOps = VectorOpsFactory.get();
        final KVCache kv = state.kvCache;
        final int layerFinal = layer;
        final int positionFinal = position;
        final int seqLenFinal = seqLen;
        final float scaleFactorFinal = scaleFactor;

        if (USE_FLASH) {
            // FlashAttention-style single-pass online-softmax attention.
            // For each head, stream over t from attnStartPos to position keeping a running
            // (max, sumExp, out[headSize]) state. Whenever a new maximum is seen, rescale
            // the partial output by exp(old_max - new_max). Finally divide by sumExp.
            it.denzosoft.llmplayer.tensor.MatmulPool.forEach(headCount, h -> {
                int kvHead = h / kvMul;
                int kvHeadOff = kvHead * headSize;
                int qOffset = h * headSize;
                int outOffset = h * headSize;

                float maxScore = Float.NEGATIVE_INFINITY;
                float sumExp = 0f;
                // Output already zeroed by Arrays.fill above

                for (int t = attnStartPos; t <= positionFinal; t++) {
                    float score = kv.dotK(layerFinal, t, kvHeadOff, headSize, state.q, qOffset);
                    float s = score * scaleFactorFinal;
                    // Attention logit soft-capping (Gemma2/3)
                    if (attnLogitSoftCap > 0f) {
                        s = attnLogitSoftCap * (float) Math.tanh(s / attnLogitSoftCap);
                    }
                    float newMax = s > maxScore ? s : maxScore;
                    // Rescale factors
                    float scaleOld = maxScore == Float.NEGATIVE_INFINITY
                        ? 0f : (float) Math.exp(maxScore - newMax);
                    float scaleNew = (float) Math.exp(s - newMax);
                    // Rescale existing partial output by scaleOld
                    if (scaleOld != 1f) {
                        for (int i = 0; i < headSize; i++) {
                            state.xb2[outOffset + i] *= scaleOld;
                        }
                    }
                    // Accumulate scaleNew * V[t] into partial output
                    kv.saxpyV(layerFinal, t, kvHeadOff, headSize, scaleNew, state.xb2, outOffset);
                    // Update running sum and max
                    sumExp = sumExp * scaleOld + scaleNew;
                    maxScore = newMax;
                }
                // Normalize by sumExp
                if (sumExp > 0f) {
                    float invSum = 1.0f / sumExp;
                    for (int i = 0; i < headSize; i++) {
                        state.xb2[outOffset + i] *= invSum;
                    }
                }
            });
        } else {
            // Legacy 2-pass: compute scores, softmax, weighted sum.
            it.denzosoft.llmplayer.tensor.MatmulPool.forEach(headCount, h -> {
                int kvHead = h / kvMul;
                int kvHeadOff = kvHead * headSize;

                int attOffset = h * seqLenFinal;
                int qOffset = h * headSize;
                for (int t = attnStartPos; t <= positionFinal; t++) {
                    float score = kv.dotK(layerFinal, t, kvHeadOff, headSize, state.q, qOffset);
                    float s = score * scaleFactorFinal;
                    if (attnLogitSoftCap > 0f) {
                        s = attnLogitSoftCap * (float) Math.tanh(s / attnLogitSoftCap);
                    }
                    state.att[attOffset + (t - attnStartPos)] = s;
                }
                vecOps.softmax(state.att, attOffset, seqLenFinal);
                int outOffset = h * headSize;
                for (int t = attnStartPos; t <= positionFinal; t++) {
                    float a = state.att[attOffset + (t - attnStartPos)];
                    kv.saxpyV(layerFinal, t, kvHeadOff, headSize, a, state.xb2, outOffset);
                }
            });
        }

        // Spark2.5 per-head output gate, computed from the normed layer input still in xb
        if (weights.attnGate() != null) {
            applyHeadGate(state, weights.attnGate(), state.xb, headCount, headSize, dim);
        }

        // Output projection: xb = Wo * xb2 (qDim -> dim)
        Arrays.fill(state.xb, 0);
        weights.wo().matmulParallel(state.xb2, state.xb, dim, qDim);
        // E13: Wo bias (attn_output.bias) — for Qwen2/SmolLM3 variants and Command-R
        if (weights.woBias() != null) {
            addBias(state.xb, weights.woBias(), dim);
        }
    }

    /**
     * Determine if a layer uses global (full) attention or local (sliding window).
     * Gemma 2: alternating — odd layers are global, even layers are local (sliding window).
     * Gemma 3: every 6th layer (layer % 6 == 5) is global, rest are local.
     * GPT-OSS: even layers are global, odd layers use sliding window.
     * Cohere2: every 4th layer (layer % 4 == 3) is global, rest are local — see llama.cpp
     *   set_swa_pattern(4) at llama-model.cpp.
     */
    /**
     * Per-head QK-norm and RoPE of {@code state.q}/{@code state.k} for one position. The norm runs
     * before RoPE (Qwen3, Gemma 3, OLMo2) except for Hunyuan, which rotates first.
     */
    private void normAndRope(InferenceState state, int layer, int position) {
        int headCount = config.headCount();
        int headCountKV = config.headCountKV();
        int headSize = config.headSize();
        boolean normAfterRope = config.qkNormAfterRope();

        // Apply per-head QK-norm if present (Qwen3, Gemma3)
        if (cachedQNorm != null && !normAfterRope) {
            applyPerHeadNorm(state.q, cachedQNorm[layer], headCount, headSize, config.normEps());
            applyPerHeadNorm(state.k, cachedKNorm[layer], headCountKV, headSize, config.normEps());
        }

        // Apply RoPE to Q and K (skip for NoPE layers in Llama4 iRoPE / Cohere2 NoPE-on-global)
        int noRopeInterval = config.noRopeLayerInterval();
        boolean iropeSkip = noRopeInterval > 0 && (layer % noRopeInterval) == (noRopeInterval - 1);
        // Cohere2: NoPE on global layers — only SWA layers get RoPE.
        boolean cohere2GlobalSkip = config.useNoPeOnGlobalLayers() && isGlobalLayer(layer);
        if (mropeMap != null) {
            // Qwen-VL: text tokens rotate with t = h = w = position (e = 0), image tokens with their
            // grid coordinates; see MRopePositions
            int[] p4 = state.ropePos4;
            if (state.mrope != null) {
                state.mrope.get(position, p4);
            } else {
                p4[0] = position; p4[1] = position; p4[2] = position; p4[3] = 0;
            }
            int half = rope.getRopeDimCount() / 2;
            if (state.mropeCos == null) {
                state.mropeCos = new float[half];
                state.mropeSin = new float[half];
            }
            rope.applyMrope(state.q, headCount, mropeMap, p4, state.mropeCos, state.mropeSin);
            rope.applyMrope(state.k, headCountKV, mropeMap, p4, state.mropeCos, state.mropeSin);
        } else if (!iropeSkip && !cohere2GlobalSkip) {
            // Gemma 3: local layers use theta=10000, global use main theta
            // Spark2.5: local layers also rotate a different number of dims
            RoPE activeRope = (ropeLocal != null && !isGlobalLayer(layer)) ? ropeLocal : rope;
            activeRope.applyAllHeads(state.q, headCount, position);
            activeRope.applyAllHeads(state.k, headCountKV, position);
        }

        // Hunyuan: QK-norm after RoPE (llama.cpp hunyuan-vl.cpp)
        if (cachedQNorm != null && normAfterRope) {
            applyPerHeadNorm(state.q, cachedQNorm[layer], headCount, headSize, config.normEps());
            applyPerHeadNorm(state.k, cachedKNorm[layer], headCountKV, headSize, config.normEps());
        }
    }

    /**
     * Spark2.5 headwise gate: {@code out[h*headSize..] *= sigmoid(gate · input)[h]}, where
     * {@code gate} maps the layer's normed input [dim] to one scalar per head (llama.cpp
     * spark2-5.cpp). Applied to the attention output before the output projection.
     */
    private static void applyHeadGate(InferenceState state, FloatTensor gate, float[] input,
                                      int headCount, int headSize, int dim) {
        float[] g = state.attnGate;
        Arrays.fill(g, 0, headCount, 0f);
        gate.matmul(input, g, headCount, dim);
        for (int h = 0; h < headCount; h++) {
            float s = 1.0f / (1.0f + (float) Math.exp(-g[h]));
            int off = h * headSize;
            for (int i = 0; i < headSize; i++) state.xb2[off + i] *= s;
        }
    }

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
        // Cohere2: 3 local + 1 global, repeating (set_swa_pattern(4))
        if (config.architecture() == ModelArchitecture.COHERE2) {
            return layer % 4 == 3;
        }
        // Spark2.5: pattern array (true=SWA/local, false=full/global), 3 local + 1 global
        if (config.architecture() == ModelArchitecture.SPARK2_5) {
            boolean[] pattern = config.slidingWindowPattern();
            if (pattern != null && layer < pattern.length) {
                return !pattern[layer];
            }
            return layer % 4 == 3;
        }
        // Gemma 4: pattern array (true=SWA/local, false=full/global)
        if (config.architecture() == ModelArchitecture.GEMMA4) {
            boolean[] pattern = config.slidingWindowPattern();
            if (pattern != null && layer < pattern.length) {
                return !pattern[layer]; // pattern[layer]=true means SWA (local), so NOT global
            }
            return layer % 6 == 5; // fallback: same as Gemma 3
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

        normAndRope(state, layer, position);
        // Store K and V in cache (transparently quantizes if KV cache is in Q8 mode)
        state.kvCache.storeK(layer, position, state.k, kvDim);
        state.kvCache.storeV(layer, position, state.v, kvDim);

        // Attention computation - parallel over heads
        float scaleFactor = config.attentionScale() > 0f
            ? config.attentionScale()
            : (float) (1.0 / Math.sqrt(headSize));

        int startPos = 0;
        if (slidingWindow > 0 && !isGlobalLayer(layer)) {
            startPos = Math.max(0, position - slidingWindow + 1);
        }
        final int attnStartPos = startPos;
        int seqLen = position + 1 - attnStartPos;

        Arrays.fill(state.xb2, 0, qDim, 0f);

        final VectorOps vecOps = VectorOpsFactory.get();
        final KVCache kv = state.kvCache;
        final int layerFinal = layer;

        it.denzosoft.llmplayer.tensor.MatmulPool.forEach(headCount, h -> {
            int kvHead = h / kvMul;
            int kvHeadOff = kvHead * headSize;

            int attOffset = h * seqLen;
            int qOffset = h * headSize;
            for (int t = attnStartPos; t <= position; t++) {
                float score = kv.dotK(layerFinal, t, kvHeadOff, headSize, state.q, qOffset);
                float s = score * scaleFactor;
                if (attnLogitSoftCap > 0f) {
                    s = attnLogitSoftCap * (float) Math.tanh(s / attnLogitSoftCap);
                }
                state.att[attOffset + (t - attnStartPos)] = s;
            }

            vecOps.softmax(state.att, attOffset, seqLen);

            int outOffset = h * headSize;
            for (int t = attnStartPos; t <= position; t++) {
                float a = state.att[attOffset + (t - attnStartPos)];
                kv.saxpyV(layerFinal, t, kvHeadOff, headSize, a, state.xb2, outOffset);
            }
        });
        // Result in state.xb2 — caller does output projection (Wo matmul)
    }

    /**
     * Whether {@link #forwardBatch} reproduces {@link #forward} for this layer: it needs separate
     * Q/K/V matrices, and the per-token core it reuses ({@link #forwardFromProjections}) is the
     * two-pass softmax, not the opt-in flash variant.
     */
    boolean supportsBatch(TransformerLayerWeights weights) {
        return !USE_FLASH && (weights.wqkv() != null || weights.wq() != null);
    }

    /**
     * Multi-token attention for batched prefill: reads the normed inputs {@code b.xb[t]}, writes the
     * projected output to {@code b.xb[t]}, for tokens at positions {@code basePos + t}. The Q/K/V
     * and output projections run as multi-token matmuls; the per-token part (bias, QK-norm, RoPE,
     * KV store, scores, softmax, weighted sum) runs token by token in position order through
     * {@link #forwardFromProjections}, so token t sees exactly the KV entries 0..t it sees in the
     * one-token path.
     */
    void forwardBatch(PrefillBatch b, InferenceState state, TransformerLayerWeights weights,
                      int layer, int basePos, int n) {
        int dim = config.embeddingLength();
        int kvDim = config.kvDim();
        int qDim = config.headCount() * config.headSize();
        if (weights.wqkv() != null) {
            // merged QKV (Phi-3/4, Spark2.5): one multi-token matmul, then split per token
            int qkvDim = qDim + 2 * kvDim;
            float[][] qkv = b.qkv(qkvDim);
            for (int t = 0; t < n; t++) Arrays.fill(qkv[t], 0, qkvDim, 0f);
            FloatTensor.matmulBatchParallel(weights.wqkv(), b.xb, qkv, n, qkvDim, dim);
            for (int t = 0; t < n; t++) {
                System.arraycopy(qkv[t], 0, b.q[t], 0, qDim);
                System.arraycopy(qkv[t], qDim, b.k[t], 0, kvDim);
                System.arraycopy(qkv[t], qDim + kvDim, b.v[t], 0, kvDim);
            }
        } else {
            for (int t = 0; t < n; t++) {
                Arrays.fill(b.q[t], 0, qDim, 0f);
                Arrays.fill(b.k[t], 0, kvDim, 0f);
                Arrays.fill(b.v[t], 0, kvDim, 0f);
            }
            FloatTensor.fusedQKVBatchParallel(weights.wq(), weights.wk(), weights.wv(),
                b.xb, b.q, b.k, b.v, n, qDim, kvDim, dim);
        }
        for (int t = 0; t < n; t++) {
            System.arraycopy(b.q[t], 0, state.q, 0, qDim);
            System.arraycopy(b.k[t], 0, state.k, 0, kvDim);
            System.arraycopy(b.v[t], 0, state.v, 0, kvDim);
            forwardFromProjections(state, weights, layer, basePos + t);
            // Spark2.5 per-head output gate from this token's normed input (still in b.xb[t])
            if (weights.attnGate() != null) {
                applyHeadGate(state, weights.attnGate(), b.xb[t], config.headCount(), config.headSize(), dim);
            }
            System.arraycopy(state.xb2, 0, b.att[t], 0, qDim);
            Arrays.fill(b.xb[t], 0, dim, 0f);
        }
        FloatTensor.matmulBatchParallel(weights.wo(), b.att, b.xb, n, dim, qDim);
        if (weights.woBias() != null) {
            for (int t = 0; t < n; t++) addBias(b.xb[t], weights.woBias(), dim);
        }
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
        VectorOps vecOps = VectorOpsFactory.get();
        for (int h = 0; h < nHeads; h++) {
            int offset = h * headSize;
            // SIMD sum of squares via dot(vec, vec)
            float ss = vecOps.dot(vec, offset, vec, offset, headSize);
            ss = 1.0f / (float) Math.sqrt(ss / headSize + eps);
            // SIMD scale with weights
            vecOps.scaleWeighted(vec, offset, normWeights, ss, headSize);
        }
    }
}
