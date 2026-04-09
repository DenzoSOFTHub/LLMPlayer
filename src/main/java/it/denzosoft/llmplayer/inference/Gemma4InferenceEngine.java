package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.model.ModelWeights;
import it.denzosoft.llmplayer.model.TransformerLayerWeights;
import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Inference engine for Gemma 4 architecture.
 * Features:
 * - Per-Layer Embeddings (PLE): parallel token-specific information injected at each layer
 * - Shared KV cache: top N layers reuse KV from earlier layers (no K/V projections)
 * - Dual RoPE: SWA layers use theta=10K, full attention layers use theta=1M
 * - Pre+post attention/FFN norms (Gemma-style)
 * - QK-norm on Q and K (K-norm has no learnable scale)
 * - V-norm without learnable scale
 * - GeGLU activation (GELU instead of SiLU)
 * - Layer output scaling
 * - Logit soft-capping
 */
public class Gemma4InferenceEngine {

    private final ModelConfig config;
    private final ModelWeights weights;
    private final int maxSeqLen;

    // Cached config
    private final int dim;
    private final int ffnDim;
    private final int vocabSize;
    private final int blockCount;
    private final float normEps;
    private final int headCount;
    private final int headCountKV;
    private final int headSizeSwa;   // 256 for SWA layers
    private final int headSizeFull;  // 512 for full attention layers
    private final int maxQDim;       // max(headCount * headSizeSwa, headCount * headSizeFull)
    private final int maxKvDim;      // max(headCountKV * headSizeSwa, headCountKV * headSizeFull)
    private final int kvMul;
    private final int slidingWindow;
    private final int sharedKvLayers;
    private final int pleDim; // per-layer embedding dim (256 for E4B)
    private final float embeddingScale;
    private final float finalLogitSoftCap;

    // RoPE (dual: SWA and full attention)
    private final RoPE ropeSwa;   // for sliding window layers (theta=10K)
    private final RoPE ropeFull;  // for full attention layers (theta=1M)
    private final boolean[] slidingWindowPattern; // true=SWA, false=full

    // PLE global weights
    private final FloatTensor pleTokenEmbd;    // [pleDim*blockCount, vocabSize]
    private final FloatTensor pleModelProj;    // [dim, pleDim*blockCount]
    private final float[] pleProjNormWeights;  // [pleDim]

    // PLE per-layer weights
    private final FloatTensor[] pleInpGate;    // [dim, pleDim] per layer
    private final FloatTensor[] pleProj;       // [pleDim, dim] per layer
    private final float[][] plePostNorm;       // [dim] per layer
    private final float[] layerOutputScale;    // scalar per layer

    // Cached norm weights
    private final float[][] attnNormCache;
    private final float[][] ffnNormCache;
    private final float[][] postAttnNormCache;
    private final float[][] postFfnNormCache;
    private final float[][] qNormCache;
    private final float[][] kNormCache;
    private final float[] outputNormCache;

    // Shared KV mapping: for layer L, kvSourceLayer[L] = which layer's KV cache to use
    private final int[] kvSourceLayer;

    // Inference state
    private Gemma4State state;

    public Gemma4InferenceEngine(ModelConfig config, ModelWeights weights, int maxSeqLen,
                                  FloatTensor pleTokenEmbd, FloatTensor pleModelProj,
                                  float[] pleProjNormWeights,
                                  FloatTensor[] pleInpGate, FloatTensor[] pleProj,
                                  float[][] plePostNorm, float[] layerOutputScale,
                                  float[] ropeFreqFactors) {
        this.config = config;
        this.weights = weights;
        this.maxSeqLen = maxSeqLen;

        this.dim = config.embeddingLength();
        this.ffnDim = config.intermediateSize();
        this.vocabSize = config.vocabSize();
        this.blockCount = config.blockCount();
        this.normEps = config.normEps();
        this.headCount = config.headCount();
        this.headCountKV = config.headCountKV();
        // Gemma 4: SWA layers use headSize=key_length_swa, full layers use headSize=key_length
        this.headSizeSwa = config.headSize();  // already clamped to key_length_swa in ModelConfig
        this.headSizeFull = config.keyLength() > 0 ? config.keyLength() : headSizeSwa;
        this.maxQDim = headCount * Math.max(headSizeSwa, headSizeFull);
        this.maxKvDim = headCountKV * Math.max(headSizeSwa, headSizeFull);
        this.kvMul = headCount / headCountKV;
        this.slidingWindow = config.slidingWindow();
        this.sharedKvLayers = config.sharedKvLayers();
        this.pleDim = config.embeddingLengthPerLayer();
        this.embeddingScale = config.embeddingScale();
        this.finalLogitSoftCap = config.finalLogitSoftCap();
        this.slidingWindowPattern = config.slidingWindowPattern();

        this.pleTokenEmbd = pleTokenEmbd;
        this.pleModelProj = pleModelProj;
        this.pleProjNormWeights = pleProjNormWeights;
        this.pleInpGate = pleInpGate;
        this.pleProj = pleProj;
        this.plePostNorm = plePostNorm;
        this.layerOutputScale = layerOutputScale;

        // Dual RoPE: SWA uses theta=10K, full uses theta=1M with freq factors
        // RoPE dim = headSize for each layer type
        float swaTheta = config.ropeFreqBaseSwa() > 0 ? config.ropeFreqBaseSwa() : 10000f;
        float fullTheta = config.ropeFreqBase();
        this.ropeSwa = new RoPE(headSizeSwa, headSizeSwa, maxSeqLen, swaTheta, config.ropeType(), null);
        this.ropeFull = new RoPE(headSizeFull, headSizeFull, maxSeqLen, fullTheta, config.ropeType(), ropeFreqFactors);

        // Cache norm weights for fast access
        attnNormCache = new float[blockCount][];
        ffnNormCache = new float[blockCount][];
        postAttnNormCache = new float[blockCount][];
        postFfnNormCache = new float[blockCount][];
        qNormCache = new float[blockCount][];
        kNormCache = new float[blockCount][];
        for (int i = 0; i < blockCount; i++) {
            TransformerLayerWeights lw = weights.layers()[i];
            attnNormCache[i] = RMSNorm.cacheWeights(lw.attnNorm(), dim);
            ffnNormCache[i] = RMSNorm.cacheWeights(lw.ffnNorm(), dim);
            if (lw.postAttnNorm() != null) postAttnNormCache[i] = RMSNorm.cacheWeights(lw.postAttnNorm(), dim);
            if (lw.postFfnNorm() != null) postFfnNormCache[i] = RMSNorm.cacheWeights(lw.postFfnNorm(), dim);
            // QK-norm size depends on layer type (SWA=256, full=512)
            int layerHeadSize = isSwaLayer(i) ? headSizeSwa : headSizeFull;
            if (lw.qNorm() != null) {
                qNormCache[i] = RMSNorm.cacheWeights(lw.qNorm(), layerHeadSize);
            }
            if (lw.kNorm() != null) {
                kNormCache[i] = RMSNorm.cacheWeights(lw.kNorm(), layerHeadSize);
                // Gemma 4 K-norm weights in GGUF are raw w (NOT pre-computed 1+w)
                // The Gemma4RMSNorm class uses (1 + self.weight) * norm(x)
                for (int j = 0; j < layerHeadSize; j++) kNormCache[i][j] += 1.0f;
            }
        }
        outputNormCache = RMSNorm.cacheWeights(weights.outputNorm(), dim);

        // Build shared KV layer mapping: ALL shared layers reuse the LAST non-shared layer of same type
        // SWA shared layers → last SWA before shared boundary (firstShared - 2)
        // Full attention shared layers → last full before shared boundary (firstShared - 1)
        kvSourceLayer = new int[blockCount];
        int firstShared = blockCount - sharedKvLayers;
        for (int i = 0; i < blockCount; i++) {
            if (i < firstShared) {
                kvSourceLayer[i] = i; // own KV cache
            } else {
                kvSourceLayer[i] = isSwaLayer(i) ? (firstShared - 2) : (firstShared - 1);
            }
        }
    }

    private boolean isSwaLayer(int layer) {
        if (slidingWindowPattern != null && layer < slidingWindowPattern.length) {
            return slidingWindowPattern[layer];
        }
        return layer % 6 != 5; // fallback: Gemma 3 pattern
    }

    public Gemma4State createState() {
        // Use max dimensions for buffers (full attention layers have larger Q/K/V)
        return new Gemma4State(config, maxSeqLen, pleDim, maxQDim, maxKvDim);
    }

    /**
     * Forward pass for a single token.
     */
    public float[] forward(int token, int position, boolean computeLogits) {
        if (state == null) state = createState();

        // 1. Token embedding lookup + scaling
        for (int i = 0; i < dim; i++) {
            state.x[i] = weights.tokenEmbedding().getFloat((long) token * dim + i);
        }
        if (embeddingScale > 0f) {
            for (int i = 0; i < dim; i++) state.x[i] *= embeddingScale;
        }

        // 2. PLE pre-computation
        boolean hasPle = pleTokenEmbd != null && pleDim > 0;
        if (hasPle) computePleInput(token, state);
        boolean doPle = hasPle; // can be toggled for debug

        // 3. Forward through all layers
        for (int layer = 0; layer < blockCount; layer++) {
            forwardLayer(state, layer, position, doPle);
        }

        if (!computeLogits) return null;

        // 4. Final RMSNorm
        VectorOpsFactory.get().rmsnorm(state.xb, state.x, outputNormCache, dim, normEps);

        // 5. Output projection
        Arrays.fill(state.logits, 0);
        weights.output().matmulParallel(state.xb, state.logits, vocabSize, dim);

        // 6. Logit soft-capping: logits = softCap * tanh(logits / softCap)
        if (finalLogitSoftCap > 0f) {
            for (int i = 0; i < vocabSize; i++) {
                state.logits[i] = finalLogitSoftCap * (float) Math.tanh(state.logits[i] / finalLogitSoftCap);
            }
        }

        return state.logits;
    }

    /**
     * Pre-compute PLE input for this token.
     * Combines token-identity embedding with context-aware projection.
     */
    private void computePleInput(int token, Gemma4State state) {
        int totalPleDim = pleDim * blockCount;

        // Step 1: Token-identity embedding lookup → [totalPleDim]
        float[] pleEmb = state.pleEmb;
        for (int i = 0; i < totalPleDim; i++) {
            pleEmb[i] = pleTokenEmbd.getFloat((long) token * totalPleDim + i);
        }
        // Scale by sqrt(pleDim)
        float pleScale = (float) Math.sqrt(pleDim);
        for (int i = 0; i < totalPleDim; i++) pleEmb[i] *= pleScale;

        // Step 2: Context-aware projection: per_layer_model_proj @ x → [totalPleDim]
        float[] pleProjected = state.pleProjected;
        Arrays.fill(pleProjected, 0, totalPleDim, 0f);
        pleModelProj.matmulParallel(state.x, pleProjected, totalPleDim, dim);
        // Scale by 1/sqrt(dim)
        float projScale = 1.0f / (float) Math.sqrt(dim);
        for (int i = 0; i < totalPleDim; i++) pleProjected[i] *= projScale;

        // Apply per-layer RMSNorm to each pleDim-sized chunk of projected
        for (int l = 0; l < blockCount; l++) {
            RMSNorm.apply(pleProjected, l * pleDim, pleProjected, l * pleDim,
                    pleProjNormWeights, pleDim, normEps);
        }

        // Step 3: Combine and scale: (projected + emb) * (1/sqrt(2))
        float combineScale = 1.0f / (float) Math.sqrt(2.0);
        float[] pleCombined = state.pleCombined;
        for (int i = 0; i < totalPleDim; i++) {
            pleCombined[i] = (pleProjected[i] + pleEmb[i]) * combineScale;
        }
    }

    /**
     * Forward pass for a single transformer layer.
     */
    private void forwardLayer(Gemma4State state, int layer, int position, boolean doPle) {
        TransformerLayerWeights lw = weights.layers()[layer];
        boolean hasPle = pleTokenEmbd != null && pleDim > 0;
        boolean hasOwnKv = layer < blockCount - sharedKvLayers;
        boolean isSwa = isSwaLayer(layer);
        RoPE rope = isSwa ? ropeSwa : ropeFull;

        // Per-layer dimensions: SWA and full attention layers have different headSize
        int headSize = isSwa ? headSizeSwa : headSizeFull;
        int qDim = headCount * headSize;
        int kvDim = headCountKV * headSize;

        // === 1. Pre-attention RMSNorm ===
        RMSNorm.apply(state.xb, state.x, attnNormCache[layer], dim, normEps);

        // === 2. Q projection === (use large buffers that fit both SWA and full layers)
        final float[] qBuf = state.qLarge;
        final float[] kBuf = state.kLarge;
        final float[] vBuf = state.vLarge;
        final float[] xb2Buf = state.xb2Large;
        Arrays.fill(qBuf, 0, qDim, 0f);
        lw.wq().matmulParallel(state.xb, qBuf, qDim, dim);

        // QK-norm on Q (with learnable scale)
        if (qNormCache[layer] != null) {
            for (int h = 0; h < headCount; h++) {
                RMSNorm.apply(qBuf, h * headSize, qBuf, h * headSize,
                        qNormCache[layer], headSize, normEps);
            }
        }

        // RoPE on Q
        rope.applyAllHeads(qBuf, headCount, position);

        // === 3. K/V projection (conditional on shared KV) ===
        int kvLayer = kvSourceLayer[layer]; // which layer's KV cache to use
        if (hasOwnKv) {
            Arrays.fill(kBuf, 0, kvDim, 0f);
            Arrays.fill(vBuf, 0, kvDim, 0f);
            lw.wk().matmulParallel(state.xb, kBuf, kvDim, dim);
            lw.wv().matmulParallel(state.xb, vBuf, kvDim, dim);

            // QK-norm on K (with learnable scale)
            if (kNormCache[layer] != null) {
                for (int h = 0; h < headCountKV; h++) {
                    RMSNorm.apply(kBuf, h * headSize, kBuf, h * headSize,
                            kNormCache[layer], headSize, normEps);
                }
            }

            // V-norm (no learnable scale — raw RMSNorm)
            for (int h = 0; h < headCountKV; h++) {
                RMSNorm.applyNoScale(vBuf, h * headSize, headSize, normEps);
            }

            // RoPE on K
            rope.applyAllHeads(kBuf, headCountKV, position);

            // Store K/V in per-layer cache (kvDim varies per layer type)
            int kvCacheDim = state.gemma4KvCache.kvDim(layer);
            System.arraycopy(kBuf, 0, state.gemma4KvCache.keyLayer(layer),
                    position * kvCacheDim, kvDim);
            System.arraycopy(vBuf, 0, state.gemma4KvCache.valueLayer(layer),
                    position * kvCacheDim, kvDim);
        }
        // else: shared KV — use kvLayer's cache (already populated)

        // === 4. Attention ===
        final int hs = headSize; // capture for lambda
        final int kvd = kvDim;
        // Gemma 4: attention scale = 1.0 (model handles scaling via QK-norm internally)
        final int startPos = (isSwa && slidingWindow > 0)
                ? Math.max(0, position - slidingWindow + 1) : 0;
        final int seqLen = position + 1;
        final int kvCacheDim = state.gemma4KvCache.kvDim(kvLayer);
        final float[] keyCache = state.gemma4KvCache.keyLayer(kvLayer);
        final float[] valueCache = state.gemma4KvCache.valueLayer(kvLayer);

        // Parallel attention over heads
        IntStream.range(0, headCount).parallel().forEach(h -> {
            int kvHead = h / kvMul;
            int qOffset = h * hs;

            // Compute attention scores
            for (int t = startPos; t < seqLen; t++) {
                float score = 0f;
                int kOffset = t * kvCacheDim + kvHead * hs;
                for (int i = 0; i < hs; i++) {
                    score += qBuf[qOffset + i] * keyCache[kOffset + i];
                }
                state.att[h * maxSeqLen + t] = score; // scale=1.0
            }

            // Softmax
            float maxVal = Float.NEGATIVE_INFINITY;
            for (int t = startPos; t < seqLen; t++) {
                if (state.att[h * maxSeqLen + t] > maxVal) maxVal = state.att[h * maxSeqLen + t];
            }
            float sum = 0f;
            for (int t = startPos; t < seqLen; t++) {
                float v = (float) Math.exp(state.att[h * maxSeqLen + t] - maxVal);
                state.att[h * maxSeqLen + t] = v;
                sum += v;
            }
            float invSum = 1.0f / sum;
            for (int t = startPos; t < seqLen; t++) {
                state.att[h * maxSeqLen + t] *= invSum;
            }

            // Weighted V sum → xb2
            int outOffset = h * hs;
            for (int i = 0; i < hs; i++) {
                float val = 0f;
                for (int t = startPos; t < seqLen; t++) {
                    val += state.att[h * maxSeqLen + t]
                            * valueCache[t * kvCacheDim + kvHead * hs + i];
                }
                xb2Buf[outOffset + i] = val;
            }
        });

        // === 5. Wo projection ===
        Arrays.fill(state.xb, 0);
        lw.wo().matmulParallel(xb2Buf, state.xb, dim, qDim);

        // === 6. Post-attention norm ===
        if (postAttnNormCache[layer] != null) {
            RMSNorm.apply(state.xb, state.xb, postAttnNormCache[layer], dim, normEps);
        }

        // === 7. Attention residual: attn_out = xb + x ===
        float[] attnOut = state.xb3; // use separate buffer for attn_out
        for (int i = 0; i < dim; i++) attnOut[i] = state.xb[i] + state.x[i];

        // === 8. Pre-FFN RMSNorm ===
        RMSNorm.apply(state.xb, attnOut, ffnNormCache[layer], dim, normEps);

        // === 9. GeGLU FFN: gate * up with GELU activation ===
        Arrays.fill(state.hb, 0, ffnDim, 0f);
        Arrays.fill(state.hb2, 0, ffnDim, 0f);
        if (lw.wGate() != null) {
            lw.wGate().matmulParallel(state.xb, state.hb, ffnDim, dim);
        }
        lw.wUp().matmulParallel(state.xb, state.hb2, ffnDim, dim);

        // GELU activation on gate, then element-wise multiply
        for (int i = 0; i < ffnDim; i++) {
            float x = state.hb[i];
            // GELU with tanh approximation (gelu_pytorch_tanh)
            state.hb[i] = 0.5f * x * (1.0f + (float) Math.tanh(
                    0.7978845608028654f * (x + 0.044715f * x * x * x)));
            state.hb[i] *= state.hb2[i];
        }

        // Down projection
        Arrays.fill(state.xb, 0);
        lw.wDown().matmulParallel(state.hb, state.xb, dim, ffnDim);

        // === 10. Post-FFN norm (post_ffw_norm.weight) ===
        if (postFfnNormCache[layer] != null) {
            RMSNorm.apply(state.xb, state.xb, postFfnNormCache[layer], dim, normEps);
        }

        // === 11. FFN residual: x = xb + attn_out ===
        for (int i = 0; i < dim; i++) state.x[i] = state.xb[i] + attnOut[i];

        // === 12. PLE injection (after full layer) ===
        if (doPle && pleInpGate[layer] != null) {
            float[] pleGate = state.pleGate;
            Arrays.fill(pleGate, 0, pleDim, 0f);

            // gate = gelu(inp_gate @ x) — inp_gate maps [dim → pleDim]
            pleInpGate[layer].matmulParallel(state.x, pleGate, pleDim, dim);
            for (int i = 0; i < pleDim; i++) {
                float g = pleGate[i];
                pleGate[i] = 0.5f * g * (1.0f + (float) Math.tanh(
                        0.7978845608028654f * (g + 0.044715f * g * g * g)));
            }

            // Element-wise multiply with pre-computed PLE for this layer
            int pleOffset = layer * pleDim;
            for (int i = 0; i < pleDim; i++) {
                pleGate[i] *= state.pleCombined[pleOffset + i];
            }

            // Up-project: proj @ gated_ple → [dim]
            float[] pleOut = state.pleOut;
            Arrays.fill(pleOut, 0, dim, 0f);
            pleProj[layer].matmulParallel(pleGate, pleOut, dim, pleDim);

            // Post-PLE norm (post_norm.weight)
            if (plePostNorm[layer] != null) {
                RMSNorm.apply(pleOut, pleOut, plePostNorm[layer], dim, normEps);
            }

            // PLE residual
            for (int i = 0; i < dim; i++) state.x[i] += pleOut[i];
        }

        // === 13. Layer output scaling ===
        if (layerOutputScale != null && layer < layerOutputScale.length) {
            float scale = layerOutputScale[layer];
            if (scale != 1.0f && scale != 0f) {
                for (int i = 0; i < dim; i++) state.x[i] *= scale;
            }
        }
    }

    // === Public API matching InferenceEngine pattern ===

    public float[] prefill(Gemma4State st, int[] tokens) {
        this.state = st;
        for (int i = 0; i < tokens.length - 1; i++) {
            forward(tokens[i], i, false);
        }
        return forward(tokens[tokens.length - 1], tokens.length - 1, true);
    }

    public float[] forwardSingleToken(int token, int position) {
        if (state == null) state = createState();
        return forward(token, position, true);
    }

    public void setState(Gemma4State st) { this.state = st; }
    public Gemma4State getState() { return state; }
}
