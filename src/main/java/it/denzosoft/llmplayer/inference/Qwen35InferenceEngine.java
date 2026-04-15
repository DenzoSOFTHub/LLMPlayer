package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.model.Qwen35LayerWeights;
import it.denzosoft.llmplayer.model.Qwen35Weights;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Inference engine for Qwen3.5 architecture.
 * Hybrid model: alternating Gated DeltaNet (linear attention) and full GQA layers.
 * Pattern: 3 DeltaNet + 1 Full Attention, controlled by full_attention_interval.
 *
 * DeltaNet layers: recurrent state-space model with gated updates, no KV cache.
 * Full Attention layers: standard GQA with KV cache, QK-norm, RoPE.
 */
public class Qwen35InferenceEngine {

    private final ModelConfig config;
    private final Qwen35Weights weights;
    private final int maxSeqLen;

    // Cached config values
    private final int dim;
    private final int ffnDim;
    private final int vocabSize;
    private final int blockCount;
    private final int timeStepRank;
    private final int stateSize;
    private final int groupCount;
    private final int innerSize;
    private final int convKernel;
    private final int fullAttnInterval;
    private final float normEps;

    // Attention parameters for full attention layers
    private final int headCount;
    private final int headCountKV;
    private final int headSize;
    private final int kvDim;
    private final int kvMul;  // headCount / headCountKV for GQA

    // DeltaNet parameters
    private final int dQK;   // Q/K dimension per head = stateSize (head_k_dim)
    private final int dV;    // V dimension per head = stateSize (head_v_dim)
    private final int qkvDim; // total QKV projection dim

    // Pre-cached norm weights
    private final float[] outputNormCache;
    // E11: pre-cache all per-layer norm weights at construction time so the forward pass
    // doesn't re-dequantize + allocate a new float[dim] every token. Before the fix, each
    // forward pass called cacheWeightsInline() several times per layer, generating ~800 KB
    // of garbage per token on the 4B model.
    private final float[][] attnNormPerLayer;    // [blockCount][dim]
    private final float[][] postAttnNormPerLayer; // [blockCount][dim] (full-attention layers)
    private final float[][] ssmNormPerLayer;     // [blockCount][stateSize] (DeltaNet layers)

    // RoPE for full attention layers
    private final RoPE rope;

    // Pre-cached per-layer norm weights for attention layers
    private final float[][] qNormCache;
    private final float[][] kNormCache;

    // GPU forward pass (loaded via reflection from java21/, null if unavailable)
    private AutoCloseable gpuForwardPass;
    private int gpuLayerCount;
    private Method gpuUploadXAndUpdateParams;
    private Method gpuForwardLayer;
    private Method gpuForwardGraph;
    private Method gpuForwardGraphArgmax;
    private Method gpuForwardFinalLogits;
    private Method gpuForwardFinalArgmax;
    private Method gpuForwardGraphPrefill;
    private Method gpuDownloadX;

    public Qwen35InferenceEngine(ModelConfig config, Qwen35Weights weights, int maxSeqLen,
                                  float[] ropeFreqFactors) {
        this.config = config;
        this.weights = weights;
        this.maxSeqLen = maxSeqLen;

        this.dim = config.embeddingLength();
        this.ffnDim = config.intermediateSize();
        this.vocabSize = config.vocabSize();
        this.blockCount = config.blockCount();
        this.timeStepRank = config.ssmTimeStepRank();
        this.stateSize = config.ssmStateSize();
        this.groupCount = config.ssmGroupCount();
        this.innerSize = config.ssmInnerSize();
        this.convKernel = config.ssmConvKernel();
        this.fullAttnInterval = config.fullAttentionInterval();
        this.normEps = config.normEps();

        this.headCount = config.headCount();
        this.headCountKV = config.headCountKV();
        this.headSize = config.headSize();
        this.kvDim = config.kvDim();
        this.kvMul = headCount / headCountKV;

        this.dQK = stateSize;  // head_k_dim = stateSize
        this.dV = stateSize;   // head_v_dim = stateSize
        this.qkvDim = groupCount * stateSize * 2 + timeStepRank * stateSize;

        // Cache output norm weights
        this.outputNormCache = new float[dim];
        for (int i = 0; i < dim; i++) {
            outputNormCache[i] = weights.outputNorm().getFloat(i);
        }

        // RoPE for full attention layers
        int ropeDimCount = config.ropeDimensionCount();
        this.rope = new RoPE(headSize, ropeDimCount, maxSeqLen, config.ropeFreqBase(),
            config.ropeType(), ropeFreqFactors);

        // Cache QK norm weights for attention layers
        this.qNormCache = new float[blockCount][];
        this.kNormCache = new float[blockCount][];
        for (int layer = 0; layer < blockCount; layer++) {
            Qwen35LayerWeights lw = weights.layers()[layer];
            if (!lw.isDeltaNet() && lw.qNorm() != null) {
                qNormCache[layer] = cacheWeights(lw.qNorm(), headSize);
                kNormCache[layer] = cacheWeights(lw.kNorm(), headSize);
            }
        }

        // E11: pre-cache per-layer norm weights (attn_norm always, post_attn + ssm_norm per type)
        this.attnNormPerLayer = new float[blockCount][];
        this.postAttnNormPerLayer = new float[blockCount][];
        this.ssmNormPerLayer = new float[blockCount][];
        for (int layer = 0; layer < blockCount; layer++) {
            Qwen35LayerWeights lw = weights.layers()[layer];
            if (lw.attnNorm() != null) {
                attnNormPerLayer[layer] = cacheWeights(lw.attnNorm(), dim);
            }
            if (lw.postAttnNorm() != null) {
                postAttnNormPerLayer[layer] = cacheWeights(lw.postAttnNorm(), dim);
            }
            if (lw.isDeltaNet() && lw.ssmNorm() != null) {
                ssmNormPerLayer[layer] = cacheWeights(lw.ssmNorm(), stateSize);
            }
        }
    }

    private float[] cacheWeights(it.denzosoft.llmplayer.tensor.FloatTensor tensor, int size) {
        float[] cache = new float[size];
        for (int i = 0; i < size; i++) cache[i] = tensor.getFloat(i);
        return cache;
    }

    /**
     * Try to initialize GPU forward pass via reflection (java21/ Qwen35CudaForwardPass).
     * Call after construction with the CudaBufferManager from LLMEngine.
     */
    public void tryInitGpuForwardPass(Object bufferManager) {
        try {
            Class<?> fwdClass = Class.forName("it.denzosoft.llmplayer.inference.Qwen35CudaForwardPass");

            // Check isSupported
            Method isSup = fwdClass.getMethod("isSupported", ModelConfig.class, Qwen35Weights.class);
            Boolean supported = (Boolean) isSup.invoke(null, config, weights);
            if (!supported) {
                System.err.println("Qwen35 CUDA forward pass: not supported (weights not on GPU)");
                return;
            }

            // Construct
            Object fwd = fwdClass.getConstructor(ModelConfig.class, Qwen35Weights.class,
                    bufferManager.getClass(), int.class)
                .newInstance(config, weights, bufferManager, maxSeqLen);

            // Cache methods
            gpuUploadXAndUpdateParams = fwdClass.getMethod("uploadXAndUpdateParams", float[].class, int.class);
            gpuForwardLayer = fwdClass.getMethod("forwardLayer", int.class, int.class);
            gpuForwardGraph = fwdClass.getMethod("forwardGraph", float[].class);
            gpuForwardGraphArgmax = fwdClass.getMethod("forwardGraphArgmax");
            gpuForwardFinalLogits = fwdClass.getMethod("forwardFinalLogits", float[].class);
            gpuForwardFinalArgmax = fwdClass.getMethod("forwardFinalArgmax");
            gpuForwardGraphPrefill = fwdClass.getMethod("forwardGraphPrefill");
            gpuDownloadX = fwdClass.getMethod("downloadX", float[].class);

            Method getGpuLayers = fwdClass.getMethod("getGpuLayerCount");
            gpuLayerCount = (Integer) getGpuLayers.invoke(fwd);
            gpuForwardPass = (AutoCloseable) fwd;

            System.err.println("Qwen35 CUDA forward pass: enabled (" + gpuLayerCount + "/" + blockCount + " layers)");
        } catch (Throwable e) {
            System.err.println("Qwen35 CUDA forward pass: unavailable — " + e.getMessage());
            gpuForwardPass = null;
        }
    }

    public Qwen35State createState(int maxSeqLen) {
        return new Qwen35State(config, maxSeqLen);
    }

    public float[] forward(Qwen35State state, int token, int position) {
        return forwardInternal(state, token, position, true);
    }

    public void forwardNoOutput(Qwen35State state, int token, int position) {
        forwardInternal(state, token, position, false);
    }

    private float[] forwardInternal(Qwen35State state, int token, int position, boolean computeLogits) {
        // 1. Token embedding
        for (int i = 0; i < dim; i++) {
            state.x[i] = weights.tokenEmbedding().getFloat((long) token * dim + i);
        }

        // 2. Try GPU forward pass
        if (gpuForwardPass != null) {
            try {
                return forwardGpu(state, position, computeLogits);
            } catch (Exception e) {
                System.err.println("Qwen35 GPU forward failed, falling back to CPU: " + e.getMessage());
                gpuForwardPass = null;
            }
        }

        // 3. CPU forward through all layers
        for (int layer = 0; layer < blockCount; layer++) {
            Qwen35LayerWeights lw = weights.layers()[layer];
            if (lw.isDeltaNet()) {
                forwardDeltaNet(state, lw, layer);
            } else {
                forwardAttention(state, lw, layer, position);
            }
        }

        if (!computeLogits) return null;

        // 4. Final RMSNorm
        VectorOpsFactory.get().rmsnorm(state.xb, state.x, outputNormCache, dim, normEps);

        // 5. Output projection
        Arrays.fill(state.logits, 0);
        weights.output().matmulParallel(state.xb, state.logits, vocabSize, dim);

        return state.logits;
    }

    private float[] forwardGpu(Qwen35State state, int position, boolean computeLogits) throws Exception {
        // Upload embedding + token params
        gpuUploadXAndUpdateParams.invoke(gpuForwardPass, state.x, position);

        // Try CUDA graph (use generation graph for both prefill and generation)
        if (gpuLayerCount == blockCount) {
            Boolean graphOk = (Boolean) gpuForwardGraph.invoke(gpuForwardPass, state.logits);
            if (graphOk) return computeLogits ? state.logits : null;
        }

        // Per-layer GPU forward
        for (int layer = 0; layer < gpuLayerCount; layer++) {
            gpuForwardLayer.invoke(gpuForwardPass, layer, position);
        }
        // Profiling hook (no-op unless -Dqwen35.profile=true)
        try {
            java.lang.reflect.Method m = gpuForwardPass.getClass().getMethod("profileTokenComplete");
            m.invoke(gpuForwardPass);
        } catch (NoSuchMethodException ignored) {}

        // If not all layers on GPU, download X and continue on CPU
        if (gpuLayerCount < blockCount) {
            gpuDownloadX.invoke(gpuForwardPass, state.x);
            for (int layer = gpuLayerCount; layer < blockCount; layer++) {
                Qwen35LayerWeights lw = weights.layers()[layer];
                if (lw.isDeltaNet()) {
                    forwardDeltaNet(state, lw, layer);
                } else {
                    forwardAttention(state, lw, layer, position);
                }
            }
            if (!computeLogits) return null;
            VectorOpsFactory.get().rmsnorm(state.xb, state.x, outputNormCache, dim, normEps);
            Arrays.fill(state.logits, 0);
            weights.output().matmulParallel(state.xb, state.logits, vocabSize, dim);
            return state.logits;
        }

        if (!computeLogits) return null;

        // All layers on GPU — try GPU output projection
        Boolean logitsOk = (Boolean) gpuForwardFinalLogits.invoke(gpuForwardPass, state.logits);
        if (logitsOk) return state.logits;

        // Fallback: download X and compute output on CPU
        gpuDownloadX.invoke(gpuForwardPass, state.x);
        VectorOpsFactory.get().rmsnorm(state.xb, state.x, outputNormCache, dim, normEps);
        Arrays.fill(state.logits, 0);
        weights.output().matmulParallel(state.xb, state.logits, vocabSize, dim);
        return state.logits;
    }

    public float[] prefill(Qwen35State state, int[] tokens) {
        for (int i = 0; i < tokens.length - 1; i++) {
            forwardNoOutput(state, tokens[i], i);
        }
        return forward(state, tokens[tokens.length - 1], tokens.length - 1);
    }

    // ==================== DeltaNet Forward Pass ====================

    private void forwardDeltaNet(Qwen35State state, Qwen35LayerWeights lw, int layer) {
        // Pre-attention RMSNorm (uses pre-cached weights, no per-token allocation)
        VectorOpsFactory.get().rmsnorm(state.xb, state.x, attnNormPerLayer[layer], dim, normEps);

        // Project to QKV: xb -> qkv
        Arrays.fill(state.qkv, 0);
        lw.attnQkv().matmulParallel(state.xb, state.qkv, qkvDim, dim);

        // Causal Conv1D
        applyConv1d(state, lw, layer);

        // Apply SiLU activation to QKV after conv
        for (int i = 0; i < qkvDim; i++) {
            float val = state.qkv[i];
            state.qkv[i] = val * sigmoid(val); // SiLU = x * sigmoid(x)
        }

        // Compute alpha and beta gates from original input (xb)
        Arrays.fill(state.alpha, 0);
        lw.ssmAlpha().matmulParallel(state.xb, state.alpha, timeStepRank, dim);
        Arrays.fill(state.beta, 0);
        lw.ssmBeta().matmulParallel(state.xb, state.beta, timeStepRank, dim);

        // Alpha (decay): exp(negExpA * softplus(alpha_proj + dt_bias))
        // ssm_a stores -exp(A_log) (pre-computed by GGUF converter)
        // Beta (update gate): sigmoid(beta_proj)
        for (int h = 0; h < timeStepRank; h++) {
            float negExpA = lw.ssmA().getFloat(h);
            float dtBias = lw.ssmDtBias().getFloat(h);
            float g = negExpA * softplus(state.alpha[h] + dtBias);
            state.alpha[h] = (float) Math.exp(g);
            state.beta[h] = sigmoid(state.beta[h]);
        }

        // Compute output gate: SiLU(attn_gate @ xb)
        Arrays.fill(state.gate, 0);
        lw.attnGate().matmulParallel(state.xb, state.gate, innerSize, dim);
        for (int i = 0; i < innerSize; i++) {
            float val = state.gate[i];
            state.gate[i] = val * sigmoid(val);
        }

        // Split QKV and run DeltaNet recurrence per head
        deltaNetRecurrence(state, layer);

        // Apply ssm_norm (per-head RMSNorm on the d_v output) — pre-cached
        applyPerHeadNorm(state.deltaOut, ssmNormPerLayer[layer], timeStepRank, dV, normEps);

        // Gate the output: deltaOut *= gate
        for (int i = 0; i < innerSize; i++) {
            state.deltaOut[i] *= state.gate[i];
        }

        // Output projection: ssm_out @ deltaOut -> xb
        Arrays.fill(state.xb, 0);
        lw.ssmOut().matmulParallel(state.deltaOut, state.xb, dim, innerSize);

        // Residual connection (attention)
        for (int i = 0; i < dim; i++) {
            state.x[i] += state.xb[i];
        }

        // Post-attention norm + FFN + residual
        forwardFFN(state, lw, layer);
    }

    private void applyConv1d(Qwen35State state, Qwen35LayerWeights lw, int layer) {
        float[][] convBuf = state.convState[layer];
        int histSize = convKernel - 1; // 3

        int pos = state.convStatePos[layer]; // number of values stored so far

        // Depthwise conv1d: for each channel, multiply kernel weights by history + current
        // GGUF stores conv1d as [kernel_size, channels] with ne[0]=kernel_size, ne[1]=channels
        // So element for (channel, kernel_pos) is at offset: channel * convKernel + kernel_pos
        // PyTorch Conv1d convention: weight[0] = oldest input, weight[K-1] = current input
        // For causal conv: output[t] = sum_k(weight[K-1-k] * input[t-k]) for k=0..K-1
        // E21: use pre-allocated state.convResult (was: new float[qkvDim] every token)
        float[] result = state.convResult;
        for (int ch = 0; ch < qkvDim; ch++) {
            float sum = 0;
            // Current value (k=0): weight[convKernel-1] * input[t]
            sum += lw.ssmConv1d().getFloat((long) ch * convKernel + (convKernel - 1)) * state.qkv[ch];
            // History (k=1..convKernel-1): weight[convKernel-1-k] * input[t-k]
            for (int k = 1; k < convKernel; k++) {
                if (pos - k >= 0) { // only if we have enough history
                    int histIdx = (pos - k) % histSize;
                    sum += lw.ssmConv1d().getFloat((long) ch * convKernel + (convKernel - 1 - k)) * convBuf[histIdx][ch];
                }
            }
            result[ch] = sum;
        }

        // Store current QKV into circular buffer AFTER conv computation
        System.arraycopy(state.qkv, 0, convBuf[pos % histSize], 0, qkvDim);
        state.convStatePos[layer] = pos + 1;

        System.arraycopy(result, 0, state.qkv, 0, qkvDim);
    }

    private void deltaNetRecurrence(Qwen35State state, int layer) {
        // Split QKV:
        // Q: [groupCount * dQK], K: [groupCount * dQK], V: [timeStepRank * dV]
        int qSize = groupCount * dQK;
        int kSize = groupCount * dQK;
        int vOffset = qSize + kSize;

        // Process per head (timeStepRank heads)
        // Q/K are repeated: head h uses Q/K group (h % groupCount)
        IntStream.range(0, timeStepRank).parallel().forEach(h -> {
            int group = h % groupCount;
            float alphaH = state.alpha[h];
            float betaH = state.beta[h];

            // Get Q and K for this head's group (shared across heads in the group)
            int qOff = group * dQK;
            int kOff = qSize + group * dQK;

            // L2-normalize Q and apply scaling (1/sqrt(head_k_dim))
            float[] qNorm = new float[dQK];
            float qLen = 0;
            for (int i = 0; i < dQK; i++) {
                qNorm[i] = state.qkv[qOff + i];
                qLen += qNorm[i] * qNorm[i];
            }
            float qScale = 1.0f / (float) Math.sqrt(qLen + 1e-12f) * (1.0f / (float) Math.sqrt(dQK));
            for (int i = 0; i < dQK; i++) qNorm[i] *= qScale;

            // L2-normalize K
            float[] kNormalized = new float[dQK];
            float kLen = 0;
            for (int i = 0; i < dQK; i++) {
                kNormalized[i] = state.qkv[kOff + i];
                kLen += kNormalized[i] * kNormalized[i];
            }
            kLen = (float) Math.sqrt(kLen + 1e-12f);
            for (int i = 0; i < dQK; i++) kNormalized[i] /= kLen;

            // Get V for this head
            int vOff = vOffset + h * dV;
            float[] vH = new float[dV];
            for (int i = 0; i < dV; i++) vH[i] = state.qkv[vOff + i];

            // State S is [dQK, dV] for this head
            float[] S = state.ssmState[layer][h];

            // DeltaNet recurrence:
            // S_new = alpha*S + beta * outer(k, v - alpha * S^T @ k)
            // o = S_new^T @ q

            // Compute S^T @ k -> sK [dV]
            float[] sK = new float[dV];
            for (int j = 0; j < dV; j++) {
                float sum = 0;
                for (int i = 0; i < dQK; i++) {
                    sum += S[i * dV + j] * kNormalized[i];
                }
                sK[j] = sum;
            }

            // Update S
            for (int i = 0; i < dQK; i++) {
                for (int j = 0; j < dV; j++) {
                    int idx = i * dV + j;
                    S[idx] = alphaH * S[idx]
                           - alphaH * betaH * sK[j] * kNormalized[i]
                           + betaH * vH[j] * kNormalized[i];
                }
            }

            // Compute output: o = S^T @ q -> o_h [dV]
            int outOff = h * dV;
            for (int j = 0; j < dV; j++) {
                float sum = 0;
                for (int i = 0; i < dQK; i++) {
                    sum += S[i * dV + j] * qNorm[i];
                }
                state.deltaOut[outOff + j] = sum;
            }
        });
    }

    // ==================== Full Attention Forward Pass ====================

    private void forwardAttention(Qwen35State state, Qwen35LayerWeights lw, int layer, int position) {
        // Pre-attention RMSNorm (pre-cached)
        VectorOpsFactory.get().rmsnorm(state.xb, state.x, attnNormPerLayer[layer], dim, normEps);

        int qDim = headCount * headSize;
        int qGateDim = qDim * 2; // Q projection packs Q + gate

        // Q projection: outputs [head0_Q(headSize), head0_gate(headSize), head1_Q(headSize), head1_gate(headSize), ...]
        Arrays.fill(state.q, 0, qGateDim, 0);
        lw.wq().matmulParallel(state.xb, state.q, qGateDim, dim);

        // Deinterleave Q and gate per-head: raw layout is [Q_h0(headSize), gate_h0(headSize), Q_h1, gate_h1, ...]
        // Extract gates first (to separate array), then compact Q in forward order to avoid overlap
        for (int h = 0; h < headCount; h++) {
            System.arraycopy(state.q, h * headSize * 2 + headSize, state.attnGate, h * headSize, headSize);
        }
        for (int h = 1; h < headCount; h++) {
            System.arraycopy(state.q, h * headSize * 2, state.q, h * headSize, headSize);
        }

        // K, V projections
        Arrays.fill(state.k, 0, kvDim, 0);
        lw.wk().matmulParallel(state.xb, state.k, kvDim, dim);
        Arrays.fill(state.v, 0, kvDim, 0);
        lw.wv().matmulParallel(state.xb, state.v, kvDim, dim);

        // Per-head QK normalization (only on Q, not on gate)
        if (qNormCache[layer] != null) {
            applyPerHeadNorm(state.q, qNormCache[layer], headCount, headSize, normEps);
            applyPerHeadNorm(state.k, kNormCache[layer], headCountKV, headSize, normEps);
        }

        // RoPE
        rope.applyAllHeads(state.q, headCount, position);
        rope.applyAllHeads(state.k, headCountKV, position);

        // Store K, V in cache (transparently quantizes in Q8 mode)
        state.kvCache.storeK(layer, position, state.k, kvDim);
        state.kvCache.storeV(layer, position, state.v, kvDim);

        // Multi-head attention
        final float invSqrt = 1.0f / (float) Math.sqrt(headSize);
        final KVCache kv = state.kvCache;
        final int layerFinal = layer;
        final int positionFinal = position;
        final int headSizeFinal = headSize;
        IntStream.range(0, headCount).parallel().forEach(h -> {
            int kvHead = h / kvMul;
            int qOff = h * headSizeFinal;
            int kvHeadOff = kvHead * headSizeFinal;

            // Attention scores
            for (int t = 0; t <= positionFinal; t++) {
                float score = kv.dotK(layerFinal, t, kvHeadOff, headSizeFinal, state.q, qOff);
                state.att[h * maxSeqLen + t] = score * invSqrt;
            }

            // Softmax
            softmax(state.att, h * maxSeqLen, positionFinal + 1);

            // Weighted sum of values
            int outOff = h * headSizeFinal;
            Arrays.fill(state.xb2, outOff, outOff + headSizeFinal, 0);
            for (int t = 0; t <= positionFinal; t++) {
                float a = state.att[h * maxSeqLen + t];
                kv.saxpyV(layerFinal, t, kvHeadOff, headSizeFinal, a, state.xb2, outOff);
            }
        });

        // Apply attention output gate: output *= sigmoid(gate)
        for (int i = 0; i < qDim; i++) {
            state.xb2[i] *= sigmoid(state.attnGate[i]);
        }

        // Output projection: wo @ xb2 -> xb
        Arrays.fill(state.xb, 0);
        lw.wo().matmulParallel(state.xb2, state.xb, dim, qDim);

        // Residual connection
        for (int i = 0; i < dim; i++) {
            state.x[i] += state.xb[i];
        }

        // Post-attention norm + FFN + residual
        forwardFFN(state, lw, layer);
    }

    // ==================== SwiGLU FFN ====================

    private void forwardFFN(Qwen35State state, Qwen35LayerWeights lw, int layer) {
        // Post-attention / FFN norm (pre-cached)
        VectorOpsFactory.get().rmsnorm(state.xb, state.x, postAttnNormPerLayer[layer], dim, normEps);

        // SwiGLU FFN: h = SiLU(gate @ xb) * (up @ xb), output = down @ h
        Arrays.fill(state.hb, 0);
        lw.ffnGate().matmulParallel(state.xb, state.hb, ffnDim, dim);
        Arrays.fill(state.hb2, 0);
        lw.ffnUp().matmulParallel(state.xb, state.hb2, ffnDim, dim);

        // SiLU(gate) * up
        for (int i = 0; i < ffnDim; i++) {
            float val = state.hb[i];
            state.hb[i] = (val * sigmoid(val)) * state.hb2[i];
        }

        // Down projection
        Arrays.fill(state.xb, 0);
        lw.ffnDown().matmulParallel(state.hb, state.xb, dim, ffnDim);

        // Residual connection
        for (int i = 0; i < dim; i++) {
            state.x[i] += state.xb[i];
        }
    }

    // ==================== Utility ====================

    private static float sigmoid(float x) {
        return 1.0f / (1.0f + (float) Math.exp(-x));
    }

    private static float softplus(float x) {
        if (x > 20) return x;
        return (float) Math.log(1.0 + Math.exp(x));
    }

    private static void softmax(float[] x, int offset, int size) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < size; i++) {
            if (x[offset + i] > max) max = x[offset + i];
        }
        float sum = 0;
        for (int i = 0; i < size; i++) {
            x[offset + i] = (float) Math.exp(x[offset + i] - max);
            sum += x[offset + i];
        }
        for (int i = 0; i < size; i++) {
            x[offset + i] /= sum;
        }
    }

    private static void applyPerHeadNorm(float[] data, float[] normWeights, int numHeads,
                                          int headDim, float eps) {
        for (int h = 0; h < numHeads; h++) {
            int off = h * headDim;
            float ss = 0;
            for (int i = 0; i < headDim; i++) {
                ss += data[off + i] * data[off + i];
            }
            ss = 1.0f / (float) Math.sqrt(ss / headDim + eps);
            for (int i = 0; i < headDim; i++) {
                data[off + i] = data[off + i] * ss * normWeights[i];
            }
        }
    }

    private float[] cacheWeightsInline(it.denzosoft.llmplayer.tensor.FloatTensor tensor, int size) {
        float[] cache = new float[size];
        for (int i = 0; i < size; i++) cache[i] = tensor.getFloat(i);
        return cache;
    }

    public ModelConfig getConfig() { return config; }
}
