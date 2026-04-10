package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.model.NemotronHLayerWeights;
import it.denzosoft.llmplayer.model.NemotronHWeights;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Inference engine for Nemotron-H hybrid architecture.
 * Three layer types: Mamba-2 SSM, GQA Attention, squared-ReLU FFN.
 * Each block in the model is exactly one of these types (not combined).
 */
public class NemotronHInferenceEngine {

    private final ModelConfig config;
    private final NemotronHWeights weights;
    private final int maxSeqLen;

    private final int dim;
    private final int vocabSize;
    private final int blockCount;
    private final float normEps;
    private final float embeddingScale;  // Granite Hybrid: 12.0
    private final float logitScale;      // Granite Hybrid: 8.0
    private final float residualScale;   // Granite Hybrid: 0.22

    // Mamba-2 dimensions
    private final int ssmInnerSize;   // 7680
    private final int ssmStateSize;   // 128
    private final int ssmGroupCount;  // 8
    private final int ssmTimeStepRank; // 96 = number of heads
    private final int ssmConvKernel;  // 4
    private final int convChannels;   // ssmInnerSize + 2*ssmGroupCount*ssmStateSize
    private final int headDim;        // ssmInnerSize / ssmTimeStepRank = 80

    // Attention dimensions
    private final int headCount;
    private final int headCountKV;
    private final int headSize;
    private final int kvDim;
    private final int kvMul;

    // Output norm cache
    private final float[] outputNormCache;

    // GPU forward pass (reflection-loaded)
    private AutoCloseable gpuForwardPass;
    private int gpuLayerCount;
    private Method gpuUploadXAndUpdateParams, gpuForwardLayer, gpuForwardGraph;
    private Method gpuForwardFinalLogits, gpuDownloadX;

    // RoPE
    private final RoPE rope;

    public NemotronHInferenceEngine(ModelConfig config, NemotronHWeights weights, int maxSeqLen,
                                     float[] ropeFreqFactors) {
        this.config = config;
        this.weights = weights;
        this.maxSeqLen = maxSeqLen;

        this.dim = config.embeddingLength();
        this.vocabSize = config.vocabSize();
        this.blockCount = config.blockCount();
        this.normEps = config.normEps();
        this.embeddingScale = config.embeddingScale();
        this.logitScale = config.logitScale();
        this.residualScale = config.residualScale();

        this.ssmInnerSize = config.ssmInnerSize();
        this.ssmStateSize = config.ssmStateSize();
        this.ssmGroupCount = config.ssmGroupCount();
        this.ssmTimeStepRank = config.ssmTimeStepRank();
        this.ssmConvKernel = config.ssmConvKernel();
        this.convChannels = ssmInnerSize + 2 * ssmGroupCount * ssmStateSize;
        this.headDim = ssmInnerSize / ssmTimeStepRank;

        this.headCount = config.headCount();
        this.headCountKV = config.headCountKV();
        this.headSize = config.headSize();
        this.kvDim = config.kvDim();
        this.kvMul = headCount / headCountKV;

        this.outputNormCache = new float[dim];
        for (int i = 0; i < dim; i++) outputNormCache[i] = weights.outputNorm().getFloat(i);

        this.rope = new RoPE(headSize, config.ropeDimensionCount(), maxSeqLen,
            config.ropeFreqBase(), config.ropeType(), ropeFreqFactors);
    }

    public void tryInitGpuForwardPass(Object bufferManager) {
        try {
            Class<?> cls = Class.forName("it.denzosoft.llmplayer.inference.NemotronHCudaForwardPass");
            Method isSup = cls.getMethod("isSupported", ModelConfig.class, NemotronHWeights.class);
            if (!(Boolean) isSup.invoke(null, config, weights)) return;
            Object fwd = cls.getConstructor(ModelConfig.class, NemotronHWeights.class,
                    bufferManager.getClass(), int.class).newInstance(config, weights, bufferManager, maxSeqLen);
            gpuUploadXAndUpdateParams = cls.getMethod("uploadXAndUpdateParams", float[].class, int.class);
            gpuForwardLayer = cls.getMethod("forwardLayer", int.class, int.class);
            gpuForwardGraph = cls.getMethod("forwardGraph", float[].class);
            gpuForwardFinalLogits = cls.getMethod("forwardFinalLogits", float[].class);
            gpuDownloadX = cls.getMethod("downloadX", float[].class);
            gpuLayerCount = (Integer) cls.getMethod("getGpuLayerCount").invoke(fwd);
            gpuForwardPass = (AutoCloseable) fwd;
            System.err.println("NemotronH CUDA forward pass: enabled (" + gpuLayerCount + "/" + blockCount + " layers)");
        } catch (Throwable e) {
            System.err.println("NemotronH CUDA forward pass: unavailable — " + e.getMessage());
        }
    }

    public NemotronHState createState(int maxSeqLen) {
        return new NemotronHState(config, maxSeqLen);
    }

    public float[] forward(NemotronHState state, int token, int position) {
        return forwardInternal(state, token, position, true);
    }

    public void forwardNoOutput(NemotronHState state, int token, int position) {
        forwardInternal(state, token, position, false);
    }

    private float[] forwardInternal(NemotronHState state, int token, int position, boolean computeLogits) {
        // 1. Token embedding
        for (int i = 0; i < dim; i++) {
            state.x[i] = weights.tokenEmbedding().getFloat((long) token * dim + i);
        }
        // Granite Hybrid: embedding scaling
        if (embeddingScale > 0f) {
            for (int i = 0; i < dim; i++) state.x[i] *= embeddingScale;
        }

        // 2. Try GPU forward pass
        if (gpuForwardPass != null) {
            try {
                return forwardGpu(state, position, computeLogits);
            } catch (Exception e) {
                System.err.println("NemotronH GPU forward failed: " + e.getMessage());
                gpuForwardPass = null;
            }
        }

        // 3. CPU forward through all layers
        for (int layer = 0; layer < blockCount; layer++) {
            NemotronHLayerWeights lw = weights.layers()[layer];
            if (lw.isMamba()) {
                forwardMamba2(state, lw, layer);
            } else if (lw.isAttention()) {
                forwardAttention(state, lw, layer, position);
            } else {
                forwardFFN(state, lw, layer);
            }
        }

        if (!computeLogits) return null;

        // 3. Final RMSNorm + output projection
        VectorOpsFactory.get().rmsnorm(state.xb, state.x, outputNormCache, dim, normEps);
        Arrays.fill(state.logits, 0);
        weights.output().matmulParallel(state.xb, state.logits, vocabSize, dim);
        // Granite Hybrid: logit scaling (divide by logitScale)
        if (logitScale > 0f) {
            float scale = 1.0f / logitScale;
            for (int i = 0; i < vocabSize; i++) state.logits[i] *= scale;
        }
        return state.logits;
    }

    private float[] forwardGpu(NemotronHState state, int position, boolean computeLogits) throws Exception {
        gpuUploadXAndUpdateParams.invoke(gpuForwardPass, state.x, position);
        if (computeLogits && gpuLayerCount == blockCount) {
            Boolean ok = (Boolean) gpuForwardGraph.invoke(gpuForwardPass, state.logits);
            if (ok) {
                // Granite Hybrid: logit scaling after GPU computation
                if (logitScale > 0f) {
                    float scale = 1.0f / logitScale;
                    for (int i = 0; i < vocabSize; i++) state.logits[i] *= scale;
                }
                return state.logits;
            }
        }
        for (int i = 0; i < gpuLayerCount; i++) gpuForwardLayer.invoke(gpuForwardPass, i, position);
        if (gpuLayerCount < blockCount) {
            gpuDownloadX.invoke(gpuForwardPass, state.x);
            for (int i = gpuLayerCount; i < blockCount; i++) {
                NemotronHLayerWeights lw = weights.layers()[i];
                if (lw.isMamba()) forwardMamba2(state, lw, i);
                else if (lw.isAttention()) forwardAttention(state, lw, i, position);
                else forwardFFN(state, lw, i);
            }
        }
        if (!computeLogits) return null;
        if (gpuLayerCount == blockCount) {
            Boolean ok = (Boolean) gpuForwardFinalLogits.invoke(gpuForwardPass, state.logits);
            if (ok) {
                if (logitScale > 0f) {
                    float scale = 1.0f / logitScale;
                    for (int i = 0; i < vocabSize; i++) state.logits[i] *= scale;
                }
                return state.logits;
            }
            gpuDownloadX.invoke(gpuForwardPass, state.x);
        }
        VectorOpsFactory.get().rmsnorm(state.xb, state.x, outputNormCache, dim, normEps);
        Arrays.fill(state.logits, 0);
        weights.output().matmulParallel(state.xb, state.logits, vocabSize, dim);
        return state.logits;
    }

    public float[] prefill(NemotronHState state, int[] tokens) {
        for (int i = 0; i < tokens.length - 1; i++) forwardNoOutput(state, tokens[i], i);
        return forward(state, tokens[tokens.length - 1], tokens.length - 1);
    }

    // ==================== Mamba-2 ====================

    private void forwardMamba2(NemotronHState state, NemotronHLayerWeights lw, int layer) {
        // RMSNorm
        float[] normW = cacheWeights(lw.attnNorm(), dim);
        VectorOpsFactory.get().rmsnorm(state.xb, state.x, normW, dim, normEps);

        // Input projection: xb -> zxBCdt [ssmInnerSize + convChannels + ssmTimeStepRank]
        int projDim = ssmInnerSize + convChannels + ssmTimeStepRank;
        Arrays.fill(state.zxBCdt, 0, projDim, 0);
        lw.ssmIn().matmulParallel(state.xb, state.zxBCdt, projDim, dim);

        // Split: z (gate), xBC (conv input), dt
        // z: [0, ssmInnerSize)
        // xBC: [ssmInnerSize, ssmInnerSize + convChannels)
        // dt: [ssmInnerSize + convChannels, projDim)
        int xbcOffset = ssmInnerSize;
        int dtOffset = ssmInnerSize + convChannels;

        // Copy xBC to state buffer for conv1d
        System.arraycopy(state.zxBCdt, xbcOffset, state.xBC, 0, convChannels);

        // Causal conv1d on xBC
        applyConv1d(state, lw, layer);

        // SiLU activation on xBC (after conv)
        for (int i = 0; i < convChannels; i++) {
            float v = state.xBC[i];
            state.xBC[i] = v / (1.0f + (float) Math.exp(-v)); // SiLU
        }

        // Split xBC after conv: x[innerSize], B[groupCount*stateSize], C[groupCount*stateSize]
        System.arraycopy(state.xBC, 0, state.ssm_x, 0, ssmInnerSize);
        int bOffset = ssmInnerSize;
        int cOffset = ssmInnerSize + ssmGroupCount * ssmStateSize;

        // Discretize dt: softplus(dt + dt_bias)
        float[] dt = new float[ssmTimeStepRank];
        for (int h = 0; h < ssmTimeStepRank; h++) {
            float d = state.zxBCdt[dtOffset + h] + lw.ssmDtBias().getFloat(h);
            dt[h] = (d > 20) ? d : (float) Math.log(1.0 + Math.exp(d)); // softplus
        }

        // SSM scan per head
        mamba2Scan(state, lw, layer, dt);

        // Gate FIRST (norm_before_gate=False): y *= SiLU(z)
        for (int i = 0; i < ssmInnerSize; i++) {
            float z = state.zxBCdt[i]; // z = first ssmInnerSize elements
            state.ssm_y[i] *= z / (1.0f + (float) Math.exp(-z)); // * SiLU(z)
        }

        // THEN grouped RMSNorm: 8 groups, each of innerSize/8 = 960 elements
        applyGroupedNorm(state.ssm_y, lw.ssmNorm(), ssmGroupCount, ssmInnerSize / ssmGroupCount, normEps);

        // Output projection
        Arrays.fill(state.xb, 0);
        lw.ssmOut().matmulParallel(state.ssm_y, state.xb, dim, ssmInnerSize);

        // Residual (with Granite scaling if configured)
        applyResidual(state);

        // Integrated FFN (Granite Hybrid: Mamba layers also have SwiGLU FFN)
        if (lw.ffnUp() != null) {
            runIntegratedFFN(state, lw);
        }
    }

    private void applyConv1d(NemotronHState state, NemotronHLayerWeights lw, int layer) {
        float[][] convBuf = state.convState[layer];
        int histSize = ssmConvKernel - 1;
        int pos = state.convStatePos[layer];

        float[] result = new float[convChannels];
        for (int ch = 0; ch < convChannels; ch++) {
            float sum = lw.ssmConv1d().getFloat((long) ch * ssmConvKernel + (ssmConvKernel - 1)) * state.xBC[ch];
            for (int k = 1; k < ssmConvKernel; k++) {
                if (pos - k >= 0) {
                    int histIdx = (pos - k) % histSize;
                    sum += lw.ssmConv1d().getFloat((long) ch * ssmConvKernel + (ssmConvKernel - 1 - k))
                         * convBuf[histIdx][ch];
                }
            }
            // Add conv1d bias
            sum += lw.ssmConv1dBias().getFloat(ch);
            result[ch] = sum;
        }

        System.arraycopy(state.xBC, 0, convBuf[pos % histSize], 0, convChannels);
        state.convStatePos[layer] = pos + 1;
        System.arraycopy(result, 0, state.xBC, 0, convChannels);
    }

    private void mamba2Scan(NemotronHState state, NemotronHLayerWeights lw, int layer, float[] dt) {
        int bOffset = ssmInnerSize;
        int cOffset = ssmInnerSize + ssmGroupCount * ssmStateSize;

        IntStream.range(0, ssmTimeStepRank).parallel().forEach(h -> {
            int group = h / (ssmTimeStepRank / ssmGroupCount); // which B/C group this head belongs to
            float dtH = dt[h];
            float logA = lw.ssmA().getFloat(h);
            float dH = lw.ssmD().getFloat(h);
            float dA = (float) Math.exp(dtH * logA); // A is stored as -exp(A_log), so dA ∈ (0,1)

            float[] S = state.ssmState[layer][h]; // [headDim * stateSize]
            int xOff = h * headDim; // this head's portion of ssm_x

            // B and C for this group
            int bOff = bOffset + group * ssmStateSize;
            int cOff = cOffset + group * ssmStateSize;

            // For each element in this head's output
            for (int d = 0; d < headDim; d++) {
                float x_val = state.ssm_x[xOff + d] * dtH;
                int sOff = d * ssmStateSize; // each head_dim element has its own state vector

                // SSM state update: S[d,n] = dA * S[d,n] + dt * B[n] * x[d]
                // Output: y[d] = sum_n(S[d,n] * C[n])
                float y_val = 0;
                for (int n = 0; n < ssmStateSize; n++) {
                    S[sOff + n] = dA * S[sOff + n] + state.xBC[bOff + n] * x_val;
                    y_val += S[sOff + n] * state.xBC[cOff + n];
                }

                // D residual
                state.ssm_y[xOff + d] = y_val + dH * state.ssm_x[xOff + d];
            }
        });
    }

    // ==================== Attention ====================

    private void forwardAttention(NemotronHState state, NemotronHLayerWeights lw, int layer, int position) {
        float[] normW = cacheWeights(lw.attnNorm(), dim);
        VectorOpsFactory.get().rmsnorm(state.xb, state.x, normW, dim, normEps);

        int qDim = headCount * headSize;

        // Q, K, V projections
        Arrays.fill(state.q, 0, qDim, 0);
        lw.wq().matmulParallel(state.xb, state.q, qDim, dim);
        Arrays.fill(state.k, 0, kvDim, 0);
        lw.wk().matmulParallel(state.xb, state.k, kvDim, dim);
        Arrays.fill(state.v, 0, kvDim, 0);
        lw.wv().matmulParallel(state.xb, state.v, kvDim, dim);

        // RoPE (partial: only ropeDimCount out of headSize dims)
        rope.applyAllHeads(state.q, headCount, position);
        rope.applyAllHeads(state.k, headCountKV, position);

        // KV cache (transparent Q8 quantization if -Dkv.q8=true)
        state.kvCache.storeK(layer, position, state.k, kvDim);
        state.kvCache.storeV(layer, position, state.v, kvDim);

        // Multi-head attention (GQA) — Granite Hybrid uses custom attentionScale if set
        final float invSqrt = config.attentionScale() > 0f
            ? config.attentionScale()
            : (1.0f / (float) Math.sqrt(headSize));
        final KVCache kv = state.kvCache;
        final int layerFinal = layer;
        final int positionFinal = position;
        final int headSizeFinal = headSize;
        IntStream.range(0, headCount).parallel().forEach(h -> {
            int kvHead = h / kvMul;
            int qOff = h * headSizeFinal;
            int kvHeadOff = kvHead * headSizeFinal;

            for (int t = 0; t <= positionFinal; t++) {
                float score = kv.dotK(layerFinal, t, kvHeadOff, headSizeFinal, state.q, qOff);
                state.att[h * maxSeqLen + t] = score * invSqrt;
            }

            softmax(state.att, h * maxSeqLen, positionFinal + 1);

            int outOff = h * headSizeFinal;
            Arrays.fill(state.xb2, outOff, outOff + headSizeFinal, 0);
            for (int t = 0; t <= positionFinal; t++) {
                float a = state.att[h * maxSeqLen + t];
                kv.saxpyV(layerFinal, t, kvHeadOff, headSizeFinal, a, state.xb2, outOff);
            }
        });

        // Output projection
        Arrays.fill(state.xb, 0);
        lw.wo().matmulParallel(state.xb2, state.xb, dim, headCount * headSize);

        // Residual (with Granite scaling if configured)
        applyResidual(state);

        // Integrated FFN (Granite Hybrid: attention layers also have SwiGLU FFN)
        if (lw.ffnUp() != null) {
            runIntegratedFFN(state, lw);
        }
    }

    // ==================== FFN (squared ReLU) ====================

    private void forwardFFN(NemotronHState state, NemotronHLayerWeights lw, int layer) {
        float[] normW = cacheWeights(lw.attnNorm(), dim);
        VectorOpsFactory.get().rmsnorm(state.xb, state.x, normW, dim, normEps);

        int ffnDim = config.nemotronLayerFfnLength(layer);

        if (lw.ffnGate() != null) {
            // SwiGLU FFN (Granite Hybrid): gate + up + SiLU + element-wise mul + down
            // Use ssm_x as temp buffer for up projection (large enough: ssmInnerSize >= ffnDim for micro)
            float[] gate = state.hb;
            float[] up = state.ssm_x.length >= ffnDim ? state.ssm_x : new float[ffnDim];
            Arrays.fill(gate, 0, ffnDim, 0);
            Arrays.fill(up, 0, ffnDim, 0);
            lw.ffnGate().matmulParallel(state.xb, gate, ffnDim, dim);
            lw.ffnUp().matmulParallel(state.xb, up, ffnDim, dim);
            for (int i = 0; i < ffnDim; i++) {
                float v = gate[i];
                gate[i] = (v / (1.0f + (float) Math.exp(-v))) * up[i]; // SiLU(gate) * up
            }
            Arrays.fill(state.xb, 0);
            lw.ffnDown().matmulParallel(gate, state.xb, dim, ffnDim);
        } else {
            // Squared ReLU FFN (Nemotron-H): up + sqReLU + down
            Arrays.fill(state.hb, 0, ffnDim, 0);
            lw.ffnUp().matmulParallel(state.xb, state.hb, ffnDim, dim);
            for (int i = 0; i < ffnDim; i++) {
                float v = state.hb[i];
                state.hb[i] = v > 0 ? v * v : 0;
            }
            Arrays.fill(state.xb, 0);
            lw.ffnDown().matmulParallel(state.hb, state.xb, dim, ffnDim);
        }

        // Residual (with Granite scaling if configured)
        applyResidual(state);
    }

    // ==================== Integrated FFN (Granite Hybrid) ====================

    /**
     * Run SwiGLU FFN after Mamba or Attention layer.
     * Uses ffnNorm (separate from attnNorm) and SwiGLU activation.
     */
    private void runIntegratedFFN(NemotronHState state, NemotronHLayerWeights lw) {
        it.denzosoft.llmplayer.tensor.FloatTensor normTensor = lw.ffnNorm() != null ? lw.ffnNorm() : lw.attnNorm();
        float[] normW = cacheWeights(normTensor, dim);
        VectorOpsFactory.get().rmsnorm(state.xb, state.x, normW, dim, normEps);

        int ffnDim = config.intermediateSize();

        // SwiGLU: gate + up + SiLU + mul + down
        float[] gate = state.hb;
        float[] up = state.ssm_x.length >= ffnDim ? state.ssm_x : new float[ffnDim];
        Arrays.fill(gate, 0, ffnDim, 0);
        Arrays.fill(up, 0, ffnDim, 0);
        lw.ffnGate().matmulParallel(state.xb, gate, ffnDim, dim);
        lw.ffnUp().matmulParallel(state.xb, up, ffnDim, dim);
        for (int i = 0; i < ffnDim; i++) {
            float v = gate[i];
            gate[i] = (v / (1.0f + (float) Math.exp(-v))) * up[i];
        }
        Arrays.fill(state.xb, 0);
        lw.ffnDown().matmulParallel(gate, state.xb, dim, ffnDim);

        // Residual (with Granite scaling if configured)
        applyResidual(state);
    }

    // ==================== Residual ====================

    private void applyResidual(NemotronHState state) {
        if (residualScale > 0f) {
            for (int i = 0; i < dim; i++) state.x[i] += residualScale * state.xb[i];
        } else {
            for (int i = 0; i < dim; i++) state.x[i] += state.xb[i];
        }
    }

    // ==================== Utility ====================

    private float[] cacheWeights(it.denzosoft.llmplayer.tensor.FloatTensor tensor, int size) {
        float[] cache = new float[size];
        for (int i = 0; i < size; i++) cache[i] = tensor.getFloat(i);
        return cache;
    }

    private static void applyGroupedNorm(float[] data, it.denzosoft.llmplayer.tensor.FloatTensor normWeights,
                                          int numGroups, int groupSize, float eps) {
        // GGUF ssm_norm shape [groupSize, numGroups]: ne[0]=groupSize is contiguous.
        // Element i of group g = normWeights[g * groupSize + i]
        for (int g = 0; g < numGroups; g++) {
            int off = g * groupSize;
            float ss = 0;
            for (int i = 0; i < groupSize; i++) ss += data[off + i] * data[off + i];
            ss = 1.0f / (float) Math.sqrt(ss / groupSize + eps);
            for (int i = 0; i < groupSize; i++)
                data[off + i] = data[off + i] * ss * normWeights.getFloat((long) g * groupSize + i);
        }
    }

    private static void softmax(float[] x, int offset, int size) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < size; i++) if (x[offset + i] > max) max = x[offset + i];
        float sum = 0;
        for (int i = 0; i < size; i++) { x[offset + i] = (float) Math.exp(x[offset + i] - max); sum += x[offset + i]; }
        for (int i = 0; i < size; i++) x[offset + i] /= sum;
    }

    public ModelConfig getConfig() { return config; }
}
