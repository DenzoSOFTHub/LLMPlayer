package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.LFM2LayerWeights;
import it.denzosoft.llmplayer.model.LFM2Weights;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Inference engine for LFM2 (Liquid Foundation Model 2): hybrid of gated short-convolution
 * mixers and GQA attention mixers, both followed by a SwiGLU FFN. Per llama.cpp src/models/lfm2.cpp.
 *
 * Per layer:
 *   prev = x; n = RMSNorm(x, operator_norm); blk = conv-or-attn(n); x = prev + blk;
 *   x += SwiGLU(RMSNorm(x, ffn_norm))
 * Final: RMSNorm(x, token_embd_norm) -> logits = output @ x.
 */
public class LFM2InferenceEngine {

    private final ModelConfig config;
    private final LFM2Weights weights;
    private final int maxSeqLen;

    private final int dim, vocabSize, blockCount;
    private final float normEps;

    private final int headCount, headCountKV, headSize, kvDim, kvMul, ffnDim;
    private final int lCache;       // shortconv kernel width
    private final int convHist;     // lCache - 1

    private final float[] outputNormCache;
    private final float[][] opNormPerLayer;
    private final float[][] ffnNormPerLayer;
    private final float[][] qNormPerLayer;   // attention layers only
    private final float[][] kNormPerLayer;

    private final RoPE rope;

    // LFM2-MoE: per-expert views over the 3D expert tensors (no copy), so the SIMD matmulRows
    // kernels run on each expert's rows. Null for dense LFM2.
    private final ExpertViews expertViews;

    // GPU-resident forward pass (reflection-loaded from java21; null on CPU/unsupported)
    private AutoCloseable gpuForwardPass;
    private int gpuLayerCount;
    private Method gpuUploadX, gpuForwardLayer, gpuForwardFinalLogits, gpuDownloadX;

    public LFM2InferenceEngine(ModelConfig config, LFM2Weights weights, int maxSeqLen, float[] ropeFreqFactors) {
        this.config = config;
        this.weights = weights;
        this.maxSeqLen = maxSeqLen;
        this.dim = config.embeddingLength();
        this.vocabSize = config.vocabSize();
        this.blockCount = config.blockCount();
        this.normEps = config.normEps();
        this.headCount = config.headCount();
        this.headCountKV = config.headCountKV();
        this.headSize = config.headSize();
        this.kvDim = config.kvDim();
        this.kvMul = headCount / headCountKV;
        this.ffnDim = config.intermediateSize();
        this.lCache = config.ssmConvKernel();
        this.convHist = Math.max(1, lCache - 1);

        this.outputNormCache = cache(weights.outputNorm(), dim);
        this.opNormPerLayer = new float[blockCount][];
        this.ffnNormPerLayer = new float[blockCount][];
        this.qNormPerLayer = new float[blockCount][];
        this.kNormPerLayer = new float[blockCount][];
        for (int i = 0; i < blockCount; i++) {
            LFM2LayerWeights lw = weights.layers()[i];
            opNormPerLayer[i] = cache(lw.operatorNorm(), dim);
            ffnNormPerLayer[i] = cache(lw.ffnNorm(), dim);
            if (lw.isAttention()) {
                qNormPerLayer[i] = cache(lw.qNorm(), headSize);
                kNormPerLayer[i] = cache(lw.kNorm(), headSize);
            }
        }

        this.rope = new RoPE(headSize, config.ropeDimensionCount(), maxSeqLen,
            config.ropeFreqBase(), config.ropeType(), ropeFreqFactors);

        if (config.expertCount() > 0) {
            int experts = config.expertCount();
            int[] all = new int[experts];
            for (int e = 0; e < experts; e++) all[e] = e;
            long elementsPerSlice = (long) config.expertFfnLength() * dim;
            this.expertViews = new ExpertViews(blockCount, experts);
            for (int i = 0; i < blockCount; i++) {
                LFM2LayerWeights lw = weights.layers()[i];
                if (lw.isMoE()) {
                    expertViews.ensure(i, lw.gateExps(), lw.upExps(), lw.downExps(), all, experts, elementsPerSlice);
                }
            }
        } else {
            this.expertViews = null;
        }
    }

    public void tryInitGpuForwardPass(Object bufferManager) {
        try {
            Class<?> cls = Class.forName("it.denzosoft.llmplayer.inference.LFM2CudaForwardPass");
            Method isSup = cls.getMethod("isSupported", ModelConfig.class, LFM2Weights.class);
            if (!(Boolean) isSup.invoke(null, config, weights)) return;
            Object fwd = cls.getConstructor(ModelConfig.class, LFM2Weights.class,
                    bufferManager.getClass(), int.class).newInstance(config, weights, bufferManager, maxSeqLen);
            gpuUploadX = cls.getMethod("uploadXAndUpdateParams", float[].class, int.class);
            gpuForwardLayer = cls.getMethod("forwardLayer", int.class, int.class);
            gpuForwardFinalLogits = cls.getMethod("forwardFinalLogits", float[].class);
            gpuDownloadX = cls.getMethod("downloadX", float[].class);
            gpuLayerCount = (Integer) cls.getMethod("getGpuLayerCount").invoke(fwd);
            gpuForwardPass = (AutoCloseable) fwd;
            System.err.println("LFM2 CUDA forward pass: enabled (" + gpuLayerCount + "/" + blockCount + " layers)");
        } catch (Throwable e) {
            System.err.println("LFM2 CUDA forward pass: unavailable — " + e.getMessage());
        }
    }

    private float[] forwardGpu(LFM2State state, int position, boolean computeLogits) throws Exception {
        gpuUploadX.invoke(gpuForwardPass, state.x, position);
        for (int i = 0; i < gpuLayerCount; i++) gpuForwardLayer.invoke(gpuForwardPass, i, position);
        if (!computeLogits) return null;
        gpuForwardFinalLogits.invoke(gpuForwardPass, state.logits);
        return state.logits;
    }

    public LFM2State createState(int maxSeqLen) { return new LFM2State(config, maxSeqLen); }

    public float[] forward(LFM2State state, int token, int position) {
        return forwardInternal(state, token, position, true);
    }

    public void forwardNoOutput(LFM2State state, int token, int position) {
        forwardInternal(state, token, position, false);
    }

    public float[] prefill(LFM2State state, int[] tokens) {
        for (int i = 0; i < tokens.length - 1; i++) forwardNoOutput(state, tokens[i], i);
        return forward(state, tokens[tokens.length - 1], tokens.length - 1);
    }

    private float[] forwardInternal(LFM2State state, int token, int position, boolean computeLogits) {
        for (int i = 0; i < dim; i++) state.x[i] = weights.tokenEmbedding().getFloat((long) token * dim + i);

        // GPU-resident forward pass (falls back to CPU on any failure)
        if (gpuForwardPass != null) {
            try {
                return forwardGpu(state, position, computeLogits);
            } catch (Exception e) {
                System.err.println("LFM2 GPU forward failed: " + e.getMessage());
                gpuForwardPass = null;
            }
        }

        for (int layer = 0; layer < blockCount; layer++) {
            LFM2LayerWeights lw = weights.layers()[layer];

            // operator_norm -> mixer -> residual
            VectorOpsFactory.get().rmsnorm(state.nrm, state.x, opNormPerLayer[layer], dim, normEps);
            if (lw.isAttention()) {
                attentionMixer(state, lw, layer, position);
            } else {
                convMixer(state, lw, layer);
            }
            for (int i = 0; i < dim; i++) state.x[i] += state.out[i];

            // ffn_norm -> SwiGLU -> residual
            VectorOpsFactory.get().rmsnorm(state.nrm, state.x, ffnNormPerLayer[layer], dim, normEps);
            if (lw.isMoE()) {
                moe(state, lw, layer);
            } else {
                swiglu(state, lw);
            }
            for (int i = 0; i < dim; i++) state.x[i] += state.out[i];
        }

        if (!computeLogits) return null;
        VectorOpsFactory.get().rmsnorm(state.nrm, state.x, outputNormCache, dim, normEps);
        Arrays.fill(state.logits, 0);
        weights.output().matmulParallel(state.nrm, state.logits, vocabSize, dim);
        return state.logits;
    }

    // ==================== Attention mixer (GQA + QK-norm + RoPE) ====================

    private void attentionMixer(LFM2State state, LFM2LayerWeights lw, int layer, int position) {
        int qDim = headCount * headSize;
        Arrays.fill(state.q, 0, qDim, 0);
        lw.wq().matmulParallel(state.nrm, state.q, qDim, dim);
        Arrays.fill(state.k, 0, kvDim, 0);
        lw.wk().matmulParallel(state.nrm, state.k, kvDim, dim);
        Arrays.fill(state.v, 0, kvDim, 0);
        lw.wv().matmulParallel(state.nrm, state.v, kvDim, dim);

        // per-head QK RMSNorm (before RoPE)
        perHeadRmsNorm(state.q, headCount, qNormPerLayer[layer]);
        perHeadRmsNorm(state.k, headCountKV, kNormPerLayer[layer]);

        rope.applyAllHeads(state.q, headCount, position);
        rope.applyAllHeads(state.k, headCountKV, position);

        state.kvCache.storeK(layer, position, state.k, kvDim);
        state.kvCache.storeV(layer, position, state.v, kvDim);

        final float invSqrt = 1.0f / (float) Math.sqrt(headSize);
        final KVCache kv = state.kvCache;
        final int layerF = layer, posF = position, hsF = headSize;
        it.denzosoft.llmplayer.tensor.MatmulPool.forEach(headCount, h -> {
            int kvHead = h / kvMul;
            int qOff = h * hsF;
            int kvHeadOff = kvHead * hsF;
            for (int t = 0; t <= posF; t++) {
                state.att[h * maxSeqLen + t] = kv.dotK(layerF, t, kvHeadOff, hsF, state.q, qOff) * invSqrt;
            }
            softmax(state.att, h * maxSeqLen, posF + 1);
            int outOff = h * hsF;
            Arrays.fill(state.attOut, outOff, outOff + hsF, 0);
            for (int t = 0; t <= posF; t++) {
                kv.saxpyV(layerF, t, kvHeadOff, hsF, state.att[h * maxSeqLen + t], state.attOut, outOff);
            }
        });

        Arrays.fill(state.out, 0);
        lw.wo().matmulParallel(state.attOut, state.out, dim, qDim);
    }

    // ==================== Short-conv mixer (gated, depthwise causal conv1d) ====================

    private void convMixer(LFM2State state, LFM2LayerWeights lw, int layer) {
        // in_proj -> [b | c | x], each [dim]
        Arrays.fill(state.bcx, 0);
        lw.convInProj().matmulParallel(state.nrm, state.bcx, 3 * dim, dim);
        final int bOff = 0, cOff = dim, xOff = 2 * dim;

        // bx = b * x (gating)
        for (int i = 0; i < dim; i++) state.bx[i] = state.bcx[bOff + i] * state.bcx[xOff + i];

        // depthwise causal conv1d over time, kernel [lCache, dim]: K[k][ch] = conv[ch*lCache + k]
        // window = [oldest state ... newest=bx]; kernel index lCache-1 multiplies the current token.
        FloatTensor K = lw.conv();
        float[][] hist = state.convState[layer];
        int pos = state.convStatePos[layer];
        for (int ch = 0; ch < dim; ch++) {
            float sum = K.getFloat((long) ch * lCache + (lCache - 1)) * state.bx[ch];
            for (int k = 1; k < lCache; k++) {
                if (pos - k >= 0) {
                    int histIdx = (pos - k) % convHist;
                    sum += K.getFloat((long) ch * lCache + (lCache - 1 - k)) * hist[histIdx][ch];
                }
            }
            state.convOut[ch] = sum;
        }
        System.arraycopy(state.bx, 0, hist[pos % convHist], 0, dim);
        state.convStatePos[layer] = pos + 1;

        // y = c * conv_out ; out = out_proj @ y  (reuse bx as y buffer)
        for (int i = 0; i < dim; i++) state.bx[i] = state.bcx[cOff + i] * state.convOut[i];
        Arrays.fill(state.out, 0);
        lw.convOutProj().matmulParallel(state.bx, state.out, dim, dim);
    }

    // ==================== SwiGLU FFN ====================

    private void swiglu(LFM2State state, LFM2LayerWeights lw) {
        Arrays.fill(state.gate, 0, ffnDim, 0);
        Arrays.fill(state.up, 0, ffnDim, 0);
        lw.ffnGate().matmulParallel(state.nrm, state.gate, ffnDim, dim);
        lw.ffnUp().matmulParallel(state.nrm, state.up, ffnDim, dim);
        for (int i = 0; i < ffnDim; i++) {
            float g = state.gate[i];
            state.gate[i] = (g / (1.0f + (float) Math.exp(-g))) * state.up[i];
        }
        Arrays.fill(state.out, 0);
        lw.ffnDown().matmulParallel(state.gate, state.out, dim, ffnDim);
    }

    // ==================== Routed-expert FFN (LFM2-MoE) ====================

    /**
     * llama.cpp build_moe_ffn as called by lfm2.cpp: probs = sigmoid(router logits) (gating func
     * 2); top-K selection on probs + exp_probs_b; the selected UNBIASED probs are sum-normalised
     * (norm_w=true, clamped at the F16 epsilon) and scaled by expert_weights_scale when it is not
     * 0 or 1; out = sum_k w_k * down_k(silu(gate_k(x)) * up_k(x)).
     */
    private void moe(LFM2State state, LFM2LayerWeights lw, int layer) {
        int experts = config.expertCount();
        int k = MoERouting.effectiveTopK(config.expertUsedCount());
        int efd = config.expertFfnLength();

        float[] probs = state.routerProbs;
        Arrays.fill(probs, 0, experts, 0f);
        lw.gateInp().matmul(state.nrm, probs, experts, dim);
        boolean sigmoid = config.expertGatingFunc() == 2;
        if (sigmoid) {
            for (int e = 0; e < experts; e++) probs[e] = 1.0f / (1.0f + (float) Math.exp(-probs[e]));
        } else {
            VectorOpsFactory.get().softmax(probs, 0, experts);
        }
        float[] sel = state.selectionScores;
        FloatTensor bias = lw.expProbsBias();
        for (int e = 0; e < experts; e++) sel[e] = probs[e] + (bias != null ? bias.getFloat(e) : 0f);

        // top-K by selection score (ties: lower index first, as a stable argsort)
        int[] ids = state.selectedExperts;
        float[] w = state.selectedWeights;
        for (int j = 0; j < k; j++) {
            int best = -1;
            for (int e = 0; e < experts; e++) {
                boolean taken = false;
                for (int q = 0; q < j; q++) if (ids[q] == e) { taken = true; break; }
                if (!taken && (best < 0 || sel[e] > sel[best])) best = e;
            }
            ids[j] = best;
            w[j] = probs[best];
        }
        float sum = 0f;
        for (int j = 0; j < k; j++) sum += w[j];
        sum = Math.max(sum, 6.103515625e-5f);
        float scale = config.expertWeightsScale();
        float mul = (scale != 0f && scale != 1f) ? scale / sum : 1f / sum;
        for (int j = 0; j < k; j++) w[j] *= mul;

        // gate/up rows of all K experts as one parallel range, then SiLU*up, then the down rows
        final int kk = k;
        final float[] xin = state.nrm;
        rowsOverExperts(kk, efd, (j, r0, r1) -> {
            float[] g = state.expGate[j], u = state.expUp[j];
            Arrays.fill(g, r0, r1, 0f);
            Arrays.fill(u, r0, r1, 0f);
            expertViews.get(layer, 0, ids[j]).matmulRows(xin, g, r0, r1, dim);
            expertViews.get(layer, 1, ids[j]).matmulRows(xin, u, r0, r1, dim);
            for (int r = r0; r < r1; r++) {
                float gv = g[r];
                g[r] = (gv / (1.0f + (float) Math.exp(-gv))) * u[r];
            }
        });
        rowsOverExperts(kk, dim, (j, r0, r1) -> {
            float[] o = state.expOut[j];
            Arrays.fill(o, r0, r1, 0f);
            expertViews.get(layer, 2, ids[j]).matmulRows(state.expGate[j], o, r0, r1, efd);
        });
        Arrays.fill(state.out, 0);
        for (int j = 0; j < k; j++) {
            VectorOpsFactory.get().saxpy(w[j], state.expOut[j], 0, state.out, 0, dim);
        }
    }

    private interface ExpertRows { void run(int expert, int rowFrom, int rowTo); }

    /**
     * Runs {@code body} over the rows [0, rowsPerExpert) of each of {@code k} experts, splitting the
     * k × rowsPerExpert rows into chunks across the matmul pool (a chunk never spans two experts).
     * With top-4 routing a per-expert loop would keep only four workers busy.
     */
    private static void rowsOverExperts(int k, int rowsPerExpert, ExpertRows body) {
        it.denzosoft.llmplayer.tensor.MatmulPool pool = it.denzosoft.llmplayer.tensor.MatmulPool.enabled()
            ? it.denzosoft.llmplayer.tensor.MatmulPool.get() : null;
        if (pool == null) {
            it.denzosoft.llmplayer.tensor.MatmulPool.forEach(k, j -> body.run(j, 0, rowsPerExpert));
            return;
        }
        pool.parallelFor(k * rowsPerExpert, 16, (from, to) -> {
            int pos = from;
            while (pos < to) {
                int j = pos / rowsPerExpert;
                int r0 = pos - j * rowsPerExpert;
                int r1 = Math.min(rowsPerExpert, r0 + (to - pos));
                body.run(j, r0, r1);
                pos += r1 - r0;
            }
        });
    }

    // ==================== Utility ====================

    private void perHeadRmsNorm(float[] data, int heads, float[] normWeight) {
        for (int h = 0; h < heads; h++) {
            int off = h * headSize;
            float ss = 0;
            for (int i = 0; i < headSize; i++) ss += data[off + i] * data[off + i];
            float scale = 1.0f / (float) Math.sqrt(ss / headSize + normEps);
            for (int i = 0; i < headSize; i++) data[off + i] = data[off + i] * scale * normWeight[i];
        }
    }

    private static float[] cache(FloatTensor t, int size) {
        float[] c = new float[size];
        for (int i = 0; i < size; i++) c[i] = t.getFloat(i);
        return c;
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
