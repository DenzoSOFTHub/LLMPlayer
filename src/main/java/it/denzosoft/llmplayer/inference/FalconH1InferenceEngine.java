package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.FalconH1LayerWeights;
import it.denzosoft.llmplayer.model.FalconH1Weights;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Inference engine for Falcon-H1: every layer runs a GQA attention path AND a Mamba-2 SSM path
 * in parallel on the same pre-normed input, sums their outputs, then applies a SwiGLU FFN.
 * Per llama.cpp src/models/falcon-h1.cpp + src/models/mamba-base.cpp (build_mamba2_layer).
 * Channel/attention scaling multipliers are baked into the GGUF weights at conversion time.
 */
public class FalconH1InferenceEngine {

    private final ModelConfig config;
    private final FalconH1Weights weights;
    private final int maxSeqLen;

    private final int dim, vocabSize, blockCount;
    private final float normEps;

    // mamba-2 dims
    private final int ssmInner, ssmState, ssmGroups, nheads, ssmConv, convChannels, headDim;
    // attention dims
    private final int headCount, headCountKV, headSize, kvDim, kvMul;
    private final int ffnDim;

    private final float[] outputNormCache;
    private final float[][] attnNormPerLayer;
    private final float[][] ffnNormPerLayer;

    private final RoPE rope;

    // GPU-resident forward pass (reflection-loaded from java21; null on CPU/unsupported)
    private AutoCloseable gpuForwardPass;
    private int gpuLayerCount;
    private Method gpuUploadX, gpuForwardLayer, gpuForwardFinalLogits, gpuDownloadX;

    public FalconH1InferenceEngine(ModelConfig config, FalconH1Weights weights, int maxSeqLen, float[] ropeFreqFactors) {
        this.config = config;
        this.weights = weights;
        this.maxSeqLen = maxSeqLen;
        this.dim = config.embeddingLength();
        this.vocabSize = config.vocabSize();
        this.blockCount = config.blockCount();
        this.normEps = config.normEps();

        this.ssmInner = config.ssmInnerSize();
        this.ssmState = config.ssmStateSize();
        this.ssmGroups = config.ssmGroupCount();
        this.nheads = config.ssmTimeStepRank();
        this.ssmConv = config.ssmConvKernel();
        this.convChannels = ssmInner + 2 * ssmGroups * ssmState;
        this.headDim = ssmInner / nheads;

        this.headCount = config.headCount();
        this.headCountKV = config.headCountKV();
        this.headSize = config.headSize();
        this.kvDim = config.kvDim();
        this.kvMul = headCount / headCountKV;
        this.ffnDim = config.intermediateSize();

        this.outputNormCache = cache(weights.outputNorm(), dim);
        this.attnNormPerLayer = new float[blockCount][];
        this.ffnNormPerLayer = new float[blockCount][];
        for (int i = 0; i < blockCount; i++) {
            attnNormPerLayer[i] = cache(weights.layers()[i].attnNorm(), dim);
            ffnNormPerLayer[i] = cache(weights.layers()[i].ffnNorm(), dim);
        }

        this.rope = new RoPE(headSize, config.ropeDimensionCount(), maxSeqLen,
            config.ropeFreqBase(), config.ropeType(), ropeFreqFactors);
    }

    public void tryInitGpuForwardPass(Object bufferManager) {
        try {
            Class<?> cls = Class.forName("it.denzosoft.llmplayer.inference.FalconH1CudaForwardPass");
            Method isSup = cls.getMethod("isSupported", ModelConfig.class, FalconH1Weights.class);
            if (!(Boolean) isSup.invoke(null, config, weights)) return;
            Object fwd = cls.getConstructor(ModelConfig.class, FalconH1Weights.class,
                    bufferManager.getClass(), int.class).newInstance(config, weights, bufferManager, maxSeqLen);
            gpuUploadX = cls.getMethod("uploadXAndUpdateParams", float[].class, int.class);
            gpuForwardLayer = cls.getMethod("forwardLayer", int.class, int.class);
            gpuForwardFinalLogits = cls.getMethod("forwardFinalLogits", float[].class);
            gpuDownloadX = cls.getMethod("downloadX", float[].class);
            gpuLayerCount = (Integer) cls.getMethod("getGpuLayerCount").invoke(fwd);
            gpuForwardPass = (AutoCloseable) fwd;
            System.err.println("Falcon-H1 CUDA forward pass: enabled (" + gpuLayerCount + "/" + blockCount + " layers)");
        } catch (Throwable e) {
            System.err.println("Falcon-H1 CUDA forward pass: unavailable — " + e.getMessage());
        }
    }

    private float[] forwardGpu(FalconH1State state, int position, boolean computeLogits) throws Exception {
        gpuUploadX.invoke(gpuForwardPass, state.x, position);
        for (int i = 0; i < gpuLayerCount; i++) gpuForwardLayer.invoke(gpuForwardPass, i, position);
        if (!computeLogits) return null;
        gpuForwardFinalLogits.invoke(gpuForwardPass, state.logits);
        return state.logits;
    }

    public FalconH1State createState(int maxSeqLen) { return new FalconH1State(config, maxSeqLen); }

    public float[] forward(FalconH1State state, int token, int position) {
        return forwardInternal(state, token, position, true);
    }

    public void forwardNoOutput(FalconH1State state, int token, int position) {
        forwardInternal(state, token, position, false);
    }

    public float[] prefill(FalconH1State state, int[] tokens) {
        for (int i = 0; i < tokens.length - 1; i++) forwardNoOutput(state, tokens[i], i);
        return forward(state, tokens[tokens.length - 1], tokens.length - 1);
    }

    private float[] forwardInternal(FalconH1State state, int token, int position, boolean computeLogits) {
        for (int i = 0; i < dim; i++) state.x[i] = weights.tokenEmbedding().getFloat((long) token * dim + i);

        if (gpuForwardPass != null) {
            try {
                return forwardGpu(state, position, computeLogits);
            } catch (Exception e) {
                System.err.println("Falcon-H1 GPU forward failed: " + e.getMessage());
                gpuForwardPass = null;
            }
        }

        for (int layer = 0; layer < blockCount; layer++) {
            FalconH1LayerWeights lw = weights.layers()[layer];

            // shared pre-norm for both mixers
            VectorOpsFactory.get().rmsnorm(state.nrm, state.x, attnNormPerLayer[layer], dim, normEps);
            attentionPath(state, lw, layer, position);   // -> state.attnOut
            mambaPath(state, lw, layer);                 // -> state.ssmOut

            // parallel aggregation: x += attn_out + ssm_out
            for (int i = 0; i < dim; i++) state.x[i] += state.attnOut[i] + state.ssmOut[i];

            // SwiGLU FFN with its own norm + residual
            VectorOpsFactory.get().rmsnorm(state.nrm, state.x, ffnNormPerLayer[layer], dim, normEps);
            swiglu(state, lw);
            for (int i = 0; i < dim; i++) state.x[i] += state.attnOut[i]; // ffn output stored in attnOut (reused)
        }

        if (!computeLogits) return null;
        VectorOpsFactory.get().rmsnorm(state.nrm, state.x, outputNormCache, dim, normEps);
        Arrays.fill(state.logits, 0);
        weights.output().matmulParallel(state.nrm, state.logits, vocabSize, dim);
        return state.logits;
    }

    // ==================== Attention path (GQA, no QK-norm) ====================

    private void attentionPath(FalconH1State state, FalconH1LayerWeights lw, int layer, int position) {
        int qDim = headCount * headSize;
        Arrays.fill(state.q, 0, qDim, 0);
        lw.wq().matmulParallel(state.nrm, state.q, qDim, dim);
        Arrays.fill(state.k, 0, kvDim, 0);
        lw.wk().matmulParallel(state.nrm, state.k, kvDim, dim);
        Arrays.fill(state.v, 0, kvDim, 0);
        lw.wv().matmulParallel(state.nrm, state.v, kvDim, dim);

        rope.applyAllHeads(state.q, headCount, position);
        rope.applyAllHeads(state.k, headCountKV, position);

        state.kvCache.storeK(layer, position, state.k, kvDim);
        state.kvCache.storeV(layer, position, state.v, kvDim);

        final float invSqrt = config.attentionScale() > 0f
            ? config.attentionScale() : (1.0f / (float) Math.sqrt(headSize));
        final KVCache kv = state.kvCache;
        final int layerF = layer, posF = position, hsF = headSize;
        IntStream.range(0, headCount).parallel().forEach(h -> {
            int kvHead = h / kvMul;
            int qOff = h * hsF;
            int kvHeadOff = kvHead * hsF;
            for (int t = 0; t <= posF; t++) {
                state.att[h * maxSeqLen + t] = kv.dotK(layerF, t, kvHeadOff, hsF, state.q, qOff) * invSqrt;
            }
            softmax(state.att, h * maxSeqLen, posF + 1);
            int outOff = h * hsF;
            Arrays.fill(state.attBuf, outOff, outOff + hsF, 0);
            for (int t = 0; t <= posF; t++) {
                kv.saxpyV(layerF, t, kvHeadOff, hsF, state.att[h * maxSeqLen + t], state.attBuf, outOff);
            }
        });

        Arrays.fill(state.attnOut, 0);
        lw.wo().matmulParallel(state.attBuf, state.attnOut, dim, qDim);
        addBias(state.attnOut, lw.woBias(), dim);
    }

    // ==================== Mamba-2 path ====================

    private void mambaPath(FalconH1State state, FalconH1LayerWeights lw, int layer) {
        int projDim = ssmInner + convChannels + nheads;
        Arrays.fill(state.zxBCdt, 0, projDim, 0);
        lw.ssmIn().matmulParallel(state.nrm, state.zxBCdt, projDim, dim);

        int xbcOffset = ssmInner;
        int dtOffset = ssmInner + convChannels;
        System.arraycopy(state.zxBCdt, xbcOffset, state.xBC, 0, convChannels);

        applyConv1d(state, lw, layer);   // conv + bias (into state.xBC)
        for (int i = 0; i < convChannels; i++) {
            float v = state.xBC[i];
            state.xBC[i] = v / (1.0f + (float) Math.exp(-v)); // SiLU
        }
        System.arraycopy(state.xBC, 0, state.ssm_x, 0, ssmInner);

        float[] dt = new float[nheads];
        for (int h = 0; h < nheads; h++) {
            float d = state.zxBCdt[dtOffset + h] + lw.ssmDtBias().getFloat(h);
            dt[h] = (d > 20) ? d : (float) Math.log(1.0 + Math.exp(d)); // softplus
        }

        mamba2Scan(state, lw, layer, dt);

        // gate first: y *= SiLU(z), z = first ssmInner elements of zxBCdt
        for (int i = 0; i < ssmInner; i++) {
            float z = state.zxBCdt[i];
            state.ssm_y[i] *= z / (1.0f + (float) Math.exp(-z));
        }
        // grouped RMSNorm only when present (Falcon-H1 0.5B has none)
        if (lw.ssmNorm() != null) {
            applyGroupedNorm(state.ssm_y, lw.ssmNorm(), ssmGroups, ssmInner / ssmGroups, normEps);
        }

        Arrays.fill(state.ssmOut, 0);
        lw.ssmOut().matmulParallel(state.ssm_y, state.ssmOut, dim, ssmInner);
    }

    private void applyConv1d(FalconH1State state, FalconH1LayerWeights lw, int layer) {
        float[][] convBuf = state.convState[layer];
        int histSize = ssmConv - 1;
        int pos = state.convStatePos[layer];
        float[] result = state.convResult;
        FloatTensor K = lw.ssmConv1d();
        for (int ch = 0; ch < convChannels; ch++) {
            float sum = K.getFloat((long) ch * ssmConv + (ssmConv - 1)) * state.xBC[ch];
            for (int k = 1; k < ssmConv; k++) {
                if (pos - k >= 0) {
                    int histIdx = (pos - k) % histSize;
                    sum += K.getFloat((long) ch * ssmConv + (ssmConv - 1 - k)) * convBuf[histIdx][ch];
                }
            }
            sum += lw.ssmConv1dBias().getFloat(ch);
            result[ch] = sum;
        }
        System.arraycopy(state.xBC, 0, convBuf[pos % histSize], 0, convChannels);
        state.convStatePos[layer] = pos + 1;
        System.arraycopy(result, 0, state.xBC, 0, convChannels);
    }

    private void mamba2Scan(FalconH1State state, FalconH1LayerWeights lw, int layer, float[] dt) {
        int bOffset = ssmInner;
        int cOffset = ssmInner + ssmGroups * ssmState;
        final it.denzosoft.llmplayer.tensor.VectorOps ops = VectorOpsFactory.get();
        IntStream.range(0, nheads).parallel().forEach(h -> {
            int group = h / (nheads / ssmGroups);
            float dtH = dt[h];
            float aH = lw.ssmA().getFloat(h);          // stored as -exp(A_log)
            float dH = lw.ssmD().getFloat(h);
            float dA = (float) Math.exp(dtH * aH);
            float[] S = state.ssmState[layer][h];      // [headDim * stateSize]
            int bOff = bOffset + group * ssmState;
            int cOff = cOffset + group * ssmState;
            int xOff = h * headDim;
            for (int d = 0; d < headDim; d++) {
                float xVal = state.ssm_x[xOff + d] * dtH;
                int sOff = d * ssmState;
                float yVal = ops.ssmStateUpdate(S, sOff, state.xBC, bOff, state.xBC, cOff,
                                                ssmState, dA, xVal);
                state.ssm_y[xOff + d] = yVal + dH * state.ssm_x[xOff + d];
            }
        });
    }

    // ==================== SwiGLU FFN ====================

    private void swiglu(FalconH1State state, FalconH1LayerWeights lw) {
        Arrays.fill(state.gate, 0, ffnDim, 0);
        Arrays.fill(state.up, 0, ffnDim, 0);
        lw.ffnGate().matmulParallel(state.nrm, state.gate, ffnDim, dim);
        addBias(state.gate, lw.ffnGateBias(), ffnDim);
        lw.ffnUp().matmulParallel(state.nrm, state.up, ffnDim, dim);
        addBias(state.up, lw.ffnUpBias(), ffnDim);
        for (int i = 0; i < ffnDim; i++) {
            float g = state.gate[i];
            state.gate[i] = (g / (1.0f + (float) Math.exp(-g))) * state.up[i];
        }
        Arrays.fill(state.attnOut, 0); // reuse attnOut as ffn output buffer
        lw.ffnDown().matmulParallel(state.gate, state.attnOut, dim, ffnDim);
        addBias(state.attnOut, lw.ffnDownBias(), dim);
    }

    // ==================== Utility ====================

    private static void addBias(float[] x, FloatTensor bias, int n) {
        if (bias == null) return;
        for (int i = 0; i < n; i++) x[i] += bias.getFloat(i);
    }

    private static float[] cache(FloatTensor t, int size) {
        float[] c = new float[size];
        for (int i = 0; i < size; i++) c[i] = t.getFloat(i);
        return c;
    }

    private static void applyGroupedNorm(float[] data, FloatTensor normWeights, int numGroups, int groupSize, float eps) {
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
