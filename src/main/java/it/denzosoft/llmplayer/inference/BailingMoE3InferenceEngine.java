package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.gguf.GGUFFile;
import it.denzosoft.llmplayer.gguf.GGUFMetadata;
import it.denzosoft.llmplayer.model.ArchitectureRegistry;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.model.ModelLoader;
import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.MatmulPool;
import it.denzosoft.llmplayer.tensor.VectorOps;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

import java.util.Arrays;

/**
 * Inference engine for BailingMoE3 (Ling-3.0, e.g. Ling-3.0-tiny 7.9B-A1.3B), following llama.cpp
 * {@code src/models/bailingmoe3.cpp}. Two layer types, chosen by the per-layer KV head count:
 *
 * <ul>
 *   <li><b>KDA</b> (Kimi Delta Attention, kv heads = 0): q/k/v projections each pass a causal
 *       depthwise short conv + SiLU; q and k are L2-normalised per head; a per-channel log-decay
 *       {@code g = lower_bound * sigmoid(A_h * (f_a x + dt_b))} and a per-head
 *       {@code beta = sigmoid(w_beta x)} drive the gated delta rule
 *       {@code S = diag(exp g) S; d = beta (v - S^T k); S += k d^T; o = S^T q / sqrt(d)};
 *       the output is RMS-normed per head and gated by {@code sigmoid(g_a x)}.</li>
 *   <li><b>Gated MLA</b> (kv heads = 1): Q-LoRA; the KV latent (kv_lora_rank) plus a shared rope
 *       key; queries are absorbed into the latent space with {@code k_b} and the attention output
 *       is expanded with {@code v_b}, then gated per head by {@code sigmoid(w_gate x)}.
 *       The cache stores the latent and the rope key, not per-head K/V.</li>
 * </ul>
 * FFN: the leading dense blocks use SwiGLU; the rest a routed MoE (sigmoid scores with a
 * selection-only bias, group-limited top-k, sum-normalised weights times expert_weights_scale)
 * plus a shared expert.
 */
public class BailingMoE3InferenceEngine {

    private final ModelConfig config;
    private final int dim, vocab, nLayer, nHead, maxSeqLen;
    private final float eps;
    // KDA
    private final int kdaHeadDim, dInner, dConv;
    private final float gateLowerBound;
    // MLA
    private final int qLora, kvLora, qkHeadDim, ropeDim, nopeDim, vHeadDim;
    private final float kqScale;
    private final float[] ropeCos, ropeSin;          // [maxSeqLen][ropeDim/2]
    // MoE
    private final int nExpert, nUsed, nGroups, nGroupsUsed, expFfn, shFfn, ffnDense, nDenseLead;
    private final boolean weightsNorm;
    private final float weightsScale;

    private final FloatTensor tokEmbd, output;
    private final float[] outputNorm;
    private final Layer[] layers;
    private final ExpertViews expertViews;

    private static final class Layer {
        boolean kda;
        float[] attnNorm, ffnNorm;
        // KDA
        FloatTensor wq, wk, wv, fA, gA, betaW, wo;
        float[] convQ, convK, convV, dtB, ssmA, ssmNorm;
        // MLA
        FloatTensor qA, qB, kvA, kB, vB, gate;
        float[] qANorm, kvANorm;
        // FFN
        FloatTensor ffnGate, ffnUp, ffnDown;
        FloatTensor router, gateExps, upExps, downExps, gateSh, upSh, downSh;
        float[] probsBias;
    }

    public BailingMoE3InferenceEngine(ModelConfig config, GGUFFile gguf, int maxSeqLen) {
        this.config = config;
        this.maxSeqLen = maxSeqLen;
        GGUFMetadata md = gguf.getMetadata();
        String p = "bailingmoe3.";
        dim = config.embeddingLength();
        vocab = config.vocabSize();
        nLayer = config.blockCount();
        nHead = config.headCount();
        eps = config.normEps();
        kdaHeadDim = md.getInt(p + "kda.head_dim", 128);
        dInner = kdaHeadDim * nHead;
        dConv = md.getInt(p + "ssm.conv_kernel", 4);
        gateLowerBound = md.getFloat(p + "kda.gate_lower_bound", -5f);
        qLora = md.getInt(p + "attention.q_lora_rank", 0);
        kvLora = md.getInt(p + "attention.kv_lora_rank");
        qkHeadDim = md.getInt(p + "attention.key_length_mla");
        vHeadDim = md.getInt(p + "attention.value_length_mla");
        ropeDim = md.getInt(p + "rope.dimension_count", 64);
        nopeDim = qkHeadDim - ropeDim;
        kqScale = (float) (1.0 / Math.sqrt(qkHeadDim));
        nExpert = config.expertCount();
        nUsed = config.expertUsedCount();
        nGroups = Math.max(1, md.getInt(p + "expert_group_count", 1));
        nGroupsUsed = Math.max(1, md.getInt(p + "expert_group_used_count", nGroups));
        expFfn = config.expertFfnLength();
        shFfn = md.getInt(p + "expert_shared_feed_forward_length", expFfn * Math.max(1, config.expertSharedCount()));
        ffnDense = config.intermediateSize();
        nDenseLead = config.leadingDenseBlockCount();
        weightsNorm = md.getBoolean(p + "expert_weights_norm", false);
        weightsScale = md.getFloat(p + "expert_weights_scale", 1f);

        // RoPE (NORM pairing) for the MLA rope part
        int half = ropeDim / 2;
        ropeCos = new float[maxSeqLen * half];
        ropeSin = new float[maxSeqLen * half];
        double base = config.ropeFreqBase();
        for (int pos = 0; pos < maxSeqLen; pos++) {
            for (int i = 0; i < half; i++) {
                double a = pos * Math.pow(base, -2.0 * i / ropeDim);
                ropeCos[pos * half + i] = (float) Math.cos(a);
                ropeSin[pos * half + i] = (float) Math.sin(a);
            }
        }

        tokEmbd = t(gguf, ArchitectureRegistry.TOKEN_EMBD);
        FloatTensor out = ModelLoader.tryLoadTensor(gguf, ArchitectureRegistry.OUTPUT);
        output = out != null ? out : tokEmbd;
        outputNorm = f(gguf, ArchitectureRegistry.OUTPUT_NORM);

        layers = new Layer[nLayer];
        for (int i = 0; i < nLayer; i++) {
            String b = "blk." + i + ".";
            Layer l = new Layer();
            l.kda = config.layerKvHeads(i) == 0;
            l.attnNorm = f(gguf, b + "attn_norm.weight");
            l.ffnNorm = f(gguf, b + "ffn_norm.weight");
            if (l.kda) {
                l.wq = t(gguf, b + "attn_q.weight");
                l.wk = t(gguf, b + "attn_k.weight");
                l.wv = t(gguf, b + "attn_v.weight");
                l.convQ = f(gguf, b + "ssm_conv1d_q.weight");
                l.convK = f(gguf, b + "ssm_conv1d_k.weight");
                l.convV = f(gguf, b + "ssm_conv1d_v.weight");
                l.fA = t(gguf, b + "ssm_f_a.weight");
                l.gA = t(gguf, b + "ssm_g_a.weight");
                l.betaW = t(gguf, b + "ssm_beta.weight");
                l.dtB = f(gguf, b + "ssm_dt.bias");
                l.ssmA = f(gguf, b + "ssm_a");
                l.ssmNorm = f(gguf, b + "ssm_norm.weight");
            } else {
                if (qLora > 0) {
                    l.qA = t(gguf, b + "attn_q_a.weight");
                    l.qANorm = f(gguf, b + "attn_q_a_norm.weight");
                    l.qB = t(gguf, b + "attn_q_b.weight");
                } else {
                    l.qB = t(gguf, b + "attn_q.weight");
                }
                l.kvA = t(gguf, b + "attn_kv_a_mqa.weight");
                l.kvANorm = f(gguf, b + "attn_kv_a_norm.weight");
                l.kB = t(gguf, b + "attn_k_b.weight");
                l.vB = t(gguf, b + "attn_v_b.weight");
                l.gate = t(gguf, b + "attn_gate.weight");
            }
            l.wo = t(gguf, b + "attn_output.weight");
            if (i < nDenseLead) {
                l.ffnGate = t(gguf, b + "ffn_gate.weight");
                l.ffnUp = t(gguf, b + "ffn_up.weight");
                l.ffnDown = t(gguf, b + "ffn_down.weight");
            } else {
                l.router = t(gguf, b + "ffn_gate_inp.weight");
                l.probsBias = fOpt(gguf, b + "exp_probs_b.bias");
                l.gateExps = t(gguf, b + "ffn_gate_exps.weight");
                l.upExps = t(gguf, b + "ffn_up_exps.weight");
                l.downExps = t(gguf, b + "ffn_down_exps.weight");
                l.gateSh = t(gguf, b + "ffn_gate_shexp.weight");
                l.upSh = t(gguf, b + "ffn_up_shexp.weight");
                l.downSh = t(gguf, b + "ffn_down_shexp.weight");
            }
            layers[i] = l;
        }

        expertViews = new ExpertViews(nLayer, nExpert);
        int[] all = new int[nExpert];
        for (int e = 0; e < nExpert; e++) all[e] = e;
        for (int i = nDenseLead; i < nLayer; i++) {
            Layer l = layers[i];
            expertViews.ensure(i, l.gateExps, l.upExps, l.downExps, all, nExpert, (long) expFfn * dim);
        }
    }

    // ==================== State ====================

    public final class State {
        final float[] x = new float[dim], h = new float[dim], out = new float[dim];
        final float[] logits = new float[vocab];
        // KDA
        final float[][][] conv = new float[nLayer][][];     // [layer][3][(dConv-1) * dInner], ring by position
        final float[][] rec = new float[nLayer][];          // [layer][nHead * d * d], S[i=key][j=value] per head
        final float[] qp = new float[dInner], kp = new float[dInner], vp = new float[dInner];
        final float[] g = new float[dInner], og = new float[dInner], o = new float[dInner];
        final float[] beta = new float[nHead];
        // MLA
        final float[][] latent = new float[nLayer][];       // [layer][maxSeq * kvLora]
        final float[][] kpe = new float[nLayer][];          // [layer][maxSeq * ropeDim]
        final float[] qa = new float[Math.max(1, qLora)], qAll = new float[nHead * qkHeadDim];
        final float[] kvAll = new float[kvLora + ropeDim], qLat = new float[nHead * kvLora];
        final float[] ctx = new float[nHead * kvLora], attnOut = new float[nHead * vHeadDim];
        final float[] headGate = new float[nHead];
        final float[] att;
        // FFN
        final float[] fg = new float[Math.max(ffnDense, shFfn)], fu = new float[Math.max(ffnDense, shFfn)];
        final float[] router = new float[nExpert], sel = new float[nExpert];
        final int[] ids = new int[nUsed];
        final float[] w = new float[nUsed];
        final float[][] eg = new float[nUsed][expFfn], eu = new float[nUsed][expFfn], eo = new float[nUsed][dim];
        int pos;

        State() {
            for (int i = 0; i < nLayer; i++) {
                if (layers[i].kda) {
                    conv[i] = new float[3][(dConv - 1) * dInner];
                    rec[i] = new float[nHead * kdaHeadDim * kdaHeadDim];
                } else {
                    latent[i] = new float[maxSeqLen * kvLora];
                    kpe[i] = new float[maxSeqLen * ropeDim];
                }
            }
            att = new float[nHead * maxSeqLen];
        }
    }

    public State createState() { return new State(); }

    public float[] forward(State s, int token, int position) {
        return forwardInternal(s, token, position, true);
    }

    public void forwardNoOutput(State s, int token, int position) {
        forwardInternal(s, token, position, false);
    }

    private float[] forwardInternal(State s, int token, int pos, boolean logits) {
        long base = (long) token * dim;
        for (int i = 0; i < dim; i++) s.x[i] = tokEmbd.getFloat(base + i);
        VectorOps ops = VectorOpsFactory.get();
        for (int li = 0; li < nLayer; li++) {
            Layer l = layers[li];
            ops.rmsnorm(s.h, s.x, l.attnNorm, dim, eps);
            if (l.kda) kda(s, l, li, pos);
            else mla(s, l, li, pos);
            ops.accumulate(s.x, s.out, dim);
            ops.rmsnorm(s.h, s.x, l.ffnNorm, dim, eps);
            if (li < nDenseLead) dense(s, l);
            else moe(s, l, li);
            ops.accumulate(s.x, s.out, dim);
        }
        if (!logits) return null;
        ops.rmsnorm(s.h, s.x, outputNorm, dim, eps);
        Arrays.fill(s.logits, 0f);
        output.matmulParallel(s.h, s.logits, vocab, dim);
        return s.logits;
    }

    // ==================== KDA ====================

    private void kda(State s, Layer l, int li, int pos) {
        int d = kdaHeadDim;
        zero(s.qp, s.kp, s.vp, s.g, s.og);
        FloatTensor.fusedQKVMatmulParallel(l.wq, l.wk, l.wv, s.h, s.qp, s.kp, s.vp, dInner, dInner, dim);
        l.fA.matmulParallel(s.h, s.g, dInner, dim);
        l.gA.matmulParallel(s.h, s.og, dInner, dim);
        Arrays.fill(s.beta, 0f);
        l.betaW.matmul(s.h, s.beta, nHead, dim);

        shortConvSilu(s.conv[li][0], l.convQ, s.qp, pos);
        shortConvSilu(s.conv[li][1], l.convK, s.kp, pos);
        shortConvSilu(s.conv[li][2], l.convV, s.vp, pos);

        for (int c = 0; c < dInner; c++) {
            float a = l.ssmA[c / d];
            s.g[c] = gateLowerBound * sigmoid(a * (s.g[c] + l.dtB[c]));
        }
        for (int hh = 0; hh < nHead; hh++) {
            s.beta[hh] = sigmoid(s.beta[hh]);
            l2norm(s.qp, hh * d, d);
            l2norm(s.kp, hh * d, d);
        }

        final float scale = (float) (1.0 / Math.sqrt(d));
        final float[] S = s.rec[li];
        MatmulPool.forEach(nHead, hh -> {
            int off = hh * d, so = hh * d * d;
            // S[i][j] stored at so + i*d + j (i = key channel, j = value channel)
            for (int i = 0; i < d; i++) {
                float decay = (float) Math.exp(s.g[off + i]);
                int r = so + i * d;
                for (int j = 0; j < d; j++) S[r + j] *= decay;
            }
            float[] delta = new float[d];
            for (int i = 0; i < d; i++) {
                float ki = s.kp[off + i];
                int r = so + i * d;
                for (int j = 0; j < d; j++) delta[j] += S[r + j] * ki;
            }
            float b = s.beta[hh];
            for (int j = 0; j < d; j++) delta[j] = (s.vp[off + j] - delta[j]) * b;
            for (int i = 0; i < d; i++) {
                float ki = s.kp[off + i];
                int r = so + i * d;
                for (int j = 0; j < d; j++) S[r + j] += ki * delta[j];
            }
            Arrays.fill(s.o, off, off + d, 0f);
            for (int i = 0; i < d; i++) {
                float qi = s.qp[off + i] * scale;
                int r = so + i * d;
                for (int j = 0; j < d; j++) s.o[off + j] += S[r + j] * qi;
            }
            // per-head RMSNorm then output gate
            float ss = 0f;
            for (int j = 0; j < d; j++) ss += s.o[off + j] * s.o[off + j];
            float inv = 1f / (float) Math.sqrt(ss / d + eps);
            for (int j = 0; j < d; j++) {
                s.o[off + j] = s.o[off + j] * inv * l.ssmNorm[j] * sigmoid(s.og[off + j]);
            }
        });
        Arrays.fill(s.out, 0f);
        l.wo.matmulParallel(s.o, s.out, dim, dInner);
    }

    /**
     * Causal depthwise conv over time + SiLU, in place on {@code x} ([dInner]); kernel [dConv][dInner]
     * in GGUF order (flat k + dConv * c); {@code hist} keeps the previous dConv-1 inputs.
     */
    private void shortConvSilu(float[] hist, float[] w, float[] x, int pos) {
        int h = dConv - 1;
        for (int c = 0; c < dInner; c++) {
            float cur = x[c];
            float sum = w[c * dConv + h] * cur;
            for (int k = 0; k < h; k++) {
                // hist slot k holds the input from (h - k) steps ago, oldest first
                sum += w[c * dConv + k] * hist[k * dInner + c];
            }
            // shift history: drop oldest, append current
            for (int k = 0; k < h - 1; k++) hist[k * dInner + c] = hist[(k + 1) * dInner + c];
            if (h > 0) hist[(h - 1) * dInner + c] = cur;
            x[c] = sum / (1f + (float) Math.exp(-sum));
        }
    }

    // ==================== Gated MLA ====================

    private void mla(State s, Layer l, int li, int pos) {
        int half = ropeDim / 2;
        if (l.qA != null) {
            Arrays.fill(s.qa, 0f);
            l.qA.matmulParallel(s.h, s.qa, qLora, dim);
            VectorOpsFactory.get().rmsnorm(s.qa, s.qa, l.qANorm, qLora, eps);
            Arrays.fill(s.qAll, 0f);
            l.qB.matmulParallel(s.qa, s.qAll, nHead * qkHeadDim, qLora);
        } else {
            Arrays.fill(s.qAll, 0f);
            l.qB.matmulParallel(s.h, s.qAll, nHead * qkHeadDim, dim);
        }
        Arrays.fill(s.kvAll, 0f);
        l.kvA.matmulParallel(s.h, s.kvAll, kvLora + ropeDim, dim);
        Arrays.fill(s.headGate, 0f);
        l.gate.matmul(s.h, s.headGate, nHead, dim);

        // rope (NORM pairing) on q_pe of every head and on the shared k_pe
        int tb = pos * half;
        for (int hh = 0; hh < nHead; hh++) ropeNorm(s.qAll, hh * qkHeadDim + nopeDim, tb, half);
        ropeNorm(s.kvAll, kvLora, tb, half);
        // latent norm, then cache latent + k_pe
        VectorOpsFactory.get().rmsnorm(s.kvAll, s.kvAll, l.kvANorm, kvLora, eps);
        System.arraycopy(s.kvAll, 0, s.latent[li], pos * kvLora, kvLora);
        System.arraycopy(s.kvAll, kvLora, s.kpe[li], pos * ropeDim, ropeDim);

        // absorb q_nope into the latent space: qLat[h][c] = sum_d kB[h][c][d] * qNope[h][d]
        final float[] lat = s.latent[li], kp = s.kpe[li];
        final int n = pos + 1;
        VectorOps ops = VectorOpsFactory.get();
        MatmulPool.forEach(nHead, hh -> {
            int qo = hh * qkHeadDim, lo = hh * kvLora;
            long kbBase = (long) hh * kvLora * nopeDim;
            for (int c = 0; c < kvLora; c++) {
                s.qLat[lo + c] = l.kB.dot(kbBase + (long) c * nopeDim, s.qAll, qo, nopeDim);
            }
            int ao = hh * maxSeqLen;
            float max = Float.NEGATIVE_INFINITY;
            for (int t = 0; t < n; t++) {
                float sc = (ops.dot(s.qLat, lo, lat, t * kvLora, kvLora)
                          + ops.dot(s.qAll, qo + nopeDim, kp, t * ropeDim, ropeDim)) * kqScale;
                s.att[ao + t] = sc;
                if (sc > max) max = sc;
            }
            float sum = 0f;
            for (int t = 0; t < n; t++) { float e = (float) Math.exp(s.att[ao + t] - max); s.att[ao + t] = e; sum += e; }
            Arrays.fill(s.ctx, lo, lo + kvLora, 0f);
            for (int t = 0; t < n; t++) ops.saxpy(s.att[ao + t] / sum, lat, t * kvLora, s.ctx, lo, kvLora);
            // expand with v_b, gate per head
            float gh = sigmoid(s.headGate[hh]);
            long vbBase = (long) hh * vHeadDim * kvLora;
            int oo = hh * vHeadDim;
            for (int e = 0; e < vHeadDim; e++) {
                s.attnOut[oo + e] = l.vB.dot(vbBase + (long) e * kvLora, s.ctx, lo, kvLora) * gh;
            }
        });
        Arrays.fill(s.out, 0f);
        l.wo.matmulParallel(s.attnOut, s.out, dim, nHead * vHeadDim);
    }

    private void ropeNorm(float[] v, int off, int tableOff, int half) {
        for (int i = 0; i < half; i++) {
            float c = ropeCos[tableOff + i], sn = ropeSin[tableOff + i];
            float x0 = v[off + 2 * i], x1 = v[off + 2 * i + 1];
            v[off + 2 * i] = x0 * c - x1 * sn;
            v[off + 2 * i + 1] = x0 * sn + x1 * c;
        }
    }

    // ==================== FFN ====================

    private void dense(State s, Layer l) {
        Arrays.fill(s.fg, 0, ffnDense, 0f);
        Arrays.fill(s.fu, 0, ffnDense, 0f);
        FloatTensor.fusedGateUpMatmulParallel(l.ffnGate, l.ffnUp, s.h, s.fg, s.fu, ffnDense, dim);
        for (int i = 0; i < ffnDense; i++) s.fg[i] = silu(s.fg[i]) * s.fu[i];
        Arrays.fill(s.out, 0f);
        l.ffnDown.matmulParallel(s.fg, s.out, dim, ffnDense);
    }

    private void moe(State s, Layer l, int li) {
        // router: sigmoid probs, biased scores for selection only
        Arrays.fill(s.router, 0f);
        l.router.matmul(s.h, s.router, nExpert, dim);
        boolean sigmoidGate = config.expertGatingFunc() == 2;
        if (sigmoidGate) {
            for (int e = 0; e < nExpert; e++) s.router[e] = sigmoid(s.router[e]);
        } else {
            VectorOpsFactory.get().softmax(s.router, 0, nExpert);
        }
        for (int e = 0; e < nExpert; e++) s.sel[e] = s.router[e] + (l.probsBias != null ? l.probsBias[e] : 0f);

        // group-limited routing: keep the groups with the best top-2 score sums
        if (nGroups > 1 && nGroupsUsed < nGroups) {
            int per = nExpert / nGroups;
            float[] gs = new float[nGroups];
            for (int gi = 0; gi < nGroups; gi++) {
                float a = Float.NEGATIVE_INFINITY, b = Float.NEGATIVE_INFINITY;
                for (int e = gi * per; e < (gi + 1) * per; e++) {
                    float v = s.sel[e];
                    if (v > a) { b = a; a = v; } else if (v > b) b = v;
                }
                gs[gi] = a + b;
            }
            boolean[] keep = new boolean[nGroups];
            for (int k = 0; k < nGroupsUsed; k++) {
                int best = -1;
                for (int gi = 0; gi < nGroups; gi++) if (!keep[gi] && (best < 0 || gs[gi] > gs[best])) best = gi;
                keep[best] = true;
            }
            for (int gi = 0; gi < nGroups; gi++) {
                if (!keep[gi]) Arrays.fill(s.sel, gi * per, (gi + 1) * per, Float.NEGATIVE_INFINITY);
            }
        }
        int k = MoERouting.effectiveTopK(nUsed);
        for (int j = 0; j < k; j++) {
            int best = -1;
            for (int e = 0; e < nExpert; e++) {
                boolean taken = false;
                for (int q = 0; q < j; q++) if (s.ids[q] == e) { taken = true; break; }
                if (!taken && (best < 0 || s.sel[e] > s.sel[best])) best = e;
            }
            s.ids[j] = best;
            s.w[j] = s.router[best];
        }
        if (weightsNorm) {
            float sum = 0f;
            for (int j = 0; j < k; j++) sum += s.w[j];
            sum = Math.max(sum, 6.103515625e-5f);
            for (int j = 0; j < k; j++) s.w[j] /= sum;
        }
        if (weightsScale != 0f && weightsScale != 1f) for (int j = 0; j < k; j++) s.w[j] *= weightsScale;

        final int kk = k;
        final float[] xin = s.h;
        rows(kk, expFfn, (j, r0, r1) -> {
            float[] g = s.eg[j], u = s.eu[j];
            Arrays.fill(g, r0, r1, 0f);
            Arrays.fill(u, r0, r1, 0f);
            expertViews.get(li, 0, s.ids[j]).matmulRows(xin, g, r0, r1, dim);
            expertViews.get(li, 1, s.ids[j]).matmulRows(xin, u, r0, r1, dim);
            for (int r = r0; r < r1; r++) g[r] = silu(g[r]) * u[r];
        });
        rows(kk, dim, (j, r0, r1) -> {
            float[] o = s.eo[j];
            Arrays.fill(o, r0, r1, 0f);
            expertViews.get(li, 2, s.ids[j]).matmulRows(s.eg[j], o, r0, r1, expFfn);
        });
        Arrays.fill(s.out, 0f);
        for (int j = 0; j < kk; j++) VectorOpsFactory.get().saxpy(s.w[j], s.eo[j], 0, s.out, 0, dim);

        // shared expert
        Arrays.fill(s.fg, 0, shFfn, 0f);
        Arrays.fill(s.fu, 0, shFfn, 0f);
        FloatTensor.fusedGateUpMatmulParallel(l.gateSh, l.upSh, s.h, s.fg, s.fu, shFfn, dim);
        for (int i = 0; i < shFfn; i++) s.fg[i] = silu(s.fg[i]) * s.fu[i];
        float[] shOut = s.eo[0];   // expert outputs already accumulated into s.out
        Arrays.fill(shOut, 0f);
        l.downSh.matmulParallel(s.fg, shOut, dim, shFfn);
        VectorOpsFactory.get().accumulate(s.out, shOut, dim);
    }

    private interface RowTask { void run(int expert, int r0, int r1); }

    /** k × rowsPerExpert rows split across the matmul pool, never spanning two experts. */
    private static void rows(int k, int rowsPerExpert, RowTask body) {
        MatmulPool pool = MatmulPool.enabled() ? MatmulPool.get() : null;
        if (pool == null) {
            MatmulPool.forEach(k, j -> body.run(j, 0, rowsPerExpert));
            return;
        }
        pool.parallelFor(k * rowsPerExpert, 16, (from, to) -> {
            int p = from;
            while (p < to) {
                int j = p / rowsPerExpert, r0 = p - j * rowsPerExpert;
                int r1 = Math.min(rowsPerExpert, r0 + (to - p));
                body.run(j, r0, r1);
                p += r1 - r0;
            }
        });
    }

    // ==================== Helpers ====================

    private static float sigmoid(float x) { return 1f / (1f + (float) Math.exp(-x)); }

    private static float silu(float x) { return x / (1f + (float) Math.exp(-x)); }

    /** llama.cpp build_gdn_l2_norm: x / sqrt(sum x^2 + eps). */
    private void l2norm(float[] v, int off, int n) {
        float ss = 0f;
        for (int i = 0; i < n; i++) ss += v[off + i] * v[off + i];
        float inv = 1f / (float) Math.sqrt(ss + eps);
        for (int i = 0; i < n; i++) v[off + i] *= inv;
    }

    private static void zero(float[]... arrays) {
        for (float[] a : arrays) Arrays.fill(a, 0f);
    }

    private static FloatTensor t(GGUFFile g, String name) {
        FloatTensor t = ModelLoader.tryLoadTensor(g, name);
        if (t == null) throw new IllegalStateException("BailingMoE3: tensor not found: " + name);
        return t;
    }

    private static float[] f(GGUFFile g, String name) {
        float[] v = fOpt(g, name);
        if (v == null) throw new IllegalStateException("BailingMoE3: tensor not found: " + name);
        return v;
    }

    private static float[] fOpt(GGUFFile g, String name) {
        FloatTensor t = ModelLoader.tryLoadTensor(g, name);
        if (t == null) return null;
        int n = (int) g.findTensor(name).elementCount();
        float[] v = new float[n];
        for (int i = 0; i < n; i++) v[i] = t.getFloat(i);
        return v;
    }

    public ModelConfig getConfig() { return config; }
}
