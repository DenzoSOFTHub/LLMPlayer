package it.denzosoft.llmplayer.tts;

import it.denzosoft.llmplayer.gguf.GGUFFile;
import it.denzosoft.llmplayer.gguf.GGUFMetadata;
import it.denzosoft.llmplayer.gguf.GGUFParser;
import it.denzosoft.llmplayer.gguf.GGUFTensorInfo;
import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.MatmulPool;
import it.denzosoft.llmplayer.tensor.TensorFactory;
import it.denzosoft.llmplayer.tensor.VectorOps;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

/**
 * The audio half of Qwen3-TTS, loaded from the llama.cpp mmproj ({@code clip.gen.audio.projector_type
 * = qwen3tts_gen}), following {@code tools/mtmd/models/qwen3tts-gen.cpp}:
 * <ul>
 *   <li><b>Code predictor</b>: from the talker's hidden state and its sampled codebook-0 code, a
 *       5-layer Qwen3-style transformer samples the 15 acoustic codebooks of the frame, one per
 *       step, and returns the embedding sum that is fed back to the talker.</li>
 *   <li><b>Code2wav</b>: 16-codebook RVQ codes → 512-d latent → causal conv → 8-layer transformer
 *       with a 72-frame attention window → two ×2 ConvNeXt upsamplers → DAC decoder (Snake,
 *       transposed convs with strides 8/5/4/3, dilated residual units) → 24 kHz PCM, 1920 samples
 *       per frame. It runs in 72-frame chunks and carries the causal state between chunks, so the
 *       result equals decoding the whole sequence at once.</li>
 * </ul>
 * The speaker encoder (voice cloning from a reference recording) is not implemented.
 */
public final class Qwen3TtsCodec {

    public static final int SAMPLE_RATE = 24000;
    static final int WINDOW = 72;                  // code2wav attention window = frames per chunk

    // ---- code predictor ----
    private final int cpDim, cpHeads, cpKvHeads, cpHeadDim, cpFfn, cpLayers, nAcoustic, cpVocab;
    private final float cpEps;
    private final float cpRopeTheta = 1_000_000f;
    private final FloatTensor projInW;          // [talkerDim -> cpDim]
    private final float[] projInB;
    private final FloatTensor codeEmbd;         // [15][2048 codes][talkerDim]
    private final FloatTensor outEmbd;          // [3072 codes][talkerDim] (codebook 0)
    private final FloatTensor codeHead;         // [15][2048][cpDim]
    private final float[] cpNorm;
    private final CpLayer[] cp;
    private final int talkerDim;

    private static final class CpLayer {
        float[] ln1, ln2, qNorm, kNorm;
        FloatTensor q, k, v, o, gate, up, down;
    }

    // ---- code2wav ----
    private final float[] cbFirst, cbRest;      // [2048][256], [15][2048][256]
    private final FloatTensor qFirstOut, qRestOut; // 256 -> 512
    private final Conv preConv;                 // 512 -> 1024, k3
    private final FloatTensor tfmIn, tfmOut;    // 1024 -> 512, 512 -> 1024
    private final float[] tfmInB, tfmOutB, tfmNorm;
    private final TfmLayer[] tfm;
    private final int tfmHeads = 16, tfmHeadDim, tfmDim = 512;
    private final float tfmEps = 1e-5f, tfmRopeTheta = 10000f;
    private final Upsample[] up;
    private final Conv dacEntry;
    private final DacBlock[] dac;
    private final float[] postAlpha, postBeta;
    private final Conv postConv;

    private static final class TfmLayer {
        float[] ln1, ln2, ls1, ls2;
        FloatTensor q, k, v, o, gate, upW, down;
    }

    private static final class Upsample {
        ConvT conv;
        Conv dw;               // depthwise
        float[] normW, normB, gamma, pw1B, pw2B;
        FloatTensor pw1, pw2;
    }

    private static final class DacBlock {
        float[] alpha, beta;
        ConvT conv;
        ResUnit[] res;
    }

    private static final class ResUnit {
        float[] a1, b1, a2, b2;
        Conv c1, c2;
        int dilation;
    }

    /** Causal conv1d: weight [K][IC][OC] in GGUF order, flat k + K*(ic + IC*oc). Depthwise when ic == 1. */
    private static final class Conv {
        final int k, ic, oc;
        final boolean depthwise;
        final float[] w, b;
        Conv(int k, int ic, int oc, boolean depthwise, float[] w, float[] b) {
            this.k = k; this.ic = ic; this.oc = oc; this.depthwise = depthwise; this.w = w; this.b = b;
        }
    }

    /** Causal transposed conv1d: weight [K][OC][IC], flat k + K*(oc + OC*ic). */
    private static final class ConvT {
        final int k, oc, ic, stride;
        final float[] w, b;
        ConvT(int k, int oc, int ic, int stride, float[] w, float[] b) {
            this.k = k; this.oc = oc; this.ic = ic; this.stride = stride; this.w = w; this.b = b;
        }
    }

    private Qwen3TtsCodec(GGUFFile g) {
        GGUFMetadata md = g.getMetadata();
        String proj = md.getString("clip.gen.audio.projector_type", "");
        if (!"qwen3tts_gen".equals(proj)) {
            throw new IllegalArgumentException("Not a Qwen3-TTS mmproj (clip.gen.audio.projector_type = '" + proj + "')");
        }
        cpDim = md.getInt("clip.gen.audio.embedding_length");
        cpHeads = md.getInt("clip.gen.audio.attention.head_count");
        cpKvHeads = md.getInt("clip.gen.audio.attention.head_count_kv", cpHeads);
        cpFfn = md.getInt("clip.gen.audio.feed_forward_length");
        cpLayers = md.getInt("clip.gen.audio.block_count");
        cpEps = md.getFloat("clip.gen.audio.attention.layer_norm_epsilon", 1e-6f);
        talkerDim = md.getInt("clip.gen.audio.projection_dim");

        projInW = tensorOpt(g, "a.gen.code.proj_in.weight");
        projInB = f32opt(g, "a.gen.code.proj_in.bias");
        codeEmbd = tensor(g, "a.gen.code.embd.weight");
        outEmbd = tensor(g, "a.gen.code.out_embd.weight");
        codeHead = tensor(g, "a.gen.code.head.weight");
        cpNorm = f32(g, "a.gen.code.output_norm.weight");
        long[] headDims = g.findTensor("a.gen.code.head.weight").dimensions();
        cpVocab = (int) headDims[1];
        nAcoustic = (int) headDims[2];
        cp = new CpLayer[cpLayers];
        for (int i = 0; i < cpLayers; i++) {
            String p = "a.gen.code.blk." + i + ".";
            CpLayer l = new CpLayer();
            l.ln1 = f32(g, p + "ln1.weight");
            l.ln2 = f32(g, p + "ln2.weight");
            l.qNorm = f32(g, p + "attn_q_norm.weight");
            l.kNorm = f32(g, p + "attn_k_norm.weight");
            l.q = tensor(g, p + "attn_q.weight");
            l.k = tensor(g, p + "attn_k.weight");
            l.v = tensor(g, p + "attn_v.weight");
            l.o = tensor(g, p + "attn_out.weight");
            l.gate = tensor(g, p + "ffn_gate.weight");
            l.up = tensor(g, p + "ffn_up.weight");
            l.down = tensor(g, p + "ffn_down.weight");
            cp[i] = l;
        }
        cpHeadDim = (int) g.findTensor("a.gen.code.blk.0.attn_q.weight").dimensions()[1] / cpHeads;

        cbFirst = f32(g, "a.gen.wav.quant.first.codebook.weight");
        cbRest = f32(g, "a.gen.wav.quant.rest.codebook.weight");
        qFirstOut = tensor(g, "a.gen.wav.quant.first.out_proj.weight");
        qRestOut = tensor(g, "a.gen.wav.quant.rest.out_proj.weight");
        preConv = conv(g, "a.gen.wav.pre_conv", false);
        tfmIn = tensor(g, "a.gen.wav.tfm.in_proj.weight");
        tfmInB = f32(g, "a.gen.wav.tfm.in_proj.bias");
        tfmOut = tensor(g, "a.gen.wav.tfm.out_proj.weight");
        tfmOutB = f32(g, "a.gen.wav.tfm.out_proj.bias");
        tfmNorm = f32(g, "a.gen.wav.tfm.output_norm.weight");
        int nTfm = 0;
        while (g.findTensor("a.gen.wav.tfm.blk." + nTfm + ".attn_q.weight") != null) nTfm++;
        tfm = new TfmLayer[nTfm];
        for (int i = 0; i < nTfm; i++) {
            String p = "a.gen.wav.tfm.blk." + i + ".";
            TfmLayer l = new TfmLayer();
            l.ln1 = f32(g, p + "ln1.weight");
            l.ln2 = f32(g, p + "ln2.weight");
            l.ls1 = f32opt(g, p + "ls1.weight");
            l.ls2 = f32opt(g, p + "ls2.weight");
            l.q = tensor(g, p + "attn_q.weight");
            l.k = tensor(g, p + "attn_k.weight");
            l.v = tensor(g, p + "attn_v.weight");
            l.o = tensor(g, p + "attn_out.weight");
            l.gate = tensor(g, p + "ffn_gate.weight");
            l.upW = tensor(g, p + "ffn_up.weight");
            l.down = tensor(g, p + "ffn_down.weight");
            tfm[i] = l;
        }
        tfmHeadDim = (int) g.findTensor("a.gen.wav.tfm.blk.0.attn_q.weight").dimensions()[1] / tfmHeads;

        int nUp = 0;
        while (g.findTensor("a.gen.wav.up.blk." + nUp + ".conv.weight") != null) nUp++;
        up = new Upsample[nUp];
        for (int i = 0; i < nUp; i++) {
            String p = "a.gen.wav.up.blk." + i + ".";
            Upsample u = new Upsample();
            u.conv = convT(g, p + "conv", 2);
            u.dw = conv(g, p + "dwconv", true);
            u.normW = f32(g, p + "norm.weight");
            u.normB = f32(g, p + "norm.bias");
            u.pw1 = tensor(g, p + "pw1.weight");
            u.pw1B = f32(g, p + "pw1.bias");
            u.pw2 = tensor(g, p + "pw2.weight");
            u.pw2B = f32(g, p + "pw2.bias");
            u.gamma = f32(g, "a.gen.wav.up.blk." + i + ".gamma");
            up[i] = u;
        }
        dacEntry = conv(g, "a.gen.wav.dac.entry", false);
        int nDac = 0;
        while (g.findTensor("a.gen.wav.dac.blk." + nDac + ".conv.weight") != null) nDac++;
        dac = new DacBlock[nDac];
        int[] dil = {1, 3, 9};
        for (int i = 0; i < nDac; i++) {
            String p = "a.gen.wav.dac.blk." + i + ".";
            DacBlock d = new DacBlock();
            d.alpha = f32(g, p + "snake.alpha");
            d.beta = f32(g, p + "snake.beta");
            int k = (int) g.findTensor(p + "conv.weight").dimensions()[0];
            d.conv = convT(g, p + "conv", k / 2);
            d.res = new ResUnit[3];
            for (int r = 0; r < 3; r++) {
                String q = p + "res." + r + ".";
                ResUnit u = new ResUnit();
                u.a1 = f32(g, q + "act1.alpha");
                u.b1 = f32(g, q + "act1.beta");
                u.a2 = f32(g, q + "act2.alpha");
                u.b2 = f32(g, q + "act2.beta");
                u.c1 = conv(g, q + "conv1", false);
                u.c2 = conv(g, q + "conv2", false);
                u.dilation = dil[r];
                d.res[r] = u;
            }
            dac[i] = d;
        }
        postAlpha = f32(g, "a.gen.wav.dac.post_snake.alpha");
        postBeta = f32(g, "a.gen.wav.dac.post_snake.beta");
        postConv = conv(g, "a.gen.wav.dac.post_conv", false);
    }

    public static Qwen3TtsCodec load(Path mmproj) throws IOException {
        // Weights are copied (F32 conv kernels) or kept mmapped (matmul tensors), so the file stays open
        GGUFFile g = GGUFParser.parse(mmproj, true);
        Qwen3TtsCodec c = new Qwen3TtsCodec(g);
        System.out.println("  TTS codec: code predictor " + c.cpLayers + " layers (dim " + c.cpDim + ", "
            + (c.nAcoustic + 1) + " codebooks), code2wav " + c.tfm.length + "-layer transformer + "
            + c.up.length + " upsamplers + " + c.dac.length + " DAC blocks, " + SAMPLE_RATE + " Hz");
        return c;
    }

    public int codebooks() { return nAcoustic + 1; }
    public int talkerDim() { return talkerDim; }

    // =====================================================================================
    // Code predictor
    // =====================================================================================

    /** Per-frame scratch for the code predictor (one per synthesis). */
    public final class PredictorState {
        final float[][][] kCache = new float[cpLayers][nAcoustic + 1][cpKvHeads * cpHeadDim];
        final float[][][] vCache = new float[cpLayers][nAcoustic + 1][cpKvHeads * cpHeadDim];
        final float[] x = new float[cpDim], h = new float[cpDim], q = new float[cpHeads * cpHeadDim];
        final float[] k = new float[cpKvHeads * cpHeadDim], v = new float[cpKvHeads * cpHeadDim];
        final float[] att = new float[cpHeads * cpHeadDim], o = new float[cpDim];
        final float[] g = new float[cpFfn], u = new float[cpFfn], logits = new float[cpVocab];
        final float[] emb = new float[talkerDim];
        final float[] scores = new float[nAcoustic + 1];
    }

    public PredictorState newPredictorState() { return new PredictorState(); }

    /**
     * Samples the acoustic codes of one frame. {@code hidden} is the talker's output-normed hidden
     * state, {@code code0} its sampled codebook-0 code. Fills {@code codesOut[0..16)} and writes the
     * sum of the 16 code embeddings (talker width) into {@code nextEmbd}.
     */
    public void predictFrame(PredictorState s, float[] hidden, int code0, int topK, float topP, float temp,
                             Random rng, int[] codesOut, float[] nextEmbd) {
        codesOut[0] = code0;
        // position 0: hidden-state bridge (seeds the cache)
        projectIn(hidden, s.x);
        cpForward(s, 0);
        // position 1: codebook-0 embedding, sample codebook 1 with head 0
        row(outEmbd, code0, talkerDim, s.emb);
        System.arraycopy(s.emb, 0, nextEmbd, 0, talkerDim);
        projectIn(s.emb, s.x);
        cpForward(s, 1);
        codesOut[1] = sampleHead(s, 0, topK, topP, temp, rng);
        // positions 2..15
        for (int step = 1; step < nAcoustic; step++) {
            rowOf3d(codeEmbd, step - 1, codesOut[step], cpVocabEmbd(), talkerDim, s.emb);
            addTo(nextEmbd, s.emb);
            projectIn(s.emb, s.x);
            cpForward(s, step + 1);
            codesOut[step + 1] = sampleHead(s, step, topK, topP, temp, rng);
        }
        // the last code is only embedded for the talker
        rowOf3d(codeEmbd, nAcoustic - 1, codesOut[nAcoustic], cpVocabEmbd(), talkerDim, s.emb);
        addTo(nextEmbd, s.emb);
    }

    private int cpVocabEmbd() { return cpVocab; }

    private void projectIn(float[] in, float[] out) {
        if (projInW == null) {
            System.arraycopy(in, 0, out, 0, cpDim);
            return;
        }
        Arrays.fill(out, 0f);
        projInW.matmulParallel(in, out, cpDim, talkerDim);
        if (projInB != null) for (int i = 0; i < cpDim; i++) out[i] += projInB[i];
    }

    /** All predictor layers for the input in s.x at position pos (writes K/V row pos). */
    private void cpForward(PredictorState s, int pos) {
        int qDim = cpHeads * cpHeadDim, kvDim = cpKvHeads * cpHeadDim, group = cpHeads / cpKvHeads;
        float scale = (float) (1.0 / Math.sqrt(cpHeadDim));
        VectorOps ops = VectorOpsFactory.get();
        for (int li = 0; li < cpLayers; li++) {
            CpLayer l = cp[li];
            ops.rmsnorm(s.h, s.x, l.ln1, cpDim, cpEps);
            Arrays.fill(s.q, 0f);
            Arrays.fill(s.k, 0f);
            Arrays.fill(s.v, 0f);
            l.q.matmulParallel(s.h, s.q, qDim, cpDim);
            l.k.matmulParallel(s.h, s.k, kvDim, cpDim);
            l.v.matmulParallel(s.h, s.v, kvDim, cpDim);
            for (int hh = 0; hh < cpHeads; hh++) headNorm(s.q, hh * cpHeadDim, l.qNorm, cpHeadDim, cpEps);
            for (int hh = 0; hh < cpKvHeads; hh++) headNorm(s.k, hh * cpHeadDim, l.kNorm, cpHeadDim, cpEps);
            ropeNeox(s.q, cpHeads, cpHeadDim, pos, cpRopeTheta);
            ropeNeox(s.k, cpKvHeads, cpHeadDim, pos, cpRopeTheta);
            System.arraycopy(s.k, 0, s.kCache[li][pos], 0, kvDim);
            System.arraycopy(s.v, 0, s.vCache[li][pos], 0, kvDim);
            Arrays.fill(s.att, 0f);
            for (int hh = 0; hh < cpHeads; hh++) {
                int kvOff = (hh / group) * cpHeadDim, qOff = hh * cpHeadDim;
                float max = Float.NEGATIVE_INFINITY;
                for (int t = 0; t <= pos; t++) {
                    float sc = ops.dot(s.q, qOff, s.kCache[li][t], kvOff, cpHeadDim) * scale;
                    s.scores[t] = sc;
                    if (sc > max) max = sc;
                }
                float sum = 0f;
                for (int t = 0; t <= pos; t++) { s.scores[t] = (float) Math.exp(s.scores[t] - max); sum += s.scores[t]; }
                for (int t = 0; t <= pos; t++) ops.saxpy(s.scores[t] / sum, s.vCache[li][t], kvOff, s.att, qOff, cpHeadDim);
            }
            Arrays.fill(s.o, 0f);
            l.o.matmulParallel(s.att, s.o, cpDim, qDim);
            for (int i = 0; i < cpDim; i++) s.x[i] += s.o[i];
            ops.rmsnorm(s.h, s.x, l.ln2, cpDim, cpEps);
            Arrays.fill(s.g, 0f);
            Arrays.fill(s.u, 0f);
            FloatTensor.fusedGateUpMatmulParallel(l.gate, l.up, s.h, s.g, s.u, cpFfn, cpDim);
            for (int i = 0; i < cpFfn; i++) {
                float gv = s.g[i];
                s.g[i] = gv / (1f + (float) Math.exp(-gv)) * s.u[i];
            }
            Arrays.fill(s.o, 0f);
            l.down.matmulParallel(s.g, s.o, cpDim, cpFfn);
            for (int i = 0; i < cpDim; i++) s.x[i] += s.o[i];
        }
    }

    /** Final norm, head {@code headIdx} (slice of the 3D head tensor), sample a code. */
    private int sampleHead(PredictorState s, int headIdx, int topK, float topP, float temp, Random rng) {
        VectorOps ops = VectorOpsFactory.get();
        ops.rmsnorm(s.h, s.x, cpNorm, cpDim, cpEps);
        long base = (long) headIdx * cpVocab * cpDim;
        for (int r = 0; r < cpVocab; r++) s.logits[r] = codeHead.dot(base + (long) r * cpDim, s.h, 0, cpDim);
        return Sampling.sample(s.logits, cpVocab, topK, topP, temp, rng);
    }

    // =====================================================================================
    // Code2wav
    // =====================================================================================

    /** Streaming decoder state carried between 72-frame chunks. */
    public final class DecoderState {
        int tfmPos;                                  // frames seen by the transformer
        final float[][][] kHist = new float[tfm.length][][];  // per layer, last WINDOW-1 frames
        final float[][][] vHist = new float[tfm.length][][];
        final java.util.Map<Object, float[][]> convLeft = new java.util.IdentityHashMap<>(); // conv -> [C][pad]
        final java.util.Map<Object, float[][]> convTail = new java.util.IdentityHashMap<>(); // convT -> [OC][K-stride]
    }

    public DecoderState newDecoderState() { return new DecoderState(); }

    /** Decode up to {@link #WINDOW} frames of codes ({@code [n][16]}) into PCM samples. */
    public float[] decodeChunk(DecoderState st, int[][] codes, int n) {
        // 1. RVQ decode -> [n][512]
        float[][] sem = new float[n][256], ac = new float[n][256];
        for (int t = 0; t < n; t++) {
            System.arraycopy(cbFirst, codes[t][0] * 256, sem[t], 0, 256);
            for (int gi = 1; gi <= nAcoustic; gi++) {
                int off = ((gi - 1) * 2048 + codes[t][gi]) * 256;
                for (int i = 0; i < 256; i++) ac[t][i] += cbRest[off + i];
            }
        }
        float[][] hidden = linear(qFirstOut, sem, n, 512, 256, null);
        float[][] hidden2 = linear(qRestOut, ac, n, 512, 256, null);
        for (int t = 0; t < n; t++) for (int i = 0; i < 512; i++) hidden[t][i] += hidden2[t][i];

        // 2. pre_conv (causal, k=3) in channel-major layout
        float[][] x = conv1d(st, preConv, transpose(hidden, n, 512), n, 1);   // [1024][n]

        // 3. transformer over the frames
        float[][] cur = linear(tfmIn, transpose(x, 1024, n), n, tfmDim, 1024, tfmInB);
        cur = transformer(st, cur, n);
        float[][] y = new float[n][];
        for (int t = 0; t < n; t++) {
            float[] h = new float[tfmDim];
            VectorOpsFactory.get().rmsnorm(h, cur[t], tfmNorm, tfmDim, tfmEps);
            y[t] = h;
        }
        x = transpose(linear(tfmOut, y, n, 1024, tfmDim, tfmOutB), n, 1024);    // [1024][n]
        int len = n;

        // 4. upsample x2 (conv transpose k=2 s=2 + ConvNeXt)
        for (Upsample u : up) {
            x = convT(st, u.conv, x, len);
            len *= 2;
            x = convNext(st, u, x, len);
        }

        // 5. DAC decoder
        x = conv1d(st, dacEntry, x, len, 1);
        for (DacBlock d : dac) {
            snake(x, d.alpha, d.beta, len);
            x = convT(st, d.conv, x, len);
            len *= d.conv.stride;
            for (ResUnit r : d.res) {
                float[][] h = copy(x, len);
                snake(h, r.a1, r.b1, len);
                h = conv1d(st, r.c1, h, len, r.dilation);
                snake(h, r.a2, r.b2, len);
                h = conv1d(st, r.c2, h, len, 1);
                for (int c = 0; c < x.length; c++) VectorOpsFactory.get().saxpy(1f, h[c], 0, x[c], 0, len);
            }
        }
        snake(x, postAlpha, postBeta, len);
        float[][] out = conv1d(st, postConv, x, len, 1);
        float[] pcm = out[0];
        for (int i = 0; i < len; i++) pcm[i] = Math.max(-1f, Math.min(1f, pcm[i]));
        return pcm;
    }

    /** Pre-norm transformer layers over n frames with a WINDOW-frame causal band (state across chunks). */
    private float[][] transformer(DecoderState st, float[][] x, int n) {
        int kvDim = tfmHeads * tfmHeadDim;
        float scale = (float) (1.0 / Math.sqrt(tfmHeadDim));
        VectorOps ops = VectorOpsFactory.get();
        int base = st.tfmPos;
        for (int li = 0; li < tfm.length; li++) {
            TfmLayer l = tfm[li];
            float[][] h = new float[n][tfmDim];
            for (int t = 0; t < n; t++) ops.rmsnorm(h[t], x[t], l.ln1, tfmDim, tfmEps);
            float[][] q = linear(l.q, h, n, kvDim, tfmDim, null);
            float[][] k = linear(l.k, h, n, kvDim, tfmDim, null);
            float[][] v = linear(l.v, h, n, kvDim, tfmDim, null);
            for (int t = 0; t < n; t++) {
                ropeNeox(q[t], tfmHeads, tfmHeadDim, base + t, tfmRopeTheta);
                ropeNeox(k[t], tfmHeads, tfmHeadDim, base + t, tfmRopeTheta);
            }
            float[][] prevK = st.kHist[li] != null ? st.kHist[li] : new float[0][];
            float[][] prevV = st.vHist[li] != null ? st.vHist[li] : new float[0][];
            int np = prevK.length;
            float[][] allK = new float[np + n][], allV = new float[np + n][];
            System.arraycopy(prevK, 0, allK, 0, np);
            System.arraycopy(prevV, 0, allV, 0, np);
            System.arraycopy(k, 0, allK, np, n);
            System.arraycopy(v, 0, allV, np, n);
            float[][] att = new float[n][kvDim];
            final int npF = np;
            MatmulPool.forEach(n * tfmHeads, task -> {
                int t = task / tfmHeads, hh = task % tfmHeads, off = hh * tfmHeadDim;
                int qi = npF + t;                       // query index in allK
                int from = Math.max(0, qi - WINDOW + 1);
                float[] sc = new float[qi - from + 1];
                float max = Float.NEGATIVE_INFINITY;
                for (int j = from; j <= qi; j++) {
                    float s = ops.dot(q[t], off, allK[j], off, tfmHeadDim) * scale;
                    sc[j - from] = s;
                    if (s > max) max = s;
                }
                float sum = 0f;
                for (int j = 0; j < sc.length; j++) { sc[j] = (float) Math.exp(sc[j] - max); sum += sc[j]; }
                for (int j = from; j <= qi; j++) ops.saxpy(sc[j - from] / sum, allV[j], off, att[t], off, tfmHeadDim);
            });
            float[][] o = linear(l.o, att, n, tfmDim, kvDim, null);
            for (int t = 0; t < n; t++) {
                for (int i = 0; i < tfmDim; i++) x[t][i] += (l.ls1 != null ? l.ls1[i] : 1f) * o[t][i];
            }
            for (int t = 0; t < n; t++) ops.rmsnorm(h[t], x[t], l.ln2, tfmDim, tfmEps);
            float[][] g = linear(l.gate, h, n, 1024, tfmDim, null);
            float[][] u = linear(l.upW, h, n, 1024, tfmDim, null);
            for (int t = 0; t < n; t++) {
                for (int i = 0; i < 1024; i++) {
                    float gv = g[t][i];
                    g[t][i] = gv / (1f + (float) Math.exp(-gv)) * u[t][i];
                }
            }
            float[][] d = linear(l.down, g, n, tfmDim, 1024, null);
            for (int t = 0; t < n; t++) {
                for (int i = 0; i < tfmDim; i++) x[t][i] += (l.ls2 != null ? l.ls2[i] : 1f) * d[t][i];
            }
            // keep the last WINDOW-1 frames of K/V for the next chunk
            int keep = Math.min(WINDOW - 1, np + n);
            st.kHist[li] = Arrays.copyOfRange(allK, np + n - keep, np + n);
            st.vHist[li] = Arrays.copyOfRange(allV, np + n - keep, np + n);
        }
        st.tfmPos += n;
        return x;
    }

    /** ConvNeXt block (channel-major): causal depthwise conv, LayerNorm, pw1, GELU, pw2, gamma, residual. */
    private float[][] convNext(DecoderState st, Upsample u, float[][] x, int len) {
        int c = x.length;
        float[][] h = conv1d(st, u.dw, x, len, 1);         // [C][len]
        float[][] ht = transpose(h, c, len);               // [len][C]
        for (int t = 0; t < len; t++) layerNorm(ht[t], u.normW, u.normB, c, 1e-6f);
        float[][] a = linear(u.pw1, ht, len, u.pw1B.length, c, u.pw1B);
        MatmulPool.forEach(len, t -> gelu(a[t]));
        float[][] b = linear(u.pw2, a, len, c, u.pw1B.length, u.pw2B);
        for (int t = 0; t < len; t++) {
            for (int i = 0; i < c; i++) x[i][t] += u.gamma[i] * b[t][i];
        }
        return x;
    }

    // ---- stateful causal convolutions (channel-major [C][T]) ----

    private float[][] conv1d(DecoderState st, Conv cv, float[][] x, int len, int dilation) {
        int pad = (cv.k - 1) * dilation;
        int inCh = cv.depthwise ? cv.oc : cv.ic;
        float[][] left = st.convLeft.get(cv);
        if (left == null) left = new float[inCh][pad];
        // input with left context: [C][pad + len]
        float[][] full = new float[inCh][];
        for (int c = 0; c < inCh; c++) {
            float[] f = new float[pad + len];
            System.arraycopy(left[c], 0, f, 0, pad);
            System.arraycopy(x[c], 0, f, pad, len);
            full[c] = f;
        }
        if (pad > 0) {
            float[][] newLeft = new float[inCh][pad];
            for (int c = 0; c < inCh; c++) System.arraycopy(full[c], len, newLeft[c], 0, pad);
            st.convLeft.put(cv, newLeft);
        }
        float[][] y = new float[cv.oc][len];
        VectorOps ops = VectorOpsFactory.get();
        MatmulPool.forEach(cv.oc, oc -> {
            float[] acc = y[oc];
            if (cv.b != null) Arrays.fill(acc, cv.b[oc]);
            if (cv.depthwise) {
                for (int kk = 0; kk < cv.k; kk++) {
                    ops.saxpy(cv.w[kk + cv.k * oc], full[oc], kk * dilation, acc, 0, len);
                }
            } else {
                for (int ic = 0; ic < cv.ic; ic++) {
                    int wBase = cv.k * (ic + cv.ic * oc);
                    float[] src = full[ic];
                    for (int kk = 0; kk < cv.k; kk++) {
                        float w = cv.w[wBase + kk];
                        if (w != 0f) ops.saxpy(w, src, kk * dilation, acc, 0, len);
                    }
                }
            }
        });
        return y;
    }

    private float[][] convT(DecoderState st, ConvT ct, float[][] x, int len) {
        int stride = ct.stride, trim = ct.k - stride, outLen = len * stride;
        float[][] y = new float[ct.oc][outLen + trim];
        VectorOps ops = VectorOpsFactory.get();
        MatmulPool.forEach(ct.oc, oc -> {
            float[] tmp = new float[len];
            float[] yo = y[oc];
            for (int kk = 0; kk < ct.k; kk++) {
                Arrays.fill(tmp, 0f);
                for (int ic = 0; ic < ct.ic; ic++) {
                    float w = ct.w[kk + ct.k * (oc + ct.oc * ic)];
                    if (w != 0f) ops.saxpy(w, x[ic], 0, tmp, 0, len);
                }
                for (int t = 0; t < len; t++) yo[t * stride + kk] += tmp[t];
            }
        });
        float[][] out = new float[ct.oc][];
        float[][] tail = st.convTail.get(ct);
        float[][] newTail = trim > 0 ? new float[ct.oc][trim] : null;
        for (int oc = 0; oc < ct.oc; oc++) {
            float[] yo = y[oc];
            if (trim > 0) {
                if (tail != null) for (int i = 0; i < trim; i++) yo[i] += tail[oc][i];
                System.arraycopy(yo, outLen, newTail[oc], 0, trim);
            }
            float[] o = Arrays.copyOf(yo, outLen);
            if (ct.b != null) for (int i = 0; i < outLen; i++) o[i] += ct.b[oc];
            out[oc] = o;
        }
        if (trim > 0) st.convTail.put(ct, newTail);
        return out;
    }

    /** SnakeBeta in place: x + sin(alpha * x)^2 * beta (alpha / 1/beta folded at conversion). */
    private static void snake(float[][] x, float[] alpha, float[] beta, int len) {
        MatmulPool.forEach(x.length, c -> {
            float a = alpha[c], b = beta[c];
            float[] v = x[c];
            for (int i = 0; i < len; i++) {
                float s = (float) Math.sin(a * v[i]);
                v[i] += s * s * b;
            }
        });
    }

    // =====================================================================================
    // Helpers
    // =====================================================================================

    /** out[t] = W · in[t] (+ b); W is a [rows][cols] tensor. */
    private static float[][] linear(FloatTensor w, float[][] in, int n, int rows, int cols, float[] b) {
        float[][] out = new float[n][rows];
        FloatTensor.matmulBatchParallel(w, in, out, n, rows, cols);
        if (b != null) for (int t = 0; t < n; t++) for (int i = 0; i < rows; i++) out[t][i] += b[i];
        return out;
    }

    private static float[][] transpose(float[][] a, int rows, int cols) {
        float[][] t = new float[cols][rows];
        for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) t[c][r] = a[r][c];
        return t;
    }

    private static float[][] copy(float[][] a, int len) {
        float[][] c = new float[a.length][];
        for (int i = 0; i < a.length; i++) c[i] = Arrays.copyOf(a[i], len);
        return c;
    }

    private static void addTo(float[] acc, float[] v) {
        for (int i = 0; i < acc.length; i++) acc[i] += v[i];
    }

    private static void row(FloatTensor t, int row, int dim, float[] out) {
        long base = (long) row * dim;
        for (int i = 0; i < dim; i++) out[i] = t.getFloat(base + i);
    }

    private static void rowOf3d(FloatTensor t, int slice, int row, int rowsPerSlice, int dim, float[] out) {
        long base = ((long) slice * rowsPerSlice + row) * dim;
        for (int i = 0; i < dim; i++) out[i] = t.getFloat(base + i);
    }

    private static void headNorm(float[] v, int off, float[] w, int n, float eps) {
        float ss = 0f;
        for (int i = 0; i < n; i++) ss += v[off + i] * v[off + i];
        float inv = 1f / (float) Math.sqrt(ss / n + eps);
        for (int i = 0; i < n; i++) v[off + i] = v[off + i] * inv * w[i];
    }

    /** NEOX RoPE over the full head dim at position pos. */
    static void ropeNeox(float[] v, int heads, int headDim, int pos, float theta) {
        int half = headDim / 2;
        for (int i = 0; i < half; i++) {
            double ang = pos * Math.pow(theta, -2.0 * i / headDim);
            float c = (float) Math.cos(ang), s = (float) Math.sin(ang);
            for (int hh = 0; hh < heads; hh++) {
                int o = hh * headDim;
                float x0 = v[o + i], x1 = v[o + i + half];
                v[o + i] = x0 * c - x1 * s;
                v[o + i + half] = x0 * s + x1 * c;
            }
        }
    }

    private static void layerNorm(float[] x, float[] w, float[] b, int n, float eps) {
        double mean = 0;
        for (int i = 0; i < n; i++) mean += x[i];
        mean /= n;
        double var = 0;
        for (int i = 0; i < n; i++) { double d = x[i] - mean; var += d * d; }
        float inv = (float) (1.0 / Math.sqrt(var / n + eps));
        float mu = (float) mean;
        for (int i = 0; i < n; i++) x[i] = (x[i] - mu) * inv * w[i] + b[i];
    }

    private static final float SQRT_2_OVER_PI = (float) Math.sqrt(2.0 / Math.PI);

    private static void gelu(float[] x) {
        for (int i = 0; i < x.length; i++) {
            float v = x[i];
            x[i] = 0.5f * v * (1f + (float) Math.tanh(SQRT_2_OVER_PI * (v + 0.044715f * v * v * v)));
        }
    }

    // ---- loading ----

    private static FloatTensor tensor(GGUFFile g, String name) {
        FloatTensor t = tensorOpt(g, name);
        if (t == null) throw new IllegalStateException("TTS mmproj tensor not found: " + name);
        return t;
    }

    private static FloatTensor tensorOpt(GGUFFile g, String name) {
        GGUFTensorInfo info = g.findTensor(name);
        if (info == null) return null;
        Object saved = TensorFactory.getGpuBufferManager();
        try {
            TensorFactory.setGpuBufferManager(null);
            return TensorFactory.create(info.type(), g.getTensorData(info), info.elementCount());
        } finally {
            TensorFactory.setGpuBufferManager(saved);
        }
    }

    private static float[] f32(GGUFFile g, String name) {
        float[] v = f32opt(g, name);
        if (v == null) throw new IllegalStateException("TTS mmproj tensor not found: " + name);
        return v;
    }

    private static float[] f32opt(GGUFFile g, String name) {
        FloatTensor t = tensorOpt(g, name);
        if (t == null) return null;
        int n = (int) g.findTensor(name).elementCount();
        float[] out = new float[n];
        for (int i = 0; i < n; i++) out[i] = t.getFloat(i);
        return out;
    }

    private static Conv conv(GGUFFile g, String prefix, boolean depthwise) {
        long[] d = g.findTensor(prefix + ".weight").dimensions();
        int k = (int) d[0], ic = (int) d[1], oc = (int) d[2];
        return new Conv(k, ic, oc, depthwise, f32(g, prefix + ".weight"), f32opt(g, prefix + ".bias"));
    }

    private static ConvT convT(GGUFFile g, String prefix, int stride) {
        long[] d = g.findTensor(prefix + ".weight").dimensions();
        int k = (int) d[0], oc = (int) d[1], ic = (int) d[2];
        return new ConvT(k, oc, ic, stride, f32(g, prefix + ".weight"), f32opt(g, prefix + ".bias"));
    }
}
