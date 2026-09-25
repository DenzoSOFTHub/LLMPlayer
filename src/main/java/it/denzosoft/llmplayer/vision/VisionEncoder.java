package it.denzosoft.llmplayer.vision;

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

/**
 * Vision encoder for the Qwen-VL mmproj projectors, following llama.cpp {@code tools/mtmd/models}:
 * {@code qwen3vl_merger} (Qwen3-VL and Qwen3.5, {@code qwen3vl.cpp}) and {@code qwen2.5vl_merger}
 * (Qwen2.5-VL, {@code qwen2vl.cpp}).
 *
 * <p>Pipeline for one image of {@code pw × ph} patches (16 px each):
 * <ol>
 *   <li>Patch embedding: the two temporal Conv3d kernels applied to the same frame and summed, plus
 *       bias. Patches are processed in 2×2 merge-block order (block row, block col, dy, dx).</li>
 *   <li>Learned absolute position embedding, bilinearly interpolated (align corners) from its
 *       square grid to {@code pw × ph}.</li>
 *   <li>{@code n} pre-LN ViT blocks (LayerNorm, fused QKV with bias, 2D vision RoPE, full
 *       attention, output projection; LayerNorm, GELU MLP). Blocks flagged as deepstack also emit
 *       a merged feature (LayerNorm over 4 × dim, GELU MLP to the LLM width).</li>
 *   <li>Post LayerNorm, 2×2 merge (4 consecutive tokens), GELU MLP to the LLM width.</li>
 * </ol>
 * Each output token is {@code [main, deepstack_0, ..., deepstack_{k-1}]}, each of the LLM width; the
 * deepstack parts are added to the LLM hidden state after its first k layers.
 *
 * <p>The Qwen2.5-VL variant differs in the blocks: RMSNorm without bias, separate Q/K/V with bias,
 * a SiLU-gated MLP, no learned position embedding and no patch bias, and <b>window attention</b>:
 * the patches are reordered so that each window of {@code 112 / patch / 2} merged tokens per side is
 * contiguous, every layer except each {@code n_wa_pattern}-th attends only within its window, and
 * the merged output is put back in grid order.
 *
 * <p>Weights are expanded to F32 at load time (there is no SIMD F16 kernel) and every projection
 * runs as a tiled multi-token matmul on the shared {@link MatmulPool}.
 */
public final class VisionEncoder {

    private final boolean qwen25;            // qwen2.5vl_merger variant
    private final int windowPattern, windowMerged; // Qwen2.5-VL: full-attention period, window side in merged tokens
    private final int nEmbd, nHead, dHead, nLayer, ffnDim, patchSize, merge, projDim, imageSize;
    private final float eps;
    private final float[] mean, std;
    private final int posGrid;          // learned position grid side (image_size / patch_size)

    private final float[] patchW;       // [nEmbd][3*p*p] (both temporal kernels summed)
    private final float[] patchB;
    private final float[] posEmbd;      // [posGrid*posGrid][nEmbd]
    private final Layer[] layers;
    private final float[] postLnW, postLnB;
    private final float[] mm0W, mm0B, mm2W, mm2B;   // merger: [4E -> 4E] GELU [4E -> proj]
    private final int[] deepstackLayers;             // layer indices with a deepstack merger
    private final Merger[] deepstack;

    private static final class Layer {
        float[] ln1W, ln1B, qkvW, qkvB, oW, oB, ln2W, ln2B, upW, upB, downW, downB;
        float[] gateW, gateB;    // Qwen2.5-VL gated MLP
    }

    private static final class Merger {
        float[] normW, normB, fc1W, fc1B, fc2W, fc2B;
    }

    /** Result of encoding one image. */
    public static final class Output {
        /** [nTokens][projDim * (1 + deepstackCount)] */
        public final float[][] embeddings;
        public final int nx, ny;            // merged grid (columns, rows)
        public final int projDim, deepstackCount;

        Output(float[][] embeddings, int nx, int ny, int projDim, int deepstackCount) {
            this.embeddings = embeddings;
            this.nx = nx;
            this.ny = ny;
            this.projDim = projDim;
            this.deepstackCount = deepstackCount;
        }
    }

    private VisionEncoder(GGUFFile gguf) {
        GGUFMetadata md = gguf.getMetadata();
        String proj = md.getString("clip.projector_type", "");
        if (!"qwen3vl_merger".equals(proj) && !"qwen2.5vl_merger".equals(proj)) {
            throw new IllegalArgumentException("Unsupported mmproj projector type: '" + proj
                + "' (supported: qwen3vl_merger — Qwen3-VL, Qwen3.5; qwen2.5vl_merger — Qwen2.5-VL)");
        }
        qwen25 = "qwen2.5vl_merger".equals(proj);
        nEmbd = md.getInt("clip.vision.embedding_length");
        nHead = md.getInt("clip.vision.attention.head_count");
        dHead = nEmbd / nHead;
        nLayer = md.getInt("clip.vision.block_count");
        ffnDim = md.getInt("clip.vision.feed_forward_length");
        patchSize = md.getInt("clip.vision.patch_size");
        merge = md.getInt("clip.vision.spatial_merge_size", 2);
        projDim = md.getInt("clip.vision.projection_dim");
        imageSize = md.getInt("clip.vision.image_size", 768);
        eps = md.getFloat("clip.vision.attention.layer_norm_epsilon", 1e-6f);
        mean = floats(md.getFloatArray("clip.vision.image_mean"), 0.5f);
        std = floats(md.getFloatArray("clip.vision.image_std"), 0.5f);
        if (merge != 2) throw new IllegalArgumentException("Only spatial_merge_size=2 is supported, got " + merge);
        windowPattern = md.getInt("clip.vision.n_wa_pattern", 0);
        windowMerged = Math.max(1, md.getInt("clip.vision.window_size", 112) / patchSize / merge);

        // Weights → F32. Patch kernels [p, p, 3, E] (GGUF order), flat index o*3pp + c*pp + ky*p + kx.
        float[] w0 = f32(gguf, "v.patch_embd.weight");
        float[] w1 = f32opt(gguf, "v.patch_embd.weight.1");
        if (w1 != null) for (int i = 0; i < w0.length; i++) w0[i] += w1[i];
        patchW = w0;
        patchB = f32opt(gguf, "v.patch_embd.bias");
        posEmbd = f32opt(gguf, "v.position_embd.weight");
        posGrid = posEmbd != null ? (int) Math.round(Math.sqrt(posEmbd.length / (double) nEmbd)) : 0;

        layers = new Layer[nLayer];
        for (int i = 0; i < nLayer; i++) {
            String p = "v.blk." + i + ".";
            Layer l = new Layer();
            l.ln1W = f32(gguf, p + "ln1.weight");
            l.ln1B = f32opt(gguf, p + "ln1.bias");
            if (gguf.findTensor(p + "attn_qkv.weight") != null) {
                l.qkvW = f32(gguf, p + "attn_qkv.weight");
                l.qkvB = f32(gguf, p + "attn_qkv.bias");
            } else {
                // separate Q/K/V (Qwen2.5-VL): stack them into one [3E][E] matrix
                l.qkvW = concat(f32(gguf, p + "attn_q.weight"), f32(gguf, p + "attn_k.weight"), f32(gguf, p + "attn_v.weight"));
                l.qkvB = concat(f32(gguf, p + "attn_q.bias"), f32(gguf, p + "attn_k.bias"), f32(gguf, p + "attn_v.bias"));
            }
            l.oW = f32(gguf, p + "attn_out.weight");
            l.oB = f32(gguf, p + "attn_out.bias");
            l.ln2W = f32(gguf, p + "ln2.weight");
            l.ln2B = f32opt(gguf, p + "ln2.bias");
            l.upW = f32(gguf, p + "ffn_up.weight");
            l.upB = f32opt(gguf, p + "ffn_up.bias");
            l.gateW = f32opt(gguf, p + "ffn_gate.weight");
            l.gateB = f32opt(gguf, p + "ffn_gate.bias");
            l.downW = f32(gguf, p + "ffn_down.weight");
            l.downB = f32opt(gguf, p + "ffn_down.bias");
            layers[i] = l;
        }
        postLnW = f32(gguf, "v.post_ln.weight");
        postLnB = f32opt(gguf, "v.post_ln.bias");
        mm0W = f32(gguf, "mm.0.weight");
        mm0B = f32(gguf, "mm.0.bias");
        mm2W = f32(gguf, "mm.2.weight");
        mm2B = f32(gguf, "mm.2.bias");

        Object dsObj = md.get("clip.vision.is_deepstack_layers");
        int count = 0;
        boolean[] ds = new boolean[nLayer];
        if (dsObj instanceof Object[]) {
            Object[] arr = (Object[]) dsObj;
            for (int i = 0; i < Math.min(arr.length, nLayer); i++) {
                ds[i] = arr[i] instanceof Boolean ? (Boolean) arr[i] : Boolean.parseBoolean(String.valueOf(arr[i]));
                if (ds[i]) count++;
            }
        }
        deepstackLayers = new int[count];
        deepstack = new Merger[count];
        for (int i = 0, j = 0; i < nLayer; i++) {
            if (!ds[i]) continue;
            String p = "v.deepstack." + i + ".";
            Merger m = new Merger();
            m.normW = f32(gguf, p + "norm.weight");
            m.normB = f32(gguf, p + "norm.bias");
            m.fc1W = f32(gguf, p + "fc1.weight");
            m.fc1B = f32(gguf, p + "fc1.bias");
            m.fc2W = f32(gguf, p + "fc2.weight");
            m.fc2B = f32(gguf, p + "fc2.bias");
            deepstackLayers[j] = i;
            deepstack[j++] = m;
        }
    }

    /** Load an mmproj GGUF. The weights are copied to the heap, so the file is closed afterwards. */
    public static VisionEncoder load(Path mmproj) throws IOException {
        try (GGUFFile gguf = GGUFParser.parse(mmproj, false)) {
            VisionEncoder enc = new VisionEncoder(gguf);
            System.out.println("  Vision encoder: " + mmproj.getFileName() + " ("
                + (enc.qwen25 ? "qwen2.5vl_merger" : "qwen3vl_merger") + ", " + enc.nLayer
                + " layers, dim " + enc.nEmbd + ", patch " + enc.patchSize + ", proj " + enc.projDim
                + ", deepstack " + enc.deepstack.length + ")");
            return enc;
        }
    }

    public int projectionDim() { return projDim; }
    public int deepstackCount() { return deepstack.length; }
    public int patchSize() { return patchSize; }
    public int mergeSize() { return merge; }
    public float[] imageMean() { return mean; }
    public float[] imageStd() { return std; }
    public int nativeImageSize() { return imageSize; }

    // ==================== Encoding ====================

    public Output encode(ImagePreprocessor.Image img) {
        int pw = img.width / patchSize, ph = img.height / patchSize;
        if (pw % 2 != 0 || ph % 2 != 0) throw new IllegalArgumentException("patch grid must be even: " + pw + "x" + ph);
        int n = pw * ph;
        int e = nEmbd;
        long t0 = System.nanoTime();

        // 1-2. patch embedding (+ interpolated position embedding for Qwen3-VL). Patches are ordered
        // merge block by merge block; Qwen2.5-VL additionally groups the merge blocks by window.
        int mw = pw / 2, mh = ph / 2, m0 = mw * mh;
        int[] order = new int[m0];          // processing slot -> merged grid index (row-major)
        int[] winStart = new int[m0], winEnd = new int[m0];  // window of each slot, in slots
        int slot = 0;
        int gw = qwen25 && windowPattern > 0 ? windowMerged : Math.max(mw, mh);
        for (int wy = 0; wy < mh; wy += gw) {
            for (int wx = 0; wx < mw; wx += gw) {
                int first = slot;
                for (int dy = 0; dy < Math.min(gw, mh - wy); dy++) {
                    for (int dx = 0; dx < Math.min(gw, mw - wx); dx++) {
                        order[slot++] = (wy + dy) * mw + (wx + dx);
                    }
                }
                for (int k = first; k < slot; k++) { winStart[k] = first; winEnd[k] = slot; }
            }
        }
        int[] posY = new int[n], posX = new int[n];
        int pp = patchSize * patchSize, kin = 3 * pp;
        float[][] patches = new float[n][kin];
        for (int si = 0; si < m0; si++) {
            int my = order[si] / mw, mx = order[si] % mw;
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    int idx = si * 4 + dy * 2 + dx;
                    int py = 2 * my + dy, px = 2 * mx + dx;
                    posY[idx] = py;
                    posX[idx] = px;
                    float[] v = patches[idx];
                    for (int c = 0; c < 3; c++) {
                        float[] ch = img.channels[c];
                        for (int ky = 0; ky < patchSize; ky++) {
                            int row = (py * patchSize + ky) * img.width + px * patchSize;
                            System.arraycopy(ch, row, v, c * pp + ky * patchSize, patchSize);
                        }
                    }
                }
            }
        }
        float[][] x = linear(patches, n, patchW, patchB, e, kin);
        if (posEmbd != null) {
            float[] pos = interpolatePos(pw, ph);
            for (int i = 0; i < n; i++) {
                int off = (posY[i] * pw + posX[i]) * e;
                for (int j = 0; j < e; j++) x[i][j] += pos[off + j];
            }
        }

        // vision RoPE tables per token: pairs (i, i + dHead/2); first quarter uses y, second x
        int half = dHead / 2, quarter = dHead / 4;
        float[][] cos = new float[n][half], sin = new float[n][half];
        for (int t = 0; t < n; t++) {
            for (int i = 0; i < half; i++) {
                int k = i % quarter;
                double theta = (i < quarter ? posY[t] : posX[t]) * Math.pow(10000.0, -(double) k / quarter);
                cos[t][i] = (float) Math.cos(theta);
                sin[t][i] = (float) Math.sin(theta);
            }
        }

        // 3. transformer blocks
        float[][][] dsFeat = new float[deepstack.length][][];
        float[][] h = new float[n][e];
        for (int li = 0; li < nLayer; li++) {
            Layer l = layers[li];
            // Qwen2.5-VL: every n_wa_pattern-th layer attends globally, the rest within their window
            boolean windowed = qwen25 && windowPattern > 0 && (li + 1) % windowPattern != 0;
            for (int i = 0; i < n; i++) norm(x[i], h[i], l.ln1W, l.ln1B, e);
            float[][] qkv = linear(h, n, l.qkvW, l.qkvB, 3 * e, e);
            // RoPE on Q and K, per head
            final float[][] cf = cos, sf = sin;
            MatmulPool.forEach(n, t -> {
                float[] v = qkv[t];
                for (int hh = 0; hh < nHead; hh++) {
                    ropeHead(v, hh * dHead, cf[t], sf[t], half);
                    ropeHead(v, e + hh * dHead, cf[t], sf[t], half);
                }
            });
            float[][] att = attention(qkv, n, windowed ? winStart : null, windowed ? winEnd : null);
            float[][] o = linear(att, n, l.oW, l.oB, e, e);
            for (int i = 0; i < n; i++) {
                float[] xi = x[i], oi = o[i];
                for (int j = 0; j < e; j++) xi[j] += oi[j];
            }
            for (int i = 0; i < n; i++) norm(x[i], h[i], l.ln2W, l.ln2B, e);
            float[][] up = linear(h, n, l.upW, l.upB, ffnDim, e);
            if (l.gateW != null) {
                float[][] gate = linear(h, n, l.gateW, l.gateB, ffnDim, e);
                MatmulPool.forEach(n, t -> {
                    float[] u = up[t], g = gate[t];
                    for (int j = 0; j < ffnDim; j++) u[j] = g[j] / (1f + (float) Math.exp(-g[j])) * u[j];
                });
            } else {
                MatmulPool.forEach(n, t -> gelu(up[t], ffnDim));
            }
            float[][] down = linear(up, n, l.downW, l.downB, e, ffnDim);
            for (int i = 0; i < n; i++) {
                float[] xi = x[i], di = down[i];
                for (int j = 0; j < e; j++) xi[j] += di[j];
            }
            for (int d = 0; d < deepstackLayers.length; d++) {
                if (deepstackLayers[d] == li) dsFeat[d] = mergerPostShuffleNorm(x, n, deepstack[d]);
            }
        }

        // 4. post LN + 2x2 merge + MLP
        for (int i = 0; i < n; i++) norm(x[i], h[i], postLnW, postLnB, e);
        int m = n / 4;
        float[][] grouped = group4(h, n, e);
        float[][] a = linear(grouped, m, mm0W, mm0B, 4 * e, 4 * e);
        MatmulPool.forEach(m, t -> gelu(a[t], 4 * e));
        float[][] main = linear(a, m, mm2W, mm2B, projDim, 4 * e);

        int width = projDim * (1 + deepstack.length);
        float[][] out = new float[m][width];
        for (int t = 0; t < m; t++) {
            // slot t holds merged grid index order[t]; the output is in grid (row-major) order
            int g = order[t];
            System.arraycopy(main[t], 0, out[g], 0, projDim);
            for (int d = 0; d < deepstack.length; d++) {
                System.arraycopy(dsFeat[d][t], 0, out[g], (d + 1) * projDim, projDim);
            }
        }
        System.out.printf("  Vision: %dx%d px -> %d patches -> %d tokens (%dx%d) in %.1f s%n",
            img.width, img.height, n, m, pw / 2, ph / 2, (System.nanoTime() - t0) / 1e9);
        return new Output(out, pw / 2, ph / 2, projDim, deepstack.length);
    }

    /** Deepstack merger: group 4 tokens, LayerNorm over 4E, fc1, GELU, fc2. */
    private float[][] mergerPostShuffleNorm(float[][] x, int n, Merger mg) {
        int e4 = 4 * nEmbd;
        float[][] g = group4(x, n, nEmbd);
        int m = n / 4;
        for (int t = 0; t < m; t++) layerNorm(g[t], g[t], mg.normW, mg.normB, e4);
        float[][] a = linear(g, m, mg.fc1W, mg.fc1B, e4, e4);
        MatmulPool.forEach(m, t -> gelu(a[t], e4));
        return linear(a, m, mg.fc2W, mg.fc2B, projDim, e4);
    }

    private static float[][] group4(float[][] x, int n, int e) {
        int m = n / 4;
        float[][] g = new float[m][4 * e];
        for (int t = 0; t < m; t++) {
            for (int k = 0; k < 4; k++) System.arraycopy(x[4 * t + k], 0, g[t], k * e, e);
        }
        return g;
    }

    /**
     * Non-causal multi-head attention; Q|K|V packed per token. When {@code winStart} is given, patch
     * i attends only to the patches of its window (merged slots winStart..winEnd, 4 patches each).
     */
    private float[][] attention(float[][] qkv, int n, int[] winStart, int[] winEnd) {
        int e = nEmbd;
        float scale = (float) (1.0 / Math.sqrt(dHead));
        float[][] out = new float[n][e];
        final int qBlock = 32;
        int qBlocks = (n + qBlock - 1) / qBlock;
        VectorOps ops = VectorOpsFactory.get();
        MatmulPool.forEach(nHead * qBlocks, task -> {
            int hh = task / qBlocks, qb = task % qBlocks;
            int hOff = hh * dHead;
            float[] scores = new float[n];
            for (int i = qb * qBlock; i < Math.min(n, (qb + 1) * qBlock); i++) {
                float[] qi = qkv[i];
                int j0 = winStart != null ? winStart[i / 4] * 4 : 0;
                int j1 = winEnd != null ? winEnd[i / 4] * 4 : n;
                float max = Float.NEGATIVE_INFINITY;
                for (int j = j0; j < j1; j++) {
                    float s = ops.dot(qi, hOff, qkv[j], e + hOff, dHead) * scale;
                    scores[j] = s;
                    if (s > max) max = s;
                }
                float sum = 0f;
                for (int j = j0; j < j1; j++) {
                    float p = (float) Math.exp(scores[j] - max);
                    scores[j] = p;
                    sum += p;
                }
                float inv = 1f / sum;
                float[] oi = out[i];
                for (int j = j0; j < j1; j++) {
                    ops.saxpy(scores[j] * inv, qkv[j], 2 * e + hOff, oi, hOff, dHead);
                }
            }
        });
        return out;
    }

    private static void ropeHead(float[] v, int off, float[] cos, float[] sin, int half) {
        for (int i = 0; i < half; i++) {
            float x0 = v[off + i], x1 = v[off + i + half];
            v[off + i] = x0 * cos[i] - x1 * sin[i];
            v[off + i + half] = x0 * sin[i] + x1 * cos[i];
        }
    }

    /** Bilinear (align corners) resize of the square learned position grid to pw × ph, row-major. */
    private float[] interpolatePos(int pw, int ph) {
        int e = nEmbd, g = posGrid;
        float[] out = new float[pw * ph * e];
        if (pw == g && ph == g) {
            System.arraycopy(posEmbd, 0, out, 0, out.length);
            return out;
        }
        double sx = pw > 1 ? (g - 1) / (double) (pw - 1) : 0;
        double sy = ph > 1 ? (g - 1) / (double) (ph - 1) : 0;
        for (int y = 0; y < ph; y++) {
            double fy = y * sy;
            int y0 = (int) Math.floor(fy), y1 = Math.min(g - 1, y0 + 1);
            float wy = (float) (fy - y0);
            for (int x = 0; x < pw; x++) {
                double fx = x * sx;
                int x0 = (int) Math.floor(fx), x1 = Math.min(g - 1, x0 + 1);
                float wx = (float) (fx - x0);
                int o = (y * pw + x) * e;
                int a = (y0 * g + x0) * e, b = (y0 * g + x1) * e, c = (y1 * g + x0) * e, d = (y1 * g + x1) * e;
                for (int j = 0; j < e; j++) {
                    float top = posEmbd[a + j] * (1 - wx) + posEmbd[b + j] * wx;
                    float bot = posEmbd[c + j] * (1 - wx) + posEmbd[d + j] * wx;
                    out[o + j] = top * (1 - wy) + bot * wy;
                }
            }
        }
        return out;
    }

    // ==================== Kernels ====================

    /** out[t] = W · in[t] + b for t < n; W is [rows][cols] row-major. Tiled over (row block, token block). */
    static float[][] linear(float[][] in, int n, float[] w, float[] b, int rows, int cols) {
        float[][] out = new float[n][rows];
        final int rb = 32, tb = 16;
        int rBlocks = (rows + rb - 1) / rb, tBlocks = (n + tb - 1) / tb;
        VectorOps ops = VectorOpsFactory.get();
        MatmulPool.forEach(rBlocks * tBlocks, task -> {
            int r0 = (task % rBlocks) * rb, t0 = (task / rBlocks) * tb;
            int r1 = Math.min(rows, r0 + rb), t1 = Math.min(n, t0 + tb);
            for (int r = r0; r < r1; r++) {
                int wOff = r * cols;
                float bias = b != null ? b[r] : 0f;
                for (int t = t0; t < t1; t++) {
                    out[t][r] = ops.dot(w, wOff, in[t], 0, cols) + bias;
                }
            }
        });
        return out;
    }

    /** LayerNorm (Qwen3-VL) or RMSNorm without bias (Qwen2.5-VL). */
    private void norm(float[] x, float[] out, float[] w, float[] b, int size) {
        if (!qwen25) {
            layerNorm(x, out, w, b, size);
            return;
        }
        double ss = 0;
        for (int i = 0; i < size; i++) ss += (double) x[i] * x[i];
        float inv = (float) (1.0 / Math.sqrt(ss / size + eps));
        for (int i = 0; i < size; i++) out[i] = x[i] * inv * w[i];
    }

    private static float[] concat(float[]... parts) {
        int len = 0;
        for (float[] p : parts) len += p.length;
        float[] out = new float[len];
        int off = 0;
        for (float[] p : parts) { System.arraycopy(p, 0, out, off, p.length); off += p.length; }
        return out;
    }

    private void layerNorm(float[] x, float[] out, float[] w, float[] b, int size) {
        double mean = 0;
        for (int i = 0; i < size; i++) mean += x[i];
        mean /= size;
        double var = 0;
        for (int i = 0; i < size; i++) {
            double d = x[i] - mean;
            var += d * d;
        }
        var /= size;
        float inv = (float) (1.0 / Math.sqrt(var + eps));
        float mu = (float) mean;
        for (int i = 0; i < size; i++) out[i] = (x[i] - mu) * inv * w[i] + (b != null ? b[i] : 0f);
    }

    private static final float SQRT_2_OVER_PI = (float) Math.sqrt(2.0 / Math.PI);

    private static void gelu(float[] x, int size) {
        for (int i = 0; i < size; i++) {
            float v = x[i];
            x[i] = 0.5f * v * (1.0f + (float) Math.tanh(SQRT_2_OVER_PI * (v + 0.044715f * v * v * v)));
        }
    }

    // ==================== Loading ====================

    private static float[] f32(GGUFFile gguf, String name) {
        float[] v = f32opt(gguf, name);
        if (v == null) throw new IllegalStateException("mmproj tensor not found: " + name);
        return v;
    }

    private static float[] f32opt(GGUFFile gguf, String name) {
        GGUFTensorInfo info = gguf.findTensor(name);
        if (info == null) return null;
        Object savedGpu = TensorFactory.getGpuBufferManager();
        FloatTensor t;
        try {
            TensorFactory.setGpuBufferManager(null);
            t = TensorFactory.create(info.type(), gguf.getTensorData(info), info.elementCount());
        } finally {
            TensorFactory.setGpuBufferManager(savedGpu);
        }
        int size = (int) info.elementCount();
        float[] out = new float[size];
        final int chunk = 1 << 16;
        MatmulPool.forEach((size + chunk - 1) / chunk, c -> {
            int from = c * chunk, to = Math.min(size, from + chunk);
            for (int i = from; i < to; i++) out[i] = t.getFloat(i);
        });
        return out;
    }

    private static float[] floats(float[] v, float dflt) {
        if (v != null && v.length >= 3) return new float[] {v[0], v[1], v[2]};
        return new float[] {dflt, dflt, dflt};
    }
}
