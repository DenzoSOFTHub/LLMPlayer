package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.Qwen35LayerWeights;
import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.MatmulPool;
import it.denzosoft.llmplayer.tensor.VectorOps;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

import java.util.Arrays;

/**
 * Qwen3.5 MoE feed-forward (llama.cpp qwen35moe.cpp build_layer_ffn): softmax router over all
 * experts, top-k selection with the selected weights renormalised to sum 1, SwiGLU experts, plus a
 * shared SwiGLU expert scaled by {@code sigmoid(ffn_gate_inp_shexp · x)}.
 *
 * <p>The routed experts' rows are split across the matmul pool as one range per projection (gate
 * and up together, then down), so every core works even though only k experts are active.
 */
final class SoftmaxMoe {

    private static final int ROW_CHUNK = 16;

    /** Per-state buffers. */
    static final class Scratch {
        final float[] logits;
        final int[] selected;
        final float[] weights;
        final float[][] gate, up, out;
        final float[] shGate, shUp, shOut;

        Scratch(int dim, int expertCount, int topK, int expertFfn, int sharedFfn) {
            logits = new float[expertCount];
            selected = new int[topK];
            weights = new float[topK];
            gate = new float[topK][expertFfn];
            up = new float[topK][expertFfn];
            out = new float[topK][dim];
            shGate = new float[Math.max(1, sharedFfn)];
            shUp = new float[Math.max(1, sharedFfn)];
            shOut = new float[dim];
        }
    }

    private final int dim, expertCount, topK, expertFfn, sharedFfn;

    /** GPU hot-expert cache (hybrid split), or null for CPU-only experts. */
    private volatile GpuExpertCache gpuCache;

    void setGpuCache(GpuExpertCache cache) { this.gpuCache = cache; }

    GpuExpertCache gpuCache() { return gpuCache; }

    /** Phase profile of the owning engine: receives the split of the expert phase (cpu.profile). */
    private DecodeProfile prof;

    void setProfile(DecodeProfile prof) { this.prof = prof; }

    private RoutingStats routingStats;

    void enableRoutingStats(int layers) { routingStats = RoutingStats.create(layers, expertCount, topK); }

    SoftmaxMoe(int dim, int expertCount, int topK, int expertFfn, int sharedFfn) {
        this.dim = dim;
        this.expertCount = expertCount;
        this.topK = topK;
        this.expertFfn = expertFfn;
        this.sharedFfn = sharedFfn;
    }

    Scratch newScratch() {
        return new Scratch(dim, expertCount, topK, expertFfn, sharedFfn);
    }

    /** {@code out = MoE(x)} for one token ({@code x} is the FFN-normed input; out is overwritten). */
    void forward(Scratch s, Qwen35LayerWeights lw, int layer, float[] x, float[] out) {
        forward(s, lw, layer, x, out, true);
    }

    /** As above; the shared expert is left out when {@code withShared} is false (GPU-computed). */
    void forward(Scratch s, Qwen35LayerWeights lw, int layer, float[] x, float[] out, boolean withShared) {
        VectorOps ops = VectorOpsFactory.get();
        final DecodeProfile p = prof != null && prof.detailed ? prof : null;
        long tRoute = p != null ? System.nanoTime() : 0;

        // Router: softmax over every expert, top-k, renormalise the selected probabilities
        Arrays.fill(s.logits, 0f);
        lw.ffnGateInp().matmul(x, s.logits, expertCount, dim);
        ops.softmax(s.logits, 0, expertCount);
        selectTopK(s.logits, s.selected, s.weights);
        float sum = 0f;
        for (int k = 0; k < topK; k++) sum += s.weights[k];
        float inv = 1f / Math.max(sum, 6.103515625e-5f);
        for (int k = 0; k < topK; k++) s.weights[k] *= inv;
        if (routingStats != null) routingStats.count(layer, s.selected, topK);
        if (p != null) tRoute = System.nanoTime() - tRoute;

        GpuExpertCache cache = gpuCache;
        if (cache != null) {
            // Resident experts on the GPU while the CPU computes the others
            try {
                long ta = p != null ? System.nanoTime() : 0;
                int mask = cache.launchResident(lw.ffnGateExps(), lw.ffnUpExps(), lw.ffnDownExps(), x,
                    s.selected, topK, layer, dim, expertFfn, false, null, null, null);
                long tb = p != null ? System.nanoTime() : 0;
                routed(s, lw, x, mask);
                long tc = p != null ? System.nanoTime() : 0;
                cache.finishResident(mask, s.out);
                if (p != null) p.moeSplit(tRoute, tb - ta, tc - tb, System.nanoTime() - tc);
            } catch (RuntimeException e) {
                System.err.println("Expert GPU cache failed, using the CPU experts — " + GpuFailureException.describe(e));
                gpuCache = null;
                routed(s, lw, x, 0);
            }
        } else {
            long ta = p != null ? System.nanoTime() : 0;
            routed(s, lw, x, 0);
            if (p != null) p.moeSplit(tRoute, 0, System.nanoTime() - ta, 0);
        }

        Arrays.fill(out, 0, dim, 0f);
        for (int k = 0; k < topK; k++) ops.saxpy(s.weights[k], s.out[k], 0, out, 0, dim);

        // Shared expert, gated by a sigmoid of one logit
        if (withShared && lw.ffnUpShexp() != null) {
            Arrays.fill(s.shGate, 0f);
            Arrays.fill(s.shUp, 0f);
            lw.ffnGateShexp().matmulParallel(x, s.shGate, sharedFfn, dim);
            lw.ffnUpShexp().matmulParallel(x, s.shUp, sharedFfn, dim);
            ops.silu(s.shGate, sharedFfn);
            ops.elementwiseMul(s.shGate, s.shUp, s.shGate, sharedFfn);
            Arrays.fill(s.shOut, 0f);
            lw.ffnDownShexp().matmulParallel(s.shGate, s.shOut, dim, sharedFfn);
            float g = 1f;
            if (lw.ffnGateInpShexp() != null) {
                float logit = lw.ffnGateInpShexp().dot(0, x, 0, dim);
                g = 1f / (1f + (float) Math.exp(-logit));
            }
            ops.saxpy(g, s.shOut, 0, out, 0, dim);
        }
    }

    /**
     * Run the decode path's routed experts once on {@code x} (router, top-k, every selected expert
     * through the pool's {@code dot} loops; no GPU cache, no statistics, nothing kept). A batched
     * prefill never executes those loops, and a warm-up of {@code dot} alone on one thread did not
     * bring the decode up to speed: after the GPU batched prefill of Qwen3.5-35B-A3B the CPU experts
     * ran about 4x slower per token (196-212 ms against 45-71 ms after a per-token prefill) until
     * this ran first.
     */
    void warmDecodePath(Scratch s, Qwen35LayerWeights lw, float[] x) {
        VectorOps ops = VectorOpsFactory.get();
        Arrays.fill(s.logits, 0f);
        lw.ffnGateInp().matmul(x, s.logits, expertCount, dim);
        ops.softmax(s.logits, 0, expertCount);
        selectTopK(s.logits, s.selected, s.weights);
        routed(s, lw, x, 0);
    }

    /** Routed experts on the CPU, skipping the slots in {@code skip} (done on the GPU). */
    private void routed(Scratch s, Qwen35LayerWeights lw, float[] x, int skip) {
        if (skip == (1 << topK) - 1) return;
        final FloatTensor wGate = lw.ffnGateExps(), wUp = lw.ffnUpExps(), wDown = lw.ffnDownExps();
        final int efd = expertFfn;
        final int[] sel = s.selected;
        // F11: the parallel ranges cover only the CPU slots (bit-identical: one dot per row)
        final int[] slots = new int[topK];
        int nCpu = 0;
        for (int k = 0; k < topK; k++) if ((skip & (1 << k)) == 0) slots[nCpu++] = k;
        final int m = nCpu;
        MatmulPool.forRange(m * efd, ROW_CHUNK, (from, to) -> {
            for (int u = from; u < to; ) {
                int j = u / efd, slot = slots[j], r0 = u - j * efd, r1 = Math.min(efd, r0 + (to - u));
                long base = (long) sel[slot] * efd * dim;
                float[] g = s.gate[slot], up = s.up[slot];
                for (int r = r0; r < r1; r++) {
                    g[r] = wGate.dot(base + (long) r * dim, x, 0, dim);
                    up[r] = wUp.dot(base + (long) r * dim, x, 0, dim);
                }
                u += r1 - r0;
            }
        });
        VectorOps ops = VectorOpsFactory.get();
        for (int k = 0; k < topK; k++) {
            if ((skip & (1 << k)) != 0) continue;
            ops.silu(s.gate[k], efd);
            ops.elementwiseMul(s.gate[k], s.up[k], s.gate[k], efd);
        }
        MatmulPool.forRange(m * dim, ROW_CHUNK, (from, to) -> {
            for (int u = from; u < to; ) {
                int j = u / dim, slot = slots[j], r0 = u - j * dim, r1 = Math.min(dim, r0 + (to - u));
                long base = (long) sel[slot] * dim * efd;
                float[] o = s.out[slot], in = s.gate[slot];
                for (int r = r0; r < r1; r++) o[r] = wDown.dot(base + (long) r * efd, in, 0, efd);
                u += r1 - r0;
            }
        });
    }

    // ==================== Batched prefill ====================

    /** Chunk buffers for {@link #forwardBatch}. */
    static final class BatchScratch {
        final int cap;
        final int[] sel, groupStart, grouped, used;
        final float[] w;
        final float[][] eGate, eUp, eOut, shGate, shUp, shOut;
        final Scratch one;

        BatchScratch(SoftmaxMoe m, int cap) {
            this.cap = cap;
            int slots = cap * m.topK;
            sel = new int[slots];
            w = new float[slots];
            groupStart = new int[m.expertCount + 1];
            grouped = new int[slots];
            used = new int[m.expertCount];
            eGate = new float[slots][m.expertFfn];
            eUp = new float[slots][m.expertFfn];
            eOut = new float[slots][m.dim];
            int sh = Math.max(1, m.sharedFfn);
            shGate = new float[cap][sh];
            shUp = new float[cap][sh];
            shOut = new float[cap][m.dim];
            one = m.newScratch();
        }
    }

    private ExpertViews views;

    BatchScratch newBatchScratch(int cap, int layers) {
        if (views == null) views = new ExpertViews(layers, expertCount);
        return new BatchScratch(this, cap);
    }

    /**
     * {@code out[t] = MoE(xn[t])} for the chunk's {@code n} tokens (CPU): every token routed with
     * the one-token code, the (token, slot) pairs grouped by expert, each expert's three projections
     * run once over all its tokens ({@code matmulRowsBatch} on its view, the experts in parallel),
     * then per token the weighted sum in slot order and the gated shared expert, batched.
     */
    void forwardBatch(BatchScratch bs, Qwen35LayerWeights lw, int layer, float[][] xn, int n, float[][] out) {
        VectorOps ops = VectorOpsFactory.get();
        Scratch s = bs.one;
        GpuExpertCache gc = gpuCache;
        for (int t = 0; t < n; t++) {
            Arrays.fill(s.logits, 0f);
            lw.ffnGateInp().matmul(xn[t], s.logits, expertCount, dim);
            ops.softmax(s.logits, 0, expertCount);
            selectTopK(s.logits, s.selected, s.weights);
            float sum = 0f;
            for (int k = 0; k < topK; k++) sum += s.weights[k];
            float inv = 1f / Math.max(sum, 6.103515625e-5f);
            for (int k = 0; k < topK; k++) {
                bs.sel[t * topK + k] = s.selected[k];
                bs.w[t * topK + k] = s.weights[k] * inv;
            }
            if (gc != null) gc.noteRouting(layer, s.selected, topK);
            if (routingStats != null) routingStats.count(layer, s.selected, topK);
        }
        int slots = n * topK;
        int nUsed = ExpertViews.groupByExpert(bs.sel, slots, expertCount, bs.groupStart, bs.grouped, bs.used);
        final int efd = expertFfn;
        long elementsPerSlice = (long) efd * dim;
        views.forEachExpert(null, layer, bs.used, nUsed, lw.ffnGateExps(), lw.ffnUpExps(), lw.ffnDownExps(),
            elementsPerSlice, (e, wg, wu, wd) -> {
                int from = bs.groupStart[e], m = bs.groupStart[e + 1] - from;
                float[][] in = new float[m][], g = new float[m][], u = new float[m][], o = new float[m][];
                for (int i = 0; i < m; i++) {
                    int sl = bs.grouped[from + i];
                    in[i] = xn[sl / topK];
                    g[i] = bs.eGate[sl]; u[i] = bs.eUp[sl]; o[i] = bs.eOut[sl];
                    Arrays.fill(g[i], 0, efd, 0f);
                    Arrays.fill(u[i], 0, efd, 0f);
                    Arrays.fill(o[i], 0, dim, 0f);
                }
                wg.matmulRowsBatch(in, g, m, 0, efd, dim);
                wu.matmulRowsBatch(in, u, m, 0, efd, dim);
                VectorOps v = VectorOpsFactory.get();
                for (int i = 0; i < m; i++) {
                    v.silu(g[i], efd);
                    v.elementwiseMul(g[i], u[i], g[i], efd);
                }
                wd.matmulRowsBatch(g, o, m, 0, dim, efd);
            });
        for (int t = 0; t < n; t++) {
            Arrays.fill(out[t], 0, dim, 0f);
            for (int k = 0; k < topK; k++) {
                int sl = t * topK + k;
                if (bs.sel[sl] >= 0) ops.saxpy(bs.w[sl], bs.eOut[sl], 0, out[t], 0, dim);
            }
        }
        if (lw.ffnUpShexp() != null) {
            for (int t = 0; t < n; t++) {
                Arrays.fill(bs.shGate[t], 0f);
                Arrays.fill(bs.shUp[t], 0f);
            }
            FloatTensor.fusedGateUpBatchParallel(lw.ffnGateShexp(), lw.ffnUpShexp(), xn, bs.shGate, bs.shUp, n, sharedFfn, dim);
            for (int t = 0; t < n; t++) {
                ops.silu(bs.shGate[t], sharedFfn);
                ops.elementwiseMul(bs.shGate[t], bs.shUp[t], bs.shGate[t], sharedFfn);
                Arrays.fill(bs.shOut[t], 0f);
            }
            FloatTensor.matmulBatchParallel(lw.ffnDownShexp(), bs.shGate, bs.shOut, n, dim, sharedFfn);
            for (int t = 0; t < n; t++) {
                float g = 1f;
                if (lw.ffnGateInpShexp() != null) {
                    float logit = lw.ffnGateInpShexp().dot(0, xn[t], 0, dim);
                    g = 1f / (1f + (float) Math.exp(-logit));
                }
                ops.saxpy(g, bs.shOut[t], 0, out[t], 0, dim);
            }
        }
    }

    /** Top-k indices of {@code p} (descending) and their values. */
    private void selectTopK(float[] p, int[] idx, float[] val) {
        Arrays.fill(val, Float.NEGATIVE_INFINITY);
        Arrays.fill(idx, 0);
        for (int e = 0; e < expertCount; e++) {
            float v = p[e];
            if (v <= val[topK - 1]) continue;
            int j = topK - 1;
            while (j > 0 && val[j - 1] < v) {
                val[j] = val[j - 1];
                idx[j] = idx[j - 1];
                j--;
            }
            val[j] = v;
            idx[j] = e;
        }
    }

    /** Warm the single-token kernels decode uses (see {@link FloatTensor#warmUpDot}). */
    void warmUp(Qwen35LayerWeights lw) {
        FloatTensor.warmUpDot(lw.ffnGateExps(), expertFfn, dim);
        FloatTensor.warmUpDot(lw.ffnDownExps(), dim, expertFfn);
        if (lw.ffnGateShexp() != null) {
            FloatTensor.warmUpRows(lw.ffnGateShexp(), sharedFfn, dim);
            FloatTensor.warmUpRows(lw.ffnDownShexp(), dim, sharedFfn);
        }
    }
}
