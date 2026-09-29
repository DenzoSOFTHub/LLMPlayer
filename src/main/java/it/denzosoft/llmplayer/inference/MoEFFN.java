package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.DeepSeek2LayerWeights;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Mixture of Experts Feed-Forward Network for DeepSeek2, GLM-4.7-Flash, and DeepSeek-V3.
 *
 * Gating modes:
 * - expertGatingFunc=0 (default): softmax over router logits, no renormalization
 * - expertGatingFunc=2 (GLM-4.7-Flash): sigmoid per expert, with exp_probs_b bias added to logits,
 *   L2 normalization of selected weights, then multiply by expertWeightsScale
 *
 * Flow:
 * 1. Router: logits = x * ffnGateInp (+ exp_probs_b bias if present) → gating → top-K expert selection
 * 2. For each selected expert: SwiGLU with sliced 3D weight tensors
 * 3. Output = weighted sum of expert outputs + shared expert output
 */
public class MoEFFN {

    private final ModelConfig config;

    /** SSD-streaming expert cache, or null when the model is resident and the mmap path is used. */
    private it.denzosoft.llmplayer.tensor.ExpertCache expertCache;
    private boolean cacheLayerReady;
    private int currentLayer;

    /** Per-expert views for batched prefill (see {@link ExpertViews}). */
    private final ExpertViews expertViews;

    public MoEFFN(ModelConfig config) {
        this.config = config;
        this.expertViews = new ExpertViews(config.blockCount(), Math.max(1, config.expertCount()));
        this.routingStats = RoutingStats.create(config.blockCount(), config.expertCount(),
            MoERouting.effectiveTopK(config.expertUsedCount()));
    }

    private final RoutingStats routingStats;
    private int routeLayer = -1;

    public void setExpertCache(it.denzosoft.llmplayer.tensor.ExpertCache cache) {
        this.expertCache = cache;
    }

    /** GPU hot-expert cache (routed experts on the GPU); null for the CPU expert path. */
    private volatile GpuExpertCache gpuExperts;

    public void setGpuExpertCache(GpuExpertCache cache) {
        this.gpuExperts = cache;
    }

    public GpuExpertCache gpuExpertCache() { return gpuExperts; }

    /** Phase profile of the owning engine: receives the split of the expert phase (cpu.profile). */
    private DecodeProfile prof;

    void setProfile(DecodeProfile prof) { this.prof = prof; }

    public void forward(DeepSeek2State state, DeepSeek2LayerWeights weights) {
        forward(state, weights, -1);
    }

    /**
     * Forward pass for MoE FFN.
     * Reads from xb (normalized input), writes result to xb (for residual addition).
     *
     * @param layer index of the layer being computed, or -1 when unknown (disables the expert cache)
     */
    public void forward(DeepSeek2State state, DeepSeek2LayerWeights weights, int layer) {
        forward(state, weights, layer, (GpuAttentionPass) null);
    }

    /**
     * As {@link #forward(DeepSeek2State, DeepSeek2LayerWeights, int)}; when {@code sharedSource} is
     * not null the shared expert is taken from it (the GPU attention pass computed it while the
     * routed experts ran) instead of being computed here.
     */
    public void forward(DeepSeek2State state, DeepSeek2LayerWeights weights, int layer, GpuAttentionPass sharedSource) {
        int dim = config.embeddingLength();
        int expertUsedCount = MoERouting.effectiveTopK(config.expertUsedCount());
        int expertFfnDim = config.expertFfnLength();
        int sharedFfnDim = config.expertSharedCount() * expertFfnDim;

        final DecodeProfile p = prof != null && prof.detailed ? prof : null;
        long tRoute = p != null ? System.nanoTime() : 0;
        // 1-5. Router, gating, top-K and weight normalisation
        routeLayer = layer;
        route(state, weights, state.xb);
        if (p != null) tRoute = System.nanoTime() - tRoute;
        long tLaunch = 0, tCpu, tWait = 0;
        int[] selectedExperts = state.selectedExperts;
        float[] selectedWeights = state.selectedWeights;

        // GPU hot-expert cache: all routed experts on the GPU in one batch, or (hybrid, default)
        // the resident ones on the GPU while the CPU computes the rest
        GpuExpertCache gpu = gpuExperts;
        int gpuMask = 0;
        if (gpu != null && layer >= 0 && !GpuExpertCache.hybrid()) {
            try {
                gpu.computeExperts(weights.ffnGateExps(), weights.ffnUpExps(), weights.ffnDownExps(),
                    state.xbSaved, selectedExperts, selectedWeights, expertUsedCount, layer, dim, expertFfnDim,
                    state.moeHbPerExpert, state.moeHb2PerExpert, state.expertOutPerExpert, false, null, null, null);
                Arrays.fill(state.xb, 0);
                for (int k = 0; k < expertUsedCount; k++) {
                    VectorOpsFactory.get().saxpy(selectedWeights[k], state.expertOutPerExpert[k], 0, state.xb, 0, dim);
                }
                addShared(state, weights, layer, dim, sharedFfnDim, sharedSource);
                return;
            } catch (RuntimeException e) {
                System.err.println("GPU expert cache failed, using the CPU experts — " + GpuFailureException.describe(e));
                gpuExperts = null; // stateless: the CPU path below computes the same experts
            }
        } else if (gpu != null && layer >= 0) {
            try {
                if (p != null) tLaunch = System.nanoTime();
                gpuMask = gpu.launchResident(weights.ffnGateExps(), weights.ffnUpExps(), weights.ffnDownExps(),
                    state.xbSaved, selectedExperts, expertUsedCount, layer, dim, expertFfnDim, false, null, null, null);
                if (p != null) tLaunch = System.nanoTime() - tLaunch;
            } catch (RuntimeException e) {
                System.err.println("GPU expert cache failed, using the CPU experts — " + GpuFailureException.describe(e));
                gpuExperts = null;
                gpu = null;
                gpuMask = 0;
            }
        }

        // SSD streaming: the top-K experts for this layer are now known. Prefer L1 — read the whole
        // slices into the RAM cache with explicit positional reads — and fall back to the L0
        // read-ahead hint when there is no cache. Both are no-ops when the model fits RAM.
        currentLayer = layer;
        cacheLayerReady = expertCache != null && layer >= 0
            && expertCache.prepare(layer, selectedExperts, expertUsedCount,
                weights.ffnGateExps(), weights.ffnUpExps(), weights.ffnDownExps(),
                (long) expertFfnDim * dim);
        if (!cacheLayerReady) {
            ExpertPrefetch.willNeed(weights.ffnGateExps(), weights.ffnUpExps(), weights.ffnDownExps(),
                selectedExperts, expertUsedCount, (long) expertFfnDim * dim);
        }

        // 2. Compute routed expert outputs, split by rows across every core: a loop over the
        // top-K experts alone would keep only K threads busy.
        Arrays.fill(state.xb, 0);  // Will accumulate weighted expert outputs here
        tCpu = p != null ? System.nanoTime() : 0;
        routedExperts(state.xbSaved, selectedExperts, expertUsedCount, dim, expertFfnDim,
            weights.ffnGateExps(), weights.ffnUpExps(), weights.ffnDownExps(),
            state.moeHbPerExpert, state.moeHb2PerExpert, state.expertOutPerExpert, gpuMask);
        if (p != null) tCpu = System.nanoTime() - tCpu;
        if (gpu != null && layer >= 0) {
            try {
                if (p != null) tWait = System.nanoTime();
                gpu.finishResident(gpuMask, state.expertOutPerExpert);
                if (p != null) tWait = System.nanoTime() - tWait;
            } catch (RuntimeException e) {
                // The GPU slots' outputs are lost: recompute them on the CPU
                System.err.println("GPU expert cache failed, using the CPU experts — " + GpuFailureException.describe(e));
                gpuExperts = null;
                routedExperts(state.xbSaved, selectedExperts, expertUsedCount, dim, expertFfnDim,
                    weights.ffnGateExps(), weights.ffnUpExps(), weights.ffnDownExps(),
                    state.moeHbPerExpert, state.moeHb2PerExpert, state.expertOutPerExpert, ~gpuMask);
            }
        }

        if (p != null) p.moeSplit(tRoute, tLaunch, tCpu, tWait);

        // Sequential accumulation of weighted expert outputs
        for (int k = 0; k < expertUsedCount; k++) {
            VectorOpsFactory.get().saxpy(selectedWeights[k], state.expertOutPerExpert[k], 0, state.xb, 0, dim);
        }

        addShared(state, weights, layer, dim, sharedFfnDim, sharedSource);
    }

    private void addShared(DeepSeek2State state, DeepSeek2LayerWeights weights, int layer, int dim, int sharedFfnDim,
                           GpuAttentionPass sharedSource) {
        if (sharedSource != null && sharedSource.takeSharedExpert(layer, state.sharedPre)) {
            VectorOpsFactory.get().accumulate(state.xb, state.sharedPre, dim);
        } else {
            sharedExpert(state, weights, dim, sharedFfnDim);
        }
    }

    /** Shared expert (standard SwiGLU) added to {@code state.xb}. */
    private void sharedExpert(DeepSeek2State state, DeepSeek2LayerWeights weights, int dim, int sharedFfnDim) {
        float[] shGate = state.sharedHb;
        float[] shUp = state.sharedHb2;
        Arrays.fill(shGate, 0, sharedFfnDim, 0f);
        Arrays.fill(shUp, 0, sharedFfnDim, 0f);

        weights.ffnGateShexp().matmulParallel(state.xbSaved, shGate, sharedFfnDim, dim);
        weights.ffnUpShexp().matmulParallel(state.xbSaved, shUp, sharedFfnDim, dim);

        VectorOpsFactory.get().silu(shGate, sharedFfnDim);
        VectorOpsFactory.get().elementwiseMul(shGate, shUp, shGate, sharedFfnDim);

        float[] sharedOut = state.expertOut;
        Arrays.fill(sharedOut, 0, dim, 0f);
        weights.ffnDownShexp().matmulParallel(shGate, sharedOut, dim, sharedFfnDim);

        // Add shared expert output
        VectorOpsFactory.get().accumulate(state.xb, sharedOut, dim);
    }

    // ==================== Batched prefill ====================

    // Indices into DeepSeek2State.moePrefill
    private static final int B_EGATE = 0, B_EUP = 1, B_EOUT = 2, B_SHB = 3, B_SHB2 = 4, B_SHOUT = 5;

    /** Whether {@link #forwardBatch} can run: the CPU matmul pool is on and the views are enabled. */
    static boolean batchAvailable() {
        return ExpertViews.active();
    }

    /**
     * Multi-token {@link #forward} for the chunk's tokens at one MoE layer: {@code out[t]} receives
     * the FFN output for the normed input {@code xn[t]}. Every token is routed with the same code as
     * the one-token path; the (token, slot) pairs are grouped by expert and each expert runs once
     * per projection over all its tokens ({@code matmulRowsBatch} on its cached slice or view); each
     * token's expert outputs are then summed in slot order and the shared expert, batched over the
     * chunk, is added — the same per-token arithmetic as {@link #forward}.
     */
    public void forwardBatch(DeepSeek2State state, DeepSeek2LayerWeights weights, int layer, int n,
                             float[][] xn, float[][] out) {
        int dim = config.embeddingLength();
        int expertCount = config.expertCount();
        int k = MoERouting.effectiveTopK(config.expertUsedCount());
        int efd = config.expertFfnLength();
        int sharedFfn = config.expertSharedCount() * efd;
        long elementsPerSlice = (long) efd * dim;
        float[][][] b = batchBuffers(state, xn.length, efd, sharedFfn);
        int[] sel = state.prefillExperts;
        float[] selW = state.prefillWeights;

        // 1. Route each token
        GpuExpertCache gc = gpuExperts;
        routeLayer = layer;
        for (int t = 0; t < n; t++) {
            route(state, weights, xn[t]);
            System.arraycopy(state.selectedExperts, 0, sel, t * k, k);
            System.arraycopy(state.selectedWeights, 0, selW, t * k, k);
            if (gc != null) gc.noteRouting(layer, state.selectedExperts, k); // warm the LFU counts
        }

        // 2. Group the (token, slot) pairs by expert
        int slots = n * k;
        int[] start = state.prefillGroupStart;
        int[] grouped = state.prefillGroupSlots;
        int[] used = state.prefillUsed;
        int nUsed = ExpertViews.groupByExpert(sel, slots, expertCount, start, grouped, used);
        for (int s = 0; s < slots; s++) {
            if (sel[s] < 0) Arrays.fill(b[B_EOUT][s], 0, dim, 0f);
        }

        // 3. Compute the experts: resident ones on the GPU over all their tokens, the others on the
        // CPU meanwhile (grouped for the SSD cache, the next group read while one computes)
        int[] cpuUsed = used;
        int nCpu = nUsed, onGpu = 0;
        if (gc != null) {
            try {
                onGpu = gc.launchResidentBatch(layer, weights.ffnGateExps(), weights.ffnUpExps(), weights.ffnDownExps(),
                    xn, n, used, nUsed, start, grouped, k, state.prefillOnGpu, false);
            } catch (RuntimeException e) {
                System.err.println("GPU expert cache failed, using the CPU experts — " + GpuFailureException.describe(e));
                gpuExperts = null;
                gc = null;
                onGpu = 0;
            }
            if (onGpu > 0) {
                cpuUsed = state.prefillCpuUsed;
                nCpu = 0;
                for (int i = 0; i < nUsed; i++) if (!state.prefillOnGpu[used[i]]) cpuUsed[nCpu++] = used[i];
            }
        }
        expertViews.forEachExpert(expertCache, layer, cpuUsed, nCpu, weights.ffnGateExps(), weights.ffnUpExps(),
            weights.ffnDownExps(), elementsPerSlice,
            (e, gate, up, down) -> expertBatch(b, xn, e, gate, up, down, start, grouped, k, efd, dim));
        if (onGpu > 0) {
            try {
                gc.finishResidentBatch(b[B_EOUT]);
            } catch (RuntimeException e) {
                System.err.println("GPU expert cache failed, using the CPU experts — " + GpuFailureException.describe(e));
                gpuExperts = null;
                nCpu = 0;
                for (int i = 0; i < nUsed; i++) if (state.prefillOnGpu[used[i]]) cpuUsed[nCpu++] = used[i];
                expertViews.forEachExpert(expertCache, layer, cpuUsed, nCpu, weights.ffnGateExps(), weights.ffnUpExps(),
                    weights.ffnDownExps(), elementsPerSlice,
                    (e2, gate, up, down) -> expertBatch(b, xn, e2, gate, up, down, start, grouped, k, efd, dim));
            }
        }

        // 4. Per token: weighted sum of its expert outputs in slot order, then the shared expert
        for (int t = 0; t < n; t++) {
            Arrays.fill(out[t], 0, dim, 0f);
            for (int j = 0; j < k; j++) {
                VectorOpsFactory.get().saxpy(selW[t * k + j], b[B_EOUT][t * k + j], 0, out[t], 0, dim);
            }
            Arrays.fill(b[B_SHB][t], 0, sharedFfn, 0f);
            Arrays.fill(b[B_SHB2][t], 0, sharedFfn, 0f);
        }
        FloatTensor.fusedGateUpBatchParallel(weights.ffnGateShexp(), weights.ffnUpShexp(), xn,
            b[B_SHB], b[B_SHB2], n, sharedFfn, dim);
        for (int t = 0; t < n; t++) {
            VectorOpsFactory.get().silu(b[B_SHB][t], sharedFfn);
            VectorOpsFactory.get().elementwiseMul(b[B_SHB][t], b[B_SHB2][t], b[B_SHB][t], sharedFfn);
            Arrays.fill(b[B_SHOUT][t], 0, dim, 0f);
        }
        FloatTensor.matmulBatchParallel(weights.ffnDownShexp(), b[B_SHB], b[B_SHOUT], n, dim, sharedFfn);
        for (int t = 0; t < n; t++) {
            VectorOpsFactory.get().accumulate(out[t], b[B_SHOUT][t], dim);
        }
    }

    /** One routed expert over all its (token, slot) pairs of the chunk: gate, up, SiLU, down. */
    private void expertBatch(float[][][] b, float[][] xn, int e, FloatTensor wGate, FloatTensor wUp,
                             FloatTensor wDown, int[] start, int[] grouped, int k, int efd, int dim) {
        int from = start[e], m = start[e + 1] - from;
        float[][] in = new float[m][], gate = new float[m][], up = new float[m][], out = new float[m][];
        for (int i = 0; i < m; i++) {
            int s = grouped[from + i];
            in[i] = xn[s / k];
            gate[i] = b[B_EGATE][s];
            up[i] = b[B_EUP][s];
            out[i] = b[B_EOUT][s];
            Arrays.fill(gate[i], 0, efd, 0f);
            Arrays.fill(up[i], 0, efd, 0f);
            Arrays.fill(out[i], 0, dim, 0f);
        }
        wGate.matmulRowsBatch(in, gate, m, 0, efd, dim);
        wUp.matmulRowsBatch(in, up, m, 0, efd, dim);
        for (int i = 0; i < m; i++) {
            VectorOpsFactory.get().silu(gate[i], efd);
            VectorOpsFactory.get().elementwiseMul(gate[i], up[i], gate[i], efd);
        }
        wDown.matmulRowsBatch(gate, out, m, 0, dim, efd);
    }

    private float[][][] batchBuffers(DeepSeek2State state, int cap, int efd, int sharedFfn) {
        int dim = config.embeddingLength();
        int slots = cap * Math.max(1, config.expertUsedCount());
        if (state.moePrefill == null || state.moePrefill[B_SHOUT].length < cap) {
            state.moePrefill = new float[][][] {
                new float[slots][efd], new float[slots][efd], new float[slots][dim],
                new float[cap][sharedFfn], new float[cap][sharedFfn], new float[cap][dim]
            };
            int experts = Math.max(1, config.expertCount());
            state.prefillExperts = new int[slots];
            state.prefillWeights = new float[slots];
            state.prefillGroupStart = new int[experts + 1];
            state.prefillGroupSlots = new int[slots];
            state.prefillUsed = new int[experts];
            state.prefillCpuUsed = new int[experts];
            state.prefillOnGpu = new boolean[experts];
        }
        return state.moePrefill;
    }

    /**
     * See {@link FloatTensor#warmUpRows}: the single-token kernels decode will use — the shared
     * expert's row kernel and the routed experts' per-row {@code dot} ({@link #expertRows}).
     */
    void warmUp(DeepSeek2LayerWeights weights) {
        int dim = config.embeddingLength();
        int efd = config.expertFfnLength();
        int sharedFfn = config.expertSharedCount() * efd;
        FloatTensor.warmUpDot(weights.ffnGateExps(), efd, dim);
        FloatTensor.warmUpDot(weights.ffnUpExps(), efd, dim);
        FloatTensor.warmUpDot(weights.ffnDownExps(), dim, efd);
        FloatTensor.warmUpRows(weights.ffnGateShexp(), sharedFfn, dim);
        FloatTensor.warmUpRows(weights.ffnUpShexp(), sharedFfn, dim);
        FloatTensor.warmUpRows(weights.ffnDownShexp(), dim, sharedFfn);
    }

    /**
     * Router for one token: logits from {@code input}, gating, top-K selection and weight
     * normalisation into {@code state.selectedExperts} / {@code state.selectedWeights}.
     */
    private void route(DeepSeek2State state, DeepSeek2LayerWeights weights, float[] input) {
        int dim = config.embeddingLength();
        int expertCount = config.expertCount();
        int expertUsedCount = MoERouting.effectiveTopK(config.expertUsedCount());

        // 1. Router: compute raw expert logits
        float[] routerLogits = state.routerLogits;
        Arrays.fill(routerLogits, 0, expertCount, 0f);
        weights.ffnGateInp().matmul(input, routerLogits, expertCount, dim);

        // 2. Gating function → probs (UNBIASED). Overwrites routerLogits in-place.
        //    See llama.cpp build_moe_ffn (llama-graph.cpp ~1230): sigmoid/softmax is applied
        //    to raw logits BEFORE any bias. The resulting `probs` is preserved unbiased for
        //    later use as the mix weights.
        int gatingFunc = config.expertGatingFunc();
        if (gatingFunc == 2) {
            // Sigmoid gating (DeepSeek-V3, GLM-4.7-Flash)
            for (int i = 0; i < expertCount; i++) {
                routerLogits[i] = 1.0f / (1.0f + (float) Math.exp(-routerLogits[i]));
            }
        } else {
            // Softmax gating (default, DeepSeek-V2)
            VectorOpsFactory.get().softmax(routerLogits, 0, expertCount);
        }

        // 3. Build selection scores: selection_probs = probs + exp_probs_b (DS-V3 only).
        //    The biased scores are used ONLY for top-K selection; the unbiased probs drive
        //    the mix weights. See llama.cpp:1249-1255: "leave probs unbiased as it's later
        //    used to get expert weights".
        float[] selectionScores;
        if (weights.expProbsBias() != null) {
            selectionScores = state.selectionScores;
            FloatTensor bias = weights.expProbsBias();
            for (int i = 0; i < expertCount; i++) {
                selectionScores[i] = routerLogits[i] + bias.getFloat(i);
            }
        } else {
            selectionScores = routerLogits;
        }

        // 4. Top-K on selection_probs; mix weights come from UNBIASED probs (routerLogits).
        int[] selectedExperts = state.selectedExperts;
        float[] selectedWeights = state.selectedWeights;
        selectTopKWithWeightSource(selectionScores, routerLogits, expertCount, expertUsedCount,
            selectedExperts, selectedWeights);

        // 5. Post-selection weights as llama.cpp build_moe_ffn: with expert_weights_norm, divide by
        //    their sum (clamped at the F16 epsilon); then multiply by expert_weights_scale. This
        //    used to be an L2 normalization inside the sigmoid branch, which for GLM-4.7-Flash's
        //    top-4 made the routed contribution up to twice as large as the reference.
        if (config.expertWeightsNorm()) {
            float sum = 0f;
            for (int k = 0; k < expertUsedCount; k++) sum += selectedWeights[k];
            float inv = 1f / Math.max(sum, 6.103515625e-5f);
            for (int k = 0; k < expertUsedCount; k++) selectedWeights[k] *= inv;
        }
        float scale = config.expertWeightsScale();
        if (scale != 0f && scale != 1f) {
            for (int k = 0; k < expertUsedCount; k++) selectedWeights[k] *= scale;
        }
        // Note: DeepSeek-V2 has norm_topk_prob=false, so we do NOT renormalize weights.
        // The sub-1.0 sum acts as intentional gating.
        if (routingStats != null && routeLayer >= 0) routingStats.count(routeLayer, selectedExperts, expertUsedCount);
    }

    /** Minimum rows per parallel chunk of the routed-expert loops. */
    private static final int EXPERT_ROW_CHUNK = 16;

    /**
     * Routed experts for one token: {@code out[k] = down_e(silu(gate_e(x)) * up_e(x))} for each
     * selected expert {@code e = selected[k]}. The gate/up rows of all K experts form one parallel
     * range and the down rows a second one, so every core works on each projection.
     */
    private void routedExperts(float[] input, int[] selected, int k, int dim, int efd,
                               FloatTensor wGate, FloatTensor wUp, FloatTensor wDown,
                               float[][] gate, float[][] up, float[][] out, int skipMask) {
        if ((skipMask & ((1 << k) - 1)) == (1 << k) - 1) return;
        // F11: the parallel ranges cover only the CPU slots (bit-identical: one dot per row)
        final int[] slots = new int[k];
        int nCpu = 0;
        for (int s = 0; s < k; s++) if ((skipMask & (1 << s)) == 0) slots[nCpu++] = s;
        final int m = nCpu;
        it.denzosoft.llmplayer.tensor.MatmulPool.forRange(m * efd, EXPERT_ROW_CHUNK, (from, to) -> {
            for (int u = from; u < to; ) {
                int j = u / efd, slot = slots[j], r0 = u - j * efd, r1 = Math.min(efd, r0 + (to - u));
                int e = selected[slot];
                expertRows(wGate, input, gate[slot], e, dim, efd, r0, r1,
                    it.denzosoft.llmplayer.tensor.ExpertCache.PROJ_GATE);
                expertRows(wUp, input, up[slot], e, dim, efd, r0, r1,
                    it.denzosoft.llmplayer.tensor.ExpertCache.PROJ_UP);
                u += r1 - r0;
            }
        });
        for (int s = 0; s < k; s++) {
            if ((skipMask & (1 << s)) != 0) continue;
            VectorOpsFactory.get().silu(gate[s], efd);
            VectorOpsFactory.get().elementwiseMul(gate[s], up[s], gate[s], efd);
        }
        it.denzosoft.llmplayer.tensor.MatmulPool.forRange(m * dim, EXPERT_ROW_CHUNK, (from, to) -> {
            for (int u = from; u < to; ) {
                int j = u / dim, slot = slots[j], r0 = u - j * dim, r1 = Math.min(dim, r0 + (to - u));
                expertRows(wDown, gate[slot], out[slot], selected[slot], efd, dim, r0, r1,
                    it.denzosoft.llmplayer.tensor.ExpertCache.PROJ_DOWN);
                u += r1 - r0;
            }
        });
    }

    /**
     * Rows {@code [r0, r1)} of one expert slice of a 3D tensor ({@code [inDim, outDim, numExperts]}
     * in GGUF layout; expert e starts at {@code e * outDim * inDim}): {@code output[row] = W[row]·input}.
     */
    private void expertRows(FloatTensor weights3D, float[] input, float[] output,
                            int expert, int inDim, int outDim, int r0, int r1, int projection) {
        // When the slice is cached, it is a standalone tensor holding just this expert, so the rows
        // start at 0 instead of the expert's base offset inside the 3D tensor.
        if (cacheLayerReady) {
            FloatTensor cached = expertCache.tensorFor(currentLayer, expert, projection);
            if (cached != null) {
                for (int row = r0; row < r1; row++) {
                    output[row] = cached.dot((long) row * inDim, input, 0, inDim);
                }
                return;
            }
        }
        long expertOffset = (long) expert * outDim * inDim;
        for (int row = r0; row < r1; row++) {
            output[row] = weights3D.dot(expertOffset + (long) row * inDim, input, 0, inDim);
        }
    }

    /**
     * Select top-K indices from {@code scores} and return the corresponding values from
     * {@code weightSource}. When {@code scores == weightSource} this behaves like a classic
     * top-K (old behavior); when they differ, the top-K is decided by scores but the stored
     * values come from weightSource — used to implement DS-V3's "biased selection, unbiased
     * mix weights" pattern.
     */
    private static void selectTopKWithWeightSource(float[] scores, float[] weightSource,
                                                    int n, int k,
                                                    int[] outIndices, float[] outValues) {
        Arrays.fill(outIndices, 0, k, -1);
        // Track the min score *within the selected set* to decide replacement
        float[] selectedScores = new float[k];
        Arrays.fill(selectedScores, Float.NEGATIVE_INFINITY);
        Arrays.fill(outValues, 0, k, Float.NEGATIVE_INFINITY);

        int minPos = 0;
        float minVal = Float.NEGATIVE_INFINITY;

        for (int i = 0; i < n; i++) {
            if (scores[i] > minVal) {
                selectedScores[minPos] = scores[i];
                outIndices[minPos] = i;
                outValues[minPos] = weightSource[i];
                // Rescan for new minimum of the selected-score set
                minPos = 0;
                minVal = selectedScores[0];
                for (int j = 1; j < k; j++) {
                    if (selectedScores[j] < minVal) {
                        minPos = j;
                        minVal = selectedScores[j];
                    }
                }
            }
        }
    }
}
