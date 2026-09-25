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
    }

    public void setExpertCache(it.denzosoft.llmplayer.tensor.ExpertCache cache) {
        this.expertCache = cache;
    }

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
        int dim = config.embeddingLength();
        int expertUsedCount = MoERouting.effectiveTopK(config.expertUsedCount());
        int expertFfnDim = config.expertFfnLength();
        int sharedFfnDim = config.expertSharedCount() * expertFfnDim;

        // 1-5. Router, gating, top-K and weight normalisation
        route(state, weights, state.xb);
        int[] selectedExperts = state.selectedExperts;
        float[] selectedWeights = state.selectedWeights;

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

        // 2. Compute routed expert outputs — parallel across experts
        Arrays.fill(state.xb, 0);  // Will accumulate weighted expert outputs here

        it.denzosoft.llmplayer.tensor.MatmulPool.forEach(expertUsedCount, k -> {
            int e = selectedExperts[k];

            // Per-expert buffers (no contention)
            float[] gate = state.moeHbPerExpert[k];
            float[] up = state.moeHb2PerExpert[k];
            float[] out = state.expertOutPerExpert[k];

            Arrays.fill(gate, 0, expertFfnDim, 0f);
            Arrays.fill(up, 0, expertFfnDim, 0f);

            expertMatmul(weights.ffnGateExps(), state.xbSaved, gate, e, dim, expertFfnDim,
                it.denzosoft.llmplayer.tensor.ExpertCache.PROJ_GATE);
            expertMatmul(weights.ffnUpExps(), state.xbSaved, up, e, dim, expertFfnDim,
                it.denzosoft.llmplayer.tensor.ExpertCache.PROJ_UP);

            VectorOpsFactory.get().silu(gate, expertFfnDim);
            VectorOpsFactory.get().elementwiseMul(gate, up, gate, expertFfnDim);

            Arrays.fill(out, 0, dim, 0f);
            expertMatmul(weights.ffnDownExps(), gate, out, e, expertFfnDim, dim,
                it.denzosoft.llmplayer.tensor.ExpertCache.PROJ_DOWN);
        });

        // Sequential accumulation of weighted expert outputs
        for (int k = 0; k < expertUsedCount; k++) {
            VectorOpsFactory.get().saxpy(selectedWeights[k], state.expertOutPerExpert[k], 0, state.xb, 0, dim);
        }

        // 3. Shared expert: standard SwiGLU with shared weights
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
        for (int t = 0; t < n; t++) {
            route(state, weights, xn[t]);
            System.arraycopy(state.selectedExperts, 0, sel, t * k, k);
            System.arraycopy(state.selectedWeights, 0, selW, t * k, k);
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

        // 3. Compute the experts (grouped for the SSD cache, the next group read while one computes)
        expertViews.forEachExpert(expertCache, layer, used, nUsed, weights.ffnGateExps(), weights.ffnUpExps(),
            weights.ffnDownExps(), elementsPerSlice,
            (e, gate, up, down) -> expertBatch(b, xn, e, gate, up, down, start, grouped, k, efd, dim));

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
        }
        return state.moePrefill;
    }

    /**
     * See {@link FloatTensor#warmUpRows}: the single-token kernels decode will use — the shared
     * expert's row kernel and the routed experts' per-row {@code dot} ({@link #expertMatmul}).
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

        // 5. Post-selection weight normalization (for sigmoid gating with expert_weights_norm=true).
        if (gatingFunc == 2) {
            // L2 normalization of selected weights, then scale
            // Note: llama.cpp does sum-normalization (ggml_sum_rows + ggml_div, clamped to 6.1e-5).
            // LLMPlayer uses L2 — preserved here to avoid regressing GLM-4.7-Flash.
            // This is a separate issue (see audit M/cross-cutting MoE) to be fixed once verified.
            float l2Norm = 0f;
            for (int k = 0; k < expertUsedCount; k++) {
                l2Norm += selectedWeights[k] * selectedWeights[k];
            }
            l2Norm = (float) Math.sqrt(l2Norm);
            if (l2Norm > 6.103515625e-5f) { // F16-epsilon clamp guards against NaN
                float scale = config.expertWeightsScale() / l2Norm;
                for (int k = 0; k < expertUsedCount; k++) {
                    selectedWeights[k] *= scale;
                }
            }
        }
        // Note: DeepSeek-V2 has norm_topk_prob=false, so we do NOT renormalize weights.
        // The sub-1.0 sum acts as intentional gating.
    }

    /**
     * Matrix-vector multiply for a single expert slice from a 3D tensor.
     * The 3D tensor has shape [inDim, outDim, numExperts] in GGUF layout.
     * Expert e starts at offset e * outDim * inDim.
     */
    private void expertMatmul(FloatTensor weights3D, float[] input, float[] output,
                              int expert, int inDim, int outDim, int projection) {
        // When the slice is cached, it is a standalone tensor holding just this expert, so the rows
        // start at 0 instead of the expert's base offset inside the 3D tensor.
        if (cacheLayerReady) {
            FloatTensor cached = expertCache.tensorFor(currentLayer, expert, projection);
            if (cached != null) {
                for (int row = 0; row < outDim; row++) {
                    output[row] += cached.dot((long) row * inDim, input, 0, inDim);
                }
                return;
            }
        }
        long expertOffset = (long) expert * outDim * inDim;
        for (int row = 0; row < outDim; row++) {
            output[row] += weights3D.dot(expertOffset + (long) row * inDim, input, 0, inDim);
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
