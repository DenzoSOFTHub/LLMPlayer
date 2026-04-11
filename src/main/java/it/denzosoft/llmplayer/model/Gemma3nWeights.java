package it.denzosoft.llmplayer.model;

import it.denzosoft.llmplayer.tensor.FloatTensor;

/**
 * Bundle of all Gemma 3n / Gemma 4-PLE-specific tensors that are NOT in the standard
 * {@link ModelWeights} or {@link TransformerLayerWeights}. Loaded once by
 * {@code LLMEngine.createGemma4Engine} and passed to {@code Gemma4InferenceEngine} as a
 * single optional argument.
 *
 * <p>Layout (matches llama.cpp gemma3n-iswa.cpp):
 *
 * <h2>Per-Layer Embeddings (PLE)</h2>
 * <ul>
 *   <li>{@link #pleTokenEmbd} — {@code per_layer_token_embd.weight}, shape
 *       {@code [n_embd_altup * n_layer, vocab_size]}. Per-token, per-layer "identity"
 *       embedding looked up at the start of each forward pass.
 *   <li>{@link #pleModelProj} — {@code per_layer_model_proj.weight}, shape
 *       {@code [n_embd, n_embd_altup * n_layer]}. Projects the main embedding into the
 *       per-layer space.
 *   <li>{@link #pleProjNormWeights} — {@code per_layer_proj_norm.weight}, shape
 *       {@code [n_embd_altup]}. RMSNorm weights applied to each per-layer slice.
 *   <li>{@link #perLayerInpGate} — per-layer {@code inp_gate.weight}, shape
 *       {@code [n_embd, n_embd_altup]}. Inside each layer, projects active stream from
 *       n_embd → n_embd_altup so first_prediction can be gated by the per-layer input.
 *   <li>{@link #perLayerProj} — per-layer {@code proj.weight}, shape
 *       {@code [n_embd_altup, n_embd]}. Projects first_prediction back from n_embd_altup
 *       → n_embd so it can be added to the non-active altup streams.
 *   <li>{@link #perLayerPostNorm} — per-layer {@code post_norm.weight}, shape
 *       {@code [n_embd]}. RMSNorm applied to first_prediction after the back-projection.
 * </ul>
 *
 * <h2>AltUp (4 parallel activation streams)</h2>
 * <ul>
 *   <li>{@link #altupProj} — {@code altup_proj.weight}, shape
 *       {@code [n_embd, n_embd, n_altup-1]}. Projects the initial active stream into the
 *       n_altup-1 "shadow" streams during pre-layer initialization.
 *   <li>{@link #altupUnembdProj} — {@code altup_unembd_proj.weight}, shape
 *       {@code [n_embd, n_embd, n_altup-1]}. Projects the n_altup-1 shadow streams BACK
 *       into the active-stream space after the last layer, before averaging.
 *   <li>{@link #altupRouter} — per-layer {@code altup_router.weight}, shape
 *       {@code [n_embd, n_altup]}. Inside the layer, projects the active stream into
 *       n_altup "modalities" used by altup_predict and altup_correct.
 *   <li>{@link #altupRouterNorm} — per-layer {@code altup_router_norm.weight}, shape
 *       {@code [n_embd]}. RMSNorm applied to the active stream BEFORE the router projection.
 *   <li>{@link #altupPredictCoef} — per-layer {@code altup_predict_coef.weight}, shape
 *       {@code [n_altup, n_altup * n_altup]}. Modalities → predict coefficients used to
 *       linearly combine the n_altup streams to produce the next-state predictions.
 *   <li>{@link #altupCorrectCoef} — per-layer {@code altup_correct_coef.weight}, shape
 *       {@code [n_altup, n_altup]}. Modalities → correct coefficients (then +1.0) used to
 *       weight the innovation (actual layer output - predicted active stream) when
 *       updating the predictions for all streams.
 *   <li>{@link #altupCorrectScale} — per-layer {@code altup_correct_scale.weight}, shape
 *       {@code [n_embd]}. Per-dim scale applied to the corrected active stream BEFORE
 *       the per-layer first_prediction projection.
 * </ul>
 *
 * <h2>Laurel (low-rank residual branch)</h2>
 * <ul>
 *   <li>{@link #laurelL} — per-layer {@code laurel_l.weight}, shape
 *       {@code [n_embd, laurel_rank]}. Left low-rank projection (e.g. n_embd=2048,
 *       laurel_rank=64).
 *   <li>{@link #laurelR} — per-layer {@code laurel_r.weight}, shape
 *       {@code [laurel_rank, n_embd]}. Right low-rank back-projection.
 *   <li>{@link #laurelPostNorm} — per-layer {@code laurel_post_norm.weight}, shape
 *       {@code [n_embd]}. RMSNorm applied to the laurel output before the final residual.
 * </ul>
 *
 * <h2>Architectural constants (read from {@link ModelConfig} / GGUF metadata)</h2>
 * <ul>
 *   <li>{@code n_embd_altup} — {@code embedding_length_per_layer} (= 256 for E4B)
 *   <li>{@code n_altup} — number of altup streams (= 4 for Gemma 3n)
 *   <li>{@code i_altup_act} — active altup stream index (= 0 for Gemma 3n)
 *   <li>{@code n_layer_sparsity} — number of leading FFN layers that use Gaussian top-k
 *       activation sparsity (typically 10 of 35 for E4B)
 *   <li>{@code f_sparsity_std_mul} — std multiplier for the sparsity cutoff (typically
 *       1.6448536 ≈ z-score for ~5%)
 *   <li>{@link #hasKv} — per-layer flag, true when this layer projects its own K/V vs
 *       reusing an earlier layer's KV cache (Gemma 3n: layers >= n_layer_kv_from_start
 *       reuse earlier layers' KV)
 * </ul>
 */
public final class Gemma3nWeights {
    // PLE
    public final FloatTensor pleTokenEmbd;
    public final FloatTensor pleModelProj;
    public final float[] pleProjNormWeights;
    public final FloatTensor[] perLayerInpGate;
    public final FloatTensor[] perLayerProj;
    public final float[][] perLayerPostNorm;

    // AltUp globals
    public final FloatTensor altupProj;
    public final FloatTensor altupUnembdProj;

    // AltUp per-layer
    public final FloatTensor[] altupRouter;
    public final float[][] altupRouterNorm;
    public final FloatTensor[] altupPredictCoef;
    public final FloatTensor[] altupCorrectCoef;
    public final float[][] altupCorrectScale;

    // Laurel per-layer
    public final FloatTensor[] laurelL;
    public final FloatTensor[] laurelR;
    public final float[][] laurelPostNorm;

    // Constants
    public final int nAltup;
    public final int iAltupAct;
    public final int nLayerSparsity;
    public final float fSparsityStdMul;
    public final boolean[] hasKv;

    public Gemma3nWeights(
            FloatTensor pleTokenEmbd, FloatTensor pleModelProj, float[] pleProjNormWeights,
            FloatTensor[] perLayerInpGate, FloatTensor[] perLayerProj, float[][] perLayerPostNorm,
            FloatTensor altupProj, FloatTensor altupUnembdProj,
            FloatTensor[] altupRouter, float[][] altupRouterNorm,
            FloatTensor[] altupPredictCoef, FloatTensor[] altupCorrectCoef, float[][] altupCorrectScale,
            FloatTensor[] laurelL, FloatTensor[] laurelR, float[][] laurelPostNorm,
            int nAltup, int iAltupAct, int nLayerSparsity, float fSparsityStdMul, boolean[] hasKv) {
        this.pleTokenEmbd = pleTokenEmbd;
        this.pleModelProj = pleModelProj;
        this.pleProjNormWeights = pleProjNormWeights;
        this.perLayerInpGate = perLayerInpGate;
        this.perLayerProj = perLayerProj;
        this.perLayerPostNorm = perLayerPostNorm;
        this.altupProj = altupProj;
        this.altupUnembdProj = altupUnembdProj;
        this.altupRouter = altupRouter;
        this.altupRouterNorm = altupRouterNorm;
        this.altupPredictCoef = altupPredictCoef;
        this.altupCorrectCoef = altupCorrectCoef;
        this.altupCorrectScale = altupCorrectScale;
        this.laurelL = laurelL;
        this.laurelR = laurelR;
        this.laurelPostNorm = laurelPostNorm;
        this.nAltup = nAltup;
        this.iAltupAct = iAltupAct;
        this.nLayerSparsity = nLayerSparsity;
        this.fSparsityStdMul = fSparsityStdMul;
        this.hasKv = hasKv;
    }

    public boolean isFullyLoaded() {
        return pleTokenEmbd != null && pleModelProj != null
            && altupProj != null && altupUnembdProj != null
            && altupRouter != null && altupRouter.length > 0 && altupRouter[0] != null
            && laurelL != null && laurelL.length > 0 && laurelL[0] != null;
    }
}
