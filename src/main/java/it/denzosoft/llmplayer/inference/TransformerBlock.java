package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.model.TransformerLayerWeights;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

/**
 * A single transformer block supporting three normalization modes:
 * 1. Pre-norm (standard): attn_norm -> attention -> residual -> ffn_norm -> ffn -> residual
 * 2. Pre+Post-norm (GLM4/Gemma2): adds post-attention and post-FFN norms
 * 3. Post-norm only (OLMo2): attention -> post_attn_norm -> residual -> ffn -> post_ffn_norm -> residual
 * 4. Parallel FFN (Command-R): attn_norm -> {attention || ffn} -> residual (no ffn_norm)
 */
public class TransformerBlock {

    private final ModelConfig config;
    private final Attention attention;
    private final SwiGLUFFN ffn;
    private final float[][] cachedAttnNorm;
    private final float[][] cachedFfnNorm;
    private final float[][] cachedPostAttnNorm; // null entries if not used
    private final float[][] cachedPostFfnNorm;  // null entries if not used

    private final float residualScale; // Granite: 0.22 (scale before residual add)

    // CPU profiling (enabled via -Dcpu.profile=true)
    private final boolean cpuProfile;
    private long profileAttnNormNs, profileAttnNs, profileFfnNormNs, profileFfnNs;
    private long profilePostAttnNormNs, profilePostFfnNormNs, profileResidualNs;
    private int profileTokenCount;

    public TransformerBlock(ModelConfig config, Attention attention, SwiGLUFFN ffn,
                            TransformerLayerWeights[] allLayers) {
        this.config = config;
        this.attention = attention;
        this.ffn = ffn;
        this.residualScale = config.residualScale();
        this.cpuProfile = "true".equals(System.getProperty("cpu.profile"));

        int dim = config.embeddingLength();
        int blockCount = allLayers.length;
        this.cachedAttnNorm = new float[blockCount][];
        this.cachedFfnNorm = new float[blockCount][];
        this.cachedPostAttnNorm = new float[blockCount][];
        this.cachedPostFfnNorm = new float[blockCount][];
        for (int i = 0; i < blockCount; i++) {
            if (allLayers[i].attnNorm() != null) {
                cachedAttnNorm[i] = RMSNorm.cacheWeights(allLayers[i].attnNorm(), dim);
            }
            if (allLayers[i].ffnNorm() != null) {
                cachedFfnNorm[i] = RMSNorm.cacheWeights(allLayers[i].ffnNorm(), dim);
            }
            if (allLayers[i].postAttnNorm() != null) {
                cachedPostAttnNorm[i] = RMSNorm.cacheWeights(allLayers[i].postAttnNorm(), dim);
            }
            if (allLayers[i].postFfnNorm() != null) {
                cachedPostFfnNorm[i] = RMSNorm.cacheWeights(allLayers[i].postFfnNorm(), dim);
            }
        }
    }

    public void forward(InferenceState state, TransformerLayerWeights weights, int layer, int position) {
        int dim = config.embeddingLength();
        long t0 = 0, t1;

        // Parallel FFN mode (Command-R): no ffn_norm, FFN uses same pre-norm as attention
        boolean parallelFfn = cachedFfnNorm[layer] == null && cachedAttnNorm[layer] != null;

        // Pre-attention norm (standard/GLM4/Gemma2/Command-R) or pass-through (OLMo2)
        if (cpuProfile) t0 = System.nanoTime();
        if (cachedAttnNorm[layer] != null) {
            RMSNorm.apply(state.xb, state.x, cachedAttnNorm[layer], dim, config.normEps());
        } else {
            System.arraycopy(state.x, 0, state.xb, 0, dim);
        }
        if (cpuProfile) { t1 = System.nanoTime(); profileAttnNormNs += t1 - t0; t0 = t1; }

        // Save pre-norm output for parallel FFN (reuse logits buffer, always large enough)
        if (parallelFfn) {
            System.arraycopy(state.xb, 0, state.logits, 0, dim);
        }

        // Attention (writes to xb via output projection)
        attention.forward(state, weights, layer, position);
        if (cpuProfile) { t1 = System.nanoTime(); profileAttnNs += t1 - t0; t0 = t1; }

        // Post-attention norm (GLM4/Gemma2/OLMo2)
        if (cachedPostAttnNorm[layer] != null) {
            RMSNorm.apply(state.xb, state.xb, cachedPostAttnNorm[layer], dim, config.normEps());
            if (cpuProfile) { t1 = System.nanoTime(); profilePostAttnNormNs += t1 - t0; t0 = t1; }
        }

        // Granite residual scaling: output *= residualScale before adding to residual stream
        if (residualScale > 0f) {
            for (int i = 0; i < dim; i++) state.xb[i] *= residualScale;
        }

        // Residual connection: x += attention_output
        VectorOpsFactory.get().accumulate(state.x, state.xb, dim);
        if (cpuProfile) { t1 = System.nanoTime(); profileResidualNs += t1 - t0; t0 = t1; }

        // FFN input: depends on mode
        if (parallelFfn) {
            // Restore pre-norm output for FFN (Command-R parallel mode)
            System.arraycopy(state.logits, 0, state.xb, 0, dim);
        } else if (cachedFfnNorm[layer] != null) {
            // Standard pre-FFN norm
            RMSNorm.apply(state.xb, state.x, cachedFfnNorm[layer], dim, config.normEps());
        } else {
            // Post-norm only (OLMo2): FFN takes x directly
            System.arraycopy(state.x, 0, state.xb, 0, dim);
        }
        if (cpuProfile) { t1 = System.nanoTime(); profileFfnNormNs += t1 - t0; t0 = t1; }

        // FFN (writes to xb)
        ffn.forward(state, weights);
        if (cpuProfile) { t1 = System.nanoTime(); profileFfnNs += t1 - t0; t0 = t1; }

        // Post-FFN norm (GLM4/Gemma2/OLMo2)
        if (cachedPostFfnNorm[layer] != null) {
            RMSNorm.apply(state.xb, state.xb, cachedPostFfnNorm[layer], dim, config.normEps());
            if (cpuProfile) { t1 = System.nanoTime(); profilePostFfnNormNs += t1 - t0; t0 = t1; }
        }

        // Granite residual scaling for FFN output
        if (residualScale > 0f) {
            for (int i = 0; i < dim; i++) state.xb[i] *= residualScale;
        }

        // Residual connection: x += ffn_output
        VectorOpsFactory.get().accumulate(state.x, state.xb, dim);
        if (cpuProfile) { t1 = System.nanoTime(); profileResidualNs += t1 - t0; }

        // Print profiling summary every N tokens (at last layer)
        if (cpuProfile && layer == config.blockCount() - 1) {
            profileTokenCount++;
            if (profileTokenCount % 10 == 0) {
                printProfile();
            }
        }
    }

    private void printProfile() {
        int n = profileTokenCount;
        int layers = config.blockCount();
        double scale = 1e6 * layers; // ns to ms, divided by layers (profiling sums all layers)
        System.out.printf("[cpu-profile] %d tokens, per-token avg (ms): attn_norm=%.1f attn=%.1f ffn_norm=%.1f ffn=%.1f residual=%.1f",
            n, profileAttnNormNs / scale / n, profileAttnNs / scale / n,
            profileFfnNormNs / scale / n, profileFfnNs / scale / n,
            profileResidualNs / scale / n);
        if (profilePostAttnNormNs > 0) {
            System.out.printf(" post_attn_norm=%.1f", profilePostAttnNormNs / scale / n);
        }
        if (profilePostFfnNormNs > 0) {
            System.out.printf(" post_ffn_norm=%.1f", profilePostFfnNormNs / scale / n);
        }
        double totalMs = (profileAttnNormNs + profileAttnNs + profileFfnNormNs + profileFfnNs
            + profileResidualNs + profilePostAttnNormNs + profilePostFfnNormNs) / 1e6 / n;
        System.out.printf(" | total=%.1f ms/tok%n", totalMs);
    }
}
