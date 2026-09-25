package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.DeepSeek2LayerWeights;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Multi-head Latent Attention (MLA) for DeepSeek2 and GLM-4.7-Flash.
 *
 * Standard MLA (DeepSeek-V2):
 * 1. Q = wq * x → [headCount * keyLength]
 * 2. c_kv = wkvA * x → [kvLoraRank + ropeDim]
 * 3. Split: c_latent [kvLoraRank], k_rope [ropeDim]
 * 4. c_latent_norm = RMSNorm(c_latent)
 * 5. [K_nope, V] = wkvB * c_latent_norm → [headCount * (keyNope + valueLen)]
 * 6. K = concat(K_nope, broadcast(RoPE(k_rope))) per head
 * 7. Standard attention: softmax(Q*K^T / sqrt(keyLen)) * V
 *
 * Q-LoRA variant (GLM-4.7-Flash / DeepSeek-V3):
 * 1. q_compressed = wqA * x → [qLoraRank]
 * 2. q_compressed_norm = RMSNorm(q_compressed)
 * 3. Q = wqB * q_compressed_norm → [headCount * keyLength]
 *
 * Separate K_B/V_B variant (GLM-4.7-Flash / DeepSeek-V3):
 * Instead of combined wkvB → [K_nope, V], uses:
 * - K_nope[h] = c_latent * wkB[h]^T  (transposed: wkB shape [keyNope, kvLoraRank, headCount])
 * - V[h] = wvB[h] * c_latent          (standard: wvB shape [kvLoraRank, valueLen, headCount])
 */
public class MLAAttention {

    private final ModelConfig config;
    private final RoPE rope;  // operates on ropeDimCount dimensions
    private final float[][] cachedKvANorm;  // [layer][kvLoraRank]
    private final float[][] cachedQANorm;   // [layer][qLoraRank] (null when no Q-LoRA)

    public MLAAttention(ModelConfig config, RoPE rope, DeepSeek2LayerWeights[] allLayers) {
        this.config = config;
        this.rope = rope;

        int blockCount = allLayers.length;
        this.cachedKvANorm = new float[blockCount][];
        for (int i = 0; i < blockCount; i++) {
            cachedKvANorm[i] = RMSNorm.cacheWeights(allLayers[i].kvANorm(), config.kvLoraRank());
        }

        // Cache Q-LoRA norm weights if present
        if (allLayers[0].hasQLoRA()) {
            this.cachedQANorm = new float[blockCount][];
            for (int i = 0; i < blockCount; i++) {
                cachedQANorm[i] = RMSNorm.cacheWeights(allLayers[i].wqANorm(), config.qLoraRank());
            }
        } else {
            this.cachedQANorm = null;
        }
    }

    /**
     * Forward pass for MLA attention at a single position.
     * Uses DeepSeek2State which has MLA-specific buffers.
     */
    public void forward(DeepSeek2State state, DeepSeek2LayerWeights weights, int layer, int position) {
        int dim = config.embeddingLength();
        int headCount = config.headCount();
        int keyLength = config.keyLength();
        int valueLength = config.valueLength();
        int kvLoraRank = config.kvLoraRank();
        int ropeDim = config.ropeDimensionCount();
        int keyNope = keyLength - ropeDim;
        int kvCompressedDim = kvLoraRank + ropeDim;
        int totalQDim = headCount * keyLength;
        int totalValDim = headCount * valueLength;

        // 1. Q projection
        if (weights.hasQLoRA()) {
            // Q-LoRA: x → qCompressed → norm → Q
            int qLoraRank = config.qLoraRank();
            Arrays.fill(state.qCompressed, 0, qLoraRank, 0f);
            weights.wqA().matmulParallel(state.xb, state.qCompressed, qLoraRank, dim);

            // RMSNorm on qCompressed
            RMSNorm.apply(state.qCompressedNorm, state.qCompressed, cachedQANorm[layer], qLoraRank, config.normEps());

            // Q = wqB * qCompressedNorm
            Arrays.fill(state.q, 0, totalQDim, 0f);
            weights.wqB().matmulParallel(state.qCompressedNorm, state.q, totalQDim, qLoraRank);
        } else {
            // Direct: x → Q
            Arrays.fill(state.q, 0, totalQDim, 0f);
            weights.wq().matmulParallel(state.xb, state.q, totalQDim, dim);
        }

        // 2. KV compression: x → c_kv [kvLoraRank + ropeDim] = [c_latent, k_rope_raw]
        Arrays.fill(state.kvCompressed, 0, kvCompressedDim, 0f);
        weights.wkvA().matmulParallel(state.xb, state.kvCompressed, kvCompressedDim, dim);

        // 3-4. RMSNorm on c_latent (the first kvLoraRank elements)
        normLatent(state.kvCompressed, state.kvLatentNormed, layer);

        // 5. KV decompression → K_nope and V per head
        if (weights.hasSeparateKVB()) {
            // Separate K_B (transposed) and V_B (standard) per head
            decompressKVSeparate(state, weights, headCount, keyNope, valueLength, kvLoraRank);
        } else {
            // Combined wkvB → [headCount * (keyNope + valueLen)]
            int kvBOutDim = headCount * (keyNope + valueLength);
            Arrays.fill(state.kvDecompressed, 0, kvBOutDim, 0f);
            weights.wkvB().matmulParallel(state.kvLatentNormed, state.kvDecompressed, kvBOutDim, kvLoraRank);
            splitKV(state.kvDecompressed, state);
        }

        // 6-8. RoPE, KV store, attention → xb2
        attentionCore(state, layer, position);

        // 9. Output projection: xb = Wo * xb2
        Arrays.fill(state.xb, 0);
        weights.wo().matmulParallel(state.xb2, state.xb, dim, totalValDim);
    }

    /** RMSNorm of the latent part of {@code kvCompressed} into {@code dst}. */
    private void normLatent(float[] kvCompressed, float[] dst, int layer) {
        int kvLoraRank = config.kvLoraRank();
        float[] kvNormWeights = cachedKvANorm[layer];
        float ss = 0f;
        for (int i = 0; i < kvLoraRank; i++) {
            ss += kvCompressed[i] * kvCompressed[i];
        }
        ss = 1.0f / (float) Math.sqrt(ss / kvLoraRank + config.normEps());
        for (int i = 0; i < kvLoraRank; i++) {
            dst[i] = kvCompressed[i] * ss * kvNormWeights[i];
        }
    }

    /** Extract K_nope and V per head from the interleaved output of the combined wkvB. */
    private void splitKV(float[] kvDecompressed, DeepSeek2State state) {
        int headCount = config.headCount();
        int keyLength = config.keyLength();
        int valueLength = config.valueLength();
        int keyNope = keyLength - config.ropeDimensionCount();
        int kvBOutPerHead = keyNope + valueLength;
        for (int h = 0; h < headCount; h++) {
            int kvBSrc = h * kvBOutPerHead;
            System.arraycopy(kvDecompressed, kvBSrc, state.k, h * keyLength, keyNope);
            System.arraycopy(kvDecompressed, kvBSrc + keyNope, state.v, h * valueLength, valueLength);
        }
    }

    /**
     * The order-dependent part of MLA for one token: RoPE on k_rope (from {@code state.kvCompressed})
     * and on each head's Q rope part, K assembly, KV store and attention over the cache. Reads
     * {@code state.q}, {@code state.kvCompressed} and the K_nope/V parts of {@code state.k/v}; writes
     * the attention output, before Wo, to {@code state.xb2}.
     */
    private void attentionCore(DeepSeek2State state, int layer, int position) {
        int headCount = config.headCount();
        int keyLength = config.keyLength();
        int valueLength = config.valueLength();
        int kvLoraRank = config.kvLoraRank();
        int ropeDim = config.ropeDimensionCount();
        int keyNope = keyLength - ropeDim;
        int totalKeyDim = headCount * keyLength;
        int totalValDim = headCount * valueLength;

        // 6. Assemble full K per head: [K_nope, k_rope]
        // Copy k_rope to temp and apply RoPE
        System.arraycopy(state.kvCompressed, kvLoraRank, state.kRopeTemp, 0, ropeDim);
        rope.apply(state.kRopeTemp, 0, position);

        // Append k_rope (shared across all heads) after K_nope
        for (int h = 0; h < headCount; h++) {
            System.arraycopy(state.kRopeTemp, 0, state.k, h * keyLength + keyNope, ropeDim);
        }

        // Apply RoPE to Q_rope parts: each head's Q has [nope, rope]
        for (int h = 0; h < headCount; h++) {
            int qRopeOffset = h * keyLength + keyNope;
            rope.apply(state.q, qRopeOffset, position);
        }

        // 7. Store K and V in cache (transparently quantized if -Dkv.q8=true)
        state.kvCache.storeK(layer, position, state.k, totalKeyDim);
        state.kvCache.storeV(layer, position, state.v, totalValDim);

        // 8. Attention computation - parallel over heads
        float mscale = rope.getMscale();
        final float scaleFactor = mscale * mscale / (float) Math.sqrt(keyLength);

        Arrays.fill(state.xb2, 0, totalValDim, 0f);

        final KVCache kv = state.kvCache;
        final int layerFinal = layer;
        final int positionFinal = position;
        final int keyLengthFinal = keyLength;
        final int valueLengthFinal = valueLength;

        it.denzosoft.llmplayer.tensor.MatmulPool.forEach(headCount, h -> {
            int attOffset = h * (positionFinal + 1);
            int qOffset = h * keyLengthFinal;
            int kHeadOff = h * keyLengthFinal;  // per-head K slice
            int vHeadOff = h * valueLengthFinal; // per-head V slice
            int outOffset = h * valueLengthFinal;

            // Compute attention scores
            for (int t = 0; t <= positionFinal; t++) {
                float score = kv.dotK(layerFinal, t, kHeadOff, keyLengthFinal, state.q, qOffset);
                state.att[attOffset + t] = score * scaleFactor;
            }

            // Softmax
            VectorOpsFactory.get().softmax(state.att, attOffset, positionFinal + 1);

            // Weighted sum of values
            for (int t = 0; t <= positionFinal; t++) {
                float a = state.att[attOffset + t];
                kv.saxpyV(layerFinal, t, vHeadOff, valueLengthFinal, a, state.xb2, outOffset);
            }
        });
    }

    // ==================== Batched prefill ====================

    // Indices into DeepSeek2State.mlaPrefill
    private static final int M_Q = 0, M_QC = 1, M_QCN = 2, M_KVC = 3, M_LAT = 4, M_KVD = 5, M_ATT = 6;

    /**
     * Multi-token {@link #forward} for the chunk's tokens at one layer: {@code out[t] = Wo * attn(xn[t])}.
     * Every projection that depends on the token alone — Q (or Q-LoRA A, norm, B), wkvA, the latent
     * norm and the combined wkvB — runs as a multi-token matmul; {@link #attentionCore} and the
     * separate K_B/V_B decompression run token by token in position order; Wo is batched again.
     */
    public void forwardBatch(DeepSeek2State state, DeepSeek2LayerWeights weights, int layer, int basePos,
                             int n, float[][] xn, float[][] out) {
        int dim = config.embeddingLength();
        int headCount = config.headCount();
        int keyLength = config.keyLength();
        int valueLength = config.valueLength();
        int kvLoraRank = config.kvLoraRank();
        int ropeDim = config.ropeDimensionCount();
        int keyNope = keyLength - ropeDim;
        int kvCompressedDim = kvLoraRank + ropeDim;
        int totalQDim = headCount * keyLength;
        int totalValDim = headCount * valueLength;
        int kvBOutDim = headCount * (keyNope + valueLength);
        int qLoraRank = Math.max(0, config.qLoraRank());
        float[][][] b = batchBuffers(state, xn.length, totalQDim, qLoraRank, kvCompressedDim, kvLoraRank,
            kvBOutDim, totalValDim);

        // Q
        for (int t = 0; t < n; t++) Arrays.fill(b[M_Q][t], 0, totalQDim, 0f);
        if (weights.hasQLoRA()) {
            for (int t = 0; t < n; t++) Arrays.fill(b[M_QC][t], 0, qLoraRank, 0f);
            FloatTensor.matmulBatchParallel(weights.wqA(), xn, b[M_QC], n, qLoraRank, dim);
            for (int t = 0; t < n; t++) {
                RMSNorm.apply(b[M_QCN][t], b[M_QC][t], cachedQANorm[layer], qLoraRank, config.normEps());
            }
            FloatTensor.matmulBatchParallel(weights.wqB(), b[M_QCN], b[M_Q], n, totalQDim, qLoraRank);
        } else {
            FloatTensor.matmulBatchParallel(weights.wq(), xn, b[M_Q], n, totalQDim, dim);
        }

        // KV compression, latent norm, combined decompression
        for (int t = 0; t < n; t++) Arrays.fill(b[M_KVC][t], 0, kvCompressedDim, 0f);
        FloatTensor.matmulBatchParallel(weights.wkvA(), xn, b[M_KVC], n, kvCompressedDim, dim);
        for (int t = 0; t < n; t++) normLatent(b[M_KVC][t], b[M_LAT][t], layer);
        boolean combined = !weights.hasSeparateKVB();
        if (combined) {
            for (int t = 0; t < n; t++) Arrays.fill(b[M_KVD][t], 0, kvBOutDim, 0f);
            FloatTensor.matmulBatchParallel(weights.wkvB(), b[M_LAT], b[M_KVD], n, kvBOutDim, kvLoraRank);
        }

        // Token by token in position order
        for (int t = 0; t < n; t++) {
            System.arraycopy(b[M_Q][t], 0, state.q, 0, totalQDim);
            System.arraycopy(b[M_KVC][t], 0, state.kvCompressed, 0, kvCompressedDim);
            if (combined) {
                splitKV(b[M_KVD][t], state);
            } else {
                System.arraycopy(b[M_LAT][t], 0, state.kvLatentNormed, 0, kvLoraRank);
                decompressKVSeparate(state, weights, headCount, keyNope, valueLength, kvLoraRank);
            }
            attentionCore(state, layer, basePos + t);
            System.arraycopy(state.xb2, 0, b[M_ATT][t], 0, totalValDim);
            Arrays.fill(out[t], 0, dim, 0f);
        }
        FloatTensor.matmulBatchParallel(weights.wo(), b[M_ATT], out, n, dim, totalValDim);
    }

    private static float[][][] batchBuffers(DeepSeek2State state, int cap, int totalQDim, int qLoraRank,
                                            int kvCompressedDim, int kvLoraRank, int kvBOutDim, int totalValDim) {
        if (state.mlaPrefill == null || state.mlaPrefill[M_Q].length < cap) {
            state.mlaPrefill = new float[][][] {
                new float[cap][totalQDim], new float[cap][qLoraRank], new float[cap][qLoraRank],
                new float[cap][kvCompressedDim], new float[cap][kvLoraRank], new float[cap][kvBOutDim],
                new float[cap][totalValDim]
            };
        }
        return state.mlaPrefill;
    }

    /** See {@link FloatTensor#warmUpRows}: the single-token kernels of this layer's projections. */
    void warmUp(DeepSeek2LayerWeights weights) {
        int dim = config.embeddingLength();
        int headCount = config.headCount();
        int keyLength = config.keyLength();
        int valueLength = config.valueLength();
        int kvLoraRank = config.kvLoraRank();
        int ropeDim = config.ropeDimensionCount();
        if (weights.hasQLoRA()) {
            FloatTensor.warmUpRows(weights.wqA(), config.qLoraRank(), dim);
            FloatTensor.warmUpRows(weights.wqB(), headCount * keyLength, config.qLoraRank());
        } else {
            FloatTensor.warmUpRows(weights.wq(), headCount * keyLength, dim);
        }
        FloatTensor.warmUpRows(weights.wkvA(), kvLoraRank + ropeDim, dim);
        if (!weights.hasSeparateKVB()) {
            FloatTensor.warmUpRows(weights.wkvB(), headCount * (keyLength - ropeDim + valueLength), kvLoraRank);
        }
        FloatTensor.warmUpRows(weights.wo(), dim, headCount * valueLength);
    }

    /**
     * Decompress K and V using separate per-head 3D tensors (GLM-4.7-Flash / DeepSeek-V3).
     *
     * K_B shape in GGUF: [keyNope, kvLoraRank, headCount] — TRANSPOSED: input=kvLoraRank, output=keyNope
     * V_B shape in GGUF: [kvLoraRank, valueLen, headCount] — standard: input=kvLoraRank, output=valueLen
     *
     * For K_B, each head's 2D slice is stored as kvLoraRank rows × keyNope cols in memory
     * (ne0=keyNope is the fast dimension), but we need matmul(input[kvLoraRank], weight) → output[keyNope].
     * Since the weight stores keyNope×kvLoraRank (ne0×ne1), we need a transposed dot:
     *   K_nope[j] = sum_i(c_latent[i] * wkB[j * kvLoraRank + i]) — but wkB is stored [keyNope][kvLoraRank]
     *   Actually: wkB expert offset = h * keyNope * kvLoraRank
     *   Row j of wkB (for output j) starts at offset h*keyNope*kvLoraRank + j*kvLoraRank (WRONG — ne0 is fast)
     *
     * GGUF 3D layout: data[h][row][col] where ne0=keyNope (col), ne1=kvLoraRank (row), ne2=headCount (h)
     * Flat index: h * ne1 * ne0 + row * ne0 + col
     * So wkB[h][row][col] = flat[h * kvLoraRank * keyNope + row * keyNope + col]
     * For transposed matmul: output[col] = sum_row(input[row] * wkB[h][row][col])
     * This is a column-wise dot product — we iterate over rows (kvLoraRank) for each output col (keyNope).
     */
    private void decompressKVSeparate(DeepSeek2State state, DeepSeek2LayerWeights weights,
                                       int headCount, int keyNope, int valueLength, int kvLoraRank) {
        // K decompression: transposed matmul per head
        // wkB 3D: [keyNope, kvLoraRank, headCount] — ne0=keyNope, ne1=kvLoraRank
        it.denzosoft.llmplayer.tensor.MatmulPool.forEach(headCount, h -> {
            long headOffset = (long) h * kvLoraRank * keyNope;
            int kDst = h * (keyNope + config.ropeDimensionCount()); // offset in state.k (keyLength per head)

            // Transposed matmul: output[col] = sum_row(input[row] * weight[row * ne0 + col])
            for (int col = 0; col < keyNope; col++) {
                float sum = 0f;
                for (int row = 0; row < kvLoraRank; row++) {
                    sum += state.kvLatentNormed[row] * weights.wkB().getFloat(headOffset + (long) row * keyNope + col);
                }
                state.k[kDst + col] = sum;
            }
        });

        // V decompression: standard matmul per head
        // wvB 3D: [kvLoraRank, valueLen, headCount] — ne0=kvLoraRank, ne1=valueLen
        // Standard: output[row] = dot(input, weight[row * ne0 ...])
        it.denzosoft.llmplayer.tensor.MatmulPool.forEach(headCount, h -> {
            long headOffset = (long) h * valueLength * kvLoraRank;
            int vDst = h * valueLength;

            for (int row = 0; row < valueLength; row++) {
                state.v[vDst + row] = weights.wvB().dot(headOffset + (long) row * kvLoraRank,
                    state.kvLatentNormed, 0, kvLoraRank);
            }
        });
    }
}
