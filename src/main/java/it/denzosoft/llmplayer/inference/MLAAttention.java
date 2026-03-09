package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.DeepSeek2LayerWeights;
import it.denzosoft.llmplayer.model.ModelConfig;
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
        int keyLength = config.keyLength();    // e.g. 192 (DS2) or 256 (GLM-4.7)
        int valueLength = config.valueLength(); // e.g. 128 (DS2) or 256 (GLM-4.7)
        int kvLoraRank = config.kvLoraRank();  // e.g. 512
        int ropeDim = config.ropeDimensionCount(); // e.g. 64
        int keyNope = keyLength - ropeDim;     // e.g. 128 (DS2) or 192 (GLM-4.7)
        int kvCompressedDim = kvLoraRank + ropeDim; // e.g. 576

        int totalQDim = headCount * keyLength;
        int totalKeyDim = headCount * keyLength;
        int totalValDim = headCount * valueLength;

        // 1. Q projection
        if (weights.hasQLoRA()) {
            // Q-LoRA: x → qCompressed → norm → Q
            int qLoraRank = config.qLoraRank();
            Arrays.fill(state.qCompressed, 0, qLoraRank, 0f);
            weights.wqA().matmulParallel(state.xb, state.qCompressed, qLoraRank, dim);

            // RMSNorm on qCompressed
            float[] qaNormWeights = cachedQANorm[layer];
            RMSNorm.apply(state.qCompressedNorm, state.qCompressed, qaNormWeights, qLoraRank, config.normEps());

            // Q = wqB * qCompressedNorm
            Arrays.fill(state.q, 0, totalQDim, 0f);
            weights.wqB().matmulParallel(state.qCompressedNorm, state.q, totalQDim, qLoraRank);
        } else {
            // Direct: x → Q
            Arrays.fill(state.q, 0, totalQDim, 0f);
            weights.wq().matmulParallel(state.xb, state.q, totalQDim, dim);
        }

        // 2. KV compression: x → c_kv [kvLoraRank + ropeDim]
        Arrays.fill(state.kvCompressed, 0, kvCompressedDim, 0f);
        weights.wkvA().matmulParallel(state.xb, state.kvCompressed, kvCompressedDim, dim);

        // 3. Split c_kv into c_latent [kvLoraRank] and k_rope_raw [ropeDim]
        // c_latent = kvCompressed[0..kvLoraRank-1]
        // k_rope_raw = kvCompressed[kvLoraRank..kvCompressedDim-1]

        // 4. RMSNorm on c_latent (only first kvLoraRank elements)
        float[] kvNormWeights = cachedKvANorm[layer];
        float ss = 0f;
        for (int i = 0; i < kvLoraRank; i++) {
            ss += state.kvCompressed[i] * state.kvCompressed[i];
        }
        ss = 1.0f / (float) Math.sqrt(ss / kvLoraRank + config.normEps());
        for (int i = 0; i < kvLoraRank; i++) {
            state.kvLatentNormed[i] = state.kvCompressed[i] * ss * kvNormWeights[i];
        }

        // 5. KV decompression
        if (weights.hasSeparateKVB()) {
            // Separate K_B (transposed) and V_B (standard) per head
            decompressKVSeparate(state, weights, headCount, keyNope, valueLength, kvLoraRank);
        } else {
            // Combined wkvB → [headCount * (keyNope + valueLen)]
            int kvBOutPerHead = keyNope + valueLength;
            int kvBOutDim = headCount * kvBOutPerHead;
            Arrays.fill(state.kvDecompressed, 0, kvBOutDim, 0f);
            weights.wkvB().matmulParallel(state.kvLatentNormed, state.kvDecompressed, kvBOutDim, kvLoraRank);

            // Extract K_nope and V from interleaved kvDecompressed
            for (int h = 0; h < headCount; h++) {
                int kvBSrc = h * kvBOutPerHead;
                // K_nope
                System.arraycopy(state.kvDecompressed, kvBSrc, state.k, h * keyLength, keyNope);
                // V
                System.arraycopy(state.kvDecompressed, kvBSrc + keyNope, state.v, h * valueLength, valueLength);
            }
        }

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

        // 7. Store K and V in cache
        float[] keyCache = state.keyCache[layer];
        float[] valueCache = state.valueCache[layer];
        System.arraycopy(state.k, 0, keyCache, position * totalKeyDim, totalKeyDim);
        System.arraycopy(state.v, 0, valueCache, position * totalValDim, totalValDim);

        // 8. Attention computation - parallel over heads
        float mscale = rope.getMscale();
        float scaleFactor = mscale * mscale / (float) Math.sqrt(keyLength);

        Arrays.fill(state.xb2, 0, totalValDim, 0f);

        IntStream.range(0, headCount).parallel().forEach(h -> {
            int attOffset = h * (position + 1);

            // Compute attention scores
            for (int t = 0; t <= position; t++) {
                float score = 0f;
                int qOffset = h * keyLength;
                int kOffset = t * totalKeyDim + h * keyLength;
                for (int i = 0; i < keyLength; i++) {
                    score += state.q[qOffset + i] * keyCache[kOffset + i];
                }
                state.att[attOffset + t] = score * scaleFactor;
            }

            // Softmax
            VectorOpsFactory.get().softmax(state.att, attOffset, position + 1);

            // Weighted sum of values
            int outOffset = h * valueLength;
            for (int t = 0; t <= position; t++) {
                float a = state.att[attOffset + t];
                int vOffset = t * totalValDim + h * valueLength;
                for (int i = 0; i < valueLength; i++) {
                    state.xb2[outOffset + i] += a * valueCache[vOffset + i];
                }
            }
        });

        // 9. Output projection: xb = Wo * xb2
        Arrays.fill(state.xb, 0);
        weights.wo().matmulParallel(state.xb2, state.xb, dim, totalValDim);
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
        IntStream.range(0, headCount).parallel().forEach(h -> {
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
        IntStream.range(0, headCount).parallel().forEach(h -> {
            long headOffset = (long) h * valueLength * kvLoraRank;
            int vDst = h * valueLength;

            for (int row = 0; row < valueLength; row++) {
                state.v[vDst + row] = weights.wvB().dot(headOffset + (long) row * kvLoraRank,
                    state.kvLatentNormed, 0, kvLoraRank);
            }
        });
    }
}
