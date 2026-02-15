package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.model.DeepSeek2LayerWeights;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Multi-head Latent Attention (MLA) for DeepSeek2.
 *
 * Instead of standard Q/K/V projections, MLA compresses KV through a low-rank bottleneck:
 * 1. Q = wq * x → [headCount * keyLength]
 * 2. c_kv = wkvA * x → [kvLoraRank + ropeDim] (compressed KV + rope dims)
 * 3. Split: c_latent [kvLoraRank], k_rope [ropeDim]
 * 4. c_latent_norm = RMSNorm(c_latent)
 * 5. [K_nope, V] = wkvB * c_latent_norm → [headCount * (keyNope + valueLen)]
 * 6. K = concat(K_nope, broadcast(RoPE(k_rope))) per head
 * 7. Standard attention: softmax(Q*K^T / sqrt(keyLen)) * V
 */
public class MLAAttention {

    private final ModelConfig config;
    private final RoPE rope;  // operates on ropeDimCount dimensions
    private final float[][] cachedKvANorm;  // [layer][kvLoraRank]

    public MLAAttention(ModelConfig config, RoPE rope, DeepSeek2LayerWeights[] allLayers) {
        this.config = config;
        this.rope = rope;

        int blockCount = allLayers.length;
        this.cachedKvANorm = new float[blockCount][];
        for (int i = 0; i < blockCount; i++) {
            cachedKvANorm[i] = RMSNorm.cacheWeights(allLayers[i].kvANorm(), config.kvLoraRank());
        }
    }

    /**
     * Forward pass for MLA attention at a single position.
     * Uses DeepSeek2State which has MLA-specific buffers.
     */
    public void forward(DeepSeek2State state, DeepSeek2LayerWeights weights, int layer, int position) {
        int dim = config.embeddingLength();
        int headCount = config.headCount();
        int keyLength = config.keyLength();    // 192
        int valueLength = config.valueLength(); // 128
        int kvLoraRank = config.kvLoraRank();  // 512
        int ropeDim = config.ropeDimensionCount(); // 64
        int keyNope = keyLength - ropeDim;     // 128
        int kvCompressedDim = kvLoraRank + ropeDim; // 576
        int kvBOutPerHead = keyNope + valueLength;  // 256

        int totalQDim = headCount * keyLength;    // 3072
        int totalKeyDim = headCount * keyLength;  // 3072
        int totalValDim = headCount * valueLength; // 2048

        // 1. Q projection: x → Q [headCount * keyLength]
        Arrays.fill(state.q, 0, totalQDim, 0f);
        weights.wq().matmulParallel(state.xb, state.q, totalQDim, dim);

        // 2. KV compression: x → c_kv [kvLoraRank + ropeDim]
        Arrays.fill(state.kvCompressed, 0, kvCompressedDim, 0f);
        weights.wkvA().matmulParallel(state.xb, state.kvCompressed, kvCompressedDim, dim);

        // 3. Split c_kv into c_latent [kvLoraRank] and k_rope_raw [ropeDim]
        // c_latent = kvCompressed[0..kvLoraRank-1]
        // k_rope_raw = kvCompressed[kvLoraRank..kvCompressedDim-1]

        // 4. RMSNorm on c_latent (in-place, only first kvLoraRank elements)
        float[] kvNormWeights = cachedKvANorm[layer];
        float ss = 0f;
        for (int i = 0; i < kvLoraRank; i++) {
            ss += state.kvCompressed[i] * state.kvCompressed[i];
        }
        ss = 1.0f / (float) Math.sqrt(ss / kvLoraRank + config.normEps());
        for (int i = 0; i < kvLoraRank; i++) {
            state.kvLatentNormed[i] = state.kvCompressed[i] * ss * kvNormWeights[i];
        }

        // 5. KV decompression: c_latent_norm → [headCount * (keyNope + valueLen)]
        int kvBOutDim = headCount * kvBOutPerHead; // 4096
        Arrays.fill(state.kvDecompressed, 0, kvBOutDim, 0f);
        weights.wkvB().matmulParallel(state.kvLatentNormed, state.kvDecompressed, kvBOutDim, kvLoraRank);

        // 6. Assemble full K per head: [K_nope(128), k_rope(64)] → [192]
        // k_rope is shared across all heads (from kvCompressed[kvLoraRank..])
        // Apply RoPE to k_rope (once, shared)
        // Apply RoPE to Q_rope parts (per head)

        // Copy k_rope to a temporary buffer and apply RoPE
        System.arraycopy(state.kvCompressed, kvLoraRank, state.kRopeTemp, 0, ropeDim);
        // RoPE on k_rope: treat as single head with ropeDim dimensions
        rope.apply(state.kRopeTemp, 0, position);

        // Build full key array: for each head, K_nope from kvDecompressed + k_rope (shared)
        for (int h = 0; h < headCount; h++) {
            int kDst = h * keyLength;
            int kvBSrc = h * kvBOutPerHead;
            // K_nope[128]
            System.arraycopy(state.kvDecompressed, kvBSrc, state.k, kDst, keyNope);
            // k_rope[64] (same for all heads)
            System.arraycopy(state.kRopeTemp, 0, state.k, kDst + keyNope, ropeDim);
        }

        // Build full value array: V from kvDecompressed
        for (int h = 0; h < headCount; h++) {
            int vDst = h * valueLength;
            int kvBSrc = h * kvBOutPerHead + keyNope;
            System.arraycopy(state.kvDecompressed, kvBSrc, state.v, vDst, valueLength);
        }

        // Apply RoPE to Q_rope parts: each head's Q has [nope(128), rope(64)]
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
        // Scale includes YaRN mscale^2 (from llama.cpp: kq_scale = mscale * mscale / sqrt(n_embd_head_k))
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
}
