package it.denzosoft.llmplayer.model;

public final class ModelConfig {
    private final ModelArchitecture architecture;
    private final String name;
    private final int embeddingLength;
    private final int blockCount;
    private final int headCount;
    private final int headCountKV;
    private final int contextLength;
    private final int vocabSize;
    private final int intermediateSize;
    private final float ropeFreqBase;
    private final float normEps;
    private final int headSize;
    private final int kvDim;
    private final int ropeType;
    private final int ropeDimensionCount;
    private final int keyLength;
    private final int valueLength;
    private final int kvLoraRank;
    private final int leadingDenseBlockCount;
    private final int expertCount;
    private final int expertUsedCount;
    private final int expertSharedCount;
    private final int expertFfnLength;
    private final float ropeScalingFactor;
    private final int ropeOrigContextLength;
    private final float yarnLogMultiplier;
    private final float finalLogitSoftCap;
    private final float attnLogitSoftCap;
    private final float logitScale;
    private final int slidingWindow;

    public ModelConfig(ModelArchitecture architecture, String name, int embeddingLength, int blockCount,
                       int headCount, int headCountKV, int contextLength, int vocabSize, int intermediateSize,
                       float ropeFreqBase, float normEps, int headSize, int kvDim, int ropeType,
                       int ropeDimensionCount, int keyLength, int valueLength, int kvLoraRank,
                       int leadingDenseBlockCount, int expertCount, int expertUsedCount,
                       int expertSharedCount, int expertFfnLength, float ropeScalingFactor,
                       int ropeOrigContextLength, float yarnLogMultiplier,
                       float finalLogitSoftCap, float attnLogitSoftCap, float logitScale,
                       int slidingWindow) {
        this.architecture = architecture;
        this.name = name;
        this.embeddingLength = embeddingLength;
        this.blockCount = blockCount;
        this.headCount = headCount;
        this.headCountKV = headCountKV;
        this.contextLength = contextLength;
        this.vocabSize = vocabSize;
        this.intermediateSize = intermediateSize;
        this.ropeFreqBase = ropeFreqBase;
        this.normEps = normEps;
        this.headSize = headSize;
        this.kvDim = kvDim;
        this.ropeType = ropeType;
        this.ropeDimensionCount = ropeDimensionCount;
        this.keyLength = keyLength;
        this.valueLength = valueLength;
        this.kvLoraRank = kvLoraRank;
        this.leadingDenseBlockCount = leadingDenseBlockCount;
        this.expertCount = expertCount;
        this.expertUsedCount = expertUsedCount;
        this.expertSharedCount = expertSharedCount;
        this.expertFfnLength = expertFfnLength;
        this.ropeScalingFactor = ropeScalingFactor;
        this.ropeOrigContextLength = ropeOrigContextLength;
        this.yarnLogMultiplier = yarnLogMultiplier;
        this.finalLogitSoftCap = finalLogitSoftCap;
        this.attnLogitSoftCap = attnLogitSoftCap;
        this.logitScale = logitScale;
        this.slidingWindow = slidingWindow;
    }

    public ModelArchitecture architecture() { return architecture; }
    public String name() { return name; }
    public int embeddingLength() { return embeddingLength; }
    public int blockCount() { return blockCount; }
    public int headCount() { return headCount; }
    public int headCountKV() { return headCountKV; }
    public int contextLength() { return contextLength; }
    public int vocabSize() { return vocabSize; }
    public int intermediateSize() { return intermediateSize; }
    public float ropeFreqBase() { return ropeFreqBase; }
    public float normEps() { return normEps; }
    public int headSize() { return headSize; }
    public int kvDim() { return kvDim; }
    public int ropeType() { return ropeType; }
    public int ropeDimensionCount() { return ropeDimensionCount; }
    public int keyLength() { return keyLength; }
    public int valueLength() { return valueLength; }
    public int kvLoraRank() { return kvLoraRank; }
    public int leadingDenseBlockCount() { return leadingDenseBlockCount; }
    public int expertCount() { return expertCount; }
    public int expertUsedCount() { return expertUsedCount; }
    public int expertSharedCount() { return expertSharedCount; }
    public int expertFfnLength() { return expertFfnLength; }
    public float ropeScalingFactor() { return ropeScalingFactor; }
    public int ropeOrigContextLength() { return ropeOrigContextLength; }
    public float yarnLogMultiplier() { return yarnLogMultiplier; }
    public float finalLogitSoftCap() { return finalLogitSoftCap; }
    public float attnLogitSoftCap() { return attnLogitSoftCap; }
    public float logitScale() { return logitScale; }
    public int slidingWindow() { return slidingWindow; }

    public static ModelConfig fromMetadata(it.denzosoft.llmplayer.gguf.GGUFMetadata metadata) {
        String archName = metadata.getString("general.architecture", "llama");
        ModelArchitecture arch = ModelArchitecture.fromGgufName(archName);
        String prefix = archName + ".";

        String name = metadata.getString("general.name", "unknown");
        int embeddingLength = metadata.getInt(prefix + "embedding_length");
        int blockCount = metadata.getInt(prefix + "block_count");
        int headCount = metadata.getInt(prefix + "attention.head_count");
        int headCountKV = metadata.getInt(prefix + "attention.head_count_kv", headCount);
        int contextLength = metadata.getInt(prefix + "context_length", 2048);
        int intermediateSize = metadata.getInt(prefix + "feed_forward_length", embeddingLength * 4);

        // Vocab size from tokenizer
        String[] tokens = metadata.getStringArray("tokenizer.ggml.tokens");
        int vocabSize = tokens != null ? tokens.length : metadata.getInt(prefix + "vocab_size", 32000);

        float ropeFreqBase = metadata.getFloat(prefix + "rope.freq_base", 10000.0f);
        float normEps = metadata.getFloat(prefix + "attention.layer_norm_rms_epsilon", 1e-5f);

        int defaultHeadSize = embeddingLength / headCount;
        int headSize = defaultHeadSize;
        int kvDim = headSize * headCountKV;

        // Llama/DeepSeek2/Mistral3/Command-R/Gemma/Llama4 use ROPE_TYPE_NORMAL (consecutive pairs),
        // Qwen/Falcon/GLM4/Phi3/Qwen3MoE/OLMo2/GPT-OSS use ROPE_TYPE_NEOX (split-half)
        int ropeType;
        if (arch == ModelArchitecture.LLAMA || arch == ModelArchitecture.DEEPSEEK2
                || arch == ModelArchitecture.MISTRAL3 || arch == ModelArchitecture.COMMAND_R
                || arch == ModelArchitecture.GEMMA2 || arch == ModelArchitecture.GEMMA3
                || arch == ModelArchitecture.LLAMA4) {
            ropeType = 0;  // ROPE_TYPE_NORMAL
        } else if (arch == ModelArchitecture.QWEN2 || arch == ModelArchitecture.QWEN3
                || arch == ModelArchitecture.GLM4 || arch == ModelArchitecture.PHI3
                || arch == ModelArchitecture.QWEN3MOE || arch == ModelArchitecture.OLMO2
                || arch == ModelArchitecture.GPT_OSS) {
            ropeType = 2;  // ROPE_TYPE_NEOX
        } else {
            ropeType = 0;
        }

        // Attention key/value lengths from metadata (may override computed headSize)
        int keyLength = metadata.getInt(prefix + "attention.key_length", headSize);
        int valueLength = metadata.getInt(prefix + "attention.value_length", headSize);
        int kvLoraRank = metadata.getInt(prefix + "attention.kv_lora_rank", 0);

        // Override headSize when metadata specifies a different key_length
        // (e.g., Mistral3/Devstral: embeddingLength/headCount=160 but keyLength=128)
        if (keyLength != headSize && arch != ModelArchitecture.DEEPSEEK2) {
            headSize = keyLength;
            kvDim = headSize * headCountKV;
        }

        // For DeepSeek2, override kvDim since Q/K/V dimensions differ from standard (MLA)
        if (arch == ModelArchitecture.DEEPSEEK2) {
            kvDim = headCountKV * valueLength;
        }

        // RoPE dimension count: read AFTER headSize override so default is correct
        int ropeDimensionCount = metadata.getInt(prefix + "rope.dimension_count", headSize);

        // MoE parameters (read before dense/MoE split to determine default)
        int expertCount = metadata.getInt(prefix + "expert_count", 0);
        int expertUsedCount = metadata.getInt(prefix + "expert_used_count", 0);
        int expertSharedCount = metadata.getInt(prefix + "expert_shared_count", 0);
        int expertFfnLength = metadata.getInt(prefix + "expert_feed_forward_length", 0);

        // Dense/MoE split: default to 0 dense blocks when experts are present
        int defaultDenseBlocks = (expertCount > 0) ? 0 : blockCount;
        int leadingDenseBlockCount = metadata.getInt(prefix + "leading_dense_block_count", defaultDenseBlocks);

        // YaRN scaling parameters
        String ropeScalingType = metadata.getString(prefix + "rope.scaling.type", "none");
        float ropeScalingFactor = 0;
        int ropeOrigContextLength = 0;
        float yarnLogMultiplier = 0;
        if ("yarn".equals(ropeScalingType)) {
            ropeScalingFactor = metadata.getFloat(prefix + "rope.scaling.factor", 1.0f);
            ropeOrigContextLength = metadata.getInt(prefix + "rope.scaling.original_context_length", contextLength);
            yarnLogMultiplier = metadata.getFloat(prefix + "rope.scaling.yarn_log_multiplier", 0.0f);
        }

        // Gemma2/3 logit soft-capping
        float finalLogitSoftCap = metadata.getFloat(prefix + "final_logit_softcapping", 0f);
        float attnLogitSoftCap = metadata.getFloat(prefix + "attn_logit_softcapping", 0f);

        // Command-R logit scale (multiplied to output logits)
        float logitScale = metadata.getFloat(prefix + "logit_scale", 0f);

        // ISWA sliding window (GPT-OSS: 128 tokens for alternating layers)
        int slidingWindow = metadata.getInt(prefix + "attention.sliding_window", 0);

        return new ModelConfig(arch, name, embeddingLength, blockCount, headCount, headCountKV,
            contextLength, vocabSize, intermediateSize, ropeFreqBase, normEps, headSize, kvDim,
            ropeType, ropeDimensionCount,
            keyLength, valueLength, kvLoraRank, leadingDenseBlockCount,
            expertCount, expertUsedCount, expertSharedCount, expertFfnLength,
            ropeScalingFactor, ropeOrigContextLength, yarnLogMultiplier,
            finalLogitSoftCap, attnLogitSoftCap, logitScale, slidingWindow);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format(
            "ModelConfig{arch=%s, name='%s', dim=%d, layers=%d, heads=%d, kvHeads=%d, ctx=%d, vocab=%d, ffn=%d, headSize=%d, ropeDim=%d",
            architecture, name, embeddingLength, blockCount, headCount, headCountKV,
            contextLength, vocabSize, intermediateSize, headSize, ropeDimensionCount));
        if (kvLoraRank > 0) {
            sb.append(String.format(", keyLen=%d, valLen=%d, kvLoraRank=%d", keyLength, valueLength, kvLoraRank));
        }
        if (expertCount > 0) {
            sb.append(String.format(", experts=%d(top%d+%dshared), expertFfn=%d, denseBlocks=%d",
                expertCount, expertUsedCount, expertSharedCount, expertFfnLength, leadingDenseBlockCount));
        }
        if (slidingWindow > 0) {
            sb.append(String.format(", slidingWindow=%d", slidingWindow));
        }
        sb.append('}');
        return sb.toString();
    }
}
