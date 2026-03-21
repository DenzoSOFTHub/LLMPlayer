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

    // Qwen3.5 SSM (Gated DeltaNet) parameters
    private final int ssmConvKernel;
    private final int ssmStateSize;
    private final int ssmGroupCount;
    private final int ssmTimeStepRank;
    private final int ssmInnerSize;
    private final int fullAttentionInterval;

    // Llama4 iRoPE: interval at which layers skip RoPE (NoPE layers)
    // e.g., 4 means every 4th layer (layer % 4 == 3) is a NoPE layer
    private final int noRopeLayerInterval;

    // GLM-4.7-Flash / DeepSeek-V3: Q-LoRA rank (0 means no Q-LoRA, use direct wq)
    private final int qLoraRank;
    // MoE gating function: 0=softmax (DeepSeek V2), 2=sigmoid (GLM-4.7-Flash)
    private final int expertGatingFunc;
    // MoE expert weight scale (applied after optional L2 normalization)
    private final float expertWeightsScale;

    // Nemotron-H: per-layer arrays (null for other architectures)
    private int[] perLayerKvHeads;    // 0=Mamba/FFN, >0=attention
    private int[] perLayerFfnLength;  // 0=Mamba/attention, >0=FFN

    public ModelConfig(ModelArchitecture architecture, String name, int embeddingLength, int blockCount,
                       int headCount, int headCountKV, int contextLength, int vocabSize, int intermediateSize,
                       float ropeFreqBase, float normEps, int headSize, int kvDim, int ropeType,
                       int ropeDimensionCount, int keyLength, int valueLength, int kvLoraRank,
                       int leadingDenseBlockCount, int expertCount, int expertUsedCount,
                       int expertSharedCount, int expertFfnLength, float ropeScalingFactor,
                       int ropeOrigContextLength, float yarnLogMultiplier,
                       float finalLogitSoftCap, float attnLogitSoftCap, float logitScale,
                       int slidingWindow) {
        this(architecture, name, embeddingLength, blockCount, headCount, headCountKV, contextLength,
             vocabSize, intermediateSize, ropeFreqBase, normEps, headSize, kvDim, ropeType,
             ropeDimensionCount, keyLength, valueLength, kvLoraRank, leadingDenseBlockCount,
             expertCount, expertUsedCount, expertSharedCount, expertFfnLength, ropeScalingFactor,
             ropeOrigContextLength, yarnLogMultiplier, finalLogitSoftCap, attnLogitSoftCap, logitScale,
             slidingWindow, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0f);
    }

    public ModelConfig(ModelArchitecture architecture, String name, int embeddingLength, int blockCount,
                       int headCount, int headCountKV, int contextLength, int vocabSize, int intermediateSize,
                       float ropeFreqBase, float normEps, int headSize, int kvDim, int ropeType,
                       int ropeDimensionCount, int keyLength, int valueLength, int kvLoraRank,
                       int leadingDenseBlockCount, int expertCount, int expertUsedCount,
                       int expertSharedCount, int expertFfnLength, float ropeScalingFactor,
                       int ropeOrigContextLength, float yarnLogMultiplier,
                       float finalLogitSoftCap, float attnLogitSoftCap, float logitScale,
                       int slidingWindow,
                       int ssmConvKernel, int ssmStateSize, int ssmGroupCount,
                       int ssmTimeStepRank, int ssmInnerSize, int fullAttentionInterval,
                       int noRopeLayerInterval,
                       int qLoraRank, int expertGatingFunc, float expertWeightsScale) {
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
        this.ssmConvKernel = ssmConvKernel;
        this.ssmStateSize = ssmStateSize;
        this.ssmGroupCount = ssmGroupCount;
        this.ssmTimeStepRank = ssmTimeStepRank;
        this.ssmInnerSize = ssmInnerSize;
        this.fullAttentionInterval = fullAttentionInterval;
        this.noRopeLayerInterval = noRopeLayerInterval;
        this.qLoraRank = qLoraRank;
        this.expertGatingFunc = expertGatingFunc;
        this.expertWeightsScale = expertWeightsScale;
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
    public int ssmConvKernel() { return ssmConvKernel; }
    public int ssmStateSize() { return ssmStateSize; }
    public int ssmGroupCount() { return ssmGroupCount; }
    public int ssmTimeStepRank() { return ssmTimeStepRank; }
    public int ssmInnerSize() { return ssmInnerSize; }
    public int fullAttentionInterval() { return fullAttentionInterval; }
    public int noRopeLayerInterval() { return noRopeLayerInterval; }
    public int qLoraRank() { return qLoraRank; }
    public int expertGatingFunc() { return expertGatingFunc; }
    public float expertWeightsScale() { return expertWeightsScale; }

    // Nemotron-H per-layer support
    public int[] perLayerKvHeads() { return perLayerKvHeads; }
    public int[] perLayerFfnLength() { return perLayerFfnLength; }

    /** Nemotron-H layer type: 0=Mamba-2, 1=Attention, 2=FFN */
    public int nemotronLayerType(int layer) {
        if (perLayerKvHeads == null || perLayerFfnLength == null) return -1;
        if (layer >= perLayerKvHeads.length) return -1;
        if (perLayerKvHeads[layer] > 0) return 1;  // Attention
        if (perLayerFfnLength[layer] > 0) return 2; // FFN
        return 0; // Mamba-2
    }

    public int nemotronLayerKvHeads(int layer) {
        return (perLayerKvHeads != null && layer < perLayerKvHeads.length) ? perLayerKvHeads[layer] : headCountKV;
    }

    public int nemotronLayerFfnLength(int layer) {
        return (perLayerFfnLength != null && layer < perLayerFfnLength.length) ? perLayerFfnLength[layer] : intermediateSize;
    }

    public static ModelConfig fromMetadata(it.denzosoft.llmplayer.gguf.GGUFMetadata metadata) {
        String archName = metadata.getString("general.architecture", "llama");
        ModelArchitecture arch = ModelArchitecture.fromGgufName(archName);
        String prefix = archName + ".";

        String name = metadata.getString("general.name", "unknown");
        int embeddingLength = metadata.getInt(prefix + "embedding_length");
        int blockCount = metadata.getInt(prefix + "block_count");
        int headCount = metadata.getInt(prefix + "attention.head_count");
        // head_count_kv and feed_forward_length may be per-layer arrays (Nemotron-H)
        int headCountKV;
        int[] perLayerKvHeads = metadata.getIntArray(prefix + "attention.head_count_kv");
        if (perLayerKvHeads != null) {
            // Per-layer array (Nemotron-H): use max non-zero value
            headCountKV = 0;
            for (int v : perLayerKvHeads) if (v > headCountKV) headCountKV = v;
            if (headCountKV == 0) headCountKV = headCount;
        } else {
            headCountKV = metadata.getInt(prefix + "attention.head_count_kv", headCount);
            perLayerKvHeads = null;
        }
        int contextLength = metadata.getInt(prefix + "context_length", 2048);
        int intermediateSize;
        int[] perLayerFfnLength = metadata.getIntArray(prefix + "feed_forward_length");
        if (perLayerFfnLength != null) {
            intermediateSize = 0;
            for (int v : perLayerFfnLength) if (v > intermediateSize) intermediateSize = v;
            if (intermediateSize == 0) intermediateSize = embeddingLength * 4;
        } else {
            intermediateSize = metadata.getInt(prefix + "feed_forward_length", embeddingLength * 4);
            perLayerFfnLength = null;
        }

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
                || arch == ModelArchitecture.LLAMA4 || arch == ModelArchitecture.SMOLLM3) {
            ropeType = 0;  // ROPE_TYPE_NORMAL
        } else if (arch == ModelArchitecture.QWEN2 || arch == ModelArchitecture.QWEN3
                || arch == ModelArchitecture.GLM4 || arch == ModelArchitecture.PHI3
                || arch == ModelArchitecture.QWEN3MOE || arch == ModelArchitecture.OLMO2
                || arch == ModelArchitecture.GPT_OSS) {
            ropeType = 2;  // ROPE_TYPE_NEOX
        } else if (arch == ModelArchitecture.QWEN35) {
            ropeType = 2;  // ROPE_TYPE_NEOX (IMROPE uses split-half pairing like NEOX)
        } else {
            ropeType = 0;
        }

        // Attention key/value lengths from metadata (may override computed headSize)
        int keyLength = metadata.getInt(prefix + "attention.key_length", headSize);
        int valueLength = metadata.getInt(prefix + "attention.value_length", headSize);
        int kvLoraRank = metadata.getInt(prefix + "attention.kv_lora_rank", 0);

        // For DeepSeek2/GLM-4.7-Flash: key_length_mla / value_length_mla are the per-head MLA dims.
        // key_length=576 is the compressed KV dim, key_length_mla=256 is the actual per-head key dim.
        // When present, override keyLength/valueLength for MLA attention.
        if (arch == ModelArchitecture.DEEPSEEK2) {
            int keyLengthMla = metadata.getInt(prefix + "attention.key_length_mla", 0);
            int valueLengthMla = metadata.getInt(prefix + "attention.value_length_mla", 0);
            if (keyLengthMla > 0) keyLength = keyLengthMla;
            if (valueLengthMla > 0) valueLength = valueLengthMla;
        }

        // Override headSize when metadata specifies a different key_length
        // (e.g., Mistral3/Devstral: embeddingLength/headCount=160 but keyLength=128,
        //  Qwen3.5: embeddingLength/headCount=160 but keyLength=256 for full attention layers)
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

        // Qwen3.5 SSM (Gated DeltaNet) parameters
        int ssmConvKernel = metadata.getInt(prefix + "ssm.conv_kernel", 0);
        int ssmStateSize = metadata.getInt(prefix + "ssm.state_size", 0);
        int ssmGroupCount = metadata.getInt(prefix + "ssm.group_count", 0);
        int ssmTimeStepRank = metadata.getInt(prefix + "ssm.time_step_rank", 0);
        int ssmInnerSize = metadata.getInt(prefix + "ssm.inner_size", 0);
        int fullAttentionInterval = metadata.getInt(prefix + "full_attention_interval", 0);

        // iRoPE / NoPE: every Nth layer is a NoPE (no RoPE) layer
        // Default 4 for Llama4 and SmolLM3 (layers where layer % 4 == 3 skip RoPE), 0 for all others
        int noRopeLayerInterval = (arch == ModelArchitecture.LLAMA4 || arch == ModelArchitecture.SMOLLM3) ? 4 : 0;

        // Q-LoRA rank (GLM-4.7-Flash / DeepSeek-V3: decompose Q into Q_A * Q_B)
        int qLoraRank = metadata.getInt(prefix + "attention.q_lora_rank", 0);

        // MoE gating function: 0=softmax (default/DeepSeek V2), 2=sigmoid (GLM-4.7-Flash)
        int expertGatingFunc = metadata.getInt(prefix + "expert_gating_func", 0);

        // MoE expert weight scale (applied after optional L2 normalization)
        float expertWeightsScale = metadata.getFloat(prefix + "expert_weights_scale", 1.0f);

        // Nemotron-H: use ROPE_TYPE_NORMAL for attention layers
        if (arch == ModelArchitecture.NEMOTRON_H) {
            ropeType = 0; // ROPE_TYPE_NORMAL
        }

        ModelConfig config = new ModelConfig(arch, name, embeddingLength, blockCount, headCount, headCountKV,
            contextLength, vocabSize, intermediateSize, ropeFreqBase, normEps, headSize, kvDim,
            ropeType, ropeDimensionCount,
            keyLength, valueLength, kvLoraRank, leadingDenseBlockCount,
            expertCount, expertUsedCount, expertSharedCount, expertFfnLength,
            ropeScalingFactor, ropeOrigContextLength, yarnLogMultiplier,
            finalLogitSoftCap, attnLogitSoftCap, logitScale, slidingWindow,
            ssmConvKernel, ssmStateSize, ssmGroupCount, ssmTimeStepRank, ssmInnerSize,
            fullAttentionInterval, noRopeLayerInterval,
            qLoraRank, expertGatingFunc, expertWeightsScale);

        // Set per-layer arrays for Nemotron-H
        if (perLayerKvHeads != null) config.perLayerKvHeads = perLayerKvHeads;
        if (perLayerFfnLength != null) config.perLayerFfnLength = perLayerFfnLength;

        return config;
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
