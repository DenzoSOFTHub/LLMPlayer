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

    // Granite-specific scaling factors (mutable — set after construction from metadata)
    private float embeddingScale;  // multiply embeddings after lookup (Granite: 12.0, Gemma: sqrt(dim))
    private float attentionScale;  // replace 1/sqrt(headSize) if non-zero (Granite: 1/128)
    private float residualScale;   // multiply output before residual add (Granite: 0.22)
    public void setEmbeddingScale(float v) { this.embeddingScale = v; }
    public void setAttentionScale(float v) { this.attentionScale = v; }
    public void setResidualScale(float v) { this.residualScale = v; }
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

    // Granite Hybrid MoE: shared-expert FFN length (mutable — set after construction)
    private int expertSharedFeedForwardLength;
    public int expertSharedFeedForwardLength() { return expertSharedFeedForwardLength; }
    public void setExpertSharedFeedForwardLength(int v) { this.expertSharedFeedForwardLength = v; }

    // Gemma 4: per-layer sliding window pattern and PLE config (mutable — set after construction)
    private boolean[] slidingWindowPattern; // true=SWA, false=full attention per layer
    private int sharedKvLayers;             // number of top layers sharing KV cache
    private int embeddingLengthPerLayer;    // PLE dim (256 for E4B)
    private float ropeFreqBaseSwa;          // theta for SWA layers (10000)
    private int ropeDimCountSwa;            // RoPE dim for SWA layers
    public void setSlidingWindowPattern(boolean[] v) { this.slidingWindowPattern = v; }
    public void setSharedKvLayers(int v) { this.sharedKvLayers = v; }
    public void setEmbeddingLengthPerLayer(int v) { this.embeddingLengthPerLayer = v; }
    public void setRopeFreqBaseSwa(float v) { this.ropeFreqBaseSwa = v; }
    public void setRopeDimCountSwa(int v) { this.ropeDimCountSwa = v; }
    public boolean[] slidingWindowPattern() { return slidingWindowPattern; }

    // Nanbeige looped depth: the GGUF stores physicalBlockCount layers that are run numLoops times
    // with shared weights (llama.cpp nanbeige.cpp). blockCount() is the LOGICAL layer count
    // (physical × loops): every logical layer has its own KV-cache slot, and the model's output norm
    // is also applied at each loop boundary unless skipLoopFinalNorm is set.
    private int numLoops = 1;
    private int physicalBlockCount;
    private boolean skipLoopFinalNorm;
    public int numLoops() { return numLoops; }
    public int physicalBlockCount() { return physicalBlockCount > 0 ? physicalBlockCount : blockCount; }
    public boolean skipLoopFinalNorm() { return skipLoopFinalNorm; }

    // Qwen2-VL / Qwen3-VL multi-axis RoPE: rope.dimension_sections splits the rotated pairs into
    // (temporal, height, width, extra) sections; Qwen3-VL interleaves them (IMROPE). Null for
    // every other model. nDeepstackLayers: Qwen3-VL adds vision "deepstack" features to the
    // hidden state after its first n layers.
    private int[] ropeSections;
    private boolean ropeSectionsInterleaved;
    private int nDeepstackLayers;
    public int[] ropeSections() { return ropeSections; }
    public boolean ropeSectionsInterleaved() { return ropeSectionsInterleaved; }
    public int nDeepstackLayers() { return nDeepstackLayers; }

    /**
     * Whether the output norm is applied to the residual stream after logical layer {@code layer}
     * (a loop boundary that is not the last layer). Always false for non-looped models.
     */
    public boolean isLoopBoundary(int layer) {
        return numLoops > 1 && !skipLoopFinalNorm
            && (layer + 1) % physicalBlockCount() == 0 && (layer + 1) < blockCount;
    }

    /**
     * Hunyuan dense applies the per-head Q/K RMSNorm AFTER RoPE, whereas Qwen3/Gemma normalise
     * before it (llama.cpp hunyuan-vl.cpp builds ggml_rope_ext first, then build_norm).
     */
    public boolean qkNormAfterRope() {
        return architecture == ModelArchitecture.HUNYUAN_DENSE;
    }

    /**
     * Architectures whose layer math only the CPU {@code TransformerBlock} implements: the
     * GPU-resident forward passes must decline them (their per-tensor GPU matmuls still apply).
     * Hunyuan (QK-norm after RoPE), Spark2.5 (per-head attention output gate, per-layer RoPE
     * dims) and looped Nanbeige (shared weights across logical layers, mid-stack norm).
     */
    public boolean requiresCpuLayerPath() {
        return architecture == ModelArchitecture.HUNYUAN_DENSE
            || architecture == ModelArchitecture.SPARK2_5
            || numLoops > 1
            || mropeDiffersFromNeox();
    }

    /**
     * True when the multi-axis RoPE sections leave some rotated pairs in the "extra" section, whose
     * position is 0 for text (Qwen3-VL, the Qwen3-TTS talker). The text rotation is then not plain
     * NEOX, which is all the GPU-resident passes implement.
     */
    public boolean mropeDiffersFromNeox() {
        if (ropeSections == null) return false;
        int[] map = mropeSectionMap(ropeSections, ropeSectionsInterleaved, ropeDimensionCount / 2);
        for (int sec : map) if (sec == 3) return true;
        return false;
    }
    public int sharedKvLayers() { return sharedKvLayers; }
    public int embeddingLengthPerLayer() { return embeddingLengthPerLayer; }
    public float ropeFreqBaseSwa() { return ropeFreqBaseSwa; }
    public int ropeDimCountSwa() { return ropeDimCountSwa; }

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
        this.embeddingScale = 0;
        this.attentionScale = 0;
        this.residualScale = 0;
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
    // expert_weights_norm: renormalise the selected routing weights to sum 1 (sigmoid routing)
    private boolean expertWeightsNorm;
    public boolean expertWeightsNorm() { return expertWeightsNorm; }
    public float expertWeightsScale() { return expertWeightsScale; }
    public float embeddingScale() { return embeddingScale; }
    public float attentionScale() { return attentionScale; }
    public float residualScale() { return residualScale; }

    /**
     * Returns true if this architecture uses standard Layer Normalization (mean-centered + variance)
     * instead of RMS Normalization. True for Command-R / Cohere2 — see llama.cpp
     * {@code command-r.cpp} and {@code cohere2-iswa.cpp} which use {@code LLM_NORM}.
     */
    public boolean useLayerNorm() {
        return architecture == ModelArchitecture.COMMAND_R
            || architecture == ModelArchitecture.COHERE2;
    }

    /**
     * Returns true if this architecture skips RoPE on global (full-attention) layers and only
     * applies it on sliding-window layers (NoPE-on-global pattern). Currently only Cohere2 — see
     * llama.cpp {@code cohere2-iswa.cpp:64} which gates {@code ggml_rope_ext} on {@code if (is_swa)}.
     */
    public boolean useNoPeOnGlobalLayers() {
        return architecture == ModelArchitecture.COHERE2;
    }

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

    /**
     * Per-layer KV head count, falling back to the scalar when the GGUF stores a single value.
     * Gemma 4 dense (12B/31B) stores attention.head_count_kv as a per-layer array — the
     * sliding-window layers use the full count (e.g. 8) while the every-6th global layers use
     * a reduced count (e.g. 1). Architectures with a scalar head_count_kv (E2B/E4B, Llama, …)
     * get the uniform value for every layer.
     */
    public int layerKvHeads(int layer) {
        return (perLayerKvHeads != null && layer < perLayerKvHeads.length) ? perLayerKvHeads[layer] : headCountKV;
    }

    /**
     * Per-layer FFN intermediate size, falling back to the scalar when uniform. Gemma 4 E2B/E4B
     * use a "double-wide MLP": the GGUF stores feed_forward_length as a per-layer array (e.g. the
     * E2B's first 15 layers are 6144-wide, the last 20 are 12288-wide). Architectures with a
     * scalar feed_forward_length get the uniform value for every layer.
     */
    public int layerFfnLength(int layer) {
        return (perLayerFfnLength != null && layer < perLayerFfnLength.length) ? perLayerFfnLength[layer] : intermediateSize;
    }

    /** LFM2: a layer is GQA attention when its per-layer kv-head count > 0, else a short-conv layer. */
    public boolean lfm2IsAttentionLayer(int layer) {
        return perLayerKvHeads != null && layer < perLayerKvHeads.length && perLayerKvHeads[layer] > 0;
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
        // GLM4-MoE: block_count also counts the NextN/MTP layers appended after the trunk (used only
        // for speculative decoding in llama.cpp, and often stripped from the file). Run the trunk only.
        if ("glm4moe".equals(archName)) {
            blockCount -= metadata.getInt(prefix + "nextn_predict_layers", 0);
        }
        // Nanbeige looped depth: block_count is the physical layer count; the model runs it
        // num_loops times, so the logical depth (and the KV cache) is block_count × num_loops.
        int physicalBlockCount = blockCount;
        int numLoops = Math.max(1, metadata.getInt(prefix + "num_loops", 1));
        blockCount = physicalBlockCount * numLoops;
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
            // Granite Hybrid: scalar feed_forward_length + per-layer kv_heads array
            // Build per-layer FFN length: attention layers (kv>0) have FFN, Mamba layers don't
            if (arch == ModelArchitecture.GRANITE_HYBRID && perLayerKvHeads != null) {
                perLayerFfnLength = new int[perLayerKvHeads.length];
                for (int i = 0; i < perLayerKvHeads.length; i++) {
                    perLayerFfnLength[i] = perLayerKvHeads[i] > 0 ? intermediateSize : 0;
                }
            } else {
                perLayerFfnLength = null;
            }
        }

        // Vocab size from tokenizer
        String[] tokens = metadata.getStringArray("tokenizer.ggml.tokens");
        int vocabSize = tokens != null ? tokens.length : metadata.getInt(prefix + "vocab_size", 32000);

        float ropeFreqBase = metadata.getFloat(prefix + "rope.freq_base", 10000.0f);
        float normEps = metadata.getFloat(prefix + "attention.layer_norm_rms_epsilon", 1e-5f);

        int defaultHeadSize = embeddingLength / headCount;
        int headSize = defaultHeadSize;
        int kvDim = headSize * headCountKV;

        // Llama/DeepSeek2/Mistral3/Command-R/Llama4/GLM4 use ROPE_TYPE_NORMAL (consecutive pairs),
        // Qwen/Falcon/Phi3/Qwen3MoE/OLMo2/GPT-OSS/Gemma use ROPE_TYPE_NEOX (split-half).
        // GLM4 is NORM as in llama.cpp (only glm4moe is NEOX): NEOX gave PPL 5.45 vs 2.60 on GLM-4-9B.
        int ropeType;
        if (arch == ModelArchitecture.LLAMA || arch == ModelArchitecture.DEEPSEEK2
                || arch == ModelArchitecture.MISTRAL3 || arch == ModelArchitecture.COMMAND_R
                || arch == ModelArchitecture.COHERE2
                || arch == ModelArchitecture.LLAMA4 || arch == ModelArchitecture.SMOLLM3
                || arch == ModelArchitecture.GRANITE
                || arch == ModelArchitecture.ERNIE4_5 || arch == ModelArchitecture.GLM4) {
            ropeType = 0;  // ROPE_TYPE_NORMAL
        } else if (arch == ModelArchitecture.QWEN2 || arch == ModelArchitecture.QWEN3
                || arch == ModelArchitecture.PHI3
                || arch == ModelArchitecture.QWEN3MOE || arch == ModelArchitecture.OLMO2
                || arch == ModelArchitecture.GPT_OSS
                || arch == ModelArchitecture.GRANITE_HYBRID
                || arch == ModelArchitecture.LFM2 || arch == ModelArchitecture.FALCON_H1
                || arch == ModelArchitecture.GEMMA2 || arch == ModelArchitecture.GEMMA3
                || arch == ModelArchitecture.GEMMA3N || arch == ModelArchitecture.GEMMA4
                || arch == ModelArchitecture.HUNYUAN_DENSE || arch == ModelArchitecture.SPARK2_5) {
            ropeType = 2;  // ROPE_TYPE_NEOX
        } else if (arch == ModelArchitecture.QWEN35) {
            ropeType = 2;  // ROPE_TYPE_NEOX (IMROPE uses split-half pairing like NEOX)
        } else {
            ropeType = 0;
        }
        // GLM-4.xV (glm4 with rope.dimension_sections): the converter permutes Q/K to NEOX order for
        // M-RoPE, whose text rotation (all axes equal, non-interleaved sections) is plain NEOX.
        // glm4moe is NEOX in llama.cpp (its checkpoints already use NEOX ordering).
        if (arch == ModelArchitecture.GLM4 && ("glm4moe".equals(archName)
                || metadata.getIntArray(prefix + "rope.dimension_sections") != null)) {
            ropeType = 2;
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
        // Gemma 4: key_length=512 is for full attention layers, key_length_swa=256 is for SWA layers
        // and all tensors use SWA dims. Use key_length_swa for headSize.
        if (arch == ModelArchitecture.GEMMA4) {
            int keyLengthSwa = metadata.getInt(prefix + "attention.key_length_swa", headSize);
            headSize = keyLengthSwa;
            kvDim = headSize * headCountKV;
        } else if (keyLength != headSize && arch != ModelArchitecture.DEEPSEEK2) {
            headSize = keyLength;
            kvDim = headSize * headCountKV;
        }

        // For DeepSeek2, override kvDim since Q/K/V dimensions differ from standard (MLA)
        if (arch == ModelArchitecture.DEEPSEEK2) {
            kvDim = headCountKV * valueLength;
        }

        // Hunyuan XDRoPE / NTK-aware scaling: base = theta * alpha^(d / (d - 2)) (llama.cpp
        // hunyuan-vl.cpp). Current converters bake this into rope.freq_base and omit the key.
        float ropeScalingAlpha = metadata.getFloat(prefix + "rope.scaling.alpha", 0f);
        if (ropeScalingAlpha > 0f && headSize > 2) {
            ropeFreqBase = (float) (ropeFreqBase * Math.pow(ropeScalingAlpha, (double) headSize / (headSize - 2)));
        }

        // RoPE dimension count: read AFTER headSize override so default is correct
        int ropeDimensionCount = metadata.getInt(prefix + "rope.dimension_count", headSize);
        // Gemma 4: ropeDimCount from metadata may exceed headSize (512 > 256); clamp to headSize
        if (arch == ModelArchitecture.GEMMA4 && ropeDimensionCount > headSize) {
            ropeDimensionCount = headSize;
        }

        // MoE parameters (read before dense/MoE split to determine default)
        int expertCount = metadata.getInt(prefix + "expert_count", 0);
        int expertUsedCount = metadata.getInt(prefix + "expert_used_count", 0);
        int expertSharedCount = metadata.getInt(prefix + "expert_shared_count", 0);
        int expertFfnLength = metadata.getInt(prefix + "expert_feed_forward_length", 0);
        // Granite Hybrid MoE: shared-expert FFN size is given directly (no expert_shared_count key).
        int expertSharedFfnLength = metadata.getInt(prefix + "expert_shared_feed_forward_length", 0);

        // Dense/MoE split: default to 0 dense blocks when experts are present
        int defaultDenseBlocks = (expertCount > 0) ? 0 : blockCount;
        int leadingDenseBlockCount = metadata.getInt(prefix + "leading_dense_block_count", defaultDenseBlocks);

        // RoPE scaling parameters. Supported types:
        //   "yarn"   — YaRN extension (DeepSeek-V2, etc.) with log-multiplier
        //   "linear" — linear position downscaling (Gemma 3 4B, Llama-2 long-context, etc.)
        // For linear, RoPE positions are divided by the factor (effectively stretching the
        // pretrained context window). For yarn, the additional yarn parameters apply.
        String ropeScalingType = metadata.getString(prefix + "rope.scaling.type", "none");
        float ropeScalingFactor = 0;
        int ropeOrigContextLength = 0;
        float yarnLogMultiplier = 0;
        if ("yarn".equals(ropeScalingType)) {
            ropeScalingFactor = metadata.getFloat(prefix + "rope.scaling.factor", 1.0f);
            ropeOrigContextLength = metadata.getInt(prefix + "rope.scaling.original_context_length", contextLength);
            yarnLogMultiplier = metadata.getFloat(prefix + "rope.scaling.yarn_log_multiplier", 0.0f);
        } else if ("linear".equals(ropeScalingType)) {
            // Linear scaling: position[i] used in RoPE becomes i / factor.
            // Stored in ropeScalingFactor; consumer (RoPE) interprets it as a divisor when
            // yarnLogMultiplier == 0 (i.e. non-yarn mode). This matches llama.cpp's
            // f_freq_scale = 1.0f / factor for linear scaling.
            ropeScalingFactor = metadata.getFloat(prefix + "rope.scaling.factor", 1.0f);
        }

        // Gemma2/3 logit soft-capping
        float finalLogitSoftCap = metadata.getFloat(prefix + "final_logit_softcapping", 0f);
        float attnLogitSoftCap = metadata.getFloat(prefix + "attn_logit_softcapping", 0f);

        // Command-R logit scale (multiplied to output logits)
        // Granite: logit_scale is DIVIDED (not multiplied) — handled in InferenceEngine
        float logitScale = metadata.getFloat(prefix + "logit_scale", 0f);

        // Granite-specific scaling factors
        float embeddingScale = metadata.getFloat(prefix + "embedding_scale", 0f);
        float attentionScale = metadata.getFloat(prefix + "attention.scale", 0f);
        float residualScale = metadata.getFloat(prefix + "residual_scale", 0f);

        // ISWA sliding window (GPT-OSS: 128 tokens for alternating layers)
        int slidingWindow = metadata.getInt(prefix + "attention.sliding_window", 0);

        // Qwen3.5 SSM (Gated DeltaNet) parameters
        int ssmConvKernel = metadata.getInt(prefix + "ssm.conv_kernel", 0);
        int ssmStateSize = metadata.getInt(prefix + "ssm.state_size", 0);
        int ssmGroupCount = metadata.getInt(prefix + "ssm.group_count", 0);
        int ssmTimeStepRank = metadata.getInt(prefix + "ssm.time_step_rank", 0);
        int ssmInnerSize = metadata.getInt(prefix + "ssm.inner_size", 0);
        int fullAttentionInterval = metadata.getInt(prefix + "full_attention_interval", 0);

        // LFM2 short convolution: reuse ssmConvKernel field to carry shortconv.l_cache (kernel width).
        if (arch == ModelArchitecture.LFM2) {
            ssmConvKernel = metadata.getInt(prefix + "shortconv.l_cache", 3);
        }

        // iRoPE / NoPE: every Nth layer is a NoPE (no RoPE) layer
        // Default 4 for Llama4 and SmolLM3 (layers where layer % 4 == 3 skip RoPE), 0 for all others
        int noRopeLayerInterval = (arch == ModelArchitecture.LLAMA4 || arch == ModelArchitecture.SMOLLM3) ? 4 : 0;

        // Q-LoRA rank (GLM-4.7-Flash / DeepSeek-V3: decompose Q into Q_A * Q_B)
        int qLoraRank = metadata.getInt(prefix + "attention.q_lora_rank", 0);

        // MoE gating function: 0=softmax (default/DeepSeek V2), 2=sigmoid (GLM-4.7-Flash)
        int expertGatingFunc = metadata.getInt(prefix + "expert_gating_func", 0);
        // llama.cpp glm4-moe.cpp: an absent (NONE) gating function means sigmoid
        if ("glm4moe".equals(archName) && expertGatingFunc == 0) expertGatingFunc = 2;

        // MoE expert weight scale (applied after optional L2 normalization)
        float expertWeightsScale = metadata.getFloat(prefix + "expert_weights_scale", 1.0f);

        // Nemotron-H / Granite Hybrid: use ROPE_TYPE_NORMAL for attention layers
        if (arch == ModelArchitecture.NEMOTRON_H || arch == ModelArchitecture.GRANITE_HYBRID) {
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

        if ("qwen3vl".equals(archName) || "qwen2vl".equals(archName) || "qwen35".equals(archName)
                || "qwen3tts".equals(archName)) {
            int[] sections = metadata.getIntArray(prefix + "rope.dimension_sections");
            if (sections != null && sections.length >= 3) {
                config.ropeSections = java.util.Arrays.copyOf(sections, 4);
                config.ropeSectionsInterleaved = !"qwen2vl".equals(archName);
            }
            config.nDeepstackLayers = metadata.getInt(prefix + "n_deepstack_layers", 0);
        }
        config.expertWeightsNorm = metadata.getBoolean(prefix + "expert_weights_norm", false);
        config.numLoops = numLoops;
        config.physicalBlockCount = physicalBlockCount;
        config.skipLoopFinalNorm = metadata.getBoolean(prefix + "skip_loop_final_norm", false);

        // Set per-layer arrays for Nemotron-H
        if (perLayerKvHeads != null) config.perLayerKvHeads = perLayerKvHeads;
        if (perLayerFfnLength != null) config.perLayerFfnLength = perLayerFfnLength;

        if (expertSharedFfnLength > 0) config.setExpertSharedFeedForwardLength(expertSharedFfnLength);

        // Set Granite scaling factors (must be mutable fields since constructor has too many params)
        if (embeddingScale != 0) config.setEmbeddingScale(embeddingScale);
        if (attentionScale != 0) config.setAttentionScale(attentionScale);
        if (residualScale != 0) config.setResidualScale(residualScale);

        // Gemma 4: attention scale = 1.0 (model handles scaling via QK-norm internally)
        if (arch == ModelArchitecture.GEMMA4 && config.attentionScale() == 0f) {
            config.setAttentionScale(1.0f);
        }

        // Gemma 3n: same PLE config as Gemma 4 + AltUp/Laurel-specific config
        if (arch == ModelArchitecture.GEMMA3N) {
            config.setEmbeddingLengthPerLayer(metadata.getInt(prefix + "embedding_length_per_layer_input", 0));
            config.setEmbeddingScale((float) Math.sqrt(embeddingLength));
            // For Gemma 3n E4B: 15 of 35 layers reuse earlier layers' KV cache
            // (n_layer_kv_from_start = blockCount - shared_kv_layers = 35 - 15 = 20)
            config.setSharedKvLayers(metadata.getInt(prefix + "attention.shared_kv_layers", 0));
            // Parse sliding window pattern
            Object swpObj = metadata.get(prefix + "attention.sliding_window_pattern");
            if (swpObj instanceof Object[]) {
                Object[] swpArr = (Object[]) swpObj;
                boolean[] pattern = new boolean[swpArr.length];
                for (int i = 0; i < swpArr.length; i++) {
                    pattern[i] = swpArr[i] instanceof Boolean
                            ? ((Boolean) swpArr[i]).booleanValue()
                            : Boolean.parseBoolean(String.valueOf(swpArr[i]));
                }
                config.setSlidingWindowPattern(pattern);
            }
        }

        // Spark2.5: 3 sliding-window layers per full-attention layer, given as a boolean pattern
        // (true = SWA). SWA layers rotate every dim at theta_swa; full layers rotate
        // rope.dimension_count dims (a quarter) at the main theta (llama.cpp spark2-5.cpp).
        if (arch == ModelArchitecture.SPARK2_5) {
            config.setRopeFreqBaseSwa(metadata.getFloat(prefix + "rope.freq_base_swa", ropeFreqBase));
            config.setRopeDimCountSwa(metadata.getInt(prefix + "rope.dimension_count_swa", ropeDimensionCount));
            config.setSlidingWindowPattern(parseBoolArray(metadata.get(prefix + "attention.sliding_window_pattern")));
        }

        // Gemma 4: PLE config, shared KV, sliding window pattern, dual RoPE
        if (arch == ModelArchitecture.GEMMA4) {
            config.setSharedKvLayers(metadata.getInt(prefix + "attention.shared_kv_layers", 0));
            config.setEmbeddingLengthPerLayer(metadata.getInt(prefix + "embedding_length_per_layer_input", 0));
            config.setRopeFreqBaseSwa(metadata.getFloat(prefix + "rope.freq_base_swa", ropeFreqBase));
            config.setRopeDimCountSwa(metadata.getInt(prefix + "rope.dimension_count_swa", ropeDimensionCount));
            // Parse sliding window pattern (boolean array from GGUF)
            Object swpObj = metadata.get(prefix + "attention.sliding_window_pattern");
            if (swpObj instanceof Object[]) {
                Object[] swpArr = (Object[]) swpObj;
                boolean[] pattern = new boolean[swpArr.length];
                for (int i = 0; i < swpArr.length; i++) {
                    pattern[i] = swpArr[i] instanceof Boolean
                            ? ((Boolean) swpArr[i]).booleanValue()
                            : Boolean.parseBoolean(String.valueOf(swpArr[i]));
                }
                config.setSlidingWindowPattern(pattern);
            }
            // Gemma 4 uses sqrt(dim) embedding scaling like Gemma 2/3
            config.setEmbeddingScale((float) Math.sqrt(embeddingLength));
        }

        return config;
    }

    /**
     * Section of each rotated pair for multi-axis RoPE (0 = temporal, 1 = height, 2 = width,
     * 3 = extra), as ggml's mrope cache: sequential sections for MROPE (Qwen2-VL), interleaved
     * t/h/w for IMROPE (Qwen3-VL, Qwen3.5), with pairs past the sections going to "extra".
     */
    public static int[] mropeSectionMap(int[] sections, boolean interleaved, int halfRope) {
        int s0 = sections[0], s1 = sections[1], s2 = sections[2], s3 = sections.length > 3 ? sections[3] : 0;
        int sectDims = s0 + s1 + s2 + s3;
        int[] map = new int[halfRope];
        for (int i = 0; i < halfRope; i++) {
            int sector = sectDims > 0 ? i % sectDims : i;
            int sec;
            if (interleaved) {
                if (sector % 3 == 1 && sector < 3 * s1) sec = 1;
                else if (sector % 3 == 2 && sector < 3 * s2) sec = 2;
                else if (sector % 3 == 0 && sector < 3 * s0) sec = 0;
                else sec = 3;
            } else {
                if (sector < s0) sec = 0;
                else if (sector < s0 + s1) sec = 1;
                else if (sector < s0 + s1 + s2) sec = 2;
                else sec = 3;
            }
            map[i] = sec;
        }
        return map;
    }

    private static boolean[] parseBoolArray(Object obj) {
        if (!(obj instanceof Object[])) return null;
        Object[] arr = (Object[]) obj;
        boolean[] out = new boolean[arr.length];
        for (int i = 0; i < arr.length; i++) {
            out[i] = arr[i] instanceof Boolean
                    ? ((Boolean) arr[i]).booleanValue()
                    : Boolean.parseBoolean(String.valueOf(arr[i]));
        }
        return out;
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
        if (embeddingScale > 0 || attentionScale > 0 || residualScale > 0) {
            sb.append(String.format(", embScale=%.1f, attnScale=%.7f, resScale=%.2f",
                embeddingScale, attentionScale, residualScale));
        }
        if (logitScale > 0) sb.append(String.format(", logitScale=%.1f", logitScale));
        if (numLoops > 1) sb.append(String.format(", loops=%dx%d", numLoops, physicalBlockCount()));
        sb.append('}');
        return sb.toString();
    }
}
