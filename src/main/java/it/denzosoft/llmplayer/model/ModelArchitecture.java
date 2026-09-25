package it.denzosoft.llmplayer.model;

public enum ModelArchitecture {
    LLAMA("llama"),
    QWEN2("qwen2"),
    QWEN3("qwen3"),
    DEEPSEEK2("deepseek2"),
    GLM4("glm4"),
    PHI3("phi3"),
    QWEN3MOE("qwen3moe"),
    MISTRAL3("mistral3"),
    COMMAND_R("command-r"),
    COHERE2("cohere2"),
    OLMO2("olmo2"),
    GEMMA2("gemma2"),
    GEMMA3("gemma3"),
    LLAMA4("llama4"),
    GPT_OSS("gpt-oss"),
    QWEN35("qwen35"),
    SMOLLM3("smollm3"),
    NEMOTRON_H("nemotron_h"),
    GRANITE("granite"),
    GEMMA4("gemma4"),
    GEMMA3N("gemma3n"),
    GRANITE_HYBRID("granitehybrid"),
    ERNIE4_5("ernie4_5"),
    LFM2("lfm2"),
    FALCON_H1("falcon-h1"),
    HUNYUAN_DENSE("hunyuan-dense"),
    NANBEIGE("nanbeige"),
    SPARK2_5("spark2_5"),
    BAILINGMOE3("bailingmoe3");

    private final String ggufName;

    ModelArchitecture(String ggufName) {
        this.ggufName = ggufName;
    }

    public String getGgufName() { return ggufName; }

    public static ModelArchitecture fromGgufName(String name) {
        for (ModelArchitecture arch : values()) {
            if (arch.ggufName.equals(name)) {
                return arch;
            }
        }
        // Handle aliases for architectures that may appear under different names
        if ("command_r".equals(name) || "cohere".equals(name)) {
            return COMMAND_R;
        }
        // Cohere2 is intentionally separate: it differs from Command-R in NoPE-on-global-layers,
        // ISWA cache layout (every 4th layer global), and lacks Q/K norms.
        if ("cohere2".equals(name)) {
            return COHERE2;
        }
        if ("gemma".equals(name)) {
            return GEMMA2; // Gemma1 uses same forward pass
        }
        // Qwen2.5-VL / Qwen3-VL text backbones: the decoder is Qwen2 / Qwen3. They use multi-axis
        // RoPE (MRoPE / IMRoPE), but for text tokens the three position axes are equal and the
        // rotation reduces to NEOX RoPE — see llama.cpp ggml_rope_multi. The vision projector
        // (mmproj) is a separate file.
        if ("qwen3vl".equals(name)) {
            return QWEN3;
        }
        if ("qwen2vl".equals(name)) {
            return QWEN2;
        }
        // Qwen3-TTS talker: the Qwen3-VL decoder (IMROPE) over a text + codec vocabulary, with a
        // 3072-row codec output head. Only usable through it.denzosoft.llmplayer.tts.Qwen3Tts.
        if ("qwen3tts".equals(name)) {
            return QWEN3;
        }
        // LFM2-MoE (LFM2.5-8B-A1B): the LFM2 conv/attention layer mix with a routed-expert FFN on
        // layers >= leading_dense_block_count (llama.cpp lfm2.cpp build_moe_feed_forward).
        if ("lfm2moe".equals(name)) {
            return LFM2;
        }
        // GLM-4.5 / 4.6 / 4.7 MoE (GLM-4.5-Air etc.): GLM4 attention with bias and optional QK-norm,
        // leading dense blocks, sigmoid-routed experts with exp_probs_b and a shared expert. Runs on
        // Qwen3MoEInferenceEngine (the GLM4 + expertCount > 0 branch); RoPE is NEOX, unlike glm4.
        if ("glm4moe".equals(name)) {
            return GLM4;
        }
        if ("granite".equals(name)) {
            return GRANITE; // Standard transformer, own chat template + NEOX RoPE
        }
        throw new IllegalArgumentException("Unknown architecture: " + name);
    }
}
