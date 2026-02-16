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
    OLMO2("olmo2"),
    GEMMA2("gemma2"),
    GEMMA3("gemma3"),
    LLAMA4("llama4"),
    GPT_OSS("gpt-oss");

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
        if ("command_r".equals(name) || "cohere".equals(name) || "cohere2".equals(name)) {
            return COMMAND_R;
        }
        if ("gemma".equals(name)) {
            return GEMMA2; // Gemma1 uses same forward pass
        }
        throw new IllegalArgumentException("Unknown architecture: " + name);
    }
}
