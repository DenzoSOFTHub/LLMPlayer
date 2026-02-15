package it.denzosoft.llmplayer.model;

public enum ModelArchitecture {
    LLAMA("llama"),
    QWEN2("qwen2"),
    QWEN3("qwen3"),
    DEEPSEEK2("deepseek2"),
    GLM4("glm4"),
    PHI3("phi3"),
    QWEN3MOE("qwen3moe"),
    MISTRAL3("mistral3");

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
        throw new IllegalArgumentException("Unknown architecture: " + name);
    }
}
