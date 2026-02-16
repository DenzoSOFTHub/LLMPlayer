package it.denzosoft.llmplayer.api;

import it.denzosoft.llmplayer.sampler.SamplerConfig;

public final class GenerationRequest {
    private final String prompt;
    private final String systemMessage;
    private final int maxTokens;
    private final SamplerConfig samplerConfig;
    private final boolean useChat;
    private final boolean rawMode;
    private final String cacheKey;

    public GenerationRequest(String prompt, String systemMessage, int maxTokens,
                             SamplerConfig samplerConfig, boolean useChat) {
        this(prompt, systemMessage, maxTokens, samplerConfig, useChat, false, null);
    }

    public GenerationRequest(String prompt, String systemMessage, int maxTokens,
                             SamplerConfig samplerConfig, boolean useChat, boolean rawMode) {
        this(prompt, systemMessage, maxTokens, samplerConfig, useChat, rawMode, null);
    }

    public GenerationRequest(String prompt, String systemMessage, int maxTokens,
                             SamplerConfig samplerConfig, boolean useChat, boolean rawMode,
                             String cacheKey) {
        this.prompt = prompt;
        this.systemMessage = systemMessage;
        this.maxTokens = maxTokens;
        this.samplerConfig = samplerConfig;
        this.useChat = useChat;
        this.rawMode = rawMode;
        this.cacheKey = cacheKey;
    }

    public String prompt() { return prompt; }
    public String systemMessage() { return systemMessage; }
    public int maxTokens() { return maxTokens; }
    public SamplerConfig samplerConfig() { return samplerConfig; }
    public boolean useChat() { return useChat; }
    /** True if prompt is pre-formatted (multi-turn) but still needs BOS prepended. */
    public boolean rawMode() { return rawMode; }
    /** Optional cache key for KV cache reuse across requests. */
    public String cacheKey() { return cacheKey; }

    public static Builder builder() { return new Builder(); }

    public static class Builder {
        private String prompt = "";
        private String systemMessage = null;
        private int maxTokens = 256;
        private SamplerConfig samplerConfig = SamplerConfig.DEFAULT;
        private boolean useChat = true;
        private boolean rawMode = false;
        private String cacheKey = null;

        public Builder prompt(String p) { this.prompt = p; return this; }
        public Builder systemMessage(String s) { this.systemMessage = s; return this; }
        public Builder maxTokens(int m) { this.maxTokens = m; return this; }
        public Builder samplerConfig(SamplerConfig sc) { this.samplerConfig = sc; return this; }
        public Builder useChat(boolean uc) { this.useChat = uc; return this; }
        public Builder rawMode(boolean rm) { this.rawMode = rm; return this; }
        public Builder cacheKey(String ck) { this.cacheKey = ck; return this; }
        public GenerationRequest build() {
            return new GenerationRequest(prompt, systemMessage, maxTokens, samplerConfig, useChat, rawMode, cacheKey);
        }
    }
}
