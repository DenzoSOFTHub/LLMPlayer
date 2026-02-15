package it.denzosoft.llmplayer.api;

import it.denzosoft.llmplayer.sampler.SamplerConfig;

public final class GenerationRequest {
    private final String prompt;
    private final String systemMessage;
    private final int maxTokens;
    private final SamplerConfig samplerConfig;
    private final boolean useChat;

    public GenerationRequest(String prompt, String systemMessage, int maxTokens,
                             SamplerConfig samplerConfig, boolean useChat) {
        this.prompt = prompt;
        this.systemMessage = systemMessage;
        this.maxTokens = maxTokens;
        this.samplerConfig = samplerConfig;
        this.useChat = useChat;
    }

    public String prompt() { return prompt; }
    public String systemMessage() { return systemMessage; }
    public int maxTokens() { return maxTokens; }
    public SamplerConfig samplerConfig() { return samplerConfig; }
    public boolean useChat() { return useChat; }

    public static Builder builder() { return new Builder(); }

    public static class Builder {
        private String prompt = "";
        private String systemMessage = null;
        private int maxTokens = 256;
        private SamplerConfig samplerConfig = SamplerConfig.DEFAULT;
        private boolean useChat = true;

        public Builder prompt(String p) { this.prompt = p; return this; }
        public Builder systemMessage(String s) { this.systemMessage = s; return this; }
        public Builder maxTokens(int m) { this.maxTokens = m; return this; }
        public Builder samplerConfig(SamplerConfig sc) { this.samplerConfig = sc; return this; }
        public Builder useChat(boolean uc) { this.useChat = uc; return this; }
        public GenerationRequest build() {
            return new GenerationRequest(prompt, systemMessage, maxTokens, samplerConfig, useChat);
        }
    }
}
