package it.denzosoft.llmplayer.tuning.analyze;

import it.denzosoft.llmplayer.gguf.GGUFFile;
import it.denzosoft.llmplayer.gguf.GGUFMetadata;
import it.denzosoft.llmplayer.gguf.GGUFParser;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.tokenizer.Tokenizer;
import it.denzosoft.llmplayer.tokenizer.TokenizerFactory;
import it.denzosoft.llmplayer.tuning.PipelineState;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Stage 1: Quick-parse the target GGUF model to extract metadata
 * needed for dataset generation (tokenizer, context length, architecture).
 * Does NOT load model weights.
 */
public class TargetAnalyzer {

    /** Analysis results. */
    public static class AnalysisResult {
        private final ModelConfig config;
        private final Tokenizer tokenizer;
        private final int maxChunkTokens;
        private final int maxResponseTokens;

        public AnalysisResult(ModelConfig config, Tokenizer tokenizer,
                              int maxChunkTokens, int maxResponseTokens) {
            this.config = config;
            this.tokenizer = tokenizer;
            this.maxChunkTokens = maxChunkTokens;
            this.maxResponseTokens = maxResponseTokens;
        }

        public ModelConfig config() { return config; }
        public Tokenizer tokenizer() { return tokenizer; }
        public int maxChunkTokens() { return maxChunkTokens; }
        public int maxResponseTokens() { return maxResponseTokens; }
    }

    /**
     * Analyze the target model GGUF without loading weights.
     * Extracts architecture, tokenizer, and computes token budgets.
     */
    public AnalysisResult analyze(Path targetModel) throws IOException {
        // Parse GGUF with preload=false to skip weight loading
        GGUFFile gguf = GGUFParser.parse(targetModel, false);
        GGUFMetadata metadata = gguf.getMetadata();

        // Extract model configuration
        ModelConfig config = ModelConfig.fromMetadata(metadata);

        // Build tokenizer from GGUF metadata
        Tokenizer tokenizer = TokenizerFactory.create(metadata);

        // Compute token budgets
        int contextLength = config.contextLength();
        // Template overhead: BOS + chat template tokens + EOS per message ≈ 50-100 tokens
        int templateOverhead = 100;
        int availableTokens = contextLength - templateOverhead;
        // Split equally: 45% chunk, 45% response, 10% margin
        int maxChunkTokens = (int) (availableTokens * 0.45);
        int maxResponseTokens = (int) (availableTokens * 0.45);

        // Close the GGUF file handle (we only needed metadata)
        gguf.close();

        return new AnalysisResult(config, tokenizer, maxChunkTokens, maxResponseTokens);
    }

    /** Store analysis results in pipeline state. */
    public void updateState(PipelineState state, AnalysisResult result) {
        state.setTargetArchitecture(result.config().architecture().name());
        state.setTargetContextLength(result.config().contextLength());
        state.setTargetEmbeddingLength(result.config().embeddingLength());
        state.setMaxChunkTokens(result.maxChunkTokens());
        state.setMaxResponseTokens(result.maxResponseTokens());
    }
}
