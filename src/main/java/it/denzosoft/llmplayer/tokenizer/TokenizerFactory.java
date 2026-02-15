package it.denzosoft.llmplayer.tokenizer;

import it.denzosoft.llmplayer.gguf.GGUFMetadata;

public final class TokenizerFactory {

    private TokenizerFactory() {}

    public static Tokenizer create(GGUFMetadata metadata) {
        String[] tokens = metadata.getStringArray("tokenizer.ggml.tokens");
        if (tokens == null) {
            throw new IllegalStateException("No tokenizer.ggml.tokens found in metadata");
        }

        float[] scores = metadata.getFloatArray("tokenizer.ggml.scores");
        if (scores == null) {
            scores = new float[tokens.length];
        }

        String model = metadata.getString("tokenizer.ggml.model", "llama");
        SpecialTokens specialTokens = SpecialTokens.fromMetadata(metadata);

        String[] merges = metadata.getStringArray("tokenizer.ggml.merges");

        if ("gpt2".equals(model) || "bpe".equals(model)) {
            return new BPETokenizer(tokens, scores, merges, specialTokens);
        } else {
            return new SentencePieceTokenizer(tokens, scores, specialTokens);
        }
    }
}
