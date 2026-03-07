package it.denzosoft.llmplayer.tuning.dataset;

/** Functional interface for counting tokens in a text string. */
public interface TokenCounter {
    int countTokens(String text);
}
