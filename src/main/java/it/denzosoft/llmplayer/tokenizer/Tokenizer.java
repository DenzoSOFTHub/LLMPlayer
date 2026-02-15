package it.denzosoft.llmplayer.tokenizer;

public interface Tokenizer {
    int[] encode(String text);
    String decode(int[] tokens);
    String decode(int token);
    int vocabSize();
    boolean isSpecialToken(int tokenId);
}
