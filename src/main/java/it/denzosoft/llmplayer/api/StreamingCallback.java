package it.denzosoft.llmplayer.api;

@FunctionalInterface
public interface StreamingCallback {
    /** @return true to continue generation, false to stop */
    boolean onToken(String token, int tokenId);
}
