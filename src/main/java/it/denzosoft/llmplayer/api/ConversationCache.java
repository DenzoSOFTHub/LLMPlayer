package it.denzosoft.llmplayer.api;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Caches inference state (including KV cache) between requests for the same conversation.
 * When a multi-turn conversation reuses the same prefix, the KV cache is reused and only
 * the new tokens need to be processed.
 *
 * Thread-safe: entries are atomically removed during use and put back after generation.
 */
public class ConversationCache {

    private static final long TTL_MS = 5 * 60 * 1000; // 5 minutes
    private static final int MAX_ENTRIES = 4;

    private final Map<String, CachedConversation> cache = new ConcurrentHashMap<>();

    public static class CachedConversation {
        public final Object state;           // InferenceState, DeepSeek2State, or Qwen3MoEState
        public final int[] promptTokens;     // tokens that were processed during prefill
        public volatile long lastAccessMs;

        public CachedConversation(Object state, int[] promptTokens) {
            this.state = state;
            this.promptTokens = promptTokens;
            this.lastAccessMs = System.currentTimeMillis();
        }
    }

    /**
     * Atomically remove and return a cached conversation.
     * The caller has exclusive access to the state until put() is called.
     */
    public CachedConversation take(String key) {
        CachedConversation cached = cache.remove(key);
        if (cached != null) {
            if (System.currentTimeMillis() - cached.lastAccessMs > TTL_MS) {
                return null; // expired
            }
        }
        return cached;
    }

    /**
     * Store a conversation state in the cache.
     */
    public void put(String key, CachedConversation entry) {
        evictExpired();
        if (cache.size() >= MAX_ENTRIES) {
            evictOldest();
        }
        entry.lastAccessMs = System.currentTimeMillis();
        cache.put(key, entry);
    }

    private void evictExpired() {
        long now = System.currentTimeMillis();
        cache.entrySet().removeIf(e -> now - e.getValue().lastAccessMs > TTL_MS);
    }

    private void evictOldest() {
        String oldestKey = null;
        long oldestTime = Long.MAX_VALUE;
        for (Map.Entry<String, CachedConversation> e : cache.entrySet()) {
            if (e.getValue().lastAccessMs < oldestTime) {
                oldestTime = e.getValue().lastAccessMs;
                oldestKey = e.getKey();
            }
        }
        if (oldestKey != null) {
            cache.remove(oldestKey);
        }
    }

    public void clear() {
        cache.clear();
    }

    public int size() {
        return cache.size();
    }
}
