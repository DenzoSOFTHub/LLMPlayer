package it.denzosoft.llmplayer.api;

/**
 * JMX MXBean interface for LLMPlayer runtime metrics.
 * Exposes model info, generation stats, memory usage, and GPU status.
 *
 * Connect via JConsole/VisualVM: domain "it.denzosoft.llmplayer", type "LLMPlayer".
 */
public interface LLMPlayerMXBean {

    // --- Model info ---
    String getModelName();
    String getArchitecture();
    long getModelFileSizeMB();
    int getContextLength();
    int getBlockCount();
    int getEmbeddingLength();
    int getVocabSize();
    String getQuantization();

    // --- Generation stats ---
    long getTotalGenerations();
    long getTotalTokensGenerated();
    long getTotalPromptTokens();
    double getLastTokensPerSecond();
    double getAverageTokensPerSecond();
    double getRecentTokensPerSecond();   // 60-second rolling window
    int getRecentSampleCount();           // generations counted in rolling window
    long getLastGenerationTimeMs();
    long getTotalGenerationTimeMs();

    // --- Memory ---
    long getHeapUsedMB();
    long getHeapMaxMB();
    long getOffHeapUsedMB();

    // --- GPU ---
    boolean isGpuEnabled();
    String getGpuDeviceName();
    int getGpuLayersUsed();
    int getGpuLayersTotal();
    boolean isMoeOptimizedGpu();
    long getGpuVramTotalMB();
    long getGpuVramFreeMB();

    // --- KV Cache ---
    long getKvCacheEstimateMB();
}
