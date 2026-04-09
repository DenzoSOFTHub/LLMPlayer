package it.denzosoft.llmplayer.api;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import javax.management.ObjectName;

/**
 * JMX MXBean implementation exposing LLMPlayer runtime metrics.
 * Thread-safe: uses atomics for all mutable state.
 */
public class LLMPlayerMetrics implements LLMPlayerMXBean {

    private static final String OBJECT_NAME = "it.denzosoft.llmplayer:type=LLMPlayer";

    // Model info (set once at load)
    private volatile String modelName = "";
    private volatile String architecture = "";
    private volatile long modelFileSizeMB;
    private volatile int contextLength;
    private volatile int blockCount;
    private volatile int embeddingLength;
    private volatile int vocabSize;
    private volatile String quantization = "";

    // GPU info (set once at load)
    private volatile boolean gpuEnabled;
    private volatile String gpuDeviceName = "";
    private volatile int gpuLayersUsed;
    private volatile int gpuLayersTotal;
    private volatile boolean moeOptimizedGpu;
    private volatile long kvCacheEstimateMB;

    // GPU VRAM query (via reflection to avoid java21 dependency)
    private volatile Object cudaContext; // CudaContext instance for VRAM queries

    // Generation stats (updated atomically)
    private final AtomicLong totalGenerations = new AtomicLong();
    private final AtomicLong totalTokensGenerated = new AtomicLong();
    private final AtomicLong totalPromptTokens = new AtomicLong();
    private final AtomicLong totalGenerationTimeMs = new AtomicLong();
    private final AtomicReference<Double> lastTokPerSec = new AtomicReference<>(0.0);
    private final AtomicLong lastGenerationTimeMs = new AtomicLong();

    // Singleton per JVM
    private static volatile LLMPlayerMetrics instance;

    public static LLMPlayerMetrics getInstance() {
        if (instance == null) {
            synchronized (LLMPlayerMetrics.class) {
                if (instance == null) {
                    instance = new LLMPlayerMetrics();
                    instance.register();
                }
            }
        }
        return instance;
    }

    private void register() {
        try {
            ObjectName name = new ObjectName(OBJECT_NAME);
            javax.management.MBeanServer server = ManagementFactory.getPlatformMBeanServer();
            if (server.isRegistered(name)) {
                server.unregisterMBean(name);
            }
            server.registerMBean(this, name);
        } catch (Exception e) {
            System.err.println("[JMX] Failed to register LLMPlayerMXBean: " + e.getMessage());
        }
    }

    /** Called by LLMEngine after model load to populate model info. */
    public void setModelInfo(String modelName, String architecture, long modelFileSizeMB,
                             int contextLength, int blockCount, int embeddingLength, int vocabSize,
                             String quantization) {
        this.modelName = modelName != null ? modelName : "";
        this.architecture = architecture != null ? architecture : "";
        this.modelFileSizeMB = modelFileSizeMB;
        this.contextLength = contextLength;
        this.blockCount = blockCount;
        this.embeddingLength = embeddingLength;
        this.vocabSize = vocabSize;
        this.quantization = quantization != null ? quantization : "";
    }

    /** Called by LLMEngine after model load to populate GPU info. */
    public void setGpuInfo(boolean gpuEnabled, String gpuDeviceName, int gpuLayersUsed,
                           int gpuLayersTotal, boolean moeOptimizedGpu, long kvCacheEstimateMB) {
        this.gpuEnabled = gpuEnabled;
        this.gpuDeviceName = gpuDeviceName != null ? gpuDeviceName : "";
        this.gpuLayersUsed = gpuLayersUsed;
        this.gpuLayersTotal = gpuLayersTotal;
        this.moeOptimizedGpu = moeOptimizedGpu;
        this.kvCacheEstimateMB = kvCacheEstimateMB;
    }

    /** Set CudaContext for VRAM queries (called from java21 code via reflection). */
    public void setCudaContext(Object cudaContext) {
        this.cudaContext = cudaContext;
    }

    /** Called after each generation to update stats. */
    public void recordGeneration(int genTokens, int promptTokens, double tokPerSec, long timeMs) {
        totalGenerations.incrementAndGet();
        totalTokensGenerated.addAndGet(genTokens);
        totalPromptTokens.addAndGet(promptTokens);
        totalGenerationTimeMs.addAndGet(timeMs);
        lastTokPerSec.set(tokPerSec);
        lastGenerationTimeMs.set(timeMs);
    }

    /** Called when model is unloaded. */
    public void reset() {
        modelName = "";
        architecture = "";
        modelFileSizeMB = 0;
        gpuEnabled = false;
        gpuDeviceName = "";
        gpuLayersUsed = 0;
        gpuLayersTotal = 0;
        totalGenerations.set(0);
        totalTokensGenerated.set(0);
        totalPromptTokens.set(0);
        totalGenerationTimeMs.set(0);
        lastTokPerSec.set(0.0);
        lastGenerationTimeMs.set(0);
        cudaContext = null;
    }

    // --- Model info ---
    @Override public String getModelName() { return modelName; }
    @Override public String getArchitecture() { return architecture; }
    @Override public long getModelFileSizeMB() { return modelFileSizeMB; }
    @Override public int getContextLength() { return contextLength; }
    @Override public int getBlockCount() { return blockCount; }
    @Override public int getEmbeddingLength() { return embeddingLength; }
    @Override public int getVocabSize() { return vocabSize; }
    @Override public String getQuantization() { return quantization; }

    // --- Generation stats ---
    @Override public long getTotalGenerations() { return totalGenerations.get(); }
    @Override public long getTotalTokensGenerated() { return totalTokensGenerated.get(); }
    @Override public long getTotalPromptTokens() { return totalPromptTokens.get(); }
    @Override public double getLastTokensPerSecond() { return lastTokPerSec.get(); }
    @Override public long getLastGenerationTimeMs() { return lastGenerationTimeMs.get(); }
    @Override public long getTotalGenerationTimeMs() { return totalGenerationTimeMs.get(); }

    @Override
    public double getAverageTokensPerSecond() {
        long tokens = totalTokensGenerated.get();
        long timeMs = totalGenerationTimeMs.get();
        return timeMs > 0 ? tokens * 1000.0 / timeMs : 0.0;
    }

    // --- Memory ---
    @Override
    public long getHeapUsedMB() {
        MemoryMXBean mem = ManagementFactory.getMemoryMXBean();
        return mem.getHeapMemoryUsage().getUsed() / (1024 * 1024);
    }

    @Override
    public long getHeapMaxMB() {
        MemoryMXBean mem = ManagementFactory.getMemoryMXBean();
        long max = mem.getHeapMemoryUsage().getMax();
        return max > 0 ? max / (1024 * 1024) : Runtime.getRuntime().maxMemory() / (1024 * 1024);
    }

    @Override
    public long getOffHeapUsedMB() {
        MemoryMXBean mem = ManagementFactory.getMemoryMXBean();
        return mem.getNonHeapMemoryUsage().getUsed() / (1024 * 1024);
    }

    // --- GPU ---
    @Override public boolean isGpuEnabled() { return gpuEnabled; }
    @Override public String getGpuDeviceName() { return gpuDeviceName; }
    @Override public int getGpuLayersUsed() { return gpuLayersUsed; }
    @Override public int getGpuLayersTotal() { return gpuLayersTotal; }
    @Override public boolean isMoeOptimizedGpu() { return moeOptimizedGpu; }

    @Override
    public long getGpuVramTotalMB() {
        long[] info = queryVram();
        return info != null ? info[1] / (1024 * 1024) : -1;
    }

    @Override
    public long getGpuVramFreeMB() {
        long[] info = queryVram();
        return info != null ? info[0] / (1024 * 1024) : -1;
    }

    private long[] queryVram() {
        Object ctx = this.cudaContext;
        if (ctx == null) return null;
        try {
            return (long[]) ctx.getClass().getMethod("getMemoryInfo").invoke(ctx);
        } catch (Throwable t) {
            return null;
        }
    }

    // --- KV Cache ---
    @Override public long getKvCacheEstimateMB() { return kvCacheEstimateMB; }
}
