package it.denzosoft.llmplayer.api;

import it.denzosoft.llmplayer.model.ModelConfig;

public final class ModelInfo {
    private final String name;
    private final String architecture;
    private final int embeddingLength;
    private final int blockCount;
    private final int headCount;
    private final int headCountKV;
    private final int contextLength;
    private final int vocabSize;
    private final int intermediateSize;
    private final long modelFileSize;
    private final long kvCacheEstimate;
    private final long totalRamEstimate;

    public ModelInfo(String name, String architecture, int embeddingLength, int blockCount,
                     int headCount, int headCountKV, int contextLength, int vocabSize,
                     int intermediateSize, long modelFileSize, long kvCacheEstimate, long totalRamEstimate) {
        this.name = name;
        this.architecture = architecture;
        this.embeddingLength = embeddingLength;
        this.blockCount = blockCount;
        this.headCount = headCount;
        this.headCountKV = headCountKV;
        this.contextLength = contextLength;
        this.vocabSize = vocabSize;
        this.intermediateSize = intermediateSize;
        this.modelFileSize = modelFileSize;
        this.kvCacheEstimate = kvCacheEstimate;
        this.totalRamEstimate = totalRamEstimate;
    }

    public String name() { return name; }
    public String architecture() { return architecture; }
    public int embeddingLength() { return embeddingLength; }
    public int blockCount() { return blockCount; }
    public int headCount() { return headCount; }
    public int headCountKV() { return headCountKV; }
    public int contextLength() { return contextLength; }
    public int vocabSize() { return vocabSize; }
    public int intermediateSize() { return intermediateSize; }
    public long modelFileSize() { return modelFileSize; }
    public long kvCacheEstimate() { return kvCacheEstimate; }
    public long totalRamEstimate() { return totalRamEstimate; }

    public static ModelInfo from(ModelConfig config, long modelFileSize, long kvCacheEstimate) {
        long totalRam = modelFileSize + kvCacheEstimate;
        return new ModelInfo(
            config.name(),
            config.architecture().getGgufName(),
            config.embeddingLength(),
            config.blockCount(),
            config.headCount(),
            config.headCountKV(),
            config.contextLength(),
            config.vocabSize(),
            config.intermediateSize(),
            modelFileSize,
            kvCacheEstimate,
            totalRam
        );
    }

    @Override
    public String toString() {
        return String.format("Model: %s (%s)\n  Embedding: %d, Layers: %d, Heads: %d/%d\n  Context: %d, Vocab: %d, FFN: %d",
            name, architecture, embeddingLength, blockCount, headCount, headCountKV,
            contextLength, vocabSize, intermediateSize);
    }
}
