package it.denzosoft.llmplayer.tuning;

/**
 * Configuration for the fine-tuning pipeline. All fields are immutable
 * and set at construction time from CLI arguments.
 */
public class PipelineConfig {

    // Input
    private final String sourcePath;       // --source (code scenario)
    private final String documentsPath;    // --documents (document scenario)
    private final String dataPath;         // --data (structured scenario)
    private final String schemaPath;       // --schema (SQL DDL for structured)
    private final String dataType;         // code, document, structured (auto-detect if null)

    // Models
    private final String targetModelPath;
    private final String generatorModelPath; // null = use target as generator
    private final String outputPath;

    // Dataset generation
    private final int pairsPerChunk;
    private final int chunkOverlap;
    private final int chunkSize;       // 0 = auto (45% of context), >0 = user override

    // LoRA
    private final int loraRank;
    private final int loraAlpha;

    // Training
    private final int epochs;
    private final float learningRate;

    // Output
    private final String quantization; // null = same as target

    // GPU
    private final boolean noGpu;
    private final int gpuDeviceId;     // -1 = auto-detect
    private final int gpuLayers;       // -1 = auto

    // Pipeline control
    private final String workDir;
    private final boolean datasetOnly;
    private final String trainDataset; // null = generate, non-null = skip to training

    private PipelineConfig(Builder b) {
        this.sourcePath = b.sourcePath;
        this.documentsPath = b.documentsPath;
        this.dataPath = b.dataPath;
        this.schemaPath = b.schemaPath;
        this.dataType = b.dataType;
        this.targetModelPath = b.targetModelPath;
        this.generatorModelPath = b.generatorModelPath;
        this.outputPath = b.outputPath;
        this.pairsPerChunk = b.pairsPerChunk;
        this.chunkOverlap = b.chunkOverlap;
        this.chunkSize = b.chunkSize;
        this.loraRank = b.loraRank;
        this.loraAlpha = b.loraAlpha;
        this.epochs = b.epochs;
        this.learningRate = b.learningRate;
        this.quantization = b.quantization;
        this.noGpu = b.noGpu;
        this.gpuDeviceId = b.gpuDeviceId;
        this.gpuLayers = b.gpuLayers;
        this.workDir = b.workDir;
        this.datasetOnly = b.datasetOnly;
        this.trainDataset = b.trainDataset;
    }

    /** Resolve data type from explicit flag or input paths. */
    public String resolvedDataType() {
        if (dataType != null) return dataType;
        if (sourcePath != null) return "code";
        if (documentsPath != null) return "document";
        if (dataPath != null) return "structured";
        return "code";
    }

    /** The effective generator model path (falls back to target if not specified). */
    public String effectiveGeneratorModel() {
        return generatorModelPath != null ? generatorModelPath : targetModelPath;
    }

    /** Whether the generator and target are the same model. */
    public boolean sameGeneratorAndTarget() {
        return generatorModelPath == null || generatorModelPath.equals(targetModelPath);
    }

    /** Input path for the resolved data type. */
    public String inputPath() {
        switch (resolvedDataType()) {
            case "code": return sourcePath;
            case "document": return documentsPath;
            case "structured": return dataPath;
            default: return sourcePath;
        }
    }

    // --- Getters ---

    public String sourcePath() { return sourcePath; }
    public String documentsPath() { return documentsPath; }
    public String dataPath() { return dataPath; }
    public String schemaPath() { return schemaPath; }
    public String dataType() { return dataType; }
    public String targetModelPath() { return targetModelPath; }
    public String generatorModelPath() { return generatorModelPath; }
    public String outputPath() { return outputPath; }
    public int pairsPerChunk() { return pairsPerChunk; }
    public int chunkOverlap() { return chunkOverlap; }
    public int chunkSize() { return chunkSize; }
    public int loraRank() { return loraRank; }
    public int loraAlpha() { return loraAlpha; }
    public int epochs() { return epochs; }
    public float learningRate() { return learningRate; }
    public String quantization() { return quantization; }
    public boolean noGpu() { return noGpu; }
    public int gpuDeviceId() { return gpuDeviceId; }
    public int gpuLayers() { return gpuLayers; }
    public String workDir() { return workDir; }
    public boolean datasetOnly() { return datasetOnly; }
    public String trainDataset() { return trainDataset; }

    public static Builder builder() { return new Builder(); }

    public static class Builder {
        private String sourcePath;
        private String documentsPath;
        private String dataPath;
        private String schemaPath;
        private String dataType;
        private String targetModelPath;
        private String generatorModelPath;
        private String outputPath;
        private int pairsPerChunk = 5;
        private int chunkOverlap = 100;
        private int chunkSize = 0;
        private int loraRank = 16;
        private int loraAlpha = 32;
        private int epochs = 3;
        private float learningRate = 2e-4f;
        private String quantization;
        private boolean noGpu;
        private int gpuDeviceId = -1;
        private int gpuLayers = -1;
        private String workDir = "work";
        private boolean datasetOnly;
        private String trainDataset;

        public Builder sourcePath(String v) { sourcePath = v; return this; }
        public Builder documentsPath(String v) { documentsPath = v; return this; }
        public Builder dataPath(String v) { dataPath = v; return this; }
        public Builder schemaPath(String v) { schemaPath = v; return this; }
        public Builder dataType(String v) { dataType = v; return this; }
        public Builder targetModelPath(String v) { targetModelPath = v; return this; }
        public Builder generatorModelPath(String v) { generatorModelPath = v; return this; }
        public Builder outputPath(String v) { outputPath = v; return this; }
        public Builder pairsPerChunk(int v) { pairsPerChunk = v; return this; }
        public Builder chunkOverlap(int v) { chunkOverlap = v; return this; }
        public Builder chunkSize(int v) { chunkSize = v; return this; }
        public Builder loraRank(int v) { loraRank = v; return this; }
        public Builder loraAlpha(int v) { loraAlpha = v; return this; }
        public Builder epochs(int v) { epochs = v; return this; }
        public Builder learningRate(float v) { learningRate = v; return this; }
        public Builder quantization(String v) { quantization = v; return this; }
        public Builder noGpu(boolean v) { noGpu = v; return this; }
        public Builder gpuDeviceId(int v) { gpuDeviceId = v; return this; }
        public Builder gpuLayers(int v) { gpuLayers = v; return this; }
        public Builder workDir(String v) { workDir = v; return this; }
        public Builder datasetOnly(boolean v) { datasetOnly = v; return this; }
        public Builder trainDataset(String v) { trainDataset = v; return this; }

        public PipelineConfig build() {
            if (targetModelPath == null) throw new IllegalArgumentException("--target-model is required");
            if (outputPath == null) throw new IllegalArgumentException("--output is required");
            if (sourcePath == null && documentsPath == null && dataPath == null && trainDataset == null) {
                throw new IllegalArgumentException("One of --source, --documents, --data, or --train-dataset is required");
            }
            return new PipelineConfig(this);
        }
    }
}
