package it.denzosoft.llmplayer.tuning;

import it.denzosoft.llmplayer.api.LLMEngine;
import it.denzosoft.llmplayer.gguf.GGUFFile;
import it.denzosoft.llmplayer.gguf.GGUFParser;
import it.denzosoft.llmplayer.gpu.GpuConfig;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.tokenizer.Tokenizer;
import it.denzosoft.llmplayer.tuning.analyze.TargetAnalyzer;
import it.denzosoft.llmplayer.tuning.dataset.Chunk;
import it.denzosoft.llmplayer.tuning.dataset.CodeChunker;
import it.denzosoft.llmplayer.tuning.dataset.DataChunker;
import it.denzosoft.llmplayer.tuning.dataset.QAGenerator;
import it.denzosoft.llmplayer.tuning.dataset.TextChunker;
import it.denzosoft.llmplayer.tuning.dataset.TokenCounter;
import it.denzosoft.llmplayer.tuning.merge.GGUFWriter;
import it.denzosoft.llmplayer.tuning.merge.LoRAMerger;
import it.denzosoft.llmplayer.tuning.train.LoRAAdapter;
import it.denzosoft.llmplayer.tuning.train.TrainingLoop;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Main orchestrator for the fine-tuning pipeline.
 * Runs stages 1-6 sequentially with checkpoint/resume support.
 */
public class FineTunePipeline {

    private final PipelineConfig config;

    public FineTunePipeline(PipelineConfig config) {
        this.config = config;
    }

    /** Run the full pipeline with checkpoint/resume. */
    public void run() throws IOException {
        // Banner
        System.out.println("=== LLMPlayer Fine-Tuning Pipeline ===");
        System.out.printf("  Target model : %s%n", config.targetModelPath());
        System.out.printf("  Generator    : %s%s%n", config.effectiveGeneratorModel(),
                config.sameGeneratorAndTarget() ? " (same as target)" : "");
        System.out.printf("  Data type    : %s%n", config.resolvedDataType());
        System.out.printf("  Input        : %s%n", config.inputPath());
        System.out.printf("  Output       : %s%n", config.outputPath());
        System.out.printf("  LoRA rank=%d alpha=%d | Epochs=%d LR=%.1e%n",
                config.loraRank(), config.loraAlpha(), config.epochs(), config.learningRate());
        System.out.printf("  GPU        : %s%n",
                config.noGpu() ? "disabled"
                    : config.gpuDeviceId() >= 0 ? "device " + config.gpuDeviceId()
                    : "auto-detect");
        System.out.println();

        Path workDir = Paths.get(config.workDir());
        if (!Files.isDirectory(workDir)) Files.createDirectories(workDir);

        // Resume check
        PipelineState state;
        if (PipelineState.hasCheckpoint(workDir)) {
            state = PipelineState.load(workDir);
            if (state == null) {
                state = new PipelineState();
            } else {
                System.out.printf("Resuming from stage %d (%s elapsed)%n%n",
                        state.currentStage(), state.elapsedFormatted());
            }
        } else {
            state = new PipelineState();
        }

        // Variables that survive across stages
        Tokenizer targetTokenizer = null;
        List<Chunk> chunks = null;
        List<LoRAAdapter> adapters = null;
        LLMEngine targetEngine = null;

        // --- Stage 1: Analyze target model ---
        if (state.currentStage() <= 1) {
            System.out.println("Stage 1/6: Analyzing target model...");
            TargetAnalyzer analyzer = new TargetAnalyzer();
            TargetAnalyzer.AnalysisResult result =
                    analyzer.analyze(Paths.get(config.targetModelPath()));
            analyzer.updateState(state, result);
            // Override chunk size if user specified --chunk-size
            if (config.chunkSize() > 0) {
                state.setMaxChunkTokens(config.chunkSize());
                // Scale response tokens: ~150 tok per Q&A pair for concise answers + JSON overhead
                int scaledResponse = config.pairsPerChunk() * 150 + 150;
                if (scaledResponse < state.maxResponseTokens()) {
                    state.setMaxResponseTokens(scaledResponse);
                }
            }
            targetTokenizer = result.tokenizer();
            System.out.printf("  Architecture: %s | Context: %d | Embedding: %d | Layers: %d%n",
                    result.config().architecture(), result.config().contextLength(),
                    result.config().embeddingLength(), result.config().blockCount());
            System.out.printf("  Max chunk tokens: %d%s | Max response tokens: %d%n%n",
                    state.maxChunkTokens(),
                    config.chunkSize() > 0 ? " (user)" : "",
                    state.maxResponseTokens());
            state.setCurrentStage(2);
            state.save(workDir);
        }

        // --- Stage 2: Parse and chunk input data ---
        if (state.currentStage() <= 2 && config.trainDataset() == null) {
            System.out.println("Stage 2/6: Parsing and chunking input data...");
            if (targetTokenizer == null) {
                targetTokenizer = analyzeTokenizer();
            }
            chunks = chunkInput(targetTokenizer, state);
            state.setTotalChunks(chunks.size());
            System.out.printf("  Chunks created: %d%n%n", chunks.size());
            state.setCurrentStage(3);
            state.save(workDir);
        }

        // --- Stage 3: Generate Q&A dataset ---
        if (state.currentStage() <= 3 && config.trainDataset() == null) {
            System.out.println("Stage 3/6: Generating training dataset...");
            // Re-chunk if needed (resume from checkpoint where chunks were not in memory)
            if (chunks == null) {
                if (targetTokenizer == null) targetTokenizer = analyzeTokenizer();
                chunks = chunkInput(targetTokenizer, state);
            }

            // Load generator engine with context sized for dataset generation
            // (no need for full model context, just prompt+response+margin)
            int genContext = state.maxChunkTokens() + state.maxResponseTokens() + 500;
            System.out.printf("  Generator context: %d tokens%n", genContext);
            LLMEngine generatorEngine;
            boolean separateGenerator;
            if (!config.datasetOnly() && config.sameGeneratorAndTarget()) {
                targetEngine = loadEngine(
                        Paths.get(config.targetModelPath()), genContext);
                generatorEngine = targetEngine;
                separateGenerator = false;
            } else {
                generatorEngine = loadEngine(
                        Paths.get(config.effectiveGeneratorModel()), genContext);
                separateGenerator = !config.sameGeneratorAndTarget();
            }

            Path datasetFile = workDir.resolve("dataset.jsonl");
            QAGenerator generator = new QAGenerator(generatorEngine,
                    config.resolvedDataType(), config.pairsPerChunk(),
                    state.maxResponseTokens(), state.targetEmbeddingLength());

            final PipelineState fState = state;
            final Path fWorkDir = workDir;
            int totalPairs = generator.generate(chunks, datasetFile, state.chunksCompleted(),
                    new QAGenerator.ProgressListener() {
                        public void onProgress(int chunkIndex, int totalChunks, int pairsGenerated) {
                            System.out.printf("  Processing chunk %d/%d...%n",
                                    chunkIndex + 1, totalChunks);
                        }
                        public void onChunkCompleted(int chunkIndex, int totalChunks,
                                                     int pairsInChunk, int totalPairs) {
                            fState.setChunksCompleted(chunkIndex + 1);
                            if ((chunkIndex + 1) % 5 == 0) {
                                try { fState.save(fWorkDir); } catch (IOException ignored) { }
                            }
                        }
                    });

            if (separateGenerator) generatorEngine.close();
            System.out.printf("  Dataset generated: %d Q&A pairs in %s%n%n", totalPairs, datasetFile);

            if (config.datasetOnly()) {
                state.setStatus("completed");
                state.save(workDir);
                System.out.println("=== Dataset-only mode: pipeline complete ===");
                System.out.printf("  Dataset: %s | Pairs: %d | Elapsed: %s%n",
                        datasetFile, totalPairs, state.elapsedFormatted());
                return;
            }
            state.setCurrentStage(4);
            state.save(workDir);
        }

        // --- Stage 4: Train LoRA adapters ---
        if (state.currentStage() <= 4) {
            System.out.println("Stage 4/6: Training LoRA adapters...");
            if (targetEngine == null) {
                targetEngine = loadEngine(
                        Paths.get(config.targetModelPath()), state.targetContextLength());
            }
            Tokenizer tokenizer = targetEngine.getTokenizer();
            ModelConfig modelConfig = targetEngine.getConfig();
            int embeddingLength = modelConfig.embeddingLength();
            int blockCount = modelConfig.blockCount();

            adapters = new ArrayList<LoRAAdapter>();
            Random rng = new Random(42);
            for (int layer = 0; layer < blockCount; layer++) {
                adapters.add(new LoRAAdapter("blk." + layer + ".attn_q",
                        embeddingLength, embeddingLength, config.loraRank(), config.loraAlpha(), rng));
                adapters.add(new LoRAAdapter("blk." + layer + ".attn_v",
                        embeddingLength, embeddingLength, config.loraRank(), config.loraAlpha(), rng));
            }

            TrainingLoop trainingLoop = new TrainingLoop(targetEngine, tokenizer,
                    adapters, state.targetContextLength(), config.learningRate());
            System.out.printf("  Trainable parameters: %,d%n", trainingLoop.totalParams());
            System.out.printf("  Adapters: %d (Q+V x %d layers)%n", adapters.size(), blockCount);
            state.setTotalEpochs(config.epochs());

            Path datasetFile = config.trainDataset() != null
                    ? Paths.get(config.trainDataset()) : workDir.resolve("dataset.jsonl");
            float finalLoss = trainingLoop.train(datasetFile, config.epochs(),
                    state.epochsCompleted(), workDir.resolve("checkpoints"), state,
                    new TrainingLoop.ProgressListener() {
                        public void onStep(int epoch, int totalEpochs, int step,
                                           int totalSteps, float avgLoss) {
                            System.out.printf("  Epoch %d/%d  Step %d/%d  Loss: %.4f%n",
                                    epoch + 1, totalEpochs, step, totalSteps, avgLoss);
                        }
                        public void onEpochComplete(int epoch, int totalEpochs, float epochLoss) {
                            System.out.printf("  Epoch %d/%d complete  Loss: %.4f%n",
                                    epoch + 1, totalEpochs, epochLoss);
                        }
                    });

            System.out.printf("  Training complete. Final loss: %.4f%n%n", finalLoss);
            state.setCurrentStage(5);
            state.save(workDir);
        }

        // --- Stage 5: Merge LoRA weights into base model ---
        List<LoRAMerger.MergedTensor> mergedTensors = null;
        GGUFFile gguf = null;
        if (state.currentStage() <= 5) {
            System.out.println("Stage 5/6: Merging LoRA weights...");
            if (adapters == null) {
                throw new IOException("LoRA adapters not available. "
                        + "Cannot resume stage 5 without completing stage 4 in this run.");
            }
            gguf = GGUFParser.parse(Paths.get(config.targetModelPath()), true);
            mergedTensors = new LoRAMerger().merge(gguf, adapters);
            System.out.printf("  Merged %d tensors%n%n", mergedTensors.size());
            state.setCurrentStage(6);
            state.save(workDir);
        }

        // --- Stage 6: Export fine-tuned GGUF ---
        if (state.currentStage() <= 6) {
            System.out.println("Stage 6/6: Writing fine-tuned GGUF...");
            if (mergedTensors == null || gguf == null) {
                throw new IOException("Merged tensors not available. "
                        + "Cannot resume stage 6 without completing stage 5 in this run.");
            }
            Path outputPath = Paths.get(config.outputPath());
            new GGUFWriter().write(Paths.get(config.targetModelPath()), gguf, mergedTensors, outputPath);
            gguf.close();

            state.setStatus("completed");
            state.save(workDir);

            long fileSize = Files.exists(outputPath) ? Files.size(outputPath) : 0;
            System.out.println();
            System.out.println("=== Fine-tuning complete ===");
            System.out.printf("  Output: %s%n", outputPath);
            System.out.printf("  File size: %.1f MB%n", fileSize / (1024.0 * 1024.0));
            System.out.printf("  Elapsed: %s%n", state.elapsedFormatted());
        }

        // Cleanup
        if (targetEngine != null) targetEngine.close();
    }

    /** Load a model with GPU configuration from pipeline config. */
    private LLMEngine loadEngine(Path modelPath, int contextLength) throws IOException {
        if (config.noGpu()) {
            // Explicitly disable GPU: pass a disabled GpuConfig to prevent auto-detection
            GpuConfig off = new GpuConfig();
            off.setEnabled(false);
            return LLMEngine.load(modelPath, contextLength, off);
        }
        GpuConfig gpuCfg = new GpuConfig();
        gpuCfg.setEnabled(true);
        if (config.gpuDeviceId() >= 0) {
            gpuCfg.setDeviceId(config.gpuDeviceId());
        } else {
            // Use auto-detection (prefers real GPU over CPU device)
            GpuConfig autoGpu = LLMEngine.autoConfigureGpu(modelPath);
            if (autoGpu != null) {
                gpuCfg.setDeviceId(autoGpu.getDeviceId());
            }
        }
        gpuCfg.setGpuLayers(config.gpuLayers());
        return LLMEngine.load(modelPath, contextLength, gpuCfg);
    }

    /** Quick-analyze the target model to obtain its tokenizer. */
    private Tokenizer analyzeTokenizer() throws IOException {
        TargetAnalyzer analyzer = new TargetAnalyzer();
        TargetAnalyzer.AnalysisResult result =
                analyzer.analyze(Paths.get(config.targetModelPath()));
        return result.tokenizer();
    }

    /** Chunk input data using the appropriate chunker for the configured data type. */
    private List<Chunk> chunkInput(Tokenizer tokenizer, PipelineState state) throws IOException {
        final Tokenizer tok = tokenizer;
        TokenCounter counter = new TokenCounter() {
            public int countTokens(String text) {
                return tok.encode(text).length;
            }
        };
        CodeChunker.TokenCounter codeCounter = new CodeChunker.TokenCounter() {
            public int countTokens(String text) {
                return tok.encode(text).length;
            }
        };

        String dataType = config.resolvedDataType();
        if ("code".equals(dataType)) {
            return new CodeChunker(state.maxChunkTokens(), config.chunkOverlap(), codeCounter)
                    .chunk(Paths.get(config.inputPath()));
        } else if ("document".equals(dataType)) {
            return new TextChunker(state.maxChunkTokens(), config.chunkOverlap(), counter)
                    .chunk(Paths.get(config.inputPath()));
        } else {
            Path schemaPath = config.schemaPath() != null ? Paths.get(config.schemaPath()) : null;
            return new DataChunker(state.maxChunkTokens(), config.chunkOverlap(), counter)
                    .chunk(Paths.get(config.inputPath()), schemaPath);
        }
    }
}
