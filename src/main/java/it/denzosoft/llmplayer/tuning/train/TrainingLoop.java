package it.denzosoft.llmplayer.tuning.train;

import it.denzosoft.llmplayer.api.LLMEngine;
import it.denzosoft.llmplayer.tokenizer.Tokenizer;
import it.denzosoft.llmplayer.tuning.PipelineState;
import it.denzosoft.llmplayer.web.ApiHandler;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * Stage 4: LoRA training loop.
 * Reads a JSONL dataset, performs forward/backward pass through the model
 * with LoRA adapters, and updates LoRA weights with AdamW.
 */
public class TrainingLoop {

    private final LLMEngine engine;
    private final Tokenizer tokenizer;
    private final List<LoRAAdapter> adapters;
    private final int contextLength;

    // Hyperparameters
    private final float learningRate;
    private final float beta1 = 0.9f;
    private final float beta2 = 0.999f;
    private final float eps = 1e-8f;
    private final float weightDecay = 0.01f;

    private int globalStep = 0;

    public TrainingLoop(LLMEngine engine, Tokenizer tokenizer,
                        List<LoRAAdapter> adapters, int contextLength,
                        float learningRate) {
        this.engine = engine;
        this.tokenizer = tokenizer;
        this.adapters = adapters;
        this.contextLength = contextLength;
        this.learningRate = learningRate;
    }

    /**
     * Train for the specified number of epochs.
     * @param datasetFile  JSONL training file
     * @param epochs       total epochs
     * @param skipEpochs   epochs already completed (resume)
     * @param checkpointDir directory for epoch checkpoints
     * @param state        pipeline state (updated with progress)
     * @param listener     progress callback
     * @return final average loss
     */
    public float train(Path datasetFile, int epochs, int skipEpochs,
                       Path checkpointDir, PipelineState state,
                       ProgressListener listener) throws IOException {

        // Load training examples
        List<TrainingExample> examples = loadDataset(datasetFile);
        if (examples.isEmpty()) {
            System.out.println("  Warning: empty training dataset");
            return 0f;
        }

        // Restore optimizer state if resuming
        if (skipEpochs > 0) {
            Path epochDir = checkpointDir.resolve("epoch-" + skipEpochs);
            if (Files.isDirectory(epochDir)) {
                loadCheckpoint(epochDir);
                globalStep = skipEpochs * examples.size();
            }
        }

        float lastLoss = 0;
        for (int epoch = skipEpochs; epoch < epochs; epoch++) {
            // Shuffle examples each epoch
            Collections.shuffle(examples, new Random(42 + epoch));

            float epochLoss = 0;
            int count = 0;

            for (int i = 0; i < examples.size(); i++) {
                TrainingExample ex = examples.get(i);
                globalStep++;

                // Zero gradients
                for (LoRAAdapter a : adapters) a.zeroGrad();

                // Forward + loss + backward
                float loss = trainStep(ex);
                epochLoss += loss;
                count++;

                // AdamW update
                for (LoRAAdapter a : adapters) {
                    a.adamWStep(learningRate, beta1, beta2, eps, weightDecay, globalStep);
                }

                if (listener != null && (i + 1) % 10 == 0) {
                    listener.onStep(epoch, epochs, i + 1, examples.size(),
                        epochLoss / count);
                }
            }

            lastLoss = epochLoss / Math.max(1, count);

            if (listener != null) {
                listener.onEpochComplete(epoch, epochs, lastLoss);
            }

            // Checkpoint
            Path epochDir = checkpointDir.resolve("epoch-" + (epoch + 1));
            saveCheckpoint(epochDir);

            // Update pipeline state
            state.setEpochsCompleted(epoch + 1);
            state.save(checkpointDir.getParent());
        }

        return lastLoss;
    }

    /**
     * Single training step: forward pass, compute cross-entropy loss,
     * backward pass through LoRA adapters.
     *
     * For the initial implementation, we use the engine's forward pass
     * to compute logits for each position, then backpropagate the loss
     * gradient through the LoRA adapter deltas.
     */
    private float trainStep(TrainingExample example) {
        int[] tokens = example.tokens;
        if (tokens.length < 2) return 0f;

        // Truncate to context length
        int seqLen = Math.min(tokens.length, contextLength);
        float totalLoss = 0;

        // Simplified training: compute loss on each position using teacher forcing.
        // The engine runs forward pass for each position; we compute cross-entropy
        // between the output logits and the target token.
        //
        // For the LoRA gradient computation, we use the cached inputs from
        // each adapter's forward() call during inference, and backpropagate
        // the logit gradient through the adapter chain.
        //
        // NOTE: This initial implementation computes an approximate gradient
        // by treating each position independently. A full implementation would
        // propagate gradients through the entire sequence.

        // Process each position (teacher forcing)
        for (int pos = 0; pos < seqLen - 1; pos++) {
            int inputToken = tokens[pos];
            int targetToken = tokens[pos + 1];

            // Forward: get logits for this position
            // The engine's internal forward pass will use LoRA adapters if injected
            float[] logits = engine.forwardSingleToken(inputToken, pos);
            if (logits == null) continue;

            // Cross-entropy loss
            float loss = crossEntropyLoss(logits, targetToken);
            totalLoss += loss;

            // Gradient of cross-entropy w.r.t. logits: softmax(logits) - one_hot(target)
            float[] gradLogits = softmax(logits);
            gradLogits[targetToken] -= 1.0f;

            // Backpropagate through LoRA adapters (in reverse order)
            // Each adapter uses its cached input from the forward pass
            for (int a = adapters.size() - 1; a >= 0; a--) {
                adapters.get(a).backward(gradLogits);
            }
        }

        return totalLoss / Math.max(1, seqLen - 1);
    }

    private float crossEntropyLoss(float[] logits, int target) {
        // Numerically stable: log(softmax(logits)[target])
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) if (v > max) max = v;
        double sumExp = 0;
        for (float v : logits) sumExp += Math.exp(v - max);
        return -(float) ((logits[target] - max) - Math.log(sumExp));
    }

    private float[] softmax(float[] logits) {
        float[] result = new float[logits.length];
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) if (v > max) max = v;
        double sum = 0;
        for (int i = 0; i < logits.length; i++) {
            result[i] = (float) Math.exp(logits[i] - max);
            sum += result[i];
        }
        for (int i = 0; i < result.length; i++) result[i] /= (float) sum;
        return result;
    }

    // --- Dataset loading ---

    private List<TrainingExample> loadDataset(Path file) throws IOException {
        List<TrainingExample> examples = new ArrayList<>();
        List<String> lines = Files.readAllLines(file, StandardCharsets.UTF_8);
        for (String line : lines) {
            line = line.trim();
            if (line.isEmpty()) continue;
            Map<String, Object> obj = ApiHandler.parseJson(line);
            Object messagesObj = obj.get("messages");
            if (!(messagesObj instanceof List)) continue;

            @SuppressWarnings("unchecked")
            List<Object> messages = (List<Object>) messagesObj;
            StringBuilder fullText = new StringBuilder();
            for (Object msgObj : messages) {
                if (!(msgObj instanceof Map)) continue;
                @SuppressWarnings("unchecked")
                Map<String, Object> msg = (Map<String, Object>) msgObj;
                String content = (String) msg.get("content");
                if (content != null) {
                    if (fullText.length() > 0) fullText.append("\n");
                    fullText.append(content);
                }
            }

            int[] tokens = tokenizer.encode(fullText.toString());
            if (tokens.length >= 2) {
                examples.add(new TrainingExample(tokens));
            }
        }
        return examples;
    }

    // --- Checkpoint ---

    private void saveCheckpoint(Path dir) throws IOException {
        if (!Files.isDirectory(dir)) Files.createDirectories(dir);
        for (LoRAAdapter a : adapters) a.save(dir);
    }

    private void loadCheckpoint(Path dir) throws IOException {
        for (LoRAAdapter a : adapters) {
            Path aFile = dir.resolve(a.name() + ".A.bin");
            if (Files.exists(aFile)) {
                TrainableTensor loadedA = TrainableTensor.load(aFile);
                System.arraycopy(loadedA.data(), 0, a.loraA().data(), 0, loadedA.size());
                TrainableTensor loadedB = TrainableTensor.load(dir.resolve(a.name() + ".B.bin"));
                System.arraycopy(loadedB.data(), 0, a.loraB().data(), 0, loadedB.size());
                a.loadState(dir);
            }
        }
    }

    /** Trainable parameters count. */
    public int totalParams() {
        int total = 0;
        for (LoRAAdapter a : adapters) total += a.paramCount();
        return total;
    }

    public List<LoRAAdapter> adapters() { return adapters; }

    // --- Inner types ---

    private static class TrainingExample {
        final int[] tokens;
        TrainingExample(int[] tokens) { this.tokens = tokens; }
    }

    public interface ProgressListener {
        void onStep(int epoch, int totalEpochs, int step, int totalSteps, float avgLoss);
        void onEpochComplete(int epoch, int totalEpochs, float epochLoss);
    }
}
