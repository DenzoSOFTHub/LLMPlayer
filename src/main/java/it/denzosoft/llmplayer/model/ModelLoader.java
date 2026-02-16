package it.denzosoft.llmplayer.model;

import it.denzosoft.llmplayer.gguf.GGUFFile;
import it.denzosoft.llmplayer.gguf.GGUFParser;
import it.denzosoft.llmplayer.gguf.GGUFTensorInfo;
import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.TensorData;
import it.denzosoft.llmplayer.tensor.TensorFactory;
import it.denzosoft.llmplayer.tokenizer.Tokenizer;
import it.denzosoft.llmplayer.tokenizer.TokenizerFactory;

import java.io.IOException;
import java.nio.file.Path;

public class ModelLoader {

    public static final class LoadedModel implements AutoCloseable {
        private final GGUFFile ggufFile;
        private final ModelConfig config;
        private final ModelWeights weights;
        private final DeepSeek2Weights deepSeek2Weights;
        private final Qwen3MoEWeights qwen3MoEWeights;
        private final Tokenizer tokenizer;

        public LoadedModel(GGUFFile ggufFile, ModelConfig config, ModelWeights weights,
                           DeepSeek2Weights deepSeek2Weights, Qwen3MoEWeights qwen3MoEWeights,
                           Tokenizer tokenizer) {
            this.ggufFile = ggufFile;
            this.config = config;
            this.weights = weights;
            this.deepSeek2Weights = deepSeek2Weights;
            this.qwen3MoEWeights = qwen3MoEWeights;
            this.tokenizer = tokenizer;
        }

        public GGUFFile ggufFile() { return ggufFile; }
        public ModelConfig config() { return config; }
        public ModelWeights weights() { return weights; }
        public DeepSeek2Weights deepSeek2Weights() { return deepSeek2Weights; }
        public Qwen3MoEWeights qwen3MoEWeights() { return qwen3MoEWeights; }
        public Tokenizer tokenizer() { return tokenizer; }

        @Override
        public void close() {
            ggufFile.close();
        }
    }

    public static LoadedModel load(Path path) throws IOException {
        return load(path, false, -1);
    }

    public static LoadedModel load(Path path, boolean preload) throws IOException {
        return load(path, preload, -1, false);
    }

    /**
     * Load model with partial GPU offloading support.
     * @param gpuLayers number of layers to keep on GPU. -1 or negative = all layers on GPU (no limit),
     *                  0 = no GPU layers (CPU only for layers), positive N = first N layers on GPU.
     *                  Only relevant when a gpuBufferManager is registered in TensorFactory.
     */
    public static LoadedModel load(Path path, boolean preload, int gpuLayers) throws IOException {
        return load(path, preload, gpuLayers, false);
    }

    /**
     * Load model with partial GPU offloading and optional MoE-optimized placement.
     * When moeOptimizedGpu=true and the model is MoE, attention tensors go on GPU for ALL layers
     * while expert tensors stay on CPU. This maximizes GPU utilization for MoE models.
     */
    public static LoadedModel load(Path path, boolean preload, int gpuLayers, boolean moeOptimizedGpu) throws IOException {
        System.out.println("Loading model from: " + path);
        long startTime = System.currentTimeMillis();

        GGUFFile gguf = GGUFParser.parse(path, preload);
        System.out.println("  Parsed GGUF: " + gguf.getHeader().tensorCount() + " tensors, " +
            gguf.getMetadata().size() + " metadata entries");

        ModelConfig config = ModelConfig.fromMetadata(gguf.getMetadata());
        System.out.println("  Config: " + config);

        ModelWeights weights = null;
        DeepSeek2Weights ds2Weights = null;
        Qwen3MoEWeights q3moeWeights = null;

        ModelArchitecture arch = config.architecture();
        boolean isMoEArch = config.expertCount() > 0;
        if (arch == ModelArchitecture.DEEPSEEK2) {
            ds2Weights = loadDeepSeek2Weights(gguf, config, gpuLayers, moeOptimizedGpu);
        } else if (arch == ModelArchitecture.QWEN3MOE
                || (arch == ModelArchitecture.LLAMA4 && isMoEArch)
                || (arch == ModelArchitecture.GPT_OSS && isMoEArch)
                || (arch == ModelArchitecture.GLM4 && isMoEArch)) {
            // Qwen3MoE weight format works for any GQA + MoE architecture
            q3moeWeights = loadQwen3MoEWeights(gguf, config, gpuLayers, moeOptimizedGpu);
        } else {
            weights = loadWeights(gguf, config, gpuLayers);
        }
        System.out.println("  Weights loaded");

        Tokenizer tokenizer = TokenizerFactory.create(gguf.getMetadata());
        System.out.println("  Tokenizer loaded: vocab size = " + tokenizer.vocabSize());

        long elapsed = System.currentTimeMillis() - startTime;
        System.out.println("  Model loaded in " + elapsed + "ms");

        return new LoadedModel(gguf, config, weights, ds2Weights, q3moeWeights, tokenizer);
    }

    private static ModelWeights loadWeights(GGUFFile gguf, ModelConfig config, int gpuLayers) {
        FloatTensor tokenEmbedding = loadTensor(gguf, ArchitectureRegistry.TOKEN_EMBD);
        FloatTensor outputNorm = loadTensor(gguf, ArchitectureRegistry.OUTPUT_NORM);

        // Output weight may be tied to token embedding
        FloatTensor output;
        GGUFTensorInfo outputInfo = gguf.findTensor(ArchitectureRegistry.OUTPUT);
        if (outputInfo != null) {
            output = createTensor(gguf, outputInfo);
        } else {
            output = tokenEmbedding; // weight tying
        }

        // For partial GPU offloading: if gpuLayers >= 0, only load first N layers on GPU
        boolean partialOffload = gpuLayers >= 0 && TensorFactory.getGpuBufferManager() != null;
        Object savedGpuManager = partialOffload ? TensorFactory.getGpuBufferManager() : null;

        TransformerLayerWeights[] layers = new TransformerLayerWeights[config.blockCount()];
        for (int i = 0; i < config.blockCount(); i++) {
            // Disable GPU for layers beyond the limit
            if (partialOffload && i >= gpuLayers) {
                TensorFactory.setGpuBufferManager(null);
            }
            // Detect merged QKV (Phi3/Phi4) vs separate Q, K, V
            FloatTensor wq = tryLoadTensor(gguf, ArchitectureRegistry.attnQ(i));
            FloatTensor wk = tryLoadTensor(gguf, ArchitectureRegistry.attnK(i));
            FloatTensor wv = tryLoadTensor(gguf, ArchitectureRegistry.attnV(i));
            FloatTensor wqkv = null;
            if (wq == null) {
                // Try merged QKV
                wqkv = tryLoadTensor(gguf, ArchitectureRegistry.attnQKV(i));
                if (wqkv == null) {
                    throw new IllegalStateException("Layer " + i + ": neither attn_q nor attn_qkv found");
                }
            }

            layers[i] = new TransformerLayerWeights(
                tryLoadTensor(gguf, ArchitectureRegistry.attnNorm(i)),   // null for OLMo2 (post-norm only)
                tryLoadTensor(gguf, ArchitectureRegistry.ffnNorm(i)),    // null for OLMo2/Command-R
                wq, wk, wv,
                loadTensor(gguf, ArchitectureRegistry.attnOutput(i)),
                wqkv,
                tryLoadTensor(gguf, ArchitectureRegistry.ffnGate(i)),  // null for GLM4/Phi4 packed FFN
                loadTensor(gguf, ArchitectureRegistry.ffnUp(i)),
                loadTensor(gguf, ArchitectureRegistry.ffnDown(i)),
                // Qwen2 biases (null if absent)
                tryLoadTensor(gguf, ArchitectureRegistry.attnQBias(i)),
                tryLoadTensor(gguf, ArchitectureRegistry.attnKBias(i)),
                tryLoadTensor(gguf, ArchitectureRegistry.attnVBias(i)),
                // Qwen3/OLMo2 QK norm (null if absent)
                tryLoadTensor(gguf, ArchitectureRegistry.attnQNorm(i)),
                tryLoadTensor(gguf, ArchitectureRegistry.attnKNorm(i)),
                // GLM4/Gemma2/OLMo2 post-norm (null if absent)
                tryLoadTensor(gguf, ArchitectureRegistry.postAttnNorm(i)),
                tryLoadTensor(gguf, ArchitectureRegistry.postFfnNorm(i))
            );
        }

        // Restore GPU manager if it was temporarily disabled
        if (partialOffload) {
            TensorFactory.setGpuBufferManager(savedGpuManager);
        }

        // Load rope_freqs.weight if present (for extended context RoPE)
        float[] ropeFreqFactors = loadRopeFreqFactors(gguf, config);

        return new ModelWeights(tokenEmbedding, outputNorm, output, layers, ropeFreqFactors);
    }

    private static DeepSeek2Weights loadDeepSeek2Weights(GGUFFile gguf, ModelConfig config,
                                                          int gpuLayers, boolean moeOptimizedGpu) {
        FloatTensor tokenEmbedding = loadTensor(gguf, ArchitectureRegistry.TOKEN_EMBD);
        FloatTensor outputNorm = loadTensor(gguf, ArchitectureRegistry.OUTPUT_NORM);

        FloatTensor output;
        GGUFTensorInfo outputInfo = gguf.findTensor(ArchitectureRegistry.OUTPUT);
        if (outputInfo != null) {
            output = createTensor(gguf, outputInfo);
        } else {
            output = tokenEmbedding;
        }

        int blockCount = config.blockCount();
        int leadingDenseCount = config.leadingDenseBlockCount();

        boolean hasGpu = TensorFactory.getGpuBufferManager() != null;
        boolean partialOffload = gpuLayers >= 0 && hasGpu;
        boolean moeMode = moeOptimizedGpu && hasGpu;
        Object savedGpuManager = (partialOffload || moeMode) ? TensorFactory.getGpuBufferManager() : null;

        DeepSeek2LayerWeights[] layers = new DeepSeek2LayerWeights[blockCount];

        for (int i = 0; i < blockCount; i++) {
            boolean isDense = i < leadingDenseCount;

            if (moeMode && !isDense) {
                // MoE-optimized: per-tensor GPU placement
                // GPU ON for attention tensors
                TensorFactory.setGpuBufferManager(savedGpuManager);
                FloatTensor attnNorm = loadTensor(gguf, ArchitectureRegistry.attnNorm(i));
                FloatTensor ffnNorm = loadTensor(gguf, ArchitectureRegistry.ffnNorm(i));
                FloatTensor attnQ = loadTensor(gguf, ArchitectureRegistry.attnQ(i));
                FloatTensor attnKvAMqa = loadTensor(gguf, ArchitectureRegistry.attnKvAMqa(i));
                FloatTensor attnKvANorm = loadTensor(gguf, ArchitectureRegistry.attnKvANorm(i));
                FloatTensor attnKvB = loadTensor(gguf, ArchitectureRegistry.attnKvB(i));
                FloatTensor attnOutput = loadTensor(gguf, ArchitectureRegistry.attnOutput(i));

                // GPU OFF for expert tensors (large, only top-K used per token)
                TensorFactory.setGpuBufferManager(null);
                FloatTensor ffnGateExps = loadTensor(gguf, ArchitectureRegistry.ffnGateExps(i));
                FloatTensor ffnUpExps = loadTensor(gguf, ArchitectureRegistry.ffnUpExps(i));
                FloatTensor ffnDownExps = loadTensor(gguf, ArchitectureRegistry.ffnDownExps(i));

                // GPU ON for router + shared experts (small)
                TensorFactory.setGpuBufferManager(savedGpuManager);
                FloatTensor ffnGateInp = loadTensor(gguf, ArchitectureRegistry.ffnGateInp(i));
                FloatTensor ffnGateShexp = loadTensor(gguf, ArchitectureRegistry.ffnGateShexp(i));
                FloatTensor ffnUpShexp = loadTensor(gguf, ArchitectureRegistry.ffnUpShexp(i));
                FloatTensor ffnDownShexp = loadTensor(gguf, ArchitectureRegistry.ffnDownShexp(i));

                layers[i] = new DeepSeek2LayerWeights(
                    attnNorm, ffnNorm, attnQ, attnKvAMqa, attnKvANorm, attnKvB, attnOutput,
                    null, null, null, // no dense FFN
                    ffnGateInp, ffnGateExps, ffnUpExps, ffnDownExps,
                    ffnGateShexp, ffnUpShexp, ffnDownShexp
                );
            } else {
                // Standard first-N-layers offload or dense leading layers
                if (moeMode && isDense) {
                    // Dense leading layers: all on GPU in MoE mode
                    TensorFactory.setGpuBufferManager(savedGpuManager);
                } else if (partialOffload && i >= gpuLayers) {
                    TensorFactory.setGpuBufferManager(null);
                }

                layers[i] = new DeepSeek2LayerWeights(
                    loadTensor(gguf, ArchitectureRegistry.attnNorm(i)),
                    loadTensor(gguf, ArchitectureRegistry.ffnNorm(i)),
                    loadTensor(gguf, ArchitectureRegistry.attnQ(i)),
                    loadTensor(gguf, ArchitectureRegistry.attnKvAMqa(i)),
                    loadTensor(gguf, ArchitectureRegistry.attnKvANorm(i)),
                    loadTensor(gguf, ArchitectureRegistry.attnKvB(i)),
                    loadTensor(gguf, ArchitectureRegistry.attnOutput(i)),
                    isDense ? loadTensor(gguf, ArchitectureRegistry.ffnGate(i)) : null,
                    isDense ? loadTensor(gguf, ArchitectureRegistry.ffnUp(i)) : null,
                    isDense ? loadTensor(gguf, ArchitectureRegistry.ffnDown(i)) : null,
                    isDense ? null : loadTensor(gguf, ArchitectureRegistry.ffnGateInp(i)),
                    isDense ? null : loadTensor(gguf, ArchitectureRegistry.ffnGateExps(i)),
                    isDense ? null : loadTensor(gguf, ArchitectureRegistry.ffnUpExps(i)),
                    isDense ? null : loadTensor(gguf, ArchitectureRegistry.ffnDownExps(i)),
                    isDense ? null : loadTensor(gguf, ArchitectureRegistry.ffnGateShexp(i)),
                    isDense ? null : loadTensor(gguf, ArchitectureRegistry.ffnUpShexp(i)),
                    isDense ? null : loadTensor(gguf, ArchitectureRegistry.ffnDownShexp(i))
                );
            }
        }

        // Restore GPU manager
        if (savedGpuManager != null) {
            TensorFactory.setGpuBufferManager(savedGpuManager);
        }

        float[] ropeFreqFactors = loadRopeFreqFactors(gguf, config);

        return new DeepSeek2Weights(tokenEmbedding, outputNorm, output, layers, ropeFreqFactors);
    }

    private static Qwen3MoEWeights loadQwen3MoEWeights(GGUFFile gguf, ModelConfig config,
                                                        int gpuLayers, boolean moeOptimizedGpu) {
        FloatTensor tokenEmbedding = loadTensor(gguf, ArchitectureRegistry.TOKEN_EMBD);
        FloatTensor outputNorm = loadTensor(gguf, ArchitectureRegistry.OUTPUT_NORM);

        FloatTensor output;
        GGUFTensorInfo outputInfo = gguf.findTensor(ArchitectureRegistry.OUTPUT);
        if (outputInfo != null) {
            output = createTensor(gguf, outputInfo);
        } else {
            output = tokenEmbedding;
        }

        int blockCount = config.blockCount();
        int leadingDenseCount = config.leadingDenseBlockCount();

        boolean hasGpu = TensorFactory.getGpuBufferManager() != null;
        boolean partialOffload = gpuLayers >= 0 && hasGpu;
        boolean moeMode = moeOptimizedGpu && hasGpu;
        Object savedGpuManager = (partialOffload || moeMode) ? TensorFactory.getGpuBufferManager() : null;

        Qwen3MoELayerWeights[] layers = new Qwen3MoELayerWeights[blockCount];

        for (int i = 0; i < blockCount; i++) {
            boolean isDense = i < leadingDenseCount;

            if (moeMode && !isDense) {
                // MoE-optimized: per-tensor GPU placement
                // GPU ON for attention tensors
                TensorFactory.setGpuBufferManager(savedGpuManager);
                FloatTensor attnNorm = loadTensor(gguf, ArchitectureRegistry.attnNorm(i));
                FloatTensor ffnNorm = loadFfnNorm(gguf, i);
                FloatTensor wq = loadTensor(gguf, ArchitectureRegistry.attnQ(i));
                FloatTensor wk = loadTensor(gguf, ArchitectureRegistry.attnK(i));
                FloatTensor wv = loadTensor(gguf, ArchitectureRegistry.attnV(i));
                FloatTensor wo = loadTensor(gguf, ArchitectureRegistry.attnOutput(i));
                FloatTensor qNorm = tryLoadTensor(gguf, ArchitectureRegistry.attnQNorm(i));
                FloatTensor kNorm = tryLoadTensor(gguf, ArchitectureRegistry.attnKNorm(i));

                // Attention biases (GPT-OSS)
                FloatTensor wqBias = tryLoadTensor(gguf, ArchitectureRegistry.attnQBias(i));
                FloatTensor wkBias = tryLoadTensor(gguf, ArchitectureRegistry.attnKBias(i));
                FloatTensor wvBias = tryLoadTensor(gguf, ArchitectureRegistry.attnVBias(i));
                FloatTensor woBias = tryLoadTensor(gguf, ArchitectureRegistry.attnOutputBias(i));

                // GPU OFF for expert tensors (large, only top-K used per token)
                TensorFactory.setGpuBufferManager(null);
                FloatTensor ffnGateExps = loadTensor(gguf, ArchitectureRegistry.ffnGateExps(i));
                FloatTensor ffnUpExps = loadTensor(gguf, ArchitectureRegistry.ffnUpExps(i));
                FloatTensor ffnDownExps = loadTensor(gguf, ArchitectureRegistry.ffnDownExps(i));

                // Expert biases (GPT-OSS)
                FloatTensor ffnGateExpsBias = tryLoadTensor(gguf, ArchitectureRegistry.ffnGateExpsBias(i));
                FloatTensor ffnUpExpsBias = tryLoadTensor(gguf, ArchitectureRegistry.ffnUpExpsBias(i));
                FloatTensor ffnDownExpsBias = tryLoadTensor(gguf, ArchitectureRegistry.ffnDownExpsBias(i));

                // GPU ON for router + shared experts (small)
                TensorFactory.setGpuBufferManager(savedGpuManager);
                FloatTensor ffnGateInp = loadTensor(gguf, ArchitectureRegistry.ffnGateInp(i));
                FloatTensor ffnGateInpBias = tryLoadTensor(gguf, ArchitectureRegistry.ffnGateInpBias(i));
                FloatTensor ffnGateShexp = tryLoadTensor(gguf, ArchitectureRegistry.ffnGateShexp(i));
                FloatTensor ffnUpShexp = tryLoadTensor(gguf, ArchitectureRegistry.ffnUpShexp(i));
                FloatTensor ffnDownShexp = tryLoadTensor(gguf, ArchitectureRegistry.ffnDownShexp(i));

                // Attention sinks (GPT-OSS)
                FloatTensor attnSinks = tryLoadTensor(gguf, ArchitectureRegistry.attnSinks(i));

                layers[i] = new Qwen3MoELayerWeights(
                    attnNorm, ffnNorm, wq, wk, wv, wo, qNorm, kNorm,
                    null, null, null, // no dense FFN
                    ffnGateInp, ffnGateExps, ffnUpExps, ffnDownExps,
                    ffnGateShexp, ffnUpShexp, ffnDownShexp,
                    wqBias, wkBias, wvBias, woBias,
                    ffnGateInpBias, ffnGateExpsBias, ffnUpExpsBias, ffnDownExpsBias,
                    attnSinks
                );
            } else {
                // Standard first-N-layers offload or dense leading layers
                if (moeMode && isDense) {
                    // Dense leading layers: all on GPU in MoE mode
                    TensorFactory.setGpuBufferManager(savedGpuManager);
                } else if (partialOffload && i >= gpuLayers) {
                    TensorFactory.setGpuBufferManager(null);
                }

                FloatTensor attnNorm = loadTensor(gguf, ArchitectureRegistry.attnNorm(i));
                FloatTensor ffnNorm = loadFfnNorm(gguf, i);
                FloatTensor wq = loadTensor(gguf, ArchitectureRegistry.attnQ(i));
                FloatTensor wk = loadTensor(gguf, ArchitectureRegistry.attnK(i));
                FloatTensor wv = loadTensor(gguf, ArchitectureRegistry.attnV(i));
                FloatTensor wo = loadTensor(gguf, ArchitectureRegistry.attnOutput(i));

                // Attention biases (GPT-OSS)
                FloatTensor wqBias = tryLoadTensor(gguf, ArchitectureRegistry.attnQBias(i));
                FloatTensor wkBias = tryLoadTensor(gguf, ArchitectureRegistry.attnKBias(i));
                FloatTensor wvBias = tryLoadTensor(gguf, ArchitectureRegistry.attnVBias(i));
                FloatTensor woBias = tryLoadTensor(gguf, ArchitectureRegistry.attnOutputBias(i));

                // Router/expert biases (GPT-OSS)
                FloatTensor ffnGateInpBias = isDense ? null : tryLoadTensor(gguf, ArchitectureRegistry.ffnGateInpBias(i));
                FloatTensor ffnGateExpsBias = isDense ? null : tryLoadTensor(gguf, ArchitectureRegistry.ffnGateExpsBias(i));
                FloatTensor ffnUpExpsBias = isDense ? null : tryLoadTensor(gguf, ArchitectureRegistry.ffnUpExpsBias(i));
                FloatTensor ffnDownExpsBias = isDense ? null : tryLoadTensor(gguf, ArchitectureRegistry.ffnDownExpsBias(i));

                // Attention sinks (GPT-OSS)
                FloatTensor attnSinks = tryLoadTensor(gguf, ArchitectureRegistry.attnSinks(i));

                layers[i] = new Qwen3MoELayerWeights(
                    attnNorm, ffnNorm, wq, wk, wv, wo,
                    tryLoadTensor(gguf, ArchitectureRegistry.attnQNorm(i)),
                    tryLoadTensor(gguf, ArchitectureRegistry.attnKNorm(i)),
                    isDense ? loadTensor(gguf, ArchitectureRegistry.ffnGate(i)) : null,
                    isDense ? loadTensor(gguf, ArchitectureRegistry.ffnUp(i)) : null,
                    isDense ? loadTensor(gguf, ArchitectureRegistry.ffnDown(i)) : null,
                    isDense ? null : loadTensor(gguf, ArchitectureRegistry.ffnGateInp(i)),
                    isDense ? null : loadTensor(gguf, ArchitectureRegistry.ffnGateExps(i)),
                    isDense ? null : loadTensor(gguf, ArchitectureRegistry.ffnUpExps(i)),
                    isDense ? null : loadTensor(gguf, ArchitectureRegistry.ffnDownExps(i)),
                    isDense ? null : tryLoadTensor(gguf, ArchitectureRegistry.ffnGateShexp(i)),
                    isDense ? null : tryLoadTensor(gguf, ArchitectureRegistry.ffnUpShexp(i)),
                    isDense ? null : tryLoadTensor(gguf, ArchitectureRegistry.ffnDownShexp(i)),
                    wqBias, wkBias, wvBias, woBias,
                    ffnGateInpBias, ffnGateExpsBias, ffnUpExpsBias, ffnDownExpsBias,
                    attnSinks
                );
            }
        }

        // Restore GPU manager
        if (savedGpuManager != null) {
            TensorFactory.setGpuBufferManager(savedGpuManager);
        }

        float[] ropeFreqFactors = loadRopeFreqFactors(gguf, config);

        return new Qwen3MoEWeights(tokenEmbedding, outputNorm, output, layers, ropeFreqFactors);
    }

    /** Load ffn_norm with fallback to post_attention_norm (GPT-OSS naming). */
    private static FloatTensor loadFfnNorm(GGUFFile gguf, int layer) {
        FloatTensor t = tryLoadTensor(gguf, ArchitectureRegistry.ffnNorm(layer));
        if (t != null) return t;
        t = tryLoadTensor(gguf, ArchitectureRegistry.postAttnNorm(layer));
        if (t != null) return t;
        throw new IllegalStateException("Required tensor not found: " + ArchitectureRegistry.ffnNorm(layer)
            + " (also tried " + ArchitectureRegistry.postAttnNorm(layer) + ")");
    }

    private static float[] loadRopeFreqFactors(GGUFFile gguf, ModelConfig config) {
        GGUFTensorInfo ropeFreqsInfo = gguf.findTensor("rope_freqs.weight");
        if (ropeFreqsInfo == null) {
            ropeFreqsInfo = gguf.findTensor(ArchitectureRegistry.ropeFreqs(0));
        }
        if (ropeFreqsInfo != null) {
            FloatTensor ropeFreqsTensor = createTensor(gguf, ropeFreqsInfo);
            int halfDim = config.ropeDimensionCount() / 2;
            float[] factors = new float[halfDim];
            for (int i = 0; i < halfDim; i++) {
                factors[i] = ropeFreqsTensor.getFloat(i);
            }
            System.out.println("  Loaded rope_freqs.weight: " + halfDim + " frequency factors");
            return factors;
        }
        return null;
    }

    private static FloatTensor loadTensor(GGUFFile gguf, String name) {
        GGUFTensorInfo info = gguf.findTensor(name);
        if (info == null) {
            throw new IllegalStateException("Required tensor not found: " + name);
        }
        return createTensor(gguf, info);
    }

    private static FloatTensor tryLoadTensor(GGUFFile gguf, String name) {
        GGUFTensorInfo info = gguf.findTensor(name);
        if (info == null) {
            return null;
        }
        return createTensor(gguf, info);
    }

    private static FloatTensor createTensor(GGUFFile gguf, GGUFTensorInfo info) {
        TensorData data = gguf.getTensorData(info);
        return TensorFactory.create(info.type(), data, info.elementCount());
    }
}
