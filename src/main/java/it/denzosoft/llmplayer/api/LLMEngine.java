package it.denzosoft.llmplayer.api;

import it.denzosoft.llmplayer.evaluator.*;
import it.denzosoft.llmplayer.inference.DeepSeek2InferenceEngine;
import it.denzosoft.llmplayer.inference.DeepSeek2State;
import it.denzosoft.llmplayer.inference.InferenceEngine;
import it.denzosoft.llmplayer.inference.InferenceState;
import it.denzosoft.llmplayer.inference.Qwen3MoEInferenceEngine;
import it.denzosoft.llmplayer.inference.Qwen3MoEState;
import it.denzosoft.llmplayer.gguf.GGUFTensorInfo;
import it.denzosoft.llmplayer.model.ArchitectureRegistry;
import it.denzosoft.llmplayer.model.ModelArchitecture;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.model.ModelLoader;
import it.denzosoft.llmplayer.sampler.CompositeSampler;
import it.denzosoft.llmplayer.sampler.SamplerConfig;
import it.denzosoft.llmplayer.tokenizer.ChatTemplate;
import it.denzosoft.llmplayer.tokenizer.SpecialTokens;
import it.denzosoft.llmplayer.tokenizer.Tokenizer;

import it.denzosoft.llmplayer.gpu.GpuConfig;
import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.TensorFactory;

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;

/**
 * Main facade for the LLM inference engine.
 * Thread-safe: model weights are immutable (mmap), each generate() creates its own state.
 */
public class LLMEngine implements AutoCloseable {

    private final ModelLoader.LoadedModel loadedModel;
    private final InferenceEngine engine;                 // standard architectures
    private final DeepSeek2InferenceEngine ds2Engine;     // DeepSeek2 only
    private final Qwen3MoEInferenceEngine q3moeEngine;   // Qwen3 MoE only
    private final Tokenizer tokenizer;
    private final ChatTemplate chatTemplate;
    private final SpecialTokens specialTokens;
    private final int maxContextLength;
    private final long modelFileSize;
    private final long kvCacheEstimate;
    private final AutoCloseable gpuResources; // OpenCLContext + GpuBufferManager, closed on engine close
    private final int gpuLayersUsed;       // how many layers are on GPU (-1 = none)
    private final String gpuDeviceName;    // GPU device name, or null
    private final boolean moeOptimizedGpu; // MoE: attention on GPU, experts on CPU
    private final ConversationCache conversationCache = new ConversationCache();

    private LLMEngine(ModelLoader.LoadedModel loadedModel, int maxContextLength) {
        this(loadedModel, maxContextLength, null, -1, null, false);
    }

    private LLMEngine(ModelLoader.LoadedModel loadedModel, int maxContextLength,
                       AutoCloseable gpuResources, int gpuLayersUsed, String gpuDeviceName,
                       boolean moeOptimizedGpu) {
        this.loadedModel = loadedModel;
        this.tokenizer = loadedModel.tokenizer();
        this.specialTokens = SpecialTokens.fromMetadata(loadedModel.ggufFile().getMetadata());
        this.chatTemplate = new ChatTemplate(loadedModel.config().architecture(),
            loadedModel.ggufFile().getMetadata().getString("tokenizer.chat_template", ""));
        this.maxContextLength = maxContextLength;
        this.modelFileSize = loadedModel.ggufFile().getFileSize();
        this.kvCacheEstimate = estimateKvCache(loadedModel.config(), maxContextLength);
        this.gpuResources = gpuResources;
        this.gpuLayersUsed = gpuLayersUsed;
        this.gpuDeviceName = gpuDeviceName;
        this.moeOptimizedGpu = moeOptimizedGpu;

        ModelArchitecture arch = loadedModel.config().architecture();
        if (arch == ModelArchitecture.DEEPSEEK2) {
            this.engine = null;
            this.ds2Engine = new DeepSeek2InferenceEngine(
                loadedModel.config(), loadedModel.deepSeek2Weights(), maxContextLength,
                loadedModel.deepSeek2Weights().ropeFreqFactors());
            this.q3moeEngine = null;
        } else if (arch == ModelArchitecture.QWEN3MOE) {
            this.engine = null;
            this.ds2Engine = null;
            this.q3moeEngine = new Qwen3MoEInferenceEngine(
                loadedModel.config(), loadedModel.qwen3MoEWeights(), maxContextLength,
                loadedModel.qwen3MoEWeights().ropeFreqFactors());
        } else if (arch == ModelArchitecture.LLAMA4 && loadedModel.config().expertCount() > 0) {
            // Llama4 MoE: reuse Qwen3MoE engine (standard GQA + MoE FFN)
            this.engine = null;
            this.ds2Engine = null;
            this.q3moeEngine = new Qwen3MoEInferenceEngine(
                loadedModel.config(), loadedModel.qwen3MoEWeights(), maxContextLength,
                loadedModel.qwen3MoEWeights().ropeFreqFactors());
        } else if (arch == ModelArchitecture.GPT_OSS && loadedModel.config().expertCount() > 0) {
            // GPT-OSS MoE: reuse Qwen3MoE engine (GQA + MoE FFN)
            this.engine = null;
            this.ds2Engine = null;
            this.q3moeEngine = new Qwen3MoEInferenceEngine(
                loadedModel.config(), loadedModel.qwen3MoEWeights(), maxContextLength,
                loadedModel.qwen3MoEWeights().ropeFreqFactors());
        } else if (arch == ModelArchitecture.GLM4 && loadedModel.config().expertCount() > 0) {
            // GLM4 MoE: reuse Qwen3MoE engine (GQA + MoE FFN)
            this.engine = null;
            this.ds2Engine = null;
            this.q3moeEngine = new Qwen3MoEInferenceEngine(
                loadedModel.config(), loadedModel.qwen3MoEWeights(), maxContextLength,
                loadedModel.qwen3MoEWeights().ropeFreqFactors());
        } else {
            // Standard dense engine: Llama, Qwen2, Qwen3, GLM4, Phi3, Mistral3,
            // Command-R, OLMo2, Gemma2/3, dense Llama4/GPT-OSS
            this.ds2Engine = null;
            this.q3moeEngine = null;
            this.engine = new InferenceEngine(loadedModel.config(), loadedModel.weights(), maxContextLength,
                loadedModel.weights().ropeFreqFactors());
        }
    }

    public static LLMEngine load(Path ggufPath) throws IOException {
        return load(ggufPath, 2048);
    }

    public static LLMEngine load(Path ggufPath, int maxContextLength) throws IOException {
        // Auto-configure GPU: use best available hardware automatically
        GpuConfig autoConfig = autoConfigureGpu(ggufPath);
        if (autoConfig != null) {
            return load(ggufPath, maxContextLength, autoConfig);
        }
        ModelLoader.LoadedModel model = ModelLoader.load(ggufPath, true);
        int maxCtx = Math.min(maxContextLength, model.config().contextLength());
        return new LLMEngine(model, maxCtx);
    }

    public static LLMEngine load(Path ggufPath, int maxContextLength, GpuConfig gpuConfig) throws IOException {
        // Initialize GPU BEFORE model loading so TensorFactory can create GPU tensor variants.
        // Disable virtual thread matmul when GPU is active: OpenCL driver threads conflict
        // with JVM virtual thread carrier threads, causing segfaults in hybrid CPU/GPU mode.
        // This applies to both PoCL and NVIDIA drivers.
        AutoCloseable gpuRes = null;
        String deviceName = null;
        int gpuLayersUsed = -1;
        boolean moeOptimized = false;
        if (gpuConfig != null && gpuConfig.isEnabled()) {
            gpuRes = initGpu(gpuConfig);
            if (gpuRes != null) {
                FloatTensor.disableVirtualThreadMatmul();
                deviceName = getGpuDeviceName(gpuConfig.getDeviceId());

                // Calculate GPU layers
                int requestedLayers = gpuConfig.getGpuLayers();
                long modelFileSize = Files.size(ggufPath);

                // Quick parse to get block count and MoE config for auto-calculation
                it.denzosoft.llmplayer.gguf.GGUFFile quickParse = it.denzosoft.llmplayer.gguf.GGUFParser.parse(ggufPath);
                ModelConfig quickConfig = ModelConfig.fromMetadata(quickParse.getMetadata());
                int blockCount = quickConfig.blockCount();

                if (requestedLayers == -1) {
                    // Auto-detect: estimate bytes per layer, fit into VRAM
                    long vram = getDeviceGlobalMemory(gpuConfig.getDeviceId());
                    if (vram > 0 && quickConfig.expertCount() > 0) {
                        // MoE model: try MoE-optimized placement (attention on GPU, experts on CPU)
                        long nonExpertBytes = sumNonExpertTensorBytes(quickParse, quickConfig);
                        long usableVram = (long) (vram * 0.80);
                        if (nonExpertBytes <= usableVram) {
                            // All attention+norms+router+shared-expert fit in VRAM
                            moeOptimized = true;
                            gpuLayersUsed = blockCount;
                            long vramUsedMB = nonExpertBytes / (1024 * 1024);
                            System.out.println("GPU: MoE-optimized offload — all " + blockCount +
                                " layers attention on GPU, experts on CPU (" + vramUsedMB + " MB VRAM used, " +
                                (vram / 1024 / 1024) + " MB available)");
                        } else {
                            // Non-expert tensors don't fit → fallback to first-N-layers
                            long bytesPerLayer = modelFileSize / blockCount;
                            int fittableLayers = (int) (usableVram / bytesPerLayer);
                            gpuLayersUsed = Math.min(fittableLayers, blockCount);
                            long vramUsedMB = (long) gpuLayersUsed * bytesPerLayer / (1024 * 1024);
                            System.out.println("GPU: offloading " + gpuLayersUsed + "/" + blockCount +
                                " layers to GPU (" + vramUsedMB + " MB VRAM used, " +
                                (vram / 1024 / 1024) + " MB available)");
                        }
                    } else if (vram > 0) {
                        // Dense model: standard first-N-layers offload
                        long bytesPerLayer = modelFileSize / blockCount;
                        long usableVram = (long) (vram * 0.80);
                        int fittableLayers = (int) (usableVram / bytesPerLayer);
                        gpuLayersUsed = Math.min(fittableLayers, blockCount);
                        long vramUsedMB = (long) gpuLayersUsed * bytesPerLayer / (1024 * 1024);
                        System.out.println("GPU: offloading " + gpuLayersUsed + "/" + blockCount +
                            " layers to GPU (" + vramUsedMB + " MB VRAM used, " +
                            (vram / 1024 / 1024) + " MB available)");
                    } else {
                        // Can't detect VRAM, put all layers on GPU
                        gpuLayersUsed = blockCount;
                        System.out.println("GPU: offloading all " + blockCount + " layers (VRAM detection unavailable)");
                    }
                } else if (requestedLayers == 0) {
                    // 0 means all layers on GPU
                    gpuLayersUsed = blockCount;
                    System.out.println("GPU: offloading all " + blockCount + " layers to GPU");
                } else {
                    gpuLayersUsed = Math.min(requestedLayers, blockCount);
                    System.out.println("GPU: offloading " + gpuLayersUsed + "/" + blockCount + " layers to GPU");
                }

                // Apply MoE-optimized flag from explicit GpuConfig if set
                if (gpuConfig.isMoeOptimized()) {
                    moeOptimized = true;
                }

                quickParse.close();
            }
        }
        ModelLoader.LoadedModel model = ModelLoader.load(ggufPath, true, gpuLayersUsed, moeOptimized);
        int maxCtx = Math.min(maxContextLength, model.config().contextLength());
        return new LLMEngine(model, maxCtx, gpuRes, gpuLayersUsed, deviceName, moeOptimized);
    }

    public GenerationResponse generate(GenerationRequest request) {
        return generate(request, null);
    }

    public GenerationResponse generate(GenerationRequest request, StreamingCallback callback) {
        ModelConfig config = loadedModel.config();
        CompositeSampler sampler = new CompositeSampler(request.samplerConfig());

        // Tokenize prompt
        String formattedPrompt;
        if (request.rawMode()) {
            formattedPrompt = request.prompt();
        } else if (request.useChat()) {
            if (request.systemMessage() != null) {
                formattedPrompt = chatTemplate.formatChat(request.systemMessage(), request.prompt());
            } else {
                formattedPrompt = chatTemplate.formatUserMessage(request.prompt());
            }
        } else {
            formattedPrompt = request.prompt();
        }

        int[] encodedTokens = tokenizer.encode(formattedPrompt);

        // Prepend BOS token if chat mode or rawMode (BOS is not included in template text)
        int[] promptTokens;
        if (request.useChat() || request.rawMode()) {
            int bosId = specialTokens.getBosId();
            promptTokens = new int[encodedTokens.length + 1];
            promptTokens[0] = bosId;
            System.arraycopy(encodedTokens, 0, promptTokens, 1, encodedTokens.length);
        } else {
            promptTokens = encodedTokens;
        }
        int promptLen = promptTokens.length;

        if (promptLen >= maxContextLength - 1) {
            return new GenerationResponse("Error: prompt too long (" + promptLen + " tokens, max " +
                (maxContextLength - 1) + ")", 0, promptLen, 0, 0, Collections.<EvaluationResult>emptyList());
        }

        // KV cache reuse: check for cached conversation state
        String cacheKey = request.cacheKey();
        ConversationCache.CachedConversation cached = null;
        int prefillStart = 0;

        if (cacheKey != null) {
            cached = conversationCache.take(cacheKey); // exclusive access
            if (cached != null) {
                prefillStart = findPrefixMatch(cached.promptTokens, promptTokens);
                if (prefillStart == 0) {
                    cached = null; // no match, create fresh state
                } else {
                    // Ensure at least the last token is re-processed to get logits
                    if (prefillStart >= promptLen) {
                        prefillStart = promptLen - 1;
                    }
                }
            }
        }

        // Dispatch to appropriate engine
        GenerationResponse response;
        Object stateForCache;
        if (ds2Engine != null) {
            DeepSeek2State state = (cached != null)
                ? (DeepSeek2State) cached.state
                : ds2Engine.createState(maxContextLength);
            stateForCache = state;
            response = generateDeepSeek2(ds2Engine, sampler, promptTokens, request, callback, state, prefillStart);
        } else if (q3moeEngine != null) {
            Qwen3MoEState state = (cached != null)
                ? (Qwen3MoEState) cached.state
                : q3moeEngine.createState(maxContextLength);
            stateForCache = state;
            response = generateQwen3MoE(q3moeEngine, sampler, promptTokens, request, callback, state, prefillStart);
        } else {
            InferenceState state = (cached != null)
                ? (InferenceState) cached.state
                : engine.createState(maxContextLength);
            stateForCache = state;
            response = generateStandard(engine, sampler, promptTokens, request, callback, state, prefillStart);
        }

        // Update cache with the state (KV cache now includes prompt + generated tokens)
        if (cacheKey != null) {
            conversationCache.put(cacheKey, new ConversationCache.CachedConversation(stateForCache, promptTokens));
        }

        return response;
    }

    private GenerationResponse generateStandard(InferenceEngine eng, CompositeSampler sampler,
                                                  int[] promptTokens, GenerationRequest request,
                                                  StreamingCallback callback,
                                                  InferenceState state, int prefillStart) {
        int promptLen = promptTokens.length;

        long startTime = System.nanoTime();
        float[] logits = null;
        for (int i = prefillStart; i < promptLen; i++) {
            logits = eng.forward(state, promptTokens[i], i);
        }
        long genStartTime = System.nanoTime();

        return generateLoop(logits, promptLen, request, sampler, callback, startTime, genStartTime,
            new ForwardFunction() {
                @Override
                public float[] forward(int token, int position) {
                    return eng.forward(state, token, position);
                }
            });
    }

    private GenerationResponse generateDeepSeek2(DeepSeek2InferenceEngine eng, CompositeSampler sampler,
                                                   int[] promptTokens, GenerationRequest request,
                                                   StreamingCallback callback,
                                                   DeepSeek2State state, int prefillStart) {
        int promptLen = promptTokens.length;

        long startTime = System.nanoTime();
        float[] logits = null;
        for (int i = prefillStart; i < promptLen; i++) {
            logits = eng.forward(state, promptTokens[i], i);
        }
        long genStartTime = System.nanoTime();

        return generateLoop(logits, promptLen, request, sampler, callback, startTime, genStartTime,
            new ForwardFunction() {
                @Override
                public float[] forward(int token, int position) {
                    return eng.forward(state, token, position);
                }
            });
    }

    private GenerationResponse generateQwen3MoE(Qwen3MoEInferenceEngine eng, CompositeSampler sampler,
                                                  int[] promptTokens, GenerationRequest request,
                                                  StreamingCallback callback,
                                                  Qwen3MoEState state, int prefillStart) {
        int promptLen = promptTokens.length;

        long startTime = System.nanoTime();
        float[] logits = null;
        for (int i = prefillStart; i < promptLen; i++) {
            logits = eng.forward(state, promptTokens[i], i);
        }
        long genStartTime = System.nanoTime();

        return generateLoop(logits, promptLen, request, sampler, callback, startTime, genStartTime,
            new ForwardFunction() {
                @Override
                public float[] forward(int token, int position) {
                    return eng.forward(state, token, position);
                }
            });
    }

    private interface ForwardFunction {
        float[] forward(int token, int position);
    }

    private GenerationResponse generateLoop(float[] logits, int promptLen,
                                             GenerationRequest request, CompositeSampler sampler,
                                             StreamingCallback callback,
                                             long startTime, long genStartTime,
                                             ForwardFunction forwardFn) {
        List<Integer> generatedTokens = new ArrayList<>();
        List<float[]> logitsHistory = new ArrayList<>();
        StringBuilder responseText = new StringBuilder();
        boolean eosReached = false;

        for (int i = 0; i < request.maxTokens(); i++) {
            int nextToken = sampler.sample(logits);

            if (specialTokens.isEos(nextToken)) {
                eosReached = true;
                break;
            }

            generatedTokens.add(nextToken);
            logitsHistory.add(Arrays.copyOf(logits, logits.length));

            String tokenText = tokenizer.decode(nextToken);
            responseText.append(tokenText);

            if (callback != null) {
                if (!callback.onToken(tokenText, nextToken)) break;
            }

            int position = promptLen + i;
            if (position >= maxContextLength - 1) break;
            logits = forwardFn.forward(nextToken, position);
        }

        long totalTimeNs = System.nanoTime() - startTime;
        long genTimeNs = System.nanoTime() - genStartTime;
        long totalTimeMs = totalTimeNs / 1_000_000;
        int genTokenCount = generatedTokens.size();
        double tokPerSec = genTokenCount > 0 ? genTokenCount * 1_000_000_000.0 / genTimeNs : 0;

        int[] genTokenArray = new int[generatedTokens.size()];
        for (int i = 0; i < genTokenArray.length; i++) {
            genTokenArray[i] = generatedTokens.get(i);
        }
        EvaluationContext evalCtx = new EvaluationContext(
            request.prompt(), responseText.toString(), genTokenArray, logitsHistory, totalTimeMs, eosReached);
        AggregateEvaluator evaluator = AggregateEvaluator.createDefault(
            specialTokens.getEosId(), request.maxTokens());
        List<EvaluationResult> evalResults = evaluator.evaluateAll(evalCtx);

        return new GenerationResponse(
            responseText.toString(), genTokenCount, promptLen, tokPerSec, totalTimeMs, evalResults);
    }

    /**
     * Generate an embedding vector for the given text.
     * Runs prefill through the model, extracts the last hidden state after final RMSNorm,
     * and returns an L2-normalized vector.
     */
    public float[] embed(String text) {
        int[] encodedTokens = tokenizer.encode(text);
        int[] tokens = new int[encodedTokens.length + 1];
        tokens[0] = specialTokens.getBosId();
        System.arraycopy(encodedTokens, 0, tokens, 1, encodedTokens.length);

        int dim = loadedModel.config().embeddingLength();

        if (engine != null) {
            InferenceState state = engine.createState(maxContextLength);
            engine.prefill(state, tokens);
            return l2Normalize(state.xb, dim);
        } else if (ds2Engine != null) {
            DeepSeek2State state = ds2Engine.createState(maxContextLength);
            ds2Engine.prefill(state, tokens);
            return l2Normalize(state.xb, dim);
        } else if (q3moeEngine != null) {
            Qwen3MoEState state = q3moeEngine.createState(maxContextLength);
            q3moeEngine.prefill(state, tokens);
            return l2Normalize(state.xb, dim);
        }
        throw new IllegalStateException("No inference engine available");
    }

    /**
     * Find the length of the matching prefix between cached tokens and new tokens.
     */
    private static int findPrefixMatch(int[] cached, int[] newTokens) {
        int limit = Math.min(cached.length, newTokens.length);
        int match = 0;
        while (match < limit && cached[match] == newTokens[match]) {
            match++;
        }
        return match;
    }

    private static float[] l2Normalize(float[] src, int dim) {
        float[] result = new float[dim];
        float sumSq = 0;
        for (int i = 0; i < dim; i++) {
            sumSq += src[i] * src[i];
        }
        float norm = (float) Math.sqrt(sumSq);
        if (norm > 0) {
            for (int i = 0; i < dim; i++) {
                result[i] = src[i] / norm;
            }
        }
        return result;
    }

    public void generateStreaming(GenerationRequest request, StreamingCallback callback) {
        generate(request, callback);
    }

    public List<GenerationResponse> generateBatch(List<GenerationRequest> requests) {
        // Try Java 25 StructuredTaskScope first
        List<GenerationResponse> result = tryStructuredBatch(requests);
        if (result != null) return result;

        // Fallback: ExecutorService
        ExecutorService executor = Executors.newCachedThreadPool();
        try {
            List<Future<GenerationResponse>> futures = new ArrayList<>();
            for (GenerationRequest request : requests) {
                futures.add(executor.submit(new Callable<GenerationResponse>() {
                    @Override
                    public GenerationResponse call() {
                        return generate(request);
                    }
                }));
            }
            List<GenerationResponse> responses = new ArrayList<>();
            for (Future<GenerationResponse> future : futures) {
                try {
                    responses.add(future.get());
                } catch (Exception e) {
                    responses.add(new GenerationResponse(
                        "Error: " + e.getMessage(), 0, 0, 0, 0, Collections.<EvaluationResult>emptyList()));
                }
            }
            return responses;
        } finally {
            executor.shutdown();
        }
    }

    /**
     * Try to use Java 25 StructuredTaskScope-based batch generation via reflection.
     * Returns null if the class is not available (running on Java < 25).
     */
    @SuppressWarnings("unchecked")
    private List<GenerationResponse> tryStructuredBatch(List<GenerationRequest> requests) {
        try {
            Class<?> cls = Class.forName("it.denzosoft.llmplayer.api.StructuredBatchGenerator");
            java.lang.reflect.Method m = cls.getMethod("generate", LLMEngine.class, List.class);
            return (List<GenerationResponse>) m.invoke(null, this, requests);
        } catch (ClassNotFoundException e) {
            return null; // Java 25 classes not available
        } catch (Exception e) {
            // StructuredTaskScope failed — fall through to legacy path
            return null;
        }
    }

    public ModelInfo getModelInfo() {
        return ModelInfo.from(loadedModel.config(), modelFileSize, kvCacheEstimate);
    }

    private static long estimateKvCache(ModelConfig config, int maxCtxLen) {
        long bytesPerFloat = 4L;
        if (config.architecture() == ModelArchitecture.DEEPSEEK2) {
            // DeepSeek2 MLA: key=[headCount * keyLength], value=[headCount * valueLength]
            long keySize = (long) config.headCount() * config.keyLength();
            long valSize = (long) config.headCount() * config.valueLength();
            return config.blockCount() * (long) maxCtxLen * (keySize + valSize) * bytesPerFloat;
        } else {
            // Standard + Qwen3MoE + Phi3 + Mistral3: 2 (K+V) * blockCount * maxCtxLen * kvDim * 4
            return 2L * config.blockCount() * maxCtxLen * config.kvDim() * bytesPerFloat;
        }
    }

    public Tokenizer getTokenizer() { return tokenizer; }
    public SpecialTokens getSpecialTokens() { return specialTokens; }
    public ChatTemplate getChatTemplate() { return chatTemplate; }
    public ModelConfig getConfig() { return loadedModel.config(); }
    public String getModelName() { return loadedModel.config().name(); }
    public int getGpuLayersUsed() { return gpuLayersUsed; }
    public String getGpuDeviceName() { return gpuDeviceName; }
    public boolean isMoeOptimizedGpu() { return moeOptimizedGpu; }

    /**
     * Enumerate available GPU (OpenCL) devices via reflection.
     * Returns a list of maps with device info, or empty list if GPU support unavailable.
     */
    @SuppressWarnings("unchecked")
    public static List<Map<String, Object>> listGpuDevices() {
        try {
            Class<?> ctxClass = Class.forName("it.denzosoft.llmplayer.gpu.OpenCLContext");
            java.lang.reflect.Method enumMethod = ctxClass.getMethod("enumerateDevices");
            List<?> devices = (List<?>) enumMethod.invoke(null);
            List<Map<String, Object>> result = new ArrayList<>();
            for (int i = 0; i < devices.size(); i++) {
                Object dev = devices.get(i);
                Map<String, Object> info = new LinkedHashMap<>();
                info.put("index", i);
                // Try to extract fields via reflection on the device info object
                try {
                    Class<?> devClass = dev.getClass();
                    try {
                        info.put("name", devClass.getMethod("name").invoke(dev));
                    } catch (Exception e1) {
                        try {
                            info.put("name", devClass.getMethod("getName").invoke(dev));
                        } catch (Exception e2) {
                            info.put("name", dev.toString());
                        }
                    }
                    try {
                        info.put("vendor", devClass.getMethod("vendor").invoke(dev));
                    } catch (Exception ignored) {}
                    try {
                        info.put("globalMemory", devClass.getMethod("globalMemory").invoke(dev));
                    } catch (Exception ignored) {}
                    try {
                        info.put("computeUnits", devClass.getMethod("computeUnits").invoke(dev));
                    } catch (Exception ignored) {}
                    try {
                        info.put("deviceType", devClass.getMethod("deviceType").invoke(dev));
                    } catch (Exception ignored) {}
                } catch (Exception e) {
                    info.put("name", dev.toString());
                }
                result.add(info);
            }
            return result;
        } catch (ClassNotFoundException e) {
            return Collections.emptyList();
        } catch (Exception e) {
            System.err.println("Error enumerating GPU devices: " + e.getMessage());
            return Collections.emptyList();
        }
    }

    /**
     * Memory safety check before loading a model.
     */
    public static MemoryCheck checkMemory(Path ggufPath, int contextLength) {
        try {
            long fileSize = Files.size(ggufPath);

            // Quick parse to get config for KV cache estimate
            int blockCount = 32; // default estimate
            int kvDim = 256;
            try {
                it.denzosoft.llmplayer.gguf.GGUFFile quickParse = it.denzosoft.llmplayer.gguf.GGUFParser.parse(ggufPath);
                ModelConfig config = ModelConfig.fromMetadata(quickParse.getMetadata());
                blockCount = config.blockCount();
                kvDim = config.kvDim();
                quickParse.close();
            } catch (Exception ignored) {}

            long kvCache = 2L * blockCount * contextLength * kvDim * 4L;
            long estimatedRam = fileSize + kvCache;

            // Get available memory
            long jvmMaxMemory = Runtime.getRuntime().maxMemory();
            long physicalMemory = getPhysicalMemorySize();
            long availableRam = physicalMemory > 0 ? Math.min(jvmMaxMemory, physicalMemory) : jvmMaxMemory;

            boolean safe = estimatedRam < (long) (availableRam * 0.90); // 10% margin
            String message;
            if (safe) {
                message = "Memory OK: ~" + formatMB(estimatedRam) + " needed, " + formatMB(availableRam) + " available";
            } else {
                message = "WARNING: Model needs ~" + formatMB(estimatedRam) +
                    " but only " + formatMB(availableRam) + " available. Loading may cause swap/OOM.";
            }

            return new MemoryCheck(estimatedRam, availableRam, safe, message);
        } catch (IOException e) {
            return new MemoryCheck(0, 0, false, "Cannot check memory: " + e.getMessage());
        }
    }

    /**
     * Result of a memory safety check.
     */
    public static final class MemoryCheck {
        private final long estimatedRam;
        private final long availableRam;
        private final boolean safe;
        private final String message;

        public MemoryCheck(long estimatedRam, long availableRam, boolean safe, String message) {
            this.estimatedRam = estimatedRam;
            this.availableRam = availableRam;
            this.safe = safe;
            this.message = message;
        }

        public long estimatedRam() { return estimatedRam; }
        public long availableRam() { return availableRam; }
        public boolean isSafe() { return safe; }
        public String message() { return message; }
    }

    private static long getPhysicalMemorySize() {
        try {
            Object osBean = ManagementFactory.getOperatingSystemMXBean();
            Class<?> sunClass = Class.forName("com.sun.management.OperatingSystemMXBean");
            if (sunClass.isInstance(osBean)) {
                java.lang.reflect.Method m = sunClass.getMethod("getTotalPhysicalMemorySize");
                return ((Number) m.invoke(osBean)).longValue();
            }
        } catch (Exception ignored) {}
        return -1;
    }

    private static String getGpuDeviceName(int deviceId) {
        List<Map<String, Object>> devices = listGpuDevices();
        if (deviceId < devices.size()) {
            Object name = devices.get(deviceId).get("name");
            return name != null ? name.toString() : null;
        }
        return null;
    }

    private static long getDeviceGlobalMemory(int deviceId) {
        List<Map<String, Object>> devices = listGpuDevices();
        if (deviceId < devices.size()) {
            Object mem = devices.get(deviceId).get("globalMemory");
            if (mem instanceof Number) {
                return ((Number) mem).longValue();
            }
        }
        return -1;
    }

    /**
     * Check if the OpenCL device is CPU-based (e.g., PoCL).
     * CPU-based OpenCL devices conflict with JVM virtual threads.
     * Real GPUs (NVIDIA, AMD) can safely coexist with virtual thread matmul on CPU tensors.
     */
    private static boolean isDeviceCpuBased(int deviceId) {
        List<Map<String, Object>> devices = listGpuDevices();
        if (deviceId < devices.size()) {
            Object deviceType = devices.get(deviceId).get("deviceType");
            if (deviceType != null) {
                // OpenCL CL_DEVICE_TYPE_CPU = 2, CL_DEVICE_TYPE_GPU = 4
                String typeStr = deviceType.toString().toUpperCase();
                return typeStr.contains("CPU") || "2".equals(typeStr);
            }
            // Fallback: check device name for PoCL/CPU indicators
            Object name = devices.get(deviceId).get("name");
            if (name != null) {
                String nameStr = name.toString().toLowerCase();
                return nameStr.contains("cpu") || nameStr.contains("pocl");
            }
        }
        return false; // assume real GPU if unknown
    }

    private static String formatMB(long bytes) {
        long mb = bytes / (1024 * 1024);
        if (mb < 1024) return mb + " MB";
        return String.format("%.1f GB", mb / 1024.0);
    }

    /**
     * Auto-detect the best GPU and configure it optimally for the given model.
     * Returns null if no GPU is available.
     */
    public static GpuConfig autoConfigureGpu(Path ggufPath) {
        List<Map<String, Object>> devices = listGpuDevices();
        if (devices.isEmpty()) {
            return null;
        }

        // Select device with most VRAM
        int bestIdx = 0;
        long bestVram = 0;
        for (int i = 0; i < devices.size(); i++) {
            Object mem = devices.get(i).get("globalMemory");
            if (mem instanceof Number) {
                long vram = ((Number) mem).longValue();
                if (vram > bestVram) {
                    bestVram = vram;
                    bestIdx = i;
                }
            }
        }

        GpuConfig config = new GpuConfig();
        config.setEnabled(true);
        config.setDeviceId(bestIdx);
        config.setGpuLayers(-1); // auto-calculate based on VRAM
        return config;
    }

    /**
     * Build a hardware optimization plan describing how the system would configure
     * itself for optimal performance with the given model and hardware.
     * The UI can show this to the user for confirmation before loading.
     */
    public static HardwarePlan buildHardwarePlan(Path ggufPath, int contextLength) {
        List<Map<String, Object>> devices = listGpuDevices();
        MemoryCheck memCheck = checkMemory(ggufPath, contextLength);

        long modelFileSize;
        int blockCount;
        String modelName;
        boolean isMoE = false;
        boolean moeOptimized = false;
        long nonExpertBytes = 0;
        it.denzosoft.llmplayer.gguf.GGUFFile quickParse = null;
        ModelConfig planConfig = null;
        try {
            modelFileSize = Files.size(ggufPath);
            quickParse = it.denzosoft.llmplayer.gguf.GGUFParser.parse(ggufPath);
            planConfig = ModelConfig.fromMetadata(quickParse.getMetadata());
            blockCount = planConfig.blockCount();
            modelName = planConfig.name();
            isMoE = planConfig.expertCount() > 0;
            if (isMoE) {
                nonExpertBytes = sumNonExpertTensorBytes(quickParse, planConfig);
            }
            quickParse.close();
        } catch (Exception e) {
            if (quickParse != null) { try { quickParse.close(); } catch (Exception ignored) {} }
            return new HardwarePlan(ggufPath.getFileName().toString(), false, -1, null, 0,
                0, 0, false, memCheck, "Cannot read model: " + e.getMessage(), false);
        }

        boolean gpuAvailable = !devices.isEmpty();
        int bestDeviceIdx = -1;
        String bestDeviceName = null;
        long bestVram = 0;
        int gpuLayers = 0;

        if (gpuAvailable) {
            // Select device with most VRAM
            for (int i = 0; i < devices.size(); i++) {
                Object mem = devices.get(i).get("globalMemory");
                if (mem instanceof Number) {
                    long vram = ((Number) mem).longValue();
                    if (vram > bestVram) {
                        bestVram = vram;
                        bestDeviceIdx = i;
                    }
                }
            }
            if (bestDeviceIdx >= 0) {
                Object name = devices.get(bestDeviceIdx).get("name");
                bestDeviceName = name != null ? name.toString() : "GPU " + bestDeviceIdx;
            }

            // Calculate layers that fit in VRAM
            if (bestVram > 0) {
                long usableVram = (long) (bestVram * 0.80);
                if (isMoE && nonExpertBytes <= usableVram) {
                    // MoE-optimized: all attention on GPU, experts on CPU
                    moeOptimized = true;
                    gpuLayers = blockCount;
                } else {
                    // Standard first-N-layers
                    long bytesPerLayer = modelFileSize / blockCount;
                    gpuLayers = Math.min((int) (usableVram / bytesPerLayer), blockCount);
                }
            } else {
                gpuLayers = blockCount; // VRAM unknown, try all
            }
        }

        // Build summary
        boolean recommended = memCheck.isSafe();
        StringBuilder summary = new StringBuilder();
        summary.append("Model: ").append(modelName != null ? modelName : ggufPath.getFileName())
               .append(" (").append(formatMB(modelFileSize)).append(")\n");
        summary.append("Layers: ").append(blockCount).append("\n");
        summary.append("RAM: ").append(memCheck.message()).append("\n");
        if (gpuAvailable && bestDeviceName != null) {
            summary.append("GPU: ").append(bestDeviceName)
                   .append(" (").append(formatMB(bestVram)).append(" VRAM)\n");
            if (moeOptimized) {
                summary.append("Plan: MoE-optimized — ALL ").append(blockCount)
                       .append(" layers attention on GPU, experts on CPU (")
                       .append(formatMB(nonExpertBytes)).append(" VRAM used)");
            } else if (gpuLayers >= blockCount) {
                summary.append("Plan: ALL ").append(blockCount).append(" layers on GPU (full offload)");
            } else if (gpuLayers > 0) {
                summary.append("Plan: ").append(gpuLayers).append("/").append(blockCount)
                       .append(" layers on GPU, rest on CPU (partial offload)");
            } else {
                summary.append("Plan: CPU only (model too large for VRAM)");
            }
        } else {
            summary.append("GPU: not available\n");
            summary.append("Plan: CPU only");
        }

        if (!memCheck.isSafe()) {
            summary.append("\n\nWARNING: This model is too large for the available RAM.\n");
            summary.append("Loading will cause disk swap, resulting in extremely slow performance.\n");
            summary.append("Consider using a smaller/more quantized model.");
            recommended = false;
        }

        return new HardwarePlan(modelName, gpuAvailable, bestDeviceIdx, bestDeviceName,
            bestVram, gpuLayers, blockCount, moeOptimized, memCheck, summary.toString(), recommended);
    }

    /**
     * Describes the auto-tuned hardware configuration plan.
     */
    public static final class HardwarePlan {
        private final String modelName;
        private final boolean gpuAvailable;
        private final int gpuDeviceId;
        private final String gpuDeviceName;
        private final long gpuVram;
        private final int gpuLayers;
        private final int totalLayers;
        private final boolean moeOptimized;
        private final MemoryCheck memoryCheck;
        private final String summary;
        private final boolean recommended;

        public HardwarePlan(String modelName, boolean gpuAvailable, int gpuDeviceId,
                            String gpuDeviceName, long gpuVram, int gpuLayers, int totalLayers,
                            boolean moeOptimized,
                            MemoryCheck memoryCheck, String summary, boolean recommended) {
            this.modelName = modelName;
            this.gpuAvailable = gpuAvailable;
            this.gpuDeviceId = gpuDeviceId;
            this.gpuDeviceName = gpuDeviceName;
            this.gpuVram = gpuVram;
            this.gpuLayers = gpuLayers;
            this.totalLayers = totalLayers;
            this.moeOptimized = moeOptimized;
            this.memoryCheck = memoryCheck;
            this.summary = summary;
            this.recommended = recommended;
        }

        public String modelName() { return modelName; }
        public boolean isGpuAvailable() { return gpuAvailable; }
        public int gpuDeviceId() { return gpuDeviceId; }
        public String gpuDeviceName() { return gpuDeviceName; }
        public long gpuVram() { return gpuVram; }
        public int gpuLayers() { return gpuLayers; }
        public int totalLayers() { return totalLayers; }
        public boolean isMoeOptimized() { return moeOptimized; }
        public MemoryCheck memoryCheck() { return memoryCheck; }
        public String summary() { return summary; }
        /** Whether loading this model is recommended given the available hardware. */
        public boolean isRecommended() { return recommended; }

        /** Build a GpuConfig from this plan (null if GPU not available or 0 layers). */
        public GpuConfig toGpuConfig() {
            if (!gpuAvailable || gpuLayers <= 0) return null;
            GpuConfig config = new GpuConfig();
            config.setEnabled(true);
            config.setDeviceId(gpuDeviceId);
            config.setGpuLayers(gpuLayers);
            config.setMoeOptimized(moeOptimized);
            return config;
        }
    }

    /**
     * Sum the byte sizes of all non-expert tensors across all layers of a MoE model.
     * This includes: attention tensors, norms, router, shared experts, plus dense leading layers in full.
     */
    private static long sumNonExpertTensorBytes(it.denzosoft.llmplayer.gguf.GGUFFile gguf, ModelConfig config) {
        long total = 0;
        int blockCount = config.blockCount();
        int leadingDenseCount = config.leadingDenseBlockCount();
        boolean isDeepSeek2 = config.architecture() == ModelArchitecture.DEEPSEEK2;

        for (int i = 0; i < blockCount; i++) {
            boolean isDense = i < leadingDenseCount;
            if (isDense) {
                // Dense leading layers: sum all tensors
                total += sumAllLayerTensorBytes(gguf, config, i);
            } else {
                // MoE layers: sum only non-expert tensors
                total += sumMoELayerNonExpertBytes(gguf, config, i, isDeepSeek2);
            }
        }
        return total;
    }

    /**
     * Sum byte sizes of all tensors in a dense layer.
     */
    private static long sumAllLayerTensorBytes(it.denzosoft.llmplayer.gguf.GGUFFile gguf, ModelConfig config, int layer) {
        long total = 0;
        boolean isDeepSeek2 = config.architecture() == ModelArchitecture.DEEPSEEK2;

        // Norms
        total += tensorByteSize(gguf, ArchitectureRegistry.attnNorm(layer));
        total += tensorByteSize(gguf, ArchitectureRegistry.ffnNorm(layer));

        // Attention
        if (isDeepSeek2) {
            total += tensorByteSize(gguf, ArchitectureRegistry.attnQ(layer));
            total += tensorByteSize(gguf, ArchitectureRegistry.attnKvAMqa(layer));
            total += tensorByteSize(gguf, ArchitectureRegistry.attnKvANorm(layer));
            total += tensorByteSize(gguf, ArchitectureRegistry.attnKvB(layer));
            total += tensorByteSize(gguf, ArchitectureRegistry.attnOutput(layer));
        } else {
            total += tensorByteSize(gguf, ArchitectureRegistry.attnQ(layer));
            total += tensorByteSize(gguf, ArchitectureRegistry.attnK(layer));
            total += tensorByteSize(gguf, ArchitectureRegistry.attnV(layer));
            total += tensorByteSize(gguf, ArchitectureRegistry.attnOutput(layer));
            total += tensorByteSize(gguf, ArchitectureRegistry.attnQNorm(layer));
            total += tensorByteSize(gguf, ArchitectureRegistry.attnKNorm(layer));
        }

        // Dense FFN
        total += tensorByteSize(gguf, ArchitectureRegistry.ffnGate(layer));
        total += tensorByteSize(gguf, ArchitectureRegistry.ffnUp(layer));
        total += tensorByteSize(gguf, ArchitectureRegistry.ffnDown(layer));

        return total;
    }

    /**
     * Sum byte sizes of non-expert tensors in a MoE layer (attention + norms + router + shared experts).
     */
    private static long sumMoELayerNonExpertBytes(it.denzosoft.llmplayer.gguf.GGUFFile gguf, ModelConfig config,
                                                   int layer, boolean isDeepSeek2) {
        long total = 0;

        // Norms
        total += tensorByteSize(gguf, ArchitectureRegistry.attnNorm(layer));
        total += tensorByteSize(gguf, ArchitectureRegistry.ffnNorm(layer));

        // Attention tensors
        if (isDeepSeek2) {
            total += tensorByteSize(gguf, ArchitectureRegistry.attnQ(layer));
            total += tensorByteSize(gguf, ArchitectureRegistry.attnKvAMqa(layer));
            total += tensorByteSize(gguf, ArchitectureRegistry.attnKvANorm(layer));
            total += tensorByteSize(gguf, ArchitectureRegistry.attnKvB(layer));
            total += tensorByteSize(gguf, ArchitectureRegistry.attnOutput(layer));
        } else {
            total += tensorByteSize(gguf, ArchitectureRegistry.attnQ(layer));
            total += tensorByteSize(gguf, ArchitectureRegistry.attnK(layer));
            total += tensorByteSize(gguf, ArchitectureRegistry.attnV(layer));
            total += tensorByteSize(gguf, ArchitectureRegistry.attnOutput(layer));
            total += tensorByteSize(gguf, ArchitectureRegistry.attnQNorm(layer));
            total += tensorByteSize(gguf, ArchitectureRegistry.attnKNorm(layer));
        }

        // Router (small)
        total += tensorByteSize(gguf, ArchitectureRegistry.ffnGateInp(layer));

        // Shared experts (small)
        total += tensorByteSize(gguf, ArchitectureRegistry.ffnGateShexp(layer));
        total += tensorByteSize(gguf, ArchitectureRegistry.ffnUpShexp(layer));
        total += tensorByteSize(gguf, ArchitectureRegistry.ffnDownShexp(layer));

        return total;
    }

    /**
     * Null-safe tensor byte size lookup. Returns 0 if tensor not found.
     */
    private static long tensorByteSize(it.denzosoft.llmplayer.gguf.GGUFFile gguf, String name) {
        GGUFTensorInfo info = gguf.findTensor(name);
        return info != null ? info.byteSize() : 0;
    }

    /**
     * Initialize GPU via reflection (Java 21 only).
     * Returns an AutoCloseable that manages OpenCLContext + GpuBufferManager lifecycle,
     * or null if GPU initialization fails.
     */
    private static AutoCloseable initGpu(GpuConfig gpuConfig) {
        try {
            // Initialize OpenCLContext
            Class<?> ctxClass = Class.forName("it.denzosoft.llmplayer.gpu.OpenCLContext");
            java.lang.reflect.Method createMethod = ctxClass.getMethod("create", int.class);
            Object clContext = createMethod.invoke(null, gpuConfig.getDeviceId());

            // Pre-compile all GPU kernels eagerly (avoids first-use latency)
            try {
                ctxClass.getMethod("precompileKernels").invoke(clContext);
            } catch (Exception ignored) {}

            // Get device info for logging
            java.lang.reflect.Method getInfoMethod = ctxClass.getMethod("getDeviceInfo");
            Object deviceInfo = getInfoMethod.invoke(clContext);
            System.out.println("GPU: " + deviceInfo);

            // Create GpuBufferManager
            Class<?> bufMgrClass = Class.forName("it.denzosoft.llmplayer.gpu.GpuBufferManager");
            Object bufManager = bufMgrClass.getConstructor(ctxClass).newInstance(clContext);

            // Register with TensorFactory
            Class<?> tfClass = Class.forName("it.denzosoft.llmplayer.tensor.TensorFactory");
            java.lang.reflect.Method setGpuMethod = tfClass.getMethod("setGpuBufferManager", Object.class);
            setGpuMethod.invoke(null, bufManager);

            // Return a composite AutoCloseable
            return new AutoCloseable() {
                @Override
                public void close() throws Exception {
                    // Unregister from TensorFactory
                    try {
                        setGpuMethod.invoke(null, (Object) null);
                    } catch (Exception ignored) {}
                    // Close buffer manager
                    ((AutoCloseable) bufManager).close();
                    // Close OpenCL context
                    ((AutoCloseable) clContext).close();
                }
            };
        } catch (ClassNotFoundException e) {
            System.err.println("GPU support requires Java 21+. Falling back to CPU.");
            return null;
        } catch (Exception e) {
            String msg = e.getMessage();
            if (e.getCause() != null) msg = e.getCause().getMessage();
            System.err.println("GPU initialization failed: " + msg + ". Falling back to CPU.");
            return null;
        }
    }

    @Override
    public void close() {
        if (gpuResources != null) {
            try { gpuResources.close(); } catch (Exception ignored) {}
        }
        loadedModel.close();
    }
}
