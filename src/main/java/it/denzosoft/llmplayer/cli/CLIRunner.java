package it.denzosoft.llmplayer.cli;

import it.denzosoft.llmplayer.api.*;
import it.denzosoft.llmplayer.evaluator.EvaluationResult;
import it.denzosoft.llmplayer.sampler.SamplerConfig;

import it.denzosoft.llmplayer.gpu.GpuConfig;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class CLIRunner {

    private final CLIOptions options;

    public CLIRunner(CLIOptions options) {
        this.options = options;
    }

    public void run() throws IOException {
        if (options.isHelp()) {
            CLIOptions.printUsage();
            return;
        }

        if (options.isGpuList()) {
            listGpuDevices();
            return;
        }

        if (options.getModelPath() == null) {
            System.err.println("Error: --model is required");
            CLIOptions.printUsage();
            return;
        }

        // Configure thread pool. Default is physical cores (CLIOptions.detectPhysicalCores) because
        // the matmul hot path is memory-bandwidth bound and hyperthreads only add port contention.
        if (options.getThreads() > 0) {
            System.setProperty("java.util.concurrent.ForkJoinPool.common.parallelism",
                String.valueOf(options.getThreads()));
            int logical = Runtime.getRuntime().availableProcessors();
            if (options.getThreads() < logical) {
                System.out.println("CPU: using " + options.getThreads() + " threads (physical cores; "
                    + logical + " logical available — bandwidth-bound matmul prefers physical; override with --threads)");
            }
            int numaNodes = CLIOptions.detectNumaNodes();
            if (numaNodes > 1) {
                System.out.println("CPU: " + numaNodes + " NUMA nodes detected — for best memory-bound "
                    + "throughput pin to one node, e.g.: numactl --cpunodebind=0 --membind=0 java ...");
            }
        }

        Path modelPath = Paths.get(options.getModelPath());

        // Configure GPU: auto-detect unless explicitly disabled
        GpuConfig gpuConfig;
        if (options.isNoGpu()) {
            gpuConfig = new GpuConfig(); // disabled
            System.out.println("GPU: disabled by --no-gpu");
        } else if (options.isGpuEnabled()) {
            // Explicit --gpu or --gpu-device: use user's choice
            gpuConfig = new GpuConfig();
            gpuConfig.setEnabled(true);
            gpuConfig.setDeviceId(options.getGpuDeviceId());
            gpuConfig.setGpuLayers(options.getGpuLayers());
            if (options.getMoeOptimized() != null) {
                gpuConfig.setMoeOptimized(options.getMoeOptimized());
            }
            gpuConfig.setBackend(parseGpuBackend(options.getGpuBackend()));
            gpuConfig.setMemoryMode(options.getGpuMemoryMode());
        } else {
            // Auto-detect: probe hardware and configure optimally
            gpuConfig = LLMEngine.autoConfigureGpu(modelPath);
            if (gpuConfig == null) {
                gpuConfig = new GpuConfig(); // no GPU found
            }
            if (options.getMoeOptimized() != null) {
                gpuConfig.setMoeOptimized(options.getMoeOptimized());
            }
            gpuConfig.setBackend(parseGpuBackend(options.getGpuBackend()));
            gpuConfig.setMemoryMode(options.getGpuMemoryMode());
        }

        // Show hardware plan
        LLMEngine.HardwarePlan plan = LLMEngine.buildHardwarePlan(modelPath, options.getContextLength());
        System.out.println("\n--- Hardware Plan ---");
        System.out.println(plan.summary());
        System.out.println("--------------------\n");

        if (!plan.isRecommended() && !options.isForce()) {
            System.out.print("This configuration is not recommended. Continue? [y/N] ");
            System.out.flush();
            BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
            String answer = br.readLine();
            if (answer == null || (!answer.trim().equalsIgnoreCase("y") && !answer.trim().equalsIgnoreCase("yes"))) {
                System.out.println("Aborted.");
                return;
            }
        }

        // --auto-tune (Phase 2.1): empirically compare GPU placement against CPU-only and keep the
        // faster. This catches the case where too little of a partial-fit model lands in VRAM (or the
        // GPU/bus is weak) so that GPU placement is actually slower than running entirely on the CPU.
        if (options.isAutoTune() && !options.isNoGpu()) {
            gpuConfig = autoTune(modelPath, gpuConfig, options);
        }

        System.out.println("Loading model: " + modelPath);

        try (LLMEngine engine = LLMEngine.load(modelPath, options.getContextLength(), gpuConfig,
                options.isGpuChainEnabled())) {
            // Enable thinking/reasoning mode if requested
            if (options.isThinking()) {
                engine.getChatTemplate().setThinkingEnabled(true);
            }
            if (options.isShowInfo()) {
                ModelInfoPrinter.print(engine.getModelInfo());
                return;
            }

            if (options.isInteractive()) {
                runInteractive(engine);
            } else if (options.getPrompt() != null) {
                runSinglePrompt(engine, options.getPrompt());
            } else {
                System.err.println("Error: provide --prompt or --interactive");
                CLIOptions.printUsage();
            }
        }
    }

    private void runSinglePrompt(LLMEngine engine, String prompt) {
        // Tier 3: speculative decoding when --draft-model provided. Standalone path,
        // does not affect the normal engine.generate() flow when the flag is absent.
        if (options.getDraftModelPath() != null) {
            runSpeculative(engine, prompt);
            return;
        }
        SamplerConfig samplerConfig = options.toSamplerConfig();
        GenerationRequest request = GenerationRequest.builder()
            .prompt(prompt)
            .maxTokens(options.getMaxTokens())
            .samplerConfig(samplerConfig)
            .build();

        System.out.println("\n--- Generation ---");
        GenerationResponse response = engine.generate(request, (token, id) -> {
            System.out.print(token);
            System.out.flush();
            return true;
        });
        System.out.println("\n--- Stats ---");
        System.out.printf("Tokens: %d prompt + %d generated in %dms (%.1f tok/s)%n",
            response.promptTokenCount(), response.tokenCount(), response.timeMs(), response.tokensPerSecond());

        if (response.evaluation() != null) {
            System.out.println("--- Evaluation ---");
            for (EvaluationResult eval : response.evaluation()) {
                System.out.println("  " + eval);
            }
        }
    }

    /**
     * --auto-tune calibration (Phase 2.1): measure steady-state decode tok/s for the heuristic GPU
     * placement and for CPU-only, then return the faster configuration. A one-time, opt-in
     * calibration (it loads the model a couple of extra times) that replaces "guess from file size"
     * with a measurement, and auto-corrects the partial-fit footgun where GPU placement is slower
     * than CPU because too little fits VRAM or the GPU/bus is weak.
     */
    private GpuConfig autoTune(Path modelPath, GpuConfig gpuConfig, CLIOptions options) {
        final String calPrompt = "Write one short paragraph about the history of computing.";
        final int calTokens = 24;
        System.out.println("=== Auto-tune calibration (measuring placements) ===");
        double tpsGpu = measurePlacement(modelPath, gpuConfig, options, calPrompt, calTokens, "GPU placement");
        GpuConfig cpuCfg = new GpuConfig(); // GPU disabled
        double tpsCpu = measurePlacement(modelPath, cpuCfg, options, calPrompt, calTokens, "CPU-only");
        if (!Double.isNaN(tpsGpu) && tpsGpu >= tpsCpu) {
            System.out.printf("Auto-tune: GPU placement wins (%.1f vs %.1f tok/s) — using GPU.%n", tpsGpu, tpsCpu);
            return gpuConfig;
        } else if (!Double.isNaN(tpsCpu)) {
            System.out.printf("Auto-tune: CPU-only is faster (%.1f vs %.1f tok/s) — too little fits VRAM "
                + "or the GPU/bus is weak; using CPU.%n", tpsCpu, tpsGpu);
            return cpuCfg;
        }
        return gpuConfig;
    }

    /** Load the model with {@code cfg}, run a discarded warm-up then a measured decode, return tok/s. */
    private double measurePlacement(Path modelPath, GpuConfig cfg, CLIOptions options,
                                    String prompt, int nTokens, String label) {
        try (LLMEngine e = LLMEngine.load(modelPath, options.getContextLength(), cfg, options.isGpuChainEnabled())) {
            SamplerConfig sampler = options.toSamplerConfig();
            e.generate(GenerationRequest.builder().prompt(prompt).maxTokens(6).samplerConfig(sampler).build());
            GenerationResponse r = e.generate(
                GenerationRequest.builder().prompt(prompt).maxTokens(nTokens).samplerConfig(sampler).build());
            System.out.printf("  %-14s %.1f tok/s%n", label + ":", r.tokensPerSecond());
            return r.tokensPerSecond();
        } catch (Exception ex) {
            System.out.println("  " + label + ": unavailable (" + ex.getMessage() + ")");
            return Double.NaN;
        }
    }

    /** Tier 3 speculative decoding path. Loads the draft model, runs the algorithm,
     *  prints stats. Completely separate from normal generation. */
    private void runSpeculative(LLMEngine target, String prompt) {
        String draftPath = options.getDraftModelPath();
        int K = options.getSpeculationDepth();
        System.out.println("\n--- Speculative Decoding ---");
        System.out.println("Target: " + target.getModelInfo().name());
        System.out.println("Draft model: " + draftPath);
        System.out.println("Speculation depth K=" + K);

        // Load draft on CPU by default to avoid GPU memory contention with target.
        // User can override with environment but for first iteration we keep it simple.
        it.denzosoft.llmplayer.gpu.GpuConfig draftGpu = new it.denzosoft.llmplayer.gpu.GpuConfig();
        draftGpu.setEnabled(false);
        try (LLMEngine draft = LLMEngine.load(java.nio.file.Path.of(draftPath),
                options.getContextLength(), draftGpu, false)) {
            it.denzosoft.llmplayer.spec.SpeculativeDecoder spec =
                new it.denzosoft.llmplayer.spec.SpeculativeDecoder(target, draft, K);
            GenerationRequest request = GenerationRequest.builder()
                .prompt(prompt)
                .maxTokens(options.getMaxTokens())
                .build();
            System.out.println("\n--- Generation ---");
            GenerationResponse response = spec.generate(request, (token, id) -> {
                System.out.print(token);
                System.out.flush();
                return true;
            });
            System.out.println("\n--- Stats ---");
            System.out.printf("Tokens: %d prompt + %d generated in %dms (%.1f tok/s)%n",
                response.promptTokenCount(), response.tokenCount(),
                response.timeMs(), response.tokensPerSecond());
            System.out.printf("Acceptance rate: %.1f%% (%d/%d draft tokens accepted, %d rounds)%n",
                spec.getAcceptanceRate() * 100,
                spec.getTotalAcceptedTokens(), spec.getTotalDraftTokens(),
                spec.getTotalSpeculationRounds());
        } catch (Exception e) {
            System.err.println("Speculative decoding failed: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private void runInteractive(LLMEngine engine) throws IOException {
        System.out.println("\nInteractive mode. Type 'quit' to exit, 'info' for model info.");
        System.out.println("Model: " + engine.getModelInfo().name());
        System.out.println();

        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        SamplerConfig samplerConfig = options.toSamplerConfig();

        while (true) {
            System.out.print("> ");
            System.out.flush();
            String input = reader.readLine();
            if (input == null || input.equalsIgnoreCase("quit") || input.equalsIgnoreCase("exit")) {
                System.out.println("Bye!");
                break;
            }
            if (input.equalsIgnoreCase("info")) {
                ModelInfoPrinter.print(engine.getModelInfo());
                continue;
            }
            if (input.trim().isEmpty()) continue;

            GenerationRequest request = GenerationRequest.builder()
                .prompt(input)
                .maxTokens(options.getMaxTokens())
                .samplerConfig(samplerConfig)
                .build();

            System.out.println();
            GenerationResponse response = engine.generate(request, (token, id) -> {
                System.out.print(token);
                System.out.flush();
                return true;
            });
            System.out.printf("%n[%d tokens, %.1f tok/s]%n%n",
                response.tokenCount(), response.tokensPerSecond());
        }
    }

    private void listGpuDevices() {
        boolean found = false;

        // CUDA devices
        try {
            Class<?> ctxClass = Class.forName("it.denzosoft.llmplayer.gpu.CudaContext");
            @SuppressWarnings("unchecked")
            List<?> devices = (List<?>) ctxClass.getMethod("enumerateDevices").invoke(null);
            if (!devices.isEmpty()) {
                System.out.println("CUDA devices:");
                for (Object dev : devices) {
                    System.out.println("  " + dev);
                }
                found = true;
            }
        } catch (Exception ignored) {}

        // OpenCL devices
        try {
            Class<?> ctxClass = Class.forName("it.denzosoft.llmplayer.gpu.OpenCLContext");
            @SuppressWarnings("unchecked")
            List<?> devices = (List<?>) ctxClass.getMethod("enumerateDevices").invoke(null);
            if (!devices.isEmpty()) {
                System.out.println("OpenCL devices:");
                for (Object dev : devices) {
                    System.out.println("  " + dev);
                }
                found = true;
            }
        } catch (ClassNotFoundException e) {
            if (!found) System.out.println("GPU support requires Java 21+.");
            return;
        } catch (Exception e) {
            System.out.println("Error enumerating OpenCL devices: " + e.getMessage());
        }

        if (!found) {
            System.out.println("No GPU devices found.");
            System.out.println("For CUDA: install NVIDIA driver (libcuda.so) and NVRTC (libnvrtc.so).");
            System.out.println("For OpenCL: install OpenCL drivers (libOpenCL.so on Linux).");
        }
    }

    private static GpuConfig.GpuBackend parseGpuBackend(String backend) {
        if ("cuda".equals(backend)) return GpuConfig.GpuBackend.CUDA;
        if ("opencl".equals(backend)) return GpuConfig.GpuBackend.OPENCL;
        return GpuConfig.GpuBackend.AUTO;
    }
}
