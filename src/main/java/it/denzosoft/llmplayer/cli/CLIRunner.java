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

        // Configure thread pool
        if (options.getThreads() > 0) {
            System.setProperty("java.util.concurrent.ForkJoinPool.common.parallelism",
                String.valueOf(options.getThreads()));
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
        } else {
            // Auto-detect: probe hardware and configure optimally
            gpuConfig = LLMEngine.autoConfigureGpu(modelPath);
            if (gpuConfig == null) {
                gpuConfig = new GpuConfig(); // no GPU found
            }
        }

        // Show hardware plan
        LLMEngine.HardwarePlan plan = LLMEngine.buildHardwarePlan(modelPath, options.getContextLength());
        System.out.println("\n--- Hardware Plan ---");
        System.out.println(plan.summary());
        System.out.println("--------------------\n");

        if (!plan.isRecommended()) {
            System.out.print("This configuration is not recommended. Continue? [y/N] ");
            System.out.flush();
            BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
            String answer = br.readLine();
            if (answer == null || (!answer.trim().equalsIgnoreCase("y") && !answer.trim().equalsIgnoreCase("yes"))) {
                System.out.println("Aborted.");
                return;
            }
        }

        System.out.println("Loading model: " + modelPath);

        try (LLMEngine engine = LLMEngine.load(modelPath, options.getContextLength(), gpuConfig)) {
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
        // Use reflection to call OpenCLContext.enumerateDevices() (Java 21 only)
        try {
            Class<?> ctxClass = Class.forName("it.denzosoft.llmplayer.gpu.OpenCLContext");
            java.lang.reflect.Method enumMethod = ctxClass.getMethod("enumerateDevices");
            @SuppressWarnings("unchecked")
            List<?> devices = (List<?>) enumMethod.invoke(null);
            if (devices.isEmpty()) {
                System.out.println("No OpenCL devices found.");
                System.out.println("Make sure OpenCL drivers are installed (libOpenCL.so on Linux).");
            } else {
                System.out.println("Available OpenCL devices:");
                for (Object dev : devices) {
                    System.out.println("  " + dev);
                }
            }
        } catch (ClassNotFoundException e) {
            System.out.println("GPU support requires Java 21+.");
        } catch (Exception e) {
            System.out.println("Error enumerating GPU devices: " + e.getMessage());
        }
    }
}
