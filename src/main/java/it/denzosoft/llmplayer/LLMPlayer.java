package it.denzosoft.llmplayer;

import it.denzosoft.llmplayer.cli.CLIOptions;
import it.denzosoft.llmplayer.cli.CLIRunner;
import it.denzosoft.llmplayer.ui.LLMPlayerUI;
import it.denzosoft.llmplayer.web.WebServer;

import java.io.IOException;

/**
 * LLMPlayer - Pure Java LLM inference engine.
 *
 * Supports GGUF models with quantization (Q3_K, Q4_K, Q4_0, Q8_0, etc.)
 * Uses Java Vector API (SIMD), Panama Foreign Memory (mmap), and Virtual Threads.
 *
 * Launch modes:
 *   (default)        Desktop GUI (Swing) - main control panel
 *   --web            Web UI server on configurable port
 *   --model <path>   CLI mode (single prompt or interactive)
 */
public class LLMPlayer {

    public static final String VERSION = "1.0.0";

    public static void main(String[] args) {
        System.out.println("LLMPlayer v" + VERSION + " - Pure Java LLM Inference Engine");
        System.out.println();

        CLIOptions options = CLIOptions.parse(args);

        try {
            if (options.isHelp()) {
                CLIOptions.printUsage();
            } else if (options.isGpuList()) {
                // GPU device listing: handled by CLIRunner
                new CLIRunner(options).run();
            } else if (options.isWebMode()) {
                new WebServer(options.getPort(), options.getGgufDirectory()).startBlocking();
            } else if (options.getModelPath() != null) {
                // CLI mode: --model was specified
                new CLIRunner(options).run();
            } else {
                // Default: launch Swing desktop UI
                LLMPlayerUI.launch(options.getGgufDirectory());
            }
        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        } catch (Exception e) {
            System.err.println("Unexpected error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
