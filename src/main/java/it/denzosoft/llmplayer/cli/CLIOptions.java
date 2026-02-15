package it.denzosoft.llmplayer.cli;

import it.denzosoft.llmplayer.sampler.SamplerConfig;

public class CLIOptions {

    private String modelPath;
    private String prompt;
    private boolean interactive;
    private int maxTokens = 256;
    private float temperature = 0.7f;
    private int topK = 40;
    private float topP = 0.9f;
    private float repetitionPenalty = 1.1f;
    private long seed = System.nanoTime();
    private int threads = Runtime.getRuntime().availableProcessors();
    private boolean showInfo;
    private boolean help;
    private int contextLength = 2048;
    private boolean webMode;
    private int port = 8080;
    private String ggufDirectory = "gguf";
    private boolean gpuEnabled;
    private boolean noGpu;
    private int gpuDeviceId;
    private boolean gpuList;
    private int gpuLayers = -1;

    public static CLIOptions parse(String[] args) {
        CLIOptions opts = new CLIOptions();
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if ("--model".equals(arg) || "-m".equals(arg)) {
                opts.modelPath = args[++i];
            } else if ("--prompt".equals(arg) || "-p".equals(arg)) {
                opts.prompt = args[++i];
            } else if ("--interactive".equals(arg) || "-i".equals(arg)) {
                opts.interactive = true;
            } else if ("--max-tokens".equals(arg) || "-n".equals(arg)) {
                opts.maxTokens = Integer.parseInt(args[++i]);
            } else if ("--temperature".equals(arg) || "-t".equals(arg)) {
                opts.temperature = Float.parseFloat(args[++i]);
            } else if ("--top-k".equals(arg)) {
                opts.topK = Integer.parseInt(args[++i]);
            } else if ("--top-p".equals(arg)) {
                opts.topP = Float.parseFloat(args[++i]);
            } else if ("--repetition-penalty".equals(arg)) {
                opts.repetitionPenalty = Float.parseFloat(args[++i]);
            } else if ("--seed".equals(arg)) {
                opts.seed = Long.parseLong(args[++i]);
            } else if ("--threads".equals(arg)) {
                opts.threads = Integer.parseInt(args[++i]);
            } else if ("--info".equals(arg)) {
                opts.showInfo = true;
            } else if ("--context-length".equals(arg) || "-c".equals(arg)) {
                opts.contextLength = Integer.parseInt(args[++i]);
            } else if ("--web".equals(arg) || "-w".equals(arg)) {
                opts.webMode = true;
            } else if ("--port".equals(arg)) {
                opts.port = Integer.parseInt(args[++i]);
            } else if ("--gguf-dir".equals(arg)) {
                opts.ggufDirectory = args[++i];
            } else if ("--no-gpu".equals(arg)) {
                opts.noGpu = true;
            } else if ("--gpu".equals(arg)) {
                opts.gpuEnabled = true;
            } else if ("--gpu-device".equals(arg)) {
                opts.gpuEnabled = true;
                opts.gpuDeviceId = Integer.parseInt(args[++i]);
            } else if ("--gpu-layers".equals(arg)) {
                opts.gpuLayers = Integer.parseInt(args[++i]);
            } else if ("--gpu-list".equals(arg)) {
                opts.gpuList = true;
            } else if ("--help".equals(arg) || "-h".equals(arg)) {
                opts.help = true;
            } else {
                // If no flag, treat as model path if not set
                if (opts.modelPath == null && !arg.startsWith("-")) {
                    opts.modelPath = arg;
                } else {
                    System.err.println("Unknown option: " + arg);
                }
            }
        }
        return opts;
    }

    public SamplerConfig toSamplerConfig() {
        return new SamplerConfig(temperature, topK, topP, repetitionPenalty, seed);
    }

    // Getters
    public String getModelPath() { return modelPath; }
    public String getPrompt() { return prompt; }
    public boolean isInteractive() { return interactive; }
    public int getMaxTokens() { return maxTokens; }
    public float getTemperature() { return temperature; }
    public int getThreads() { return threads; }
    public boolean isShowInfo() { return showInfo; }
    public boolean isHelp() { return help; }
    public int getContextLength() { return contextLength; }
    public boolean isWebMode() { return webMode; }
    public int getPort() { return port; }
    public String getGgufDirectory() { return ggufDirectory; }
    public boolean isGpuEnabled() { return gpuEnabled; }
    public int getGpuDeviceId() { return gpuDeviceId; }
    public boolean isGpuList() { return gpuList; }
    public boolean isNoGpu() { return noGpu; }
    public int getGpuLayers() { return gpuLayers; }

    public static void printUsage() {
        System.out.println("Usage: java -jar LLMPlayer.jar [options]");
        System.out.println();
        System.out.println("Options:");
        System.out.println("  --model, -m <path>       Path to GGUF model file");
        System.out.println("  --prompt, -p <text>      Input prompt");
        System.out.println("  --interactive, -i        Interactive chat mode");
        System.out.println("  --max-tokens, -n <num>   Max tokens to generate (default: 256)");
        System.out.println("  --temperature, -t <num>  Sampling temperature (default: 0.7)");
        System.out.println("  --top-k <num>            Top-K sampling (default: 40)");
        System.out.println("  --top-p <num>            Top-P nucleus sampling (default: 0.9)");
        System.out.println("  --repetition-penalty <n> Repetition penalty (default: 1.1)");
        System.out.println("  --seed <num>             Random seed");
        System.out.println("  --threads <num>          Number of threads");
        System.out.println("  --context-length, -c <n> Max context length (default: 2048)");
        System.out.println("  --info                   Show model info and exit");
        System.out.println("  --web, -w                Start web UI server");
        System.out.println("  --port <num>             Web UI port (default: 8080)");
        System.out.println("  --gguf-dir <path>        GGUF models directory (default: gguf)");
        System.out.println("  --gpu                    Force GPU (default: auto-detect)");
        System.out.println("  --no-gpu                 Disable GPU, CPU only");
        System.out.println("  --gpu-device <id>        Select GPU device by index (default: best)");
        System.out.println("  --gpu-layers <n>         GPU layers: -1=auto, 0=all, N=first N (default: -1)");
        System.out.println("  --gpu-list               List available GPU devices and exit");
        System.out.println("  --help, -h               Show this help");
    }
}
