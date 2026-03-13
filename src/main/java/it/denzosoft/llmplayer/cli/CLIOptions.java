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
    private boolean gpuChainEnabled = true;  // GPU kernel chaining (default: enabled)
    private String gpuBackend = "auto";      // auto, cuda, opencl
    private String gpuMemoryMode = "device"; // device, managed, host-mapped

    // Fine-tuning options
    private boolean fineTune;
    private String targetModel;
    private String generatorModel;
    private String sourcePath;
    private String documentsPath;
    private String dataPath;
    private String schemaPath;
    private String ftOutput;
    private String ftDataType;
    private int loraRank = 16;
    private int loraAlpha = 32;
    private int epochs = 3;
    private float learningRate = 2e-4f;
    private int pairsPerChunk = 5;
    private int chunkSize = 0;
    private String ftWorkDir = "work";
    private boolean datasetOnly;
    private String trainDataset;

    private boolean force;         // Skip confirmation prompts (e.g., RAM warning)
    private boolean thinking;      // Enable thinking/reasoning mode (SmolLM3, Qwen3, Qwen3.5)

    // Download options
    private String downloadSpec;   // HuggingFace repo: "owner/repo" or "owner/repo/file.gguf"
    private String hfToken;        // HuggingFace API token (optional, for private repos)

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
            } else if ("--gpu-chain".equals(arg)) {
                opts.gpuChainEnabled = true;
            } else if ("--no-gpu-chain".equals(arg)) {
                opts.gpuChainEnabled = false;
            } else if ("--gpu-backend".equals(arg)) {
                opts.gpuBackend = args[++i].toLowerCase();
            } else if ("--gpu-memory".equals(arg)) {
                opts.gpuMemoryMode = args[++i].toLowerCase();
            } else if ("--fine-tune".equals(arg)) {
                opts.fineTune = true;
            } else if ("--target-model".equals(arg)) {
                opts.targetModel = args[++i];
                opts.fineTune = true;
            } else if ("--generator-model".equals(arg)) {
                opts.generatorModel = args[++i];
            } else if ("--source".equals(arg)) {
                opts.sourcePath = args[++i];
            } else if ("--documents".equals(arg)) {
                opts.documentsPath = args[++i];
            } else if ("--data".equals(arg)) {
                opts.dataPath = args[++i];
            } else if ("--schema".equals(arg)) {
                opts.schemaPath = args[++i];
            } else if ("--ft-output".equals(arg)) {
                opts.ftOutput = args[++i];
            } else if ("--ft-data-type".equals(arg)) {
                opts.ftDataType = args[++i];
            } else if ("--lora-rank".equals(arg)) {
                opts.loraRank = Integer.parseInt(args[++i]);
            } else if ("--lora-alpha".equals(arg)) {
                opts.loraAlpha = Integer.parseInt(args[++i]);
            } else if ("--epochs".equals(arg)) {
                opts.epochs = Integer.parseInt(args[++i]);
            } else if ("--learning-rate".equals(arg)) {
                opts.learningRate = Float.parseFloat(args[++i]);
            } else if ("--pairs-per-chunk".equals(arg)) {
                opts.pairsPerChunk = Integer.parseInt(args[++i]);
            } else if ("--chunk-size".equals(arg)) {
                opts.chunkSize = Integer.parseInt(args[++i]);
            } else if ("--ft-work-dir".equals(arg)) {
                opts.ftWorkDir = args[++i];
            } else if ("--dataset-only".equals(arg)) {
                opts.datasetOnly = true;
            } else if ("--train-dataset".equals(arg)) {
                opts.trainDataset = args[++i];
            } else if ("--thinking".equals(arg)) {
                opts.thinking = true;
            } else if ("--force".equals(arg) || "-y".equals(arg)) {
                opts.force = true;
            } else if ("--download".equals(arg)) {
                opts.downloadSpec = args[++i];
            } else if ("--hf-token".equals(arg)) {
                opts.hfToken = args[++i];
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
    public boolean isGpuChainEnabled() { return gpuChainEnabled; }
    public String getGpuBackend() { return gpuBackend; }
    public String getGpuMemoryMode() { return gpuMemoryMode; }

    // Fine-tuning getters
    public boolean isFineTune() { return fineTune; }
    public String getTargetModel() { return targetModel; }
    public String getGeneratorModel() { return generatorModel; }
    public String getSourcePath() { return sourcePath; }
    public String getDocumentsPath() { return documentsPath; }
    public String getDataPath() { return dataPath; }
    public String getSchemaPath() { return schemaPath; }
    public String getFtOutput() { return ftOutput; }
    public String getFtDataType() { return ftDataType; }
    public int getLoraRank() { return loraRank; }
    public int getLoraAlpha() { return loraAlpha; }
    public int getEpochs() { return epochs; }
    public float getLearningRate() { return learningRate; }
    public int getPairsPerChunk() { return pairsPerChunk; }
    public int getChunkSize() { return chunkSize; }
    public String getFtWorkDir() { return ftWorkDir; }
    public boolean isDatasetOnly() { return datasetOnly; }
    public String getTrainDataset() { return trainDataset; }

    public boolean isForce() { return force; }
    public boolean isThinking() { return thinking; }

    // Download getters
    public String getDownloadSpec() { return downloadSpec; }
    public String getHfToken() { return hfToken; }

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
        System.out.println("  --gpu-chain              Enable GPU kernel chaining (default: on)");
        System.out.println("  --no-gpu-chain           Disable GPU kernel chaining");
        System.out.println("  --gpu-backend <backend>  GPU backend: auto, cuda, opencl (default: auto)");
        System.out.println("  --gpu-memory <mode>      GPU memory: device, managed, host-mapped (default: device)");
        System.out.println("  --thinking               Enable extended thinking/reasoning (SmolLM3, Qwen3, Qwen3.5)");
        System.out.println();
        System.out.println("Fine-tuning:");
        System.out.println("  --fine-tune              Enable fine-tuning mode");
        System.out.println("  --target-model <path>    Target GGUF model to fine-tune");
        System.out.println("  --generator-model <path> Generator model for dataset (default: target)");
        System.out.println("  --source <path>          Source code directory (code scenario)");
        System.out.println("  --documents <path>       Documents directory (document scenario)");
        System.out.println("  --data <path>            Data file CSV/JSON (structured scenario)");
        System.out.println("  --schema <path>          SQL DDL schema file (structured scenario)");
        System.out.println("  --ft-output <path>       Output fine-tuned GGUF path");
        System.out.println("  --ft-data-type <type>    Force data type: code, document, structured");
        System.out.println("  --lora-rank <n>          LoRA rank (default: 16)");
        System.out.println("  --lora-alpha <n>         LoRA alpha (default: 32)");
        System.out.println("  --epochs <n>             Training epochs (default: 3)");
        System.out.println("  --learning-rate <f>      Learning rate (default: 2e-4)");
        System.out.println("  --pairs-per-chunk <n>    Q&A pairs per chunk (default: 5)");
        System.out.println("  --chunk-size <n>         Max tokens per chunk (default: auto from context)");
        System.out.println("  --ft-work-dir <path>     Working directory (default: work)");
        System.out.println("  --dataset-only           Stop after dataset generation");
        System.out.println("  --train-dataset <path>   Skip to training with existing dataset");
        System.out.println();
        System.out.println("Model download:");
        System.out.println("  --download <repo>        Download GGUF from HuggingFace (e.g. \"bartowski/Llama-3.2-1B-Instruct-GGUF\")");
        System.out.println("                           Or specify file: \"owner/repo/model-Q4_K_M.gguf\"");
        System.out.println("  --hf-token <token>       HuggingFace API token (for private/gated repos)");
        System.out.println("  --force, -y              Skip confirmation prompts");
        System.out.println("  --help, -h               Show this help");
    }
}
