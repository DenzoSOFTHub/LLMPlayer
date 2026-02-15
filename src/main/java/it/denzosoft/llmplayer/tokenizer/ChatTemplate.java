package it.denzosoft.llmplayer.tokenizer;

import it.denzosoft.llmplayer.model.ModelArchitecture;

public class ChatTemplate {

    private final ModelArchitecture architecture;
    private final String chatTemplate;

    public ChatTemplate(ModelArchitecture architecture, String chatTemplate) {
        this.architecture = architecture;
        this.chatTemplate = chatTemplate;
    }

    public String formatUserMessage(String userMessage) {
        if (architecture == ModelArchitecture.LLAMA) {
            return formatLlama3(userMessage);
        } else if (architecture == ModelArchitecture.QWEN2 || architecture == ModelArchitecture.QWEN3
                || architecture == ModelArchitecture.QWEN3MOE) {
            return formatQwen(userMessage);
        } else if (architecture == ModelArchitecture.GLM4) {
            return formatGLM4(userMessage);
        } else if (architecture == ModelArchitecture.DEEPSEEK2) {
            return formatDeepSeek(userMessage);
        } else if (architecture == ModelArchitecture.PHI3) {
            return formatPhi3(userMessage);
        } else if (architecture == ModelArchitecture.MISTRAL3) {
            return formatMistral3(userMessage);
        }
        return formatLlama3(userMessage); // default fallback
    }

    public String formatChat(String systemMessage, String userMessage) {
        if (architecture == ModelArchitecture.LLAMA) {
            return formatLlama3Chat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.QWEN2 || architecture == ModelArchitecture.QWEN3
                || architecture == ModelArchitecture.QWEN3MOE) {
            return formatQwenChat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.GLM4) {
            return formatGLM4Chat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.DEEPSEEK2) {
            return formatDeepSeekChat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.PHI3) {
            return formatPhi3Chat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.MISTRAL3) {
            return formatMistral3Chat(systemMessage, userMessage);
        }
        return formatLlama3Chat(systemMessage, userMessage); // default fallback
    }

    // Llama 3 format (BOS is prepended by engine, not included here)
    private String formatLlama3(String userMessage) {
        return "<|start_header_id|>user<|end_header_id|>\n\n" +
               userMessage + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    }

    private String formatLlama3Chat(String systemMessage, String userMessage) {
        return "<|start_header_id|>system<|end_header_id|>\n\n" +
               systemMessage + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" +
               userMessage + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    }

    // Qwen format
    private String formatQwen(String userMessage) {
        return "<|im_start|>user\n" + userMessage + "<|im_end|>\n<|im_start|>assistant\n";
    }

    private String formatQwenChat(String systemMessage, String userMessage) {
        return "<|im_start|>system\n" + systemMessage + "<|im_end|>\n" +
               "<|im_start|>user\n" + userMessage + "<|im_end|>\n<|im_start|>assistant\n";
    }

    // GLM4 format
    private String formatGLM4(String userMessage) {
        return "[gMASK]<sop><|user|>\n" + userMessage + "<|assistant|>\n";
    }

    private String formatGLM4Chat(String systemMessage, String userMessage) {
        return "[gMASK]<sop><|system|>\n" + systemMessage +
               "<|user|>\n" + userMessage + "<|assistant|>\n";
    }

    // DeepSeek format (BOS is prepended by engine, not included here)
    private String formatDeepSeek(String userMessage) {
        return "User: " + userMessage + "\n\nAssistant:";
    }

    private String formatDeepSeekChat(String systemMessage, String userMessage) {
        return systemMessage + "\n\nUser: " + userMessage + "\n\nAssistant:";
    }

    // Phi-3/Phi-4 format
    private String formatPhi3(String userMessage) {
        return "<|user|>\n" + userMessage + "<|end|>\n<|assistant|>\n";
    }

    private String formatPhi3Chat(String systemMessage, String userMessage) {
        return "<|system|>\n" + systemMessage + "<|end|>\n" +
               "<|user|>\n" + userMessage + "<|end|>\n<|assistant|>\n";
    }

    // Mistral3/Devstral format
    private String formatMistral3(String userMessage) {
        return "[INST] " + userMessage + " [/INST]";
    }

    private String formatMistral3Chat(String systemMessage, String userMessage) {
        return "[INST] " + systemMessage + "\n\n" + userMessage + " [/INST]";
    }
}
