package it.denzosoft.llmplayer.tokenizer;

import it.denzosoft.llmplayer.model.ModelArchitecture;

import java.util.List;

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

    /**
     * Format a multi-turn conversation for the OpenAI-compatible API.
     * Each message is a String[] of {role, content}. Roles: "system", "user", "assistant".
     * Returns the formatted prompt ready for tokenization (BOS is handled by the engine).
     */
    public String formatConversation(List<String[]> messages) {
        if (architecture == ModelArchitecture.LLAMA) {
            return formatLlama3Conversation(messages);
        } else if (architecture == ModelArchitecture.QWEN2 || architecture == ModelArchitecture.QWEN3
                || architecture == ModelArchitecture.QWEN3MOE) {
            return formatQwenConversation(messages);
        } else if (architecture == ModelArchitecture.GLM4) {
            return formatGLM4Conversation(messages);
        } else if (architecture == ModelArchitecture.DEEPSEEK2) {
            return formatDeepSeekConversation(messages);
        } else if (architecture == ModelArchitecture.PHI3) {
            return formatPhi3Conversation(messages);
        } else if (architecture == ModelArchitecture.MISTRAL3) {
            return formatMistral3Conversation(messages);
        }
        return formatLlama3Conversation(messages);
    }

    private String formatLlama3Conversation(List<String[]> messages) {
        StringBuilder sb = new StringBuilder();
        for (String[] msg : messages) {
            sb.append("<|start_header_id|>").append(msg[0]).append("<|end_header_id|>\n\n");
            sb.append(msg[1]).append("<|eot_id|>");
        }
        sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n");
        return sb.toString();
    }

    private String formatQwenConversation(List<String[]> messages) {
        StringBuilder sb = new StringBuilder();
        for (String[] msg : messages) {
            sb.append("<|im_start|>").append(msg[0]).append("\n");
            sb.append(msg[1]).append("<|im_end|>\n");
        }
        sb.append("<|im_start|>assistant\n");
        return sb.toString();
    }

    private String formatGLM4Conversation(List<String[]> messages) {
        StringBuilder sb = new StringBuilder("[gMASK]<sop>");
        for (String[] msg : messages) {
            sb.append("<|").append(msg[0]).append("|>\n");
            sb.append(msg[1]);
        }
        sb.append("<|assistant|>\n");
        return sb.toString();
    }

    private String formatDeepSeekConversation(List<String[]> messages) {
        StringBuilder sb = new StringBuilder();
        for (String[] msg : messages) {
            if ("system".equals(msg[0])) {
                sb.append(msg[1]).append("\n\n");
            } else if ("user".equals(msg[0])) {
                sb.append("User: ").append(msg[1]).append("\n\n");
            } else if ("assistant".equals(msg[0])) {
                sb.append("Assistant: ").append(msg[1]).append("\n\n");
            }
        }
        sb.append("Assistant:");
        return sb.toString();
    }

    private String formatPhi3Conversation(List<String[]> messages) {
        StringBuilder sb = new StringBuilder();
        for (String[] msg : messages) {
            sb.append("<|").append(msg[0]).append("|>\n");
            sb.append(msg[1]).append("<|end|>\n");
        }
        sb.append("<|assistant|>\n");
        return sb.toString();
    }

    private String formatMistral3Conversation(List<String[]> messages) {
        StringBuilder sb = new StringBuilder();
        String systemMsg = null;
        boolean firstUser = true;
        for (String[] msg : messages) {
            if ("system".equals(msg[0])) {
                systemMsg = msg[1];
            } else if ("user".equals(msg[0])) {
                sb.append("[INST] ");
                if (firstUser && systemMsg != null) {
                    sb.append(systemMsg).append("\n\n");
                }
                sb.append(msg[1]).append(" [/INST]");
                firstUser = false;
            } else if ("assistant".equals(msg[0])) {
                sb.append(msg[1]).append("</s>");
            }
        }
        return sb.toString();
    }
}
