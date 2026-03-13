package it.denzosoft.llmplayer.tokenizer;

import it.denzosoft.llmplayer.model.ModelArchitecture;

import java.util.List;
import java.util.Map;

public class ChatTemplate {

    private final ModelArchitecture architecture;
    private final String chatTemplate;
    // GLM-4.7-Flash uses DEEPSEEK2 GGUF architecture but needs GLM4 chat format
    private final boolean isGlmVariant;
    // Thinking/reasoning mode: when true, models with <think> support will reason before answering.
    // Affects SmolLM3 (/think system msg), Qwen3 (no suppressor), Qwen3.5 (remove suppressor).
    private boolean thinkingEnabled;

    public ChatTemplate(ModelArchitecture architecture, String chatTemplate) {
        this.architecture = architecture;
        this.chatTemplate = chatTemplate;
        // Detect GLM models masquerading as deepseek2 (e.g., GLM-4.7-Flash)
        this.isGlmVariant = architecture == ModelArchitecture.DEEPSEEK2
                && chatTemplate != null
                && (chatTemplate.contains("[gMASK]") || chatTemplate.contains("<|user|>"));
    }

    public void setThinkingEnabled(boolean enabled) { this.thinkingEnabled = enabled; }
    public boolean isThinkingEnabled() { return thinkingEnabled; }

    /**
     * Returns true if this model architecture supports thinking/reasoning mode.
     */
    public boolean supportsThinking() {
        return architecture == ModelArchitecture.SMOLLM3
                || architecture == ModelArchitecture.QWEN3
                || architecture == ModelArchitecture.QWEN35;
    }

    public String formatUserMessage(String userMessage) {
        if (architecture == ModelArchitecture.LLAMA || architecture == ModelArchitecture.LLAMA4) {
            return formatLlama3(userMessage);
        } else if (architecture == ModelArchitecture.QWEN35) {
            return formatQwen35(userMessage);
        } else if (architecture == ModelArchitecture.QWEN2 || architecture == ModelArchitecture.QWEN3
                || architecture == ModelArchitecture.QWEN3MOE || architecture == ModelArchitecture.SMOLLM3) {
            return formatQwen(userMessage);
        } else if (architecture == ModelArchitecture.GLM4) {
            return formatGLM4(userMessage);
        } else if (architecture == ModelArchitecture.DEEPSEEK2) {
            return isGlmVariant ? formatGLM4(userMessage) : formatDeepSeek(userMessage);
        } else if (architecture == ModelArchitecture.PHI3) {
            return formatPhi3(userMessage);
        } else if (architecture == ModelArchitecture.MISTRAL3) {
            return formatMistral3(userMessage);
        } else if (architecture == ModelArchitecture.COMMAND_R) {
            return formatCommandR(userMessage);
        } else if (architecture == ModelArchitecture.OLMO2) {
            return formatOLMo2(userMessage);
        } else if (architecture == ModelArchitecture.GEMMA2 || architecture == ModelArchitecture.GEMMA3) {
            return formatGemma(userMessage);
        } else if (architecture == ModelArchitecture.GPT_OSS) {
            return formatGptOss(userMessage);
        }
        return formatLlama3(userMessage); // default fallback
    }

    public String formatChat(String systemMessage, String userMessage) {
        if (architecture == ModelArchitecture.LLAMA || architecture == ModelArchitecture.LLAMA4) {
            return formatLlama3Chat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.QWEN35) {
            return formatQwen35Chat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.QWEN2 || architecture == ModelArchitecture.QWEN3
                || architecture == ModelArchitecture.QWEN3MOE || architecture == ModelArchitecture.SMOLLM3) {
            return formatQwenChat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.GLM4) {
            return formatGLM4Chat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.DEEPSEEK2) {
            return isGlmVariant ? formatGLM4Chat(systemMessage, userMessage) : formatDeepSeekChat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.PHI3) {
            return formatPhi3Chat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.MISTRAL3) {
            return formatMistral3Chat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.COMMAND_R) {
            return formatCommandRChat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.OLMO2) {
            return formatOLMo2Chat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.GEMMA2 || architecture == ModelArchitecture.GEMMA3) {
            return formatGemmaChat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.GPT_OSS) {
            return formatGptOssChat(systemMessage, userMessage);
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

    /**
     * Returns the thinking suffix for Qwen-style models.
     * When thinking is disabled: "<think>\n\n</think>\n\n" (suppresses reasoning).
     * When thinking is enabled: "" (model reasons freely).
     */
    private String thinkingSuffix() {
        // SmolLM3: uses /think or /no_think in system message, no suffix needed
        if (architecture == ModelArchitecture.SMOLLM3) return "";
        // Qwen3.5: suppress thinking by default, enable when flag is set
        if (architecture == ModelArchitecture.QWEN35) {
            return thinkingEnabled ? "" : "<think>\n\n</think>\n\n";
        }
        // Qwen3: thinking is natural, suppress only when explicitly disabled
        if (architecture == ModelArchitecture.QWEN3) {
            return thinkingEnabled ? "" : "<think>\n\n</think>\n\n";
        }
        return "";
    }

    // Qwen format (also used by SmolLM3)
    private String formatQwen(String userMessage) {
        // SmolLM3 with thinking: inject /think system message
        if (architecture == ModelArchitecture.SMOLLM3 && thinkingEnabled) {
            return "<|im_start|>system\n/think<|im_end|>\n" +
                   "<|im_start|>user\n" + userMessage + "<|im_end|>\n<|im_start|>assistant\n";
        }
        return "<|im_start|>user\n" + userMessage + "<|im_end|>\n<|im_start|>assistant\n" + thinkingSuffix();
    }

    private String formatQwenChat(String systemMessage, String userMessage) {
        // SmolLM3 with thinking: prepend /think to system message
        if (architecture == ModelArchitecture.SMOLLM3 && thinkingEnabled) {
            systemMessage = "/think\n" + systemMessage;
        }
        return "<|im_start|>system\n" + systemMessage + "<|im_end|>\n" +
               "<|im_start|>user\n" + userMessage + "<|im_end|>\n<|im_start|>assistant\n" + thinkingSuffix();
    }

    // Qwen3.5 format (thinking controlled by thinkingSuffix())
    private String formatQwen35(String userMessage) {
        return "<|im_start|>user\n" + userMessage + "<|im_end|>\n<|im_start|>assistant\n" + thinkingSuffix();
    }

    private String formatQwen35Chat(String systemMessage, String userMessage) {
        return "<|im_start|>system\n" + systemMessage + "<|im_end|>\n" +
               "<|im_start|>user\n" + userMessage + "<|im_end|>\n<|im_start|>assistant\n" + thinkingSuffix();
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
        if (architecture == ModelArchitecture.LLAMA || architecture == ModelArchitecture.LLAMA4) {
            return formatLlama3Conversation(messages);
        } else if (architecture == ModelArchitecture.QWEN35) {
            return formatQwen35Conversation(messages);
        } else if (architecture == ModelArchitecture.QWEN2 || architecture == ModelArchitecture.QWEN3
                || architecture == ModelArchitecture.QWEN3MOE || architecture == ModelArchitecture.SMOLLM3) {
            return formatQwenConversation(messages);
        } else if (architecture == ModelArchitecture.GLM4) {
            return formatGLM4Conversation(messages);
        } else if (architecture == ModelArchitecture.DEEPSEEK2) {
            return isGlmVariant ? formatGLM4Conversation(messages) : formatDeepSeekConversation(messages);
        } else if (architecture == ModelArchitecture.PHI3) {
            return formatPhi3Conversation(messages);
        } else if (architecture == ModelArchitecture.MISTRAL3) {
            return formatMistral3Conversation(messages);
        } else if (architecture == ModelArchitecture.COMMAND_R) {
            return formatCommandRConversation(messages);
        } else if (architecture == ModelArchitecture.OLMO2) {
            return formatOLMo2Conversation(messages);
        } else if (architecture == ModelArchitecture.GEMMA2 || architecture == ModelArchitecture.GEMMA3) {
            return formatGemmaConversation(messages);
        } else if (architecture == ModelArchitecture.GPT_OSS) {
            return formatGptOssConversation(messages);
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
        // SmolLM3 with thinking: inject /think system message if no system in messages
        if (architecture == ModelArchitecture.SMOLLM3 && thinkingEnabled) {
            boolean hasSystem = false;
            for (String[] msg : messages) {
                if ("system".equals(msg[0])) { hasSystem = true; break; }
            }
            if (!hasSystem) {
                sb.append("<|im_start|>system\n/think<|im_end|>\n");
            }
        }
        for (String[] msg : messages) {
            sb.append("<|im_start|>").append(msg[0]).append("\n");
            // SmolLM3 with thinking: prepend /think to system message content
            if (architecture == ModelArchitecture.SMOLLM3 && thinkingEnabled && "system".equals(msg[0])) {
                sb.append("/think\n");
            }
            sb.append(msg[1]).append("<|im_end|>\n");
        }
        sb.append("<|im_start|>assistant\n").append(thinkingSuffix());
        return sb.toString();
    }

    private String formatQwen35Conversation(List<String[]> messages) {
        StringBuilder sb = new StringBuilder();
        for (String[] msg : messages) {
            sb.append("<|im_start|>").append(msg[0]).append("\n");
            sb.append(msg[1]).append("<|im_end|>\n");
        }
        sb.append("<|im_start|>assistant\n").append(thinkingSuffix());
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

    // Command-R / Cohere format
    private String formatCommandR(String userMessage) {
        return "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>" + userMessage +
               "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>";
    }

    private String formatCommandRChat(String systemMessage, String userMessage) {
        return "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>" + systemMessage +
               "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>" + userMessage +
               "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>";
    }

    private String formatCommandRConversation(List<String[]> messages) {
        StringBuilder sb = new StringBuilder();
        for (String[] msg : messages) {
            sb.append("<|START_OF_TURN_TOKEN|>");
            if ("system".equals(msg[0])) {
                sb.append("<|SYSTEM_TOKEN|>");
            } else if ("user".equals(msg[0])) {
                sb.append("<|USER_TOKEN|>");
            } else if ("assistant".equals(msg[0])) {
                sb.append("<|CHATBOT_TOKEN|>");
            }
            sb.append(msg[1]).append("<|END_OF_TURN_TOKEN|>");
        }
        sb.append("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>");
        return sb.toString();
    }

    // OLMo2 format
    private String formatOLMo2(String userMessage) {
        return "<|user|>\n" + userMessage + "\n<|assistant|>\n";
    }

    private String formatOLMo2Chat(String systemMessage, String userMessage) {
        return "<|system|>\n" + systemMessage + "\n<|user|>\n" + userMessage + "\n<|assistant|>\n";
    }

    private String formatOLMo2Conversation(List<String[]> messages) {
        StringBuilder sb = new StringBuilder();
        for (String[] msg : messages) {
            sb.append("<|").append(msg[0]).append("|>\n");
            sb.append(msg[1]).append("\n");
        }
        sb.append("<|assistant|>\n");
        return sb.toString();
    }

    // Gemma2/3 format
    private String formatGemma(String userMessage) {
        return "<start_of_turn>user\n" + userMessage + "<end_of_turn>\n<start_of_turn>model\n";
    }

    private String formatGemmaChat(String systemMessage, String userMessage) {
        // Gemma uses system message as part of user turn
        return "<start_of_turn>user\n" + systemMessage + "\n\n" + userMessage +
               "<end_of_turn>\n<start_of_turn>model\n";
    }

    private String formatGemmaConversation(List<String[]> messages) {
        StringBuilder sb = new StringBuilder();
        String pendingSystem = null;
        for (String[] msg : messages) {
            if ("system".equals(msg[0])) {
                pendingSystem = msg[1];
            } else {
                sb.append("<start_of_turn>").append(msg[0].equals("assistant") ? "model" : msg[0]).append("\n");
                if (pendingSystem != null && "user".equals(msg[0])) {
                    sb.append(pendingSystem).append("\n\n");
                    pendingSystem = null;
                }
                sb.append(msg[1]).append("<end_of_turn>\n");
            }
        }
        sb.append("<start_of_turn>model\n");
        return sb.toString();
    }

    // GPT-OSS / Sonar-OSS format uses channels: analysis, commentary, final.
    // System message follows the model's expected format from its chat template.
    private static final String GPT_OSS_SYSTEM = "You are a helpful assistant.";

    private static final String GPT_OSS_GEN_PROMPT = "<|start|>assistant<|message|>";

    private String formatGptOss(String userMessage) {
        return "<|start|>system<|message|>" + GPT_OSS_SYSTEM + "<|end|>" +
               "<|start|>user<|message|>" + userMessage + "<|end|>" +
               GPT_OSS_GEN_PROMPT;
    }

    private String formatGptOssChat(String systemMessage, String userMessage) {
        return "<|start|>system<|message|>" + GPT_OSS_SYSTEM + "<|end|>" +
               "<|start|>developer<|message|>" + systemMessage + "<|end|>" +
               "<|start|>user<|message|>" + userMessage + "<|end|>" +
               GPT_OSS_GEN_PROMPT;
    }

    private String formatGptOssConversation(List<String[]> messages) {
        StringBuilder sb = new StringBuilder();
        sb.append("<|start|>system<|message|>").append(GPT_OSS_SYSTEM).append("<|end|>");
        for (String[] msg : messages) {
            String role = msg[0];
            // Map "system" role from API to "developer" role for GPT-OSS
            if ("system".equals(role)) role = "developer";
            else if ("assistant".equals(role)) {
                // Wrap assistant content in final channel format
                sb.append("<|start|>assistant<|channel|>final<|message|>").append(msg[1]).append("<|end|>");
                continue;
            }
            sb.append("<|start|>").append(role).append("<|message|>").append(msg[1]).append("<|end|>");
        }
        sb.append(GPT_OSS_GEN_PROMPT);
        return sb.toString();
    }

    // --- Tool Calling ---

    /**
     * Returns true if this model architecture uses native tool calling format
     * (SmolLM3 Hermes-style xml_tools with &lt;tool_call&gt; tags).
     */
    public boolean usesNativeToolFormat() {
        return architecture == ModelArchitecture.SMOLLM3;
    }

    /**
     * Format tool definitions as a system prompt injection.
     * SmolLM3 uses Hermes-style XML tool definitions.
     * Other architectures use a generic JSON-based format.
     *
     * @param tools list of tool maps with "function" sub-maps containing "name", "description", "parameters"
     * @param toolNames populated with function names for later matching
     * @return system prompt text describing available tools
     */
    @SuppressWarnings("unchecked")
    public String formatToolsSystemPrompt(List<?> tools, List<String> toolNames, java.util.function.Function<Object, String> toJson) {
        if (architecture == ModelArchitecture.SMOLLM3) {
            return formatSmolLM3Tools(tools, toolNames, toJson);
        }
        return formatGenericTools(tools, toolNames, toJson);
    }

    @SuppressWarnings("unchecked")
    private String formatSmolLM3Tools(List<?> tools, List<String> toolNames, java.util.function.Function<Object, String> toJson) {
        StringBuilder sb = new StringBuilder();
        sb.append("You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. ");
        sb.append("You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. ");
        sb.append("Here are the available tools:\n<tools>\n");
        for (Object toolObj : tools) {
            Map<String, Object> tool = (Map<String, Object>) toolObj;
            Map<String, Object> function = (Map<String, Object>) tool.get("function");
            if (function == null) continue;
            String name = (String) function.get("name");
            if (name == null) continue;
            toolNames.add(name);
            sb.append("{\"type\": \"function\", \"function\": ");
            sb.append(toJson.apply(function));
            sb.append("}\n");
        }
        sb.append("</tools>\n\n");
        sb.append("For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n");
        sb.append("<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>");
        return sb.toString();
    }

    @SuppressWarnings("unchecked")
    private String formatGenericTools(List<?> tools, List<String> toolNames, java.util.function.Function<Object, String> toJson) {
        StringBuilder sb = new StringBuilder();
        sb.append("You have access to the following tools:\n\n");
        for (Object toolObj : tools) {
            Map<String, Object> tool = (Map<String, Object>) toolObj;
            Map<String, Object> function = (Map<String, Object>) tool.get("function");
            if (function == null) continue;
            String name = (String) function.get("name");
            if (name == null) continue;
            toolNames.add(name);
            sb.append("Function: ").append(name).append("\n");
            if (function.containsKey("description")) {
                sb.append("Description: ").append(function.get("description")).append("\n");
            }
            if (function.containsKey("parameters")) {
                sb.append("Parameters: ").append(toJson.apply(function.get("parameters"))).append("\n");
            }
            sb.append("\n");
        }
        sb.append("When you need to call a tool, respond ONLY with a JSON object in this exact format:\n");
        sb.append("{\"name\": \"function_name\", \"arguments\": {\"arg1\": \"value1\"}}\n");
        sb.append("Do not include any other text when making a tool call.");
        return sb.toString();
    }

    /**
     * Format a tool result message content for the model.
     * SmolLM3 wraps tool results in &lt;tool_response&gt; tags.
     */
    public String formatToolResult(String toolCallId, String content) {
        if (architecture == ModelArchitecture.SMOLLM3) {
            return "<tool_response>\n" + content + "\n</tool_response>";
        }
        String prefix = toolCallId != null ? "[Tool result for " + toolCallId + "]: " : "[Tool result]: ";
        return prefix + content;
    }

    /**
     * Format an assistant message that contains tool calls, for multi-turn conversations.
     * SmolLM3 wraps each tool call in &lt;tool_call&gt; tags.
     */
    @SuppressWarnings("unchecked")
    public String formatAssistantToolCalls(String content, List<?> toolCalls) {
        if (architecture == ModelArchitecture.SMOLLM3) {
            StringBuilder sb = new StringBuilder();
            if (content != null && !content.isEmpty()) sb.append(content);
            for (Object tcObj : toolCalls) {
                Map<String, Object> tc = (Map<String, Object>) tcObj;
                Map<String, Object> fn = (Map<String, Object>) tc.get("function");
                if (fn == null) continue;
                if (sb.length() > 0) sb.append("\n");
                sb.append("<tool_call>\n");
                sb.append("{\"name\": \"").append(fn.get("name")).append("\", \"arguments\": ");
                Object args = fn.get("arguments");
                sb.append(args != null ? args : "{}");
                sb.append("}\n</tool_call>");
            }
            return sb.toString();
        }
        // Generic format
        StringBuilder sb = new StringBuilder();
        if (content != null) sb.append(content);
        for (Object tcObj : toolCalls) {
            Map<String, Object> tc = (Map<String, Object>) tcObj;
            Map<String, Object> fn = (Map<String, Object>) tc.get("function");
            if (fn != null) {
                if (sb.length() > 0) sb.append("\n");
                sb.append("[Called tool: ").append(fn.get("name"));
                sb.append("(").append(fn.get("arguments") != null ? fn.get("arguments") : "{}");
                sb.append(")]");
            }
        }
        return sb.toString();
    }
}
