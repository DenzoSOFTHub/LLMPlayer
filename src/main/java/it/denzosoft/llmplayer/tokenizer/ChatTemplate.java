package it.denzosoft.llmplayer.tokenizer;

import it.denzosoft.llmplayer.model.ModelArchitecture;

import java.util.List;
import java.util.Map;

public class ChatTemplate {

    private final ModelArchitecture architecture;
    private final String chatTemplate;
    // GLM-4.7-Flash uses DEEPSEEK2 GGUF architecture but needs GLM4 chat format
    private final boolean isGlmVariant;
    // Olmo 3 uses olmo2 GGUF architecture but ships a ChatML-style chat template
    private final boolean isOlmo3ChatML;
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
        // Detect Olmo 3 (uses olmo2 GGUF arch but ChatML-style template with <|im_start|>)
        this.isOlmo3ChatML = architecture == ModelArchitecture.OLMO2
                && chatTemplate != null
                && chatTemplate.contains("<|im_start|>");
    }

    public void setThinkingEnabled(boolean enabled) { this.thinkingEnabled = enabled; }
    public boolean isThinkingEnabled() { return thinkingEnabled; }

    /**
     * Returns true if this model architecture supports thinking/reasoning mode.
     */
    public boolean supportsThinking() {
        return architecture == ModelArchitecture.SMOLLM3
                || architecture == ModelArchitecture.QWEN3
                || architecture == ModelArchitecture.QWEN35
                || architecture == ModelArchitecture.NEMOTRON_H
                || architecture == ModelArchitecture.NANBEIGE
                || architecture == ModelArchitecture.SPARK2_5
                || architecture == ModelArchitecture.BAILINGMOE3
                || isGlmHybridThinking();
    }

    public String formatUserMessage(String userMessage) {
        if (architecture == ModelArchitecture.LLAMA || architecture == ModelArchitecture.LLAMA4) {
            return formatLlama3(userMessage);
        } else if (architecture == ModelArchitecture.QWEN35) {
            return formatQwen35(userMessage);
        } else if (architecture == ModelArchitecture.QWEN2 || architecture == ModelArchitecture.QWEN3
                || architecture == ModelArchitecture.QWEN3MOE || architecture == ModelArchitecture.SMOLLM3
                || architecture == ModelArchitecture.NEMOTRON_H
                || architecture == ModelArchitecture.LFM2 || architecture == ModelArchitecture.FALCON_H1
                || architecture == ModelArchitecture.NANBEIGE) {
            return formatQwen(userMessage);
        } else if (architecture == ModelArchitecture.HUNYUAN_DENSE) {
            return formatHunyuan(userMessage);
        } else if (architecture == ModelArchitecture.BAILINGMOE3) {
            return formatLing(null, userMessage);
        } else if (architecture == ModelArchitecture.SPARK2_5) {
            return formatSpark(null, userMessage);
        } else if (architecture == ModelArchitecture.ERNIE4_5) {
            return formatErnie(userMessage);
        } else if (architecture == ModelArchitecture.GLM4) {
            return formatGLM4(userMessage);
        } else if (architecture == ModelArchitecture.DEEPSEEK2) {
            return isGlmVariant ? formatGLM4(userMessage) : formatDeepSeek(userMessage);
        } else if (architecture == ModelArchitecture.PHI3) {
            return formatPhi3(userMessage);
        } else if (architecture == ModelArchitecture.MISTRAL3) {
            return formatMistral3(userMessage);
        } else if ((architecture == ModelArchitecture.COMMAND_R || architecture == ModelArchitecture.COHERE2)) {
            return formatCommandR(userMessage);
        } else if (architecture == ModelArchitecture.OLMO2) {
            return formatOLMo2(userMessage);
        } else if (architecture == ModelArchitecture.GEMMA2 || architecture == ModelArchitecture.GEMMA3 || architecture == ModelArchitecture.GEMMA3N) {
            return formatGemma(userMessage);
        } else if (architecture == ModelArchitecture.GEMMA4) {
            return formatGemma4(userMessage);
        } else if (architecture == ModelArchitecture.GPT_OSS) {
            return formatGptOss(userMessage);
        } else if (architecture == ModelArchitecture.GRANITE || architecture == ModelArchitecture.GRANITE_HYBRID) {
            return formatGranite(userMessage);
        }
        return formatLlama3(userMessage); // default fallback
    }

    public String formatChat(String systemMessage, String userMessage) {
        if (architecture == ModelArchitecture.LLAMA || architecture == ModelArchitecture.LLAMA4) {
            return formatLlama3Chat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.QWEN35) {
            return formatQwen35Chat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.QWEN2 || architecture == ModelArchitecture.QWEN3
                || architecture == ModelArchitecture.QWEN3MOE || architecture == ModelArchitecture.SMOLLM3
                || architecture == ModelArchitecture.NEMOTRON_H
                || architecture == ModelArchitecture.LFM2 || architecture == ModelArchitecture.FALCON_H1
                || architecture == ModelArchitecture.NANBEIGE) {
            return formatQwenChat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.HUNYUAN_DENSE) {
            return formatHunyuanChat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.BAILINGMOE3) {
            return formatLing(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.SPARK2_5) {
            return formatSpark(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.ERNIE4_5) {
            return formatErnieChat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.GLM4) {
            return formatGLM4Chat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.DEEPSEEK2) {
            return isGlmVariant ? formatGLM4Chat(systemMessage, userMessage) : formatDeepSeekChat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.PHI3) {
            return formatPhi3Chat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.MISTRAL3) {
            return formatMistral3Chat(systemMessage, userMessage);
        } else if ((architecture == ModelArchitecture.COMMAND_R || architecture == ModelArchitecture.COHERE2)) {
            return formatCommandRChat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.OLMO2) {
            return formatOLMo2Chat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.GEMMA2 || architecture == ModelArchitecture.GEMMA3 || architecture == ModelArchitecture.GEMMA3N) {
            return formatGemmaChat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.GEMMA4) {
            return formatGemma4Chat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.GPT_OSS) {
            return formatGptOssChat(systemMessage, userMessage);
        } else if (architecture == ModelArchitecture.GRANITE || architecture == ModelArchitecture.GRANITE_HYBRID) {
            return formatGraniteChat(systemMessage, userMessage);
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

    // ERNIE 4.5: <|begin_of_sentence|> (cls) + "User: ...\nAssistant: ..."<|end_of_sentence|> (sep).
    // Assistant turn ends with <|end_of_sentence|>; generation prompt is "Assistant: ".
    private String formatErnie(String userMessage) {
        return "<|begin_of_sentence|>User: " + userMessage + "\nAssistant: ";
    }

    private String formatErnieChat(String systemMessage, String userMessage) {
        return "<|begin_of_sentence|>" + systemMessage + "\nUser: " + userMessage + "\nAssistant: ";
    }

    // Hunyuan dense (Hy-MT2, Hunyuan-*-Instruct). The engine prepends BOS
    // (<｜hy_begin▁of▁sentence｜>); a system prompt is closed by <｜hy_place▁holder▁no▁3｜> and an
    // assistant turn by <｜hy_place▁holder▁no▁2｜> (the EOS).
    private static final String HY_USER = "<\uFF5Chy_User\uFF5C>";
    private static final String HY_ASSISTANT = "<\uFF5Chy_Assistant\uFF5C>";
    private static final String HY_SYS_END = "<\uFF5Chy_place\u2581holder\u2581no\u25813\uFF5C>";
    private static final String HY_TURN_END = "<\uFF5Chy_place\u2581holder\u2581no\u25812\uFF5C>";

    private String formatHunyuan(String userMessage) {
        return HY_USER + userMessage + HY_ASSISTANT;
    }

    private String formatHunyuanChat(String systemMessage, String userMessage) {
        return systemMessage + HY_SYS_END + HY_USER + userMessage + HY_ASSISTANT;
    }

    private String formatHunyuanConversation(List<String[]> messages) {
        StringBuilder sb = new StringBuilder();
        for (String[] msg : messages) {
            if ("system".equals(msg[0])) {
                sb.append(msg[1]).append(HY_SYS_END);
            } else if ("user".equals(msg[0])) {
                sb.append(HY_USER).append(msg[1]);
            } else if ("assistant".equals(msg[0])) {
                sb.append(HY_ASSISTANT).append(msg[1]).append(HY_TURN_END);
            }
        }
        sb.append(HY_ASSISTANT);
        return sb.toString();
    }

    // Ling 3.0 (BailingMoE3): <role>SYSTEM</role>...detailed thinking on|off<|role_end|>, then
    // <role>HUMAN</role>...<|role_end|> and <role>ASSISTANT</role>; the generation prompt opens
    // <think> when thinking is on and emits an empty <think></think> otherwise.
    private String formatLing(String systemMessage, String userMessage) {
        java.util.List<String[]> messages = new java.util.ArrayList<>();
        if (systemMessage != null) messages.add(new String[] {"system", systemMessage});
        messages.add(new String[] {"user", userMessage});
        return formatLingConversation(messages);
    }

    private String formatLingConversation(List<String[]> messages) {
        String thinking = "detailed thinking " + (thinkingEnabled ? "on" : "off");
        StringBuilder sb = new StringBuilder("<role>SYSTEM</role>");
        int start = 0;
        if (!messages.isEmpty() && "system".equals(messages.get(0)[0])) {
            String sys = messages.get(0)[1];
            if (sys.contains("detailed thinking on") || sys.contains("detailed thinking off")) {
                sb.append(sys);
            } else {
                sb.append(sys).append('\n').append(thinking);
            }
            start = 1;
        } else {
            sb.append(thinking);
        }
        sb.append("<|role_end|>");
        for (int i = start; i < messages.size(); i++) {
            String[] msg = messages.get(i);
            if ("user".equals(msg[0])) {
                sb.append("<role>HUMAN</role>").append(msg[1]).append("<|role_end|>");
            } else if ("assistant".equals(msg[0])) {
                sb.append("<role>ASSISTANT</role>\n<think></think>").append(msg[1]).append("<|role_end|>");
            } else if ("system".equals(msg[0])) {
                sb.append("<role>SYSTEM</role>").append(msg[1]).append("<|role_end|>");
            }
        }
        sb.append("<role>ASSISTANT</role>").append(thinkingEnabled ? "\n<think>" : "\n<think></think>");
        return sb.toString();
    }

    // Spark2.5: every turn is <｜start▁of▁sentence｜><|Role|>...<｜end▁of▁sentence｜>. The system
    // block always starts with the default prompt, and a user system message is appended to it.
    // The generation prompt opens <think> when thinking is on and closes it (</think>) otherwise.
    private static final String SPARK_BOS = "<\uFF5Cstart\u2581of\u2581sentence\uFF5C>";
    private static final String SPARK_EOS = "<\uFF5Cend\u2581of\u2581sentence\uFF5C>";

    private String formatSpark(String systemMessage, String userMessage) {
        java.util.List<String[]> messages = new java.util.ArrayList<>();
        if (systemMessage != null) messages.add(new String[] {"system", systemMessage});
        messages.add(new String[] {"user", userMessage});
        return formatSparkConversation(messages);
    }

    private String formatSparkConversation(List<String[]> messages) {
        StringBuilder sb = new StringBuilder();
        sb.append(SPARK_BOS).append("<|System|>\nyou are a helpful assistant.");
        int start = 0;
        if (!messages.isEmpty() && "system".equals(messages.get(0)[0])) {
            sb.append("\n\n").append(messages.get(0)[1]);
            start = 1;
        }
        sb.append(SPARK_EOS);
        for (int i = start; i < messages.size(); i++) {
            String[] msg = messages.get(i);
            if ("system".equals(msg[0])) {
                sb.append(SPARK_BOS).append("<|System|>\n").append(msg[1]).append(SPARK_EOS);
            } else if ("user".equals(msg[0])) {
                sb.append(SPARK_BOS).append("<|User|>").append(msg[1]).append(SPARK_EOS);
            } else if ("assistant".equals(msg[0])) {
                sb.append(SPARK_BOS).append("<|Bot|></think>").append(msg[1]).append(SPARK_EOS);
            }
        }
        sb.append(SPARK_BOS).append("<|Bot|>").append(thinkingEnabled ? "<think>" : "</think>");
        return sb.toString();
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
        // Qwen3: thinking is natural, suppress only when explicitly disabled. Instruct-only
        // checkpoints (Qwen3-2507-Instruct, Qwen3-VL-Instruct) have no <think> in their template
        // and get no suffix.
        if (architecture == ModelArchitecture.QWEN3) {
            if (chatTemplate != null && !chatTemplate.isEmpty() && !chatTemplate.contains("<think>")) return "";
            return thinkingEnabled ? "" : "<think>\n\n</think>\n\n";
        }
        // Nanbeige 4.2: the template opens a <think> block itself when thinking is on, and emits an
        // empty one when enable_thinking=false.
        if (architecture == ModelArchitecture.NANBEIGE) {
            return thinkingEnabled ? "<think>\n" : "<think>\n\n</think>\n\n";
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
        if (architecture == ModelArchitecture.NANBEIGE) {
            return formatQwenChat(NANBEIGE_DEFAULT_SYSTEM, userMessage);
        }
        return "<|im_start|>user\n" + userMessage + "<|im_end|>\n<|im_start|>assistant\n" + thinkingSuffix();
    }

    // Nanbeige 4.2's template inserts this system turn when the conversation has none.
    private static final String NANBEIGE_DEFAULT_SYSTEM = "你是南北阁，一款由BOSS直聘自主研发并训练的专业大语言模型。";

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
        return "[gMASK]<sop><|user|>\n" + glmUser(userMessage) + glmGenerationPrompt();
    }

    private String formatGLM4Chat(String systemMessage, String userMessage) {
        return "[gMASK]<sop><|system|>\n" + systemMessage +
               "<|user|>\n" + glmUser(userMessage) + glmGenerationPrompt();
    }

    /**
     * GLM-4.5 and later hybrid-reasoning templates (GLM4-MoE): thinking is on unless
     * enable_thinking=false, which appends "/nothink" to each user turn and an empty
     * "\n<think></think>" to the generation prompt. Other GLM4 templates are left as they were.
     */
    private boolean isGlmHybridThinking() {
        return architecture == ModelArchitecture.GLM4 && chatTemplate != null
            && chatTemplate.contains("enable_thinking") && chatTemplate.contains("<think></think>");
    }

    private String glmUser(String content) {
        if (isGlmHybridThinking() && !thinkingEnabled && !content.endsWith("/nothink")) return content + "/nothink";
        return content;
    }

    private String glmGenerationPrompt() {
        if (isGlmHybridThinking()) return thinkingEnabled ? "<|assistant|>" : "<|assistant|>\n<think></think>";
        return "<|assistant|>\n";
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
                || architecture == ModelArchitecture.QWEN3MOE || architecture == ModelArchitecture.SMOLLM3
                || architecture == ModelArchitecture.NEMOTRON_H
                || architecture == ModelArchitecture.LFM2 || architecture == ModelArchitecture.FALCON_H1
                || architecture == ModelArchitecture.NANBEIGE) {
            return formatQwenConversation(messages);
        } else if (architecture == ModelArchitecture.HUNYUAN_DENSE) {
            return formatHunyuanConversation(messages);
        } else if (architecture == ModelArchitecture.BAILINGMOE3) {
            return formatLingConversation(messages);
        } else if (architecture == ModelArchitecture.SPARK2_5) {
            return formatSparkConversation(messages);
        } else if (architecture == ModelArchitecture.ERNIE4_5) {
            return formatErnieConversation(messages);
        } else if (architecture == ModelArchitecture.GLM4) {
            return formatGLM4Conversation(messages);
        } else if (architecture == ModelArchitecture.DEEPSEEK2) {
            return isGlmVariant ? formatGLM4Conversation(messages) : formatDeepSeekConversation(messages);
        } else if (architecture == ModelArchitecture.PHI3) {
            return formatPhi3Conversation(messages);
        } else if (architecture == ModelArchitecture.MISTRAL3) {
            return formatMistral3Conversation(messages);
        } else if ((architecture == ModelArchitecture.COMMAND_R || architecture == ModelArchitecture.COHERE2)) {
            return formatCommandRConversation(messages);
        } else if (architecture == ModelArchitecture.OLMO2) {
            return formatOLMo2Conversation(messages);
        } else if (architecture == ModelArchitecture.GEMMA2 || architecture == ModelArchitecture.GEMMA3 || architecture == ModelArchitecture.GEMMA3N) {
            return formatGemmaConversation(messages);
        } else if (architecture == ModelArchitecture.GEMMA4) {
            return formatGemma4Conversation(messages);
        } else if (architecture == ModelArchitecture.GPT_OSS) {
            return formatGptOssConversation(messages);
        } else if (architecture == ModelArchitecture.GRANITE || architecture == ModelArchitecture.GRANITE_HYBRID) {
            return formatGraniteConversation(messages);
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
        if (architecture == ModelArchitecture.NANBEIGE
                && (messages.isEmpty() || !"system".equals(messages.get(0)[0]))) {
            sb.append("<|im_start|>system\n").append(NANBEIGE_DEFAULT_SYSTEM).append("<|im_end|>\n");
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

    private String formatErnieConversation(List<String[]> messages) {
        StringBuilder sb = new StringBuilder("<|begin_of_sentence|>");
        for (String[] msg : messages) {
            if ("user".equals(msg[0])) {
                sb.append("User: ").append(msg[1]).append("\n");
            } else if ("assistant".equals(msg[0])) {
                sb.append("Assistant: ").append(msg[1]).append("<|end_of_sentence|>");
            } else if ("system".equals(msg[0])) {
                sb.append(msg[1]).append("\n");
            }
        }
        sb.append("Assistant: ");
        return sb.toString();
    }

    private String formatGLM4Conversation(List<String[]> messages) {
        StringBuilder sb = new StringBuilder("[gMASK]<sop>");
        boolean hybrid = isGlmHybridThinking();
        for (String[] msg : messages) {
            if (hybrid && "assistant".equals(msg[0])) {
                // Past turns carry an empty think block and the visible answer only
                String content = msg[1];
                int end = content.indexOf("</think>");
                if (end >= 0) content = content.substring(end + "</think>".length());
                sb.append("<|assistant|>\n<think></think>");
                if (!content.trim().isEmpty()) sb.append('\n').append(content.trim());
                continue;
            }
            sb.append("<|").append(msg[0]).append("|>\n");
            sb.append("user".equals(msg[0]) ? glmUser(msg[1]) : msg[1]);
        }
        sb.append(glmGenerationPrompt());
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

    // OLMo2 format (Olmo 3 uses ChatML — see isOlmo3ChatML)
    private String formatOLMo2(String userMessage) {
        if (isOlmo3ChatML) {
            // Olmo 3's Jinja template injects a default system prompt when none is given.
            // We replicate it verbatim so the model sees the same context it was trained on.
            return "<|im_start|>system\nYou are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions><|im_end|>\n"
                    + "<|im_start|>user\n" + userMessage + "<|im_end|>\n"
                    + "<|im_start|>assistant\n";
        }
        return "<|user|>\n" + userMessage + "\n<|assistant|>\n";
    }

    private String formatOLMo2Chat(String systemMessage, String userMessage) {
        if (isOlmo3ChatML) {
            return "<|im_start|>system\n" + systemMessage + "<|im_end|>\n<|im_start|>user\n"
                    + userMessage + "<|im_end|>\n<|im_start|>assistant\n";
        }
        return "<|system|>\n" + systemMessage + "\n<|user|>\n" + userMessage + "\n<|assistant|>\n";
    }

    private String formatOLMo2Conversation(List<String[]> messages) {
        StringBuilder sb = new StringBuilder();
        if (isOlmo3ChatML) {
            for (String[] msg : messages) {
                sb.append("<|im_start|>").append(msg[0]).append("\n");
                sb.append(msg[1]).append("<|im_end|>\n");
            }
            sb.append("<|im_start|>assistant\n");
            return sb.toString();
        }
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

    // Granite 3.x multi-turn conversation
    private String formatGraniteConversation(List<String[]> messages) {
        StringBuilder sb = new StringBuilder();
        for (String[] msg : messages) {
            String role = msg[0];
            String content = msg[1];
            sb.append("<|start_of_role|>").append(role).append("<|end_of_role|>");
            sb.append(content);
            if (!"assistant".equals(role) || messages.indexOf(msg) < messages.size() - 1) {
                sb.append("<|end_of_text|>\n");
            }
        }
        // If last message is not assistant, add assistant header
        if (messages.isEmpty() || !"assistant".equals(messages.get(messages.size() - 1)[0])) {
            sb.append("<|start_of_role|>assistant<|end_of_role|>\n");
        }
        return sb.toString();
    }

    // Granite 3.x: <|start_of_role|>user<|end_of_role|>...<|end_of_text|>
    private String formatGranite(String userMessage) {
        return "<|start_of_role|>user<|end_of_role|>" + userMessage + "<|end_of_text|>\n"
             + "<|start_of_role|>assistant<|end_of_role|>\n";
    }

    private String formatGraniteChat(String systemMessage, String userMessage) {
        StringBuilder sb = new StringBuilder();
        if (systemMessage != null && !systemMessage.isEmpty()) {
            sb.append("<|start_of_role|>system<|end_of_role|>").append(systemMessage).append("<|end_of_text|>\n");
        }
        sb.append("<|start_of_role|>user<|end_of_role|>").append(userMessage).append("<|end_of_text|>\n");
        sb.append("<|start_of_role|>assistant<|end_of_role|>\n");
        return sb.toString();
    }

    // Gemma 4 format: <|turn>role\nmessage<turn|>
    private String formatGemma4(String userMessage) {
        return "<|turn>user\n" + userMessage + "<turn|>\n<|turn>model\n";
    }

    private String formatGemma4Chat(String systemMessage, String userMessage) {
        StringBuilder sb = new StringBuilder();
        if (systemMessage != null && !systemMessage.isEmpty()) {
            sb.append("<|turn>system\n").append(systemMessage).append("<turn|>\n");
        }
        sb.append("<|turn>user\n").append(userMessage).append("<turn|>\n");
        sb.append("<|turn>model\n");
        return sb.toString();
    }

    private String formatGemma4Conversation(List<String[]> messages) {
        StringBuilder sb = new StringBuilder();
        for (String[] msg : messages) {
            String role = "assistant".equals(msg[0]) ? "model" : msg[0];
            sb.append("<|turn>").append(role).append("\n");
            sb.append(msg[1]).append("<turn|>\n");
        }
        if (!messages.isEmpty() && !"assistant".equals(messages.get(messages.size() - 1)[0])) {
            sb.append("<|turn>model\n");
        }
        return sb.toString();
    }
}
