package it.denzosoft.llmplayer.web;

import com.sun.net.httpserver.HttpExchange;
import it.denzosoft.llmplayer.api.*;
import it.denzosoft.llmplayer.sampler.SamplerConfig;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * OpenAI-compatible API handler.
 * Implements POST /v1/chat/completions and GET /v1/models.
 */
public class OpenAIHandler {

    private final ApiHandler apiHandler;

    public OpenAIHandler(ApiHandler apiHandler) {
        this.apiHandler = apiHandler;
    }

    public void handle(HttpExchange exchange) throws IOException {
        String path = exchange.getRequestURI().getPath();
        String method = exchange.getRequestMethod();

        // CORS headers
        exchange.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        exchange.getResponseHeaders().set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        exchange.getResponseHeaders().set("Access-Control-Allow-Headers", "Content-Type, Authorization");

        if ("OPTIONS".equals(method)) {
            exchange.sendResponseHeaders(204, -1);
            exchange.close();
            return;
        }

        try {
            if ("/v1/chat/completions".equals(path)) {
                if ("POST".equals(method)) handleChatCompletions(exchange);
                else sendError(exchange, 405, "Method not allowed");
            } else if ("/v1/models".equals(path)) {
                if ("GET".equals(method)) handleModels(exchange);
                else sendError(exchange, 405, "Method not allowed");
            } else if ("/v1/embeddings".equals(path)) {
                if ("POST".equals(method)) handleEmbeddings(exchange);
                else sendError(exchange, 405, "Method not allowed");
            } else {
                sendError(exchange, 404, "Not found");
            }
        } catch (Exception e) {
            System.err.println("OpenAI API error: " + e.getMessage());
            e.printStackTrace();
            try {
                sendError(exchange, 500, e.getMessage() != null ? e.getMessage() : "Internal error");
            } catch (IOException ignored) {}
        }
    }

    @SuppressWarnings("unchecked")
    private void handleChatCompletions(HttpExchange exchange) throws IOException {
        LLMEngine engine = apiHandler.getEngine();
        if (engine == null) {
            sendError(exchange, 503, "No model loaded. Load a model first.");
            return;
        }
        if (apiHandler.isGenerating()) {
            sendError(exchange, 429, "Generation already in progress");
            return;
        }

        Map<String, Object> body = ApiHandler.parseJson(readBody(exchange));

        // Parse messages
        List<?> messagesRaw = (List<?>) body.get("messages");
        if (messagesRaw == null || messagesRaw.isEmpty()) {
            sendError(exchange, 400, "messages is required and must be non-empty");
            return;
        }

        List<String[]> messages = new ArrayList<>();
        for (Object msgObj : messagesRaw) {
            Map<String, Object> msg = (Map<String, Object>) msgObj;
            String role = (String) msg.get("role");
            if (role == null) continue;

            if ("tool".equals(role)) {
                // Tool result message: convert to user message
                Object contentObj = msg.get("content");
                String content = contentObj != null ? contentObj.toString() : "";
                String toolCallId = (String) msg.get("tool_call_id");
                String prefix = toolCallId != null ? "[Tool result for " + toolCallId + "]: " : "[Tool result]: ";
                messages.add(new String[]{"user", prefix + content});
                continue;
            }

            Object contentObj = msg.get("content");
            String content = contentObj instanceof String ? (String) contentObj : null;

            // Handle assistant messages with tool_calls
            if ("assistant".equals(role) && msg.containsKey("tool_calls")) {
                StringBuilder sb = new StringBuilder();
                if (content != null) sb.append(content);
                List<?> toolCalls = (List<?>) msg.get("tool_calls");
                if (toolCalls != null) {
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
                }
                messages.add(new String[]{"assistant", sb.toString()});
                continue;
            }

            if (content == null) continue;
            messages.add(new String[]{role, content});
        }

        if (messages.isEmpty()) {
            sendError(exchange, 400, "messages must contain at least one message with role and content");
            return;
        }

        // Parse tools for function calling
        List<?> tools = (List<?>) body.get("tools");
        String toolChoice = null;
        if (body.containsKey("tool_choice")) {
            Object tc = body.get("tool_choice");
            if (tc instanceof String) {
                toolChoice = (String) tc;
            } else if (tc instanceof Map) {
                // Object format: {"type":"function","function":{"name":"..."}} → treat as "auto"
                toolChoice = "auto";
            }
        }
        boolean toolsActive = tools != null && !tools.isEmpty() && !"none".equals(toolChoice);

        // Parse response_format for JSON mode
        Map<String, Object> responseFormat = null;
        if (body.containsKey("response_format")) {
            responseFormat = (Map<String, Object>) body.get("response_format");
        }

        // Build system prompt injections for tools and/or JSON mode
        StringBuilder systemInjection = new StringBuilder();
        List<String> toolNames = new ArrayList<>();
        if (toolsActive) {
            systemInjection.append(formatToolsSystemPrompt(tools, toolNames));
        }
        if (responseFormat != null) {
            String formatType = (String) responseFormat.get("type");
            if ("json_object".equals(formatType)) {
                if (systemInjection.length() > 0) systemInjection.append("\n\n");
                systemInjection.append("You must respond with valid JSON only. Do not include any text outside the JSON object.");
            } else if ("json_schema".equals(formatType)) {
                Map<String, Object> jsonSchema = (Map<String, Object>) responseFormat.get("json_schema");
                if (jsonSchema != null && jsonSchema.containsKey("schema")) {
                    if (systemInjection.length() > 0) systemInjection.append("\n\n");
                    systemInjection.append("You must respond with valid JSON matching this schema:\n");
                    systemInjection.append(ApiHandler.toJson(jsonSchema.get("schema")));
                }
            }
        }

        // Inject system additions into messages
        if (systemInjection.length() > 0) {
            String injection = systemInjection.toString();
            boolean hasSystem = false;
            for (int i = 0; i < messages.size(); i++) {
                if ("system".equals(messages.get(i)[0])) {
                    messages.set(i, new String[]{"system", messages.get(i)[1] + "\n\n" + injection});
                    hasSystem = true;
                    break;
                }
            }
            if (!hasSystem) {
                messages.add(0, new String[]{"system", injection});
            }
        }

        // Parse parameters
        boolean stream = body.containsKey("stream") && Boolean.TRUE.equals(body.get("stream"));
        float temperature = body.containsKey("temperature")
            ? ((Number) body.get("temperature")).floatValue() : 0.7f;
        int maxTokens = 256;
        if (body.containsKey("max_tokens")) {
            maxTokens = ((Number) body.get("max_tokens")).intValue();
        } else if (body.containsKey("max_completion_tokens")) {
            maxTokens = ((Number) body.get("max_completion_tokens")).intValue();
        }
        float topP = body.containsKey("top_p")
            ? ((Number) body.get("top_p")).floatValue() : 0.9f;
        int topK = body.containsKey("top_k")
            ? ((Number) body.get("top_k")).intValue() : 40;

        // Repetition penalty: accept custom repetition_penalty or map from frequency_penalty
        float repPenalty = 1.1f;
        if (body.containsKey("repetition_penalty")) {
            repPenalty = ((Number) body.get("repetition_penalty")).floatValue();
        } else if (body.containsKey("frequency_penalty")) {
            float fp = ((Number) body.get("frequency_penalty")).floatValue();
            repPenalty = 1.0f + fp * 0.5f;
        } else if (body.containsKey("presence_penalty")) {
            float pp = ((Number) body.get("presence_penalty")).floatValue();
            repPenalty = 1.0f + pp * 0.5f;
        }

        // Parse stop sequences
        List<String> stopSequences = new ArrayList<>();
        if (body.containsKey("stop")) {
            Object stopObj = body.get("stop");
            if (stopObj instanceof String) {
                stopSequences.add((String) stopObj);
            } else if (stopObj instanceof List) {
                for (Object s : (List<?>) stopObj) {
                    if (s instanceof String) stopSequences.add((String) s);
                }
            }
        }

        // Build the formatted prompt from messages using chat template
        String formattedPrompt = engine.getChatTemplate().formatConversation(messages);

        // Compute conversation cache key from messages (hash of all messages)
        String cacheKey = computeCacheKey(messages);

        SamplerConfig samplerConfig = new SamplerConfig(temperature, topK, topP, repPenalty, System.nanoTime());
        GenerationRequest request = GenerationRequest.builder()
            .prompt(formattedPrompt)
            .maxTokens(maxTokens)
            .samplerConfig(samplerConfig)
            .useChat(false)
            .rawMode(true)
            .cacheKey(cacheKey)
            .build();

        String modelName = engine.getModelName();
        String completionId = "chatcmpl-" + UUID.randomUUID().toString().replace("-", "").substring(0, 29);
        long created = System.currentTimeMillis() / 1000;

        if (stream) {
            handleStreamingChat(exchange, engine, request, completionId, modelName, created, stopSequences, toolsActive ? toolNames : null);
        } else {
            handleNonStreamingChat(exchange, engine, request, completionId, modelName, created, stopSequences, toolsActive ? toolNames : null);
        }
    }

    private void handleStreamingChat(HttpExchange exchange, LLMEngine engine,
                                      GenerationRequest request, String completionId,
                                      String modelName, long created,
                                      final List<String> stopSequences,
                                      final List<String> toolNames) throws IOException {
        exchange.getResponseHeaders().set("Content-Type", "text/event-stream");
        exchange.getResponseHeaders().set("Cache-Control", "no-cache");
        exchange.getResponseHeaders().set("Connection", "keep-alive");
        exchange.sendResponseHeaders(200, 0);

        final OutputStream out = exchange.getResponseBody();
        apiHandler.setStopRequested(false);
        apiHandler.setGenerating(true);

        try {
            // Send initial chunk with role
            Map<String, Object> roleDelta = new LinkedHashMap<>();
            roleDelta.put("role", "assistant");
            roleDelta.put("content", "");
            writeSSE(out, buildStreamChunk(completionId, modelName, created, roleDelta, null));

            final StringBuilder accumulated = new StringBuilder();
            final int[] tokenCount = {0};

            GenerationResponse response = engine.generate(request, new StreamingCallback() {
                @Override
                public boolean onToken(String token, int tokenId) {
                    if (apiHandler.isStopRequested()) return false;

                    accumulated.append(token);
                    tokenCount[0]++;

                    // Check stop sequences
                    String text = accumulated.toString();
                    for (String seq : stopSequences) {
                        if (text.contains(seq)) {
                            return false;
                        }
                    }

                    try {
                        Map<String, Object> delta = new LinkedHashMap<>();
                        delta.put("content", token);
                        writeSSE(out, buildStreamChunk(completionId, modelName, created, delta, null));
                    } catch (IOException e) {
                        return false;
                    }
                    return true;
                }
            });

            // Determine finish_reason
            String finishReason = (response.tokenCount() >= request.maxTokens()) ? "length" : "stop";
            if (toolNames != null && tryParseToolCall(accumulated.toString().trim(), toolNames) != null) {
                finishReason = "tool_calls";
            }

            // Send final chunk with finish_reason
            Map<String, Object> emptyDelta = new LinkedHashMap<>();
            writeSSE(out, buildStreamChunk(completionId, modelName, created, emptyDelta, finishReason));

            // Send usage chunk
            Map<String, Object> usageChunk = new LinkedHashMap<>();
            usageChunk.put("id", completionId);
            usageChunk.put("object", "chat.completion.chunk");
            usageChunk.put("created", created);
            usageChunk.put("model", modelName);
            usageChunk.put("choices", Collections.emptyList());
            Map<String, Object> usage = new LinkedHashMap<>();
            usage.put("prompt_tokens", response.promptTokenCount());
            usage.put("completion_tokens", response.tokenCount());
            usage.put("total_tokens", response.promptTokenCount() + response.tokenCount());
            usageChunk.put("usage", usage);
            writeSSE(out, ApiHandler.toJson(usageChunk));

            // Send [DONE]
            out.write("data: [DONE]\n\n".getBytes(StandardCharsets.UTF_8));
            out.flush();
        } catch (Exception e) {
            try {
                Map<String, Object> errChunk = new LinkedHashMap<>();
                Map<String, Object> errBody = new LinkedHashMap<>();
                errBody.put("message", e.getMessage() != null ? e.getMessage() : "Generation error");
                errChunk.put("error", errBody);
                writeSSE(out, ApiHandler.toJson(errChunk));
                out.write("data: [DONE]\n\n".getBytes(StandardCharsets.UTF_8));
                out.flush();
            } catch (IOException ignored) {}
        } finally {
            apiHandler.setGenerating(false);
            out.close();
        }
    }

    private void handleNonStreamingChat(HttpExchange exchange, LLMEngine engine,
                                         GenerationRequest request, String completionId,
                                         String modelName, long created,
                                         List<String> stopSequences,
                                         List<String> toolNames) throws IOException {
        apiHandler.setStopRequested(false);
        apiHandler.setGenerating(true);

        try {
            GenerationResponse response = engine.generate(request);

            String text = response.text();
            String finishReason = (response.tokenCount() >= request.maxTokens()) ? "length" : "stop";

            // Check stop sequences
            for (String seq : stopSequences) {
                int idx = text.indexOf(seq);
                if (idx >= 0) {
                    text = text.substring(0, idx);
                    finishReason = "stop";
                    break;
                }
            }

            // Check for tool calls in the output
            Map<String, Object> toolCall = (toolNames != null) ? tryParseToolCall(text.trim(), toolNames) : null;

            // Build response
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("id", completionId);
            result.put("object", "chat.completion");
            result.put("created", created);
            result.put("model", modelName);

            Map<String, Object> message = new LinkedHashMap<>();
            message.put("role", "assistant");

            if (toolCall != null) {
                // Tool call response: content is null, tool_calls array present
                message.put("content", null);
                String callId = "call_" + UUID.randomUUID().toString().replace("-", "").substring(0, 24);
                Map<String, Object> tc = new LinkedHashMap<>();
                tc.put("id", callId);
                tc.put("type", "function");
                Map<String, Object> fn = new LinkedHashMap<>();
                fn.put("name", toolCall.get("name"));
                Object args = toolCall.get("arguments");
                fn.put("arguments", args instanceof String ? args : ApiHandler.toJson(args));
                tc.put("function", fn);
                List<Object> toolCalls = new ArrayList<>();
                toolCalls.add(tc);
                message.put("tool_calls", toolCalls);
                finishReason = "tool_calls";
            } else {
                message.put("content", text);
            }

            Map<String, Object> choice = new LinkedHashMap<>();
            choice.put("index", 0);
            choice.put("message", message);
            choice.put("finish_reason", finishReason);

            List<Object> choices = new ArrayList<>();
            choices.add(choice);
            result.put("choices", choices);

            Map<String, Object> usage = new LinkedHashMap<>();
            usage.put("prompt_tokens", response.promptTokenCount());
            usage.put("completion_tokens", response.tokenCount());
            usage.put("total_tokens", response.promptTokenCount() + response.tokenCount());
            result.put("usage", usage);

            sendJson(exchange, 200, result);
        } finally {
            apiHandler.setGenerating(false);
        }
    }

    private void handleModels(HttpExchange exchange) throws IOException {
        LLMEngine engine = apiHandler.getEngine();
        List<Object> data = new ArrayList<>();

        if (engine != null) {
            Map<String, Object> model = new LinkedHashMap<>();
            model.put("id", engine.getModelName());
            model.put("object", "model");
            model.put("created", System.currentTimeMillis() / 1000);
            model.put("owned_by", "local");
            data.add(model);
        } else {
            // List available GGUF files
            Path dir = Paths.get(apiHandler.getGgufDirectory());
            if (Files.isDirectory(dir)) {
                try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir, "*.gguf")) {
                    for (Path file : stream) {
                        Map<String, Object> model = new LinkedHashMap<>();
                        model.put("id", file.getFileName().toString());
                        model.put("object", "model");
                        try {
                            model.put("created", Files.getLastModifiedTime(file).toMillis() / 1000);
                        } catch (IOException e) {
                            model.put("created", 0);
                        }
                        model.put("owned_by", "local");
                        data.add(model);
                    }
                }
            }
        }

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("object", "list");
        result.put("data", data);
        sendJson(exchange, 200, result);
    }

    // --- Embeddings ---

    @SuppressWarnings("unchecked")
    private void handleEmbeddings(HttpExchange exchange) throws IOException {
        LLMEngine engine = apiHandler.getEngine();
        if (engine == null) {
            sendError(exchange, 503, "No model loaded. Load a model first.");
            return;
        }

        Map<String, Object> body = ApiHandler.parseJson(readBody(exchange));

        // Parse input: can be a string or array of strings
        Object inputObj = body.get("input");
        if (inputObj == null) {
            sendError(exchange, 400, "input is required");
            return;
        }

        List<String> inputs = new ArrayList<>();
        if (inputObj instanceof String) {
            inputs.add((String) inputObj);
        } else if (inputObj instanceof List) {
            for (Object item : (List<?>) inputObj) {
                if (item instanceof String) inputs.add((String) item);
            }
        }

        if (inputs.isEmpty()) {
            sendError(exchange, 400, "input must be a non-empty string or array of strings");
            return;
        }

        String modelName = engine.getModelName();
        List<Object> data = new ArrayList<>();
        int totalTokens = 0;

        for (int i = 0; i < inputs.size(); i++) {
            float[] embedding = engine.embed(inputs.get(i));
            totalTokens += engine.getTokenizer().encode(inputs.get(i)).length;

            List<Object> embeddingList = new ArrayList<>();
            for (float v : embedding) {
                embeddingList.add((double) v);
            }

            Map<String, Object> item = new LinkedHashMap<>();
            item.put("object", "embedding");
            item.put("embedding", embeddingList);
            item.put("index", i);
            data.add(item);
        }

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("object", "list");
        result.put("data", data);
        result.put("model", modelName);
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("prompt_tokens", totalTokens);
        usage.put("total_tokens", totalTokens);
        result.put("usage", usage);

        sendJson(exchange, 200, result);
    }

    // --- Tool Calling Helpers ---

    /**
     * Format tool definitions into a system prompt injection.
     * Populates toolNames with the function names for later matching.
     */
    @SuppressWarnings("unchecked")
    private String formatToolsSystemPrompt(List<?> tools, List<String> toolNames) {
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
                sb.append("Parameters: ").append(ApiHandler.toJson(function.get("parameters"))).append("\n");
            }
            sb.append("\n");
        }
        sb.append("When you need to call a tool, respond ONLY with a JSON object in this exact format:\n");
        sb.append("{\"name\": \"function_name\", \"arguments\": {\"arg1\": \"value1\"}}\n");
        sb.append("Do not include any other text when making a tool call.");
        return sb.toString();
    }

    /**
     * Try to parse the model output as a tool call JSON.
     * Returns a map with "name" and "arguments" if valid, null otherwise.
     */
    @SuppressWarnings("unchecked")
    private Map<String, Object> tryParseToolCall(String text, List<String> toolNames) {
        if (text.isEmpty()) return null;
        // Find JSON object boundaries
        int start = text.indexOf('{');
        int end = text.lastIndexOf('}');
        if (start < 0 || end <= start) return null;

        String jsonCandidate = text.substring(start, end + 1);
        try {
            Map<String, Object> parsed = ApiHandler.parseJson(jsonCandidate);
            String name = (String) parsed.get("name");
            if (name != null && toolNames.contains(name)) {
                Map<String, Object> result = new LinkedHashMap<>();
                result.put("name", name);
                Object args = parsed.get("arguments");
                result.put("arguments", args != null ? args : Collections.emptyMap());
                return result;
            }
        } catch (Exception ignored) {
            // Not valid JSON or doesn't match tool pattern
        }
        return null;
    }

    // --- Helpers ---

    /**
     * Compute a cache key from the conversation messages.
     * Uses a hash of all message roles and contents to identify the conversation state.
     */
    private static String computeCacheKey(List<String[]> messages) {
        StringBuilder sb = new StringBuilder();
        for (String[] msg : messages) {
            sb.append(msg[0]).append(':').append(msg[1]).append('\n');
        }
        return Integer.toHexString(sb.toString().hashCode());
    }

    private String buildStreamChunk(String id, String model, long created,
                                     Map<String, Object> delta, String finishReason) {
        Map<String, Object> chunk = new LinkedHashMap<>();
        chunk.put("id", id);
        chunk.put("object", "chat.completion.chunk");
        chunk.put("created", created);
        chunk.put("model", model);

        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        choice.put("delta", delta);
        choice.put("finish_reason", finishReason);

        List<Object> choices = new ArrayList<>();
        choices.add(choice);
        chunk.put("choices", choices);

        return ApiHandler.toJson(chunk);
    }

    private static void writeSSE(OutputStream out, String json) throws IOException {
        out.write(("data: " + json + "\n\n").getBytes(StandardCharsets.UTF_8));
        out.flush();
    }

    private void sendError(HttpExchange exchange, int status, String message) throws IOException {
        Map<String, Object> error = new LinkedHashMap<>();
        Map<String, Object> errorBody = new LinkedHashMap<>();
        errorBody.put("message", message);
        errorBody.put("type", status == 429 ? "rate_limit_error" : "invalid_request_error");
        errorBody.put("code", null);
        error.put("error", errorBody);
        sendJson(exchange, status, error);
    }

    private void sendJson(HttpExchange exchange, int status, Object data) throws IOException {
        byte[] bytes = ApiHandler.toJson(data).getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.sendResponseHeaders(status, bytes.length);
        exchange.getResponseBody().write(bytes);
        exchange.getResponseBody().close();
    }

    private String readBody(HttpExchange exchange) throws IOException {
        try (InputStream is = exchange.getRequestBody()) {
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            byte[] tmp = new byte[8192];
            int n;
            while ((n = is.read(tmp)) != -1) {
                buffer.write(tmp, 0, n);
            }
            return new String(buffer.toByteArray(), StandardCharsets.UTF_8);
        }
    }
}
