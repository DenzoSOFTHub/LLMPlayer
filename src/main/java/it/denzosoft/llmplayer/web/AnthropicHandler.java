package it.denzosoft.llmplayer.web;

import com.sun.net.httpserver.HttpExchange;
import it.denzosoft.llmplayer.api.*;
import it.denzosoft.llmplayer.sampler.SamplerConfig;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Anthropic Messages API handler.
 * Implements POST /v1/messages and POST /v1/messages/count_tokens.
 * Compatible with Claude Code and other Anthropic API clients.
 */
public class AnthropicHandler {

    private final ApiHandler apiHandler;

    public AnthropicHandler(ApiHandler apiHandler) {
        this.apiHandler = apiHandler;
    }

    public void handle(HttpExchange exchange) throws IOException {
        String path = exchange.getRequestURI().getPath();
        String method = exchange.getRequestMethod();

        // CORS headers
        exchange.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        exchange.getResponseHeaders().set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        exchange.getResponseHeaders().set("Access-Control-Allow-Headers",
                "Content-Type, Authorization, x-api-key, anthropic-version, anthropic-beta");

        if ("OPTIONS".equals(method)) {
            exchange.sendResponseHeaders(204, -1);
            exchange.close();
            return;
        }

        try {
            if ("/v1/messages".equals(path)) {
                if ("POST".equals(method)) handleMessages(exchange);
                else sendError(exchange, 405, "method_not_allowed", "Method not allowed");
            } else if ("/v1/messages/count_tokens".equals(path)) {
                if ("POST".equals(method)) handleCountTokens(exchange);
                else sendError(exchange, 405, "method_not_allowed", "Method not allowed");
            } else {
                sendError(exchange, 404, "not_found", "Not found");
            }
        } catch (Exception e) {
            System.err.println("Anthropic API error: " + e.getMessage());
            e.printStackTrace();
            try {
                sendError(exchange, 500, "api_error",
                        e.getMessage() != null ? e.getMessage() : "Internal error");
            } catch (IOException ignored) {}
        }
    }

    @SuppressWarnings("unchecked")
    private void handleMessages(HttpExchange exchange) throws IOException {
        LLMEngine engine = apiHandler.getEngine();
        if (engine == null) {
            sendError(exchange, 503, "api_error", "No model loaded. Load a model first.");
            return;
        }
        if (apiHandler.isGenerating()) {
            sendError(exchange, 429, "rate_limit_error", "Generation already in progress");
            return;
        }

        Map<String, Object> body = ApiHandler.parseJson(readBody(exchange));

        // max_tokens is required in Anthropic API
        if (!body.containsKey("max_tokens")) {
            sendError(exchange, 400, "invalid_request_error", "max_tokens is required");
            return;
        }

        // Parse messages
        List<?> messagesRaw = (List<?>) body.get("messages");
        if (messagesRaw == null || messagesRaw.isEmpty()) {
            sendError(exchange, 400, "invalid_request_error",
                    "messages is required and must be non-empty");
            return;
        }

        // Convert Anthropic messages to List<String[]> for ChatTemplate
        List<String[]> messages = new ArrayList<>();

        // Handle top-level system field
        Object systemObj = body.get("system");
        if (systemObj != null) {
            String systemText = extractSystemText(systemObj);
            if (systemText != null && !systemText.isEmpty()) {
                messages.add(new String[]{"system", systemText});
            }
        }

        // Parse each message
        for (Object msgObj : messagesRaw) {
            Map<String, Object> msg = (Map<String, Object>) msgObj;
            String role = (String) msg.get("role");
            if (role == null) continue;

            Object contentObj = msg.get("content");
            String content = extractMessageContent(contentObj);

            if ("user".equals(role) || "assistant".equals(role)) {
                messages.add(new String[]{role, content != null ? content : ""});
            }
        }

        if (messages.isEmpty() || (messages.size() == 1 && "system".equals(messages.get(0)[0]))) {
            sendError(exchange, 400, "invalid_request_error",
                    "messages must contain at least one user or assistant message");
            return;
        }

        // Parse tools for system prompt injection
        List<?> tools = (List<?>) body.get("tools");
        List<String> toolNames = new ArrayList<>();
        if (tools != null && !tools.isEmpty()) {
            String toolPrompt = formatToolsSystemPrompt(tools, toolNames);
            // Inject into system message
            boolean hasSystem = false;
            for (int i = 0; i < messages.size(); i++) {
                if ("system".equals(messages.get(i)[0])) {
                    messages.set(i, new String[]{"system",
                            messages.get(i)[1] + "\n\n" + toolPrompt});
                    hasSystem = true;
                    break;
                }
            }
            if (!hasSystem) {
                messages.add(0, new String[]{"system", toolPrompt});
            }
        }

        // Parse parameters
        boolean stream = body.containsKey("stream") && Boolean.TRUE.equals(body.get("stream"));
        int maxTokens = ((Number) body.get("max_tokens")).intValue();
        float temperature = body.containsKey("temperature")
                ? ((Number) body.get("temperature")).floatValue() : 0.7f;
        float topP = body.containsKey("top_p")
                ? ((Number) body.get("top_p")).floatValue() : 0.9f;
        int topK = body.containsKey("top_k")
                ? ((Number) body.get("top_k")).intValue() : 40;

        // Parse stop sequences
        List<String> stopSequences = new ArrayList<>();
        if (body.containsKey("stop_sequences")) {
            Object stopObj = body.get("stop_sequences");
            if (stopObj instanceof List) {
                for (Object s : (List<?>) stopObj) {
                    if (s instanceof String) stopSequences.add((String) s);
                }
            }
        }

        // Build the formatted prompt
        String formattedPrompt = engine.getChatTemplate().formatConversation(messages);
        String cacheKey = computeCacheKey(messages);

        SamplerConfig samplerConfig = new SamplerConfig(temperature, topK, topP, 1.1f, System.nanoTime());
        GenerationRequest request = GenerationRequest.builder()
                .prompt(formattedPrompt)
                .maxTokens(maxTokens)
                .samplerConfig(samplerConfig)
                .useChat(false)
                .rawMode(true)
                .cacheKey(cacheKey)
                .build();

        String modelName = engine.getModelName();
        String messageId = "msg_" + UUID.randomUUID().toString().replace("-", "").substring(0, 24);

        if (stream) {
            handleStreamingMessages(exchange, engine, request, messageId, modelName,
                    stopSequences, toolNames.isEmpty() ? null : toolNames);
        } else {
            handleNonStreamingMessages(exchange, engine, request, messageId, modelName,
                    stopSequences, toolNames.isEmpty() ? null : toolNames);
        }
    }

    private void handleNonStreamingMessages(HttpExchange exchange, LLMEngine engine,
                                             GenerationRequest request, String messageId,
                                             String modelName, List<String> stopSequences,
                                             List<String> toolNames) throws IOException {
        apiHandler.setStopRequested(false);
        apiHandler.setGenerating(true);

        try {
            GenerationResponse response = engine.generate(request);

            String text = response.text();
            String stopReason = (response.tokenCount() >= request.maxTokens()) ? "max_tokens" : "end_turn";
            String stopSequence = null;

            // Check stop sequences
            for (String seq : stopSequences) {
                int idx = text.indexOf(seq);
                if (idx >= 0) {
                    text = text.substring(0, idx);
                    stopReason = "end_turn";
                    stopSequence = seq;
                    break;
                }
            }

            // Check for tool call in output
            Map<String, Object> toolCall = (toolNames != null) ? tryParseToolCall(text.trim(), toolNames) : null;

            // Build content blocks
            List<Object> contentBlocks = new ArrayList<>();
            if (toolCall != null) {
                // Text before tool call (if any)
                String beforeTool = text.trim();
                int braceIdx = beforeTool.indexOf('{');
                if (braceIdx > 0) {
                    String preText = beforeTool.substring(0, braceIdx).trim();
                    if (!preText.isEmpty()) {
                        Map<String, Object> textBlock = new LinkedHashMap<>();
                        textBlock.put("type", "text");
                        textBlock.put("text", preText);
                        contentBlocks.add(textBlock);
                    }
                }

                // Tool use block
                Map<String, Object> toolUseBlock = new LinkedHashMap<>();
                toolUseBlock.put("type", "tool_use");
                toolUseBlock.put("id", "toolu_" + UUID.randomUUID().toString().replace("-", "").substring(0, 24));
                toolUseBlock.put("name", toolCall.get("name"));
                toolUseBlock.put("input", toolCall.get("arguments"));
                contentBlocks.add(toolUseBlock);
                stopReason = "tool_use";
            } else {
                Map<String, Object> textBlock = new LinkedHashMap<>();
                textBlock.put("type", "text");
                textBlock.put("text", text);
                contentBlocks.add(textBlock);
            }

            // Build response
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("id", messageId);
            result.put("type", "message");
            result.put("role", "assistant");
            result.put("content", contentBlocks);
            result.put("model", modelName);
            result.put("stop_reason", stopReason);
            result.put("stop_sequence", stopSequence);

            Map<String, Object> usage = new LinkedHashMap<>();
            usage.put("input_tokens", response.promptTokenCount());
            usage.put("output_tokens", response.tokenCount());
            result.put("usage", usage);

            sendJson(exchange, 200, result);
        } finally {
            apiHandler.setGenerating(false);
        }
    }

    private void handleStreamingMessages(HttpExchange exchange, LLMEngine engine,
                                          GenerationRequest request, String messageId,
                                          String modelName, final List<String> stopSequences,
                                          final List<String> toolNames) throws IOException {
        exchange.getResponseHeaders().set("Content-Type", "text/event-stream");
        exchange.getResponseHeaders().set("Cache-Control", "no-cache");
        exchange.getResponseHeaders().set("Connection", "keep-alive");
        exchange.sendResponseHeaders(200, 0);

        final OutputStream out = exchange.getResponseBody();
        apiHandler.setStopRequested(false);
        apiHandler.setGenerating(true);

        try {
            // Send message_start
            Map<String, Object> messageShell = new LinkedHashMap<>();
            messageShell.put("id", messageId);
            messageShell.put("type", "message");
            messageShell.put("role", "assistant");
            messageShell.put("content", Collections.emptyList());
            messageShell.put("model", modelName);
            messageShell.put("stop_reason", null);
            messageShell.put("stop_sequence", null);
            Map<String, Object> startUsage = new LinkedHashMap<>();
            startUsage.put("input_tokens", 0);
            startUsage.put("output_tokens", 0);
            messageShell.put("usage", startUsage);

            Map<String, Object> messageStart = new LinkedHashMap<>();
            messageStart.put("type", "message_start");
            messageStart.put("message", messageShell);
            writeNamedSSE(out, "message_start", ApiHandler.toJson(messageStart));

            // Send content_block_start
            Map<String, Object> blockStart = new LinkedHashMap<>();
            blockStart.put("type", "content_block_start");
            blockStart.put("index", 0);
            Map<String, Object> contentBlock = new LinkedHashMap<>();
            contentBlock.put("type", "text");
            contentBlock.put("text", "");
            blockStart.put("content_block", contentBlock);
            writeNamedSSE(out, "content_block_start", ApiHandler.toJson(blockStart));

            // Send ping
            Map<String, Object> ping = new LinkedHashMap<>();
            ping.put("type", "ping");
            writeNamedSSE(out, "ping", ApiHandler.toJson(ping));

            // Stream content deltas
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
                        delta.put("type", "content_block_delta");
                        delta.put("index", 0);
                        Map<String, Object> deltaContent = new LinkedHashMap<>();
                        deltaContent.put("type", "text_delta");
                        deltaContent.put("text", token);
                        delta.put("delta", deltaContent);
                        writeNamedSSE(out, "content_block_delta", ApiHandler.toJson(delta));
                    } catch (IOException e) {
                        return false;
                    }
                    return true;
                }
            });

            // Send content_block_stop
            Map<String, Object> blockStop = new LinkedHashMap<>();
            blockStop.put("type", "content_block_stop");
            blockStop.put("index", 0);
            writeNamedSSE(out, "content_block_stop", ApiHandler.toJson(blockStop));

            // Determine stop reason
            String stopReason = (response.tokenCount() >= request.maxTokens()) ? "max_tokens" : "end_turn";
            if (toolNames != null && tryParseToolCall(accumulated.toString().trim(), toolNames) != null) {
                stopReason = "tool_use";
            }

            // Send message_delta with stop_reason and output usage
            Map<String, Object> messageDelta = new LinkedHashMap<>();
            messageDelta.put("type", "message_delta");
            Map<String, Object> deltaObj = new LinkedHashMap<>();
            deltaObj.put("stop_reason", stopReason);
            deltaObj.put("stop_sequence", null);
            messageDelta.put("delta", deltaObj);
            Map<String, Object> deltaUsage = new LinkedHashMap<>();
            deltaUsage.put("output_tokens", response.tokenCount());
            messageDelta.put("usage", deltaUsage);
            writeNamedSSE(out, "message_delta", ApiHandler.toJson(messageDelta));

            // Send message_stop
            Map<String, Object> messageStop = new LinkedHashMap<>();
            messageStop.put("type", "message_stop");
            writeNamedSSE(out, "message_stop", ApiHandler.toJson(messageStop));

        } catch (Exception e) {
            try {
                Map<String, Object> errEvent = new LinkedHashMap<>();
                errEvent.put("type", "error");
                Map<String, Object> errBody = new LinkedHashMap<>();
                errBody.put("type", "api_error");
                errBody.put("message", e.getMessage() != null ? e.getMessage() : "Generation error");
                errEvent.put("error", errBody);
                writeNamedSSE(out, "error", ApiHandler.toJson(errEvent));
            } catch (IOException ignored) {}
        } finally {
            apiHandler.setGenerating(false);
            out.close();
        }
    }

    @SuppressWarnings("unchecked")
    private void handleCountTokens(HttpExchange exchange) throws IOException {
        LLMEngine engine = apiHandler.getEngine();
        if (engine == null) {
            sendError(exchange, 503, "api_error", "No model loaded. Load a model first.");
            return;
        }

        Map<String, Object> body = ApiHandler.parseJson(readBody(exchange));

        List<?> messagesRaw = (List<?>) body.get("messages");
        if (messagesRaw == null || messagesRaw.isEmpty()) {
            sendError(exchange, 400, "invalid_request_error",
                    "messages is required and must be non-empty");
            return;
        }

        // Convert messages
        List<String[]> messages = new ArrayList<>();

        Object systemObj = body.get("system");
        if (systemObj != null) {
            String systemText = extractSystemText(systemObj);
            if (systemText != null && !systemText.isEmpty()) {
                messages.add(new String[]{"system", systemText});
            }
        }

        for (Object msgObj : messagesRaw) {
            Map<String, Object> msg = (Map<String, Object>) msgObj;
            String role = (String) msg.get("role");
            if (role == null) continue;

            Object contentObj = msg.get("content");
            String content = extractMessageContent(contentObj);
            messages.add(new String[]{role, content != null ? content : ""});
        }

        // Format with chat template and tokenize
        String formattedPrompt = engine.getChatTemplate().formatConversation(messages);
        int[] tokens = engine.getTokenizer().encode(formattedPrompt);

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("input_tokens", tokens.length);
        sendJson(exchange, 200, result);
    }

    // --- Message content extraction ---

    /**
     * Extract system text from the top-level system field.
     * Can be a string or an array of content blocks.
     */
    @SuppressWarnings("unchecked")
    private String extractSystemText(Object systemObj) {
        if (systemObj instanceof String) {
            return (String) systemObj;
        }
        if (systemObj instanceof List) {
            StringBuilder sb = new StringBuilder();
            for (Object block : (List<?>) systemObj) {
                if (block instanceof Map) {
                    Map<String, Object> b = (Map<String, Object>) block;
                    if ("text".equals(b.get("type")) && b.containsKey("text")) {
                        if (sb.length() > 0) sb.append("\n");
                        sb.append(b.get("text"));
                    }
                }
            }
            return sb.toString();
        }
        return null;
    }

    /**
     * Extract message content from a content field.
     * Can be a string or array of content blocks (text, tool_use, tool_result).
     */
    @SuppressWarnings("unchecked")
    private String extractMessageContent(Object contentObj) {
        if (contentObj instanceof String) {
            return (String) contentObj;
        }
        if (contentObj instanceof List) {
            StringBuilder sb = new StringBuilder();
            for (Object block : (List<?>) contentObj) {
                if (block instanceof Map) {
                    Map<String, Object> b = (Map<String, Object>) block;
                    String type = (String) b.get("type");
                    if ("text".equals(type)) {
                        if (sb.length() > 0) sb.append("\n");
                        sb.append(b.get("text"));
                    } else if ("tool_use".equals(type)) {
                        if (sb.length() > 0) sb.append("\n");
                        String name = (String) b.get("name");
                        Object input = b.get("input");
                        sb.append("[Called tool: ").append(name).append("(");
                        sb.append(input != null ? ApiHandler.toJson(input) : "{}");
                        sb.append(")]");
                    } else if ("tool_result".equals(type)) {
                        if (sb.length() > 0) sb.append("\n");
                        Object resultContent = b.get("content");
                        String resultText;
                        if (resultContent instanceof String) {
                            resultText = (String) resultContent;
                        } else if (resultContent instanceof List) {
                            StringBuilder rsb = new StringBuilder();
                            for (Object rb : (List<?>) resultContent) {
                                if (rb instanceof Map) {
                                    Map<String, Object> rblock = (Map<String, Object>) rb;
                                    if ("text".equals(rblock.get("type"))) {
                                        if (rsb.length() > 0) rsb.append("\n");
                                        rsb.append(rblock.get("text"));
                                    }
                                }
                            }
                            resultText = rsb.toString();
                        } else {
                            resultText = resultContent != null ? resultContent.toString() : "";
                        }
                        sb.append("[Tool result: ").append(resultText).append("]");
                    }
                }
            }
            return sb.toString();
        }
        return contentObj != null ? contentObj.toString() : null;
    }

    // --- Tool calling ---

    @SuppressWarnings("unchecked")
    private String formatToolsSystemPrompt(List<?> tools, List<String> toolNames) {
        StringBuilder sb = new StringBuilder();
        sb.append("You have access to the following tools:\n\n");
        for (Object toolObj : tools) {
            Map<String, Object> tool = (Map<String, Object>) toolObj;
            String name = (String) tool.get("name");
            if (name == null) continue;
            toolNames.add(name);
            sb.append("Function: ").append(name).append("\n");
            if (tool.containsKey("description")) {
                sb.append("Description: ").append(tool.get("description")).append("\n");
            }
            if (tool.containsKey("input_schema")) {
                sb.append("Parameters: ").append(ApiHandler.toJson(tool.get("input_schema"))).append("\n");
            }
            sb.append("\n");
        }
        sb.append("When you need to call a tool, respond ONLY with a JSON object in this exact format:\n");
        sb.append("{\"name\": \"function_name\", \"arguments\": {\"arg1\": \"value1\"}}\n");
        sb.append("Do not include any other text when making a tool call.");
        return sb.toString();
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> tryParseToolCall(String text, List<String> toolNames) {
        if (text.isEmpty()) return null;
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
        } catch (Exception ignored) {}
        return null;
    }

    // --- Helpers ---

    private static String computeCacheKey(List<String[]> messages) {
        StringBuilder sb = new StringBuilder();
        for (String[] msg : messages) {
            sb.append(msg[0]).append(':').append(msg[1]).append('\n');
        }
        return Integer.toHexString(sb.toString().hashCode());
    }

    private static void writeNamedSSE(OutputStream out, String event, String data) throws IOException {
        out.write(("event: " + event + "\ndata: " + data + "\n\n").getBytes(StandardCharsets.UTF_8));
        out.flush();
    }

    private void sendError(HttpExchange exchange, int status, String type, String message) throws IOException {
        Map<String, Object> error = new LinkedHashMap<>();
        error.put("type", "error");
        Map<String, Object> errorBody = new LinkedHashMap<>();
        errorBody.put("type", type);
        errorBody.put("message", message);
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
