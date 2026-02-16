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
            String content = (String) msg.get("content");
            if (role == null || content == null) continue;
            messages.add(new String[]{role, content});
        }

        if (messages.isEmpty()) {
            sendError(exchange, 400, "messages must contain at least one message with role and content");
            return;
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

        SamplerConfig samplerConfig = new SamplerConfig(temperature, topK, topP, repPenalty, System.nanoTime());
        GenerationRequest request = GenerationRequest.builder()
            .prompt(formattedPrompt)
            .maxTokens(maxTokens)
            .samplerConfig(samplerConfig)
            .useChat(false)
            .rawMode(true)
            .build();

        String modelName = engine.getModelName();
        String completionId = "chatcmpl-" + UUID.randomUUID().toString().replace("-", "").substring(0, 29);
        long created = System.currentTimeMillis() / 1000;

        if (stream) {
            handleStreamingChat(exchange, engine, request, completionId, modelName, created, stopSequences);
        } else {
            handleNonStreamingChat(exchange, engine, request, completionId, modelName, created, stopSequences);
        }
    }

    private void handleStreamingChat(HttpExchange exchange, LLMEngine engine,
                                      GenerationRequest request, String completionId,
                                      String modelName, long created,
                                      final List<String> stopSequences) throws IOException {
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
                                         List<String> stopSequences) throws IOException {
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

            // Build response
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("id", completionId);
            result.put("object", "chat.completion");
            result.put("created", created);
            result.put("model", modelName);

            Map<String, Object> message = new LinkedHashMap<>();
            message.put("role", "assistant");
            message.put("content", text);

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

    // --- Helpers ---

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
