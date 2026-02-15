package it.denzosoft.llmplayer.web;

import com.sun.net.httpserver.HttpExchange;
import it.denzosoft.llmplayer.api.*;
import it.denzosoft.llmplayer.gpu.GpuConfig;
import it.denzosoft.llmplayer.sampler.SamplerConfig;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * REST API handler for the LLMPlayer Web UI.
 * Manages model loading/unloading, info, and streaming chat generation.
 */
public class ApiHandler {

    private final String ggufDirectory;
    private volatile LLMEngine engine;
    private volatile boolean stopRequested;
    private volatile boolean generating;

    public ApiHandler(String ggufDirectory) {
        this.ggufDirectory = ggufDirectory;
    }

    public void handle(HttpExchange exchange) throws IOException {
        String path = exchange.getRequestURI().getPath();
        String method = exchange.getRequestMethod();

        // CORS headers
        exchange.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        exchange.getResponseHeaders().set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        exchange.getResponseHeaders().set("Access-Control-Allow-Headers", "Content-Type");

        if ("OPTIONS".equals(method)) {
            exchange.sendResponseHeaders(204, -1);
            exchange.close();
            return;
        }

        try {
            if ("/api/models".equals(path)) {
                if ("GET".equals(method)) handleListModels(exchange);
                else methodNotAllowed(exchange);
            } else if ("/api/models/load".equals(path)) {
                if ("POST".equals(method)) handleLoadModel(exchange);
                else methodNotAllowed(exchange);
            } else if ("/api/models/unload".equals(path)) {
                if ("POST".equals(method)) handleUnloadModel(exchange);
                else methodNotAllowed(exchange);
            } else if ("/api/models/info".equals(path)) {
                if ("GET".equals(method)) handleModelInfo(exchange);
                else methodNotAllowed(exchange);
            } else if ("/api/chat".equals(path)) {
                if ("POST".equals(method)) handleChat(exchange);
                else methodNotAllowed(exchange);
            } else if ("/api/chat/stop".equals(path)) {
                if ("POST".equals(method)) handleChatStop(exchange);
                else methodNotAllowed(exchange);
            } else if ("/api/gpu/devices".equals(path)) {
                if ("GET".equals(method)) handleGpuDevices(exchange);
                else methodNotAllowed(exchange);
            } else if ("/api/memory/check".equals(path)) {
                if ("POST".equals(method)) handleMemoryCheck(exchange);
                else methodNotAllowed(exchange);
            } else if ("/api/hardware/plan".equals(path)) {
                if ("POST".equals(method)) handleHardwarePlan(exchange);
                else methodNotAllowed(exchange);
            } else {
                Map<String, Object> err = new LinkedHashMap<>();
                err.put("error", "Not found");
                sendJson(exchange, 404, err);
            }
        } catch (Exception e) {
            System.err.println("API error: " + e.getMessage());
            e.printStackTrace();
            try {
                Map<String, Object> err = new LinkedHashMap<>();
                err.put("error", e.getMessage() != null ? e.getMessage() : "Internal error");
                sendJson(exchange, 500, err);
            } catch (IOException ignored) {}
        }
    }

    private void handleListModels(HttpExchange exchange) throws IOException {
        List<Map<String, Object>> models = new ArrayList<>();
        Path dir = Paths.get(ggufDirectory);
        if (Files.isDirectory(dir)) {
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir, "*.gguf")) {
                for (Path file : stream) {
                    Map<String, Object> m = new LinkedHashMap<>();
                    m.put("name", file.getFileName().toString());
                    m.put("path", file.toString());
                    try {
                        m.put("size", Files.size(file));
                    } catch (IOException e) {
                        m.put("size", 0);
                    }
                    models.add(m);
                }
            }
        }
        // Sort by name
        models.sort(new Comparator<Map<String, Object>>() {
            @Override
            public int compare(Map<String, Object> a, Map<String, Object> b) {
                return ((String) a.get("name")).compareToIgnoreCase((String) b.get("name"));
            }
        });
        sendJson(exchange, 200, models);
    }

    private void handleLoadModel(HttpExchange exchange) throws IOException {
        Map<String, Object> body = parseJson(readBody(exchange));
        String path = (String) body.get("path");
        int contextLength = body.containsKey("contextLength")
            ? ((Number) body.get("contextLength")).intValue() : 2048;

        if (path == null || path.trim().isEmpty()) {
            Map<String, Object> err = new LinkedHashMap<>();
            err.put("error", "Missing 'path' in request body");
            sendJson(exchange, 400, err);
            return;
        }

        // Unload previous model
        if (engine != null) {
            engine.close();
            engine = null;
        }

        // GPU config
        GpuConfig gpuConfig = new GpuConfig();
        if (body.containsKey("gpu") && Boolean.TRUE.equals(body.get("gpu"))) {
            gpuConfig.setEnabled(true);
            if (body.containsKey("gpuDevice")) {
                gpuConfig.setDeviceId(((Number) body.get("gpuDevice")).intValue());
            }
            if (body.containsKey("gpuLayers")) {
                gpuConfig.setGpuLayers(((Number) body.get("gpuLayers")).intValue());
            }
        }

        System.out.println("Loading model: " + path + " (ctx=" + contextLength +
            ", gpu=" + gpuConfig.isEnabled() + ")");
        long start = System.currentTimeMillis();
        engine = LLMEngine.load(Paths.get(path), contextLength, gpuConfig);
        long elapsed = System.currentTimeMillis() - start;
        System.out.println("Model loaded in " + elapsed + "ms");

        ModelInfo info = engine.getModelInfo();
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("status", "loaded");
        result.put("loadTimeMs", elapsed);
        result.put("model", modelInfoToMap(info));
        // Include GPU info in response
        if (engine.getGpuDeviceName() != null) {
            result.put("gpuDevice", engine.getGpuDeviceName());
            result.put("gpuLayers", engine.getGpuLayersUsed());
            result.put("totalLayers", info.blockCount());
        }
        sendJson(exchange, 200, result);
    }

    private void handleUnloadModel(HttpExchange exchange) throws IOException {
        if (engine != null) {
            engine.close();
            engine = null;
            System.out.println("Model unloaded");
        }
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("status", "unloaded");
        sendJson(exchange, 200, result);
    }

    private void handleModelInfo(HttpExchange exchange) throws IOException {
        if (engine == null) {
            Map<String, Object> err = new LinkedHashMap<>();
            err.put("error", "No model loaded");
            sendJson(exchange, 400, err);
            return;
        }
        ModelInfo info = engine.getModelInfo();
        Map<String, Object> result = modelInfoToMap(info);
        // Include GPU placement info
        result.put("gpuLayers", engine.getGpuLayersUsed());
        result.put("gpuDeviceName", engine.getGpuDeviceName());
        result.put("moeOptimizedGpu", engine.isMoeOptimizedGpu());
        sendJson(exchange, 200, result);
    }

    private void handleChat(HttpExchange exchange) throws IOException {
        if (engine == null) {
            Map<String, Object> err = new LinkedHashMap<>();
            err.put("error", "No model loaded");
            sendJson(exchange, 400, err);
            return;
        }
        if (generating) {
            Map<String, Object> err = new LinkedHashMap<>();
            err.put("error", "Generation already in progress");
            sendJson(exchange, 409, err);
            return;
        }

        Map<String, Object> body = parseJson(readBody(exchange));
        String prompt = body.containsKey("prompt") ? (String) body.get("prompt") : "";
        String systemMessage = (String) body.get("systemMessage");
        float temperature = body.containsKey("temperature")
            ? ((Number) body.get("temperature")).floatValue() : 0.7f;
        int maxTokens = body.containsKey("maxTokens")
            ? ((Number) body.get("maxTokens")).intValue() : 256;
        int topK = body.containsKey("topK")
            ? ((Number) body.get("topK")).intValue() : 40;
        float topP = body.containsKey("topP")
            ? ((Number) body.get("topP")).floatValue() : 0.9f;
        float repPenalty = body.containsKey("repPenalty")
            ? ((Number) body.get("repPenalty")).floatValue() : 1.1f;

        SamplerConfig samplerConfig = new SamplerConfig(temperature, topK, topP, repPenalty, System.nanoTime());
        GenerationRequest request = GenerationRequest.builder()
            .prompt(prompt)
            .systemMessage(systemMessage)
            .maxTokens(maxTokens)
            .samplerConfig(samplerConfig)
            .useChat(true)
            .build();

        // Set up SSE streaming
        exchange.getResponseHeaders().set("Content-Type", "text/event-stream");
        exchange.getResponseHeaders().set("Cache-Control", "no-cache");
        exchange.getResponseHeaders().set("Connection", "keep-alive");
        exchange.sendResponseHeaders(200, 0); // chunked

        OutputStream out = exchange.getResponseBody();
        stopRequested = false;
        generating = true;

        try {
            GenerationResponse response = engine.generate(request, new StreamingCallback() {
                @Override
                public boolean onToken(String token, int tokenId) {
                    if (stopRequested) return false;
                    try {
                        Map<String, Object> tokenEvent = new LinkedHashMap<>();
                        tokenEvent.put("token", token);
                        tokenEvent.put("done", false);
                        String event = "data: " + toJson(tokenEvent) + "\n\n";
                        out.write(event.getBytes(StandardCharsets.UTF_8));
                        out.flush();
                    } catch (IOException e) {
                        return false;
                    }
                    return true;
                }
            });

            // Send final event with stats
            Map<String, Object> stats = new LinkedHashMap<>();
            stats.put("tokenCount", response.tokenCount());
            stats.put("promptTokenCount", response.promptTokenCount());
            stats.put("tokensPerSecond", Math.round(response.tokensPerSecond() * 10.0) / 10.0);
            stats.put("timeMs", response.timeMs());

            Map<String, Object> doneEvent = new LinkedHashMap<>();
            doneEvent.put("done", true);
            doneEvent.put("stats", stats);
            if (stopRequested) doneEvent.put("stopped", true);

            String finalEvent = "data: " + toJson(doneEvent) + "\n\n";
            out.write(finalEvent.getBytes(StandardCharsets.UTF_8));
            out.flush();
        } catch (Exception e) {
            try {
                Map<String, Object> errEvent = new LinkedHashMap<>();
                errEvent.put("error", e.getMessage() != null ? e.getMessage() : "Generation error");
                errEvent.put("done", true);
                String errorEvent = "data: " + toJson(errEvent) + "\n\n";
                out.write(errorEvent.getBytes(StandardCharsets.UTF_8));
                out.flush();
            } catch (IOException ignored) {}
        } finally {
            generating = false;
            out.close();
        }
    }

    private void handleChatStop(HttpExchange exchange) throws IOException {
        stopRequested = true;
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("status", "stopping");
        sendJson(exchange, 200, result);
    }

    private void handleGpuDevices(HttpExchange exchange) throws IOException {
        List<Map<String, Object>> devices = LLMEngine.listGpuDevices();
        sendJson(exchange, 200, devices);
    }

    private void handleMemoryCheck(HttpExchange exchange) throws IOException {
        Map<String, Object> body = parseJson(readBody(exchange));
        String path = (String) body.get("path");
        int contextLength = body.containsKey("contextLength")
            ? ((Number) body.get("contextLength")).intValue() : 2048;

        if (path == null || path.trim().isEmpty()) {
            Map<String, Object> err = new LinkedHashMap<>();
            err.put("error", "Missing 'path' in request body");
            sendJson(exchange, 400, err);
            return;
        }

        LLMEngine.MemoryCheck check = LLMEngine.checkMemory(Paths.get(path), contextLength);
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("estimatedRam", check.estimatedRam());
        result.put("availableRam", check.availableRam());
        result.put("safe", check.isSafe());
        result.put("message", check.message());
        sendJson(exchange, 200, result);
    }

    private void handleHardwarePlan(HttpExchange exchange) throws IOException {
        Map<String, Object> body = parseJson(readBody(exchange));
        String path = (String) body.get("path");
        int contextLength = body.containsKey("contextLength")
            ? ((Number) body.get("contextLength")).intValue() : 2048;

        if (path == null || path.trim().isEmpty()) {
            Map<String, Object> err = new LinkedHashMap<>();
            err.put("error", "Missing 'path' in request body");
            sendJson(exchange, 400, err);
            return;
        }

        LLMEngine.HardwarePlan plan = LLMEngine.buildHardwarePlan(Paths.get(path), contextLength);
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("modelName", plan.modelName());
        result.put("gpuAvailable", plan.isGpuAvailable());
        result.put("gpuDeviceName", plan.gpuDeviceName());
        result.put("gpuVram", plan.gpuVram());
        result.put("gpuLayers", plan.gpuLayers());
        result.put("totalLayers", plan.totalLayers());
        result.put("recommended", plan.isRecommended());
        result.put("summary", plan.summary());
        result.put("memorySafe", plan.memoryCheck().isSafe());
        result.put("estimatedRam", plan.memoryCheck().estimatedRam());
        result.put("availableRam", plan.memoryCheck().availableRam());
        sendJson(exchange, 200, result);
    }

    public void shutdown() {
        stopRequested = true;
        if (engine != null) {
            engine.close();
            engine = null;
        }
    }

    // --- Helpers ---

    private Map<String, Object> modelInfoToMap(ModelInfo info) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("name", info.name());
        m.put("architecture", info.architecture());
        m.put("embeddingLength", info.embeddingLength());
        m.put("blockCount", info.blockCount());
        m.put("headCount", info.headCount());
        m.put("headCountKV", info.headCountKV());
        m.put("contextLength", info.contextLength());
        m.put("vocabSize", info.vocabSize());
        m.put("intermediateSize", info.intermediateSize());
        return m;
    }

    private void methodNotAllowed(HttpExchange exchange) throws IOException {
        Map<String, Object> err = new LinkedHashMap<>();
        err.put("error", "Method not allowed");
        sendJson(exchange, 405, err);
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

    private void sendJson(HttpExchange exchange, int status, Object data) throws IOException {
        byte[] bytes = toJson(data).getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.sendResponseHeaders(status, bytes.length);
        exchange.getResponseBody().write(bytes);
        exchange.getResponseBody().close();
    }

    // --- Minimal JSON serializer/parser ---

    @SuppressWarnings("unchecked")
    static String toJson(Object obj) {
        if (obj == null) return "null";
        if (obj instanceof Boolean) return obj.toString();
        if (obj instanceof Number) {
            Number n = (Number) obj;
            // Avoid trailing .0 for integers
            if (n instanceof Double) {
                double d = (Double) n;
                if (d == Math.floor(d) && !Double.isInfinite(d)) {
                    return String.valueOf((long) d);
                }
            }
            if (n instanceof Float) {
                float f = (Float) n;
                if (f == Math.floor(f) && !Float.isInfinite(f)) {
                    return String.valueOf((int) f);
                }
            }
            return n.toString();
        }
        if (obj instanceof String) return escapeJsonString((String) obj);
        if (obj instanceof Map) {
            Map<?, ?> map = (Map<?, ?>) obj;
            StringBuilder sb = new StringBuilder("{");
            boolean first = true;
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                if (!first) sb.append(",");
                first = false;
                sb.append(escapeJsonString(entry.getKey().toString()));
                sb.append(":");
                sb.append(toJson(entry.getValue()));
            }
            sb.append("}");
            return sb.toString();
        }
        if (obj instanceof List) {
            List<?> list = (List<?>) obj;
            StringBuilder sb = new StringBuilder("[");
            boolean first = true;
            for (Object item : list) {
                if (!first) sb.append(",");
                first = false;
                sb.append(toJson(item));
            }
            sb.append("]");
            return sb.toString();
        }
        return escapeJsonString(obj.toString());
    }

    private static String escapeJsonString(String s) {
        StringBuilder sb = new StringBuilder("\"");
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '"') sb.append("\\\"");
            else if (c == '\\') sb.append("\\\\");
            else if (c == '\n') sb.append("\\n");
            else if (c == '\r') sb.append("\\r");
            else if (c == '\t') sb.append("\\t");
            else if (c == '\b') sb.append("\\b");
            else if (c == '\f') sb.append("\\f");
            else if (c < 0x20) sb.append(String.format("\\u%04x", (int) c));
            else sb.append(c);
        }
        sb.append("\"");
        return sb.toString();
    }

    static Map<String, Object> parseJson(String json) {
        json = json.trim();
        if (json.isEmpty()) return Collections.emptyMap();
        JsonParser parser = new JsonParser(json);
        Object result = parser.parseValue();
        if (result instanceof Map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> map = (Map<String, Object>) result;
            return map;
        }
        return Collections.emptyMap();
    }

    /**
     * Minimal recursive-descent JSON parser.
     */
    private static class JsonParser {
        private final String json;
        private int pos;

        JsonParser(String json) {
            this.json = json;
            this.pos = 0;
        }

        Object parseValue() {
            skipWhitespace();
            if (pos >= json.length()) return null;
            char c = json.charAt(pos);
            if (c == '{') return parseObject();
            if (c == '[') return parseArray();
            if (c == '"') return parseString();
            if (c == 't' || c == 'f') return parseBoolean();
            if (c == 'n') return parseNull();
            return parseNumber();
        }

        Map<String, Object> parseObject() {
            Map<String, Object> map = new LinkedHashMap<>();
            pos++; // skip '{'
            skipWhitespace();
            if (pos < json.length() && json.charAt(pos) == '}') {
                pos++;
                return map;
            }
            while (pos < json.length()) {
                skipWhitespace();
                String key = parseString();
                skipWhitespace();
                expect(':');
                Object value = parseValue();
                map.put(key, value);
                skipWhitespace();
                if (pos < json.length() && json.charAt(pos) == ',') {
                    pos++;
                } else {
                    break;
                }
            }
            skipWhitespace();
            if (pos < json.length() && json.charAt(pos) == '}') pos++;
            return map;
        }

        List<Object> parseArray() {
            List<Object> list = new ArrayList<>();
            pos++; // skip '['
            skipWhitespace();
            if (pos < json.length() && json.charAt(pos) == ']') {
                pos++;
                return list;
            }
            while (pos < json.length()) {
                list.add(parseValue());
                skipWhitespace();
                if (pos < json.length() && json.charAt(pos) == ',') {
                    pos++;
                } else {
                    break;
                }
            }
            skipWhitespace();
            if (pos < json.length() && json.charAt(pos) == ']') pos++;
            return list;
        }

        String parseString() {
            pos++; // skip opening '"'
            StringBuilder sb = new StringBuilder();
            while (pos < json.length()) {
                char c = json.charAt(pos++);
                if (c == '"') return sb.toString();
                if (c == '\\' && pos < json.length()) {
                    char esc = json.charAt(pos++);
                    if (esc == '"') sb.append('"');
                    else if (esc == '\\') sb.append('\\');
                    else if (esc == '/') sb.append('/');
                    else if (esc == 'n') sb.append('\n');
                    else if (esc == 'r') sb.append('\r');
                    else if (esc == 't') sb.append('\t');
                    else if (esc == 'b') sb.append('\b');
                    else if (esc == 'f') sb.append('\f');
                    else if (esc == 'u') {
                        if (pos + 4 <= json.length()) {
                            String hex = json.substring(pos, pos + 4);
                            sb.append((char) Integer.parseInt(hex, 16));
                            pos += 4;
                        }
                    } else {
                        sb.append(esc);
                    }
                } else {
                    sb.append(c);
                }
            }
            return sb.toString();
        }

        Number parseNumber() {
            int start = pos;
            if (pos < json.length() && json.charAt(pos) == '-') pos++;
            while (pos < json.length() && Character.isDigit(json.charAt(pos))) pos++;
            boolean isFloat = false;
            if (pos < json.length() && json.charAt(pos) == '.') {
                isFloat = true;
                pos++;
                while (pos < json.length() && Character.isDigit(json.charAt(pos))) pos++;
            }
            if (pos < json.length() && (json.charAt(pos) == 'e' || json.charAt(pos) == 'E')) {
                isFloat = true;
                pos++;
                if (pos < json.length() && (json.charAt(pos) == '+' || json.charAt(pos) == '-')) pos++;
                while (pos < json.length() && Character.isDigit(json.charAt(pos))) pos++;
            }
            String numStr = json.substring(start, pos);
            if (isFloat) return Double.parseDouble(numStr);
            long val = Long.parseLong(numStr);
            if (val >= Integer.MIN_VALUE && val <= Integer.MAX_VALUE) return (int) val;
            return val;
        }

        Boolean parseBoolean() {
            if (json.startsWith("true", pos)) {
                pos += 4;
                return Boolean.TRUE;
            }
            if (json.startsWith("false", pos)) {
                pos += 5;
                return Boolean.FALSE;
            }
            throw new IllegalArgumentException("Expected boolean at pos " + pos);
        }

        Object parseNull() {
            if (json.startsWith("null", pos)) {
                pos += 4;
                return null;
            }
            throw new IllegalArgumentException("Expected null at pos " + pos);
        }

        void expect(char c) {
            skipWhitespace();
            if (pos < json.length() && json.charAt(pos) == c) {
                pos++;
            }
        }

        void skipWhitespace() {
            while (pos < json.length() && Character.isWhitespace(json.charAt(pos))) {
                pos++;
            }
        }
    }
}
