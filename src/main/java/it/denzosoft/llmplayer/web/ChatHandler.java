package it.denzosoft.llmplayer.web;

import com.sun.net.httpserver.HttpExchange;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * REST API handler for chat persistence with branching support.
 * Manages conversations stored as JSON files in a chats/ directory.
 */
public class ChatHandler {

    private final Path chatsDirectory;
    private final Object writeLock = new Object();

    public ChatHandler(String baseDirectory) {
        this.chatsDirectory = Paths.get(baseDirectory, "chats");
    }

    public void handle(HttpExchange exchange) throws IOException {
        String path = exchange.getRequestURI().getPath();
        String method = exchange.getRequestMethod();

        // CORS headers
        exchange.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        exchange.getResponseHeaders().set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
        exchange.getResponseHeaders().set("Access-Control-Allow-Headers", "Content-Type");

        if ("OPTIONS".equals(method)) {
            exchange.sendResponseHeaders(204, -1);
            exchange.close();
            return;
        }

        try {
            // Strip /api/chats prefix
            String subPath = path.length() > "/api/chats".length()
                ? path.substring("/api/chats".length()) : "";

            if (subPath.isEmpty() || "/".equals(subPath)) {
                // /api/chats
                if ("GET".equals(method)) listConversations(exchange);
                else if ("POST".equals(method)) createConversation(exchange);
                else methodNotAllowed(exchange);

            } else if (subPath.startsWith("/export/")) {
                // /api/chats/export/{id}
                String id = subPath.substring("/export/".length());
                if ("GET".equals(method)) exportConversation(exchange, id);
                else methodNotAllowed(exchange);

            } else {
                // /api/chats/{id}[/...]
                String rest = subPath.substring(1); // strip leading /
                int slash = rest.indexOf('/');

                if (slash == -1) {
                    // /api/chats/{id}
                    String id = rest;
                    if ("GET".equals(method)) getConversation(exchange, id);
                    else if ("DELETE".equals(method)) deleteConversation(exchange, id);
                    else methodNotAllowed(exchange);

                } else {
                    String id = rest.substring(0, slash);
                    String action = rest.substring(slash + 1);

                    if ("title".equals(action)) {
                        if ("PUT".equals(method)) updateTitle(exchange, id);
                        else methodNotAllowed(exchange);

                    } else if ("messages".equals(action)) {
                        if ("POST".equals(method)) addMessage(exchange, id);
                        else methodNotAllowed(exchange);

                    } else if (action.startsWith("messages/")) {
                        String msgId = action.substring("messages/".length());
                        if ("PUT".equals(method)) editMessage(exchange, id, msgId);
                        else methodNotAllowed(exchange);

                    } else if ("active-leaf".equals(action)) {
                        if ("PUT".equals(method)) updateActiveLeaf(exchange, id);
                        else methodNotAllowed(exchange);

                    } else if ("settings".equals(action)) {
                        if ("PUT".equals(method)) updateSettings(exchange, id);
                        else methodNotAllowed(exchange);

                    } else {
                        sendError(exchange, 404, "Not found");
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("Chat API error: " + e.getMessage());
            e.printStackTrace();
            try {
                sendError(exchange, 500, e.getMessage() != null ? e.getMessage() : "Internal error");
            } catch (IOException ignored) {}
        }
    }

    // --- Endpoints ---

    private void listConversations(HttpExchange exchange) throws IOException {
        List<Map<String, Object>> list = new ArrayList<>();
        if (Files.isDirectory(chatsDirectory)) {
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(chatsDirectory, "conv_*.json")) {
                for (Path file : stream) {
                    try {
                        Map<String, Object> conv = readConversation(fileToId(file));
                        Map<String, Object> summary = new LinkedHashMap<>();
                        summary.put("id", conv.get("id"));
                        summary.put("title", conv.get("title"));
                        summary.put("created", conv.get("created"));
                        summary.put("updated", conv.get("updated"));
                        @SuppressWarnings("unchecked")
                        Map<String, Object> messages = (Map<String, Object>) conv.get("messages");
                        summary.put("messageCount", messages != null ? messages.size() : 0);
                        list.add(summary);
                    } catch (IOException e) {
                        // Skip corrupted files
                    }
                }
            }
        }
        // Sort by updated descending (most recent first)
        list.sort(new Comparator<Map<String, Object>>() {
            @Override
            public int compare(Map<String, Object> a, Map<String, Object> b) {
                long ua = a.get("updated") instanceof Number ? ((Number) a.get("updated")).longValue() : 0;
                long ub = b.get("updated") instanceof Number ? ((Number) b.get("updated")).longValue() : 0;
                return Long.compare(ub, ua);
            }
        });
        sendJson(exchange, 200, list);
    }

    private void createConversation(HttpExchange exchange) throws IOException {
        String body = readBody(exchange);
        Map<String, Object> input = body.isEmpty() ? new LinkedHashMap<String, Object>()
            : ApiHandler.parseJson(body);

        long now = System.currentTimeMillis();
        String id = "conv_" + now;

        Map<String, Object> conv = new LinkedHashMap<>();
        conv.put("id", id);
        conv.put("title", input.containsKey("title") ? input.get("title") : "New Chat");
        conv.put("created", now);
        conv.put("updated", now);

        // Default settings
        Map<String, Object> settings = new LinkedHashMap<>();
        settings.put("temperature", 0.7);
        settings.put("maxTokens", 256);
        settings.put("topK", 40);
        settings.put("topP", 0.9);
        settings.put("repetitionPenalty", 1.1);
        settings.put("systemMessage", "");
        if (input.containsKey("settings")) {
            @SuppressWarnings("unchecked")
            Map<String, Object> inputSettings = (Map<String, Object>) input.get("settings");
            if (inputSettings != null) settings.putAll(inputSettings);
        }
        conv.put("settings", settings);

        conv.put("messages", new LinkedHashMap<String, Object>());
        conv.put("rootChildren", new ArrayList<String>());
        conv.put("activeLeafId", null);

        synchronized (writeLock) {
            writeConversation(id, conv);
        }
        sendJson(exchange, 201, conv);
    }

    private void getConversation(HttpExchange exchange, String id) throws IOException {
        if (!conversationExists(id)) {
            sendError(exchange, 404, "Conversation not found");
            return;
        }
        Map<String, Object> conv = readConversation(id);
        sendJson(exchange, 200, conv);
    }

    private void deleteConversation(HttpExchange exchange, String id) throws IOException {
        Path file = chatsDirectory.resolve(id + ".json");
        if (!Files.exists(file)) {
            sendError(exchange, 404, "Conversation not found");
            return;
        }
        synchronized (writeLock) {
            Files.delete(file);
        }
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("status", "deleted");
        sendJson(exchange, 200, result);
    }

    private void updateTitle(HttpExchange exchange, String id) throws IOException {
        if (!conversationExists(id)) {
            sendError(exchange, 404, "Conversation not found");
            return;
        }
        Map<String, Object> body = ApiHandler.parseJson(readBody(exchange));
        String title = (String) body.get("title");
        if (title == null || title.trim().isEmpty()) {
            sendError(exchange, 400, "Missing 'title'");
            return;
        }

        synchronized (writeLock) {
            Map<String, Object> conv = readConversation(id);
            conv.put("title", title.trim());
            conv.put("updated", System.currentTimeMillis());
            writeConversation(id, conv);
            sendJson(exchange, 200, conv);
        }
    }

    @SuppressWarnings("unchecked")
    private void addMessage(HttpExchange exchange, String id) throws IOException {
        if (!conversationExists(id)) {
            sendError(exchange, 404, "Conversation not found");
            return;
        }
        Map<String, Object> body = ApiHandler.parseJson(readBody(exchange));
        String role = (String) body.get("role");
        String content = body.containsKey("content") ? String.valueOf(body.get("content")) : "";
        String parentId = (String) body.get("parentId");

        if (role == null || (!role.equals("user") && !role.equals("assistant"))) {
            sendError(exchange, 400, "Invalid role (must be 'user' or 'assistant')");
            return;
        }

        synchronized (writeLock) {
            Map<String, Object> conv = readConversation(id);
            Map<String, Object> messages = (Map<String, Object>) conv.get("messages");
            List<String> rootChildren = (List<String>) conv.get("rootChildren");

            // Generate new message ID
            String msgId = nextMessageId(messages);

            Map<String, Object> msg = new LinkedHashMap<>();
            msg.put("id", msgId);
            msg.put("role", role);
            msg.put("content", content);
            msg.put("parentId", parentId);
            msg.put("children", new ArrayList<String>());
            msg.put("timestamp", System.currentTimeMillis());

            // Copy stats if provided (for assistant messages)
            if (body.containsKey("stats")) {
                msg.put("stats", body.get("stats"));
            }

            // Link to parent
            if (parentId != null && messages.containsKey(parentId)) {
                Map<String, Object> parent = (Map<String, Object>) messages.get(parentId);
                List<String> children = (List<String>) parent.get("children");
                children.add(msgId);
            } else {
                rootChildren.add(msgId);
            }

            messages.put(msgId, msg);
            conv.put("activeLeafId", msgId);
            conv.put("updated", System.currentTimeMillis());

            // Auto-title from first user message
            if ("user".equals(role) && "New Chat".equals(conv.get("title"))) {
                String title = content.length() > 50 ? content.substring(0, 50) + "..." : content;
                title = title.replace("\n", " ").trim();
                if (!title.isEmpty()) conv.put("title", title);
            }

            writeConversation(id, conv);

            Map<String, Object> result = new LinkedHashMap<>();
            result.put("message", msg);
            result.put("conversationId", id);
            sendJson(exchange, 201, result);
        }
    }

    @SuppressWarnings("unchecked")
    private void editMessage(HttpExchange exchange, String convId, String msgId) throws IOException {
        if (!conversationExists(convId)) {
            sendError(exchange, 404, "Conversation not found");
            return;
        }
        Map<String, Object> body = ApiHandler.parseJson(readBody(exchange));
        String newContent = (String) body.get("content");
        if (newContent == null) {
            sendError(exchange, 400, "Missing 'content'");
            return;
        }

        synchronized (writeLock) {
            Map<String, Object> conv = readConversation(convId);
            Map<String, Object> messages = (Map<String, Object>) conv.get("messages");
            List<String> rootChildren = (List<String>) conv.get("rootChildren");

            Map<String, Object> original = (Map<String, Object>) messages.get(msgId);
            if (original == null) {
                sendError(exchange, 404, "Message not found");
                return;
            }

            // Create sibling message with same parentId
            String newId = nextMessageId(messages);
            String parentId = (String) original.get("parentId");

            Map<String, Object> newMsg = new LinkedHashMap<>();
            newMsg.put("id", newId);
            newMsg.put("role", original.get("role"));
            newMsg.put("content", newContent);
            newMsg.put("parentId", parentId);
            newMsg.put("children", new ArrayList<String>());
            newMsg.put("timestamp", System.currentTimeMillis());

            // Link to parent (as sibling of original)
            if (parentId != null && messages.containsKey(parentId)) {
                Map<String, Object> parent = (Map<String, Object>) messages.get(parentId);
                List<String> children = (List<String>) parent.get("children");
                children.add(newId);
            } else {
                rootChildren.add(newId);
            }

            messages.put(newId, newMsg);
            conv.put("activeLeafId", newId);
            conv.put("updated", System.currentTimeMillis());

            writeConversation(convId, conv);

            Map<String, Object> result = new LinkedHashMap<>();
            result.put("message", newMsg);
            result.put("conversationId", convId);
            sendJson(exchange, 200, result);
        }
    }

    private void updateActiveLeaf(HttpExchange exchange, String id) throws IOException {
        if (!conversationExists(id)) {
            sendError(exchange, 404, "Conversation not found");
            return;
        }
        Map<String, Object> body = ApiHandler.parseJson(readBody(exchange));
        String leafId = (String) body.get("activeLeafId");

        synchronized (writeLock) {
            Map<String, Object> conv = readConversation(id);
            conv.put("activeLeafId", leafId);
            conv.put("updated", System.currentTimeMillis());
            writeConversation(id, conv);

            Map<String, Object> result = new LinkedHashMap<>();
            result.put("activeLeafId", leafId);
            sendJson(exchange, 200, result);
        }
    }

    private void updateSettings(HttpExchange exchange, String id) throws IOException {
        if (!conversationExists(id)) {
            sendError(exchange, 404, "Conversation not found");
            return;
        }
        Map<String, Object> body = ApiHandler.parseJson(readBody(exchange));

        synchronized (writeLock) {
            Map<String, Object> conv = readConversation(id);
            @SuppressWarnings("unchecked")
            Map<String, Object> settings = (Map<String, Object>) conv.get("settings");
            if (settings == null) {
                settings = new LinkedHashMap<>();
                conv.put("settings", settings);
            }
            settings.putAll(body);
            conv.put("updated", System.currentTimeMillis());
            writeConversation(id, conv);

            Map<String, Object> result = new LinkedHashMap<>();
            result.put("settings", settings);
            sendJson(exchange, 200, result);
        }
    }

    private void exportConversation(HttpExchange exchange, String id) throws IOException {
        if (!conversationExists(id)) {
            sendError(exchange, 404, "Conversation not found");
            return;
        }
        Map<String, Object> conv = readConversation(id);
        String json = ApiHandler.toJson(conv);
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.getResponseHeaders().set("Content-Disposition",
            "attachment; filename=\"" + id + ".json\"");
        exchange.sendResponseHeaders(200, bytes.length);
        exchange.getResponseBody().write(bytes);
        exchange.getResponseBody().close();
    }

    // --- I/O Helpers ---

    private Map<String, Object> readConversation(String id) throws IOException {
        Path file = chatsDirectory.resolve(id + ".json");
        byte[] bytes = Files.readAllBytes(file);
        String json = new String(bytes, StandardCharsets.UTF_8);
        return ApiHandler.parseJson(json);
    }

    private void writeConversation(String id, Map<String, Object> conv) throws IOException {
        if (!Files.isDirectory(chatsDirectory)) {
            Files.createDirectories(chatsDirectory);
        }
        Path target = chatsDirectory.resolve(id + ".json");
        Path tmp = chatsDirectory.resolve(id + ".json.tmp");
        byte[] bytes = ApiHandler.toJson(conv).getBytes(StandardCharsets.UTF_8);
        Files.write(tmp, bytes);
        Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }

    private boolean conversationExists(String id) {
        return Files.exists(chatsDirectory.resolve(id + ".json"));
    }

    private String fileToId(Path file) {
        String name = file.getFileName().toString();
        return name.substring(0, name.length() - ".json".length());
    }

    @SuppressWarnings("unchecked")
    private String nextMessageId(Map<String, Object> messages) {
        int max = 0;
        for (String key : messages.keySet()) {
            if (key.startsWith("msg_")) {
                try {
                    int n = Integer.parseInt(key.substring(4));
                    if (n > max) max = n;
                } catch (NumberFormatException ignored) {}
            }
        }
        return "msg_" + (max + 1);
    }

    // --- HTTP Helpers ---

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
        byte[] bytes = ApiHandler.toJson(data).getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.sendResponseHeaders(status, bytes.length);
        exchange.getResponseBody().write(bytes);
        exchange.getResponseBody().close();
    }

    private void sendError(HttpExchange exchange, int status, String message) throws IOException {
        Map<String, Object> err = new LinkedHashMap<>();
        err.put("error", message);
        sendJson(exchange, status, err);
    }

    private void methodNotAllowed(HttpExchange exchange) throws IOException {
        sendError(exchange, 405, "Method not allowed");
    }
}
