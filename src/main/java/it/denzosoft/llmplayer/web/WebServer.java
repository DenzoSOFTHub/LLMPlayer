package it.denzosoft.llmplayer.web;

import com.sun.net.httpserver.HttpServer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.Executors;

/**
 * Embedded HTTP server for the LLMPlayer Web UI.
 * Uses JDK built-in com.sun.net.httpserver.HttpServer.
 */
public class WebServer {

    private final int port;
    private final String ggufDirectory;
    private HttpServer server;
    private ApiHandler apiHandler;
    private OpenAIHandler openAIHandler;

    public WebServer(int port, String ggufDirectory) {
        this.port = port;
        this.ggufDirectory = ggufDirectory;
    }

    /** Start the server (non-blocking). */
    public void start() throws IOException {
        apiHandler = new ApiHandler(ggufDirectory);
        openAIHandler = new OpenAIHandler(apiHandler);
        String html = loadHtml();

        server = HttpServer.create(new InetSocketAddress(port), 0);

        server.createContext("/", exchange -> {
            String path = exchange.getRequestURI().getPath();
            if ("/".equals(path) || "/index.html".equals(path)) {
                byte[] response = html.getBytes(StandardCharsets.UTF_8);
                exchange.getResponseHeaders().set("Content-Type", "text/html; charset=utf-8");
                exchange.sendResponseHeaders(200, response.length);
                exchange.getResponseBody().write(response);
                exchange.getResponseBody().close();
            } else if (path.startsWith("/v1/")) {
                openAIHandler.handle(exchange);
            } else if (path.startsWith("/api/")) {
                apiHandler.handle(exchange);
            } else {
                exchange.sendResponseHeaders(404, -1);
                exchange.close();
            }
        });

        server.setExecutor(Executors.newCachedThreadPool());
        server.start();

        System.out.println("Web API server started on http://localhost:" + port);
    }

    /** Start the server and block the calling thread (for CLI --web mode). */
    public void startBlocking() throws IOException {
        start();

        System.out.println("===========================================");
        System.out.println("  LLMPlayer Web UI");
        System.out.println("  http://localhost:" + port);
        System.out.println("===========================================");
        System.out.println("Press Ctrl+C to stop the server.");

        Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
            @Override
            public void run() {
                System.out.println("\nShutting down server...");
                stop();
            }
        }));

        try {
            Thread.currentThread().join();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    public void stop() {
        if (apiHandler != null) {
            apiHandler.shutdown();
            apiHandler = null;
        }
        if (server != null) {
            server.stop(1);
            server = null;
        }
    }

    public boolean isRunning() {
        return server != null;
    }

    public int getPort() {
        return port;
    }

    private String loadHtml() throws IOException {
        try (InputStream is = getClass().getResourceAsStream("/web-ui.html")) {
            if (is == null) {
                throw new IOException("web-ui.html not found in resources");
            }
            return new String(readAllBytes(is), StandardCharsets.UTF_8);
        }
    }

    private static byte[] readAllBytes(InputStream is) throws IOException {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        byte[] tmp = new byte[8192];
        int n;
        while ((n = is.read(tmp)) != -1) {
            buffer.write(tmp, 0, n);
        }
        return buffer.toByteArray();
    }
}
