package it.denzosoft.llmplayer.cli;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Downloads GGUF model files from HuggingFace Hub.
 * Uses only JDK HTTP classes — zero external dependencies.
 *
 * Usage: --download "owner/repo" or --download "owner/repo/filename.gguf"
 *
 * If only repo is given, lists available GGUF files and lets user choose.
 * If filename is included, downloads that specific file.
 * Skips download if the file already exists locally with matching size.
 */
public class HuggingFaceDownloader {

    private static final String HF_API = "https://huggingface.co/api/models/";
    private static final String HF_RESOLVE = "https://huggingface.co/%s/resolve/main/%s";
    private static final int CONNECT_TIMEOUT = 15_000;
    private static final int READ_TIMEOUT = 30_000;
    private static final int BUFFER_SIZE = 1024 * 1024; // 1 MB

    private final String ggufDirectory;
    private final String hfToken;

    public HuggingFaceDownloader(String ggufDirectory, String hfToken) {
        this.ggufDirectory = ggufDirectory;
        this.hfToken = hfToken;
    }

    /**
     * Download a model from HuggingFace.
     * @param spec "owner/repo" or "owner/repo/filename.gguf"
     */
    public void download(String spec) throws IOException {
        // Parse spec: "owner/repo" or "owner/repo/path/to/file.gguf"
        String[] parts = spec.split("/", 3);
        if (parts.length < 2) {
            System.err.println("Error: invalid HuggingFace model spec. Use: owner/repo or owner/repo/filename.gguf");
            return;
        }

        String repo = parts[0] + "/" + parts[1];
        String filename = (parts.length == 3) ? parts[2] : null;

        if (filename != null && filename.endsWith(".gguf")) {
            // Direct download of specific file
            downloadFile(repo, filename);
        } else {
            // List GGUF files in the repo and download
            List<GGUFFileInfo> ggufFiles = listGGUFFiles(repo);
            if (ggufFiles.isEmpty()) {
                System.err.println("No GGUF files found in " + repo);
                return;
            }

            if (ggufFiles.size() == 1) {
                downloadFile(repo, ggufFiles.get(0).path);
            } else {
                // Show list and let user choose
                System.out.println("GGUF files in " + repo + ":");
                System.out.println();
                for (int i = 0; i < ggufFiles.size(); i++) {
                    GGUFFileInfo f = ggufFiles.get(i);
                    System.out.printf("  %2d. %-60s %s%n", i + 1, f.path, formatSize(f.size));
                }
                System.out.println();

                // Auto-select: pick the Q4_K_M variant if available, otherwise first
                int selected = 0;
                for (int i = 0; i < ggufFiles.size(); i++) {
                    String name = ggufFiles.get(i).path.toLowerCase();
                    if (name.contains("q4_k_m")) {
                        selected = i;
                        break;
                    }
                }

                System.out.println("Downloading: " + ggufFiles.get(selected).path);
                System.out.println("(Use --download \"" + repo + "/" + ggufFiles.get(selected).path + "\" to pick a specific file)");
                System.out.println();
                downloadFile(repo, ggufFiles.get(selected).path);
            }
        }
    }

    private List<GGUFFileInfo> listGGUFFiles(String repo) throws IOException {
        String apiUrl = HF_API + repo;
        HttpURLConnection conn = openConnection(apiUrl);
        conn.setRequestMethod("GET");

        int code = conn.getResponseCode();
        if (code != 200) {
            throw new IOException("HuggingFace API returned " + code + " for " + repo +
                ". Check the repository name.");
        }

        String json = readResponse(conn);
        conn.disconnect();

        // Parse siblings array from JSON (minimal parser — no external deps)
        return parseGGUFSiblings(json);
    }

    private void downloadFile(String repo, String filename) throws IOException {
        // Ensure target directory exists
        Path dir = Paths.get(ggufDirectory);
        Files.createDirectories(dir);

        // Target local path (flatten subdirectories: use only the filename)
        String localName = filename.contains("/") ? filename.substring(filename.lastIndexOf('/') + 1) : filename;
        Path target = dir.resolve(localName);

        // Check remote file size first
        String url = String.format(HF_RESOLVE, repo, filename);
        long remoteSize = getRemoteFileSize(url);

        // Skip if already downloaded with matching size
        if (Files.exists(target)) {
            long localSize = Files.size(target);
            if (remoteSize > 0 && localSize == remoteSize) {
                System.out.println("Already downloaded: " + target + " (" + formatSize(localSize) + ")");
                return;
            } else if (remoteSize > 0) {
                System.out.println("Existing file size mismatch (local: " + formatSize(localSize) +
                    ", remote: " + formatSize(remoteSize) + "). Re-downloading...");
            }
        }

        System.out.println("Downloading " + repo + "/" + filename);
        System.out.println("  -> " + target);
        if (remoteSize > 0) {
            System.out.println("  Size: " + formatSize(remoteSize));
        }
        System.out.println();

        // Download with progress
        HttpURLConnection conn = openConnection(url);
        conn.setRequestMethod("GET");
        conn.setReadTimeout(0); // no timeout for large files

        int code = conn.getResponseCode();
        // Follow redirects (HuggingFace CDN)
        if (code == 302 || code == 301) {
            String redirect = conn.getHeaderField("Location");
            conn.disconnect();
            conn = (HttpURLConnection) new URL(redirect).openConnection();
            conn.setConnectTimeout(CONNECT_TIMEOUT);
            conn.setReadTimeout(0);
            code = conn.getResponseCode();
        }

        if (code != 200) {
            conn.disconnect();
            throw new IOException("Download failed with HTTP " + code + " for " + url);
        }

        long contentLength = conn.getContentLengthLong();
        if (contentLength <= 0) contentLength = remoteSize;

        Path tempFile = dir.resolve(localName + ".downloading");
        try (InputStream in = conn.getInputStream();
             OutputStream out = new BufferedOutputStream(Files.newOutputStream(tempFile), BUFFER_SIZE)) {

            byte[] buf = new byte[BUFFER_SIZE];
            long downloaded = 0;
            int lastPercent = -1;
            long lastTime = System.currentTimeMillis();
            long lastBytes = 0;

            int read;
            while ((read = in.read(buf)) != -1) {
                out.write(buf, 0, read);
                downloaded += read;

                // Progress display
                long now = System.currentTimeMillis();
                if (contentLength > 0) {
                    int percent = (int) (downloaded * 100 / contentLength);
                    if (percent != lastPercent || now - lastTime > 2000) {
                        long elapsed = now - lastTime;
                        double speed = elapsed > 0 ? (downloaded - lastBytes) / 1024.0 / 1024.0 / (elapsed / 1000.0) : 0;
                        System.out.printf("\r  [%3d%%] %s / %s  %.1f MB/s",
                            percent, formatSize(downloaded), formatSize(contentLength), speed);
                        lastPercent = percent;
                        lastTime = now;
                        lastBytes = downloaded;
                    }
                } else if (now - lastTime > 2000) {
                    System.out.printf("\r  %s downloaded", formatSize(downloaded));
                    lastTime = now;
                }
            }
        }

        // Rename temp file to final
        Files.deleteIfExists(target);
        Files.move(tempFile, target);

        System.out.println();
        System.out.println("Download complete: " + target);
        conn.disconnect();
    }

    private long getRemoteFileSize(String url) {
        try {
            HttpURLConnection conn = openConnection(url);
            conn.setRequestMethod("HEAD");

            int code = conn.getResponseCode();
            if (code == 302 || code == 301) {
                String redirect = conn.getHeaderField("Location");
                conn.disconnect();
                conn = (HttpURLConnection) new URL(redirect).openConnection();
                conn.setRequestMethod("HEAD");
                conn.setConnectTimeout(CONNECT_TIMEOUT);
                conn.setReadTimeout(READ_TIMEOUT);
                conn.getResponseCode();
            }

            long size = conn.getContentLengthLong();
            conn.disconnect();
            return size;
        } catch (IOException e) {
            return -1;
        }
    }

    private HttpURLConnection openConnection(String url) throws IOException {
        HttpURLConnection conn = (HttpURLConnection) new URL(url).openConnection();
        conn.setConnectTimeout(CONNECT_TIMEOUT);
        conn.setReadTimeout(READ_TIMEOUT);
        conn.setInstanceFollowRedirects(true);
        conn.setRequestProperty("User-Agent", "LLMPlayer/1.5.0");
        if (hfToken != null && !hfToken.isEmpty()) {
            conn.setRequestProperty("Authorization", "Bearer " + hfToken);
        }
        return conn;
    }

    private String readResponse(HttpURLConnection conn) throws IOException {
        try (InputStream in = conn.getInputStream();
             BufferedReader reader = new BufferedReader(new InputStreamReader(in, "UTF-8"))) {
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
            return sb.toString();
        }
    }

    /**
     * Parse GGUF file entries from HuggingFace model API JSON response.
     * Minimal JSON parser — extracts "rfilename" and "size" from siblings array.
     */
    private List<GGUFFileInfo> parseGGUFSiblings(String json) {
        List<GGUFFileInfo> results = new ArrayList<>();

        // Find "siblings" array
        int siblingsIdx = json.indexOf("\"siblings\"");
        if (siblingsIdx < 0) return results;

        int arrayStart = json.indexOf('[', siblingsIdx);
        if (arrayStart < 0) return results;

        // Walk through sibling objects
        int pos = arrayStart;
        while (pos < json.length()) {
            int objStart = json.indexOf('{', pos);
            if (objStart < 0) break;
            int objEnd = json.indexOf('}', objStart);
            if (objEnd < 0) break;

            String obj = json.substring(objStart, objEnd + 1);

            // Extract rfilename
            String rfilename = extractJsonString(obj, "rfilename");
            if (rfilename != null && rfilename.endsWith(".gguf")) {
                long size = extractJsonLong(obj, "size");
                results.add(new GGUFFileInfo(rfilename, size));
            }

            pos = objEnd + 1;
            // Check if we've left the siblings array
            if (json.indexOf(']', arrayStart) > 0 && pos > json.indexOf(']', arrayStart)) break;
        }

        return results;
    }

    private String extractJsonString(String json, String key) {
        String search = "\"" + key + "\"";
        int idx = json.indexOf(search);
        if (idx < 0) return null;
        int colonIdx = json.indexOf(':', idx + search.length());
        if (colonIdx < 0) return null;
        int quoteStart = json.indexOf('"', colonIdx + 1);
        if (quoteStart < 0) return null;
        int quoteEnd = json.indexOf('"', quoteStart + 1);
        if (quoteEnd < 0) return null;
        return json.substring(quoteStart + 1, quoteEnd);
    }

    private long extractJsonLong(String json, String key) {
        String search = "\"" + key + "\"";
        int idx = json.indexOf(search);
        if (idx < 0) return -1;
        int colonIdx = json.indexOf(':', idx + search.length());
        if (colonIdx < 0) return -1;
        // Find start of number
        int numStart = colonIdx + 1;
        while (numStart < json.length() && (json.charAt(numStart) == ' ' || json.charAt(numStart) == '\t')) numStart++;
        int numEnd = numStart;
        while (numEnd < json.length() && Character.isDigit(json.charAt(numEnd))) numEnd++;
        if (numEnd == numStart) return -1;
        try {
            return Long.parseLong(json.substring(numStart, numEnd));
        } catch (NumberFormatException e) {
            return -1;
        }
    }

    private static String formatSize(long bytes) {
        if (bytes < 0) return "unknown";
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.1f KB", bytes / 1024.0);
        if (bytes < 1024L * 1024 * 1024) return String.format("%.1f MB", bytes / (1024.0 * 1024));
        return String.format("%.1f GB", bytes / (1024.0 * 1024 * 1024));
    }

    private static class GGUFFileInfo {
        final String path;
        final long size;

        GGUFFileInfo(String path, long size) {
            this.path = path;
            this.size = size;
        }
    }
}
