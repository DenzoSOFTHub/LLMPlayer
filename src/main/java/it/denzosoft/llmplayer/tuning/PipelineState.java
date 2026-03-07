package it.denzosoft.llmplayer.tuning;

import it.denzosoft.llmplayer.web.ApiHandler;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * Persistent pipeline state for checkpoint/resume support.
 * Serialized as JSON to {workDir}/pipeline.json.
 */
public class PipelineState {

    private int currentStage;       // 1-6
    private int chunksCompleted;    // stage 3 progress
    private int totalChunks;        // stage 3 total
    private int epochsCompleted;    // stage 4 progress
    private int totalEpochs;        // stage 4 total
    private long startTime;
    private long lastUpdateTime;
    private String status;          // running, suspended, completed, error

    // Stored analysis results (from stage 1)
    private String targetArchitecture;
    private int targetContextLength;
    private int targetEmbeddingLength;
    private int maxChunkTokens;
    private int maxResponseTokens;

    public PipelineState() {
        this.currentStage = 1;
        this.status = "running";
        this.startTime = System.currentTimeMillis();
        this.lastUpdateTime = this.startTime;
    }

    // --- Persistence ---

    public void save(Path workDir) throws IOException {
        lastUpdateTime = System.currentTimeMillis();
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("currentStage", currentStage);
        map.put("chunksCompleted", chunksCompleted);
        map.put("totalChunks", totalChunks);
        map.put("epochsCompleted", epochsCompleted);
        map.put("totalEpochs", totalEpochs);
        map.put("startTime", startTime);
        map.put("lastUpdateTime", lastUpdateTime);
        map.put("status", status);
        map.put("targetArchitecture", targetArchitecture);
        map.put("targetContextLength", targetContextLength);
        map.put("targetEmbeddingLength", targetEmbeddingLength);
        map.put("maxChunkTokens", maxChunkTokens);
        map.put("maxResponseTokens", maxResponseTokens);

        if (!Files.isDirectory(workDir)) Files.createDirectories(workDir);
        Path target = workDir.resolve("pipeline.json");
        Path tmp = workDir.resolve("pipeline.json.tmp");
        byte[] bytes = ApiHandler.toJson(map).getBytes(StandardCharsets.UTF_8);
        Files.write(tmp, bytes);
        Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }

    public static PipelineState load(Path workDir) throws IOException {
        Path file = workDir.resolve("pipeline.json");
        if (!Files.exists(file)) return null;
        byte[] bytes = Files.readAllBytes(file);
        Map<String, Object> map = ApiHandler.parseJson(new String(bytes, StandardCharsets.UTF_8));

        PipelineState s = new PipelineState();
        s.currentStage = getInt(map, "currentStage", 1);
        s.chunksCompleted = getInt(map, "chunksCompleted", 0);
        s.totalChunks = getInt(map, "totalChunks", 0);
        s.epochsCompleted = getInt(map, "epochsCompleted", 0);
        s.totalEpochs = getInt(map, "totalEpochs", 0);
        s.startTime = getLong(map, "startTime", System.currentTimeMillis());
        s.lastUpdateTime = getLong(map, "lastUpdateTime", s.startTime);
        s.status = (String) map.get("status");
        s.targetArchitecture = (String) map.get("targetArchitecture");
        s.targetContextLength = getInt(map, "targetContextLength", 2048);
        s.targetEmbeddingLength = getInt(map, "targetEmbeddingLength", 0);
        s.maxChunkTokens = getInt(map, "maxChunkTokens", 0);
        s.maxResponseTokens = getInt(map, "maxResponseTokens", 0);
        return s;
    }

    public static boolean hasCheckpoint(Path workDir) {
        return Files.exists(workDir.resolve("pipeline.json"));
    }

    // --- Getters/Setters ---

    public int currentStage() { return currentStage; }
    public void setCurrentStage(int v) { currentStage = v; }

    public int chunksCompleted() { return chunksCompleted; }
    public void setChunksCompleted(int v) { chunksCompleted = v; }

    public int totalChunks() { return totalChunks; }
    public void setTotalChunks(int v) { totalChunks = v; }

    public int epochsCompleted() { return epochsCompleted; }
    public void setEpochsCompleted(int v) { epochsCompleted = v; }

    public int totalEpochs() { return totalEpochs; }
    public void setTotalEpochs(int v) { totalEpochs = v; }

    public long startTime() { return startTime; }
    public long lastUpdateTime() { return lastUpdateTime; }

    public String status() { return status; }
    public void setStatus(String v) { status = v; }

    public String targetArchitecture() { return targetArchitecture; }
    public void setTargetArchitecture(String v) { targetArchitecture = v; }

    public int targetContextLength() { return targetContextLength; }
    public void setTargetContextLength(int v) { targetContextLength = v; }

    public int targetEmbeddingLength() { return targetEmbeddingLength; }
    public void setTargetEmbeddingLength(int v) { targetEmbeddingLength = v; }

    public int maxChunkTokens() { return maxChunkTokens; }
    public void setMaxChunkTokens(int v) { maxChunkTokens = v; }

    public int maxResponseTokens() { return maxResponseTokens; }
    public void setMaxResponseTokens(int v) { maxResponseTokens = v; }

    public String elapsedFormatted() {
        long elapsed = lastUpdateTime - startTime;
        long hours = elapsed / 3_600_000;
        long minutes = (elapsed % 3_600_000) / 60_000;
        return hours > 0 ? hours + "h " + minutes + "m" : minutes + "m";
    }

    // --- Helpers ---

    private static int getInt(Map<String, Object> m, String key, int def) {
        Object v = m.get(key);
        return v instanceof Number ? ((Number) v).intValue() : def;
    }

    private static long getLong(Map<String, Object> m, String key, long def) {
        Object v = m.get(key);
        return v instanceof Number ? ((Number) v).longValue() : def;
    }
}
