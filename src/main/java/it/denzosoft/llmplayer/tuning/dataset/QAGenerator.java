package it.denzosoft.llmplayer.tuning.dataset;

import it.denzosoft.llmplayer.api.GenerationRequest;
import it.denzosoft.llmplayer.api.GenerationResponse;
import it.denzosoft.llmplayer.api.LLMEngine;
import it.denzosoft.llmplayer.sampler.SamplerConfig;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * Generates Q&A pairs from chunks using an LLMEngine as the generator.
 * Supports progressive checkpointing for suspend/resume.
 */
public class QAGenerator {

    private final LLMEngine engine;
    private final String dataType;
    private final int pairsPerChunk;
    private final int maxResponseTokens;
    private final int embeddingLength;

    public QAGenerator(LLMEngine engine, String dataType, int pairsPerChunk,
                       int maxResponseTokens, int embeddingLength) {
        this.engine = engine;
        this.dataType = dataType;
        this.pairsPerChunk = pairsPerChunk;
        this.maxResponseTokens = maxResponseTokens;
        this.embeddingLength = embeddingLength;
    }

    /**
     * Generate Q&A pairs for all chunks, appending to the output file.
     * Skips the first {@code skipChunks} chunks (for resume support).
     *
     * @param chunks       all chunks to process
     * @param outputFile   JSONL output file (appended to)
     * @param skipChunks   number of chunks already processed (resume offset)
     * @param listener     progress callback (may be null)
     * @return total number of Q&A pairs generated in this run
     */
    public int generate(List<Chunk> chunks, Path outputFile, int skipChunks,
                        ProgressListener listener) throws IOException {
        int totalPairs = 0;

        try (BufferedWriter writer = Files.newBufferedWriter(outputFile,
                StandardCharsets.UTF_8,
                skipChunks > 0 ? StandardOpenOption.APPEND : StandardOpenOption.CREATE,
                StandardOpenOption.WRITE)) {

            for (int i = skipChunks; i < chunks.size(); i++) {
                Chunk chunk = chunks.get(i);

                if (listener != null) {
                    listener.onProgress(i, chunks.size(), totalPairs);
                }

                String prompt = PromptTemplates.buildPrompt(chunk, dataType,
                    pairsPerChunk, embeddingLength);

                // Generate with low temperature for structured JSON output
                SamplerConfig sampler = new SamplerConfig(0.3f, 40, 0.9f, 1.1f, System.nanoTime());
                GenerationRequest request = GenerationRequest.builder()
                    .prompt(prompt)
                    .maxTokens(maxResponseTokens)
                    .samplerConfig(sampler)
                    .useChat(true)
                    .build();

                GenerationResponse response = engine.generate(request);
                String output = response.text();

                // Debug: show raw output (first 500 chars) and parsed pair count
                String preview = output.length() > 500 ? output.substring(0, 500) + "..." : output;
                System.out.printf("  [DEBUG] Raw output (%d chars): %s%n",
                        output.length(), preview.replace("\n", "\\n"));

                // Parse JSON array of Q&A pairs from the output
                List<QAPair> pairs = parseQAPairs(output);
                System.out.printf("  [DEBUG] Parsed %d Q&A pairs%n", pairs.size());

                // Write each pair as a JSONL line in chat format
                for (QAPair pair : pairs) {
                    String jsonLine = formatAsTrainingLine(pair.question, pair.answer);
                    writer.write(jsonLine);
                    writer.newLine();
                    totalPairs++;
                }
                writer.flush();

                if (listener != null) {
                    listener.onChunkCompleted(i, chunks.size(), pairs.size(), totalPairs);
                }
            }
        }

        return totalPairs;
    }

    /** Parse the generator's JSON output into Q&A pairs. Tolerant of formatting issues. */
    List<QAPair> parseQAPairs(String output) {
        List<QAPair> pairs = new ArrayList<>();
        // Find the JSON array in the output (may have surrounding text)
        int start = output.indexOf('[');
        int end = output.lastIndexOf(']');
        if (start < 0 || end < 0 || end <= start) {
            // Fallback: try to extract Q&A from plain text
            return parsePlainTextQA(output);
        }

        String json = output.substring(start, end + 1);
        // Simple parsing of [{q:..., a:...}, ...] without a full JSON parser
        // Split on }, { boundaries
        int depth = 0;
        int objStart = -1;
        for (int i = 0; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == '{') {
                if (depth == 0) objStart = i;
                depth++;
            } else if (c == '}') {
                depth--;
                if (depth == 0 && objStart >= 0) {
                    String obj = json.substring(objStart, i + 1);
                    QAPair pair = parseOneQA(obj);
                    if (pair != null) pairs.add(pair);
                    objStart = -1;
                }
            }
        }
        return pairs;
    }

    private QAPair parseOneQA(String obj) {
        // Strategy: find the q/question field, then the a/answer field.
        // Extract the value text between them using positional parsing,
        // which is more robust than trying to parse malformed JSON field-by-field.
        String q = null, a = null;

        // Find q/question key position
        int qKeyEnd = findKeyEnd(obj, "q");
        if (qKeyEnd < 0) qKeyEnd = findKeyEnd(obj, "question");

        // Find a/answer key position
        int aKeyEnd = findKeyEnd(obj, "a");
        if (aKeyEnd < 0) aKeyEnd = findKeyEnd(obj, "answer");

        if (qKeyEnd >= 0 && aKeyEnd >= 0 && qKeyEnd < aKeyEnd) {
            // q value is between qKeyEnd and aKeyEnd, a value is between aKeyEnd and }
            q = cleanValue(obj.substring(qKeyEnd, aKeyEnd));
            a = cleanValue(obj.substring(aKeyEnd));
        } else if (qKeyEnd >= 0) {
            // Only q found — try extracting both from remaining text
            q = extractJsonString(obj, "q");
            if (q == null) q = extractJsonString(obj, "question");
            a = extractJsonString(obj, "a");
            if (a == null) a = extractJsonString(obj, "answer");
        }

        if (q != null && a != null && !q.isEmpty() && !a.isEmpty()) {
            return new QAPair(q, a);
        }
        return null;
    }

    /** Find position right after "key": (or key:) colon+whitespace. Returns -1 if not found. */
    private int findKeyEnd(String obj, String key) {
        // Try quoted key: "key"
        String[] patterns = {"\"" + key + "\"", key};
        for (String pat : patterns) {
            int idx = obj.indexOf(pat);
            if (idx < 0) continue;
            int afterPat = idx + pat.length();
            // Skip whitespace, find colon
            while (afterPat < obj.length() && obj.charAt(afterPat) == ' ') afterPat++;
            if (afterPat < obj.length() && obj.charAt(afterPat) == ':') {
                afterPat++; // skip colon
                while (afterPat < obj.length() && obj.charAt(afterPat) == ' ') afterPat++;
                return afterPat;
            }
        }
        return -1;
    }

    /** Clean a raw value substring: strip quotes, trailing commas, key prefixes. */
    private String cleanValue(String raw) {
        String s = raw.trim();
        // Remove leading quote
        if (s.startsWith("\"")) s = s.substring(1);
        // Remove trailing content after the value: look for patterns that start the next field
        // Look for   ,"   or   ,\n   or   "\n   or ending   }  or   ]
        // Strategy: find the last meaningful content before a key-like pattern
        // Remove everything from the last  "  that precedes a comma/brace/key-pattern
        int end = findValueEnd(s);
        if (end > 0 && end < s.length()) s = s.substring(0, end);
        s = s.trim();
        // Strip trailing punctuation artifacts
        while (s.endsWith(",") || s.endsWith("}") || s.endsWith("]"))
            s = s.substring(0, s.length() - 1).trim();
        // Strip trailing quote
        if (s.endsWith("\"")) s = s.substring(0, s.length() - 1);
        return s.trim();
    }

    /** Find where the value text ends in a raw substring. */
    private int findValueEnd(String s) {
        // If value is properly quoted, find the closing quote
        // that is followed by comma, whitespace+key, or brace
        boolean inQuote = false;
        int lastQuote = -1;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '\\' && i + 1 < s.length()) { i++; continue; }
            if (c == '"') {
                if (!inQuote) {
                    // Check if this looks like a key start: "q" or "a" preceded by comma/whitespace
                    if (i > 0 && (s.charAt(i - 1) == ',' || s.charAt(i - 1) == ' '
                            || s.charAt(i - 1) == '\n' || s.charAt(i - 1) == '\t')) {
                        // Check if followed by a single-char key + quote + colon
                        if (i + 2 < s.length() && s.charAt(i + 2) == '"') {
                            int colonPos = i + 3;
                            while (colonPos < s.length() && s.charAt(colonPos) == ' ') colonPos++;
                            if (colonPos < s.length() && s.charAt(colonPos) == ':') {
                                return i - 1; // end before the comma/whitespace
                            }
                        }
                    }
                }
                lastQuote = i;
            }
        }
        return s.length();
    }

    /** Extract a string value for a given key from a JSON-like object string.
     *  Tolerant of LLM quirks: handles unquoted keys/values. */
    private String extractJsonString(String obj, String key) {
        // Try quoted key first: "key": or "key" :
        String quotedKey = "\"" + key + "\"";
        int idx = obj.indexOf(quotedKey);
        // Fallback: unquoted key like   a:  (preceded by comma/brace + whitespace)
        if (idx < 0) {
            int searchFrom = 0;
            while (searchFrom < obj.length()) {
                int bare = obj.indexOf(key, searchFrom);
                if (bare < 0) return null;
                // Check it's a standalone key: preceded by {, comma, or whitespace
                // and followed by optional whitespace then :
                boolean validStart = bare == 0 || " \t\n,{".indexOf(obj.charAt(bare - 1)) >= 0;
                int afterKey = bare + key.length();
                // Skip whitespace after key name
                while (afterKey < obj.length() && obj.charAt(afterKey) == ' ') afterKey++;
                boolean validEnd = afterKey < obj.length() && obj.charAt(afterKey) == ':';
                if (validStart && validEnd) {
                    idx = bare;
                    break;
                }
                searchFrom = bare + 1;
            }
        }
        if (idx < 0) return null;

        // Find the colon after the key
        int colon = obj.indexOf(':', idx + key.length());
        if (colon < 0) return null;

        // Skip whitespace after colon
        int afterColon = colon + 1;
        while (afterColon < obj.length() && obj.charAt(afterColon) == ' ') afterColon++;
        if (afterColon >= obj.length()) return null;

        if (obj.charAt(afterColon) == '"') {
            // Quoted value — parse with escape handling
            StringBuilder sb = new StringBuilder();
            for (int i = afterColon + 1; i < obj.length(); i++) {
                char c = obj.charAt(i);
                if (c == '\\' && i + 1 < obj.length()) {
                    char next = obj.charAt(i + 1);
                    if (next == '"') { sb.append('"'); i++; }
                    else if (next == 'n') { sb.append('\n'); i++; }
                    else if (next == '\\') { sb.append('\\'); i++; }
                    else if (next == 't') { sb.append('\t'); i++; }
                    else { sb.append(c); }
                } else if (c == '"') {
                    return sb.toString();
                } else {
                    sb.append(c);
                }
            }
            return sb.toString();
        } else {
            // Unquoted value — read until next "key": pattern or } or end
            // Find the end: look for next comma followed by a key, or closing brace
            StringBuilder sb = new StringBuilder();
            int depth = 0;
            for (int i = afterColon; i < obj.length(); i++) {
                char c = obj.charAt(i);
                if (c == '{' || c == '[') depth++;
                else if (c == '}' || c == ']') {
                    if (depth == 0) break;
                    depth--;
                }
                sb.append(c);
            }
            String val = sb.toString().trim();
            // Strip trailing comma
            if (val.endsWith(",")) val = val.substring(0, val.length() - 1).trim();
            // Strip surrounding quotes that might be partial
            if (val.startsWith("\"")) val = val.substring(1);
            if (val.endsWith("\"")) val = val.substring(0, val.length() - 1);
            return val.isEmpty() ? null : val;
        }
    }

    private List<QAPair> parsePlainTextQA(String text) {
        // Last resort: try to find Q: ... A: ... patterns
        List<QAPair> pairs = new ArrayList<>();
        String[] lines = text.split("\n");
        String currentQ = null;
        StringBuilder currentA = new StringBuilder();

        for (String line : lines) {
            String trimmed = line.trim();
            if (trimmed.startsWith("Q:") || trimmed.startsWith("Question:")) {
                if (currentQ != null && currentA.length() > 0) {
                    pairs.add(new QAPair(currentQ, currentA.toString().trim()));
                }
                currentQ = trimmed.substring(trimmed.indexOf(':') + 1).trim();
                currentA = new StringBuilder();
            } else if (trimmed.startsWith("A:") || trimmed.startsWith("Answer:")) {
                currentA.append(trimmed.substring(trimmed.indexOf(':') + 1).trim());
            } else if (currentQ != null) {
                currentA.append(" ").append(trimmed);
            }
        }
        if (currentQ != null && currentA.length() > 0) {
            pairs.add(new QAPair(currentQ, currentA.toString().trim()));
        }
        return pairs;
    }

    /** Format a Q&A pair as a JSONL training line with messages array. */
    private String formatAsTrainingLine(String question, String answer) {
        // Escape for JSON
        String q = escapeJson(question);
        String a = escapeJson(answer);
        return "{\"messages\":[{\"role\":\"user\",\"content\":\"" + q
            + "\"},{\"role\":\"assistant\",\"content\":\"" + a + "\"}]}";
    }

    private String escapeJson(String s) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '"') sb.append("\\\"");
            else if (c == '\\') sb.append("\\\\");
            else if (c == '\n') sb.append("\\n");
            else if (c == '\r') sb.append("\\r");
            else if (c == '\t') sb.append("\\t");
            else if (c < 0x20) sb.append(String.format("\\u%04x", (int) c));
            else sb.append(c);
        }
        return sb.toString();
    }

    // --- Inner types ---

    static class QAPair {
        final String question;
        final String answer;
        QAPair(String question, String answer) {
            this.question = question;
            this.answer = answer;
        }
    }

    public interface ProgressListener {
        void onProgress(int chunkIndex, int totalChunks, int pairsGenerated);
        void onChunkCompleted(int chunkIndex, int totalChunks, int pairsInChunk, int totalPairs);
    }
}
