package it.denzosoft.llmplayer.tuning.dataset;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * Chunks structured data (CSV/JSON export + optional SQL schema) for Q&A generation.
 * Produces chunks combining schema info, data samples, and statistics.
 */
public class DataChunker {

    private final int maxChunkTokens;
    private final TokenCounter tokenCounter;

    public DataChunker(int maxChunkTokens, int overlapTokens, TokenCounter tokenCounter) {
        this.maxChunkTokens = maxChunkTokens;
        this.tokenCounter = tokenCounter;
    }

    /**
     * Chunk a data file, optionally enriched with a schema file.
     * @param dataFile CSV or JSON data file
     * @param schemaFile optional SQL DDL file (may be null)
     */
    public List<Chunk> chunk(Path dataFile, Path schemaFile) throws IOException {
        List<Chunk> chunks = new ArrayList<>();

        String schema = schemaFile != null && Files.exists(schemaFile)
            ? new String(Files.readAllBytes(schemaFile), StandardCharsets.UTF_8) : "";

        List<String> lines = Files.readAllLines(dataFile, StandardCharsets.UTF_8);
        if (lines.isEmpty()) return chunks;

        String fileName = dataFile.getFileName().toString();
        boolean isCsv = fileName.endsWith(".csv") || fileName.endsWith(".tsv");
        String separator = fileName.endsWith(".tsv") ? "\t" : ",";

        String header = "";
        List<String> dataLines = lines;
        String[] columnNames = new String[0];

        if (isCsv && lines.size() > 1) {
            header = lines.get(0);
            columnNames = splitCsvLine(header, separator);
            dataLines = lines.subList(1, lines.size());
        }

        // Compute basic statistics
        String stats = computeStats(columnNames, dataLines, separator);

        // Chunk 1: Schema + overview + statistics
        int chunkIdx = 0;
        StringBuilder schemaChunk = new StringBuilder();
        if (!schema.isEmpty()) {
            schemaChunk.append("SQL Schema:\n").append(schema).append("\n\n");
        }
        schemaChunk.append("Data file: ").append(fileName).append("\n");
        schemaChunk.append("Total rows: ").append(dataLines.size()).append("\n");
        if (header.length() > 0) {
            schemaChunk.append("Columns: ").append(header).append("\n");
        }
        schemaChunk.append("\n").append(stats);

        chunks.add(new Chunk("chunk_" + (++chunkIdx), schemaChunk.toString(), fileName,
            "Schema & Statistics", "schema", tokenCounter.countTokens(schemaChunk.toString())));

        // Chunk 2..N: Data samples in groups
        int rowsPerChunk = estimateRowsPerChunk(header, dataLines, separator);
        if (rowsPerChunk < 5) rowsPerChunk = 5;

        for (int i = 0; i < dataLines.size(); i += rowsPerChunk) {
            int end = Math.min(i + rowsPerChunk, dataLines.size());
            StringBuilder sb = new StringBuilder();
            if (!schema.isEmpty()) {
                sb.append("Schema:\n").append(truncate(schema, 500)).append("\n\n");
            }
            if (header.length() > 0) sb.append(header).append("\n");
            for (int j = i; j < end; j++) {
                sb.append(dataLines.get(j)).append("\n");
            }
            sb.append("\nRows ").append(i + 1).append("-").append(end)
              .append(" of ").append(dataLines.size());

            String content = sb.toString();
            int tokens = tokenCounter.countTokens(content);
            if (tokens > maxChunkTokens) {
                // Too large — reduce rows and retry
                int reducedEnd = Math.min(i + rowsPerChunk / 2, dataLines.size());
                sb = new StringBuilder();
                if (header.length() > 0) sb.append(header).append("\n");
                for (int j = i; j < reducedEnd; j++) sb.append(dataLines.get(j)).append("\n");
                content = sb.toString();
                tokens = tokenCounter.countTokens(content);
            }

            chunks.add(new Chunk("chunk_" + (++chunkIdx), content, fileName,
                "Rows " + (i + 1) + "-" + end, "data", tokens));
        }

        return chunks;
    }

    private int estimateRowsPerChunk(String header, List<String> lines, String sep) {
        // Estimate based on average row token count
        int sampleSize = Math.min(10, lines.size());
        int totalTokens = 0;
        for (int i = 0; i < sampleSize; i++) {
            totalTokens += tokenCounter.countTokens(lines.get(i));
        }
        int avgTokensPerRow = totalTokens / Math.max(1, sampleSize);
        int headerTokens = tokenCounter.countTokens(header);
        int budget = maxChunkTokens - headerTokens - 100; // margin for metadata
        return Math.max(5, budget / Math.max(1, avgTokensPerRow));
    }

    private String computeStats(String[] columns, List<String> lines, String sep) {
        if (columns.length == 0 || lines.isEmpty()) return "No statistics available.\n";

        StringBuilder sb = new StringBuilder("Statistics:\n");
        sb.append("- Row count: ").append(lines.size()).append("\n");

        // Per-column stats (sample first 1000 rows)
        int sampleSize = Math.min(1000, lines.size());
        for (int col = 0; col < columns.length && col < 20; col++) {
            Set<String> distinct = new HashSet<>();
            int numericCount = 0;
            double sum = 0, min = Double.MAX_VALUE, max = Double.MIN_VALUE;

            for (int i = 0; i < sampleSize; i++) {
                String[] vals = splitCsvLine(lines.get(i), sep);
                if (col >= vals.length) continue;
                String val = vals[col].trim();
                distinct.add(val);
                try {
                    double d = Double.parseDouble(val);
                    numericCount++;
                    sum += d;
                    if (d < min) min = d;
                    if (d > max) max = d;
                } catch (NumberFormatException ignored) {}
            }

            sb.append("- ").append(columns[col]).append(": ");
            sb.append(distinct.size()).append(" distinct values");
            if (numericCount > sampleSize / 2) {
                sb.append(String.format(" | min=%.2f, max=%.2f, avg=%.2f",
                    min, max, sum / numericCount));
            } else if (distinct.size() <= 10) {
                sb.append(" [");
                int count = 0;
                for (String v : distinct) {
                    if (count++ > 0) sb.append(", ");
                    if (count > 5) { sb.append("..."); break; }
                    sb.append(v);
                }
                sb.append("]");
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    private String[] splitCsvLine(String line, String sep) {
        // Simple CSV split (handles quoted fields minimally)
        List<String> fields = new ArrayList<>();
        boolean inQuote = false;
        StringBuilder current = new StringBuilder();
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c == '"') {
                inQuote = !inQuote;
            } else if (!inQuote && String.valueOf(c).equals(sep)) {
                fields.add(current.toString());
                current = new StringBuilder();
            } else {
                current.append(c);
            }
        }
        fields.add(current.toString());
        return fields.toArray(new String[0]);
    }

    private String truncate(String s, int maxLen) {
        return s.length() <= maxLen ? s : s.substring(0, maxLen) + "...";
    }
}
