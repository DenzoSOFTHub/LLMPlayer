package it.denzosoft.llmplayer.tuning;

import it.denzosoft.llmplayer.tokenizer.Tokenizer;
import it.denzosoft.llmplayer.tuning.analyze.TargetAnalyzer;
import it.denzosoft.llmplayer.tuning.dataset.Chunk;
import it.denzosoft.llmplayer.tuning.dataset.CodeChunker;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.List;

/**
 * Standalone utility: dumps all chunks to a JSONL file for inspection.
 * Usage: java ... ChunkDumper <model.gguf> <source-dir> <output.jsonl>
 */
public class ChunkDumper {

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.out.println("Usage: ChunkDumper <model.gguf> <source-dir> <output.jsonl> [chunk-size]");
            return;
        }
        Path modelPath = Paths.get(args[0]);
        Path sourceDir = Paths.get(args[1]);
        Path outputFile = Paths.get(args[2]);
        int userChunkSize = args.length >= 4 ? Integer.parseInt(args[3]) : 0;

        System.out.println("Analyzing model for tokenizer...");
        TargetAnalyzer analyzer = new TargetAnalyzer();
        TargetAnalyzer.AnalysisResult result = analyzer.analyze(modelPath);
        final Tokenizer tok = result.tokenizer();
        int maxChunkTokens = userChunkSize > 0 ? userChunkSize : result.maxChunkTokens();
        System.out.printf("  Max chunk tokens: %d%s%n", maxChunkTokens,
                userChunkSize > 0 ? " (user)" : "");

        CodeChunker.TokenCounter counter = new CodeChunker.TokenCounter() {
            public int countTokens(String text) { return tok.encode(text).length; }
        };

        System.out.printf("Chunking %s...%n", sourceDir);
        List<Chunk> chunks = new CodeChunker(maxChunkTokens, 100, counter).chunk(sourceDir);
        System.out.printf("  Total chunks: %d%n%n", chunks.size());

        try (BufferedWriter w = Files.newBufferedWriter(outputFile, StandardCharsets.UTF_8)) {
            for (int i = 0; i < chunks.size(); i++) {
                Chunk c = chunks.get(i);
                w.write("{\"index\":" + i
                    + ",\"id\":\"" + esc(c.id()) + "\""
                    + ",\"sourceFile\":\"" + esc(c.sourceFile()) + "\""
                    + ",\"sectionTitle\":\"" + esc(c.sectionTitle()) + "\""
                    + ",\"contextType\":\"" + esc(c.contextType()) + "\""
                    + ",\"tokenCount\":" + c.tokenCount()
                    + ",\"charCount\":" + c.content().length()
                    + ",\"content\":\"" + esc(c.content()) + "\""
                    + "}");
                w.newLine();

                System.out.printf("  [%3d] %-50s %5d tok  %s%n",
                    i, c.sourceFile() + " / " + c.sectionTitle(),
                    c.tokenCount(), c.contextType());
            }
        }

        System.out.printf("%nSaved to %s%n", outputFile);
    }

    private static String preview(String s, int maxLen) {
        String clean = s.replace("\n", "\\n").replace("\r", "");
        return clean.length() > maxLen ? clean.substring(0, maxLen) + "..." : clean;
    }

    private static String esc(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\n", "\\n").replace("\r", "").replace("\t", "\\t");
    }
}
