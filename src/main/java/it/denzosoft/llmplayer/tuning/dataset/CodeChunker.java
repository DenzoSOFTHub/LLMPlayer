package it.denzosoft.llmplayer.tuning.dataset;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Scans a directory of source files and produces semantically meaningful chunks.
 * <p>
 * Java files are chunked using a skeleton+method strategy:
 * <ol>
 *   <li>A "skeleton" chunk: the class with full field declarations but method
 *       bodies replaced by {@code { ... }} — teaches the model the API surface.</li>
 *   <li>One chunk per method (or group of small methods): the method body preceded
 *       by the class header (package, imports, class declaration, fields) as context.</li>
 * </ol>
 * Other source languages fall back to fixed-size text chunking with overlap.
 */
public class CodeChunker {

    /** Functional interface for counting tokens in a string. */
    public interface TokenCounter {
        int countTokens(String text);
    }

    private static final Set<String> CODE_EXTENSIONS = new HashSet<String>(Arrays.asList(
            ".java", ".py", ".js", ".ts", ".c", ".cpp", ".h", ".hpp",
            ".go", ".rs", ".cs", ".kt", ".scala", ".rb", ".swift", ".m"));
    private static final Pattern CLASS_PATTERN = Pattern.compile(
            "^\\s*(public\\s+|protected\\s+|private\\s+|abstract\\s+|final\\s+|static\\s+)*"
            + "(class|interface|enum|record)\\s+(\\w+)");
    private static final Pattern METHOD_PATTERN = Pattern.compile(
            "^\\s*(public|protected|private|static|final|abstract|synchronized|native|default|"
            + "void|int|long|float|double|boolean|byte|char|short|String|\\w+)"
            + "[^;=]*\\([^)]*\\)\\s*(throws\\s+[^{]*)?(\\{|$)");

    private final int maxChunkTokens;
    private final int overlapTokens;
    private final TokenCounter tokenCounter;
    private int chunkSeq;

    public CodeChunker(int maxChunkTokens, int overlapTokens, TokenCounter tokenCounter) {
        this.maxChunkTokens = maxChunkTokens;
        this.overlapTokens = overlapTokens;
        this.tokenCounter = tokenCounter;
    }

    /** Recursively scans the directory and chunks all source files found. */
    public List<Chunk> chunk(Path directory) throws IOException {
        chunkSeq = 0;
        List<Chunk> chunks = new ArrayList<Chunk>();
        Files.walkFileTree(directory, new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                String name = file.getFileName().toString();
                int dot = name.lastIndexOf('.');
                if (dot >= 0 && CODE_EXTENSIONS.contains(name.substring(dot))) {
                    String content = new String(Files.readAllBytes(file), StandardCharsets.UTF_8);
                    String rel = directory.relativize(file).toString();
                    if (name.endsWith(".java")) {
                        chunks.addAll(chunkJava(content, rel));
                    } else {
                        chunks.addAll(chunkFixed(content, rel, name));
                    }
                }
                return FileVisitResult.CONTINUE;
            }
        });
        return chunks;
    }

    /**
     * Chunk a Java source file using skeleton+method strategy.
     *
     * 1. Parse the file into a header (package/imports) and method sections.
     * 2. Build a skeleton: header + class declaration + fields + method signatures with { ... }
     * 3. Emit the skeleton as a "skeleton" chunk (teaches API surface).
     * 4. For each method, emit a chunk: header context + method body.
     *    Small methods are grouped together up to maxChunkTokens.
     */
    private List<Chunk> chunkJava(String source, String relativePath) {
        List<Chunk> result = new ArrayList<Chunk>();
        String[] lines = source.split("\n", -1);
        String className = stripExtension(relativePath);

        // --- Phase 1: identify the file header and method boundaries ---
        int headerEnd = 0; // line index where header (package+imports+class decl+fields) ends
        List<MethodBlock> methods = new ArrayList<MethodBlock>();
        List<String> headerLines = new ArrayList<String>();
        boolean inClassBody = false;
        int braceDepth = 0;
        int methodStart = -1;
        String methodTitle = null;

        for (int i = 0; i < lines.length; i++) {
            String trimmed = lines[i].trim();

            // Track brace depth
            for (int c = 0; c < lines[i].length(); c++) {
                char ch = lines[i].charAt(c);
                if (ch == '{') braceDepth++;
                else if (ch == '}') braceDepth--;
            }

            // Detect class declaration
            if (!inClassBody) {
                Matcher cm = CLASS_PATTERN.matcher(lines[i]);
                if (cm.find()) {
                    className = cm.group(3);
                    inClassBody = true;
                    headerEnd = i;
                }
                continue;
            }

            // Inside class body: detect method starts
            Matcher mm = METHOD_PATTERN.matcher(lines[i]);
            if (mm.find() && !trimmed.startsWith("//") && !trimmed.startsWith("*")
                    && braceDepth <= 2) { // depth 1 = class body, depth 2 = just entered method
                // Close previous method if any
                if (methodStart >= 0) {
                    methods.add(new MethodBlock(methodTitle, methodStart, i - 1));
                }
                // Everything between headerEnd and first method is "class header"
                if (methods.isEmpty() && methodStart < 0) {
                    headerEnd = i - 1;
                }
                methodTitle = methodName(lines[i], className);
                methodStart = i;
            }
        }
        // Close last method (goes to end of file, excluding final closing brace)
        if (methodStart >= 0) {
            int end = lines.length - 1;
            // Trim trailing closing braces (class closing brace)
            while (end > methodStart && lines[end].trim().equals("}")) end--;
            methods.add(new MethodBlock(methodTitle, methodStart, end));
        }

        // --- Phase 2: build the class header (package + imports + class decl + fields) ---
        StringBuilder classHeader = new StringBuilder();
        for (int i = 0; i <= Math.min(headerEnd, lines.length - 1); i++) {
            if (classHeader.length() > 0) classHeader.append("\n");
            classHeader.append(lines[i]);
        }
        String classHeaderStr = classHeader.toString();

        // --- Phase 3: build skeleton (class with method bodies replaced by { ... }) ---
        if (methods.size() > 1) { // Only emit skeleton if there's more than 1 method
            StringBuilder skeleton = new StringBuilder();
            skeleton.append(classHeaderStr);

            for (MethodBlock mb : methods) {
                skeleton.append("\n\n    ");
                // Method signature (first line)
                skeleton.append(lines[mb.startLine].trim());
                // Replace body with { ... }
                if (!lines[mb.startLine].trim().endsWith("{")) {
                    skeleton.append(" { ... }");
                } else {
                    // Remove trailing { and add { ... }
                    String sig = skeleton.toString();
                    if (sig.endsWith("{")) {
                        skeleton.setLength(skeleton.length() - 1);
                        skeleton.append("{ ... }");
                    } else {
                        skeleton.append(" ... }");
                    }
                }
            }
            skeleton.append("\n}");

            String skeletonStr = skeleton.toString();
            int skeletonTokens = tokenCounter.countTokens(skeletonStr);
            if (skeletonTokens <= maxChunkTokens) {
                result.add(emit(skeletonStr, relativePath, className + " (skeleton)", "skeleton"));
            }
            // If skeleton is too large, skip it (rare — would mean hundreds of methods)
        }

        // --- Phase 4: emit method chunks with class header as context ---
        // For small files with no methods detected, emit the whole file
        if (methods.isEmpty()) {
            result.add(emit(source, relativePath, className, "class"));
            return result;
        }

        int headerTokens = tokenCounter.countTokens(classHeaderStr);
        int budgetForMethods = maxChunkTokens - headerTokens - 20; // 20 token margin
        if (budgetForMethods < 200) {
            // Header alone is too large — fall back to whole-file chunk
            result.add(emit(source, relativePath, className, "class"));
            return result;
        }

        // Group small methods together, emit large methods individually
        StringBuilder methodBatch = new StringBuilder();
        String batchTitle = null;
        int batchTokens = 0;
        int methodsInBatch = 0;

        for (MethodBlock mb : methods) {
            String methodText = join(lines, mb.startLine, mb.endLine + 1);
            int methodTokens = tokenCounter.countTokens(methodText);

            // If this single method exceeds the budget, emit it alone (possibly truncated)
            if (methodTokens > budgetForMethods) {
                // Flush current batch first
                if (methodBatch.length() > 0) {
                    String content = classHeaderStr + "\n\n" + methodBatch.toString() + "\n}";
                    String title = methodsInBatch == 1 ? batchTitle : className + " (methods)";
                    result.add(emit(content, relativePath, title, "method"));
                    methodBatch.setLength(0);
                    batchTokens = 0;
                    methodsInBatch = 0;
                    batchTitle = null;
                }
                // Emit the large method with header
                String content = classHeaderStr + "\n\n" + methodText + "\n}";
                result.add(emit(content, relativePath, mb.title, "method"));
                continue;
            }

            // Would adding this method overflow the budget?
            if (batchTokens + methodTokens > budgetForMethods && methodBatch.length() > 0) {
                // Flush current batch
                String content = classHeaderStr + "\n\n" + methodBatch.toString() + "\n}";
                String title = methodsInBatch == 1 ? batchTitle : className + " (methods)";
                result.add(emit(content, relativePath, title, "method"));
                methodBatch.setLength(0);
                batchTokens = 0;
                methodsInBatch = 0;
                batchTitle = null;
            }

            // Add to batch
            if (methodBatch.length() > 0) methodBatch.append("\n\n");
            methodBatch.append(methodText);
            batchTokens += methodTokens;
            methodsInBatch++;
            if (batchTitle == null) batchTitle = mb.title;
        }

        // Flush remaining batch
        if (methodBatch.length() > 0) {
            String content = classHeaderStr + "\n\n" + methodBatch.toString() + "\n}";
            String title = methodsInBatch == 1 ? batchTitle : className + " (methods)";
            result.add(emit(content, relativePath, title, "method"));
        }

        return result;
    }

    private List<Chunk> chunkFixed(String content, String relativePath, String fileName) {
        List<Chunk> result = new ArrayList<Chunk>();
        String[] lines = content.split("\n", -1);
        int start = 0;
        while (start < lines.length) {
            StringBuilder sb = new StringBuilder();
            int end = start;
            while (end < lines.length) {
                String candidate = sb.length() > 0 ? sb + "\n" + lines[end] : lines[end];
                if (tokenCounter.countTokens(candidate) > maxChunkTokens && sb.length() > 0) break;
                if (sb.length() > 0) sb.append("\n");
                sb.append(lines[end]);
                end++;
            }
            result.add(emit(sb.toString(), relativePath, fileName, "file"));
            // Step back by overlapTokens worth of lines
            int back = 0;
            for (int i = end - 1; i > start && tokenCounter.countTokens(join(lines, i, end)) < overlapTokens; i--) {
                back++;
            }
            int next = end - back;
            start = next > start ? next : end; // prevent infinite loop
        }
        return result;
    }

    private Chunk emit(String content, String relPath, String title, String type) {
        return new Chunk("chunk_" + (chunkSeq++), content, relPath, title, type,
                tokenCounter.countTokens(content));
    }

    private static String join(String[] lines, int from, int to) {
        StringBuilder sb = new StringBuilder();
        for (int i = from; i < to; i++) {
            if (i > from) sb.append("\n");
            sb.append(lines[i]);
        }
        return sb.toString();
    }

    private static String stripExtension(String path) {
        int slash = path.lastIndexOf('/');
        String name = slash >= 0 ? path.substring(slash + 1) : path;
        int dot = name.lastIndexOf('.');
        return dot >= 0 ? name.substring(0, dot) : name;
    }

    private static String methodName(String line, String className) {
        String t = line.trim();
        int paren = t.indexOf('(');
        if (paren > 0) {
            String before = t.substring(0, paren).trim();
            int sp = before.lastIndexOf(' ');
            if (sp >= 0) return className + "." + before.substring(sp + 1);
        }
        return className;
    }

    /** A method block identified by its line range in the source. */
    private static class MethodBlock {
        final String title;
        final int startLine;
        final int endLine; // inclusive
        MethodBlock(String title, int startLine, int endLine) {
            this.title = title;
            this.startLine = startLine;
            this.endLine = endLine;
        }
    }
}
