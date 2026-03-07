package it.denzosoft.llmplayer.tuning.dataset;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
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
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/** Scans a directory for document files (.txt, .md, .csv, .tsv, .docx, .xlsx)
 *  and produces text chunks split by paragraphs with configurable token budget and overlap. */
public class TextChunker {

    private static final Set<String> TEXT_EXTENSIONS = new HashSet<String>(
            Arrays.asList(".txt", ".md", ".csv", ".tsv"));

    private static final Pattern XML_TAG = Pattern.compile("<[^>]+>");
    private static final Pattern XLSX_T_TAG = Pattern.compile("<t[^>]*>([^<]*)</t>");
    private static final Pattern PARAGRAPH_SPLIT = Pattern.compile("\\n\\s*\\n");
    private static final Pattern WHITESPACE_RUN = Pattern.compile("\\s+");

    private final int maxChunkTokens;
    private final int overlapTokens;
    private final TokenCounter tokenCounter;

    public TextChunker(int maxChunkTokens, int overlapTokens, TokenCounter tokenCounter) {
        this.maxChunkTokens = maxChunkTokens;
        this.overlapTokens = overlapTokens;
        this.tokenCounter = tokenCounter;
    }

    /** Recursively scans the directory for supported files, extracts text, and returns chunks. */
    public List<Chunk> chunk(Path directory) throws IOException {
        List<Path> files = collectFiles(directory);
        List<Chunk> chunks = new ArrayList<Chunk>();
        int chunkIndex = 0;

        for (Path file : files) {
            String text = extractText(file);
            if (text == null || text.trim().isEmpty()) {
                continue;
            }

            String relativePath = directory.relativize(file).toString();
            String sectionTitle = file.getFileName().toString();

            List<String> paragraphs = splitParagraphs(text);
            List<Chunk> fileChunks = buildChunks(paragraphs, relativePath, sectionTitle, chunkIndex);
            chunkIndex += fileChunks.size();
            chunks.addAll(fileChunks);
        }

        return chunks;
    }

    private List<Path> collectFiles(Path directory) throws IOException {
        final List<Path> files = new ArrayList<Path>();
        Files.walkFileTree(directory, new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
                String name = file.getFileName().toString().toLowerCase();
                if (isSupportedFile(name)) {
                    files.add(file);
                }
                return FileVisitResult.CONTINUE;
            }
        });
        return files;
    }

    private boolean isSupportedFile(String lowerName) {
        for (String ext : TEXT_EXTENSIONS) {
            if (lowerName.endsWith(ext)) {
                return true;
            }
        }
        return lowerName.endsWith(".docx") || lowerName.endsWith(".xlsx");
    }

    private String extractText(Path file) throws IOException {
        String name = file.getFileName().toString().toLowerCase();
        if (name.endsWith(".docx")) {
            return extractDocx(file);
        } else if (name.endsWith(".xlsx")) {
            return extractXlsx(file);
        } else {
            return new String(Files.readAllBytes(file), StandardCharsets.UTF_8);
        }
    }

    /** Extract text from a .docx file by reading word/document.xml from the ZIP. */
    private String extractDocx(Path file) throws IOException {
        String xml = readZipEntry(file, "word/document.xml");
        if (xml == null) {
            return "";
        }
        String text = XML_TAG.matcher(xml).replaceAll(" ");
        return WHITESPACE_RUN.matcher(text).replaceAll(" ").trim();
    }

    /** Extract text from a .xlsx file by reading shared strings and sheet data. */
    private String extractXlsx(Path file) throws IOException {
        StringBuilder sb = new StringBuilder();
        String sharedStrings = readZipEntry(file, "xl/sharedStrings.xml");
        if (sharedStrings != null) {
            Matcher m = XLSX_T_TAG.matcher(sharedStrings);
            while (m.find()) {
                if (sb.length() > 0) {
                    sb.append('\t');
                }
                sb.append(m.group(1));
            }
        }

        String sheet = readZipEntry(file, "xl/worksheets/sheet1.xml");
        if (sheet != null) {
            Matcher m = XLSX_T_TAG.matcher(sheet);
            while (m.find()) {
                if (sb.length() > 0) {
                    sb.append('\t');
                }
                sb.append(m.group(1));
            }
        }

        return sb.toString().trim();
    }

    /** Read a single entry from a ZIP file by name, or null if not found. */
    private String readZipEntry(Path zipFile, String entryName) throws IOException {
        ZipInputStream zis = new ZipInputStream(Files.newInputStream(zipFile));
        try {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                if (entry.getName().equals(entryName)) {
                    return readStreamFully(zis);
                }
            }
            return null;
        } finally {
            zis.close();
        }
    }

    private String readStreamFully(InputStream is) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte[] buf = new byte[8192];
        int n;
        while ((n = is.read(buf)) != -1) {
            baos.write(buf, 0, n);
        }
        return new String(baos.toByteArray(), StandardCharsets.UTF_8);
    }

    private List<String> splitParagraphs(String text) {
        String[] parts = PARAGRAPH_SPLIT.split(text);
        List<String> paragraphs = new ArrayList<String>();
        for (String p : parts) {
            String trimmed = p.trim();
            if (!trimmed.isEmpty()) {
                paragraphs.add(trimmed);
            }
        }
        return paragraphs;
    }

    /** Build chunks from paragraphs, starting a new chunk when the token budget is exceeded. */
    private List<Chunk> buildChunks(List<String> paragraphs, String sourceFile,
                                     String sectionTitle, int startIndex) {
        List<Chunk> chunks = new ArrayList<Chunk>();
        if (paragraphs.isEmpty()) {
            return chunks;
        }

        List<String> current = new ArrayList<String>();
        int currentTokens = 0;

        for (int i = 0; i < paragraphs.size(); i++) {
            String para = paragraphs.get(i);
            int paraTokens = tokenCounter.countTokens(para);

            if (!current.isEmpty() && currentTokens + paraTokens > maxChunkTokens) {
                // Emit the current chunk
                String content = joinParagraphs(current);
                int tokens = tokenCounter.countTokens(content);
                String id = "chunk_" + (startIndex + chunks.size());
                chunks.add(new Chunk(id, content, sourceFile, sectionTitle, "section", tokens));

                // Build overlap from the tail of the current chunk
                current = buildOverlap(current);
                currentTokens = tokenCounter.countTokens(joinParagraphs(current));
            }

            current.add(para);
            currentTokens += paraTokens;
        }

        // Emit the final chunk
        if (!current.isEmpty()) {
            String content = joinParagraphs(current);
            int tokens = tokenCounter.countTokens(content);
            String id = "chunk_" + (startIndex + chunks.size());
            chunks.add(new Chunk(id, content, sourceFile, sectionTitle, "section", tokens));
        }

        return chunks;
    }

    /** Take paragraphs from the end of the list until the overlap token budget is reached. */
    private List<String> buildOverlap(List<String> paragraphs) {
        if (overlapTokens <= 0) {
            return new ArrayList<String>();
        }

        List<String> overlap = new ArrayList<String>();
        int tokens = 0;
        for (int i = paragraphs.size() - 1; i >= 0; i--) {
            int paraTokens = tokenCounter.countTokens(paragraphs.get(i));
            if (tokens + paraTokens > overlapTokens && !overlap.isEmpty()) {
                break;
            }
            overlap.add(0, paragraphs.get(i));
            tokens += paraTokens;
        }
        return overlap;
    }

    private String joinParagraphs(List<String> paragraphs) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < paragraphs.size(); i++) {
            if (i > 0) {
                sb.append("\n\n");
            }
            sb.append(paragraphs.get(i));
        }
        return sb.toString();
    }
}
