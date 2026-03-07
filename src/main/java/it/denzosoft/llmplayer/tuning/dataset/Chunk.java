package it.denzosoft.llmplayer.tuning.dataset;

/**
 * A chunk of content ready for Q&A generation.
 * Contains the text plus contextual metadata for the generator prompt.
 */
public class Chunk {

    private final String id;
    private final String content;
    private final String sourceFile;
    private final String sectionTitle;
    private final String contextType; // class, method, section, table, etc.
    private final int tokenCount;

    public Chunk(String id, String content, String sourceFile, String sectionTitle,
                 String contextType, int tokenCount) {
        this.id = id;
        this.content = content;
        this.sourceFile = sourceFile;
        this.sectionTitle = sectionTitle;
        this.contextType = contextType;
        this.tokenCount = tokenCount;
    }

    public String id() { return id; }
    public String content() { return content; }
    public String sourceFile() { return sourceFile; }
    public String sectionTitle() { return sectionTitle; }
    public String contextType() { return contextType; }
    public int tokenCount() { return tokenCount; }
}
