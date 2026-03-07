package it.denzosoft.llmplayer.tuning.dataset;

/**
 * Meta-prompts for Q&A generation across the three scenarios.
 * Prompts are calibrated by target model capacity (embedding size).
 */
public class PromptTemplates {

    /** Complexity hint based on target model embedding dimension. */
    public static String complexityHint(int embeddingLength) {
        if (embeddingLength <= 2048) {
            return "Generate concise answers (max 150 words). One question = one concept. Use simple vocabulary.";
        } else if (embeddingLength <= 4096) {
            return "Generate detailed answers with examples when helpful. You may cover multiple aspects per answer.";
        } else {
            return "Generate thorough, detailed answers with examples, edge cases, and multi-step reasoning.";
        }
    }

    /** Meta-prompt for source code Q&A generation. */
    public static String codePrompt(String chunk, String sourceFile, String contextType,
                                    int pairsCount, String complexityHint) {
        return "You are an expert programmer. Given the following " + contextType
            + " from file \"" + sourceFile + "\", generate exactly " + pairsCount
            + " question/answer pairs.\n\n"
            + "The questions must cover different aspects:\n"
            + "- Explanation: What does this code do?\n"
            + "- Parameters/API: What are the inputs, outputs, return values?\n"
            + "- Flow: How does execution proceed? What calls what?\n"
            + "- Issues: Are there potential bugs, edge cases, or improvements?\n"
            + "- Usage: How would you use this code? Show an example.\n\n"
            + complexityHint + "\n\n"
            + "Code:\n```\n" + chunk + "\n```\n\n"
            + "Generate in this exact JSON format (array of objects):\n"
            + "[{\"q\": \"question text\", \"a\": \"answer text\"}, ...]\n"
            + "Output ONLY the JSON array, no other text.";
    }

    /** Meta-prompt for document Q&A generation. */
    public static String documentPrompt(String chunk, String sourceFile, String sectionTitle,
                                        int pairsCount, String complexityHint) {
        return "You are a document analyst. Given the following text extracted from \""
            + sourceFile + "\""
            + (sectionTitle != null && !sectionTitle.isEmpty() ? ", section \"" + sectionTitle + "\"" : "")
            + ", generate exactly " + pairsCount + " question/answer pairs.\n\n"
            + "The questions must cover different types:\n"
            + "- Factual: Ask about specific facts, dates, numbers mentioned in the text\n"
            + "- Summary: Ask for a summary of the key points\n"
            + "- Inferential: Ask about implications or consequences\n"
            + "- Comparative: Compare elements mentioned in the text\n"
            + "- Procedural: Ask about processes or steps described\n\n"
            + "Answers must cite or reference the text when possible.\n\n"
            + complexityHint + "\n\n"
            + "Text:\n" + chunk + "\n\n"
            + "Generate in this exact JSON format (array of objects):\n"
            + "[{\"q\": \"question text\", \"a\": \"answer text\"}, ...]\n"
            + "Output ONLY the JSON array, no other text.";
    }

    /** Meta-prompt for structured data Q&A generation. */
    public static String structuredDataPrompt(String chunk, String contextType,
                                              int pairsCount, String complexityHint) {
        String typeHint;
        if ("schema".equals(contextType)) {
            typeHint = "Focus on:\n"
                + "- Schema understanding: table structures, column types, relationships\n"
                + "- Data overview: row counts, value distributions\n"
                + "- SQL queries: write queries that answer analytical questions\n"
                + "- Data quality: identify potential issues or anomalies\n";
        } else {
            typeHint = "Focus on:\n"
                + "- Counts and aggregations from the data shown\n"
                + "- Trends and patterns visible in the rows\n"
                + "- Comparisons between different rows or groups\n"
                + "- SQL queries that would extract the answer\n"
                + "- Anomalies or notable values\n";
        }

        return "You are a data analyst. Given the following " + contextType
            + " information, generate exactly " + pairsCount
            + " question/answer pairs.\n\n"
            + typeHint + "\n"
            + complexityHint + "\n\n"
            + "Data:\n" + chunk + "\n\n"
            + "Generate in this exact JSON format (array of objects):\n"
            + "[{\"q\": \"question text\", \"a\": \"answer text\"}, ...]\n"
            + "Output ONLY the JSON array, no other text.";
    }

    /** Build the meta-prompt for a given chunk and scenario type. */
    public static String buildPrompt(Chunk chunk, String dataType, int pairsCount,
                                     int embeddingLength) {
        String hint = complexityHint(embeddingLength);
        switch (dataType) {
            case "code":
                return codePrompt(chunk.content(), chunk.sourceFile(), chunk.contextType(),
                    pairsCount, hint);
            case "document":
                return documentPrompt(chunk.content(), chunk.sourceFile(), chunk.sectionTitle(),
                    pairsCount, hint);
            case "structured":
                return structuredDataPrompt(chunk.content(), chunk.contextType(),
                    pairsCount, hint);
            default:
                return codePrompt(chunk.content(), chunk.sourceFile(), chunk.contextType(),
                    pairsCount, hint);
        }
    }
}
