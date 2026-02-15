package it.denzosoft.llmplayer.cli;

import it.denzosoft.llmplayer.api.ModelInfo;

public class ModelInfoPrinter {

    public static void print(ModelInfo info) {
        String separator = "+" + repeat("-", 26) + "+" + repeat("-", 30) + "+";
        System.out.println(separator);
        System.out.printf("| %-24s | %-28s |%n", "Property", "Value");
        System.out.println(separator);
        printRow("Name", info.name());
        printRow("Architecture", info.architecture());
        printRow("Embedding Length", String.valueOf(info.embeddingLength()));
        printRow("Layers", String.valueOf(info.blockCount()));
        printRow("Attention Heads", String.valueOf(info.headCount()));
        printRow("KV Heads", String.valueOf(info.headCountKV()));
        printRow("Context Length", String.valueOf(info.contextLength()));
        printRow("Vocab Size", String.valueOf(info.vocabSize()));
        printRow("FFN Size", String.valueOf(info.intermediateSize()));
        System.out.println(separator);
    }

    private static void printRow(String property, String value) {
        System.out.printf("| %-24s | %-28s |%n", property, value);
    }

    private static String repeat(String s, int count) {
        StringBuilder sb = new StringBuilder(s.length() * count);
        for (int i = 0; i < count; i++) {
            sb.append(s);
        }
        return sb.toString();
    }
}
