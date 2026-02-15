package it.denzosoft.llmplayer.tokenizer;

import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * SentencePiece-style tokenizer using Unigram/greedy longest-match with scores.
 * Used by Qwen, DeepSeek, and other models with tokenizer.ggml.model="llama".
 */
public class SentencePieceTokenizer implements Tokenizer {

    private final String[] vocab;
    private final float[] scores;
    private final Map<String, Integer> tokenToId;
    private final SpecialTokens specialTokens;
    private final Map<String, Integer> specialTokenMap;

    // SentencePiece uses U+2581 (▁) as space replacement
    private static final char SPACE_REPLACEMENT = '\u2581';

    public SentencePieceTokenizer(String[] vocab, float[] scores, SpecialTokens specialTokens) {
        this.vocab = vocab;
        this.scores = scores;
        this.specialTokens = specialTokens;
        this.tokenToId = new HashMap<>(vocab.length * 2);
        this.specialTokenMap = new HashMap<>();

        for (int i = 0; i < vocab.length; i++) {
            tokenToId.put(vocab[i], i);
            String token = vocab[i];
            if (token.startsWith("<") && token.endsWith(">") && token.length() > 2) {
                specialTokenMap.put(token, i);
            }
            if (token.startsWith("<|") && token.endsWith("|>")) {
                specialTokenMap.put(token, i);
            }
        }
    }

    @Override
    public int[] encode(String text) {
        if (text == null || text.isEmpty()) return new int[0];

        List<Integer> allTokens = new ArrayList<>();

        // Split on special tokens first
        List<String> parts = splitOnSpecialTokens(text);

        for (String part : parts) {
            Integer specialId = specialTokenMap.get(part);
            if (specialId != null) {
                allTokens.add(specialId);
                continue;
            }

            // Replace spaces with ▁ for SentencePiece
            String processed = part.replace(' ', SPACE_REPLACEMENT);
            // Prepend ▁ if text starts fresh (SentencePiece convention)
            if (!processed.isEmpty() && processed.charAt(0) != SPACE_REPLACEMENT) {
                processed = SPACE_REPLACEMENT + processed;
            }

            List<Integer> partTokens = greedyEncode(processed);
            allTokens.addAll(partTokens);
        }

        return allTokens.stream().mapToInt(Integer::intValue).toArray();
    }

    private List<String> splitOnSpecialTokens(String text) {
        List<String> parts = new ArrayList<>();
        if (specialTokenMap.isEmpty()) {
            parts.add(text);
            return parts;
        }

        List<String> sortedSpecials = new ArrayList<>(specialTokenMap.keySet());
        sortedSpecials.sort((a, b) -> b.length() - a.length());

        int pos = 0;
        while (pos < text.length()) {
            boolean found = false;
            for (String special : sortedSpecials) {
                if (text.startsWith(special, pos)) {
                    parts.add(special);
                    pos += special.length();
                    found = true;
                    break;
                }
            }
            if (!found) {
                int nextSpecial = text.length();
                for (String special : sortedSpecials) {
                    int idx = text.indexOf(special, pos);
                    if (idx >= 0 && idx < nextSpecial) {
                        nextSpecial = idx;
                    }
                }
                parts.add(text.substring(pos, nextSpecial));
                pos = nextSpecial;
            }
        }
        return parts;
    }

    /**
     * Greedy BPE-style encoding using scores as merge priorities.
     * This implements a simplified version of SentencePiece encoding.
     */
    private List<Integer> greedyEncode(String text) {
        // Start with individual characters/bytes as tokens
        List<Integer> tokens = new ArrayList<>();
        List<String> pieces = new ArrayList<>();
        int i = 0;
        while (i < text.length()) {
            // Try longest match first
            boolean matched = false;
            int maxLen = Math.min(text.length() - i, 32); // max token length
            for (int len = maxLen; len >= 1; len--) {
                String sub = text.substring(i, i + len);
                Integer id = tokenToId.get(sub);
                if (id != null) {
                    tokens.add(id);
                    pieces.add(sub);
                    i += len;
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                // Byte fallback: encode as <0xHH>
                byte[] bytes = text.substring(i, i + 1).getBytes(StandardCharsets.UTF_8);
                for (byte b : bytes) {
                    String byteToken = String.format("<0x%02X>", Byte.toUnsignedInt(b));
                    Integer id = tokenToId.get(byteToken);
                    if (id != null) {
                        tokens.add(id);
                        pieces.add(byteToken);
                    }
                }
                i++;
            }
        }

        // Now iteratively merge adjacent pairs with highest score
        while (pieces.size() > 1) {
            float bestScore = Float.NEGATIVE_INFINITY;
            int bestPos = -1;

            for (int j = 0; j < pieces.size() - 1; j++) {
                String merged = pieces.get(j) + pieces.get(j + 1);
                Integer mergedId = tokenToId.get(merged);
                if (mergedId != null && scores[mergedId] > bestScore) {
                    bestScore = scores[mergedId];
                    bestPos = j;
                }
            }

            if (bestPos < 0) break;

            String merged = pieces.get(bestPos) + pieces.get(bestPos + 1);
            int mergedId = tokenToId.get(merged);
            pieces.set(bestPos, merged);
            tokens.set(bestPos, mergedId);
            pieces.remove(bestPos + 1);
            tokens.remove(bestPos + 1);
        }

        return tokens;
    }

    @Override
    public String decode(int[] tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            sb.append(decode(token));
        }
        return sb.toString();
    }

    @Override
    public String decode(int token) {
        if (token < 0 || token >= vocab.length) return "";
        String piece = vocab[token];

        // Handle byte tokens <0xHH>
        if (piece.startsWith("<0x") && piece.endsWith(">") && piece.length() == 6) {
            int byteVal = Integer.parseInt(piece.substring(3, 5), 16);
            return new String(new byte[]{(byte) byteVal}, StandardCharsets.UTF_8);
        }

        // Replace ▁ back to space
        return piece.replace(SPACE_REPLACEMENT, ' ');
    }

    @Override
    public int vocabSize() { return vocab.length; }

    @Override
    public boolean isSpecialToken(int tokenId) {
        if (tokenId == specialTokens.getBosId() || tokenId == specialTokens.getEosId()) return true;
        if (tokenId < 0 || tokenId >= vocab.length) return false;
        String token = vocab[tokenId];
        return (token.startsWith("<") && token.endsWith(">") && !token.startsWith("<0x"));
    }
}
