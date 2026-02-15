package it.denzosoft.llmplayer.tokenizer;

import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Byte-Pair Encoding tokenizer (GPT-2 / Llama 3 style).
 * Uses regex pre-tokenization + BPE merge table.
 */
public class BPETokenizer implements Tokenizer {

    private final String[] vocab;
    private final Map<String, Integer> tokenToId;
    private final Map<Long, Integer> mergeRanks; // pair hash -> merge priority
    private final SpecialTokens specialTokens;
    private final Map<String, Integer> specialTokenMap;

    // GPT-2 / Llama 3 pre-tokenization pattern
    private static final Pattern PRE_TOKENIZE = Pattern.compile(
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    );

    public BPETokenizer(String[] vocab, float[] scores, String[] merges, SpecialTokens specialTokens) {
        this.vocab = vocab;
        this.specialTokens = specialTokens;
        this.tokenToId = new HashMap<>(vocab.length * 2);
        this.specialTokenMap = new HashMap<>();

        for (int i = 0; i < vocab.length; i++) {
            tokenToId.put(vocab[i], i);
        }

        // Build special token map for quick lookup of multi-char special tokens
        for (int i = 0; i < vocab.length; i++) {
            String token = vocab[i];
            if (token.startsWith("<|") && token.endsWith("|>")) {
                specialTokenMap.put(token, i);
            } else if (token.startsWith("<") && token.endsWith(">") && token.length() > 2) {
                specialTokenMap.put(token, i);
            }
        }

        // Parse merge table
        this.mergeRanks = new HashMap<>();
        if (merges != null) {
            for (int i = 0; i < merges.length; i++) {
                String merge = merges[i];
                int space = merge.indexOf(' ');
                if (space > 0) {
                    String left = merge.substring(0, space);
                    String right = merge.substring(space + 1);
                    Integer leftId = tokenToId.get(left);
                    Integer rightId = tokenToId.get(right);
                    if (leftId != null && rightId != null) {
                        mergeRanks.put(pairKey(leftId, rightId), i);
                    }
                }
            }
        }
    }

    private static long pairKey(int left, int right) {
        return ((long) left << 32) | (right & 0xFFFFFFFFL);
    }

    @Override
    public int[] encode(String text) {
        if (text == null || text.isEmpty()) return new int[0];

        List<Integer> allTokens = new ArrayList<>();

        // First, check for special tokens and split around them
        List<String> parts = splitOnSpecialTokens(text);

        for (String part : parts) {
            Integer specialId = specialTokenMap.get(part);
            if (specialId != null) {
                allTokens.add(specialId);
                continue;
            }

            // Pre-tokenize using regex
            Matcher matcher = PRE_TOKENIZE.matcher(part);
            while (matcher.find()) {
                String word = matcher.group();
                List<Integer> wordTokens = bpeEncode(word);
                allTokens.addAll(wordTokens);
            }
        }

        return allTokens.stream().mapToInt(Integer::intValue).toArray();
    }

    private List<String> splitOnSpecialTokens(String text) {
        List<String> parts = new ArrayList<>();
        if (specialTokenMap.isEmpty()) {
            parts.add(text);
            return parts;
        }

        // Sort special tokens by length (longest first) for greedy matching
        List<String> sortedSpecials = new ArrayList<>(specialTokenMap.keySet());
        sortedSpecials.sort((a, b) -> b.length() - a.length());

        int pos = 0;
        while (pos < text.length()) {
            boolean found = false;
            for (String special : sortedSpecials) {
                if (text.startsWith(special, pos)) {
                    if (pos > 0) {
                        String before = text.substring(pos - (pos - (parts.isEmpty() ? 0 : pos)), pos);
                        // Already handled by previous iterations
                    }
                    parts.add(special);
                    pos += special.length();
                    found = true;
                    break;
                }
            }
            if (!found) {
                // Find next special token
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

    private List<Integer> bpeEncode(String word) {
        // Convert to byte-level tokens
        byte[] bytes = word.getBytes(StandardCharsets.UTF_8);
        List<Integer> tokens = new ArrayList<>(bytes.length);

        for (byte b : bytes) {
            String byteStr = byteToToken(b);
            Integer id = tokenToId.get(byteStr);
            if (id != null) {
                tokens.add(id);
            } else {
                // Try direct single-char lookup
                id = tokenToId.get(String.valueOf((char) (b & 0xFF)));
                if (id != null) {
                    tokens.add(id);
                }
            }
        }

        if (tokens.isEmpty()) return tokens;

        // Iteratively merge pairs
        while (tokens.size() > 1) {
            // Find best merge (lowest rank)
            int bestRank = Integer.MAX_VALUE;
            int bestPos = -1;

            for (int i = 0; i < tokens.size() - 1; i++) {
                long key = pairKey(tokens.get(i), tokens.get(i + 1));
                Integer rank = mergeRanks.get(key);
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestPos = i;
                }
            }

            if (bestPos < 0) break;

            // Merge the pair
            String merged = vocab[tokens.get(bestPos)] + vocab[tokens.get(bestPos + 1)];
            Integer mergedId = tokenToId.get(merged);
            if (mergedId == null) break;

            tokens.set(bestPos, mergedId);
            tokens.remove(bestPos + 1);
        }

        return tokens;
    }

    /**
     * Maps a byte value to the GPT-2 byte token representation.
     * GPT-2 uses a specific mapping from byte values to Unicode characters.
     */
    private static String byteToToken(byte b) {
        int v = Byte.toUnsignedInt(b);
        // GPT-2 byte encoder: printable ASCII stays as-is, others get shifted to Unicode range
        if (v >= 33 && v <= 126) return String.valueOf((char) v);
        if (v == 32) return "\u0120"; // space -> Ġ
        if (v == 10) return "\u010A"; // newline -> Ċ
        if (v == 13) return "\u010D"; // carriage return
        if (v == 9) return "\u0109";  // tab
        // General mapping for other bytes
        if (v >= 0 && v <= 32) return String.valueOf((char) (v + 0x100));
        if (v >= 127 && v <= 160) return String.valueOf((char) (v + 0x100 - 127 + 0x7F));
        return String.valueOf((char) v);
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
        // Handle GPT-2 byte tokens: decode byte-level representation back to UTF-8
        return decodeTokenPiece(piece);
    }

    private String decodeTokenPiece(String piece) {
        // Check if it's a special token
        if (piece.startsWith("<") && piece.endsWith(">")) {
            return piece;
        }
        // Convert byte-level BPE back to actual bytes
        byte[] bytes = new byte[piece.length()];
        int byteLen = 0;
        for (int i = 0; i < piece.length(); i++) {
            char c = piece.charAt(i);
            int b = tokenCharToByte(c);
            if (b >= 0) {
                bytes[byteLen++] = (byte) b;
            } else {
                // Multi-byte chars: encode to UTF-8
                byte[] charBytes = String.valueOf(c).getBytes(StandardCharsets.UTF_8);
                if (byteLen + charBytes.length > bytes.length) {
                    bytes = Arrays.copyOf(bytes, bytes.length * 2);
                }
                System.arraycopy(charBytes, 0, bytes, byteLen, charBytes.length);
                byteLen += charBytes.length;
            }
        }
        return new String(bytes, 0, byteLen, StandardCharsets.UTF_8);
    }

    private static int tokenCharToByte(char c) {
        if (c >= 33 && c <= 126) return c;
        if (c == 0x0120) return 32;  // Ġ -> space
        if (c == 0x010A) return 10;  // Ċ -> newline
        if (c == 0x010D) return 13;
        if (c == 0x0109) return 9;
        if (c >= 0x100 && c <= 0x120) return c - 0x100;
        if (c >= 0x17F && c <= 0x19F) return c - 0x100 + 127;
        return -1;
    }

    @Override
    public int vocabSize() { return vocab.length; }

    @Override
    public boolean isSpecialToken(int tokenId) {
        if (tokenId == specialTokens.getBosId() || tokenId == specialTokens.getEosId()) return true;
        if (tokenId < 0 || tokenId >= vocab.length) return false;
        String token = vocab[tokenId];
        return token.startsWith("<|") && token.endsWith("|>");
    }
}
