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
    private final boolean useGpt2ByteMapping; // false for gemma4 (uses SentencePiece-style encoding)

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
        // Gemma 4: <|turn>, <turn|> etc. should NOT be special tokens — they are encoded
        // as regular BPE text (multi-token). Only true control tokens (BOS/EOS) are special.
        boolean isGemma4 = "gemma4".equals(
            specialTokens != null ? specialTokens.getModelType() : null);
        this.useGpt2ByteMapping = !isGemma4;
        for (int i = 0; i < vocab.length; i++) {
            String token = vocab[i];
            if (isGemma4) {
                // Gemma 4: register angle-bracket tokens AND \n/\n\n as special tokens.
                // Use SentencePiece byte mapping for regular text (not GPT-2 byte mapping).
                if (token.startsWith("<") && token.endsWith(">") && token.length() > 2) {
                    specialTokenMap.put(token, i);
                } else if ("\n".equals(token) || "\n\n".equals(token)) {
                    specialTokenMap.put(token, i);
                }
            } else {
                if (token.startsWith("<|") && token.endsWith("|>")) {
                    specialTokenMap.put(token, i);
                } else if (token.startsWith("<") && token.endsWith(">") && token.length() > 2) {
                    specialTokenMap.put(token, i);
                }
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

            if (useGpt2ByteMapping) {
                // GPT-2/Llama3 pre-tokenize using regex
                Matcher matcher = PRE_TOKENIZE.matcher(part);
                while (matcher.find()) {
                    String word = matcher.group();
                    allTokens.addAll(bpeEncode(word));
                }
            } else {
                // Gemma 4: replace spaces with ▁ (SentencePiece style), then BPE the whole piece
                String spPart = part.replace(' ', '\u2581');
                allTokens.addAll(bpeEncode(spPart));
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
        List<Integer> tokens;
        if (useGpt2ByteMapping) {
            byte[] bytes = word.getBytes(StandardCharsets.UTF_8);
            tokens = new ArrayList<>(bytes.length);
            for (byte b : bytes) {
                String byteStr = byteToToken(b);
                Integer id = tokenToId.get(byteStr);
                if (id != null) {
                    tokens.add(id);
                } else {
                    id = tokenToId.get(String.valueOf((char) (b & 0xFF)));
                    if (id != null) tokens.add(id);
                }
            }
        } else {
            // Non-GPT-2 mode (Gemma 4): lookup each character directly in vocabulary
            tokens = new ArrayList<>(word.length());
            for (int i = 0; i < word.length(); ) {
                int cp = word.codePointAt(i);
                String ch = new String(Character.toChars(cp));
                Integer id = tokenToId.get(ch);
                if (id != null) {
                    tokens.add(id);
                } else {
                    // Fallback: try UTF-8 byte-level tokens (<0xXX>)
                    byte[] bytes = ch.getBytes(StandardCharsets.UTF_8);
                    for (byte b : bytes) {
                        String hexTok = String.format("<0x%02X>", b & 0xFF);
                        id = tokenToId.get(hexTok);
                        if (id != null) tokens.add(id);
                    }
                }
                i += Character.charCount(cp);
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
        if (v >= 127 && v <= 160) return String.valueOf((char) (v - 127 + 0x121));
        if (v == 173) return String.valueOf((char) 0x143);
        return String.valueOf((char) v);
    }

    @Override
    public String decode(int[] tokens) {
        StringBuilder sb = new StringBuilder();
        // SentencePiece-mode (Gemma 4): coalesce consecutive <0xHH> byte tokens so multi-byte
        // UTF-8 chars (e.g. 'é' = 0xC3 0xA9) decode correctly.
        if (!useGpt2ByteMapping) {
            java.util.ArrayList<Byte> pendingBytes = new java.util.ArrayList<>();
            for (int token : tokens) {
                if (token < 0 || token >= vocab.length) continue;
                String piece = vocab[token];
                if (piece.length() == 6 && piece.startsWith("<0x") && piece.charAt(5) == '>') {
                    try {
                        pendingBytes.add((byte) Integer.parseInt(piece.substring(3, 5), 16));
                        continue;
                    } catch (NumberFormatException ignored) { }
                }
                if (!pendingBytes.isEmpty()) {
                    byte[] arr = new byte[pendingBytes.size()];
                    for (int i = 0; i < arr.length; i++) arr[i] = pendingBytes.get(i);
                    sb.append(new String(arr, StandardCharsets.UTF_8));
                    pendingBytes.clear();
                }
                sb.append(decodeTokenPiece(piece));
            }
            if (!pendingBytes.isEmpty()) {
                byte[] arr = new byte[pendingBytes.size()];
                for (int i = 0; i < arr.length; i++) arr[i] = pendingBytes.get(i);
                sb.append(new String(arr, StandardCharsets.UTF_8));
            }
            return sb.toString();
        }
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
            // SentencePiece byte fallback token <0xHH>
            if (!useGpt2ByteMapping && piece.length() == 6 && piece.startsWith("<0x")) {
                try {
                    int byteVal = Integer.parseInt(piece.substring(3, 5), 16);
                    return new String(new byte[]{(byte) byteVal}, StandardCharsets.ISO_8859_1);
                } catch (NumberFormatException ignored) { }
            }
            return piece;
        }
        // Gemma 4 / SentencePiece-style tokens: vocab strings are already UTF-8.
        // Just replace U+2581 (▁) with space.
        if (!useGpt2ByteMapping) {
            return piece.replace('\u2581', ' ');
        }
        // Convert byte-level BPE back to actual bytes
        // Allocate extra space since chars may produce multi-byte UTF-8
        byte[] bytes = new byte[piece.length() * 4];
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
        if (c >= 33 && c <= 126) return c;       // printable ASCII
        if (c >= 161 && c <= 172) return c;       // identity-mapped (¡-¬)
        if (c >= 174 && c <= 255) return c;       // identity-mapped (®-ÿ)
        if (c == 0x0120) return 32;  // Ġ -> space
        if (c == 0x010A) return 10;  // Ċ -> newline
        if (c == 0x010D) return 13;
        if (c == 0x0109) return 9;
        if (c >= 0x100 && c <= 0x120) return c - 0x100; // bytes 0-32
        if (c >= 0x121 && c <= 0x142) return c - 0x121 + 127; // bytes 127-160
        if (c == 0x143) return 173; // soft hyphen
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
