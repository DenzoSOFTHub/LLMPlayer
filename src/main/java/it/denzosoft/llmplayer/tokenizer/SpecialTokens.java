package it.denzosoft.llmplayer.tokenizer;

import it.denzosoft.llmplayer.gguf.GGUFMetadata;

import java.util.HashMap;
import java.util.Map;

public class SpecialTokens {
    private final int bosId;
    private final int eosId;
    private final int padId;
    private final int eotId;  // end-of-turn token (Llama 3: <|eot_id|>)
    private final int[] additionalEosIds; // extra stop tokens (GPT-OSS: <|end|>)
    private final Map<String, Integer> specialTokenMap;

    public SpecialTokens(int bosId, int eosId, int padId, int eotId, int[] additionalEosIds,
                          Map<String, Integer> specialTokenMap) {
        this.bosId = bosId;
        this.eosId = eosId;
        this.padId = padId;
        this.eotId = eotId;
        this.additionalEosIds = additionalEosIds;
        this.specialTokenMap = specialTokenMap;
    }

    public int getBosId() { return bosId; }
    public int getEosId() { return eosId; }
    public int getPadId() { return padId; }
    public int getEotId() { return eotId; }

    public int getTokenId(String name) {
        return specialTokenMap.getOrDefault(name, -1);
    }

    public boolean isEos(int tokenId) {
        if (tokenId == eosId || tokenId == eotId) return true;
        if (additionalEosIds != null) {
            for (int id : additionalEosIds) {
                if (tokenId == id) return true;
            }
        }
        return false;
    }

    public static SpecialTokens fromMetadata(GGUFMetadata metadata) {
        int bosId = metadata.getInt("tokenizer.ggml.bos_token_id", 1);
        int eosId = metadata.getInt("tokenizer.ggml.eos_token_id", 2);
        int padId = metadata.getInt("tokenizer.ggml.padding_token_id", -1);
        int eotId = metadata.getInt("tokenizer.ggml.eot_token_id", -1);

        // Load additional EOS token IDs from metadata array
        int[] additionalEos = metadata.getIntArray("tokenizer.ggml.eos_token_ids");

        // If no explicit additional EOS, scan vocabulary for common end-of-message tokens
        if (additionalEos == null) {
            String[] tokens = metadata.getStringArray("tokenizer.ggml.tokens");
            if (tokens != null) {
                java.util.List<Integer> found = new java.util.ArrayList<>();
                for (int i = 0; i < tokens.length; i++) {
                    if ("<|end|>".equals(tokens[i]) && i != eosId && i != eotId) {
                        found.add(i);
                    }
                }
                if (!found.isEmpty()) {
                    additionalEos = new int[found.size()];
                    for (int i = 0; i < found.size(); i++) {
                        additionalEos[i] = found.get(i);
                    }
                }
            }
        }

        Map<String, Integer> specialMap = new HashMap<>();
        specialMap.put("bos", bosId);
        specialMap.put("eos", eosId);
        if (padId >= 0) specialMap.put("pad", padId);
        if (eotId >= 0) specialMap.put("eot", eotId);

        return new SpecialTokens(bosId, eosId, padId, eotId, additionalEos, specialMap);
    }
}
