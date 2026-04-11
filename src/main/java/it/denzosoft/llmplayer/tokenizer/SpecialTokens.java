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
    private final boolean addBos;  // whether to prepend BOS token
    private String modelType;     // tokenizer model type (e.g. "gemma4")

    public SpecialTokens(int bosId, int eosId, int padId, int eotId, int[] additionalEosIds,
                          Map<String, Integer> specialTokenMap) {
        this(bosId, eosId, padId, eotId, additionalEosIds, specialTokenMap, true);
    }

    public SpecialTokens(int bosId, int eosId, int padId, int eotId, int[] additionalEosIds,
                          Map<String, Integer> specialTokenMap, boolean addBos) {
        this.bosId = bosId;
        this.addBos = addBos;
        this.eosId = eosId;
        this.padId = padId;
        this.eotId = eotId;
        this.additionalEosIds = additionalEosIds;
        this.specialTokenMap = specialTokenMap;
    }

    public int getBosId() { return bosId; }
    public boolean shouldAddBos() { return addBos; }
    public String getModelType() { return modelType; }
    public void setModelType(String type) { this.modelType = type; }
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
        // Respect add_bos_token flag; default true unless explicitly false or no bos_token_id defined
        boolean addBos = metadata.getBoolean("tokenizer.ggml.add_bos_token",
            metadata.getInt("tokenizer.ggml.bos_token_id", -1) >= 0);

        // Load additional EOS token IDs from metadata array
        int[] additionalEos = metadata.getIntArray("tokenizer.ggml.eos_token_ids");

        // Augment with explicit-EOT lookups: many GGUFs only set eos_token_id to <eos> but
        // their chat template emits a different turn-terminator (e.g. Gemma's <end_of_turn>,
        // GPT-OSS's <|end|>, Llama 3's <|eot_id|>). Without scanning the vocab for these,
        // generation runs past the model's actual end-of-response and loops into garbage.
        // We always scan, then merge with whatever the metadata array provided.
        String[] tokens = metadata.getStringArray("tokenizer.ggml.tokens");
        if (tokens != null) {
            java.util.LinkedHashSet<Integer> stopSet = new java.util.LinkedHashSet<>();
            if (additionalEos != null) {
                for (int id : additionalEos) stopSet.add(id);
            }
            // Known turn-terminator tokens across families. Each is added if found and
            // distinct from the primary eos/eot we already track.
            String[] candidates = {
                "<end_of_turn>",      // Gemma 2 / 3 / 3n / 4
                "<|end_of_turn|>",    // some Gemma variants
                "<|end|>",            // GPT-OSS / Phi
                "<|eot_id|>",         // Llama 3 / SmolLM3
                "<|im_end|>",         // Qwen / ChatML
                "<|endoftext|>"       // GPT-2 family fallback
            };
            for (String name : candidates) {
                for (int i = 0; i < tokens.length; i++) {
                    if (name.equals(tokens[i]) && i != eosId && i != eotId) {
                        stopSet.add(i);
                        break;
                    }
                }
            }
            if (!stopSet.isEmpty()) {
                additionalEos = new int[stopSet.size()];
                int j = 0;
                for (int id : stopSet) additionalEos[j++] = id;
            }
        }

        Map<String, Integer> specialMap = new HashMap<>();
        specialMap.put("bos", bosId);
        specialMap.put("eos", eosId);
        if (padId >= 0) specialMap.put("pad", padId);
        if (eotId >= 0) specialMap.put("eot", eotId);

        return new SpecialTokens(bosId, eosId, padId, eotId, additionalEos, specialMap, addBos);
    }
}
