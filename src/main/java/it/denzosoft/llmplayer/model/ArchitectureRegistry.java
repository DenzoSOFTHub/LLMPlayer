package it.denzosoft.llmplayer.model;

/**
 * Maps architecture-specific tensor name patterns.
 * GGUF standardizes tensor names, so most architectures use the same names.
 */
public final class ArchitectureRegistry {

    private ArchitectureRegistry() {}

    // Standard GGUF tensor names (used by llama, qwen2, qwen3, etc.)
    public static final String TOKEN_EMBD = "token_embd.weight";
    public static final String OUTPUT_NORM = "output_norm.weight";
    public static final String OUTPUT = "output.weight";

    // Per-layer tensor name patterns (N = layer index)
    public static String attnNorm(int layer) { return "blk." + layer + ".attn_norm.weight"; }
    public static String ffnNorm(int layer) { return "blk." + layer + ".ffn_norm.weight"; }
    public static String attnQ(int layer) { return "blk." + layer + ".attn_q.weight"; }
    public static String attnK(int layer) { return "blk." + layer + ".attn_k.weight"; }
    public static String attnV(int layer) { return "blk." + layer + ".attn_v.weight"; }
    public static String attnOutput(int layer) { return "blk." + layer + ".attn_output.weight"; }
    public static String ffnGate(int layer) { return "blk." + layer + ".ffn_gate.weight"; }
    public static String ffnUp(int layer) { return "blk." + layer + ".ffn_up.weight"; }
    public static String ffnDown(int layer) { return "blk." + layer + ".ffn_down.weight"; }
    public static String ropeFreqs(int layer) { return "blk." + layer + ".rope_freqs.weight"; }

    // Phi3/Phi4: merged QKV
    public static String attnQKV(int layer) { return "blk." + layer + ".attn_qkv.weight"; }

    // Qwen2: attention biases
    public static String attnQBias(int layer) { return "blk." + layer + ".attn_q.bias"; }
    public static String attnKBias(int layer) { return "blk." + layer + ".attn_k.bias"; }
    public static String attnVBias(int layer) { return "blk." + layer + ".attn_v.bias"; }

    // Qwen3: per-head QK normalization
    public static String attnQNorm(int layer) { return "blk." + layer + ".attn_q_norm.weight"; }
    public static String attnKNorm(int layer) { return "blk." + layer + ".attn_k_norm.weight"; }

    // GLM4: post-normalization
    public static String postAttnNorm(int layer) { return "blk." + layer + ".post_attention_norm.weight"; }
    public static String postFfnNorm(int layer) { return "blk." + layer + ".post_ffw_norm.weight"; }

    // DeepSeek2 MLA (Multi-head Latent Attention)
    public static String attnKvAMqa(int layer) { return "blk." + layer + ".attn_kv_a_mqa.weight"; }
    public static String attnKvANorm(int layer) { return "blk." + layer + ".attn_kv_a_norm.weight"; }
    public static String attnKvB(int layer) { return "blk." + layer + ".attn_kv_b.weight"; }

    // Attention output bias (GPT-OSS)
    public static String attnOutputBias(int layer) { return "blk." + layer + ".attn_output.bias"; }

    // DeepSeek2 MoE
    public static String ffnGateInp(int layer) { return "blk." + layer + ".ffn_gate_inp.weight"; }
    public static String ffnGateInpBias(int layer) { return "blk." + layer + ".ffn_gate_inp.bias"; }
    public static String ffnGateExps(int layer) { return "blk." + layer + ".ffn_gate_exps.weight"; }
    public static String ffnUpExps(int layer) { return "blk." + layer + ".ffn_up_exps.weight"; }
    public static String ffnDownExps(int layer) { return "blk." + layer + ".ffn_down_exps.weight"; }
    public static String ffnGateShexp(int layer) { return "blk." + layer + ".ffn_gate_shexp.weight"; }
    public static String ffnUpShexp(int layer) { return "blk." + layer + ".ffn_up_shexp.weight"; }
    public static String ffnDownShexp(int layer) { return "blk." + layer + ".ffn_down_shexp.weight"; }

    // MoE expert biases (GPT-OSS)
    public static String ffnGateExpsBias(int layer) { return "blk." + layer + ".ffn_gate_exps.bias"; }
    public static String ffnUpExpsBias(int layer) { return "blk." + layer + ".ffn_up_exps.bias"; }
    public static String ffnDownExpsBias(int layer) { return "blk." + layer + ".ffn_down_exps.bias"; }

    // GPT-OSS attention sinks: per-head learned biases for softmax
    public static String attnSinks(int layer) { return "blk." + layer + ".attn_sinks.weight"; }
}
