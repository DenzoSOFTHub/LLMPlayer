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
    public static final String OUTPUT_BIAS = "output.bias"; // E12: optional, some Qwen2 variants

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

    // GLM-4.7-Flash / DeepSeek-V3: Q-LoRA decomposition (Q = Q_A * norm * Q_B)
    public static String attnQA(int layer) { return "blk." + layer + ".attn_q_a.weight"; }
    public static String attnQANorm(int layer) { return "blk." + layer + ".attn_q_a_norm.weight"; }
    public static String attnQB(int layer) { return "blk." + layer + ".attn_q_b.weight"; }

    // GLM-4.7-Flash / DeepSeek-V3: separate K_B and V_B (3D per-head tensors)
    public static String attnKB(int layer) { return "blk." + layer + ".attn_k_b.weight"; }
    public static String attnVB(int layer) { return "blk." + layer + ".attn_v_b.weight"; }

    // GLM-4.7-Flash: expert probability bias
    public static String expProbsBias(int layer) { return "blk." + layer + ".exp_probs_b.bias"; }

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

    // Qwen3.5 Gated DeltaNet (SSM) tensors
    public static String attnGate(int layer) { return "blk." + layer + ".attn_gate.weight"; }
    public static String ssmA(int layer) { return "blk." + layer + ".ssm_a"; }
    public static String ssmAlpha(int layer) { return "blk." + layer + ".ssm_alpha.weight"; }
    public static String ssmBeta(int layer) { return "blk." + layer + ".ssm_beta.weight"; }
    public static String ssmConv1d(int layer) { return "blk." + layer + ".ssm_conv1d.weight"; }
    public static String ssmDtBias(int layer) { return "blk." + layer + ".ssm_dt.bias"; }
    public static String ssmNorm(int layer) { return "blk." + layer + ".ssm_norm.weight"; }
    public static String ssmOut(int layer) { return "blk." + layer + ".ssm_out.weight"; }
    public static String postAttnNormWeight(int layer) { return "blk." + layer + ".post_attention_norm.weight"; }

    // Nemotron-H specific tensors
    public static String ssmIn(int layer) { return "blk." + layer + ".ssm_in.weight"; }
    public static String ssmD(int layer) { return "blk." + layer + ".ssm_d"; }
    public static String ssmConv1dBias(int layer) { return "blk." + layer + ".ssm_conv1d.bias"; }

    // Gemma 4 PLE (Per-Layer Embeddings)
    public static final String PER_LAYER_TOKEN_EMBD = "per_layer_token_embd.weight";
    public static final String PER_LAYER_MODEL_PROJ = "per_layer_model_proj.weight";
    public static final String PER_LAYER_PROJ_NORM = "per_layer_proj_norm.weight";
    public static String pleInpGate(int layer) { return "blk." + layer + ".inp_gate.weight"; }
    public static String pleProj(int layer) { return "blk." + layer + ".proj.weight"; }
    public static String plePostNorm(int layer) { return "blk." + layer + ".post_norm.weight"; }
    public static String layerOutputScale(int layer) { return "blk." + layer + ".layer_output_scale.weight"; }
}
