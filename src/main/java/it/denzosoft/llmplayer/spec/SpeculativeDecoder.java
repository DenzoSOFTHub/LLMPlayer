package it.denzosoft.llmplayer.spec;

import it.denzosoft.llmplayer.api.GenerationRequest;
import it.denzosoft.llmplayer.api.GenerationResponse;
import it.denzosoft.llmplayer.api.LLMEngine;
import it.denzosoft.llmplayer.api.StreamingCallback;
import it.denzosoft.llmplayer.tokenizer.SpecialTokens;
import it.denzosoft.llmplayer.tokenizer.Tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Speculative decoding (Leviathan et al. 2023). Uses a small <b>draft</b> model
 * to propose K candidate tokens, then a large <b>target</b> model verifies them
 * via rejection sampling. Accepted tokens come from the target's distribution
 * (statistically equivalent to direct sampling from the target).
 *
 * <h2>Important: this is a STANDALONE class</h2>
 *
 * It does NOT modify any existing API. Both {@link LLMEngine#generate} and
 * {@link LLMEngine#forwardSingleToken} continue to work exactly as before.
 * SpeculativeDecoder uses {@code forwardSingleToken} on TWO separate engine
 * instances (target + draft) and applies the speculative-decoding algorithm
 * on top.
 *
 * <h2>Speedup characteristics</h2>
 *
 * The win depends on two factors:
 * <ul>
 *   <li><b>Draft/target time ratio</b> {@code t_draft/t_target}: smaller is
 *       better. For Qwen3-0.6B drafting Qwen3-8B: ratio ≈ 0.05.</li>
 *   <li><b>Acceptance rate</b>: how often the draft's token matches what the
 *       target would have sampled. Typically 50-80% on coherent text.</li>
 * </ul>
 *
 * <h3>Verification mode trade-off</h3>
 *
 * <b>This implementation uses SEQUENTIAL verification</b> — the target runs K
 * separate {@code forward} calls to verify K draft tokens. Maximum theoretical
 * speedup with all-accepted: {@code (K+1) * t_target / (K * t_target + K * t_draft)}.
 * For K=4, ratio 0.1: ~1.14×. <b>Marginal in practice.</b>
 *
 * <p>Real speedup (2-3×) requires <b>batched verification</b> — one target
 * forward pass that processes K query tokens against the existing KV history.
 * That requires a {@code forwardBatch(int[] tokens, int startPos)} API that
 * does not exist yet. See the followup task in the project tracker.
 *
 * <p>This implementation focuses on <b>algorithmic correctness</b>: the output
 * distribution is provably equivalent to direct target sampling.
 */
public final class SpeculativeDecoder {

    private final LLMEngine target;
    private final LLMEngine draft;
    private final int speculationDepth;   // K: number of draft tokens per round
    private final Random rng;

    // Per-call statistics (reset on each generate())
    private long totalDraftTokens;
    private long totalAcceptedTokens;
    private long totalSpeculationRounds;

    public SpeculativeDecoder(LLMEngine target, LLMEngine draft, int speculationDepth, long seed) {
        if (target == null || draft == null) {
            throw new IllegalArgumentException("target and draft engines must be non-null");
        }
        if (speculationDepth < 1 || speculationDepth > 32) {
            throw new IllegalArgumentException("speculationDepth must be in [1, 32], got " + speculationDepth);
        }
        // Vocabularies must match — otherwise token IDs don't compare.
        int tVocab = target.getModelInfo().vocabSize();
        int dVocab = draft.getModelInfo().vocabSize();
        if (tVocab != dVocab) {
            throw new IllegalArgumentException(String.format(
                "Target and draft vocabularies differ: target=%d, draft=%d. " +
                "Both models must share the same tokenizer.", tVocab, dVocab));
        }
        this.target = target;
        this.draft = draft;
        this.speculationDepth = speculationDepth;
        this.rng = new Random(seed);
    }

    public SpeculativeDecoder(LLMEngine target, LLMEngine draft, int speculationDepth) {
        this(target, draft, speculationDepth, System.nanoTime());
    }

    /**
     * Generate text using speculative decoding.
     *
     * @param request standard {@link GenerationRequest}; only {@code prompt},
     *                {@code maxTokens}, {@code useChat} are honored. Sampling
     *                parameters (temp, top_p, etc.) are ignored — speculative
     *                decoding requires direct probability comparison so we
     *                always use the underlying target distribution after a
     *                fixed temperature scaling.
     * @param callback optional streaming callback (called per accepted token).
     * @return the generated response.
     */
    public GenerationResponse generate(GenerationRequest request, StreamingCallback callback) {
        long t0 = System.nanoTime();
        totalDraftTokens = 0;
        totalAcceptedTokens = 0;
        totalSpeculationRounds = 0;

        Tokenizer tok = target.getTokenizer();
        SpecialTokens specials = target.getSpecialTokens();

        // 1. Format prompt (use target's chat template — both models share vocab)
        String formatted = request.useChat()
            ? target.getChatTemplate().formatUserMessage(request.prompt())
            : request.prompt();
        int[] promptTokens = tok.encode(formatted);
        if (specials.shouldAddBos() && specials.getBosId() >= 0) {
            int[] withBos = new int[promptTokens.length + 1];
            withBos[0] = specials.getBosId();
            System.arraycopy(promptTokens, 0, withBos, 1, promptTokens.length);
            promptTokens = withBos;
        }
        int promptLen = promptTokens.length;

        // 2. Prefill BOTH models on the prompt. Use forwardSingleToken so each
        //    engine accumulates the prompt into its own KV cache.
        float[] targetLogits = null;
        float[] draftLogits = null;
        for (int i = 0; i < promptLen; i++) {
            targetLogits = target.forwardSingleToken(promptTokens[i], i);
            draftLogits = draft.forwardSingleToken(promptTokens[i], i);
        }
        // After prefill: position = promptLen. Both models hold logits for the next
        // (first generated) token at index promptLen.
        int position = promptLen;
        int prevAcceptedToken = promptTokens[promptLen - 1]; // last input token

        // 3. Speculation loop
        StringBuilder output = new StringBuilder();
        List<Integer> generated = new ArrayList<>();
        int maxNew = request.maxTokens();
        int K = speculationDepth;

        outer:
        while (generated.size() < maxNew) {
            totalSpeculationRounds++;

            // ---- Phase A: draft proposes K candidate tokens ----
            int[] draftTokens = new int[K];
            float[][] draftProbs = new float[K][];   // probability vectors at each draft step
            // The draft's "current" logits are draftLogits (last computed).
            float[] curDraftLogits = draftLogits;
            for (int k = 0; k < K; k++) {
                float[] p = softmax(curDraftLogits);
                int y = sampleFromDistribution(p);
                draftTokens[k] = y;
                draftProbs[k] = p;
                // Forward draft to get next-step logits (unless this is the last in the batch
                // — but we still need to forward to maintain KV cache for next round if accepted)
                curDraftLogits = draft.forwardSingleToken(y, position + k);
            }
            totalDraftTokens += K;

            // ---- Phase B: target verifies K candidates ----
            // We need P_target(y_k) at positions p+0..p+K-1, plus an extra at p+K for bonus.
            // - Logits at position p (for verifying y_0) we ALREADY have in targetLogits.
            // - Logits at positions p+1..p+K we get by feeding draft tokens y_0..y_{K-1}.
            //
            // Use forwardBatch to allow future batched implementation. Currently sequential
            // (same cost as forwardSingleToken K times) — see API javadoc.
            float[][] targetProbs = new float[K + 1][];   // +1 for bonus position
            targetProbs[0] = softmax(targetLogits);
            float[][] batchLogits = target.forwardBatch(draftTokens, position);
            for (int k = 0; k < K; k++) {
                targetProbs[k + 1] = softmax(batchLogits[k]);
            }
            // After this batch, target's KV cache holds positions 0..position+K-1
            // (each draftTokens[k] was committed at position+k).

            // ---- Phase C: rejection sampling ----
            int accepted = 0;
            int finalToken = -1;       // the extra/correction token for this round
            boolean allAccepted = true;
            for (int k = 0; k < K; k++) {
                int y = draftTokens[k];
                float pTarget = targetProbs[k][y];
                float pDraft  = draftProbs[k][y];
                float ratio = (pDraft > 0f) ? Math.min(1f, pTarget / pDraft) : 0f;
                if (rng.nextFloat() < ratio) {
                    accepted++;
                    generated.add(y);
                    output.append(tok.decode(y));
                    if (callback != null && !callback.onToken(tok.decode(y), y)) break outer;
                    if (specials.isEos(y) || generated.size() >= maxNew) break outer;
                } else {
                    // Reject. Sample correction from adjusted distribution: max(0, p_t - p_d).
                    float[] adjusted = new float[targetProbs[k].length];
                    float sum = 0f;
                    for (int i = 0; i < adjusted.length; i++) {
                        float v = targetProbs[k][i] - draftProbs[k][i];
                        if (v > 0f) { adjusted[i] = v; sum += v; }
                    }
                    if (sum <= 0f) {
                        // Distributions identical (numerically) — fall back to target sampling.
                        finalToken = sampleFromDistribution(targetProbs[k]);
                    } else {
                        for (int i = 0; i < adjusted.length; i++) adjusted[i] /= sum;
                        finalToken = sampleFromDistribution(adjusted);
                    }
                    allAccepted = false;
                    break;
                }
            }
            totalAcceptedTokens += accepted;

            if (allAccepted && finalToken < 0) {
                // All K accepted — sample bonus token from target's K+1 position logits.
                finalToken = sampleFromDistribution(targetProbs[K]);
            }

            if (finalToken >= 0 && generated.size() < maxNew) {
                generated.add(finalToken);
                output.append(tok.decode(finalToken));
                if (callback != null && !callback.onToken(tok.decode(finalToken), finalToken)) break;
                if (specials.isEos(finalToken)) break;
            }

            // ---- Phase D: synchronize KV cache state ----
            // After this round, generated tokens are: accepted draft tokens + 1 final/bonus token.
            // The engines' KV caches naturally hold:
            //   target: positions 0..position+K-1 (we ran K forwards)
            //   draft:  positions 0..position+K-1 (we ran K forwards)
            // We accepted `accepted` draft tokens then added 1 extra token (`finalToken`).
            // The next position we need to compute logits for is: position + accepted + 1.
            //
            // Target needs forward(finalToken, position + accepted) → gives logits at position + accepted + 1.
            // Draft  needs forward(finalToken, position + accepted) → gives logits at position + accepted + 1.
            //
            // For positions > position + accepted, the existing KV entries are STALE but unused
            // (next forward call will overwrite them when we eventually re-reach those positions).
            int newPos = position + accepted;
            if (allAccepted) {
                // We accepted K draft tokens, all KV cache writes were valid.
                // Plus the bonus token at position+K — need to commit it via a forward.
                if (finalToken >= 0) {
                    targetLogits = target.forwardSingleToken(finalToken, position + K);
                    draftLogits  = draft.forwardSingleToken(finalToken, position + K);
                    prevAcceptedToken = finalToken;
                    position = position + K + 1;
                } else {
                    position = position + K;
                }
            } else {
                // Rejected at position newPos. KV writes at positions newPos..position+K-1 are STALE.
                // Need to re-forward the correction token to overwrite KV at slot newPos.
                if (finalToken >= 0) {
                    targetLogits = target.forwardSingleToken(finalToken, newPos);
                    draftLogits  = draft.forwardSingleToken(finalToken, newPos);
                    prevAcceptedToken = finalToken;
                    position = newPos + 1;
                } else {
                    position = newPos;
                }
            }
        }

        long elapsedNs = System.nanoTime() - t0;
        double tokPerSec = generated.size() * 1e9 / elapsedNs;
        long elapsedMs = elapsedNs / 1_000_000;

        // Print stats to stderr (caller can ignore)
        double acceptRate = totalDraftTokens > 0
            ? (100.0 * totalAcceptedTokens / totalDraftTokens) : 0.0;
        System.err.printf("[speculative] generated=%d, rounds=%d, draft=%d, accepted=%d (%.1f%% accept rate), tok/s=%.1f%n",
            generated.size(), totalSpeculationRounds, totalDraftTokens, totalAcceptedTokens, acceptRate, tokPerSec);

        return new GenerationResponse(output.toString(), generated.size(), promptLen,
            tokPerSec, elapsedMs, java.util.Collections.emptyList());
    }

    /** Acceptance rate of the most recent generate() call (0.0 if none). */
    public double getAcceptanceRate() {
        return totalDraftTokens > 0 ? (double) totalAcceptedTokens / totalDraftTokens : 0.0;
    }

    public long getTotalDraftTokens() { return totalDraftTokens; }
    public long getTotalAcceptedTokens() { return totalAcceptedTokens; }
    public long getTotalSpeculationRounds() { return totalSpeculationRounds; }

    // ----- Sampling helpers -----

    /** Numerically stable softmax. */
    private static float[] softmax(float[] logits) {
        float[] out = new float[logits.length];
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) if (v > max) max = v;
        float sum = 0f;
        for (int i = 0; i < logits.length; i++) {
            out[i] = (float) Math.exp(logits[i] - max);
            sum += out[i];
        }
        if (sum <= 0f || Float.isNaN(sum)) {
            // Degenerate: uniform fallback to avoid NaN propagation
            float u = 1f / logits.length;
            for (int i = 0; i < logits.length; i++) out[i] = u;
            return out;
        }
        float inv = 1f / sum;
        for (int i = 0; i < logits.length; i++) out[i] *= inv;
        return out;
    }

    /** Sample token id from a probability distribution. */
    private int sampleFromDistribution(float[] probs) {
        float r = rng.nextFloat();
        float cum = 0f;
        for (int i = 0; i < probs.length; i++) {
            cum += probs[i];
            if (r < cum) return i;
        }
        return probs.length - 1;  // fallback for floating-point edge case
    }
}
