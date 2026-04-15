# Speculative Decoding (Tier 3)

> **Status (2026-04-13)**:
> - **Phase 1 — algorithm**: ✅ implemented and algorithmically correct (`SpeculativeDecoder` class).
> - **Phase 2 — `forwardBatch` API**: ✅ added to `LLMEngine` with sequential implementation (no per-layer batching yet).
> - **Phase 3 — actual batched kernels**: ❌ not yet implemented. This is the missing piece for real speedup.

## What it is

Per [Leviathan et al. 2023](https://arxiv.org/abs/2211.17192), speculative decoding uses a small **draft** model to propose `K` candidate tokens, then a large **target** model verifies them via rejection sampling. Accepted tokens are statistically equivalent to direct sampling from the target.

The win comes from:
- Draft is small/fast (e.g. Qwen3-0.6B at ~50 tok/s)
- Target verifies all K in **one batched forward pass** (~1 target forward instead of K)
- Net: target runs `M+1` times less often, where `M` is the average accept count

## How to use

```bash
java ... it.denzosoft.llmplayer.LLMPlayer \
  --model gguf/Qwen3-8B-Q4_K_M.gguf \
  --draft-model gguf/Qwen3-0.6B-Q8_0.gguf \
  --spec-depth 4 \
  --prompt "..." \
  --max-tokens 80 --gpu-layers -1
```

**Requirements**:
- Both models must share the same vocabulary (same tokenizer family).
- Currently tested: Qwen3-0.6B / Qwen3-1.7B drafting Qwen3-8B (all use Qwen3 tokenizer, vocab 151,936).

The CLI flag is **opt-in**. Without `--draft-model`, normal generation is used (zero impact on existing flow).

## Implementation

- `it.denzosoft.llmplayer.spec.SpeculativeDecoder` — standalone class
- Uses `LLMEngine.forwardSingleToken(token, position)` on TWO independent engine instances
- KV cache is naturally managed by position semantics (no explicit truncation needed): writing at slot `p` overwrites previous content; the model uses positions `0..currentPos`
- Algorithm: standard rejection sampling
  1. Draft samples `K` candidates `y_0..y_{K-1}` (with their probabilities)
  2. Target computes its probabilities at the same positions
  3. For each k: accept `y_k` with probability `min(1, p_target(y_k) / p_draft(y_k))`; on first rejection, sample a correction from `max(0, p_target - p_draft)` normalized
  4. If all K accepted, sample a bonus token from the target's `K+1` position logits

## Measured results (sequential verification)

Hardware: RTX 4050 Laptop GPU, target Qwen3-8B Q4_K_M on GPU, draft on CPU.

### Test 1: short (8-token answer)

| Config | tok/s | Accept rate | Output |
|---|---:|---:|---|
| Baseline (no spec) | 9.8 | — | "The capital of France is Paris." |
| Spec K=4, draft Qwen3-0.6B | 1.1 | 33% | "The capital of France is Paris." |
| Spec K=4, draft Qwen3-1.7B | 0.7 | 50% | "The capital of France is Paris." |

### Test 2: longer (80-token paragraph)

| Config | tok/s | Accept rate | Output |
|---|---:|---:|---|
| Spec K=4, draft Qwen3-0.6B | 3.2 | **67.9%** | "Photosynthesis is the process by which..." |

**Output is correct and matches what the target alone would produce** (modulo sampling noise). Algorithm correctness verified.

## Why no speedup yet

Sequential verification: the target runs `K` separate `forward()` calls to verify `K` draft tokens. Each call goes through the full transformer (matmul, attention, etc.).

For K=4 with 67% accept rate:
- Tokens produced per round: `K * accept_rate + 1` = `4 * 0.67 + 1` = 3.68
- Target forwards per round: K = 4
- Target forwards per token: 4 / 3.68 = 1.09 (vs 1.0 baseline)
- Plus K draft forwards per round (CPU, slow)

Result: speculative decoding with sequential verification is **slower** than direct generation, despite the high accept rate.

## Phase 2 (added 2026-04-13) — `LLMEngine.forwardBatch` API

A new public API was added to `LLMEngine`:

```java
public float[][] forwardBatch(int[] tokens, int startPosition)
```

Semantics: feed `tokens[0]` at position `startPosition`, then `tokens[1]` at `startPosition+1`, etc., and return `float[K][vocabSize]` of next-token logits at each position.

`SpeculativeDecoder` now uses this API for the verification phase. The contract is stable, so future batched implementations will plug in transparently.

**Current implementation** (`forwardBatch`): K sequential calls to `forwardSingleToken`. **Same cost as before** — the API is in place but does not yet provide speedup.

## Why no speedup yet — the real bottleneck

For Qwen3-8B forward pass (~100ms on RTX 4050):
- ~17 ms is bandwidth-bound (model weight reads)
- ~83 ms is overhead (kernel launches, attention compute, sync)

The lm_head matmul alone is only ~1.6 ms (310 MB Q4_K weights / 192 GB/s). Even batching the lm_head perfectly across K=4 would save ~5 ms per round → 0.5% wall-clock improvement. **Not worth the engineering.**

The REAL win requires batching the per-layer matmuls (FFN gate/up/down, attention QKV/Wo). These are the dominant cost. A batched kernel reads weights ONCE for K input vectors, amortizing bandwidth.

**For Qwen3-8B Q4_K_M:**
- Per-token weight reads: 2.4 GB FFN + 0.6 GB attention + 0.3 GB lm_head ≈ 3.3 GB
- At 192 GB/s: theoretical minimum 17 ms/token
- We measure 100 ms/token → 17% bandwidth utilization (consistent with earlier analysis)

**With K=4 batched:**
- Per-round weight reads: ~3.3 GB (same as 1 token, weights reused for 4)
- Plus draft cost: 4 × ~10ms = 40 ms (CPU-bound, can pipeline with target)
- Per-round time: ~17 ms target + 40 ms draft = 57 ms → 4 tokens / 57 ms = 70 tok/s effective

vs current 9.8 tok/s baseline = **7× theoretical speedup** (assuming 100% accept rate).

Realistic with 67% accept rate: ~3-4× actual speedup.

## What's needed (Phase 3) — batched per-layer kernels

To actually achieve the speedup, the per-layer compute must process K tokens together:

### Required new CUDA kernels

| Kernel | Current | Batched K=4 |
|---|---|---|
| `matmul_q4_k` | reads `[dim] FP32` input, writes `[rows] FP32` | reads `[K, dim] FP32` inputs, writes `[K, rows] FP32` outputs |
| `matmul_q5_k` | same | same shape extension |
| `matmul_q6_k` | same | same shape extension |
| `attention_kernel` | 1 query, N keys/values | K queries, N+K-1 keys/values (causal mask) |
| `rmsnorm_fused` | 1 vector | K vectors |
| `rope` | 1 vector | K vectors at K positions |

### Required Java changes

- `InferenceState` needs to hold K parallel residual streams (`float[K][dim]` instead of `float[dim]`)
- `InferenceEngine.forwardBatchInternal()` — new method that processes K tokens through all layers
- `Attention` — multi-query path that handles K queries + causal masking
- KV cache writes K positions at once

### Effort estimate

- Phase 3 minimum (Q4_K only, standard `InferenceEngine` only): **2 weeks**
- Phase 3 full (all engine paths, all quant types): **1-2 months**

For now, the API contract is in place. Real speedup awaits Phase 3.

## Code locations

- Algorithm: `src/main/java/it/denzosoft/llmplayer/spec/SpeculativeDecoder.java`
- CLI integration: `CLIRunner.runSpeculative()` (called when `--draft-model` is set)
- CLI options: `CLIOptions.draftModelPath`, `CLIOptions.speculationDepth`

## Future work

1. **Batched verification**: implement `forwardBatch()` for at least the standard `InferenceEngine` path. Estimated 1-2 weeks of work; this unlocks the 2-3× speedup.
2. **Tree-based speculation** (Medusa-style): instead of a linear K-token chain, generate a tree of K candidates and verify in parallel. Slightly higher accept rate, more code complexity.
3. **Same-model self-speculation** (look-ahead): use the same model in two ways (forward N positions, then verify). Avoids the need for a separate draft model.
