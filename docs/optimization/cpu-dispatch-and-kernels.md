# CPU dispatch, kernels and batched prefill (2026-09-23)

This document records a pass over the CPU inference path aimed at three things: fixing defects that
kept the engine from using the hardware it runs on, raising single-core kernel throughput, and making
prompt processing (prefill) scale with prompt length the way it should. It continues
[`per-token-latency-analysis.md`](per-token-latency-analysis.md), which concluded that the CPU forward
pass ran at roughly 12 % of the machine's arithmetic peak and that the gap was "latency and
dependency stalls inside the kernels". That diagnosis turned out to be right, and it was not the
only problem.

## 1. Measurement conditions, and why most numbers here are A/B ratios

The machine was a VirtualBox guest with 8 vCPUs of an Intel Core Ultra 7 155H (AVX2, FMA and F16C;
no AVX-512 and no AVX-VNNI exposed to the guest), 15 GB of RAM, no GPU, and the models on a `vboxsf`
share. Throughout the session the host also ran unrelated Java workloads from another project, which
held the load average between 5 and 22. Wall-clock numbers therefore swing by 2–4× between
consecutive runs of the same binary, and an unpaired before/after comparison is meaningless.

Every claim below is therefore one of the following:

- an **in-process A/B**, where the variants run alternately in the same JVM on the same data, so they
  experience the same background load;
- a **back-to-back pair** of end-to-end runs where only one flag differs, reported as a ratio;
- a **correctness** result, such as relative error against the scalar reference kernel or
  byte-identical generated text at `--temperature 0`, which does not depend on load at all.

Absolute tok/s figures from this session should not be copied into `BENCHMARKS.md`.

## 2. Defects fixed

### 2.1 AVX-512 hosts fell back to the scalar kernels

Every SIMD tensor guarded its fast path with `FloatVector.SPECIES_PREFERRED.length() != 8`. The
intent was "no 256-bit vectors, use the scalar path", but on an AVX-512 machine the preferred species
has 16 lanes, so the test sent the entire model through the scalar `Q*_KFloatTensor.dot` fallback,
which is several times slower. The kernels use `SPECIES_256` explicitly and run correctly on AVX-512
hardware, so the guard is now `length() < 8` (only NEON-class 128-bit hardware falls back). Affected:
`SimdQ3_K`, `SimdQ4_K`, `SimdQ5_0`, `SimdQ5_K`, `SimdQ6_K`, `SimdQ8_0` and `SimdQ8KvOps`.

### 2.2 `--threads` was ignored by the default matmul path, and entirely in `--web` and GUI mode

`CLIRunner` translated `--threads` (default: physical cores) into the ForkJoin common-pool
parallelism. But on Java 25 the matmul went through `VirtualThreadMatmul`, which split every matrix
into `availableProcessors()` chunks regardless, so the setting never reached the hot path. In
addition, `--web` and the Swing GUI never passed through `CLIRunner`, so there the setting was not
applied at all. `LLMPlayer.main` now sets `-Dmatmul.threads` from `--threads` for every launch mode,
and the new pool (section 3) honours it, including `--threads 1`.

### 2.3 Gemma 2, 3, 3n and 4 used the wrong RoPE pairing

`ModelConfig` assigned `ROPE_TYPE_NORMAL` (rotate consecutive pairs) to the whole Gemma family.
llama.cpp uses NEOX (rotate the two halves of each head) for Gemma, because the GGUF converter
permutes Q/K only for Llama-style checkpoints. The symptom is characteristic: position 0 is the
identity rotation, so the first tokens of a prompt are processed correctly, and the error grows with
position. Short chats looked plausible — `docs/models/gemma-3-1b-it-Q4_K_M.md` attributed the
"somewhat coherent but imperfect" output to the 1B scale — while longer prompts broke down.

Measured with the top-1 next token after the chat-formatted prompt, on prefixes of one prompt:

| Model | Prompt tokens | NORMAL (before) | NEOX (after) |
|---|---:|---|---|
| Gemma-3-1B Q4_K_M | 32 | `Okay` (15.4) | `Okay` (46.5) |
| Gemma-3-1B Q4_K_M | 75 | `<end_of_turn>` (15.5) | `Okay` (49.9) |
| Gemma-3-1B Q4_K_M | 121 | `<end_of_turn>` (15.3) | `Okay` (45.4) |
| Gemma-2-2B IQ4_XS | 75 | `' '` (14.4) | `This` (22.2) |
| Gemma-2-2B IQ4_XS | 163 | `'  '` (12.1) | `Let` (21.1) |

Gemma 4 runs on its own engine but reads the same setting. On Gemma-4-E2B Q4_K_M with a 174-token
prompt, NORMAL pairing generated only an empty code fence and stopped (PPL 4.49), while NEOX answered
the question correctly ("This is a classic problem in distributed systems involving the 'dual
write'…", PPL 1.12). Gemma-3n-E4B, on the same engine, also answers correctly with NEOX (PPL 1.15).

With NORMAL pairing, Gemma-3-1B ended its turn without generating anything on any prompt longer than
roughly 50 tokens, and Gemma-2-2B degenerated into whitespace. With NEOX both answer normally at
every length tested, with much larger logit margins. Because every GPU forward pass reads the type
from the same `RoPE` object, the fix also applies to the CUDA paths.

## 3. `MatmulPool`: one persistent pool with dynamic scheduling

`MatmulPool` (base source root, Java 8 compatible) replaces `VirtualThreadMatmul` and the ForkJoin
parallel streams on the CPU path. It keeps `threads − 1` platform worker threads alive for the life
of the process. A dispatch publishes an immutable job; the rows are cut into about four chunks per
thread and claimed from an atomic counter, and the calling thread works too. Workers spin briefly
between dispatches (a decode step issues them back to back) and then park.

It fixes three things at once:

- **Static partitioning.** A matmul used to finish when its slowest of N equal chunks did. On a
  hybrid CPU (P-cores and E-cores), a VM with time-shared vCPUs, or a machine where another process
  holds a core, that stalls every other thread. With dynamic claiming, faster cores take more chunks.
- **Per-call task creation.** No futures or virtual threads are created per matmul.
- **Thread count.** It honours `--threads` (section 2.2).

The Q/K/V projection is dispatched as a single unit space of `qRows + 2·kvRows` rows, which removes
the GQA load imbalance described as F7 in `per-token-latency-analysis.md` without any special
boundary arithmetic. The per-head attention loops and the per-expert MoE loops of every engine use
`MatmulPool.forEach`, so they share the workers instead of competing with them from the common pool.
A dispatch attempted while the pool is busy (a nested parallel region, or a second generation
thread) runs inline on the caller, so the pool cannot deadlock.

**Measured:** in-process interleaved A/B on Llama-3.2-1B Q4_K_M decode, pool against the
virtual-thread dispatcher with the same kernels. A first run (eight repetitions, load average about
13) gave 305.6 against 368.1 ms/token, **1.20×**. Four later runs of twelve repetitions each (load
average 9–11) gave **1.28×, 1.51×, 1.77× and 1.70×**, with no consistent difference between
`matmul.spin` 20000 and 2000.

The pool is disabled when a GPU backend is initialised, together with `disableVirtualThreadMatmul`,
so GPU runs keep the dispatch they were validated with. `-Dmatmul.pool=false` restores the old
dispatchers on CPU, and `-Dmatmul.pool=force` keeps the pool on under a GPU backend.

## 4. Kernels

### 4.1 What limited the old kernels

The Q4_K kernel dequantised each weight (`q·d·sc − dmin·m`) and fed a single accumulator, so every
FMA waited on the previous one — two dependent FMAs per 16 weights. A pure FP32 dot product in L1 on
this machine runs at 15–20 Gelem/s on one core, while the Q4_K kernel managed 4–5. The FMA units
were not the limit; the dependency chain and the per-element unpacking were.

### 4.2 Q4_K: packed loads, raw accumulation, input sums hoisted per range

`FloatTensor.matmulRows(input, out, rowFrom, rowTo, cols)` is new: the unit of work each thread
receives, which lets a kernel do per-input preparation once per row range instead of once per row.
`SimdQ4_KFloatTensor.matmulRows` does three things:

1. It computes the input's per-32-element sums once. Then `dmin·m·Σx` becomes eight scalar FMAs per
   256-weight block, and the vector loop only accumulates the raw `q·x`.
2. It accumulates `q·x` per sub-block in short, independent chains and applies `d·sc` once per 32
   weights.
3. It loads the 32 quant bytes of a group as one 8-int vector instead of four 8-byte loads each widened
   with `B2I`. Lane `k` then holds bytes `4k..4k+3`, so shifting by `8p` exposes element `4k+p` with
   shifts and masks only. The input is permuted once per range to match
   (`xp[p·8+k] = x[4k+p]` within each sub-block), which leaves every sub-block's dot product
   unchanged.

In-process A/B on a synthetic 1024×2048 Q4_K matrix (single thread, three runs): original kernel
4.4 / 5.0 / 5.0 Gelem/s, new kernel **5.7 / 6.6 / 6.9**, +33–38 %. Relative error against the scalar
`Q4_KFloatTensor` is about 1e-6 on real model rows, which is FP32 summation-order noise. The
single-row `dot` (used by callers that do not go through `matmulRows`, such as the MoE expert loops)
uses the same raw-accumulation scheme without the permutation, and is about 10 % faster than before.

On a quiet machine (load average 1–4, measured later in the session) the difference is larger:
on Llama-1B's real tensors the packed `matmulRows` runs at 9.6–9.9 Gelem/s single-thread against
5.7–5.9 for the single-row `dot`, **1.7×**, and 37–38 Gelem/s with 8 threads.

### 4.2b Q5_K: the same packed scheme

Q5_K is the fourth most common format in the local model set after Q4_K, Q6_K and Q8_0: 37–51 % of
the weights of Gemma-4 E2B/E4B, 9–31 % across the Qwen3.5 family, 11–13 % in Phi-4. Its block is the
Q4_K layout plus a 32-byte `qh` array carrying the fifth bit, and its 176-byte block keeps every array
16-byte aligned, so the Q4_K packed kernel carries over directly: `qs` and `qh` are each loaded as one
8-int vector, and the fifth bit of element `4k+p` is bit `8p + 2·group` (low nibble) or
`8p + 2·group + 1` (high nibble) of lane `k` of `qh`. The shared input preparation (permutation,
sub-block sums, 6-bit scale unpacking) lives in `KQuantInput`, used by both kernels.

| Measurement (Llama-3.2-3B Q3_K_L, real tensors, quiet machine) | Before | After |
|---|---:|---:|
| `ffn_down` Q5_K, single thread | 4.52 Gelem/s (`dot`) | 5.49 (`matmulRows`) |
| `ffn_down` Q5_K, batched prefill kernel per token | 6.45 | 12.91 (2.0×) |
| `attn_v` Q5_K, batched prefill kernel per token | 6.13 | 11.76 (1.9×) |

Relative error against the scalar reference is about 1e-6. End to end, the same build with
`-Dmatmul.pool=false` (old `dot` path) against the default, one pair each at `--temperature 0` with a
174-token prompt: **Gemma-4-E2B decode 2.4 → 4.7 tok/s** and **Qwen3.5-0.8B 9.0 → 12.0 tok/s**, with
byte-identical generated text and identical perplexity in both.

### 4.2c IQ4_NL and IQ4_XS: codebook lookup in registers

IQ4_NL (97 % of Phi-3-mini IQ4_NL, 77 % of Llama-1B IQ4_NL) and IQ4_XS (77 % of Gemma-2-2B IQ4_XS)
store 4-bit indices into a fixed 16-entry non-linear codebook. The previous kernels built a scaled
copy of the codebook and two index arrays with a scalar loop for every 32 weights and then used a
gather, which ran at **1.0–1.3 Gelem/s** (IQ4_NL) and **0.85–0.97** (IQ4_XS), several times slower than
any other format.

The new kernels keep the codebook in two 8-lane `IntVector`s holding the float bit patterns of
entries 0–7 and 8–15. Each index selects from both halves (`selectFrom`, i.e. `vpermd`), a blend on
bit 3 picks the right one, and a reinterpret yields the float weights without any conversion. The
raw `codebook[q]·x` of a 32-weight (sub-)block is formed in two short chains, and the block scale
(`d`, or `d·(ls − 32)` for IQ4_XS) is applied once per 32 weights.

| Measurement (real tensors) | Before | After |
|---|---:|---:|
| Llama-1B IQ4_NL, single thread | 1.20–1.26 Gelem/s | 6.41–7.20 (5.4×) |
| Llama-1B IQ4_NL, 8 threads | 4.2–5.5 | 22.8–36.1 |
| Gemma-2-2B IQ4_XS, single thread | 0.85–0.97 | 3.91–4.21 (4.4×) |
| Llama-1B IQ4_NL end to end, decode | 2.6 tok/s | 8.0 tok/s, text identical |
| Phi-3-mini IQ4_NL end to end, decode | 0.6 tok/s | 1.7 tok/s, text identical |
| Gemma-2-2B IQ4_XS end to end (with the RoPE fix), 173-token prompt | 216 s, 0.9 tok/s, incoherent text, PPL 7.89 | 50 s, 2.7 tok/s, correct answer, PPL 1.43 |

A trap worth recording: with the codebook vectors held in `static final IntVector` fields, C2 failed
to intrinsify `selectFrom` ("missing constant" in `-XX:+PrintIntrinsics`), and the same kernel ran at
0.46 Gelem/s in most runs and 7.3 in an occasional lucky one. Building the two vectors from `int[]`
arrays at the top of `dot` and passing them down made it fast in 6 runs out of 6.

### 4.3 Q8_0: raw per-block accumulation

The old kernel multiplied the scale into every input vector and fed one FMA chain. The new one forms
the raw per-block `q·x` in two short chains, applies the scale once per 32 weights, and alternates
blocks between two accumulators. Synthetic A/B: 5.3 → 7.9 Gelem/s (+50 %). On Qwen3-0.6B's real
tensors: `ffn_down` 5.6 → 7.4–7.6 Gelem/s, `ffn_gate` 4.9 → 4.9–6.1, relative error about 1e-7.

A caution that belongs next to this kernel: a variant that differed only in using one accumulator
instead of two alternating ones ran at **1.5 Gelem/s**, five times slower, because the Vector API
intrinsics stopped being applied. Small changes of code shape can fall off this cliff without
warning, so every kernel change must be benchmarked in place, not just reasoned about.

### 4.4 Q6_K: lossless repack to an int8 layout

Q6_K stayed the slowest common kernel after the Q4_K and Q8_0 work, at 2.2–3 Gelem/s single-thread,
and Q4_K_M files use it for `output.weight` and for part of `ffn_down` and `attn_v`. On Llama-3.2-1B
that is about a third of the weights touched per token. Several reshuffles of the unpacking code
measured within noise or worse (section 4.5), because the cost is structural: each group of 8 weights
needs three widening loads and about ten integer operations to reassemble 4 + 2 bit fields.

The fix changes the data instead of the code. A Q6_K weight is `d · sc[j] · (q − 32)` with `q` in
0..63; every `q − 32` fits an int8, and `d` (fp16) times `sc` (int8) is exact in float32. So on first
use by the dense matmul paths, `SimdQ6_KFloatTensor` rewrites the tensor once, in parallel, into an
off-heap layout of 276 bytes per 256 weights: 256 int8 values already minus 32, the 16 int8 scales,
and `d` as a float. The kernel then has the Q8_0 shape (widen, convert, FMA, scale once per 16
weights). The weight values are bit-identical; only the summation order differs.

| Measurement | Original layout | Repacked |
|---|---:|---:|
| Synthetic A/B, single thread, 3 runs | 2.91 / 2.98 / 2.92 Gelem/s | 7.73 / 7.68 / 7.57 |
| Llama-1B `ffn_down`, single thread | 2.15 | 4.47 |
| Llama-1B `output`, single thread | 2.38 | 5.40 |
| Llama-1B `ffn_down`, 8 threads | 7.5–10.4 | 15.7 |
| Llama-1B `output`, 8 threads | 6.4–8.5 | 11.7 |
| Batched prefill kernel, per token | 4.3–4.75 | 9.4–9.8 |

**End to end it did not pay on this machine, so it is opt-in (`-Dq6k.repack=true`).** A paired A/B
of the new build with and without the repack (same JVM, alternating, ten repetitions, Llama-3.2-1B)
gave prefill 4.99 s → 4.13 s (1.21×) but decode slower in 9 of 10 pairs (median 735 → 1075 ms/token
under a load average of 13–16; in the quieter first pairs 178 → 199, 164 → 195, 105 → 191). With every
thread running, decode is bound by memory bandwidth, not by the kernel, and the repacked tensor is
31 % more bytes to stream per token. A faster single-core kernel does not help when the cores are
waiting on memory. The generated text with the repack enabled is byte-identical to the original
build. It may pay on hosts with much more memory bandwidth per core, or with fewer threads, and should
be re-measured there before being turned on.

The cost is memory: the copy is 1.31× the Q6_K bytes (about 280 MB for Llama-1B's output matrix).
The original mapped pages are no longer read by these paths, so the OS can drop them, but that is up
to the page cache. The repack is therefore skipped for lazy larger-than-RAM loads (`mmap.advise`
`random` or `sequential`), where the extra copy would compete with the page cache for the working set,
and it is off unless `-Dq6k.repack=true`. `dot()`, which the MoE expert slices and a few other
callers use, keeps reading the original layout, so MoE expert tensors are never duplicated.

### 4.5 Tried and rejected

| Idea | Result | Why |
|---|---|---|
| Integer path (activations quantised to int8, llama.cpp style) | 4.55 vs 5.38 Gelem/s — slower | The Vector API has no widening byte multiply-add (`pmaddubsw`, `vpdpbusd`), so the int8 product must be emulated through 16-bit lanes plus conversions, which costs more than it saves. This is the path llama.cpp uses, and it is the right one on the GPU (`cuda.dp4a`), but in Java on CPU the float path wins. |
| "Magic float" conversion (`0x4B000000 \| q`, subtract 2²³) instead of `I2F` | Faster but wrong | Cancelling 2²³·Σx in FP32 destroys precision (−528.0 vs −516.86 on the check row). |
| Q4_K through `B2F` on 32-byte vectors | 2.7–2.9 Gelem/s — slower | `convertShape` with part > 0 compiles to extra shuffles. |
| Q8_0-style raw accumulation for Q5_0 | 0.8–1.2 vs 3.0–5.6 Gelem/s — slower | Same idea that gave Q8_0 +50 %, but in this code shape the intrinsics stopped applying, and building the shift vector locally (the IQ4 fix) did not recover it. Reverted; the existing Q5_0 kernel runs at about 5.4 Gelem/s on a quiet machine. |
| Four-token batched kernels for IQ4_NL / IQ4_XS | 0.47–0.60× the single-token path | Same lookup as the fast `dot`, but in the multi-token shape the intrinsics stopped applying, first with the accumulators in a `FloatVector[]` (boxed) and still after moving them to locals. Removed; batched prefill for these formats uses the default `matmulRowsBatch`, which runs the new `dot` per token over a cache-resident row range. |
| Packed loads for Q6_K | 0.74–2.2 Gelem/s vs 2.5–3.0 — slower | The 210-byte block is not 4-byte aligned, and several intrinsics (`load`, `blend`, `broadcast`) stopped being applied in that code shape. Reverted. |

## 5. Batched prefill for the standard engine

### 5.1 What it does

`InferenceEngine.forwardPrefill(state, tokens, from, to)` processes the prompt layer by layer in
chunks of `-Dprefill.batch` tokens (default 64). Inside a layer, every projection (Q/K/V, `wo`,
gate/up, down) runs as one **multi-token matmul**, `FloatTensor.matmulRowsBatch`. Attention still runs
token by token in position order through the existing `Attention.forwardFromProjections`, so token `t`
sees exactly the KV entries `0..t` it sees in the one-token path. The per-token math is unchanged;
only the summation order inside the matmul kernels differs.

The default `matmulRowsBatch` processes one token at a time over a row range small enough to stay in
cache, so each weight row comes from memory once per range instead of once per token. `SimdQ4_K`,
`SimdQ6_K` and `SimdQ8_0` override it with four-token tiles: each weight vector is unpacked,
converted and scaled once and then FMA'd against four inputs.

Single-thread kernel throughput per token, batched against the per-token path on real tensors
(16 tokens):

| Tensor | Per-token | Batched | Ratio |
|---|---:|---:|---:|
| Llama-1B `attn_q` Q4_K | 4.26 Gelem/s | 5.71 | 1.34× |
| Llama-1B `ffn_gate` Q4_K | 4.23 | 7.36 | 1.74× |
| Llama-1B `attn_v` Q6_K | 1.39 | 4.75 | 3.41× |
| Llama-1B `ffn_down` Q6_K | 1.07 | 4.26 | 3.97× |
| Qwen3-0.6B `ffn_gate` Q8_0 | 4.09 | 7.44 | 1.82× |
| Qwen3-0.6B `ffn_down` Q8_0 | 3.98 | 7.31 | 1.84× |

Relative error of the batched kernels against the per-token path is between 1e-7 and 1e-6.

Later in the session four-token kernels were added for Q5_K (2.0×, section 4.2b) and Q3_K (reusing its existing per-8-weight dequantisation: 2.2–2.6× per token on Llama-3.2-3B Q3_K_L, relative error 2e-7).

### 5.2 Coverage

Batched prefill covers pre-norm layers (RMSNorm or LayerNorm), optional post-attention and post-FFN
norms (Gemma 2/3), Granite residual scaling, Q/K/V and output biases, QK-norm, sliding-window
attention and iRoPE/NoPE layers, because those parts all live in the per-token core it reuses. The
following fall back to the token-by-token loop automatically: merged QKV (Phi-3/4), packed gate/up
(GLM4), parallel FFN (Command-R), post-norm-only layers (OLMo2), the opt-in flash attention, any
GPU-resident forward pass, and the lazy larger-than-RAM prefetcher. Disable it with
`-Dprefill.batched=false`, the same flag the Qwen3 MoE and DeepSeek2 layer-outer prefill already use.

### 5.3 Measured

Back-to-back pairs with `-Dprefill.batched` false and true, same binary, `--temperature 0`, CPU only.
The totals include model load and 16–24 generated tokens, so they understate the prefill speedup.

| Model | Prompt | Off | On | Ratio | Generated text |
|---|---:|---:|---:|---:|---|
| Llama-3.2-1B Q4_K_M | 292 tokens | 117.7 s | 31.4 s | 3.7× | identical |
| Qwen3-1.7B Q8_0 | 294 | 96.6 s | 47.8 s | 2.0× | identical |
| Qwen2.5-Coder-3B Q4_K_M | 290 | 745.8 s | 217.9 s | 3.4× | identical |
| Gemma-3-1B Q4_K_M (post-norms, GELU, SWA) | 174 | 27.4 s | 13.5 s | 2.0× | identical |
| SmolLM3 Q4_K_M | 173 | 71.1 s | 48.2 s | 1.5× | identical |
| Llama-3.2-3B Q3_K_L (Q3_K and Q5_K batched kernels) | 174 | 160.8 s | 68.4 s | 2.35× | identical |
| Phi-4-mini Q4_K_M (merged QKV → falls back) | 170 | 69.7 s | 83.0 s | same path | identical |

The Phi-4 pair runs the same per-token path twice, so its timing difference is background noise and a useful reminder of how large that noise is. The Qwen3 row predates the Q8_0 batched kernel, so it shows only the cache-locality effect of the
default `matmulRowsBatch`. Perplexity and `avg_nll` were identical in every pair.

### 5.4 Qwen3.5 (DeltaNet hybrid)

`Qwen35InferenceEngine.forwardPrefill` applies the same scheme to the hybrid engine. Both layer types
were first split, without changing behaviour, into projections, an order-dependent core and an output
projection: `deltaNetCore` (conv1d, gates, DeltaNet recurrence, SSM norm) and `attentionCore` (Q/gate
deinterleave, QK norm, RoPE, KV store, attention, output gate). The one-token path calls the pieces in
the same order as before; the only reordering is that the alpha, beta and output-gate projections are
now computed before the conv1d rather than after it, which is equivalent because all three depend only
on the normed input. After the split, the generated text of Qwen3.5-0.8B was byte-identical to the
previous build. The batched path runs every projection — DeltaNet QKV, alpha, beta, gate and
`ssm_out`; attention Q+gate, K, V and O; FFN gate, up and down — as multi-token matmuls, and feeds the
cores token by token in position order.

| Model (174–176-token prompt, 24 generated) | Off | On | Generated text |
|---|---:|---:|---|
| Qwen3.5-0.8B Q4_K_M | 27.2 s | 15.5 s (1.75×) | identical |
| Qwen3.5-4B Q4_K_M | 53.5 s | 37.4 s (1.43×) | identical |

### 5.4b Gemma 4 (per-layer embeddings)

`Gemma4InferenceEngine.forwardPrefill` (2026-09-24) brings the same scheme to Gemma 4 E2B/E4B and
the dense 12B/31B, and `LLMEngine.generateGemma4` now calls it instead of looping over `forward`.
As with Qwen3.5, the one-token path was first split without changing its arithmetic:
`attentionCore` holds the order-dependent part of an attention layer (QK-norm, V-norm, RoPE, KV store
and the attention itself, including the "V = raw K" case of the global layers that ship no `wv`), and
`computePleInput` became the `per_layer_model_proj` matmul followed by `combinePle`, which scales,
per-layer-normalises and combines one token's projection with its token-identity embedding.

The batched path runs every projection as a multi-token matmul: the PLE model projection for the
whole chunk, Q/K/V (as one fused unit space when the layer has its own `wv`), Wo, FFN gate and up
(fused), down, and the per-layer PLE gate and PLE projection. `attentionCore` is fed token by token in
position order. Gemma 3n keeps the one-token loop, because its AltUp streams would need their own
batched predict/correct step, and so does the GPU-resident pass.

Correctness, greedy decoding with `prefill.batched` off and on in separate processes:

| Model | Prompt | Chunking | Generated text and PPL |
|---|---:|---|---|
| Gemma-4-E2B Q4_K_M | 314 tokens | 64 (five chunks) | identical |
| Gemma-4-12B Q4_K_M | 104 tokens | `-Dprefill.batch=16` (seven chunks) | identical |

Gemma-3n-E4B, which keeps the one-token loop but shares the refactored PLE code, generated the same
text and PPL as the previous build on the 104-token prompt.

The 12B run exercises the global layers without `wv` and the per-layer KV head count. Its timings
are not reported, because with 7 GB free for a 7.1 GB model the first run paged the weights in from
the shared folder.

Speed on Gemma-4-E2B, ~300-token prompt, load average 6–13:

| Measurement | Token by token | Batched | Ratio |
|---|---:|---:|---:|
| One JVM, same build, modes alternated, 4 repetitions | 44.9–53.7 s | 23.9–27.7 s | **1.9×** |
| Two builds in one JVM (`ABG`, see 5b), median of 3 | 96.4 s | 33.9 s | **2.8×** |
| CLI end to end, 314 prompt + 24 generated tokens | 135.5 s | 72.9 s | 1.9× |

**Decode after a batched prefill.** The first separate-process runs suggested that decode was slower
after a batched prefill: across twelve runs at load average 9–13, the median was 2.7 tok/s after a
batched prefill against 4.2 after a token-by-token one, bimodally (2.5–2.7 or 3.6–4.7 tok/s). The
closer look did not confirm a lasting effect:

- JFR of the decode window showed the same hot methods in both modes, all C2 code, with none of the
  Vector API Java fallbacks that marked the problem of section 5.5. `-XX:+PrintCompilation` put the
  Q4_K and Q6_K single-token kernels at tier 4 about 2.5 s into the run, during the warm-up, long
  before decode.
- In one JVM, with the two modes alternating on fresh states, decode measured 177–220 ms/token after
  a batched prefill against 148–240 after a token-by-token one (medians 186 and 171), within this
  machine's noise.
- Timing decode in blocks of eight tokens, one mode per process, showed no persistent penalty: the
  batched steady state matched or beat the other in two of three pairs, and diverged in the third
  only as the load rose during the run. The first 16 tokens after a batched prefill were slower
  (about 500 against 400 ms/token), which fits the one-off compilation of the decode-only callers.
- With the warm-up disabled (`-Dmatmul.warmup.ms=0`) decode was the slowest configuration in all
  three runs (1.2–2.4 tok/s), so `warmUpRows` still matters here.

The start-up transient in a fresh process is therefore the only decode cost found. It is a few
seconds at most and is dwarfed by the prefill saving.

### 5.4c MoE engines: Qwen3 MoE, GPT-OSS and DeepSeek2

`Qwen3MoEInferenceEngine` and `DeepSeek2InferenceEngine` already prefilled layer-outer (see F2 in
`per-token-latency-analysis.md`): same one-token matmuls, reordered so that each layer's experts are
requested once per chunk instead of once per token. Since 2026-09-24 the CPU path also batches the
matmuls themselves:

- **Attention.** Qwen3 MoE: Q/K/V as one fused multi-token projection, `attentionCore` (biases,
  QK-norm, RoPE, KV store, attention with sinks and sliding window) token by token, then Wo batched.
  DeepSeek2 (`MLAAttention.forwardBatch`): Q or Q-LoRA A/norm/B, `wkvA`, the latent norm and the
  combined `wkvB` batched; RoPE, KV store and attention token by token; Wo batched. The GLM-style
  separate K_B/V_B decompression stays per token.
- **Routed experts.** Every token of the chunk is routed with the one-token router code. The
  (token, slot) pairs are then grouped by expert (a counting sort, `ExpertViews.groupByExpert`), and
  each distinct expert runs once per projection over all its tokens with `matmulRowsBatch`, on its
  SSD-cache slice or on a per-expert view of the 3D tensor (`ExpertViews`: `TensorData.slice` wrapped
  by `TensorFactory.create`, no copy). Experts run in parallel on the pool. With 64 tokens and top-8
  of 128 experts an expert sees about four tokens on average — the four-input kernel shape. Each
  token's expert outputs are then summed in slot order, exactly as in the one-token path.
- **Shared expert and leading dense layers** run as multi-token matmuls over the chunk.
- **SSD cache.** Experts are prepared in groups of 16 (`-Dmoe.prefill.cache.group`); the next
  group's reads run on a background thread while the current group computes, and with the cache
  active the chunk is 256 tokens instead of 64 (`ssd-streaming-cache.md`, "Three prefill changes"). Hit and miss counts are now per (chunk, expert) during prefill rather than per
  (token, slot), so the hit rates of the two paths are not comparable; the bytes read are.

Decode is unchanged: routed experts still run with one `dot` per row. Running them through
`matmulRows` on the views instead — which is 33–38 % faster for Q4_K on one core — measured slower
end to end on Qwen3-Coder-30B streamed from SSD. With a 19-token prompt and 24 generated tokens, three
rotating repetitions per configuration, decode was 0.6 / 0.7 / 0.3 tok/s with it against
0.9 / 1.0 / 0.8 with the `dot` loop and 0.9 / 1.1 / 0.7 for the previous build. The cause was not
identified; the change was dropped. Since batched prefill never runs the experts' `dot` kernel,
`FloatTensor.warmUpDot` warms it alongside `warmUpRows`. `-Dmoe.expert.rows=false` restores the
layer-outer prefill with one-token matmuls.

Measured in separate processes, greedy decoding, previous path (`-Dmoe.expert.rows=false`) against
the batched one, prefill time = total minus the decode tokens at the reported decode rate:

| Model | Prompt | Prefill before → after | Read from disk | Generated text and PPL |
|---|---:|---|---|---|
| Qwen3-Coder-30B-A3B Q4_K_M, 2 GB cache | 103 | 163 → 61 s, 134 → 61 s, 137 → 68 s; final build 139 → 102 s, 81 → 50 s (**1.4–2.7×**) | 37–41 GB → 19–23 GB | identical in all five pairs |
| DeepSeek-Coder-V2-Lite Q4_K_M (preloaded) | 102 | 204 → 112 s, 116 → 61 s (**1.8–1.9×**) | — | identical |
| sonar-oss-20b (GPT-OSS, MXFP4), 2 GB cache | 113 | 238 → 165 s (**1.44×**) | 47.8 → 21.1 GB | identical |

The first three Qwen3-Coder pairs ran a build that also used `matmulRows` for decode (see above);
the last two ran the final build. In those two, decode after the batched prefill measured 0.7 and
1.3 tok/s against 1.5 and 1.3 for the previous path. The decode code is now identical to the previous
build's, and the dedicated decode test above found the two equal, so the first pair is most likely
load noise — but it is one of two, and worth re-checking on an idle machine.

GPT-OSS gains least because MXFP4 has no `matmulRowsBatch` override: the default runs the tokens one
by one over each expert, so the batch saves reads but not dequantisation. A multi-token MXFP4 kernel
is the obvious next step for it. Load average during these runs was 6–15, so single timings vary;
the prefill ratios held in every pair.

### 5.5 Batched prefill made the following decode slower, and why

After batched prefill was in place, a careful look at decode — three generations per process,
several processes — showed decode after a batched prefill running at 3.8–7.4 tok/s on Llama-1B
against 10–12 after a token-by-token prefill, in a bimodal pattern (most runs slow, some fast), even
in the second and third generations of a process. The fault was not in the arithmetic. Three
contributing effects were found:

1. **JIT timing (the main cause).** Batched prefill never runs the single-token kernels
   (`matmulRows` and the kernels under it), so their C2 compilation started only when decode began.
   These are large methods full of Vector API calls, compiled while every core was busy with the
   matmul and with background load, and until C2 finished, decode ran on C1 code, which does not
   intrinsify the Vector API. JFR showed it directly: `IntVector.lanewiseShiftTemplate` and
   `AbstractVector.convert0` — the Java fallbacks of vector shifts and conversions — as hot methods
   under the single-token Q4_K kernel. Running 48 throwaway decode steps before the first generation
   brought decode after a batched prefill back to 8–11.8 tok/s. The fix is
   `FloatTensor.warmUpRows`: before the first batched prefill, each weight class's single-token
   kernel runs on 64 rows, on one thread while the pool is idle, until an iteration costs less than
   half of the first one (the compiled code is installed) or `-Dmatmul.warmup.ms` (1500 ms) passes.
   After it, decode after a batched prefill measured in line with decode after a token-by-token
   prefill in three paired runs, while the batched runs kept their much shorter totals
   (9.7–19.8 s against 18–60 s for three generations each).
2. **Allocation churn.** The packed Q4_K and Q5_K kernels allocated the permuted input and the
   sub-block sums on every row range: about 30 MB per decoded token, and 968 young collections over
   a short three-generation run, each stopping the matmul threads. They now reuse per-thread buffers
   (`KQuantInput.Scratch`; the pool's workers are long-lived platform threads). Same run: 20
   collections.
3. **Loop-variable shift counts.** The batched Q4_K and Q5_K kernels shifted by `8 * p`; they now
   advance a shifted copy by a constant each step, so every vector shift count is a compile-time
   constant. This was a precaution suggested by the JFR evidence rather than a separately measured
   win. Separately, the short last group of a batched chunk (fewer than four tokens) now goes through
   the four-token kernel with its last input repeated, so the single-token kernels are never compiled
   from those rare calls.

The lesson for future kernel work on this engine: a change that removes a code path from the hot
loop (as batched prefill did for the single-token kernels) also removes the JIT warm-up that path was
providing, and on this JVM the price is paid by the next phase.

## 5b. End-to-end A/B of the whole change set

The most robust end-to-end measurement available on this machine loads **both builds into one JVM**,
each through its own classloader, and alternates them on every repetition, so both see the same
background load (harness `ABX`, kept in the session scratchpad; it drives `LLMEngine.generate` with a
greedy sampler and `maxTokens=1` for prefill, and `forwardSingleToken` after warm-up for decode).
Baseline is the code at the start of the session; "new" is everything above with the Q6_K repack off (its default).
Prompt of about 290 tokens, 3–4 repetitions per build, medians, load average 10–16:

| Model | Prefill base → new | Ratio | Decode base → new (ms/token) | Ratio |
|---|---:|---:|---:|---:|
| Llama-3.2-1B Q4_K_M | 54.5 s → 16.7 s | **3.26×** | 232 → 196 | **1.19×** |
| Qwen3-1.7B Q8_0 | 68.4 s → 16.3 s | **4.21×** | 244 → 252 | 0.97× (noise: runs spread 92–287) |
| Qwen2.5-Coder-3B Q4_K_M | 190.4 s → 54.8 s | **3.47×** | 754 → 444 | **1.70×** |
| Llama-3.1-8B Q4_K_M | 680.9 s → 134.0 s | **5.08×** | 1191 → 969 | **1.23×** |

The prefill ratios are stable across every repetition (for example Llama-1B: 50–65 s against 15–18
s). **The decode column of this harness is not trustworthy**, and a later run showed why: at load
average 2 it reported Llama-1B decode *slower* in the new build (0.76×), while every other method
disagreed. Both builds share the JDK's Vector API classes, so the JIT's type profiles in those shared
methods see two unrelated callers, and whichever build is compiled second can lose inlining. The
large prefill ratios survive this; a 20 % decode effect does not.

The decode claim therefore rests on two measurements that avoid the problem:

- **Separate processes, alternating order, sign test.** Six pairs of Llama-3.2-1B decode runs
  (32 tokens, best of 3 per run, load average 5.8–7.5): the new build won **6 of 6**, 112–124 ms/token
  against 74–93, medians 119.2 → 86.2 ms/token, **1.38×**.
- **One build, dispatcher toggled in process.** Pool against the previous virtual-thread dispatcher,
  same kernels, twelve interleaved repetitions per run, four runs: **1.28×, 1.51×, 1.77×, 1.70×**.
  Varying `matmul.spin` between 20000 and 2000 made no consistent difference.

### 5c. Final end-to-end suite (2026-09-24)

A last run compared the build at the start of the session with the final one through the CLI:
eight models covering every format and engine touched, a ~300-token prompt, 64 generated tokens,
`--temperature 0`, CPU only, two repetitions per build in alternating order. The machine was idle
for the first pair only (load average 1.7–2.2) and then back under external load (5–18), so the
decode column is noisy; the totals are dominated by prefill and hold across repetitions.

| Model | Total, original build | Total, final build | Ratio (per pair) |
|---|---:|---:|---:|
| Llama-3.2-1B Q4_K_M | 30.4 s / 135.7 s | 14.0 s / 33.9 s | 2.2× (quiet pair; decode 9.3 → 13.2 tok/s) / 4.0× |
| Qwen3-1.7B Q8_0 | 161.8 / 136.9 | 66.5 / 55.0 | 2.4× / 2.5× |
| Qwen2.5-Coder-3B Q4_K_M | 355.3 / 347.5 | 116.7 / 125.6 | 3.0× / 2.8× |
| Phi-3-mini IQ4_NL | 868.7 / 534.6 | 100.9 / 102.7 | 8.6× / 5.2× |
| Gemma-2-2B IQ4_XS | 772.2 / 836.6 | 320.8 / 313.0 | 2.4× / 2.7× |
| Gemma-4-E2B Q4_K_M | 151.7 / 176.0 | 133.5 / 174.4 | 1.1× / 1.0× |
| Qwen3.5-4B Q4_K_M | 1078.2 / 260.2 | 289.2 / 135.1 | 3.7× / 1.9× |
| Llama-3.1-8B Q4_K_M | 461.0 / 435.5 | 187.0 / 196.2 | 2.5× / 2.2× |

Gemma-4 gains least because its engine still prefilled token by token when this suite ran (its
batched prefill followed; see section 5.4b). The original build also stopped after 52 tokens there,
a symptom of the RoPE defect (section 2.3), so its row compares different amounts of work.

## 6. New and changed properties

| Property | Default | Meaning |
|---|---|---|
| `matmul.pool` | `true` | Use `MatmulPool` on the CPU path. `false` restores the previous dispatchers; `force` keeps the pool on under a GPU backend. |
| `matmul.threads` | from `--threads` | Pool size. The launcher sets it from `--threads` (default: physical cores) in every launch mode; for programmatic use without the launcher, the ForkJoin common-pool parallelism or else all logical CPUs. |
| `matmul.chunks.per.thread` | `4` | Target chunks per thread per dispatch: more gives better balance on uneven cores, fewer means less claiming overhead. |
| `matmul.min.chunk` | `32` | Minimum rows per chunk for single-token matmuls (kernels do some per-chunk input preparation). |
| `matmul.spin` | `20000` | Spin iterations a worker waits for the next dispatch before parking. |
| `prefill.batched` | `true` | Now also enables batched prefill in the standard engine, in Qwen3.5 and in Gemma 4 (not Gemma 3n). |
| `moe.expert.rows` | `true` | Multi-token matmuls in the prefill of Qwen3 MoE, GPT-OSS and DeepSeek2, with routed experts grouped by expert (section 5.4c). `false` keeps the layer-outer prefill with one-token matmuls. |
| `moe.prefill.cache.group` | `16` | Experts prepared per SSD-cache call during batched MoE prefill. |
| `moe.prefill.stream.batch` | `256` | Prefill chunk of the MoE engines when the SSD cache is active (bytes read are proportional to the number of chunks; see `ssd-streaming-cache.md`). |
| `moe.prefill.overlap` | `true` | Read the next expert group from disk while the current group is computed. |
| `q6k.repack` | `false` | Opt-in lossless int8 repack of Q6_K tensors for the dense matmul paths (section 4.4): 2.2× kernel, 1.21× prefill, but slower decode on a bandwidth-bound machine. Never applied to lazy larger-than-RAM loads. |
| `prefill.batch` | `64` | Tokens per prefill chunk; also sizes the per-token buffers (`PrefillBatch`). |
| `matmul.warmup.ms` | `1500` | Upper bound per weight class on the single-token kernel warm-up that runs before the first batched prefill (section 5.5); `0` disables it. |

## 7. What remains

- **Q3_K has a `matmulRowsBatch` override but no single-token `matmulRows`**, so decode on Q3_K
  models (55 % of Llama-3.2-3B Q3_K_L) still runs the generic per-row path; the packed scheme should
  apply with the 3-bit layout's own bit arithmetic. The batch kernel also allocates small buffers
  per call.
- **A multi-token MXFP4 kernel.** GPT-OSS experts are MXFP4, which falls back to the generic
  `matmulRowsBatch`; batched MoE prefill therefore saves reads there but not dequantisation
  (section 5.4c).
- **Gemma 3n, Nemotron-H / Granite Hybrid, LFM2 and Falcon-H1** still prefill token by token.
- **Clean-machine numbers.** Everything here should be re-measured on an idle machine and, ideally,
  on an AVX-512 host, where fix 2.1 alone should be worth several times.
