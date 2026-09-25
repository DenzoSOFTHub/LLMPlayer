# SSD streaming with a hot-expert cache

This document analyses whether LLMPlayer can run models substantially larger than physical RAM by
streaming weights from SSD, keeping only a hot working set resident. It reports the measurements
taken on the reference box, describes the design in three levers of increasing cost, and records
what was implemented and what was deliberately deferred.

The short answer is that it is feasible, that it applies to **MoE architectures only**, and that the
first and cheapest lever is not a cache at all — it is fixing the read granularity.

Related: [`placement-autotuning.md`](placement-autotuning.md) for the VRAM placement cost model that
this continues, and [`autotuning-heuristics.md`](autotuning-heuristics.md) for the decision rules
already in the code.

## 1. The problem, measured

The reference box now runs models several times larger than its RAM:

| Resource | Value |
|---|---|
| RAM | 7.8 GB total, 3–4 GB available |
| VRAM | 6 GB (RTX 4050 Laptop) |
| Physical cores | 4 (Intel Ultra 7 155H) |
| Model volume | `vboxsf` share, 3.7 TB |

Local MoE models above that RAM budget: Qwen3-Coder-30B-A3B (18.6 GB), Qwen3.5-35B-A3B (22 GB),
MiniMax-M2-55B (24 GB), GLM-4.7-Flash (18.3 GB).

Running Qwen3-Coder-30B-A3B Q4_K_M today, CPU-only, on v1.16.1 defaults (lazy mmap plus
`MADV_RANDOM`), gives:

```
Model loaded in 391ms
Tokens: 9 prompt + 5 generated in 152257ms (0.1 tok/s)
Perplexity: 0.99 (EXCELLENT)   Aggregate: 0.87 (EXCELLENT)
```

Two things matter in that output. The model **works** — it loads in 391 ms and generates coherent
text at excellent perplexity, so nothing is functionally broken. And it takes roughly **30 seconds
per token**, which makes it unusable in practice. The problem is entirely I/O.

There is also a usability gate in front of it: the CLI currently refuses models above the RAM
budget with an interactive prompt, and the warning text is stale.

```
WARNING: This model is too large for the available RAM.
Loading will cause disk swap, resulting in extremely slow performance.
This configuration is not recommended. Continue? [y/N]
```

Since v1.16.0 there is no swap involved — `LLMEngine.load` skips the preload and reads pages from
the model file on demand. The message predates that change.

## 2. Why it is I/O, and where the I/O goes wrong

Measured on the actual model volume with in-process `pread` at random offsets, fixed seed. An
initial attempt with `dd` was discarded because roughly 3.5 ms of process-spawn overhead per
invocation dominated the result.

| Read granularity | MB/s | IOPS |
|---|---|---|
| 4 KB | **15.9** | 4058 |
| 64 KB | 76.4 | 1223 |
| 256 KB | 205.1 | 820 |
| 1 MB | 395.6 | 396 |
| 2.5 MB (one expert slice) | **481.7** | 194 |
| 8 MB | 521.0 | 65 |
| sequential | 613 | — |

The cost is per request, not per byte. Bandwidth scales almost linearly with request size until it
saturates around 500 MB/s at a few megabytes. **The spread between page-fault granularity and
expert granularity is about 30×**, and that single fact is what makes this whole feature worth
building.

Now compare that with how the weights are actually read. When `LLMEngine.load` skips the preload it
sets `mmap.advise=random`, and `MemorySegmentTensorData.mapFile` issues `madvise(MADV_RANDOM)` over
the whole mapping, which switches kernel read-ahead **off**.

That advice was chosen in v1.16.1 by reasoning about sparsity — cold MoE experts are touched
sparsely, so read-ahead looks like pure waste — and it was never measured against a model larger
than RAM, since the verification sweep for that release covered only small models. The reasoning
holds *between* experts and fails *within* one. A routed expert is a single contiguous slice of
about 2.6 MB. With read-ahead disabled, faulting it costs roughly 650 independent 4 KB page faults
served at ~16 MB/s, instead of one large read served at ~482 MB/s.

## 3. The arithmetic

For Qwen3-Coder-30B-A3B Q4_K_M: 48 layers, 128 experts, top-8 routing, `expertFfnDim` 768, `dim`
2048. One expert's gate, up and down slices are each 2048 × 768 elements, which at Q4_K
(144 bytes per 256-element block) is about 0.88 MB per projection, so **about 2.65 MB per expert**.
The routed experts account for roughly 88–90 % of the file.

Cold demand per token is therefore 48 layers × 8 experts × 2.65 MB ≈ **1.0 GB**.

| Regime | s/token | tok/s |
|---|---|---|
| measured today (4 KB faults, read-ahead off) | ~30 | **0.1** |
| expert-granular reads, no cache | ~2.1 | ~0.5 |
| plus a hot set serving 79 % of routing | ~0.44 | ~2.3 |

The last row is not a guess about how concentrated routing is. It was already measured in this
repository with `-Dmoe.routing.stats` on this exact model (128 experts, top-8, 48 MoE layers, 34k
selections):

| Hot set | Share of routing captured |
|---|---|
| top-8 (6 % of experts) | 38.5 % |
| top-16 (12.5 %) | 57.3 % |
| **top-32 (25 %)** | **79.2 %** |
| top-64 (50 %) | 96.2 % |

Routing is moderately concentrated: the top 32 experts capture 3.2× their uniform share. That is
the empirical basis for a hot-set cache, and the instrumentation that produces it is the direct
equivalent of Colibri's learned `.coli_usage` profile.

## 4. Reference designs

**[Colibri](https://github.com/JustVugg/colibri)** is the closest match, and notably shares
LLMPlayer's philosophy — a pure C engine with zero dependencies, streaming MoE experts from disk. It
runs GLM-5.2 (744B, ~372 GB int4) on consumer hardware by treating storage as the bottom tier of a
memory hierarchy:

- The dense part (attention, shared experts, embeddings, ~17B params at int4) stays **resident in
  RAM**; the 19,456 routed experts live on disk and stream on demand; an optional VRAM tier holds
  the hottest experts.
- Admission and eviction use a **per-layer LRU plus a learned pinned hot store**. Usage history is
  persisted in `.coli_usage` and updated every turn, and the hottest experts are pinned
  automatically. Colibri describes cache warmth as "the single most valuable knob" on
  storage-constrained runs.
- I/O is explicit `pread` with a batched async pool, optionally `O_DIRECT` to bypass the page cache
  (reported +34 % decode on some hardware). The three projection matrices of an expert are stored
  adjacently and read in **one** `pread` — the same granularity argument as section 2.
- A router-lookahead thread prefetches the next layer's experts; Colibri reports routing is
  **71.6 % predictable one layer ahead**.
- Reported throughput: ~1.8 tok/s warm on a 128 GB CPU-only desktop, 1.07 tok/s on a single
  RTX 5070 Ti, and 0.05–0.1 tok/s cold on a 25 GB box. Decode is disk-bound on most machines.

**[LLM in a flash](https://arxiv.org/abs/2312.11514)** (Apple, 2023) is the academic reference for
the same problem on dense models. Its two techniques are *windowing* (reuse neurons activated by
recent tokens, so only the delta is loaded) and *row-column bundling* (store the up-projection row
and down-projection column together to increase the size of each flash read). The second is exactly
the granularity lever; the first depends on activation sparsity that LLMPlayer's architectures do
not currently expose.

The **KTransformers** placement strategy (attention on GPU, experts on CPU) is already implemented
here as MoE-optimized placement, and the **AirLLM** next-layer prefetch pattern is already
implemented as `LayerPrefetcher`.

## 5. What LLMPlayer already has

A surprising amount of the substrate exists. This work is an increment, not a new subsystem.

| Piece | Where | State |
|---|---|---|
| Lazy mmap above 85 % of RAM | `LLMEngine.load` (preload decision) | done, v1.16.0 |
| Pattern-aware `madvise` | `LLMEngine.load` → `MemorySegmentTensorData.mapFile` | done, v1.16.1 |
| Routing-frequency profile | `Qwen3MoEInferenceEngine`, `-Dmoe.routing.stats` | done, v1.16.0 |
| Next-layer prefetch (dense) | `LayerPrefetcher` | done, dense engine only |
| Slot-based LRU expert cache into **VRAM** | `ExpertGpuCache` | MXFP4 validated; K-quant path incorrect, gated off |
| MoE-optimized placement | `GpuConfig.moeOptimized` | done |
| Per-expert offset matmul | `GraniteExpertGpu` | done, requires full VRAM residency |

What does **not** exist: any `pread` or direct-I/O path for weights (mmap is the only mechanism), any
`MADV_WILLNEED` or `MADV_DONTNEED` call (`madvise` is issued exactly once, over the entire file, at
map time), any host-RAM eviction or residency accounting, any prefetcher for the MoE engines, and
any disk-I/O or cache-hit metric.

## 6. Scope: this is a MoE feature

**Dense models larger than RAM cannot benefit from a cache**, and the document should be explicit
about that rather than leave it as an assumption. Every weight of a dense model is needed for every
token, so the hit rate of any cache over the portion that does not fit is structurally zero. The
only available lever is overlapping I/O with compute, which `LayerPrefetcher` already does. Adding a
cache there would consume RAM to no effect.

Streaming plus caching pays off precisely where the model is large but the *active* fraction per
token is small — which is the definition of MoE, and why Colibri targets MoE exclusively.

In scope: `Qwen3MoEInferenceEngine` (Qwen3-Coder, Llama 4 MoE, GLM4 MoE, GPT-OSS) and
`DeepSeek2InferenceEngine` via `MoEFFN` (DeepSeek2, GLM-4.7-Flash). Granite Hybrid MoE goes through
`GraniteExpertGpu`, which requires the whole 3D expert tensor to be VRAM-resident, and is a separate
case.

## 7. Design — three levers

### L0 — expert-granular read-ahead (implemented)

Immediately after the router selects top-K, issue one `madvise(MADV_WILLNEED)` per selected expert
slice, before the expert matmuls run. `MADV_WILLNEED` overrides the blanket `MADV_RANDOM` for those
ranges, and it is asynchronous, so queueing all K up front lets the kernel run the reads
concurrently and overlap them with compute rather than serialising fault by fault.

This is a hint, not a data path: it cannot change results, it allocates nothing, and it leaves every
tensor class untouched.

Implementation, mirroring the existing `preload()` precedent:

1. `TensorData` gains `default void adviseWillNeed(long offset, long length) {}` — a no-op, so the
   Java 8 path and `ByteBufferTensorData` are unaffected.
2. `MemorySegmentTensorData` overrides it with a `madvise` downcall. The `madvise` method handle and
   the page size are resolved once into statics, because this sits on the per-token path (K experts
   × 3 projections × 48 layers per token) and building a downcall handle per call would be pure
   overhead. `madvise` requires a page-aligned start address, so the range is rounded down and the
   length extended to match.
3. `ExpertPrefetch` computes each expert's byte range with the same arithmetic as
   `MoEFFN.expertMatmul` and `ExpertGpuCache.uploadExpertSlice`, and skips negative expert indices
   (the backfilled slot from a NaN router row).
4. Call sites: `Qwen3MoEInferenceEngine.moeFFN` and `MoEFFN.forward`, both immediately after top-K
   selection and before the expert loop.

Gated by `-Dmoe.expert.willneed`: `auto` (default) enables it exactly when the lazy >RAM MoE path is
active (`mmap.advise=random`), `true` and `false` force it. Models that fit RAM never trigger it.

### L1 — `MappedExpertCache`: RAM cache filled by explicit reads (implemented)

This was originally scoped as optional — build it only if L0 left a gap worth closing. The L0
measurement settled that: the mmap layer will not carry an expert-sized request on this storage
stack, so an explicit read path is the only route to the numbers in section 3.

`ExpertCache` (base, Java 8) is the interface; `MappedExpertCache` (java21) is the implementation,
loaded reflectively by `ExpertCacheFactory` in the same way `VectorOpsFactory` loads `SimdVectorOps`.

**One slot holds one expert of one layer — all three projections together.** The router never needs
gate without up and down, so that is the natural eviction unit, and it keeps each sub-slice at a
single quantization type. This matters on a Q4_K_M mix, where `ffn_down_exps` is Q6_K on some layers
and Q4_K on others: slots are sized by the largest expert across all layers (2988 KB on
Qwen3-Coder-30B, versus 2.65 MB if every projection were Q4_K).

**Reads go through `FileChannel.read(ByteBuffer, position)` into a `ByteBuffer` view of the slot
segment**, so bytes land straight in the cache with no intermediate copy, and the call lowers to
`pread` on Linux. Panama FFM is not needed for the I/O itself. Positional reads do not touch the
channel position and are safe to issue concurrently, so the misses of one layer are filled in
parallel — which is where the effective read bandwidth in section 10 comes from.

**Retention is least-frequently-used with a least-recently-used tiebreak.** That is the hot set,
without needing a persistence format: counting selections keeps the top experts resident and lets
the cold tail cycle through the remaining slots. A plain LRU — which is what `ExpertGpuCache` does,
admitting every miss unconditionally — would let that tail evict the hot experts on every token.
Slots claimed earlier in the same `prepare()` are excluded from eviction, so one layer can never
evict its own experts.

The API is two calls per MoE layer: `prepare()` right after top-K resolves slots and reads misses,
then `tensorFor()` inside the compute loop is a pure lookup. Redirection happens at the point of
computation — `MoEFFN.expertMatmul` and `Qwen3MoEInferenceEngine.expertMatmul` switch from
`weights3D.dot(expertOffset + rowOff, …)` to `cached.dot(rowOff, …)`, since a cached slice is a
standalone tensor whose rows start at zero. That placement is forced by the constraint in section 8:
`TensorData` is not a viable interposition point.

Every failure path falls back to the mmap: a short read, a wrong slot size, a cache too small to
hold one layer's top-K, or a `ClassCastException` on the Java 8 mapping all leave the original
`weights3D.dot` path in charge, and `prepare()` returning false re-enables the L0 hint for that
layer.

Budget: `--expert-cache-size <MB>`, or `-Dmoe.expert.cache.mb`. The default is a quarter of physical
RAM capped at 4 GB — the cache displaces page cache rather than competing with it, since the pages
it holds are the ones the kernel was thrashing anyway. Enabled by `-Dmoe.expert.cache` on the same
`auto` gate as L0.

### L2 — router-lookahead prefetch (designed, not built)

Prefetch layer L+1's experts during layer L's compute, reusing `LayerPrefetcher`'s single-slot queue
with a discard policy so a lagging disk degrades gracefully. Colibri's 71.6 % one-layer-ahead
predictability is encouraging but architecture-specific; it must be measured on Qwen3-Coder before
this is worth building. The existing routing instrumentation is the natural place to add that
measurement.

## 8. Constraints discovered

**`TensorData` cannot be used as an interposition point.** The obvious design — wrap `TensorData` so
every weight read passes through a cache — does not work. All twelve `Simd*FloatTensor` classes take
a concrete cast and capture the raw segment at construction:

```java
this.segment = ((MemorySegmentTensorData) data).segment();
```

and the three GPU upload paths do the same. A wrapper would be bypassed by the entire hot CPU path
and would break those casts. Any cache must expose real `MemorySegment` slots, and redirection must
happen where the matmul offset is computed.

**`vboxsf` is not bare NVMe.** The measurements here come from a VirtualBox shared folder: no
`O_DIRECT`, and a higher fixed per-request cost than a direct NVMe device. The numbers are therefore
a lower bound. A bare-metal NVMe will do better, but the shape of the curve — bandwidth rising with
request size and saturating in the low megabytes — is a property of the storage stack generally, not
of `vboxsf`.

**The `ExpertGpuCache` K-quant bug is still open** (`placement-autotuning.md`). The Q4_K path
uploads bytes that dequantize to huge or NaN values from a cache slot while the same bytes
dequantize correctly in place, and the root cause resisted diagnosis because `compute-sanitizer` and
`cuda-gdb` cannot instrument a JVM that reaches `libcuda` through Panama FFM. L1 must not depend on
that path: it is a separate CPU-side cache and should stay that way. Note also that the summary
label "byte offset / kernel mismatch" used elsewhere in the docs is contradicted by the
investigation itself, which explicitly ruled both out.

**Measurement noise.** The box is shared with other workloads and thermally constrained, so deltas
below roughly 2× should be treated as noise. The effects targeted here are far above that threshold,
which is what makes them measurable at all on this hardware.

## 9. User interface

- **`--ssd-streaming`** — run a model larger than RAM by streaming weights from the model file. It
  sets `no.preload=true` and `moe.expert.cache=true`, and stands in for `--force` on this one case,
  so the configuration no longer stops at an interactive prompt. Both are already the automatic
  behaviour above 85 % of RAM; the flag makes it explicit and applies it wherever the model falls
  against that line.
- **`--expert-cache-size <MB>`** — the L1 budget, forwarded to `moe.expert.cache.mb` by `CLIRunner`
  the same way `--threads` is forwarded, so the cache needs no extra constructor parameter.
- **Warning text corrected.** Two messages claimed a model above the RAM budget would swap. Since
  v1.16.0 that has been false — the loader skips the preload and reads pages from the model file —
  and with L1 the MoE case is a supported mode rather than a discouraged one. The plan now says so,
  and distinguishes MoE (streams, with a cache) from dense (streams, but nothing can cache it).
- **Counters on `LLMPlayerMXBean` and `/api/metrics`**, under an `expertCache` block: active flag,
  hit rate, hits, misses, MB read, read time, slot count and total size. No reflection is needed —
  unlike the CUDA context, `ExpertCache` is a base-code interface and only its implementation lives
  in java21. The block is always present so clients can render it unconditionally; when the cache is
  inactive every counter is `-1`. A shutdown line reports the same totals.

On metrics, note that `ExpertGpuCache.getHits()` / `getMisses()` and
`Qwen3MoEInferenceEngine.getExpertCacheStats()` already exist but have **no callers anywhere** —
cache behaviour is currently unobservable. Exposing them needs a getter on `LLMPlayerMXBean`, a
field plus a `reset()` entry on `LLMPlayerMetrics`, and a separate entry in
`ApiHandler.handleMetrics`, since the JSON is hand-built rather than reflected from the MXBean.

The stale swap warning in `CLIRunner` should be corrected regardless of the rest.

## 10. Results

Measured with:

```bash
echo y | java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED --enable-preview \
  -Xmx3g -cp target/classes it.denzosoft.llmplayer.LLMPlayer \
  --model gguf/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  --prompt "Hello" --max-tokens 5 --context-length 512 --no-gpu
```

| Run | Wall clock, 5 tokens | Decode tok/s | PPL | Output |
|---|---|---|---|---|
| Baseline, before L0 existed (cold cache) | 152.3 s | 0.1 | 0.99 | `Hello! How can I` |
| **L0 on** (cache warmed by the run above) | **93.9 s** | 0.1 | 0.99 | identical |
| **L0 off**, control (cache warmer still) | **107.9 s** | 0.1 | 0.99 | identical |
| **L1**, 2 GB expert cache | **34.1 s** | **1.0** | 0.99 | identical |

Two numbers, because they measure different things. Wall clock includes prefill, where every expert
is a cold miss, so **L1 is worth 3.2× there** (34.1 s against 107.9 s), and 4.5× against the cold
baseline. The reported tok/s is decode only — `genTokenCount / genTimeNs`, excluding prefill — and
that is where the cache does its real work: **0.1 → 1.0 tok/s, about 10×**, because by then the hot
experts are resident. The decode figure is the one a user generating text experiences; the wall
clock is the conservative one, so the 3.2× is the honest headline.

Output and perplexity are identical in every row, which is the correctness gate for a change that
alters where the weight bytes come from.

A longer run shows the hot set warming as predicted: 20 requested tokens (9 generated before EOS)
reached a **52.2 % hit rate**, up from 48.4 % over 5 tokens, at 3.7 s per token wall clock.

### The other engine, and a false alarm worth recording

GLM-4.7-Flash exercises the second call site, `MoEFFN` on the DeepSeek2 engine. With the cache it
runs 3 tokens in 164.8 s against 247.1 s without — **1.5×**, at a 44.5 % hit rate. The smaller gain
is consistent with its geometry: its experts are twice the size (5976 KB slots against 2988 KB), so
the same 2 GB budget holds 350 experts instead of 701.

Its first measurement looked like a correctness failure — PPL 2.42 with the cache against 1.38
without. It was not. The CLI defaults to `--temperature 0.7`, so both runs were sampling
stochastically with no seed. Re-run at `--temperature 0`, the cache produces `PPL=1.38,
avg_nll=0.3233`, digit for digit the same as the pre-cache run. The lesson for anyone benchmarking
this path: **compare quality at `--temperature 0`**, because MoE output is especially sensitive —
noise in the attention output perturbs the router's matmul and flips the top-K selection, so a
sampled run can diverge for reasons that have nothing to do with the change under test.

Models that fit RAM are untouched: the cache never engages (`mmap.advise` is `none` when preloading)
and Llama-3.2-1B and Qwen3-0.6B return their usual perplexity and throughput.

The honest reading of those three numbers is that **L0 is worth about 1.15×, not the 30× the I/O
curve suggested**. Most of the 152 s → 94 s drop is page-cache warming from the preceding run, which
is why the control matters: comparing L0 on against L0 off, with the control holding the *more*
warmed cache, leaves 93.9 s against 107.9 s. That is below the noise threshold this box warrants.

Output and perplexity are unchanged in every run, as expected for a pure read-ahead hint.

### Why the gain is small — and what it proves

A direct `mincore()` probe explains it. Issuing `MADV_WILLNEED` over a 2.65 MB expert range on this
volume:

```
MADV_WILLNEED -> rc=0 in 2.0 ms
resident pages: before=0/647  after=32/647
explicit read of same range: 21.7 ms (116 MB/s)
```

The call **succeeds** and then does almost nothing: 1.5 seconds later only 32 of 647 pages are
resident, roughly 128 KB of the requested 2.65 MB. `vboxsf` caps read-ahead near 128 KB and does not
honour a large `MADV_WILLNEED`. An explicit read of the identical range completes in 21.7 ms.

This is a more useful result than a speedup would have been, because it settles a design question.
The earlier bandwidth curve was measured with explicit `pread`, so it proves that *the storage* can
serve expert-sized reads quickly — but the mmap path cannot ask it to. **Reaching that bandwidth
requires an explicit read data path; it is not achievable with mmap hints on this stack.**

That is precisely the choice Colibri made, and it is now grounded in a measurement here rather than
inherited from another project's design: explicit `pread` into managed buffers is a *requirement*,
not an optimisation over mmap.

### Where the time goes after L1

The cache reports its own counters at shutdown:

```
Expert RAM cache: 701 experts x 2988 KB = 2045 MB, filled by positional reads (SSD streaming)
Expert RAM cache: 48.4% hit rate (2601 hits, 2775 misses), 7.42 GB read from disk in 4.2 s
```

Two things follow. The 48.4 % hit rate is close to what section 3 predicts for a cache of this size
— 701 slots over 48 layers is about 14 experts per layer, between the measured top-8 (38.5 %) and
top-16 (57.3 %) — and it is depressed by the cold start, since the first token can only miss.

More importantly, **7.42 GB was read in 4.2 seconds of a 34.1 second run**. Two conclusions:
parallel positional reads reach roughly 1.8 GB/s effective, far above the 482 MB/s single-threaded
figure in section 2, because the misses of one layer overlap and some reads hit the OS cache; and
**I/O is now about 12 % of the run, so the bottleneck has moved from disk to CPU**. The remaining
~30 seconds is SIMD dequantization and matmul over roughly 1 GB of expert weights per token on four
cores.

That reframes the remaining headroom. Growing the cache to a 100 % hit rate could save at most the
4.2 seconds still spent reading — around 12 % — so further work on the streaming path has little
left to win. Making this model materially faster is now a compute problem, which is the domain of
MoE-optimized GPU placement and the expert GPU cache, not of streaming.

## 11. Disposition

- **L0 — shipped, enabled, honestly small.** Worth ~1.15× here and demonstrably capped by the
  filesystem, not by the design. Left on (`auto`) because it is never negative, costs about 1150
  cheap syscalls per token in a regime where a token takes ~20 s, and should do considerably better
  on storage whose read-ahead honours the hint. Retest on bare NVMe before drawing conclusions
  about the technique itself.
- **L1 — shipped, and it is the one that matters.** 3.2× on Qwen3-Coder-30B with bit-identical
  output, moving the model from unusable to slow-but-usable. It also moved the bottleneck: I/O is
  now ~12 % of the run, so the streaming problem is substantially solved on this hardware.
- **L2 — designed, deferred.** Router-lookahead prefetch would overlap disk I/O with compute, but
  after L1 there were only ~4 seconds of I/O in a 34 second run to hide, and `prepare()` already
  overlaps the misses of a layer with each other. *Update 2026-09-24:* compute has since become
  several times faster and I/O is again 35–65 % of a run, so the case for overlap is back. Batched
  prefill now overlaps the next expert group's reads with the current group's compute (see "Three
  prefill changes" below); for decode, cross-layer lookahead remains unbuilt and would first need
  its predictability measured on this model.
- **Dense >RAM — closed, won't do.** No cache policy can help; `LayerPrefetcher` already does the
  only useful thing.

### Budget sweep: more cache is not monotonically better

Qwen3-Coder-30B, 5 tokens at `--temperature 0`, 2 GB heap on a 7.8 GB box:

| Budget | Slots | Hit rate | Read from disk | Read time | Wall clock |
|---|---|---|---|---|---|
| 256 MB | 87 | 6.4 % | 13.40 GB | 5.7 s | 15.8 s |
| 512 MB | 175 | 16.8 % | 11.93 GB | 5.2 s | 16.5 s |
| 1024 MB | 350 | 31.0 % | 9.91 GB | 4.2 s | 15.6 s |
| **2048 MB** | 701 | **48.4 %** | 7.42 GB | 4.0 s | **17.5 s** |
| 3072 MB | 1052 | **57.3 %** | 6.14 GB | 2.6 s | **68.4 s** |

Hit rate scales cleanly with budget, and the 57.3 % at 1052 slots — about 22 experts per layer —
lands almost exactly on the top-16 figure from section 3, which is a good sign that the LFU
retention really is holding the hot set rather than churning.

**But the 3 GB row is four times slower while reading the least and hitting the most.** It wins
every cache-internal metric and loses badly overall, because 3 GB of slots plus a 2 GB heap on a
7.8 GB machine leaves the kernel nothing for the page cache. The expert cache stops displacing page
cache and starts competing with it.

That is the practical ceiling on this technique, and it is worth stating plainly: the budget must
leave room for the page cache, because the page cache is what absorbs everything the expert cache
misses. `ExpertCacheFactory` now warns when the requested budget plus the JVM heap exceeds 60 % of
physical RAM. The default — a quarter of physical RAM, capped at 4 GB — lands at 1.95 GB here, right
at the measured sweet spot, so it is validated rather than merely plausible.

Note also how flat wall clock is from 256 MB to 2048 MB. With a warm page cache the two caches are
substitutes, and the expert cache's marginal value is modest; its large win in section 10 was against
a system that was thrashing. Absolute wall-clock numbers therefore drift downward across repeated
runs on the same file as the page cache warms — the deterministic quantities are hit rate and bytes
read, which is why the table reports those alongside.

**Batched MoE prefill changes what the counters mean (2026-09-24).** Since batched prefill runs each
routed expert once per chunk over all its tokens (`cpu-dispatch-and-kernels.md`, section 5.4c), it
calls `prepare` once per (chunk, layer, group of experts) rather than once per (token, layer). Hits
and misses during prefill are therefore counted per (chunk, expert), and a run's hit rate is no
longer comparable with runs from before that change; bytes read still are. On Qwen3-Coder-30B with a
2 GB cache and a 103-token prompt, a run read 19–23 GB instead of 37–41 GB, because an expert
requested by several tokens of a chunk is read at most once for that chunk.

### 2026-09-24: disk is the bottleneck again, and three prefill changes

Section 10 concluded that after L1 the I/O was about 12 % of the run and that streaming had little
left to win. The CPU kernel work since then (`cpu-dispatch-and-kernels.md`) made compute several
times faster, and the balance moved back: on Qwen3-Coder-30B with a 2 GB cache, runs of this session
spent 43 s reading 22.9 GB in a 68 s run, and 15–25 s of 40–60 s in a decode-dominated run. Three
changes to batched MoE prefill were measured, all on Qwen3-Coder-30B Q4_K_M, a 290-token prompt, a
2 GB cache, greedy decoding, separate processes in rotating order (load average 5–14, 5–6 GB of free
RAM). The generated text was identical in every run below.

**Larger chunks when streaming — kept.** Batched prefill reads each expert at most once per chunk and
layer, so the bytes read are proportional to the number of chunks. `-Dprefill.batch` sweep, 4
generated tokens, two runs each (bytes are deterministic, times are not):

| Chunk | Chunks for 290 tokens | Read from disk | Read time | Wall clock |
|---:|---:|---:|---:|---:|
| 64 | 5 | 47.4 GB | 109–125 s | 180–282 s |
| 128 | 3 | 31.8 GB | 72–95 s | 161–197 s |
| 320 | 1 | **16.2 GB** | **48 s** | 128–157 s |

The global default stays 64, which is sized for the CPU caches of the multi-token kernels (every
weight row sweeps all the chunk's inputs), but when the SSD cache is active the MoE engines now use
`-Dmoe.prefill.stream.batch` (256).

**Overlapping reads with compute — kept.** Experts are prepared in groups of 16; previously each
group's misses were read and then computed, so disk and CPU took turns. Now the next group's
`prepare()` runs on a background thread (`moe-expert-io`) while the current group computes. To make
that safe, `findVictim` protects the slots of the previous `prepare()` as well as the current one,
and the engine resolves a group's tensors before starting the next prepare, so the cache is never
read and written concurrently. Only the first group of each layer stays exposed, because a layer's
experts are known only after its router runs. `-Dmoe.prefill.overlap=false` restores the
synchronous order. Chunk 256, 4 generated tokens:

| | Run 1 | Run 2 | Read from disk |
|---|---:|---:|---:|
| synchronous | 117.7 s | 116.8 s | 22.12 GB |
| **overlapped** | **91.6 s** | **94.4 s** | 22.12 GB |

About 1.25× on the whole run with identical reads. Together with the larger chunk, the same prompt
went from 180–282 s and 47.4 GB to about 92 s and 22.1 GB.

The two changes together on the other engines, with the same text as before in every case:
sonar-oss-20b (GPT-OSS, MXFP4, `-Dno.preload=true`, 2 GB cache, 113-token prompt, 8 generated),
previous behaviour (`-Dmoe.prefill.overlap=false -Dmoe.prefill.stream.batch=64`) against the new
default in two rotating pairs: 192 → 123 s and 128 → 108 s, 18.1 → 11.7 GB read. DeepSeek-Coder-V2-Lite
forced to stream (`-Dno.preload=true`, 1.5 GB cache) ran its 102-token prompt plus 16 tokens in 74 s,
with output identical to the resident run. One GPT-OSS run in between took 468 s with the new build —
at a moment when another workload left 5 GB of RAM free, so the lazily mapped non-expert weights were
being re-read page by page; the paired runs above, taken back to back, are the valid comparison.

**Counting every token's selection for the LFU — rejected.** Batched prefill calls `prepare()` once
per chunk, so each expert's selection count grows by one per chunk rather than one per token. Adding
the missing counts (so that decode inherits the "true" hot set) made things worse, deterministically:
with 24 generated tokens, 37.25 GB read and 13,963 misses against 32.45 GB and 12,146 without, and
decode 0.7 against 0.8 tok/s. The most plausible reading is that the full counts pin the experts the
prompt used most, while decode wants others; counting once per chunk damps the prompt's weight in
the retention. The change was removed.

### What is still open

- **Only one hardware profile.** Everything here is one box with a `vboxsf` volume. The shape of the
  budget curve, and especially where the page-cache cliff sits, will differ on bare NVMe and on a
  machine whose RAM is not 2.4× smaller than the model.
- **`ExpertGpuCache` counters remain dead code.** `getHits()` / `getMisses()` and
  `Qwen3MoEInferenceEngine.getExpertCacheStats()` still have no callers; only the RAM cache is
  surfaced through metrics. Wiring the GPU one is a small, separate job.
- **No cross-session persistence.** Colibri persists its routing profile (`.coli_usage`) so a fresh
  process starts warm. Here the hot set is relearned each run, which costs the first token or two.
