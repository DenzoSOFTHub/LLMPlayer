# Per-token latency — a re-analysis of where the time actually goes

A fresh look at the whole inference path, driven by measurement rather than by which subsystem is
most interesting to optimise. It was prompted by the SSD-streaming work
([`ssd-streaming-cache.md`](ssd-streaming-cache.md)), which solved the disk bottleneck and thereby
exposed what was underneath it.

The headline is that the profile does not look the way the codebase's optimisation history assumes.
Nearly all of the tuning effort recorded in `llamacpp-comparison.md`, `jvm-flags.md` and the
`option-*` documents went into CUDA kernels. On the current reference machine that work cannot run at
all, and the two biggest CPU-side costs — one of them a straightforward defect — have never been
addressed.

## 1. The reference machine changed, and that changes the priorities

| | Documented in BENCHMARKS.md (through 2026-06-17) | Now (2026-08-11) |
|---|---|---|
| Cores | 22 | **4** (1 thread/core, 96 MiB L3) |
| RAM | 31 GB | **7.4 GB** |
| GPU | RTX 4050 Laptop, 6 GB VRAM | **none** |
| Model volume | — | `vboxsf` share |

The GPU is not merely idle, it is absent: no `/dev/nvidia*`, no `libcuda.so`, no `nvidia-smi`, and
`--gpu-list` reports "No GPU devices found". Every CUDA path in the project — the forward passes, the
dp4a work, the graph capture, the expert GPU cache — is currently unreachable. The CPU path is the
only thing that executes.

That is worth stating plainly because it inverts the project's usual cost/benefit. A 2 % gain in a
Q4_K CUDA kernel is worth nothing here; a 15 % gain in the CPU forward pass is worth everything.

## 2. Where the time goes, measured

`-Dcpu.profile=true`, Qwen3-Coder-30B-A3B Q4_K_M (18.6 GB, 48 layers, 128 experts, top-8), CPU-only,
2 GB expert cache, 12 requested tokens:

```
[cpu-profile Qwen3MoE] 10 tokens, per-token avg (ms):
  attn_norm=6.0  attn(GQA)=479.6  ffn_norm=0.7  dense_ffn=0.0
  moe_ffn=641.3  residual=1.4  output=185.7  | total=1314.8
```

| Phase | ms/token | Share |
|---|---:|---:|
| `moe_ffn` (routed experts, **including the cache's disk reads**) | 641.3 | 49 % |
| `attn(GQA)` | 479.6 | 36 % |
| `output` (final logit projection) | 185.7 | 14 % |
| norms + residuals | 8.1 | 1 % |

**Read this profile carefully.** It averages the first 10 forward passes, and 9 of those are prefill
passes running against a cold cache. So `moe_ffn` here carries most of the run's disk I/O (3.6 s
total across the run) and is flattered downward in steady-state decode, while `attn` and `output` —
which touch no expert weights and are pure compute — are *understated* as a share of decode. The true
decode split is more attention- and output-heavy than the table suggests.

### The cost model behind it

Parameters touched per token, from the model geometry (dim 2048, 32 heads / 4 KV heads, head 128,
expert FFN 768, vocab 151936):

| Component | Per layer | × 48 layers |
|---|---:|---:|
| Attention (wq + wk + wv + wo) | 18.87 M | **0.91 G** |
| Routed experts (8 × gate/up/down) | 37.75 M | **1.81 G** |
| Router | 0.26 M | 0.01 G |
| Output projection (once) | — | **0.31 G** |
| **Total** | | **3.04 G** |

Dividing measured time by parameters gives the effective throughput of each phase:

| Phase | Params | Time | Throughput |
|---|---:|---:|---:|
| `moe_ffn` | 1.81 G | 641.3 ms | **2.83 G param/s** |
| `attn` | 0.91 G | 479.6 ms | **1.89 G param/s** |
| `output` | 0.31 G | 185.7 ms | **1.68 G param/s** |

This is the surprise. The MoE path — the one everyone treats as the problem — is the **most
efficient** phase, and it is carrying the disk I/O on top. Attention and the output projection are
1.5–1.7× less efficient per parameter despite being plain dense matmuls over resident weights.

At ~1.86 GB of weight bytes per token in 1.315 s, the aggregate is ~1.4 GB/s. That is far below what
DDR5 on this part can sustain, so the forward pass is **compute-bound in the dequantise-and-FMA
loop**, not memory-bound. Layout tricks will not help; reducing dispatches and eliminating redundant
work will.

## 3. Findings, ranked

### F1 — Prefill computed the output projection for every prompt token and threw it away — FIXED

**Verified defect, now fixed and measured.**

Only the final prompt token's logits are used to sample the first generated token. But four of the
eight `generate*` prefill loops in `LLMEngine` were written as:

```java
for (int i = prefillStart; i < promptLen; i++) {
    logits = eng.forward(state, promptTokens[i], i);   // every iteration overwrites the last
}
```

so every earlier prompt token paid the full 151936 × 2048 output projection and the result was
immediately discarded.

The striking part is that **every engine already had `forwardNoOutput`** — the no-logits entry point
that stops after the layer loop — and half the call sites already used it correctly. The defect was
purely in the four loops that did not:

| Engine | Before | |
|---|---|---|
| Falcon-H1, LFM2, Qwen3.5 | `if (i < promptLen-1) eng.forwardNoOutput(...) else eng.forward(...)` | ✅ |
| Gemma 4 | `eng.forward(promptTokens[i], i, i == promptLen - 1)` | ✅ |
| **Standard** (Llama, Qwen2/3, Gemma 2/3, Phi, Mistral, OLMo2, Granite, …) | `logits = eng.forward(...)` | ❌ |
| **Qwen3 MoE** | `logits = eng.forward(...)` | ❌ |
| **DeepSeek2** | `logits = eng.forward(...)` | ❌ |
| **Nemotron-H / Granite Hybrid** | `logits = eng.forward(...)` | ❌ |

The standard engine covers most architectures in the project, so this was not a MoE curiosity. The
fix is the same four-line conditional in each loop, calling the method that already existed.

**Measured, back-to-back A/B against a build of the previous code:**

| Model | Engine | Prompt | Before | After | Δ |
|---|---|---:|---:|---:|---:|
| Llama-3.2-1B Q4_K_M | Standard | 153 tokens | 35.48 s | **28.87 s** | **−18.6 %** |
| Qwen3-Coder-30B Q4_K_M | Qwen3 MoE | 9 tokens | 58.86 s | **37.74 s** | −35.9 % |
| Nemotron-3-Nano-4B Q4_K_M | Nemotron-H | 15 tokens | 215.05 s | **74.82 s** | −65.2 % |
| DeepSeek-Coder-V2-Lite Q4_K_M | DeepSeek2 | 11 tokens | 153.65 s | **98.47 s** | −35.9 % |

All four modified engines are covered: Standard, Qwen3 MoE, Nemotron-H and DeepSeek2.

Perplexity, `avg_nll` and the generated text are identical to the digit in every pair, and on the 30B
the expert-cache counters are byte-for-byte identical (48.4 % hit rate, 2601 hits / 2775 misses,
7.42 GB read) — proving the two runs did exactly the same work apart from the skipped projection.

**Only the Llama row is a clean measurement of the matmul itself**: 6.61 s saved over 152 skipped
tokens is 43.5 ms each, which is the output projection and nothing else. The other three rows are
real but inflated, and none of them should be quoted as a steady-state figure:

- The 30B pair ran while the machine was under heavy variable load — the identical "before"
  configuration measured 10.9 s earlier in the session. The two runs were consecutive, so they are
  comparable to each other, but not to anything else.
- The Nemotron run was invoked with `--ssd-streaming`, which forces `no.preload` even though a 4B
  model fits RAM comfortably. That made the output tensor stream from disk on *every* prompt token,
  so the measurement conflates the matmul with repeated disk reads of the same weights.

Both inflated rows point at the same secondary effect, which is worth stating because it is not
obvious from the code: the output tensor is large (~255 MB in Q6_K on the 30B), so projecting it once
per prompt token pushes gigabytes through a 7.4 GB machine and evicts everything else. On the 30B the
measured disk-read time fell alongside the wall clock (4.1 s → 3.4 s), consistent with reduced
page-cache pressure rather than with the saved matmul alone. **Skipping the projection saves more
than its own compute whenever memory is tight.**

The saving scales with prompt length, which is the direction real usage goes.

### F2 — Prefill was token-by-token, so a >RAM MoE re-read its experts once per prompt token — FIXED

**Shipped for `Qwen3MoEInferenceEngine`. The largest win of the session.**

`forwardPrefill(state, tokens, fromPos, toPos)` drives the layers in the outer loop and the tokens in
the inner loop, in chunks of `-Dprefill.batch` (default 64). It computes exactly the same values in
exactly the same per-(layer, token) order — token *t* at layer *L* still reads its residual stream
from layer *L−1* and attends over KV[L][0..t], written by the tokens processed before it in that same
layer — so **only the loop nesting changes and the output is bit-identical**. Disable with
`-Dprefill.batched=false`.

What changes is the working set. Token-outer order walks all 48 layers for one token before moving
on, so keeping an expert resident across tokens would require the cache to hold `layers × top-K`
slots at once; streamed from SSD, most are evicted before the next token needs them. Layer-outer
collapses the working set to the union of experts selected by one chunk at one layer, which the
existing cache absorbs. This is the "batch-union" effect that makes batched prefill fast — obtained
by reordering alone, with no batched matmul kernel. (Since 2026-09-24 the CPU path also batches the
matmuls — attention projections, shared expert, and each routed expert once over all the chunk's
tokens routed to it; see section 5.4c of [`cpu-dispatch-and-kernels.md`](cpu-dispatch-and-kernels.md).)

Measured on Qwen3-Coder-30B, 109-token prompt, 12 generated, `--temperature 0`, 2 GB cache:

| | token-outer | layer-outer | Δ |
|---|---:|---:|---:|
| Wall clock | 283.2 s | **65.7 s** | **−76.8 % (4.3×)** |
| Decode tok/s | 0.4 | **1.7** | 4.3× |
| Cache misses | 19 989 | **14 774** | −26.1 % |
| Hit rate | 57.0 % | **68.2 %** | +11.2 pt |
| Read from disk | 53.35 GB | **39.52 GB** | −25.9 % |
| Time in disk reads | 52.8 s | **17.9 s** | **−66 %** |
| PPL / `avg_nll` | 0.98 / 0.1043 | 0.98 / 0.1043 | identical |

Total expert selections are identical in both runs (26475+19989 = 31690+14774 = 46 464 = 121 forward
passes × 48 layers × 8 experts), confirming the routing is unchanged and only the *order* of access
differs.

Note that disk time fell by 66 % while the miss count fell by only 26 %. Fewer misses is the smaller
half of the story: the remaining misses are also cheaper, because consecutive tokens at one layer
touch the same region of the file instead of jumping across 48 layers between reads. Miss count alone
understates the benefit, which is why the table reports both.

The chunk size trades two things off: a larger chunk amortises each expert read over more tokens, but
the per-layer union grows and can exceed the cache. At 64 tokens the union stays well inside a 2 GB
cache on this model. It also bounds the residual-stream buffer to `chunk × dim` floats (512 KB here).

**Also applied to DeepSeek2** (`DeepSeek2InferenceEngine`, covering DeepSeek2 and GLM-4.7-Flash),
which turned out to be structurally identical. Measured on DeepSeek-Coder-V2-Lite Q4_K_M (9.7 GB
against 7.4 GB of RAM), 98-token prompt, 10 generated, 1.5 GB cache:

| | token-outer | layer-outer | Δ |
|---|---:|---:|---:|
| Wall clock | 137.6 s | **84.3 s** | −38.8 % |
| Cache misses | 11 112 | **8 726** | −21.5 % |
| Hit rate | 34.0 % | **48.2 %** | +14.2 pt |
| Read from disk | 59.46 GB | **46.70 GB** | −21.5 % |
| Time in disk reads | 66.3 s | **24.7 s** | **−62.7 %** |
| PPL / `avg_nll` | 1.00 / 0.0005 | 1.00 / 0.0005 | identical |

Selections identical in both runs (16 848 = 108 passes × 26 MoE layers × top-6), and the same
signature as Qwen3 MoE: disk time falls roughly three times as much as the miss count, because the
surviving misses are cheaper.

**Nemotron-H was deliberately left alone.** The reordering would be correct there — inside a layer
the tokens still advance in order, which is exactly how an SSM scan is normally batched, so the
Mamba-2 recurrent state and the conv rolling state stay consistent — but it carries per-layer
recurrent state and a separate `forwardGpu` path, and the only variant that would benefit is Granite
MoE, which fits RAM and therefore streams nothing. Real risk, no measurable gain on this hardware.

Dense engines gain nothing from the reordering either: a dense model touches every weight for every
token regardless of order.

#### Original analysis

Prefill runs `forward()` per token, layer-inner. For a model streaming experts from SSD this is the
worst possible order: token *t* walks all 48 layers pulling its top-8 experts, then token *t+1* does
it again. Across a prompt of *N* tokens the same expert can be read up to *N* times.

Inverting the loop — layer-outer, token-inner — lets each expert be loaded once per layer and serve
every prompt token that routes to it. The union of experts selected across *N* tokens in one layer is
far smaller than *N* × 8: with routing 79 % concentrated in the top 32 of 128, a 200-token prompt
would touch perhaps 60–90 distinct experts per layer instead of issuing 1600 selections. This is
exactly Colibri's "batch-union: each unique expert is read once".

It is correct to do: attention for token *i* needs the KV of tokens 0..*i*, all produced within the
same layer, so processing tokens in order inside a layer preserves causality. It is also the
precondition for a batched matmul later — amortising a weight read over *N* input vectors is the
whole reason batched inference is fast.

Note that `cuda.batched` / `BatchedCudaForwardPass` already exists for the GPU and is disabled
because of a context bug. The CPU side has no batched path at all.

### F3 — On a GPU-equipped machine, half of this profile is already solvable

`attn` + `output` = 665 ms/token, **50 % of the measured total**, and both are dense, resident,
modest-sized tensors. MoE-optimized placement already puts exactly these on GPU using ~540 MB of
VRAM. On the previously-documented RTX 4050 this configuration should roughly halve per-token time
for this model with no new code.

This cannot be verified here (no GPU), and it is recorded as an expectation, not a measurement. But
it does mean the highest-value hardware change for this workload is restoring GPU access to the VM,
not writing new CPU code.

### F4 — The output projection reads 255 MB per token because it is Q6_K

`output.weight` is 311 M parameters. At Q6_K (210 bytes per 256 elements, 0.82 B/param) that is
**255 MB read per token**; at Q4_K it would be 175 MB. Q4_K_M ships Q6_K for `output.weight` by
convention, and its SIMD dot is also the slower of the two per byte.

There is no safe algorithmic shortcut — sampling needs the full logit vector — so the levers are: fix
F1 so it runs once per generation instead of once per prompt token; move it to GPU (F3, it is a
single tensor); or pick a model file whose output tensor is Q4_K. Worth knowing when choosing a
quantisation for CPU-bound use.

### F5 — Fused gate+up is now applicable to MoE experts, and is not used

`FloatTensor.fusedGateUpMatmulParallel` exists and is used by `SwiGLUFFN` for dense FFN, but not by
the MoE expert path, because it assumes row offsets start at zero and the 3D expert tensors need a
per-expert base offset.

**The L1 cache removes that obstacle**: a cached slice is a standalone tensor whose rows start at
zero, which is why `expertMatmul` can call `cached.dot(row * inDim, …)`. So the cached path could use
the fused variant directly — one parallel dispatch instead of two, and the shared input vector stays
in L1 across both projections.

Expected value is modest (the input is 8 KB and already cache-resident; the saving is one dispatch
and one barrier per expert per layer, so ~384 barriers per token) and it only applies to cache hits.
Worth trying because it is cheap, but it should be measured, not assumed — the codebase's own history
has several fused variants that measured neutral or negative.

Care is needed with nesting: the expert loop is already an `IntStream.parallel()`, so calling a
second parallel dispatch inside it puts nested tasks on the common ForkJoinPool. With 4 cores and 8
experts the current coarse split is already reasonable; a flattened parallel-over-(expert, row) may
be the better structure.

### F6 — Top-K pruning as an explicit quality/speed knob — SHIPPED

`--expert-top-k N` / `-Dmoe.top.k=N` routes to fewer experts than the model specifies. Expert work
scales linearly with the count, and the routed experts are 1.81 G of the 3.04 G parameters touched
per token, so halving the count removes roughly a quarter of the total. The reduction is applied at
selection time, so the paths that renormalise the routing weights do so over the reduced set and the
gate mass still sums to one; it also shrinks upstream work, since fewer experts are read from disk.
Wired into `Qwen3MoEInferenceEngine`, `MoEFFN` (DeepSeek2 / GLM-4.7-Flash) and
`NemotronHInferenceEngine` (Granite MoE). Values at or above the model's own top-K are ignored, so it
can only reduce.

Measured on Qwen3-Coder-30B (top-8 native), `--temperature 0`. Wall clock was unusable — the machine
was under fluctuating external load — so the table reports the **load-independent** quantities, plus
the generated text, which is what actually answers the question:

| top-K | Expert work | Disk read | Cache hit rate | Output on *"capital of France"* |
|---:|---:|---:|---:|---|
| 8 (native) | 100 % | 13.6 s | 47.8 % | `The capital of France is Paris.` |
| 6 | 75 % | 7.2 s | 51.2 % | identical |
| **4** | **50 %** | 4.7 s | 54.2 % | identical |
| 2 | 25 % | 2.7 s | 55.3 % | `The capital of an Indianapolis-based answer in the Uundernessanswers odcortexe…` |

Disk-read time and hit rate move monotonically and are count-driven rather than timing artefacts:
requesting fewer experts means fewer misses, and the same cache covers a larger fraction of what is
requested.

A harder prompt (*"reverse a singly linked list, only the code"*) sharpens the picture. top-8 and
top-6 produced **byte-identical, correct Java**. top-4 produced **correct Java that is not
byte-identical** — the same algorithm, differing only by a `static` modifier on the signature. So the
degradation is graded rather than binary: top-6 is transparent, top-4 diverges without becoming
wrong, top-2 collapses.

**Usable down to top-4 on this model; catastrophic at top-2.** Half the expert work for output that
is still correct is a good trade on a machine like this one; a quarter is not. If bit-identical
output matters, top-6 is the limit.

#### The perplexity gate does not catch this, and that matters beyond this feature

At top-2, with the output above being obvious gibberish, the evaluator reported
**`Perplexity: 1.00 (EXCELLENT)`**. It is computed over 3 tokens and measures how confident the model
is in its own continuation, not whether that continuation is right — a model can be serenely
confident while producing nonsense.

Every other optimisation document in this project uses PPL as the quality gate. That is sound for
changes that are meant to be numerically transparent, where PPL functions as a checksum and any
divergence is a red flag. It is **not** sound for a deliberately lossy knob, where the whole question
is how much quality was traded away. For those, compare generated text on several prompts, at
`--temperature 0`.

### F7 — The attention efficiency gap: cause identified, two fixes measured, both reverted

Attention runs at 1.89 G param/s against the MoE path's 2.83, despite being dense matmuls over
resident weights while the MoE path is simultaneously doing disk I/O.

**A JFR profile confirms where the time is.** `VirtualThreadMatmul.lambda` is the dominant frame by a
wide margin (19591 execution samples), ahead of `SimdQ4_KFloatTensor.dot` (14786) and
`SimdQ6_KFloatTensor.dot` (5683). That lambda is the fused-matmul task body, which serves attention
and the output projection; the MoE expert path does not go through it, because `expertMatmul` runs a
plain row loop over `weights3D.dot`.

**The cause is a load imbalance, and it is arithmetic rather than conjecture.**
`VirtualThreadMatmul.fusedMatmulQKV` chunks a single `0..max(qRows, kvRows)` range and tests both
bounds inside the loop body:

```java
int maxRows = Math.max(qRows, kvRows);
int chunkSize = Math.max(1, maxRows / procs);
// ... if (row < qRows) q...;  if (row < kvRows) { k...; v...; }
```

A row below `kvRows` costs three dots; above it, one. Under GQA `kvRows` is a small prefix —
Qwen3-Coder-30B has qRows=4096, kvRows=512 — so on 4 cores the first chunk carries
512×3 + 512×1 = 2048 dots while the other three carry 1024 each. The join barrier waits on a task
doing twice the work of its peers, so three carriers idle for half the phase: **62.5 % utilisation**,
which matches the observed 1.89 / 2.83 ratio almost exactly. These tasks run on virtual threads that
never block, so a mounted task holds its carrier to completion and the imbalance cannot be stolen
away.

**Two rebalancings were implemented and measured. Neither won, so both were reverted.**

| Variant | Paired wins | Median Δ |
|---|---|---|
| Separate Q and K/V chunk sets (8 tasks instead of 4) | 1 / 6 | **+6.0 % slower** |
| Equal-*work* chunk boundaries, O(1), same 4 tasks | 3 / 6 | −4.2 %, i.e. noise |

Both were verified correct (perplexity and `avg_nll` identical). The first is a clear regression —
doubling the task count costs more in virtual-thread creation and `Future` bookkeeping than the
imbalance it removes, at these matrix sizes. The second is neutral.

**The most likely reason the fix does not pay is the machine, not the code.** These measurements ran
with a load average of 6.5–9.2 on 4 cores, with two foreign JVMs at ~108 % CPU. Load imbalance only
costs you when you own the cores: here the idle carriers created by the imbalance are immediately
filled by other tenants' work, so removing the imbalance frees nothing the JVM can use. The first
unpaired attempt appeared to show −18 % purely because external load happened to fall during the run
— an interleaved paired design reversed the sign.

**Disposition: reverted, worth retrying on a quiet machine.** The imbalance is real and the equal-work
boundary computation is the right shape for a fix; it simply cannot be validated under contention.
Anyone retrying this should use a paired interleaved design and check `uptime` first — an unpaired
A/B on this box produces whatever answer the background load dictates.

## 3b. Hypotheses tested and ruled out

A second pass looked for overhead outside the arithmetic. Three plausible candidates were checked
directly and none of them is significant. Recording the negatives matters as much as the positives —
they are what narrows the remaining search.

**Attention scores are not iterating over the context.** The obvious suspicion for the `attn`
efficiency gap was a softmax or score loop bounded by `maxSeqLen` instead of the current position,
which at ctx 512 with a ~100-token prompt would be a 5× waste. It is not: `gqaAttention` computes
`attLen = position - startPos + 1` and loops `for (t = startPos; t <= position; t++)`. Correctly
O(position). Ruled out.

**GC is not a factor.** A JFR profile over an 82-second run shows 19 collections, 16 of them G1 young,
with individual pauses of 5.8–19.3 ms — well under 0.5 % of wall clock. Total allocation is
~457–493 MB per worker thread, an allocation rate near 28 MB/s. Ruled out.

**The Vector API is not boxing.** The classic failure mode — the JIT failing to intrinsify, so
`FloatVector` operations allocate real objects instead of using registers — would show as heavy
`Float256Vector` allocation. It accounts for 53 of 5314 allocation samples, about 1 %. The SIMD path
is being intrinsified. Ruled out.

Two real but small findings did come out of the allocation profile, joined by allocation site:

| Site | Type | Volume over the run |
|---|---|---:|
| `SimdQ4_KFloatTensor.dot` | `int[]` | 2.3 GB |
| `SimdQ6_KFloatTensor.dot` | `byte[]` | 236 MB |
| `MappedExpertCache.findVictim` | `java.lang.Long` | 37 MB |

The first two are the small fixed-size scratch arrays each kernel allocates per call
(`new int[8]` twice in Q4_K, `new byte[16]` in Q6_K). Escape analysis eliminates most of them —
otherwise the volume would be far higher — and at a 28 MB/s allocation rate with GC under 0.5 %, the
residue is not worth restructuring. Making them fields would break thread safety; `ThreadLocal` is
actively wrong here because `VirtualThreadMatmul` creates a fresh virtual thread per task, so every
task would allocate its own.

The third was **this session's own regression** and is fixed: `findVictim` looked each slot's key up
in a `HashMap<Long,Integer>`, boxing a `Long` per slot per miss across an O(maxSlots) scan — 701 slots
at a 2 GB budget. It now reads a plain `int[] slotFreq` kept alongside the slots. Same LFU semantics,
since frequency only changes when a key is selected, which also touches its slot. Worth noting the
irony: this is exactly the pattern criticised in `ExpertGpuCache` earlier in the same document.

### Where the remaining gap actually is

With overhead ruled out, the time really is in the dequantise-and-FMA arithmetic. Sizing it: 3.04 G
parameters per token at roughly 5 scalar-equivalent operations each is ~15 G ops, against a 4-core
AVX2 ceiling of ~96 G ops/s — about **158 ms/token theoretical against 1315 ms measured, so ~12 % of
peak**. Even if the per-element op count is double that estimate, the gap is 4×.

It is not memory-bandwidth-bound either: 1.86 GB of weight bytes per token at DDR5 speeds would be
~37 ms. So the pass is neither at the compute ceiling nor at the bandwidth ceiling, which points at
latency and dependency stalls inside the kernels.

Closing that needs instruction-level evidence — hardware performance counters, or a reference
implementation on the same box to size the achievable target. `perf` is not available here and
llama.cpp is not built, and neither would be trustworthy at a load average of 6–13 on 4 cores.
**This is the largest remaining unknown, and it is blocked on tooling and a quiet machine rather
than on ideas.**

## 4. What not to try

Recorded so the measurements are not repeated:

- **Storing the expert cache dequantised (F16/F32).** Makes slices 3.6–7.1× larger, collapsing hit
  rate from 48.4 % to ~19 %/~6.5 % on a 2 GB budget. There is also no `Simd*FloatTensor` for F16/F32
  — `F32FloatTensor.dot` copies each row into a ThreadLocal buffer before the SIMD FMA, whereas
  `SimdQ4_KFloatTensor.dot` reads quantised bytes straight from the segment. See
  [`ssd-streaming-cache.md`](ssd-streaming-cache.md).
- **Re-quantising into the cache to shrink slices.** No K-quant encoder exists (`GGUFWriter` has
  F32/F16/BF16/Q8_0/Q4_0 only), the gain is ~15 % capacity, and it would destroy the bit-identical
  property that makes the cache verifiable. Quantise the model file instead.
- **A larger expert cache.** Measured cliff: 3 GB reached the best hit rate of the sweep and ran 4×
  slower, because cache plus heap starved the page cache.
- **`MADV_WILLNEED` as the streaming mechanism** (~1.15×, capped by the filesystem), **`matmul.tiled`**
  (−50 %), **`cuda.q4k.cpasync`** (−2.8 %).

## 5. Suggested order

| # | Change | Expected value | Confidence | Effort |
|---|---|---|---|---|
| ~~1~~ | ~~**F1** — skip the output projection on non-final prefill tokens (4 engines)~~ | **Done: −18.6 % measured on a 153-token prompt** | — | — |
| ~~2~~ | ~~**F7** — attention phase diagnosis~~ | **Done: cause found (GQA chunk imbalance, 62.5 % utilisation). Two fixes measured, neither won under contention, both reverted.** | — | — |
| ~~3~~ | ~~**F6** — `--expert-top-k` prune knob~~ | **Done: usable to top-4 (half the expert work, output unchanged); top-2 collapses** | — | — |
| 4 | **F5** — fused gate+up on cached slices | Small, hit-rate-limited | Medium | Low |
| 5 | **F2** — layer-outer batched prefill | Large for long prompts | High on principle, real engine change | High |
| 6 | **F3** — restore GPU access to the VM | ~50 % of per-token time | Expectation, unverifiable here | Not a code change |

Items 2–4 are independent and individually verifiable. Item 5 is the one that changes the shape of
the engine, and it was correctly sequenced after F1: F1 removed a cost that would otherwise have been
misattributed to prefill structure and inflated the apparent payoff of batching.

A note on method, since it cost a wrong claim in the first draft of this document. The initial pass
concluded that `Qwen3MoEInferenceEngine` had no no-logits variant and that one would have to be
written. It already had one — the grep that established otherwise had been truncated by `head`. The
defect was real and the fix was right, but it was four call sites rather than four call sites plus two
new methods. When a finding rests on the *absence* of something, verify the absence directly rather
than inferring it from a filtered search.

## 6. How to reproduce

```bash
# Phase profile (the table in section 2)
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED --enable-preview \
  -Xmx2g -Dcpu.profile=true -cp target/classes it.denzosoft.llmplayer.LLMPlayer \
  --model gguf/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  --prompt "Hello" --max-tokens 12 --context-length 512 --no-gpu \
  --temperature 0 --ssd-streaming --expert-cache-size 2048
```

Always compare quality at `--temperature 0`. The CLI default is 0.7 with no seed, and MoE output is
especially sensitive: noise perturbs the router's matmul and flips the top-K selection, so a sampled
run diverges for reasons unrelated to the change under test.
