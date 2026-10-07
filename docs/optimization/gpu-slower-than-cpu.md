# Why GPU mode can decode slower than CPU-only — MoE placements on a laptop GPU

This document records the 2026-09-28 investigation into a specific complaint: on the reference box,
MoE models that take the MoE-optimized placement (every attention layer on the GPU, routed experts on
the CPU, a hybrid expert cache in the remaining VRAM) appeared to decode slower than the same models
with `--no-gpu`. It gives the measured picture, the root causes that survived a three-lens
adversarial review, the causes that did not, and a fix plan written for an engineer who has no
access to the session that produced it.

Hardware and software of every measurement: NVIDIA GeForce RTX 4050 Laptop GPU (6140 MiB, sm_89)
under WSL2 on a Windows host (WDDM driver model), Intel Core Ultra 7 155H (6P + 8E + 2LP-E, 22
logical CPUs, exposed to the VM as 11 pairs), 31 GB RAM, `--threads` defaulting to 11, JDK 25,
the working tree described in section 7.3. All timings are milliseconds per decoded token unless
stated otherwise.

Related documents: [`cuda-forward-pass.md`](cuda-forward-pass.md) for the GPU-resident passes and
the kernel conventions the fixes build on, [`autotuning-heuristics.md`](autotuning-heuristics.md)
for the placement rules as they stand today, [`placement-autotuning.md`](placement-autotuning.md)
for the bandwidth cost model, [`ssd-streaming-cache.md`](ssd-streaming-cache.md) for the other
expert cache (RAM, filled from SSD) and the routing-concentration table,
[`jvm-flags.md`](jvm-flags.md) for the `-D` properties that already have a row there (two used
below do not; they are defined at the top of section 7), and
[`llamacpp-comparison.md`](llamacpp-comparison.md) for the thermal-noise caveat this box carries.

---

## 1. Summary

### The question

With `-Dcpu.profile=true`, MiniMax-M2 (IQ1_M, 62 layers, 116 experts, top-8) reported 2427 ms per
token with the default GPU placement and 3659 with the expert cache off, against 1802 with
`--no-gpu`. Qwen3-Coder-30B-A3B (Q4_K_M, 48 layers, 128 experts, top-8) reported roughly 210 to
310 ms per token on the GPU against 190 to 210 on the CPU. Why is the GPU losing, and what should
change so that GPU mode is never slower than CPU-only and as fast as the hardware allows?

### The answer

The MiniMax headline was a measurement artifact: the Qwen3MoE profiler folded the per-token
GPU-mode prefill of the 29-token prompt into its per-token averages while the CPU-only run prefilled
in a batch that never touched the counters, and once that is removed (already fixed in the working
tree) the default GPU placement decodes MiniMax at 1218–1313 ms per token against 1620–2164 for
CPU-only in the same token windows, which is 18–31 % faster (24–34 % in the clean m3b run), not
35 % slower. On Qwen3-Coder the default placement ties CPU-only within the noise (210 vs 194 ms in
the cleanest batch) and the cache-off configuration is genuinely slower (260 ms, 228–300 per window,
m4d), because the GPU spends the decode in its idle or near-idle states (P8/P5: 210–510 MHz SM,
405–810 MHz memory; deep P8 in every sample only on MiniMax and in the m3i run) and every GPU phase
runs several times below spec; a light clock keeper on a side stream that merely stops the governor
from idling took the same model from 291 to 191 ms per token in an interleaved pair, the single
largest effect measured. The structural costs that remain are second order but real: the expert
cache allocates one 2 MiB-rounded buffer per slot and wastes 26–39 % of its physical VRAM (and
oversubscribes the card on Qwen3-Coder), each attention layer is 14 individual kernel launches
between a pageable upload and a blocking download with no CUDA graph, prefill runs token by token
whenever the GPU is involved, and `MatmulPool` is switched off for every GPU placement that is not
MoE-optimized. Nine other explanations were tested by the data and are recorded in section 6 so
they are not proposed again: six rejected outright (more pool threads, C1-compiled kernels, RAM
pressure, CPU/GPU power coupling as it was measured, VRAM oversubscription on MiniMax, heavy or
memory-bound keepers) and three whose mechanism was kept as engineering under F5, F6 and F8. The
plan is therefore: fix the instruments first, productize the light keeper behind a flag, remove the
cache's allocation waste, capture per-layer graphs, and only then touch placement policy, validating
every step with the interleaved protocol of section 2.6, because identical configurations differ by
1.8–7× from run to run on this machine.

### Priority table

| Id | Fix | Cause status | Effort | Expected gain (with the arithmetic in section 5) | Depends on |
|---|---|---|---|---|---|
| F0 | Instrumentation and benchmark protocol | confirmed | small | none directly; precondition for every other row | — |
| F1 | Clock keeper productization (adaptive, lowest priority, generation-scoped, opt-in) | disputed, corrected | medium | Qwen3-Coder cache-on 291 → 191 and 524 → 263 ms measured (−35 to −50 %); MiniMax unmeasured with the light variant, ceiling ~120–220 ms of 1218–1313 (9–18 %) | F0 |
| F2 | Expert cache arena with per-projection unit sizes | confirmed | small | 1.7–2.1× more resident experts in the same physical VRAM (1.25–1.3× at the same printed budget); Qwen3-Coder stops oversubscribing; tok/s effect via hit rate unmeasured | F0 |
| F3 | Per-layer CUDA graphs, pinned host blocks, event waits in `MoeAttentionCudaPass` / `MlaAttentionCudaPass` | confirmed | medium | 5–6 ms (Qwen3-Coder), 6–8 (MiniMax), 7–9 (GLM-4.7-Flash) per token at today's clocks; a larger share once F1 holds the clock | F0; F1 for measurability |
| F4 | `MatmulPool` stays on for every GPU placement | confirmed | small | −15 to −30 % of the CPU share on dense partial offloads and MoE first-N placements; measured floor 1.26× on GLM-4.7-Flash; also unbiases `--auto-tune` | — |
| F9 | Small-matmul byte threshold and output-projection routing to the CPU twin | disputed, corrected | small | MiniMax output 60–75 → 30–34 ms (−26 to −45 ms per token, 2–3.5 %); zero on Qwen3-Coder; shared-expert models unmeasured | F1 (the decision changes with the clock) |
| F5 | Expert cache redesign: async pinned promotion, calibrate-then-pin policy, speed-aware split, stats | cause claims refuted, fix retained with corrected gains | medium | ≤ 6–20 ms per token from async promotion in steady state; 15–22 ms per token (Qwen3-Coder) from convergence; the split guard prevents regressions rather than adding speed | F0, F1, F2 |
| F6 | Batched prefill in GPU mode (`attentionLayerBatch` + `ExpertViews`) | "2–3× slower" refuted; direction holds for Q4_K models | large | Qwen3-Coder time-to-first-token roughly halved (432–574 → ~216–396 ms per prompt token); MiniMax at most −12 % | F3, F5 |
| F7 | VRAM planner, allocation verification, canary | disputed, corrected | medium | no direct tok/s; makes the 55 ms WDDM-spill trap impossible by construction | F2 |
| F8 | In-process placement calibration replacing the reload-based `--auto-tune` | cause claim refuted; design retained with corrections | medium | no direct tok/s; stops shipping a losing configuration per model; needs F4 and F9 to produce a true CPU baseline | F0, F4, F9 |
| F10 | Qwen3.5-MoE graph mode (per-layer graphs around the host expert callback) | confirmed (part of F3) | medium | estimated 10–13 ms per token of ~178 (launch count read from the code, not measured) | F3 |
| F11 | CPU expert chunk compaction over the non-resident slots | disputed, corrected | small | 0–8 % of the moe phase at measured residency (20–80 ms MiniMax, 5–15 ms Qwen3-Coder); zero on GPT-OSS | — (test with `-Dmatmul.chunks.per.thread=16` first) |
| F12 | `MatmulPool` worker hold across the GPU phase | disputed, corrected | small | ceiling 7–13 ms per token on Qwen3-Coder; A/B not yet run | — (test with `-Dmatmul.spin` first) |

The order is expected gain × confidence ÷ effort, with dependencies pulled forward; the ids are the
historical numbers of the findings list, hence out of sequence. F0 first because nothing is
measurable without it; F1 because it is the only intervention that moved a whole token in an
interleaved pair; F2 because it is small, confirmed and a precondition for F5 and F7. F3 sits above
F4 and F9 despite a 2–4 % gain at today's clocks because F6 and F10 depend on it and its share rises
to 25–30 % of the attention phase once F1 holds the clock; F4 affects no configuration measured this
session; F9 precedes F5 on effort at a similar gain; F5–F8 are larger engineering with bounded or
indirect gains; F10–F12 are cheap experiments whose ceiling is a few percent.

---

## 2. Measurement methodology and its pitfalls

Every number in this document was produced by the scripts in section 7.1. Before reading section 3,
read the five traps below; each of them produced a wrong conclusion at some point during the day.

### 2.1 The profiler artifact (fixed in the working tree)

`Qwen3MoEInferenceEngine.forwardPrefill` takes the batched CPU prefill (`prefillBatched`) only when
`ExpertViews.active() && expertGpuCache == null && gpuAttention == null`. In any GPU mode the prompt
goes through the layer-outer per-token loop, which calls `forwardLayer` per (layer, token) and
therefore accumulates every prompt token's attention and expert time into `profAttnNs` and
`profMoeFfnNs`, while `profTokenCount` is incremented once per `outputProjection`, i.e. once for the
whole prompt. The printed `[cpu-profile Qwen3MoE] N tokens, per-token avg` therefore contained
29 prompt tokens' worth of layer time divided by N. The batched CPU-only path does not touch the
layer counters, so its average was clean (its first line divides 9 decoded tokens by 10, a 10 % low
bias on that one window).

The cost of the bias is `promptTokens × prefillMsPerToken / N`: on MiniMax at N = 60 it is
29 × 2.1 s / 60 ≈ 1.0 s per token on a ~1.45 s steady state, which is the entire "2427 vs 1802"
gap. On Qwen3-Coder with a 19-token prompt over 100 tokens the formula gives 19 × ~200 / 100 ≈ 38 ms
(about 18 % of 210) if a prefill token costs about a decode token, and 19 × 432–574 / 100 ≈
80–110 ms with the measured GPU-mode prefill cost including one-time warm-up (section 3.2). No
pre-fix Qwen3-Coder profile line exists in `an/`, so this bias was never measured directly; the
review's "on the order of 10 ms" compared an unsourced pre-fix 218 with the post-fix 210.5.

The working tree fixes it: `outputProjection` zeroes the phase sums at the first projection
(`profPrefillDropped`, mirrored from `DeepSeek2InferenceEngine`), so every m3, m4 and m5 file is per
decoded token. The m1 and m2 files predate the fix; their cumulative lines can still be used by
re-deriving 10-token windows: `window_k = (avg_N × N − avg_{N−10} × (N−10)) / 10`, which cancels the
prefill constant from the second window on. Section 3 quotes windows for those runs.

Two residual gaps: `profPrefillDropped` is never reset, so the second generation of a process
(auto-tune, multi-turn) folds its prefill in again; and `Qwen35InferenceEngine` has the same
per-token prefill in GPU mode with no drop.

### 2.2 Thermal and run-to-run drift

Identical configurations, minutes apart, on the same box: Qwen3-Coder CPU-only 194 ms (m4a) then
384 ms (m4e, seven minutes later) then 1423 ms (m3f, earlier in the day, after a 22.5 GB MiniMax
preload); GPU default 210 (m4c), 291 (m5a), 524 (m5c). MiniMax CPU-only drifts +34 % within one
run (m1c windows 1620 → 2164) and +93 % in another (m3e 1592 → 3067). The 1-minute load average in
the m4–m5 traces rises from 11.9 (the pool alone) to 17–30 (per-file maxima: m4a 13.7, m4b 17.8,
m4c 21.2, m4d 22.4, m4e 22.9, m5a 18.3, m5b 22.9, m5c 26.1, m5d 21.6, m5e 26.9, m5f 30.3) on a
22-vCPU VM, i.e. other host work was present in the later runs. Sequential single-sample comparisons
cannot resolve differences under about 30 %; the earlier `llamacpp-comparison.md` caveat of
±15–30 % is optimistic for this day. Every fix below is validated only by interleaved pairs.

### 2.3 The P-state ramp and 2-second sampling

nvidia-smi was sampled every 2 s. Every P-state count in this document is taken over the
*generation window*: the samples at the loaded VRAM plateau (the trace maximum of `memory.used`)
with `utilization.gpu > 0`, which spans the prompt prefill and the decode. Whole-trace histograms
are dominated by the 60–120 s model load, during which the idle GPU sits at P8 and ~4 W with
164 MiB used, and must not be read as decode states (an earlier draft of this document made that
mistake). The counts are in sections 3.1 and 3.2; in short, MiniMax decodes at P8 in every cache-off
sample and in 85 % of the cache-on samples (the rest P5, 810 MHz memory); Qwen3-Coder decodes mostly
at P5 with the cache off (m4d; m3i, earlier in the day, all deep P8) and at P5/P4/P3 with no P8
sample at all with the cache on (m4c), where the governor hunts rather than sits — m4c's attention
windows swing 27–79 ms inside one run. The 2 s samples do register the light keeper (m5b P4 10 /
P5 3 against m5a P5 10 / P4 5 / P8 2), but a 2 s sample still cannot resolve a 2 ms burst, so the
P-state column is a coarse indicator of the governor's state, not a measurement of the clock during
the model's own kernels. `clk.txt` (200 ms sampling, an earlier Qwen3-Coder run) shows the governor
needs a sustained load to reach P0 and drops back on single low samples; that ramp does not occur at
all on MiniMax, whose duty cycle it treats as idle.

### 2.4 The CPU clock is unobservable under WSL2

`/proc/cpuinfo` reports a constant 2995.2 MHz on every vCPU in every sample (`m5.txt`, the batch 1
`*_smi.txt`); `/sys/devices/system/cpu/cpu0/cpufreq` does not exist and `cpuidle/current_driver` is
`none`. CPU throttling, C-state exits and the effect of GPU power on CPU boost cannot be seen from
inside the VM. This WSL2 instance has no interop binfmt, so `powershell.exe Get-Counter` cannot be
run from a script; it has to be run manually in a Windows terminal alongside a bench, or replaced by
the pure-Java probe of F0.

### 2.5 WDDM allocation semantics

Under WSL2 the CUDA driver forwards allocations to the Windows WDDM driver, which over-commits into
shared system memory rather than failing: `cuMemAlloc` succeeds past physical VRAM, and a buffer that
lands in shared memory is read over PCIe at a few GB/s with no error and no counter. The one visible
symptom is time: the Qwen3-Coder output projection (Q6_K, 255 MB) cost 54–67 ms per token when it
was uploaded lazily after the cache had taken the free VRAM (`vr.txt`, `[dbg.out] Q6_KCudaTensor
... 54.82 ms`) and 9–14 ms once `tryInitExpertGpuCache` touched it before sizing the cache (m3h,
m4c). Two further facts matter for every budget: each `cuMemAlloc` is backed by 2 MiB pages (a
1.48 MiB slot occupies 2 MiB, measured exactly in section 3.5), and `cuMemGetInfo` reports about
984 MiB less free memory than `6140 − nvidia-smi used` at both fill levels examined (a fixed
under-report, not a proportional one). Also note that nvidia-smi inside WSL2 reports 0 MiB used
before the process starts, i.e. it does not show the Windows host's own VRAM use.

### 2.6 Mandatory benchmark protocol for validating a fix

Use this for every fix in section 5; the pass criterion per fix refers to it as "protocol P".

1. Build: `export JAVA_HOME=/usr/lib/jvm/jdk-25.0.2+10 && export PATH=$JAVA_HOME/bin:$PATH`, then
   `mvn -q clean compile`. Record `git diff --stat` and the class timestamps in the result file, so
   that a run cannot be attributed to code it did not contain (the m1/m2 vs m3 confusion of this
   day). Commit or stash the unrelated working-tree changes first (section 7.3), or the diff will
   not isolate the fix.
2. Configurations: A = baseline, B = fix. Same JVM flags (`--add-modules jdk.incubator.vector
   --enable-native-access=ALL-UNNAMED --enable-preview -Xmx8g -Dcpu.profile=true`), same prompt,
   `--temperature 0 --force`, fixed `--max-tokens`. Reference workloads: MiniMax-M2 with
   "What is the capital of France?" (29 tokens after the chat template) and `--max-tokens 60`;
   Qwen3-Coder-30B with "Write a Java method that reverses a linked list." (19 tokens) and
   `--max-tokens 100`. Add one 512-token prompt when the fix touches prefill or attention scaling.
3. Interleave A-B-A-B; at least two pairs, three when the expected effect is under 20 %. Sleep 30 s
   between runs. Before each run check `uptime`: a 1-minute load average above `threads + 2` means
   the host is busy; wait.
4. Sampling: two background loops per run, started before the JVM and killed after it —
   `while true; do echo "$(date +%T.%N | cut -c1-12) $(nvidia-smi --query-gpu=memory.used,clocks.sm,clocks.mem,pstate,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader)"; sleep 0.2; done > <run>_smi.txt`
   and `while true; do echo "$(date +%T) $(uptime | sed 's/.*load average/load/') $(vmstat 1 2 | tail -1 | awk '{print "si="$7" so="$8" bi="$9" us="$13" id="$15}')"; sleep 1; done > <run>_sys.txt`.
   The `run` function of section 7.1 is the historical 2 s variant (its `vmstat 1 2` inside the
   same loop makes the real period about 3 s), kept only to reproduce the `an/` files. Report the
   P-state histogram of the generation window (section 2.3) next to the numbers.
5. Compare phases, not totals: attention, moe_ffn, output, and the cache hit rate, from the
   `[cpu-profile ...]` lines re-derived into 10-token windows (section 2.1). Discard the first window
   (cache fill, kernel compile, first-use uploads). Report the median window and the best window of
   each run, and the decode-only tok/s from the `--- Stats ---` line; `timeMs` is wall time including
   prefill, so `timeMs − N × total_N` (the profile's cumulative per-token total times N; the
   printed tok/s has one decimal, ±3 %) is the prefill plus warm-up cost.
6. Pass: the targeted phase is compared on the best window of each run (best-of-N) and the total
   on the median window. B beats A on the targeted phase in every pair, and B's median total is not
   worse than A's in any pair by more than the pair-to-pair spread of A itself (the difference
   between A's median windows across the pairs). For capacity fixes (F2, F7) the criterion is the
   nvidia-smi `memory.used` delta, not time.
7. CPU frequency: when a fix changes CPU-side load (F1, F12), read
   `Get-Counter '\Processor Information(_Total)\% Processor Performance'` on the Windows side during
   the run, or print the F0 probe.
8. Results go into a dated "Validation" subsection appended to this document, into the `-D` flag's
   row in `jvm-flags.md`, and, when a default changes, into the MoE rows of `BENCHMARKS.md`.

---

## 3. The measured picture per model

Run ids are the file stems under the investigation's `an/` directory (section 7.2). "Profiler
fixed" means `profPrefillDropped` was compiled in (runs m3a and later). "Pre-fix" windows are
re-derived as in section 2.1; the first window of a pre-fix GPU run contains the 29-token prefill
and is shown in parentheses.

### 3.1 MiniMax-M2 THRIFT IQ1_M (62 layers, 116 experts top-8, dim 3072, expert FFN 1536)

Prompt "What is the capital of France?" (29 tokens), 60 tokens requested, 11 threads. 2348 MiB VRAM
with the cache off, 5320 MiB with it on (1486 slots printed as 2194 MB).

| Run | Configuration | Windows (ms per decoded token) | attn | moe_ffn | output | Wall (prompt + gen) | tok/s |
|---|---|---|---|---|---|---|---|
| m1c | CPU-only, 11 threads, pre-fix (first window = closing projection + 9 tokens) | 1612, 1620, 1677, 1937, 2164 | 308–445 | 1262–1683 | 28–35 | 165.3 s (29 + 56) | 0.5 |
| m1a | GPU attention, cache off, pre-fix | (7284), 1818, 1877 | 260–262 | 1488–1546 | 69–70 | 117.4 s (29 + 33) | 0.5 |
| m1b | GPU attention + hybrid cache, pre-fix | (7337), 1332, 1297, 1460, 1494, 1644 | 208–225 | 1027–1366 | 59–65 | 147.6 s (29 + 60) | 0.7 |
| m2a | m1b + same-priority spin keeper 32×32, 500/500 µs, started at context creation, pre-fix | (5338), 1754, 1814, 1877, 1807 | 164–179 | 1520–1643 | 62–66 | 129.7 s (29 + 51) | 0.6 |
| m3a | GPU attention, cache off, profiler fixed | 1755, 1743 | 258–261 | 1411–1425 | 70–72 | 84.9 s (29 + 23) | 0.6 |
| m3b | GPU attention + hybrid cache, profiler fixed | 1218, 1302, 1258, 1314, 1356, 1426 (cumulative 1218 → 1313) | 191–216 | 965–1148 | 56–62 | 118.9 s (29 + 60) | 0.8 |
| m3e | CPU-only, 16 threads | 1592, 1777, 1922, 2106, 3066 | 320–916 | 1239–2108 | 31–41 | 170.8 s (29 + 56) | 0.5 |
| m5f | m3b + lowest-priority 16 MB memory-fill keeper, 1000/1000 µs | 5756, 5327, 6271, 6637, 5986, 6602 | 201–255 | 5086–6331 | 39–71 | 516.8 s (29 + 60) | 0.2 |
| m3c, m3d | same-priority heavy keepers (160×256 spin; 64 MB fill), 2000/0 µs, started at context creation | never finished loading within the batch's 1500 s timeout | — | — | — | — | — |

Cache hit rate in m3b: cumulative 17.5 % → 22.3 % over 60 tokens, per-window 17.5, 23.5, 26.8,
25.9, 26.6, 27.7 %; 495 units cover 6.9 % of the 7192 (layer, expert) pairs. P-states in the
generation window (section 2.3): m1a P8 55 of 55; m3a P8 26 of 27 (the other the load's residual
P4); m1b P8 63 / P5 7; m3b P8 31 / P5 6; m2a P8 36 / P5 5 (the same-priority keeper did not lift
it); m5f P4 189 / P3 3 / P0 1 at 5501–8001 MHz memory and 10.6–17.2 W (the memory keeper lifted it
and starved the experts). SM 210–495 MHz and 4.6–8.8 W in every base run.

Caveats. The m1c and m1a/m1b comparison is only valid window by window, pairing the same decoded
token positions (windows 2–5 of each run; both paths count the prompt's closing projection as their
first token): the default placement is 18–31 % faster than CPU-only, and GPU-no-cache is 12 % slower
than the CPU windows at the same position and faster than m1c's late ones (1937, 2164). The clean
m3a/m3b pair confirms this without arithmetic: GPU + cache 1218–1426 against CPU-only 1612–2164
(m1c) is 24–34 % faster window for window; GPU no-cache 1743–1755 sits inside the CPU range. Wall
time per generated token including prefill also favours the default (m3b 118.9 s for 89 forwards =
1.34 s against m1c 1.94 s and m3e 2.01 s). The m2a keeper was the pre-fix, same-priority variant
started before the weights were uploaded; its "+15–50 % on the expert phase" is a single
non-interleaved pair against m1b and is not the CPU-side cost of a keeper (section 6.4). The m3e
16-thread run is slower than the 11-thread run on every phase (section 6.1).

### 3.2 Qwen3-Coder-30B-A3B-Instruct Q4_K_M (48 layers, 128 experts top-8, dim 2048, expert FFN 768)

Prompt "Write a Java method that reverses a linked list." (19 tokens), 100 tokens, 11 threads unless
stated. 1436 MiB VRAM with the cache off, 5916 MiB (the card's ceiling) with it on (2525 slots
printed as 3106 MB). The m2 runs used the France prompt (15 tokens) with only 7 tokens generated
and printed no profile line. Each run's cache series is near-identical (deterministic routing at
temperature 0: 17,965–18,401 hits of 45,696 selections, within 0.5 % of each other; 39–40 %
cumulative, 29–54 % per window). The last column counts the 2 s samples of the generation window
(section 2.3).

| Run | Configuration | Cumulative at 100 | Window range | attn | moe_ffn | output | Wall | tok/s | P-states, generation window (2 s samples) |
|---|---|---|---|---|---|---|---|---|---|
| m4a | CPU-only, 11 threads | 194.1 | 186–206 | 43.6 | 122.2 | 28.0 | 23.5 s | 5.1 | — |
| m4b | CPU-only, 16 threads | 208.7 | 189–273 | 52.1 | 129.5 | 26.6 | 26.2 s | 4.8 | — |
| m4e | CPU-only, 11 threads, seven minutes after m4a | 383.6 | 315–425 | 102.1 | 238.5 | 42.3 | 45.9 s | 2.6 | — |
| m4c | GPU attention + hybrid cache | 210.5 | 148–278 (148, 159, 159, 242, 246, 249, 278, 242, 203, 180) | 47.7 | 153.4 | 9.4 | 32.0 s | 4.7 | P5 7, P4 5, P3 1 (no P8) |
| m4d | GPU attention, cache off | 260.2 | 228–300 | 103.7 | 134.7 | 21.7 | 34.2 s | 3.8 | P5 12, P8 3 |
| m5a | GPU attention + hybrid cache (base) | 290.7 | 240–376 | 68.6 | 209.2 | 12.8 | 39.7 s | 3.4 | P5 10, P4 5, P8 2 |
| m5b | m5a + light lowest-priority spin keeper 4×64, 1000/1000 µs | 190.6 | 164–240 | 29.9 | 156.1 | 4.5 | 27.3 s | 5.2 | P4 10, P5 3 |
| m5c | GPU attention + hybrid cache (base, again) | 523.8 | 430–856 | 109.9 | 391.4 | 22.5 | 68.5 s | 1.9 | P5 20, P8 9 |
| m5d | m5c + lowest-priority 16 MB memory-fill keeper, 1000/1000 µs | 614.2 | 223 → 1583, rising | 77.0 | 532.2 | 4.9 | 74.5 s | 1.6 | P0 22, P4 8, P5 2, P3 1 |
| m5e | + light spin keeper (again) | 262.7 | 204–324 | 41.6 | 214.2 | 6.8 | 35.7 s | 3.8 | P4 13, P5 3, P8 1 |
| m3g | CPU-only, 16 threads | 342.7 | 314–387 | 95.0 | 214.1 | 33.0 | 41.7 s | 2.9 | — |
| m3f | CPU-only, 11 threads, right after the 22.5 GB MiniMax preload (disk reads in the trace) | 1422.7 | 1054–1776 | 516.4 | 788.6 | 116.3 | 164.2 s | 0.7 | — |
| m3h | GPU attention + hybrid cache | 308.3 | 266–340 | 71.6 | 223.0 | 13.6 | 45.8 s | 3.2 | P5 7, P8 3, P4 2, P3 1 |
| m3i | GPU attention, cache off | 719.3 | 251–817 | 166.5 | 513.4 | 39.3 | 90.4 s | 1.4 | P8 26 of 26 |
| m3j | + same-priority 160×256 spin keeper, 2000/0 µs, started at context creation | 18,131 at 30 tokens | 14,680–24,740 | 247–252 | 14,424–24,473 | 7–20 | timeout | — | P4 146, P3 18, P5 7, P0 4 |
| m3k | + same-priority 64 MB fill keeper, 2000/0 µs, started at context creation | never finished loading (1500 s) | — | — | — | — | — | — | no generation; whole trace P3 191, P4 49, P0 15 |
| m2b | GPU attention, cache off, 15 + 7 tokens, pre-fix | — | — | — | — | — | 8.1 s | 2.9 | P5 1, P8 1 |
| m2c | m2b + same-priority 500/500 keeper | — | — | — | — | — | 13.9 s | 1.5 | P8 5 of 5 |
| m2d | GPU + cache + same-priority 500/500 keeper | — | — | — | — | — | 10.1 s | 3.0 | P5 2, P8 1, P4 1 |

Clocks and power in the generation window: m4c SM 405–1185 MHz, memory 810–7001 MHz, 7.1–12.6 W;
m4d SM 210–510, memory 405–810, 5.8–9.0 W; m5a / m5c (base) SM 210–1110, memory 405–5501,
5.5–10.6 W; m5b / m5e (light keeper) SM 360–1425, memory 405–5501, 6.2–12.3 W; m5d (memory keeper)
SM 960–2400, memory 810–8001, 10.4–24.6 W; m3i SM 210–225, memory 405, 5.4–6.5 W.

Caveats. Only the m4 batch (m4a → m4c → m4d → m4e, back to back) and the m5 batch (interleaved
base / keeper) are usable as pairs; m3f, m4e and m5c show what an unpaired run can do. The clean
phase picture in m4: GPU attention with the cache on 26–79 ms per window against the CPU's 41–48;
GPU attention with the cache off 91–123 against the CPU's 41–48, i.e. 2.1–2.6 ms per layer for
10.9 MB of weights (4–5 GB/s effective, the idle memory clock); GPU output projection 5–16 ms
against the CPU's 27–29; moe_ffn with the hybrid cache 113–193 against CPU-only 117–130 while 40 %
of the expert selections are served by the GPU, which means the resident experts are on or near
the critical path of the layer at these clocks; moe_ffn with the cache off 119–160, i.e. the CPU
expert phase is 3–10 % slower with the GPU active than in CPU-only mode. Prefill plus warm-up per
prompt token (`timeMs − N × total_N`, divided by 19): m4a 216, m4e 396, m3g 391 (CPU batched);
m4d 432, m4c 574, m3h 788, m3i 972 (GPU-mode per-token), used in F6.

### 3.3 GLM-4.7-Flash Q4_K_M (DeepSeek2 engine, MLA attention, `MlaAttentionCudaPass`)

No GLM run belongs to the `an/` batch; the figures come from the earlier part of the day and are
recorded in the code and in the findings text (section 7.4). The model has 47 layers, of which the
first is dense (`leading_dense_block_count`) and 46 are MoE. CPU-only 253 ms per token. GPU
MoE-optimized after the day's fixes 184 ms per token, attention 65 ms (1.4 ms per layer over 47
layers), output 10.6 ms, moe 104 ms after the shared expert was folded into the MLA pass (282 ms
before). Re-enabling `MatmulPool` for the MoE-optimized placement alone moved decode from 393 to
312 ms per token (comment in `LLMEngine` next to `MatmulPool.enable()`). An early run without the
expert cache (`o.txt`) shows the output projection at 846 then 567 ms per token: the first token's
lazy 250 MB upload and a P8 clock, not oversubscription (the run held an estimated 1.7 GB of VRAM;
`o.txt` has no smi trace). The DeepSeek2 profiler already dropped the prefill (`profPrefillDropped`
in `DeepSeek2InferenceEngine.outputProjection`), so these averages are per decoded token.

### 3.4 GPT-OSS-20B, DeepSeek-Coder-V2-Lite, Qwen3.5-35B-A3B

These were measured earlier in the session or in earlier releases, not in the `an/` batch; they are
listed so the fix plan's scope is complete, and each number is flagged in section 7.4.

| Model | Engine | Numbers on record | Where recorded |
|---|---|---|---|
| GPT-OSS-20B MXFP4 (32 experts top-4) | Qwen3MoE, hybrid cache | cache on 378 ms per token (attn 34, output 13, moe 331) vs cache off 1059 (attn 139, output 68, moe 852) — the same attention and output code is 4–5× faster with the cache on because the extra expert launches lengthen the GPU bursts, i.e. part of the cache's benefit is clock state | findings text only |
| DeepSeek-Coder-V2-Lite Q4_K_M (64 experts) | DeepSeek2, hybrid cache | hybrid split 3.1 → 6.8 tok/s; upload-on-miss cache 2.5 → 0.8 tok/s (why DS2 is gated on `GpuExpertCache.hybrid()`); 1103 slots printed as 3222 MB because one Q8_0 `ffn_down_exps` sizes every slot to 2.92 MiB | `LLMEngine.tryInitExpertGpuCache` comment; `smoke_out.txt` (one 32-token run at 1.0 tok/s) |
| Qwen3.5-35B-A3B | Qwen3.5 pass + `SoftmaxMoe` | CPU 2.1 → GPU pass 3.6 → with hybrid cache 5.6 tok/s (~178 ms per token); the pass runs per launch because `graphAvailable = !hasMoe` | findings text only |

### 3.5 Microbenchmarks and VRAM facts

`m4.txt` (RoundTrip, one layer's host round trip with the GPU at P4 right after start-up), three
repetitions:

| Measurement | Run 1 | Run 2 | Run 3 | Unit |
|---|---|---|---|---|
| Pinned 8 KB upload + 10 launches of a one-block kernel + 16 KB download + sync | 189.1 | 154.4 | 159.6 | µs |
| Single launch | 8.2 | 8.2 | 8.6 | µs |
| Idle download + sync | 26.7 | 32.1 | 25.9 | µs |

The engine measures 0.56–7.8 ms per layer for the same shape, so the floor is 2–30 % of the
attention phase depending on the clock state.

`m5.txt` (MtDot, the IQ1_M expert dot on one 120 MB slab, static row partition):

| Condition | 1 thread | 4 | 8 | 16 | 22 | Unit |
|---|---|---|---|---|---|---|
| GPU idle | 1.23 | 5.12 | 8.64 | 12.94 | 13.81 | Gelem/s |
| Second JVM looping the output-projection matmul (GPU at P4, 1605 / 5501 MHz, 10.6 W) | 1.30 | 4.97 | 8.00 | 10.62 | 13.43 | Gelem/s |

The kernel is compute-bound (13.8 Gelem/s is ~3 GB/s of weights).

VRAM: MiniMax 2348 MiB (cache off) → 5320 MiB (cache on), a delta of 2972 MiB = 1486 slots × 2 MiB
exactly, for a cache that printed 2194 MB; Qwen3-Coder 1436 → 5916 MiB against a demand of
1436 + 2525 × 2 MiB = 6486 MiB, so at least 570 MiB sits outside dedicated VRAM. No swap and no
disk reads during any GPU-mode decode (`mem_samples.txt`: si = so = 0, bi ≈ 0, MemAvailable
≥ 27.5 GB).

---

## 4. Root causes

Each cause is stated as it survived the review: the mechanism, the evidence, the affected engines,
the cost, and what the three verdicts corrected. Refuted causes are in section 6.

### 4.1 C1 — Measurement methodology (confirmed)

Mechanism: section 2.1 (profiler), 2.2 (drift), 2.3 (sampling), 2.4 (CPU clock). Two further
points the review added: `ExpertGpuCache.getStats()` had no caller at HEAD, so no pre-m3 run knows
its hit rate (the working tree now appends it to every `Qwen3MoEInferenceEngine.printProfile` line;
`DeepSeek2InferenceEngine` and `Qwen35InferenceEngine` still print nothing); and nothing times the
three parts of the hybrid moe phase (`launchResident`, `cpuExpertCompute`, `finishResident`), so the
GPU wait inside it is inferred, never measured. Evidence: `Qwen3MoEInferenceEngine.forwardPrefill`
(gate and per-token loop), `forwardLayer` (counters), `outputProjection` (`profTokenCount++`,
`profPrefillDropped`); m1a/m1b vs m3a/m3b in section 3.1; `CLIRunner.autoTune` / `measurePlacement`
(24 tokens after a 6-token warm-up, one fresh load per candidate, ties to the GPU).

Corrections applied: `--auto-tune` is not biased by the profiler (its tok/s is decode-only, from
`GenerationResponse`), only by drift; the cost estimate "200–300 ms of the reported 218 on
Qwen3-Coder" was arithmetically impossible and is replaced by the formula value of ~38 ms
(section 2.1), which was never measured directly; the "16 s ramp" of `clk.txt` does not apply to
MiniMax, which never ramps.

Affected: every Qwen3MoE-engine GPU-mode number of runs m1a–m2d, the MoE rows of `BENCHMARKS.md`
that were taken with `-Dcpu.profile=true` in GPU mode, and the validation of every other cause.
Cost: none in tok/s; it decides the conclusion.

### 4.2 C2 — The GPU decodes in its idle or near-idle P-state (disputed; corrected as below)

Mechanism: per layer the main thread does one host-to-device copy, 14 kernel launches and one
blocking device-to-host copy (`MoeAttentionCudaPass.attentionLayer`; the MLA pass ~21–24 launches),
then computes the routed experts on the CPU for 2.5–27 ms (Qwen3-Coder 122–130 ms / 48 layers,
MiniMax 965–1683 / 62; about 2.3 on GLM from the unverified 104 ms figure) while the GPU idles.
The driver's governor (in the Windows WDDM driver; WSL2 only forwards submissions) treats a 2–15 %
duty cycle as idle. At P8 the memory clock is 405 MHz instead of 5501–8001 (about 9.7 GB/s on the
96-bit bus instead of 192) and the SM clock 210–300 MHz instead of 1605–2400; at P5 the memory
clock is 810 MHz and the SM 400–500 MHz. Bandwidth-bound kernels (the attention projections, the
output projection, the resident expert matmuls) and compute-bound ones (the IQ1_M / IQ2_XXS
grid-lookup kernels, which `Dp4aMatmul.kernelFor` does not cover) stretch several-fold, and so
does every launch and sync gap.

Evidence: every MiniMax cache-off generation-window sample and 85 % of the cache-on samples are
P8 (section 2.3); MiniMax attention 192–262 ms = 3.1–4.2 ms per layer for 11.4 MB of weights, and
the 422 MB Q5_K output projection 59–75 ms (5.6–7 GB/s) against 30–34 ms on the CPU; Qwen3-Coder
cache-off attention 91–123 ms (4–5 GB/s, at P5/P8) against 41–48 on the CPU (m4d/m4a). The
decisive evidence is the interleaved keeper pair: m5a → m5b attention 68.6 → 29.9, output
12.8 → 4.5, moe 209 → 156, total 291 → 191 ms with identical hit counts, and m5c → m5e 524 → 263;
the memory-bound 16 MB keeper (m5d) lifted the memory clock to 8001 MHz at 22–25 W and made moe
worse (229 → 1390 ms across the run) because it competes for the bandwidth the resident-expert
kernels need.

Corrections applied: the blanket "5–15× below spec on every GPU phase" is an upper bound computed
from clock ratios, not a measurement; Qwen3-Coder with the cache on is at P5/P4/P3 in every
generation-window sample of m4c, and at P5 in most cache-off samples (m4d), so for Qwen3-Coder the
loss is a low-clock hunting loss rather than a deep-P8 one (deep P8 throughout only in m3i and on
MiniMax); the earlier same-priority keeper (m2a, 32×32, 500/500 µs, started at context creation)
did not lift the P-state and its m2a-vs-m1b CPU penalty is a single unpaired comparison; the
memory-bound variant proposed as the fix had already been measured and starved the model (m3d/m3k
never loaded, m3j decoded at 18–24 s per token); the keeper in the tree is not yet on a
lower-priority stream — `least` from `cuCtxGetStreamPriorityRange` is 0, the same priority the work
stream gets from plain `cuStreamCreate` (F1 step 1; the `cuda.clockkeeper` row of `jvm-flags.md`
carries the same overstatement). Even at P8 the GPU attention phase beats the CPU's on MiniMax
(192–262 vs 308–445), so the P8 loss is a contributor on MiniMax and the whole GPU-mode penalty on
Qwen3-Coder cache-off.

Affected: every MoE-optimized placement (Qwen3MoE, DeepSeek2, Qwen3.5-MoE engines) and dense
partial offloads whose CPU layers dominate; not full-offload graph mode, where the GPU is
continuously busy. Cost: Qwen3-Coder cache-on ~100 ms of 291 (m5a/m5b), cache-off ~60–100 ms of
260 (m4d, windows 228–300: attention 91–123 against the keeper-level 26–36); MiniMax ceiling
~120–220 ms of 1218–1313, i.e. 9–18 % (attention 192–216 → 50–100 saves 92–166 ms, output 60 →
4–30 saves 30–56 ms), unmeasured with the light keeper.

### 4.3 C3 — Expert cache slots sized by the largest slice and allocated one `cuMemAlloc` each (confirmed)

Mechanism: `GpuExpertCache.create` computes one `slotBytes` as the maximum over every layer and every
projection of `elementsPerSlice / blockSize × typeSize`, and `maxSlots = maxCacheBytes / slotBytes`;
the `ExpertGpuCache` constructor then issues `maxSlots` separate `CudaContext.allocBuffer(slotBytes)`
calls, each backed by 2 MiB pages, and prints `maxSlots × slotBytes`, not the physical footprint.
Mixed-quant GGUFs always carry an outlier: MiniMax IQ1_M ships Q2_K `ffn_down_exps` in 7 of 62
layers (1,548,288 B per slice against 1,032,192 for IQ1_M); Qwen3-Coder Q4_K_M ships Q6_K
`ffn_down_exps` in 24 of 48 layers (1,290,240 B against 884,736 for Q4_K); DS-Coder-V2-Lite ships
Q8_0 in 12 of 26 (3,063,808 B); GLM-4.7-Flash Q6_K in 22 of its 46 MoE layers.

Evidence: section 3.5 (2972 MiB = 1486 × 2 MiB; 5050 + 1436 > 6140 on Qwen3-Coder); the printed
slot counts reproduce exactly from the slice sizes (1486 × 1,548,288 = 2194 MB; 2525 × 1,290,240 =
3106 MB; 1103 × 3,063,808 = 3222 MB). `LLMEngine.tryInitExpertGpuCache` budgets
`free − max(512 MiB, total / 10)` from `cuMemGetInfo`; the 614 MiB margin was consumed on MiniMax
(778 MiB over the printed size) and the card was pegged on Qwen3-Coder.

Corrections applied: the waste has two parts — 2 MiB rounding alone is 26 % on MiniMax and 39 % on
Qwen3-Coder, and rounding plus slot-versus-slice slack is 50–55 %; the unit gain of the fix is
1.74× (MiniMax) and 2.06× (Qwen3-Coder) with per-projection sizes in the same physical bytes, 2.0–2.4×
with a per-layer table, not 2.6×; MiniMax was not oversubscribed (820 MiB stayed free), only
Qwen3-Coder was; the "218 → 150–170 ms" speed estimate is not derivable from this session's data,
because resident reads at the P8 memory clock (~14 GB/s) and PCIe-backed slots (6–12 GB/s) are
indistinguishable at this level — treat the fix as capacity and correctness, and measure its speed
effect through the hit rate after F1.

Affected: every hybrid-cache user (Qwen3MoE, DeepSeek2, Qwen3.5-MoE), worst where `slotBytes` is
just above 1 MiB. Cost: MiniMax 778 MiB of VRAM beyond the printed size, and 495 resident units
where an arena at the global slot size would hold 670 (+35 %) and per-projection sizes 862 (+74 %,
i.e. 43 % of the possible units are lost); Qwen3-Coder 1944 MiB and 841 units against 1730
(+106 %), plus the oversubscription that produced the 55 ms output-projection trap.

### 4.4 C4 — Per-layer launch and synchronization floor, no graphs (confirmed)

Mechanism: `MoeAttentionCudaPass.attentionLayer` writes position and `x` from `hostBlock` (allocated
with `arena.allocate`, i.e. pageable, so the driver stages the copy synchronously), launches
rmsnorm, quantize, wq, wk, wv, two QK-norms, two RoPE, kv update, flash attention, quantize, wo and
rmsnorm individually (14 launches; up to 4 bias accumulates more on GPT-OSS), then
`CudaContext.readBuffer` = `cuMemcpyDtoHAsync` + `cuStreamSynchronize`. `MlaAttentionCudaPass` does
the same with ~21–24 launches (counted from the code; no GLM run in `an/`) and computes the shared
expert before the same download, so it cannot overlap the CPU experts. With the hybrid cache,
`ExpertGpuCache.finishResident` adds a second `cudaContext.finish()` per MoE layer.
`Qwen35CudaForwardPass` sets `graphAvailable = !hasMoe` and `forwardMoeFFN` syncs once per MoE layer
around the host callback, so Qwen3.5-35B-A3B runs every layer per launch — an estimated 30–40
launches per layer, from 10 and 21 direct launch calls in `forwardDeltaNetLayer` /
`forwardAttentionLayer` before helper expansion (unverified; see F10). The context is created with
flags 0 (`CU_CTX_SCHED_AUTO`), so every `cuStreamSynchronize` spin-waits on one core. Everything
between the upload and the download has a fixed launch configuration (positions read on the device
from `tokenParams`, the flash grid from `smCount`, constant weight pointers), so the sequence is
capturable with the existing `beginCapture` / `endCapture` / `instantiateGraph` / `launchGraph` API.

Evidence: `m4.txt` floor (8.2 µs per launch, 26–32 µs idle download + sync, 155–190 µs per 10-launch
round trip); host submit alone is 48 × 14 × 8.2 µs = 5.5 ms per Qwen3-Coder token, 62 × 14 × 8.2 =
7.1 ms on MiniMax, 47 × 21 × 8.2 = 8.1 ms on GLM (launch count from the code).

Corrections applied: a per-layer graph keeps one download + sync per layer, so the recoverable part
is the per-layer floor minus that sync — ~100–130 µs per layer for the 14-launch GQA pass and
~150–190 µs for the 21–24-launch MLA pass — which gives 5–6 ms per token on Qwen3-Coder (48 layers),
6–8 on MiniMax (62) and 7–9 on GLM-4.7-Flash (47), 2–4 % of a token at today's clocks; the
"2–10 ms of GPU-side inter-kernel gaps" and "1–2.5 ms of pageable staging" are unmeasured; the
second sync of the hybrid cache is mostly idle (the resident experts finished long before) and is
skipped when no expert is resident; the sync-count targets (97 → 49 etc.) are not reached by graphs
alone; BLOCKING_SYNC cannot buy throughput because nothing runs on the CPU during the attention
pass. Under the keeper (m5b, 30 ms attention) the floor becomes 25–30 % of the attention phase,
which is why F3 depends on F1 for measurability.

Affected: `MoeAttentionCudaPass` (Qwen3-Coder, MiniMax, GPT-OSS, GLM4-MoE, Llama 4),
`MlaAttentionCudaPass` (GLM-4.7-Flash, DeepSeek-V2/V3 family), `Qwen35CudaForwardPass` with experts.

### 4.5 C5 — `MatmulPool` is switched off for every GPU placement that is not MoE-optimized (confirmed)

Mechanism: `LLMEngine.load` calls `FloatTensor.disableVirtualThreadMatmul()` right after `initGpu`,
before the placement is decided; that helper also calls `MatmulPool.disable()`. The pool is
re-enabled only for `q35Engine` with experts and for `q3moeEngine` / `ds2Engine` when
`moeOptimizedGpu` is true (both guarded by `-Dmoe.cpu.pool`). Dense partial offloads, the MoE
first-N fallback (taken when non-expert bytes exceed 80 % of VRAM) and explicit `--gpu-layers N` on
a MoE model without `--moe-optimized` run their CPU layers on `IntStream.parallel()` per row,
per-head attention on the stream fallback of `MatmulPool.forEach`, and expert loops on the chunked
stream fallback of `forRange`. `MatmulPool.enabled()` is also used as a "no GPU tensors here" proxy
that gates batched prefill (`InferenceEngine.canBatchPrefill`, `Qwen35InferenceEngine`,
`Gemma4InferenceEngine`, `ExpertViews.active()` and hence `MoEFFN.batchAvailable()`), which is why
the flag cannot simply be flipped. The disable is static and process-sticky: a CPU-only load after
a GPU load in the same JVM never gets the pool back, so `CLIRunner.autoTune` measures its CPU-only
candidate on the fallback dispatcher and is biased against the CPU.

Evidence: `FloatTensor.disableVirtualThreadMatmul`, `MatmulPool.disable` / `enable` /
`configuredThreads`, the two `MatmulPool.enable()` call sites in `LLMEngine`; the GLM-4.7-Flash
393 → 312 ms comment next to the second one (1.26×, the only measured pool-vs-fallback datum under
CUDA); `cpu-dispatch-and-kernels.md` for the 1.28–1.77× pool gain, measured against the
virtual-thread dispatcher, not against the `IntStream` fallback GPU mode actually takes.

Corrections applied: this cause explains none of the session's numbers (every GPU run was
MoE-optimized with the pool on); a GPU tensor reaching `matmulRowsBatch` through the pool computes
correctly on its CPU twin (`CudaFloatTensor.dot` delegates to `cpuTwin()`), so the residency gate is
a performance safeguard, not a crash guard; `-Dmatmul.pool=force` already keeps the pool on under a
GPU backend, so the hypothesis for dense partial offloads can be measured before any code changes.

Affected: dense models with `--gpu-layers N < blockCount` or an auto partial offload (above ~8B on
this GPU), Nemotron-H / LFM2 first-N prefixes, MoE first-N and explicit `--gpu-layers` placements,
`-Dmoe.cpu.pool=false`, and every `--auto-tune` CPU-only calibration.

### 4.6 C6 — No single VRAM planner and no allocation is ever verified (disputed; not the cause on either model)

Mechanism: placement budgets 0.80 × device total from `sumNonExpertTensorBytes` plus the FP32 KV
estimate; `sumMoELayerNonExpertBytes` has no output-projection term (`estimateNonLayerBytes` is used
only on the dense branch), counts `ffn_gate_inp` although the router is created as an `F32CudaTensor`
but consumed only through `FloatTensor.matmul` → `dot` → CPU twin and never uploaded, and ignores the
FP16 copies the MLA pass uploads for `wk_b` / `wv_b`, the pass buffers, the CUDA context and modules,
and the 2 MiB granularity. The cache then takes `cuMemGetInfo` free minus a blanket margin, after
which shared experts, dense-layer FFN tensors, the cache's own input and per-expert buffers, the
expert biases and the flash kernel variant allocate lazily during the first forward pass. Under WDDM
nothing fails (section 2.5). For DeepSeek2 models with Q-LoRA and separate `attn_k_b` / `attn_v_b`
tensors, `sumMoELayerNonExpertBytes` asks for `ArchitectureRegistry.attnQ(layer)` and
`attnKvB(layer)`, names those models do not carry, and `tensorByteSize` returns 0 for a missing
name, so those projections are counted at zero bytes.

Corrections applied: MiniMax never oversubscribed (5320 of 6140 MiB), so this is not the MiniMax
cause; on Qwen3-Coder the residual spill after the output-upload reorder costs at most ~6–9 ms per
token; the "1.3–3.5 ms at 192 GB/s" baseline for a resident output projection is wrong in the engine
at P8 (39–41 ms in m3i, 21.7 in m4d, 9–14 with the cache on); `o.txt` was not an oversubscribed run;
the "context + modules" residual is 25–115 MiB, not 55–60; a cuMemGetInfo-after-alloc check is
unverified as a residency signal under WSL2; a single multi-GiB arena is the wrong granularity under
WDDM (VidMm residency is per allocation). What survives is plan-side: count the output, the MLA
FP16 copies and the Q-LoRA tensors, stop counting the router, warm every lazy consumer before sizing
the cache, and add a byte cap — see F7.

### 4.7 C7 — Small per-tensor GPU matmuls and the output projection at idle clocks (disputed)

Mechanism: `CudaFloatTensor.matmulParallel` → `gpuMatmul` performs, under `perTensorLock`, a staging
copy and `cuMemcpyHtoDAsync`, one launch of the tensor's FP32 kernel (never `Dp4aMatmul`), a blocking
`readBuffer` and a scalar accumulate loop; nothing overlaps. This path is taken by the Qwen3MoE-family
shared experts (three round trips per layer on GLM-4.5 / Llama 4; MiniMax and Qwen3-Coder have none),
by the whole attention when the pass is off or declines a layer (4–6 round trips per layer), by
engines without a pass (Ling, LFM2-MoE), and by the output projection of every MoE model.

Corrections applied: on Qwen3-Coder the GPU output projection beats the CPU (9–14 ms with the cache
on, 21.7 cache-off at P5, 39–41 at deep P8 in m3i; CPU 26.5–42 ms), so "GPU loses" holds only for
MiniMax (GPU 59–75 vs CPU 30–34 ms, 422 MB Q5_K through the FP32 kernel at P8); the earlier
"6 ms CPU / 3.5 ms GPU" triple mixed GPU-only figures (the 3.5 ms is an `LLMEngine` comment, not a
measurement file); the byte-threshold formula gives a negative crossover at P8 (nothing wins) and
0.6–1.5 MB at full clocks; a load-time "prefer the twin" decision would measure the P4 state right
after upload and route Qwen3-Coder's output to the CPU, a regression — the decision must be made at
runtime or by placement; the router occupies no VRAM today because it is never uploaded; the
shared-expert "25–40 ms of pure overhead" multiplied a 10-launch layer floor by single-launch calls
and is 7–9 ms (sync) plus ~50 ms of P8 bandwidth on a GLM-4.5-class model, which folding into the
pass does not remove.

Affected: MiniMax output projection (measured), GLM-4.5 / Llama 4 shared experts and pass-declined
attention (unmeasured this session). Cost: MiniMax +26 to +45 ms per token (2–3.5 %).

### 4.8 C8 — CPU expert rows chunked over the full top-K slot range (disputed)

Mechanism: `Qwen3MoEInferenceEngine.cpuExpertCompute`, `MoEFFN.routedExperts` and
`SoftmaxMoe.routed` call `MatmulPool.forRange(expertUsedCount × efd, EXPERT_ROW_CHUNK, ...)` and skip
the GPU-resident slots inside the lambda, so `parallelFor` sizes chunks as
`units / (threads × CHUNKS_PER_THREAD = 4)` over the full range: MiniMax 12288 rows / 44 = 280-row
chunks, 5.49 chunks per expert; Qwen3-Coder 140-row chunks, same ratio; GPT-OSS 262-row chunks,
exactly 11 per expert.

Corrections applied: the imbalance is a sawtooth in the number of CPU slots, bounded by one chunk
(dynamic claiming): 0 % at 2, 4 or 6 resident experts, +15 % at 1, +20 % at 3, +33 % at 5 on the
linear model, and about half of that once the tail round runs on fewer threads with more bandwidth
each; GPT-OSS is a non-case at 11 threads; measured residency is ~2 of 8 on MiniMax (17–28 % hit
rate) and ~3–4 on Qwen3-Coder (37–54 %), so the expected loss is ~0–8 % of the moe phase (20–80 ms
per token on MiniMax, 5–15 on Qwen3-Coder), not 150–230; the finishResident wait can be the critical
path instead. The fix (F11) is correct, cheap and bit-identical, but second order.

### 4.9 C9 — `MatmulPool` workers park during every GPU attention phase (disputed)

Mechanism: workers spin `SPIN_LIMIT = 20000` `onSpinWait` iterations (0.07–0.9 ms) and then
`LockSupport.park`; in GPU mode the main thread blocks in `cuStreamSynchronize` for the attention
phase, so the workers park once per layer and are unparked serially by the next `parallelFor`. This
holds where the attention phase clearly exceeds the spin window: MiniMax (3.4–4.2 ms per layer) and
Qwen3-Coder with the cache off (2.1 ms per layer). With the cache on, Qwen3-Coder attention is
0.56–1.0 ms per layer, at the edge of the spin window, so whether the workers park there is unknown.
In CPU-only mode the dispatches are back to back and the workers never park inside a token.

Corrections applied: parking happens once per layer, not twice (the SiLU loop between the two
`forRange` calls is microseconds); the penalty this would explain is 3–10 % of the CPU expert phase
(m4d vs m4a: 129–135 vs 119–122 ms), 7–13 ms per token, and the m5 GPU-busy loss already covers it;
the "core frequency ramp" term is unobservable under WSL2; dense partial offloads are not affected
because the pool is off there (C5). The `-Dmatmul.spin` A/B has never been run (the `_spin` files
are keeper runs). See F12.

---

## 5. Fix plan

Fixes are ordered by priority; dependencies are stated so an implementer can reorder within them.
Line numbers are deliberately omitted — cite file and method. Every fix uses the same template:
**Why** (where needed), **Where**, **How**, **Validate**, **Expected gain**, **Risks, effort,
dependencies**.

### F0 — Instrumentation and protocol (small; confirmed; partly in the tree)

**Why.** The instruments decide the conclusion (sections 2.1, 4.1). Already in the tree
(section 7.3): `profPrefillDropped` in `Qwen3MoEInferenceEngine.outputProjection` and
`DeepSeek2InferenceEngine.outputProjection`; `getExpertCacheStats()` appended by
`Qwen3MoEInferenceEngine.printProfile`.

**Where.** The three MoE engines, `MoEFFN`, `SoftmaxMoe`, `LLMEngine`, `LLMPlayerMXBean` /
`LLMPlayerMetrics`, `CLIRunner`, `jvm-flags.md`.

**How.**
1. Reset `profPrefillDropped`, the phase sums and `profTokenCount` at the top of `forwardPrefill` in
   both engines, so the second generation of a process is clean; add the same drop to
   `Qwen35InferenceEngine` (its GPU-mode prompt goes through `forwardNoOutput` per token and
   `finishLogits` counts once). `NemotronHInferenceEngine` prefills per token in both modes, so it
   needs only the reset.
2. Print the last-10-token window next to the cumulative average in every `printProfile` (keep the
   previous sums in seven `long` fields; no allocation). Keep the per-layer window timers always on
   (two `nanoTime` calls per layer), not only under `cpu.profile`, because F1's adaptive controller
   reads them.
3. Time the hybrid moe phase in three parts under `cpu.profile`: `nanoTime` around
   `launchResident`, `cpuExpertCompute` and `finishResident` in `Qwen3MoEInferenceEngine.moeFFN`,
   around `launchResident` / `routedExperts` / `finishResident` in `MoEFFN.forward`, and in
   `SoftmaxMoe.forward` (add a `static final boolean` flag there; it has none). They are consecutive,
   so they sum to the phase and the third is the GPU wait.
4. Print `getStats()` for DeepSeek2 (`MoEFFN.gpuExpertCache()`) and Qwen3.5 (`SoftmaxMoe`, which
   needs a package-private accessor) too, and print it once in the CLI `--- Stats ---` block through a
   new `LLMEngine` accessor. Expose hits, misses and residency on `LLMPlayerMXBean` /
   `LLMPlayerMetrics` under new names (`gpuExpertCache*`); the existing `getExpertCache*` getters
   describe the SSD RAM cache and must not be reused.
5. Add a pure-Java CPU speed probe printed with every profile line: a fixed-iteration dependent
   integer loop timed with `nanoTime` on one thread (~1 ms), as the only in-VM proxy for the CPU
   clock (section 2.4).
6. Rewrite the comment above `warmDecodeKernels()` in `Qwen3MoEInferenceEngine.forwardPrefill`: it
   quotes "3111 ms/token in GPU mode against 1414" which compares a prefill-inclusive average with a
   clean one; the like-for-like effect of the call is at most 5–8 % (section 6.2). Gate the
   `Qwen35InferenceEngine.forwardGpu` call behind a boolean so it does not run its lookups on every
   token.
7. Documentation: add rows for `moe.expert.gpu` and `moe.cpu.pool` to `jvm-flags.md` (defined at
   the top of section 7), add the stream-priority caveat to its `cuda.clockkeeper` row until F1
   step 1 lands, and add `gpu.tensor.min.bytes` (F9) and `cuda.sched` (F3) when those ship.

**Validate.** Protocol P on m3b-style runs; the window print must reproduce the section 3 windows to
±1 ms; the three moe timers must sum to the moe phase to within 1 %.

**Expected gain.** None in tok/s. **Risks, effort, dependencies.** No functional risk. Effort small.
No dependencies; everything else depends on it.

### F1 — Clock keeper productization (medium; opt-in until measured end to end)

**Why.** Section 4.2. The light keeper is the only intervention that moved a whole token by a third
in an interleaved pair (m5a/m5b, m5c/m5e). Its power cost is real (GPU 5–9 W in the base decode
windows → 9–12 W with the light keeper; the 16 MB memory variant 10–25 W) and the CPU side of it
is unmeasured, so it ships opt-in.

**Where.** `CudaContext.maybeStartClockKeeper` and `CudaContext.create` (java21), the start call in
the `LLMEngine` constructor, `GpuAttentionPass` and `LayerGpuForwardPass` (base interfaces),
`kernels/cuda/spin.cu`, the `cuda.clockkeeper` row of `jvm-flags.md`.

**How.**
1. Stream priorities: in `CudaContext.create`, query `cuCtxGetStreamPriorityRange` and create the
   work stream with `cuStreamCreateWithPriority(..., CU_STREAM_NON_BLOCKING, greatest)`; keep the
   keeper's side stream at `least`. Today both are 0 (section 4.2), so the "lowest priority" the
   code and the `jvm-flags.md` row claim does not exist yet. Re-run the m5 pair after this change
   first — it may already be part of why m3j starved.
2. Scope: start the keeper from `LLMEngine.generate` (inside the generation lock) and stop it at the
   end with a ~3 s linger so multi-turn requests do not pay the ramp each time; never during load
   (m3c/m3d/m3k hung the preload); stop it in `gpuAttentionFailed` / `disableGpuForwardPass` so a
   CPU-fallback run does not keep burning power. Plumb it through a `default void setBusyHint(boolean)`
   on the two base interfaces, implemented by the CUDA passes, so the base code never imports java21.
3. Adaptive duty cycle: keep the 4×64 spin kernel (it never fills the SMs; the 160×256 grid is what
   starved m3j) and adjust `busyUs` / `sleepUs` at runtime from the last measured attention phase —
   target the smallest duty that keeps the attention phase within 1.5× of its best value, starting
   at 1000/1000 µs, halving the busy period whenever the moe phase grows by more than 10 % while
   attention does not, and doubling the sleep when attention stays flat. The signal must exist when
   profiling is off: use the always-on window timers of F0 step 2, or time the pass with
   `cuEventElapsedTime` (F3 step 5); otherwise the controller silently degrades to the fixed
   1000/1000 duty. Re-calibrate `iters` from `usPerLaunch` every few seconds: it is computed once at
   start-up at whatever clock the GPU has then (P4), so at P8 each burst runs 5–7× longer than
   intended.
4. Replace the busy-wait: poll with a new `cuEventQuery` binding (must not go through `checkError`,
   since `CUDA_ERROR_NOT_READY` is the normal answer) and `LockSupport.parkNanos`, instead of
   `cuStreamSynchronize` under `CU_CTX_SCHED_AUTO`, which spins a core.
5. Logging: the measured runs printed a pointer word as the iteration count (`m3c` "2080586288
   iters", `m5b` "-2011345040 iters"); the source now prints the `iters` variable — verify the line
   at the next run. Print the P-state histogram of the generation window (from `nvidia-smi`,
   `ProcessBuilder`, ignore failures) in the CLI stats block when the keeper is on.
6. Never enable the memory-fill variant (`cuda.clockkeeper.mb`) by default or in the adaptive
   controller; keep it only as an experiment flag (section 6.6).
7. Make the keeper a calibration axis of F8 so its CPU-side cost is measured end to end before any
   default flips.

**Validate.** Protocol P on Qwen3-Coder (cache on and off) and MiniMax (cache on), three interleaved
pairs each, `-Dcuda.clockkeeper=1000 -Dcuda.clockkeeper.sleep=1000 -Dcuda.clockkeeper.blocks=4
-Dcuda.clockkeeper.threads=64` (or the adaptive default) against no keeper, with the Windows-side
CPU counter recorded. Pass: total ms per token better in every pair by more than the base-vs-base
spread (m5a vs m5c was 1.8×, so expect to need the three pairs), attention and output phases better
in every pair, moe phase not worse by more than 10 %, GPU power reported.

**Expected gain.** Qwen3-Coder cache-on −35 to −50 % per token (m5a 291 → m5b 191; m5c 524 → m5e
263; attention 69 → 30, output 13 → 4.5, moe 209 → 156 with identical hit counts). MiniMax:
unmeasured with the light variant (m5f used the memory variant and was harmful); ceiling = attention
192–216 → ~50–100 (saves 92–166 ms) plus output 60 → 4–30 (saves 30–56 ms) = ~120–220 ms of
1218–1313 (9–18 %), against which any CPU-side penalty must be netted.
**Risks, effort, dependencies.** +4–7 W GPU in a shared laptop envelope, possibly lowering CPU boost
(unobservable from the VM — hence the Windows-side counter); WDDM TDR limits kernel duration (keep
launches ≤ 100 ms); a keeper active during load stalls it; graph capture in `Qwen35CudaForwardPass`
is THREAD_LOCAL, so a keeper thread cannot invalidate it, but its buffers must be allocated before
any capture. Effort medium. Depends on F0.

### F2 — Expert cache arena with per-projection unit sizes (small; confirmed)

**Where.** `GpuExpertCache.create` (base), `ExpertGpuCache` constructor, `launchExpert`,
`victimUnit`, `upload`, `uploadExpertSlice`, `close` (java21), `LLMEngine.tryInitExpertGpuCache`.

**How.**
1. In `GpuExpertCache.create` compute a per-projection triple `long[3]` (`sizeGate`, `sizeUp`,
   `sizeDown`, each the maximum over layers of that projection's slice bytes, 256-byte aligned) and
   `unitBytes = sum`; pass the triple through the reflective constructor
   (`getConstructor(CudaContext.class, int.class, long[].class)`), and compute units from `unitBytes`.
2. In `ExpertGpuCache` allocate chunks of 64–256 MiB, each holding `floor(chunkBytes / unitBytes)`
   units, so the 2 MiB rounding is paid once per chunk (≤ 3 % waste) and VidMm can demote a chunk
   rather than the whole cache; keep `unit → (chunk, offset)` and a precomputed `long[3 × units]`
   pointer table filled in the constructor so `launchExpert` stays allocation-free. Shrinking means
   dropping chunks, which replaces the "halve and retry" loop. Check `byteSize > sizeP` per
   projection in `uploadExpertSlice`.
3. Print physical numbers: "Expert GPU cache: N experts, X MiB (cuMemGetInfo delta Y MiB)", and add
   a new property `-Dmoe.expert.gpu.cache.mb` (CLI `--gpu-expert-cache <MB>`) applied as
   `min(cacheBytes, cap)` in `tryInitExpertGpuCache`. `moe.expert.cache.mb` / `--expert-cache-size`
   stay what they are today — the budget of the SSD RAM cache, read in `ExpertCacheFactory` and set
   by `CLIRunner` — and must not be overloaded.
4. Optional second step: a per-layer size class (every GGUF examined has exactly two triples), which
   turns `victimUnit` / `upload` into one chunk list and victim scan per class — an allocator change,
   not pointer arithmetic.
5. Wire `ExpertGpuCache.close()` from `LLMEngine.close` (only the SSD cache is closed today).

**Validate.** nvidia-smi `memory.used` with the cache on minus with it off must equal the printed
physical size within 8 MiB on MiniMax and Qwen3-Coder. Two unit-count checks, because the gain
depends on what is held constant: (a) at the same printed budget (2194 MB MiniMax, 3106 MB
Qwen3-Coder) the unit count must be ≥ 1.25× the old one (636 vs 495; 1064 vs 841), and Qwen3-Coder
must then stay under 6140 MiB (expected 1436 + ~3108 ≈ 4544 MiB); (b) with the cap raised to the old
physical footprint (2972 MiB / 5050 MiB) the unit count must be ≥ 1.7× (862 vs 495; 1730 vs 841).
Generated text identical to the old cache at temperature 0; then protocol P for the hit rate and
moe phase (expect the hit rate to rise, the time effect to depend on F1).

**Expected gain.** Same physical bytes: MiniMax 495 → 862 units (+74 %; +103 % with the per-layer
table), Qwen3-Coder 841 → 1730 (+106 %). Same printed budget: 495 → 636 and 841 → 1064 (+27–29 %).
Same unit count (495): a footprint of 495 × 3,612,672 B ≈ 1706 MiB instead of 2972, i.e. ~1266 MiB
returned to the margin, of which 778 MiB is the 2 MiB rounding alone. Speed: unquantified this
session (section 4.3). **Risks, effort, dependencies.** WDDM residency per chunk (hence the chunk
size); a single large `cuMemAlloc` can fail where many small ones succeeded (fragmentation) — chunks
handle it; keep the 512 MiB reserve until the WDDM accounting question in F7 is measured, because
the ~1 GB `cuMemGetInfo` under-report is what kept MiniMax from overflowing. Effort small. Depends
on F0.

### F3 — Per-layer CUDA graphs, pinned host blocks, event waits (medium; confirmed)

**Where.** `MoeAttentionCudaPass` and `MlaAttentionCudaPass` (constructor, `attentionLayer`,
`takeSharedExpert`, `close`), `CudaContext` (`allocPinnedHost`, `writeBufferAsync`, `readBuffer`,
`beginCapture`, `endCapture`, `instantiateGraph`, `launchGraph`, `createExtraStream`, events),
`CudaBindings` (add `cuEventQuery`, `cuEventElapsedTime`, `CU_CTX_SCHED_*` constants),
`DeepSeek2InferenceEngine.forwardLayer` and `MoEFFN.forward` for the shared-expert reordering.

**How.**
1. Allocate `hostBlock` with `ctx.allocPinnedHost` in both passes and upload with
   `writeBufferAsync` (`writeBuffer` adds a full sync for pinned segments); the trailing `readBuffer`
   makes reuse safe. Free it in `close()`.
2. Per-layer graphs: on the first call of a layer run uncaptured (it compiles the flash split kernel
   and the FP32 fallback kernels for non-dp4a weight types lazily, and the CLAUDE.md invariant
   forbids allocation or module loads inside a capture); on the second call wrap the launches
   between the upload and the download in `beginCapture` / `endCapture` / `instantiateGraph`, store
   the exec per layer, and make `attentionLayer` = `writeBufferAsync` + `launchGraph` + `readBuffer`.
   On any capture error call `endCapture` to clear the invalidated capture (as `Qwen35CudaForwardPass`
   does) and fall back to per-launch for good. Make sure `ExpertGpuCache`'s lazy kernel compile and
   buffer growth happen outside any capture (they do today: between attention calls). Instantiating
   47–62 small graphs is a one-time cost, estimated at about a millisecond each (not measured).
3. MLA shared expert out of the attention sync: queue its six launches after the attention
   `readBuffer` on the same stream, and read `gpuSh` in `takeSharedExpert` with its own `readBuffer`
   after the routed experts. This changes the `GpuAttentionPass.takeSharedExpert` contract ("same
   download, no extra synchronisation") and needs `DeepSeek2InferenceEngine.forwardLayer` to call
   `takeSharedExpert` after `moeFFN.forward` (or `MoEFFN.forward` to take a supplier), since
   `addShared` runs after the routed loop; the hybrid cache issues its expert kernels on the same
   stream, so the overlap is only against CPU work. A failure from the deferred launches must go
   through `gpuAttentionFailed` (CUDA errors are sticky and the CPU never had this sequence's KV).
4. Define `CU_CTX_SCHED_SPIN / YIELD / BLOCKING_SYNC` and create the context with the value of
   `-Dcuda.sched=auto|spin|yield|blocking` (default `auto`); expected neutral for throughput, less
   CPU burned; measure, do not assume.
5. Add `cuEventQuery` and `cuEventElapsedTime` bindings (also needed by F1, F5 and the F0 timers).

**Validate.** Protocol P with F1's keeper on (the floor is invisible at P8 inside the noise),
Qwen3-Coder cache-off and MiniMax cache-off, attention phase; also `-Dcuda.nograph=true`
equivalence (output identical at temperature 0). Pass: attention phase −3 ms or better in every
pair on Qwen3-Coder, −4 ms on MiniMax, no change in generated text.

**Expected gain.** Per-layer floor minus the one download + sync that remains (section 4.4): 5–6 ms
per token on Qwen3-Coder, 6–8 on MiniMax, 7–9 on GLM-4.7-Flash at today's clocks; 25–30 % of the
attention phase once the clock is held. **Risks, effort, dependencies.** The CLAUDE.md capture
invariant (error 900/906); THREAD_LOCAL capture means pool workers must not touch the stream during
capture (they do not); pinned memory under WSL2 is limited (a few hundred MB is fine). Effort
medium. Depends on F0; F1 for measurability.

### F4 — `MatmulPool` policy for all GPU placements (small; confirmed)

**Where.** `FloatTensor.disableVirtualThreadMatmul`, `MatmulPool`, `LLMEngine.load` (the two
`MatmulPool.enable()` sites), `FloatTensor` (new `isGpuResident()`), `CudaFloatTensor` and
`GpuFloatTensor` (override to `true`), `InferenceEngine.canBatchPrefill`,
`Qwen35InferenceEngine`, `Gemma4InferenceEngine`, `ExpertViews.active`, `MoEFFN.batchAvailable`.

**How.**
1. First measure with no code change: a dense partial offload (Phi-4 14B or Devstral-24B with
   `--gpu-layers` below the block count) and a MoE model with explicit `--gpu-layers 10`, with and
   without `-Dmatmul.pool=force`, protocol P. If the pooled run does not win, stop here.
2. Remove `MatmulPool.disable()` from `disableVirtualThreadMatmul()` (keep the virtual-thread and
   fused-reflection disables, the actual crash surface) and delete the two conditional enables.
3. Replace every `MatmulPool.enabled()` used as a "no GPU tensors" proxy by an explicit
   `anyLayerGpuResident` computed once per engine at construction over the tensors it batches
   (`wq/wk/wv/wqkv/wo/wGate/wUp/wDown`, Qwen3.5's `attnQkv`/`ssm*`, Gemma 4's PLE tensors, MoE
   `ffn_*_exps` and `shexp`), using a base `public boolean isGpuResident() { return false; }`
   overridden in the java21 subclasses. `BailingMoE3InferenceEngine`, `LFM2InferenceEngine` and
   `SimdQ6_KFloatTensor` use `enabled()` only to choose a dispatcher and need no change.
4. Keep `-Dmatmul.pool=false` as the escape hatch; smoke-test `--gpu-backend opencl` once.

**Validate.** Protocol P on the two placements of step 1 and on the MoE-optimized default (must be
unchanged: it already had the pool); `test-architectures.sh --gpu` for regressions; `--auto-tune` on
Qwen3-Coder must now measure its CPU-only candidate at the pooled speed (compare with a standalone
`--no-gpu` run of the same prompt).

**Expected gain.** CPU share × (1 − 1/1.26..1.8): ~20 % measured floor on a MoE (GLM 393 → 312),
15–30 % extrapolated for dense partial offloads at 0.3–0.8 tok/s (`BENCHMARKS.md` partial-offload
rows are pre-pool numbers and will need re-measuring). Removes the CPU-side bias of `--auto-tune`.
**Risks, effort, dependencies.** A GPU tensor reaching the batch kernels silently runs on its CPU
twin (correct, slower) — that is what the residency gate prevents. Effort small. No dependencies.

### F9 — Small-matmul byte threshold and output-projection routing (small; disputed, corrected)

**Where.** `CudaFloatTensor.matmulParallel` / `gpuMatmul` / `cpuTwin` (java21),
`Qwen3MoEInferenceEngine.outputProjection`, `DeepSeek2InferenceEngine.outputProjection`,
`LLMEngine.tryInitExpertGpuCache`, `MoeAttentionCudaPass` (shared expert), `ModelLoader`.

**How.**
1. Threshold at the dispatch point: `if (getWeightsBytes() < MIN_GPU_BYTES && cpuTwin() != null)
   cpuTwin().matmulParallel(...)` with `MIN_GPU_BYTES = Long.getLong("gpu.tensor.min.bytes", 1 << 20)`
   (the session supports 0.6–1.5 MB at full clocks; passes read weights through `getGpuWeights` /
   `Dp4aMatmul` and are unaffected). `FloatTensor` has no `byteSize()`; use `getWeightsBytes()`.
   This yields zero on MiniMax and Qwen3-Coder (no shared experts, all attention on the pass) and
   pays on GLM-4.5 / Llama 4 shared experts and pass-declined layers.
2. Output projection: a runtime decision, not a load-time one. Keep an EMA of the last N GPU output
   timings per tensor and one CPU-twin timing taken during the first decode tokens (after
   `MatmulPool.enable()`, so the twin is measured on the pool it will use — the pre-touch in
   `tryInitExpertGpuCache` runs before it); route to the twin when the CPU is faster by 20 % with
   hysteresis; re-evaluate every 64 tokens. With F1 on, the GPU wins again and the switch must
   follow.
3. Fold the Qwen3MoE-family shared expert into `MoeAttentionCudaPass` as the MLA pass does (detect
   `ffn*Shexp instanceof CudaFloatTensor` per layer, queue gate/up/silu_mul/down after the FFN
   RMSNorm with a second `Dp4aMatmul` instance, read it in `takeSharedExpert`); consumed by
   `Qwen3MoEInferenceEngine.moeFFN`. Upload shexp tensors eagerly in the constructor (they upload
   lazily on the first forward today, after the cache has taken the VRAM, i.e. into shared memory
   under WDDM).
4. Loader: keep `TensorFactory.gpuBufferManager` null for `ffn_gate_inp` (never uploaded, only
   counted) so the fit estimate stops charging 84 MiB (MiniMax) for nothing; never for attention
   tensors, because `layerSupported` requires `CudaFloatTensor`.
5. Replace the scalar accumulate loop in `gpuMatmul` with a bulk copy plus `VectorOps` accumulate
   (estimated 0.2–0.4 ms per token on 150–200k vocabularies; not measured).

**Validate.** Protocol P on MiniMax cache-on (output phase; expect 60 → ~30–34 ms), Qwen3-Coder
cache-on (output phase must not regress: 9–14 ms), and, when available, a GLM-4.5-class model for
step 3. Output text identical to CPU-only (the twin is the same kernel), but note it may differ from
the current GPU-mode output because summation order changes the routing on MoE models.

**Expected gain.** MiniMax −26 to −45 ms per token (2–3.5 %); zero on Qwen3-Coder; shared-expert
models unmeasured (7–9 ms of sync plus whatever dp4a saves over the FP32 kernel).
**Risks, effort, dependencies.** The threshold must not drop below ~0.5 MB (a pool dispatch has its
own cost, estimated at 10–20 µs and not measured). Effort small. Depends on F1.

### F5 — Expert cache redesign (medium; cause claims refuted, engineering retained)

**Why.** Three refuted causes (sections 6.7–6.9) share the same object. Their mechanisms are real;
what was wrong was the size of their contribution. The redesign below is ordered by what the
measurements support.

**Where.** `ExpertGpuCache` (`launchResident`, `finishResident`, `computeExperts`, `upload`,
`uploadExpertSlice`, `victimUnit`, `count`, `getStats`), `GpuExpertCache` (base interface),
`CudaContext` / `CudaBindings` (event query), call sites `Qwen3MoEInferenceEngine.moeFFN`,
`MoEFFN.forward`, `SoftmaxMoe.forward`, `LLMEngine.tryInitExpertGpuCache`.

**How.**
1. Stats first (F0): the three moe timers and `cuEventElapsedTime` around the resident launches,
   printed with `getStats()`. Nothing below is decidable without them.
2. Async pinned promotion, bounded per token: a pinned staging ring of 2–3 units (`allocPinnedHost`),
   decide the promotion in `launchResident` (mask and counts are known there), copy the three slices
   into a free ring slot before `cpuExpertCompute` runs (or as one extra chunk of the pool job),
   issue `cuMemcpyHtoDAsync` on a dedicated copy stream (`createExtraStream`) and record an event; at
   a later `launchResident` poll it with `cuEventQuery` and only then insert the (layer, expert)
   into `keyToUnit` and free the ring slot; the compute stream waits on the event once
   (`streamWaitEvent`) before the unit's first use; remove the victim's key at enqueue. Make
   `hostBuf` pinned so the per-token input upload is a true DMA — and at the same time switch the
   non-hybrid `computeExperts` path from `writeBuffer` to `writeBufferAsync`, because
   `CudaContext.writeBuffer` synchronises the stream after a copy from a pinned segment (the hybrid
   `launchResident` path already uses `writeBufferAsync`). Keep a byte budget on the fill phase
   too, but not one that pushes the fill into decode: today's fill (one unit per layer call, 8 calls
   on MiniMax, 18 on Qwen3-Coder) completes inside the prompt; budget it per token, not per layer
   call, and key the "new token" detector on the position, not on `pLayer <= lastLayer`, which fires
   on every call of the layer-outer prompt loop. Queue the resident-output downloads in
   `launchResident` so `finish()` finds them done.
3. Calibrate-then-pin policy: for the prompt plus the first T decode tokens only count
   (`launchResident` still serves what is resident); then rank (layer, expert) by count and enqueue
   the top `units` through the async uploader; then allow ≤ 1 replacement per token with the existing
   hysteresis. Persist the counts next to the model (`<gguf>.routing`, `java.io`, keyed by file size,
   tensor count and version) so later runs are warm from token 0 — the in-run counts (0–8 per pair
   after 4–9 tokens) cannot rank 6144–7192 pairs, only a cross-run profile can. Replace the
   `Long`-keyed `HashMap`s with `int[]` count / unit-of tables indexed by `layer × E + expert` and a
   lazily maintained minimum for victims. The boxing is hygiene, not throughput: one `Long` per
   (layer, expert) lookup and per count update is 2 × layers × top-k ≈ 770–1,000 per token, plus one
   boxed lookup per unit scanned in `victimUnit`, up to units × promotions ≈ 2,000–3,400 per token
   at four promotions (an estimate from the code, not a measurement). Extend `-Dmoe.routing.stats`
   to `MoEFFN` and `SoftmaxMoe` (it covers MiniMax already, through the Qwen3MoE engine).
4. Speed-aware split with event timing: per layer, once the resident launches are timed, cap the
   number of experts launched to the top-h by count whenever the measured `finishResident` wait
   exceeds ~20 % of the layer's CPU time for 8 consecutive tokens, leaving the rest to the CPU — a
   measured cap, never a switch, because removing GPU work drops the P-state and slows attention
   (Qwen3-Coder m4c/m4d: attention 48 vs 104, output 9 vs 22 with the cache on vs off). Any runtime
   guard that parks the cache must score total token time, not the moe phase alone, and must also
   park the promotion path (`finishResident` keeps uploading otherwise). Do not add a coverage-based
   admission rule: "decline below 15 % coverage" would decline MiniMax, where the cache is a measured
   25 % win at 6.9 % coverage.
5. Multi-expert launch: one `matmul_<type>_multi` kernel per projection taking h weight/output
   pointers (grid = h × blocks), folding the GPT-OSS bias adds into the epilogue, for the types the
   three models use (IQ1_M, IQ2_XXS, Q2_K, Q4_K, Q6_K, MXFP4) with the per-expert path as fallback;
   and a `Dp4aMatmul` overload that takes `(GGMLType, devPtr, ...)` for Q4_K/Q5_K/Q8_0 slots (the
   existing `weightOffset` form addresses a GPU-resident 3D tensor, which the cache does not hold).
   Corrected gain: the per-layer fixed cost is ~5–6 ms per token, under 4 % of the moe phase, so
   this is last.

**Validate.** Protocol P with F1 on, MiniMax and Qwen3-Coder cache-on; the moe timers must show the
`finishResident` wait falling; hit rate ≥ the old policy after 30 tokens; generated text identical at
temperature 0 (routing is deterministic; a promotion published before its DMA lands would show as
wrong output, which the event gating prevents). Pass: moe phase better in every pair, hit rate not
worse, total not worse.

**Expected gain (corrected).** Async promotion ≤ 4 uploads × 2.65–3.1 MB per token = 6–20 ms in
steady state (2–7 % Qwen3-Coder, < 0.5 % MiniMax) plus ~1–2.5 s once per process inside the prompt;
convergence ~15–22 ms per token on Qwen3-Coder (from ~40 % to ~59 % coverage at 13.7 % residency;
the docs' concentration table says 57 % at 12.5 %, 79 % at 25 %) and ~13 points on MiniMax at 6.9 %
— worth roughly 0–15 ms on Qwen3-Coder at today's clocks, because each extra resident expert moves
work to the slower device at P8; the split cap prevents the loss case rather than adding speed.
**Risks, effort, dependencies.** A second CUDA-issuing thread must call
`CudaContext.ensureCurrent()` (a bug there is a hang or error 700); `close()` must sync the copy
stream before freeing; prompt-dependent profiles may not fit another prompt (key them, keep the LFU
running); higher residency without F1 can make the GPU side critical. Effort medium (step 5 large).
Depends on F0, F1, F2.

### F6 — Batched prefill in GPU mode (large; "2–3× slower" refuted, direction holds for Q4_K models)

**Why.** With GPU attention or the cache active, `Qwen3MoEInferenceEngine.forwardPrefill` and
`DeepSeek2InferenceEngine.forwardPrefill` run the layer-outer per-token loop (one `attentionLayer`
round trip per (layer, token), one-token CPU experts, nothing overlaps), and
`Qwen35InferenceEngine` runs its MoE layers token by token even in `ffnBatch`. Measured
(section 3.2, prefill plus warm-up per prompt token): Qwen3-Coder GPU-mode 432–574 ms (m4d/m4c;
788–972 in m3h/m3i) against 216–396 for the CPU-only batched path (m4a/m4e/m3g). The ratio depends
on the pairing because the CPU baseline itself drifts 1.8×: m4d/m4e 1.1×, m4c/m4e 1.45×, m4d/m4a
and m3h/m3g 2.0×, m3i/m3g 2.5×, m4c/m4a 2.7× — all including one-time kernel compiles and uploads
on a 19-token prompt. On MiniMax the GPU-mode prefill (1.39–1.54 s per token, m3b/m3a) is already
faster than the CPU-only batched prefill (1.65–2.14 s, m3e/m1c), because `SimdIQ1_MFloatTensor`
overrides only `dot`, so `matmulRowsBatch` degrades to a per-token loop and the 1.9–2.0×
batched-kernel factor exists only for the K-quants that override it (Q3_K, Q4_K, Q5_K, Q6_K, Q8_0).

**Where.** `GpuAttentionPass` (base interface), `MoeAttentionCudaPass`, `MlaAttentionCudaPass`,
`FlashAttention` (`maxBatchTokens`, `launchBatch`), `batch_ops.cu` (`rmsnorm_batch`,
`rope_apply_batch`, `kv_cache_update_batch`, `add_bias_batch`), `Dp4aMatmul`, `CudaForwardPass.gemm`
(cuBLAS FP16 GEMM to extract), `Qwen3MoEInferenceEngine` (`forwardPrefill`, `prefillBatched`,
`attentionBatch`, `ffnBatch`, `moeBatch`), `DeepSeek2InferenceEngine`, `MoEFFN.forwardBatch`,
`ExpertViews`, `GpuExpertCache` (`noteRouting`), `SoftmaxMoe`, `Qwen35CudaForwardPass`.

**How.**
1. Interface: `default int maxBatchTokens() { return 0; }` and
   `default void attentionLayerBatch(int layer, float[][] x, float[][] xbOut, int basePos, int n)`
   (contract: `x[t] += Attention(attnNorm(x[t]))` at position `basePos + t` over the device KV,
   `xbOut[t] = ffnNorm(x[t])`), plus `takeSharedExpertBatch`.
2. `MoeAttentionCudaPass`: constructor-allocated n-row buffers (allocated before
   `tryInitExpertGpuCache` takes the free VRAM), `FlashAttention` built with `maxBatchTokens = n`, the
   batch kernels above, projections either as a per-token `Dp4aMatmul` loop (bit-identical to decode,
   but it re-reads each layer's weights n times, tens of ms at P8) or through a `GemmF16` helper
   extracted from `CudaForwardPass.gemm` when `CublasBindings.isAvailable()` (rounds activations to
   FP16, as the dense engine already accepts), one `flash.launchBatch` (add a sinks pointer for
   GPT-OSS), Wo accumulate, one download of `[pX | pXb]`. Same skeleton for the MLA pass in the
   latent layout (`rmsnorm_batch` needs a row-stride parameter; per-head FP16 matvecs per token or
   with `grid.z`).
3. Engines: change the gates to allow the batched CPU path when `gpuAttention.maxBatchTokens() > 0`
   (and keep the pool proxy consistent with F4); in `attentionBatch` call `attentionLayerBatch` for
   GPU layers inside the existing `gpuAttentionFailed` try/catch (throw `GpuFailureException` when
   `basePos > 0 || layer > 0`; never write the CPU KV for GPU layers); `ffnBatch` skips its RMSNorm
   when the GPU wrote `xn`; DeepSeek2 threads `sharedOut` into `MoEFFN.forwardBatch`; shared experts
   on Qwen3MoE-family models need the F9 fold, or they fall to the twin loop.
4. Experts stay on the CPU through `ExpertViews.forEachExpert` / `matmulRowsBatch`. Add
   `default void noteRouting(int layer, int[] experts, int count)` to `GpuExpertCache`, counted once
   per chunk, so the LFU is warm at decode; this seeds counts, not contents — with the batched path
   the cache starts decode empty and fills at ≤ 1 upload per layer call, so combine it with F5's
   ranked async fill.
5. Qwen3.5: step 1 `SoftmaxMoe.forwardBatch` (route n, `groupByExpert`, `forEachExpert`) used by
   `ffnBatch` (a CPU-only win too); step 2 `LayerGpuForwardPass.prefillLayersBatch` with batched
   DeltaNet projections and per-token recurrence launches in position order on one stream.
6. Do not gate on a minimum prompt length: every token saved is a saved sync.

**Validate.** Protocol P with the 512-token prompt on Qwen3-Coder (cache on) and GLM-4.7-Flash;
metric = `timeMs − N × total_N` divided by prompt tokens; generated text identical to the per-token
GPU path with the dp4a option, and within the documented FP16 divergence with the GEMM option;
decode unchanged.

**Expected gain.** Qwen3-Coder time-to-first-token roughly halved (GPU-mode prefill to about the CPU
batched level of 216–396 ms per prompt token, minus the attention half at 27–49 ms); GLM-4.7-Flash
similar in direction, unmeasured; MiniMax at most −12 % (the attention half is 6.7 s of a 56 s
prompt; the expert half stays CPU-bound and IQ1_M has no batched kernel). The "70–75 ms per token"
and "TTFT 55–61 s → 15–20 s" targets of the original finding are withdrawn.
**Risks, effort, dependencies.** The pass must remain the only KV writer for GPU layers; the hybrid
cache must not be called with per-token semantics from the batched loop; cuBLAS is optional and
rounds to FP16. Effort large. Depends on F3 (batch kernels and pinned blocks), F5 (ranked fill).

### F7 — VRAM planner, allocation verification, canary (medium; disputed, plan-side only)

**Where.** `LLMEngine.load` (placement branch), `sumMoELayerNonExpertBytes`,
`estimateNonLayerBytes`, `tryInitExpertGpuCache`, `buildHardwarePlan`, `ModelLoader` (router
placement), `ArchitectureRegistry`, `CudaContext` (`allocBuffer`, `getMemoryInfo`),
`CudaBufferManager`, `ExpertGpuCache` (buffers and kernels in the constructor),
`MoeAttentionCudaPass` / `MlaAttentionCudaPass` constructors, `LLMPlayerMetrics`.

**How**, in this order.
1. F2 first — it removes the overflow with no planner at all.
2. Plan side: add the output projection (`estimateNonLayerBytes`) to the MoE branch; in
   `sumMoELayerNonExpertBytes` count the Q-LoRA and split-KV tensors through the existing
   `ArchitectureRegistry.attnQA` / `attnQB` / `attnKB` / `attnVB` helpers next to `attnQ` /
   `attnKvB` (a name the model does not carry contributes 0 through `tensorByteSize`, so adding
   the helpers is safe on both layouts); count `wk_b` / `wv_b` at FP16 size when the MLA pass is
   latent; stop counting `ffn_gate_inp` and create it with the GPU manager off (F9 step 4); replace
   the flat `usableVram = 0.80 × total` by `min(total, free + measured WSL2 offset) − sum(rounded
   to 2 MiB) − reserve(lazy consumers)`, with the same figure feeding the cache; print the plan in
   the Hardware Plan block, from one `PlacementPlanner` used by both `LLMEngine.load` and
   `buildHardwarePlan`. The placement runs before any tensor exists, so the plan stays a
   quick-parse estimate plus the missing terms; `cuMemGetInfo` after a proper warm-up remains the
   source of truth for the cache.
3. Eager warm-up before sizing: an engine `warmGpu()` that runs one dummy `matmulParallel` on every
   `CudaFloatTensor` the CPU path calls (shared experts, dense-layer FFN, output, biases); the
   `ExpertGpuCache` constructor allocating its input, per-expert and staging buffers and compiling
   its kernels; the passes pre-compiling the flash variant. Only then read `getMemoryInfo`, and keep
   the 512 MiB reserve until the WDDM accounting is measured (the under-report is what saved
   MiniMax).
4. Verification: `allocBufferChecked(bytes, tag, reserve)` for allocations ≥ 16 MiB — free before and
   after, `cuMemFree` plus `VramExhaustedException` when `free_after < reserve`; `CudaFloatTensor.getGpuWeights`
   catches it and sets `gpuFailed` (the tensor becomes its twin); the cache drops a chunk; the pass
   constructors catch it per layer and set `onGpu[i] = false` (a Throwable from the constructor drops
   the whole pass today). Whether the driver "charges" VRAM for a shared-memory allocation is the
   open question — measure `cuMemGetInfo` at 0/1/2/4/5 GiB fill on this box before relying on it.
5. Canary: a 32–64 MiB reference buffer allocated first and a bandwidth probe kernel; compare the
   output weights and one cache chunk against it under the same clock; a ratio below 1/3 shrinks the
   cache by 25 % and re-probes. It is a load-time diagnostic that shrinks the cache, never a
   per-tensor demotion (pass-resident weights cannot be demoted without rebuilding the pass), and it
   must never run inside a capture or a generation. Expose in the plan and in `LLMPlayerMetrics`.

**Validate.** On Qwen3-Coder the plan's predicted VRAM must match nvidia-smi within 5 %; a
deliberately oversized `-Dmoe.expert.gpu.cache.mb` (the F2 cap; not `moe.expert.cache.mb`, which is
the SSD cache) must be shrunk with a log line and the output phase must stay at 9–14 ms (cache on);
`test-architectures.sh --gpu` unchanged.

**Expected gain.** None directly; it converts the 55 ms output-projection trap and evicted attention
layers from per-model surprises into impossibilities. **Risks, effort, dependencies.** Two sources
of truth during the transition (keep the dense branch on the planner too); WDDM can evict a verified
allocation later. Effort medium. Depends on F2.

### F8 — In-process placement calibration replacing the reload-based `--auto-tune` (medium; design retained with corrections)

**Why.** `LLMEngine.load` picks MoE-optimized iff `nonExpertBytes ≤ 0.80 × VRAM`, `buildHardwarePlan`
duplicates the rule, and nothing measures benefit; `CLIRunner.autoTune` reloads the model per
candidate (MiniMax preload 56–229 s), compares only the heuristic GPU placement against CPU-only
over 24 tokens after 6, ties to the GPU, and runs its CPU candidate on the disabled pool (C5). The
review refuted the claim that this ships a losing default on the models measured (with the
artifacts removed, the default wins or ties everywhere), so the calibrator's job is to stop
regressions on models nobody has measured, and to make the keeper, the cache and the thread count
measurable in place.

**Where.** A new base-code `PlacementCalibrator` (api package, Java 8); `Qwen3MoEInferenceEngine`,
`DeepSeek2InferenceEngine`, `Qwen35InferenceEngine`, `InferenceEngine` (park switches);
`CudaFloatTensor` (a `forceCpuMatmul` volatile and `matmulRows` / `matmulRowsBatch` delegation to the
twin); `MatmulPool` (`setActiveThreads`, `reconfigure`); `CLIRunner.autoTune`; `LLMPlayerMetrics`.

**How.**
1. Park switches: `setGpuAttentionEnabled` / `setExpertGpuCacheEnabled` that park, not close, the pass
   and the cache; `hasGpuForwardPass()` reports the effective state so `LLMEngine.gpuHoldsHistoryOf`
   stays correct; reset `lastGpuResidentState` after calibration; the prefill gates must read the
   effective state. Cover Qwen3.5 (`disableGpuForwardPass` is one-way today) and the dense engine
   (`gpuForwardPass`, `gpuKvOwner`).
2. A true CPU-attention candidate: with the pass parked, `gqaAttention` calls `matmulParallel` on
   `CudaFloatTensor` weights, i.e. four per-tensor GPU round trips per layer, which is the `!kvFits`
   mode, not CPU-only. Add a base volatile checked before `gpuMatmul` in `matmulParallel`, and
   delegate `matmulRows` / `matmulRowsBatch` to the twin (they fall to the default per-row loop
   today, losing the SIMD batch kernels). The same switch covers the output projection (F9).
3. Procedure: fixed 8-token list, own small state (`createState(64)`, not the full KV: 992 MB on
   MiniMax); per configuration discard 1 warm token, then 3 rounds × 6 tokens from position 0, ABAB
   interleaved; score = best-of-N ms per token; stage A {attention GPU on/off} with the cache off,
   stage B {cache on/off} with the winner, extended sweep {thread count, `cuda.kv.fp16`, keeper};
   change a default only when the alternative wins by > 8 %; log both numbers and the P-state line.
   Cost: 38 passes per stage, 100–145 s on MiniMax for A + B — accept it or persist.
4. Persist verdicts in `~/.cache/llmplayer/placement/<size+name+gpu+ctx+version>.properties`;
   `-Dplacement.calibrate=false|force`.
5. Cache warm-up bias: stage B measures a cold cache in 6-token rounds; run stage B after the
   prompt-sized fill, or seed it from the F5 profile.
6. Thread count: `MatmulPool` reads its size once; add `setActiveThreads(n)` (workers with index
   ≥ n − 1 skip claiming) for the sweep; `--threads N` pins it. The ForkJoin fallback's parallelism
   cannot be changed after first use.
7. Prefill awareness: also time one 32-token batched CPU prefill against the per-token GPU prefill
   and print both, with `--workload decode|prompt|balanced` until F6 lands.
8. Add an analytic `kvTerm(ctx)` to the GPU-attention score, or calibrate at position ≥ 256 when
   the model does > 5 tok/s (a position-0 calibration under-weights attention).
9. Expose `getPlacementReport()` on `LLMPlayerMetrics` / `GET /api/metrics`; `--auto-tune` keeps
   its name and calls the calibrator.

**Validate.** The calibrator must reproduce the section 3 ordering on MiniMax (cache on > cache off
≈ CPU) and Qwen3-Coder (cache on ≈ CPU > cache off) in three consecutive process starts; its verdict
must never flip between runs on the same box when the two candidates differ by less than 8 %.

**Expected gain.** None directly; a guard. **Risks, effort, dependencies.** Load time on first use;
a parked pass keeps its KV VRAM; persisted verdicts go stale after kernel changes (key them by
version). Effort medium. Depends on F0, F4, F9.

### F10 — Qwen3.5-MoE graph mode (medium; part of C4)

**Where.** `Qwen35CudaForwardPass` (`graphAvailable`, `forwardFFN`, `forwardMoeFFN`, the graph
capture paths), `Qwen35InferenceEngine` (per-layer driver).

**How.** Split `forwardFFN` into a pre-callback half (norm + shared expert) and a post-callback half
(`writeBufferAsync` + axpy); capture per layer up to the `readBuffer` of `forwardMoeFFN` as A_l and
the post-callback tail as the head of A_(l+1); store execs per layer; separate the output projection
now inside the decode graph; drop the `graphAvailable = !hasMoe` predicate; the engine already falls
back to `forwardLayer` per layer, so the per-layer graphs can live there with no interface change.
The `readBuffer` sync must stay outside every capture.

**Validate.** Protocol P on Qwen3.5-35B-A3B (no session run exists; establish the baseline first),
DeltaNet and attention layers, output identical to the per-launch path.

**Expected gain.** An estimate only: 40 layers × 30–40 launches × 8.2 µs = 10–13 ms of ~178 ms per
token plus the sync savings. The launch count is read from the code (section 4.4), not measured, and
is probably lower for DeltaNet layers, so treat the figure as an upper bound.
**Risks, effort, dependencies.** The capture invariants; the MoE host callback and the hybrid
cache's own `finish()` inside it. Effort medium. Depends on F3.

### F11 — CPU expert chunk compaction (small; disputed, corrected)

**Where.** `Qwen3MoEInferenceEngine.cpuExpertCompute` (both `forRange` dispatches),
`MoEFFN.routedExperts`, `SoftmaxMoe.routed`, the three state classes (`Qwen3MoEState`,
`DeepSeek2State`, `SoftmaxMoe.Scratch`).

**How.** First the zero-code test — `-Dmatmul.chunks.per.thread=16` (176 chunks of 70 rows on
MiniMax; the 3-resident case becomes 109.7 chunks / 11 = 9.97 rounds, +0.3 %); if that A/B shows
nothing, compaction will not either. Otherwise: a per-state `int[] cpuSlots` sized like
`selectedExperts`, filled with the non-resident slot indices (bits `s < k` only — `MoEFFN`'s
failure recovery re-dispatches with `~gpuMask`), then `forRange(nCpu × efd, EXPERT_ROW_CHUNK, ...)`
with `slot = cpuSlots[u / efd]`, same for the down projection; zero-fill the `e < 0` slots outside
the parallel loop (their weight is 0, but stale NaN × 0 is NaN). Bit-identical by construction (each
row is one independent dot).

**Validate.** Protocol P on MiniMax cache-on, moe phase, plus the F0 residency print; CPU-only runs
exercise nothing (mask 0).

**Expected gain.** 0–8 % of the moe phase at the measured residency (20–80 ms per token MiniMax,
5–15 Qwen3-Coder, 0 GPT-OSS). **Risks, effort, dependencies.** None. Effort small. No dependencies.

### F12 — `MatmulPool` worker hold across the GPU phase (small; disputed, corrected)

**Where.** `MatmulPool` (worker loop, `parallelFor`), `Qwen3MoEInferenceEngine.forwardLayer` /
`DeepSeek2InferenceEngine.forwardLayer` (around `gpu.attentionLayer`).

**How.** First the zero-code A/B — `-Dmatmul.spin=2000000` against `20000` and `2000` on
Qwen3-Coder cache-off and MiniMax cache-off, protocol P (the `_spin` files of this session are keeper
runs; this A/B has never been run). If it wins: prefer a timed pre-wake to spin-forever — the engine
calls `MatmulPool.setHold(nanos)` with the last measured attention duration before
`gpu.attentionLayer`, workers `parkNanos(hold − margin)` and then resume spinning, keeping the cores
in C6 for most of the 2–4 ms GPU window; `setHot(false)` in a `finally`, because `gpuAttentionFailed`
throws out of the token at any position > 0. Idempotent or depth-counted (two engines share the
singleton).

**Validate.** Protocol P, moe phase and total; stop if the gain is below the 7–13 ms per token
ceiling the m4 batch establishes.

**Expected gain.** Ceiling 7–13 ms per token on Qwen3-Coder (section 4.9); unknown with the cache
on, where the attention phase sits at the edge of the spin window. **Risks, effort, dependencies.**
Spinning workers draw power during a phase that is 40–45 % of the token; validate on totals. Effort
small. No dependencies.

---

## 6. Investigated and rejected

Each item records what was claimed, what refuted it, and where the surviving pieces went. Do not
re-propose these without new measurements that address the stated reason.

### 6.1 More pool threads (16 or 22 instead of 11) — rejected

Claim: WSL2 fabricates an 11 × 2 topology for a 16-core hybrid CPU, `CLIOptions.detectPhysicalCores`
returns 11, and the compute-bound IQ1_M expert kernel would be 26–34 % faster at 16–22 threads
(`m5.txt`: 12.94 / 13.81 Gelem/s against ~10.3 interpolated at 11), worth −250 to −420 ms per token
on MiniMax.

Rejected because the direct end-to-end A/B exists and points the other way: MiniMax CPU-only at 16
threads (m3e) is +9.6 % slower on the expert phase and +16 % per token than at 11 (m1c) at 50 tokens,
and −1.8 % / −1.2 % at the first window; Qwen3-Coder 16 threads (m4b) is +7.5 % slower than 11
(m4a); the one clean dense A/B on this box (`autotuning-heuristics.md`, the `--threads` caveat:
physical-11 ≈ 5.0 vs logical-22 ≈ 3.8 tok/s) lost ~24 % at 22 threads. The micro-benchmark does not
transfer: `MtDot` partitions rows statically over a fixed pool with no spinning workers and no
per-layer barriers, whereas the pool spins 20,000 iterations between phases on SMT siblings and
interleaves 8 experts × 3 projections × 62 layers. The topology misdetection is real; the only
defensible change is to print the caveat, and to make the thread count a calibration axis (F8). The
physical-core default stays.

### 6.2 C1-compiled decode kernels as the cause of the GPU-active expert penalty — rejected

Claim: GPU-mode prefill skipped `warmDecodeKernels()`, so the whole decode ran the routed-expert dot
kernels on C1 code, explaining the 10–25 % slower CPU expert phase with the GPU active (the comment
in the tree: "3111 ms/token in GPU mode against 1414").

Rejected because the CLAUDE.md invariant is about batched prefill, which never runs the single-token
kernels; the GPU-mode per-token prefill runs the very `dot` decode uses — 29 prompt tokens × 62
layers × 8 experts × 6144 rows per expert (1536 gate + 1536 up + 3072 down) ≈ 88 M invocations
before the first decoded token — so C2 code is in place during the prompt. The m1a windows are flat,
not decaying; m3a (with the call) vs m1a (without) moved the moe phase from 1488–1546 to 1411–1425,
5–8 %, inside the run-to-run band of the two CPU-only runs (m1c vs m3e, 6–10 % at the same windows);
the "3111 vs 1414" comparison was the profiler artifact. The added calls in `Qwen3MoEInferenceEngine`,
`DeepSeek2InferenceEngine` and `Qwen35InferenceEngine.forwardGpu` are harmless and kept as a start-up
nicety (F0 rewrites the comment); the residual GPU-active penalty (3–10 % on Qwen3-Coder, m4d vs
m4a) is covered by the m5 GPU-busy loss and C9.

### 6.3 RAM pressure, paging or disk reads — rejected

`mem_samples.txt` during m1b/m1c and the `_smi.txt` vmstat columns of every m3 GPU run show
si = so = 0, bi ≈ 0, MemAvailable ≥ 27.5 GB and the page cache stable at ~27.3 GB; MiniMax's 22.5 GB
preload completed before generation in every run. The single run with disk reads (m3f, Qwen3-Coder
CPU-only right after the MiniMax preload) is excluded from every comparison. Promotion uploads
read experts that `cpuExpertCompute` has just touched, so the "9p read on the main thread" risk
does not occur.

### 6.4 CPU/GPU power coupling as measured — rejected as measured

Claim: GPU activity slows the CPU experts by 3–18 % (`m5.txt`) and the spin keeper by 15–50 % (m2a
vs m1b), so any keeper must be netted against a coupling term of 0.03–0.18 × t_moe.

Rejected because neither measurement isolates a coupling: the m5 "GPU busy" load was a loop of 40
fresh JVMs (`measure.sh`), each parsing the 22.5 GB GGUF, uploading `output.weight` and spinning in
`cuStreamSynchronize` — a second CPU-heavy process whose preemption makes one `MtDot` thread the
straggler under its static partition (the 1-thread point got faster, +5.7 %, contradicting a
shared-envelope mechanism); m2a's keeper was same-priority, started at context creation, contended
with the resident-expert kernels, and is one unpaired run against a control that drifts +33 % by
itself. The interleaved m5 batch shows the opposite sign (keeper runs faster on every phase including
moe). What survives: within-run drift with the GPU idle (m1c, m3e) is unexplained; per-phase deltas
must be read end to end; the CPU clock must be read from the Windows side (F0, F1). No coupling term
enters any cost model until it is measured in-process with the keeper at known duty.

### 6.5 VRAM oversubscription as the MiniMax cause — rejected

MiniMax peaks at 2348 MiB (cache off) and 5320 MiB (cache on) of 6140 with 820 MiB free through the
whole decode, its output projection is 71–75 ms per token with nothing spilled, and its GPU-vs-CPU
gap sits in the CPU expert phase and the P8 clock. Oversubscription is real on Qwen3-Coder (section
3.5) and cost 55 ms per token on the output projection until the upload was reordered; the residual
spill is ≤ 6–9 ms per token. Handled by F2 and F7.

### 6.6 Heavy and memory-fill clock keepers — rejected

Same-priority 160×256 spin grid at 2000/0 µs, started at context creation: MiniMax never finished
loading within batch 3's 1500 s timeout (m3c); Qwen3-Coder loaded in 389 s instead of 67 and
decoded at 18–24 s per token (m3j) while held at P4/P3 — a 40,960-thread grid exceeds the 20 SMs'
resident capacity and a low-priority stream cannot yield mid-block, so the model's expert kernels
queued behind it. Memory-fill keepers (64 MB, 2000/0: m3d/m3k never loaded within 1500 s at
22–26 W; 16 MB, 1000/1000, lowest priority: m5d drifted from 223 to 1583 ms per token, m5f made
MiniMax 4–5× slower) lift the memory clock to 7001–8001 MHz but compete for the bandwidth the
resident experts and the output projection need: utilisation is not the governor's signal (m2a held
18–54 % at P8), memory traffic or power is, and that is exactly what must not be consumed. Only the
light (4×64, 1000/1000) spin variant on a genuinely lower-priority stream is worth productizing
(F1). `nvidia-smi -lgc` returned "Unknown Error" under WSL2 (task brief only; not captured in a
file), so locking the clocks from inside the VM is not an option; from the Windows host it remains
the decisive but unrun experiment.

### 6.7 Fit-only placement as the reason a losing configuration ships — rejected as a cause, design kept (F8)

Claim: the fit-only placement rule and the reload-based `--auto-tune` ship a losing configuration.
Refuted because, with the artifacts removed, the shipped default is the fastest MiniMax configuration
and ties CPU-only on Qwen3-Coder inside the 8 % hysteresis the finding itself proposed, and every
measured "losing" configuration was opt-in (`-Dmoe.expert.gpu=false`, `-Dcuda.clockkeeper`); the
calibrator survives as a guard for unmeasured models (F8).

### 6.8 Token-by-token prefill "2–3× slower than the CPU batched prefill" — refuted as stated, direction kept (F6)

Claim: GPU-mode per-token prefill is 2–3× slower than the CPU batched prefill on every MoE model.
Refuted because on MiniMax it is 0.7–0.9× the CPU batched prefill (IQ1_M has no batched kernel),
on Qwen3-Coder the ratio is 1.1–2.7× depending on the pairing (section F6), and the CPU batched
path costs 216–396 ms per prompt token, not the 90–100 the finding assumed; the direction holds for
K-quant models and is engineered in F6.

### 6.9 Synchronous pageable promotion, the hybrid split and the non-converging LFU as causes — refuted, engineering kept (F5)

Claim: cache promotion, the hybrid split and the LFU's failure to converge each cost a large share of
the token. Refuted because the fill phase completes inside the prompt and the profiler now excludes
it, steady-state promotion is ≤ 4 uploads per token (6–20 ms), the "a cache hit is itself a loss
(169 vs 132)" figures appear in no measurement file, and hit rates printed from m3b on (17–28 %
MiniMax, 37–54 % Qwen3-Coder) are ~3/4 of the converged ceiling, so the headroom is ~1.5×, not
3–4×; the mechanisms are engineered in F5.

---

## 7. Appendix

Two properties used below have no row in `jvm-flags.md` yet (F0 step 7): `-Dmoe.expert.gpu=false`
disables the hybrid expert GPU cache (read in `LLMEngine.tryInitExpertGpuCache`), and
`-Dmoe.cpu.pool=false` keeps `MatmulPool` off for the MoE-optimized placement (read in
`LLMEngine.load`).

### 7.1 Reproducing every measurement

Common prefix (JDK 25 at `/usr/lib/jvm/jdk-25.0.2+10`, run from the repository root). The GGUF
files the runs and the fix validations need are all present in `gguf/` except GPT-OSS-20B, which
has no file there at the time of writing:

```bash
export JAVA_HOME=/usr/lib/jvm/jdk-25.0.2+10 && export PATH=$JAVA_HOME/bin:$PATH
mvn -q clean compile
J=$JAVA_HOME/bin/java
JF="--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED --enable-preview"
MM=gguf/MiniMax-M2-THRIFT-55.i1-IQ1_M.gguf
QC=gguf/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
GLM=gguf/GLM-4.7-Flash-Q4_K_M.gguf                      # F3, F4, F6 validation
DS2=gguf/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf     # F5 validation
Q35=gguf/Qwen3.5-35B-A3B-Q4_K_M.gguf                     # F10 validation
P1="What is the capital of France?"
P2="Write a Java method that reverses a linked list."
T=1500   # batches 1-3 used 1500 s; batches 4-5 used 900 s
run() { # name model prompt maxtok "jvm -D flags" extra-cli...   (historical 2 s sampler; protocol P uses section 2.6 step 4)
  local name=$1 model=$2 prompt=$3 maxtok=$4 jopts=$5; shift 5
  ( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=memory.used,clocks.sm,clocks.mem,pstate,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader) $(uptime | sed 's/.*load average/load/') $(vmstat 1 2 | tail -1 | awk '{print "si="$7" so="$8" bi="$9" us="$13" id="$15}')"; sleep 2; done ) > ${name}_smi.txt 2>&1 &
  local SP=$!
  timeout $T $J $JF -Xmx8g -Dcpu.profile=true $jopts -cp target/classes it.denzosoft.llmplayer.LLMPlayer \
    --model $model --prompt "$prompt" --max-tokens $maxtok --temperature 0 --force "$@" > ${name}.txt 2>&1
  kill $SP 2>/dev/null
}
```

The runs, in the order they were made (`measure.sh` … `measure5.sh`). The sampler columns differ
per batch: batch 1 appended only the nominal `cpuMHz`, batches 2–3 `free -m` and `vmstat`, batches
4–5 `uptime`; `temperature.gpu` exists only from batch 5.

```bash
# Batch 1 (13:49-14:03, pre-fix profiler, timeout 1500)
$J $JF -cp target/classes:t RoundTrip                                   # m4.txt (t/RoundTrip.java, not in the repo)
$J $JF -cp target/classes:t MtDot $MM blk.1.ffn_gate_exps.weight        # m5.txt idle half (t/MtDot.java)
( for i in $(seq 1 40); do $J $JF -Xmx4g -cp target/classes:t GpuBench $MM output.weight; done ) &  # m5 "GPU busy" load
run m1a_gpu_nocache $MM "$P1" 60 "-Dmoe.expert.gpu=false"
run m1b_gpu_default $MM "$P1" 60 ""
run m1c_cpu         $MM "$P1" 60 "" --no-gpu
# Batch 2 (14:15-14:23, pre-fix, keeper started at context creation, same priority, timeout 1500)
run m2a_mm_gpu_clockkeeper         $MM "$P1" 60 "-Dcuda.clockkeeper=true"
run m2b_qc_gpu_nocache             $QC "$P1" 60 "-Dmoe.expert.gpu=false"
run m2c_qc_gpu_nocache_clockkeeper $QC "$P1" 60 "-Dmoe.expert.gpu=false -Dcuda.clockkeeper=true"
run m2d_qc_gpu_default_clockkeeper $QC "$P1" 60 "-Dcuda.clockkeeper=true"
# Batch 3 (15:03-17:09, profiler fixed + warmDecodeKernels in the per-token prefill, timeout 1500)
run m3a_mm_gpu_nocache_warm $MM "$P1" 60 "-Dmoe.expert.gpu=false"
run m3b_mm_gpu_default_warm $MM "$P1" 60 ""
run m3c_mm_gpu_default_warm_spin $MM "$P1" 60 "-Dcuda.clockkeeper=2000 -Dcuda.clockkeeper.sleep=0 -Dcuda.clockkeeper.blocks=160 -Dcuda.clockkeeper.threads=256"
run m3d_mm_gpu_default_warm_mem  $MM "$P1" 60 "-Dcuda.clockkeeper=2000 -Dcuda.clockkeeper.sleep=0 -Dcuda.clockkeeper.mb=64"
run m3e_mm_cpu_t16 $MM "$P1" 60 "" --no-gpu --threads 16
run m3f_qc_cpu     $QC "$P2" 100 "" --no-gpu
run m3g_qc_cpu_t16 $QC "$P2" 100 "" --no-gpu --threads 16
run m3h_qc_gpu_default_warm $QC "$P2" 100 ""
run m3i_qc_gpu_nocache_warm $QC "$P2" 100 "-Dmoe.expert.gpu=false"
run m3j_qc_gpu_default_warm_spin $QC "$P2" 100 "-Dcuda.clockkeeper=2000 -Dcuda.clockkeeper.sleep=0 -Dcuda.clockkeeper.blocks=160 -Dcuda.clockkeeper.threads=256"
run m3k_qc_gpu_default_warm_mem  $QC "$P2" 100 "-Dcuda.clockkeeper=2000 -Dcuda.clockkeeper.sleep=0 -Dcuda.clockkeeper.mb=64"
# Batch 4 (19:11-19:19, clean Qwen3-Coder comparison, back to back, timeout 900)
T=900
run m4a_qc_cpu              $QC "$P2" 100 "" --no-gpu
run m4b_qc_cpu_t16          $QC "$P2" 100 "" --no-gpu --threads 16
run m4c_qc_gpu_default_warm $QC "$P2" 100 ""
run m4d_qc_gpu_nocache_warm $QC "$P2" 100 "-Dmoe.expert.gpu=false"
run m4e_qc_cpu_again        $QC "$P2" 100 "" --no-gpu
# Batch 5 (19:22-19:46, keeper on the side stream created with the "least" priority, started after load, timeout 900)
K="-Dcuda.clockkeeper=1000 -Dcuda.clockkeeper.sleep=1000 -Dcuda.clockkeeper.blocks=4 -Dcuda.clockkeeper.threads=64"
KM="-Dcuda.clockkeeper=1000 -Dcuda.clockkeeper.sleep=1000 -Dcuda.clockkeeper.mb=16"
run m5a_qc_gpu_base         $QC "$P2" 100 ""
run m5b_qc_gpu_keeper_spin  $QC "$P2" 100 "$K"
run m5c_qc_gpu_base2        $QC "$P2" 100 ""
run m5d_qc_gpu_keeper_mem   $QC "$P2" 100 "$KM"
run m5e_qc_gpu_keeper_spin2 $QC "$P2" 100 "$K"
run m5f_mm_gpu_keeper_mem   $MM "$P1" 60  "$KM"
```

`RoundTrip`, `MtDot` and `GpuBench` were throw-away scratchpad classes (`t/`), not in the repository,
a few dozen lines each: `RoundTrip` times one pinned 8 KB upload, ten launches of a one-block
`rmsnorm_fused` kernel, a 16 KB download and a sync, then one launch alone, then one idle download +
sync; `MtDot` runs `SimdIQ1_MFloatTensor.dot` over every row of one expert tensor on a fixed thread
pool with a static row partition, three passes, reporting the third; `GpuBench` uploads
`output.weight` and runs 80 matmuls.

Window re-derivation from a run file (the profiler prints cumulative averages; this prints the
per-10-token windows of section 3 and works with or without the cache suffix on the line):

```bash
python3 - m4c_qc_gpu_default_warm.txt <<'EOF'
import re, sys
pat = re.compile(r'\[cpu-profile \w+\] (\d+) tokens.*?attn\((?:GQA|MLA)\)=([\d.]+).*?moe_ffn=([\d.]+).*?output=([\d.]+) \| total=([\d.]+)')
pn = pa = pm = po = pt = 0
for line in open(sys.argv[1], errors='replace'):
    m = pat.search(line)
    if not m: continue
    n = int(m.group(1)); a, mo, o, t = (float(m.group(i)) * n for i in (2, 3, 4, 5))
    d = n - pn
    print(f"window {pn}-{n}: attn={(a-pa)/d:.1f} moe={(mo-pm)/d:.1f} output={(o-po)/d:.1f} total={(t-pt)/d:.1f} ms/token")
    pn, pa, pm, po, pt = n, a, mo, o, t
EOF
```

Generation-window P-state histogram from an `_smi.txt` file (section 2.3): keep the rows whose
`memory.used` equals the trace maximum and whose `utilization.gpu` is above 0, then count the
`pstate` column.

### 7.2 Measurement files

All under the investigation scratchpad's `an/` directory (not in the repository):

| File | Contents |
|---|---|
| `m1a_gpu_nocache.txt`, `m1b_gpu_default.txt`, `m1c_cpu.txt` | MiniMax batch 1 run logs (pre-fix profiler); cumulative `[cpu-profile Qwen3MoE]` lines every 10 tokens, `--- Stats ---` |
| `m2a_mm_gpu_clockkeeper.txt`, `m2b..m2d_qc_*.txt` | batch 2 (same-priority 500/500 keeper); m2b–m2d have no profile line (7 tokens) |
| `m3a..m3e_mm_*.txt`, `m3f..m3k_qc_*.txt` | batch 3 (profiler fixed); m3c/m3d/m3k end during preload; hit rates printed from m3b on |
| `m4a..m4e_qc_*.txt` | batch 4, the cleanest Qwen3-Coder set |
| `m5a..m5e_qc_*.txt`, `m5f_mm_gpu_keeper_mem.txt` | batch 5, interleaved keeper A/B |
| `*_smi.txt` | nvidia-smi every 2 s: `memory.used, clocks.sm, clocks.mem, pstate, utilization.gpu, power.draw[, temperature.gpu]`, plus `free`/`vmstat` (batches 2–3) or `uptime` (4–5), plus the nominal cpuMHz (batch 1) |
| `m4.txt` | per-layer host round-trip floor (three repetitions) |
| `m5.txt` | IQ1_M expert dot throughput at 1/4/8/16/22 threads, GPU idle vs a second JVM busy |
| `mem_samples.txt` | `free -m` + `vmstat` every 5 s across m1b and m1c |
| `measure.sh` … `measure5.sh` | the batch scripts reproduced above |
| `phase1_findings.json`, `findings_flat.txt`, `phase2_causes.json` | the seven-lens findings and the 17 canonical causes with three verdicts each, from which sections 4–6 were written |
| `../clk.txt` | 200 ms nvidia-smi trace of an earlier Qwen3-Coder GPU run (908 samples: 632 P8, 104 P0, 66 P4, 58 P3, 48 P5) |
| `../vr.txt`, `../smoke_out.txt`, `../o.txt` | earlier runs: the 3059-slot Qwen3-Coder cache with the spilled 55 ms output projection; DS-Coder-V2-Lite 1103 slots; an early GLM-4.7-Flash run without the cache |

### 7.3 Working-tree changes made during the investigation (uncommitted)

These are present in the tree the fixes will be built on and must be kept (or finished) rather than
re-invented. The same working tree also carries a large set of uncommitted v1.18 changes unrelated
to the investigation (`git status` lists roughly 110 modified source files and 170 entries in all),
so protocol P's `git diff --stat` will not isolate the investigation's edits; commit or stash before
starting the fixes.

- `Qwen3MoEInferenceEngine`: `profPrefillDropped` in `outputProjection` (the first projection zeroes
  the phase sums, so profile averages are per decoded token); `printProfile` appends
  `getExpertCacheStats()` (hits, misses, hit rate) to every line; `warmDecodeKernels()` is called at
  the top of the per-token (GPU-mode) branch of `forwardPrefill`, with a comment that overstates its
  effect (F0 rewrites it); the `expertGpuCache` field is typed as the base interface `GpuExpertCache`
  instead of `Object` plus reflection.
- `DeepSeek2InferenceEngine`: the same `profPrefillDropped`; `warmDecodeKernels()` in the per-token
  prefill branch; `tryInitGpuAttention` / `MlaAttentionCudaPass` wiring, `initExpertGpuCache`,
  `takeSharedExpert` consumption.
- `Qwen35InferenceEngine`: `if (moe != null) warmDecodeKernels();` inside `forwardGpu` (runs its
  idempotent lookups on every GPU token; F0 gates it); `SoftmaxMoe` MoE path and `initExpertGpuCache`.
- `CudaBindings`: `cuStreamCreateWithPriority` and `cuCtxGetStreamPriorityRange` bindings (plus,
  from the same day's GPU work, `cuMemcpyDtoHAsync`, `cuMemsetD32Async`, `cuFuncSetAttribute`, NVRTC
  cubin and version calls).
- `CudaContext.maybeStartClockKeeper`: the experimental keeper — `-Dcuda.clockkeeper=<busy µs>`
  (`true` = 500), `.sleep` (default 500), `.blocks` / `.threads` (default 32 × 32), `.mb` (memory-fill
  variant); a daemon thread that calls `ensureCurrent()`, creates a side stream with the `least`
  priority from `cuCtxGetStreamPriorityRange` (which equals the work stream's default priority — see
  F1 step 1), calibrates `iters` from twenty launches, then loops launch → `cuStreamSynchronize` →
  `parkNanos`; `kernels/cuda/spin.cu` (`gpu_spin`, an FMA loop) and the existing `fill_zero.cu`.
  The runs of this day printed a pointer word as the iteration count; the current source prints the
  `iters` variable and has not been run since.
- `LLMEngine` constructor: the keeper is started reflectively (`getCudaContext().maybeStartClockKeeper()`)
  after every weight upload and after `MatmulPool.enable()`, not from `CudaContext.create` (the
  m2/m3 runs started it at context creation, which stalled the preload). The comment next to
  `MatmulPool.enable()` records the GLM-4.7-Flash 393 → 312 ms figure.
- `jvm-flags.md` already documents `cuda.clockkeeper` and points here; its "lowest-priority side
  stream" wording is ahead of the code until F1 step 1 lands (F0 step 7 adds the caveat).

### 7.4 Numbers that could not be traced to a measurement file

The following figures were measured by the author earlier the same day, in runs whose logs live
outside the repository (single sequential runs, not interleaved pairs, and, for the Qwen3MoE-engine
models, with the profiler artifact of section 2.1 still present), or come from the task brief;
none of them has an `an/` file behind it. Treat them as indicative and re-measure with protocol P
before building on them:
GLM-4.7-Flash CPU-only 253 ms per token (brief only); GLM-4.7-Flash GPU 184 ms per token with
attention 65, output 10.6 and moe 282 → 104 after the shared-expert fold (findings text; the
393 → 312 pool figure is in `LLMEngine`), and the ~2.3 ms per layer CPU expert time derived from
it; GPT-OSS-20B 378 vs 1059 ms per token and its phases (findings text, "task measurements");
Qwen3.5-35B-A3B 2.1 / 3.6 / 5.6 tok/s and 178 ms per token (findings text); the Qwen3-Coder
"132 / 169 / 237 ms moe" and "210 / 218 / 308 ms" totals of the earlier session, superseded by the
m4 batch, and the pre-fix profiler bias on Qwen3-Coder, which was never measured (section 2.1);
DS-Coder-V2-Lite 3.1 → 6.8 and 2.5 → 0.8 tok/s (`LLMEngine` comment; `smoke_out.txt` holds one
32-token run at 1.0 tok/s); `nvidia-smi -lgc` returning "Unknown Error" under WSL2 (brief only);
the 3.5 ms isolated Qwen3-Coder output-projection matmul (`LLMEngine` comment; the 55 ms spilled
figure is in `vr.txt`); the IQ1_M expert kernel's 17.7 GB/s at P4 and the Q4_K "~5 GB/s at P8"
(derived in the findings, not in `m5.txt`); the ~1.7 GB of VRAM held by the `o.txt` run (no smi
trace).

Estimates derived from the code rather than measured, marked as such where they appear: the
Qwen3.5-MoE 30–40 launches per layer and the 10–13 ms per token built on them (F10); the MLA pass's
21–24 launches and the 8.1 ms GLM host-submit figure (section 4.4); the ~1 ms per graph
instantiation (F3); the 10–20 µs pool dispatch cost and the 0.2–0.4 ms accumulate loop (F9); the
boxed-`Long` counts (F5).

---

## 8. Implementation (2026-09-29)

This section records what was built from the plan of section 5, where the implementation departs
from it and why, and what the findings made along the way changed. Section 9 holds the
measurements.

### 8.1 Findings that changed the plan

**The expert cache's upload source and GPU-mode determinism.** GPU-mode runs of the same build
produced different text from one process to the next (two Qwen3-Coder runs diverged after 96
characters), while CPU-only runs were identical. Teacher-forced logit dumps (a fixed token sequence
fed to two processes, full logit vectors compared) located it: re-running a MoE attention layer on
the same input inside one process gives bit-identical output, the first layer is identical across
processes, and from the second layer on the residual differs in the last digit — the CPU side's
Vector API code, compiled at different moments, rounds differently in rare cases. In GPU mode that
1e-9 difference goes through the Q8_1 quantization of every dp4a input and through the routing,
both discontinuous, and reaches about 1% of the logits within a few dozen layers; CPU-only mode has
neither amplifier. It is a property, not a bug, and it means a GPU-mode change is validated by logit
distances compared with the run-to-run noise (or by an in-process A/B), never by generated text.
The expert cache's uploads were moved to page-locked staging anyway (the old code freed the source
right after an asynchronous copy), but that was not the cause.

**PoCL replaces the JVM's signal handlers.** A GPU-mode run that decoded token by token after
`LLMEngine.autoConfigureGpu` died with a bare "Segmentation fault" and no hs_err report, and only
with the per-layer graphs enabled — which turned out to be a coincidence of code shape. The OpenCL
device probe loads every ICD, and PoCL with its LLVM runtime installs its own SIGSEGV, SIGBUS,
SIGFPE, SIGILL and SIGTRAP handlers over the JVM's; the JVM relies on those signals for its implicit
null checks, so the first one that trips kills the process. With the JDK's `libjsig.so` preloaded
the same run completed. `OpenCLContext.enumerateDevices` now saves the handlers before the probe and
restores them after it (`JvmSignalGuard`). This affected every CLI run, since the hardware plan and
`autoConfigureGpu` both probe OpenCL; it had simply not tripped often.

**CPU and GPU share the laptop's power budget.** The F0 CPU probe (a fixed integer chain timed on
one thread) showed the CPU 15-30% slower whenever the clock keeper held the GPU at P0 (17-23 W): on
Qwen3-Coder the attention phase fell from 50-100 to 22-26 ms per token and the output projection
from 7-18 to 2.6 ms, but the routed experts on the CPU grew by 10-40%, and on MiniMax, where the CPU
does most of the work, the keeper made the token slower (2120 against 1986 ms). Section 6.4 had
rejected a CPU/GPU coupling "as measured"; measured in-process it exists. The keeper's adaptive
policy was rewritten accordingly (F1 below): the first version, which minimised the attention phase,
drove the duty to 94% once the per-layer graphs made attention fast and halved the CPU experts'
speed.

**Four defects found while completing the plan (second half of 2026-09-29).** None of them was
in the plan, and each distorted either the output or a measurement:

- *The GPU passes never reset their recurrent state.* The DeltaNet state of the Qwen3.5 pass and
  the Mamba-2 SSM state of the Nemotron-H and Falcon-H1 passes live on the device and were zeroed
  only when the pass was built; the kernels that update them never read the position. Every
  generation after the first in one process therefore started from the previous sequence's state:
  in interactive mode, a second prompt produced different text from the same prompt run alone
  (Qwen3.5-0.8B: "Three primary colors are **Red, Yellow, and Blue**." against "The three primary
  colors are **Red**, **Yellow**, and **Blue**. These are the basic pigments..."; Nemotron-3-Nano
  likewise). The web server, interactive mode and the placement calibrator, which decodes dozens
  of short sequences before the real prompt, were all affected. The passes now zero that state
  when a token at position 0 is uploaded, and the second generation matches the single run on all
  three architectures.
- *`-Dmatmul.pool=force` left the pool off.* The pool's initial state was "on only when the
  property is `true`", so `force` started it disabled, and the GPU placements that were not
  MoE-optimized never re-enabled it. The first F4 measurement (pool forced on a partial offload)
  therefore compared no pool with no pool; its apparent 23% win was drift. Only `false` turns the
  pool off now.
- *`--gpu-layers N` was ignored without `--gpu`.* The auto-detection branch of the CLI built its
  own GPU configuration and dropped the requested layer count, so a first-N placement could only
  be forced together with `--gpu`.
- *The GPU batched prefill left the MoE decode loops cold.* After the batched prefill of
  Qwen3.5-35B-A3B, the routed CPU experts decoded about four times slower (196-212 ms per token
  against 45-71 ms after a per-token prefill), although `FloatTensor.warmUpDot` had run: the
  decode path's pool lambdas around `dot` had never executed. `SoftmaxMoe.warmDecodePath` runs that
  path on the first chunks of each layer, and the experts decode at 70-80 ms again.

The placement calibrator also exposed a latent bug in the CPU attention of the Qwen3.5 engine,
which indexed the score buffer with the engine's context length instead of the state's own: any
state smaller than the engine's context (the calibrator's) failed with an index out of bounds at
the first attention layer past position 0. The Falcon-H1 and Gemma 4 engines had the same pattern
and were fixed after the v1.19.0 release.

### 8.2 What was implemented, fix by fix

- **F0** — `DecodeProfile` (per-token phase timing shared by the three MoE engines; attention and
  expert phases always on and published through `GpuActivity`; prefill dropped at the first
  projection and reset per generation; last-10-token window; expert-phase split into routing, GPU
  launch, CPU experts, GPU wait and rest; CPU probe; cache statistics); Nemotron-H's profile drops the
  prompt too; cache statistics in the CLI stats block, on JMX (`gpuExpertCache*`) and `/api/metrics`;
  the warm-up comment corrected and the Qwen3.5 call gated; `jvm-flags.md` rows for every new
  property.
- **F1** — `CudaClockKeeper`: work stream at the greatest priority (−5 against the keeper's 0),
  generation-scoped through `GpuActivity` hooks with a 3 s linger, stopped when the engine drops its
  GPU path, bursts waited for with `cuEventQuery` and `parkNanos`, burst length recalibrated from
  event timing after every burst, NVML sampling (`NvmlSampler`, also available alone with
  `-Dgpu.pstate.sample=true`) with the P-state histogram in the CLI stats block, and an adaptive duty
  that chooses among off, 1/8, 1/4 and 1/2 by the mean wall time per decoded token (epsilon-greedy)
  instead of the plan's attention-based rule, for the reason of section 8.1. The plan's
  `setBusyHint` on the pass interfaces became the process-wide `GpuActivity` listener, which keeps
  base code free of java21 imports the same way. Stays opt-in: once the per-layer graphs and the
  dp4a expert cache were in, Qwen3-Coder no longer sat in P8 without it, the A-B-A gave 167 ms per
  token with the keeper against 141 and 180 ms without, and the adaptive controller chose `off` by
  itself (off 163 ms, 1/8 172 ms, 1/4 199 ms).
- **F2** — per-projection slot sizes (256-byte aligned), 128 MiB chunks, precomputed slot pointers,
  physical size and `cuMemGetInfo` delta printed, `-Dmoe.expert.gpu.cache.mb` / `--gpu-expert-cache`,
  the cache closed from `LLMEngine.close`. The optional per-layer size class (step 4) was added
  after the v1.19.1 release: every distinct per-layer gate/up/down triple is a size class, the
  budget is split so that every layer gets about the same number of units, chunks are allocated
  one class at a time in turn, and victims are searched only in the layer's class
  (`-Dmoe.expert.gpu.classes=false` restores one class). Qwen3-Coder has 24 layers with a Q6_K down
  projection and 24 with Q4_K: at the same 3066 MiB the cache holds 1116 experts instead of 1047
  (+6.6%), with a hit rate of 76.0% against 70.6% and 73.8% in the A-B-A (the times drifted too
  much to compare).
- **F3** — per-layer graphs in `MoeAttentionCudaPass` and `MlaAttentionCudaPass` (captured on a
  layer's second call; graph replays checked bit-identical to per-launch runs on the same input with
  `-Dcuda.moe.check=true`, 912 of 960 layer calls in the check replayed a graph), page-locked host
  blocks, the MLA shared expert queued after the attention download and taken after the routed
  experts, `cuda.sched`, event query and elapsed-time bindings.
- **F4** — done. Step 1 on the corrected build (see 8.1: `force` had measured no pool): Qwen3-Coder
  with its first 10 layers on the GPU decoded at 333 ms per token with the pool against 408 and
  667 ms without it, A-B-A, the win in the CPU share (attention of the 38 CPU layers and the
  experts). Steps 2-3: `FloatTensor.disableVirtualThreadMatmul` no longer disables the pool, the
  two conditional enables in `LLMEngine` became the `moe.cpu.pool=false` opt-out, and the five
  engines with batched prefill (standard, Gemma 4, Qwen3.5, Qwen3-MoE, DeepSeek2) gate it on
  `FloatTensor.anyGpuResident` over their layer weights (and on GPU matmuls being enabled, so the
  calibrator's CPU candidate still prefills batched). The MoE-optimized placement is unchanged: it
  already had the pool.
- **F9** — `gpu.tensor.min.bytes` (1 MiB) routes small per-tensor GPU matmuls to the CPU twin;
  `OutputRouter` times both devices for the output projection and switches with a 20% margin; the
  router tensor stays on the CPU; the per-tensor accumulate is a bulk copy plus a vector add; GPU
  tensors send `matmulRows` / `matmulRowsBatch` to their SIMD twin. The Qwen3-MoE-family shared
  expert inside `MoeAttentionCudaPass` (step 3) was validated after the v1.19.1 release on the local
  tiny GLM4-MoE test model: teacher-forced logits with the shared expert on the GPU and on the CPU
  differ by 1.8e-7 relative (FP32 rounding), two GPU runs are bit-identical. It is now on by default
  (`-Dmoe.attn.shared=false` disables it), as in the MLA pass; no full-size model of the family was
  available to time it. Running the pass on that model also fixed two edge cases: an empty RoPE
  table (a rope dimension of 1) allocated 0 bytes, which `cuMemAlloc` rejects, and the RoPE launches
  then had a grid of 0 blocks.
- **F5** — `int[]` routing tables instead of boxed maps; asynchronous promotion through a page-locked
  ring on a copy stream, the newcomer used only after its copy event; the per-token promotion budget
  keyed on the position (`noteToken`); resident-output downloads queued at launch; a per-layer
  speed-aware cap on the experts launched on the GPU; routing profiles saved at close and loaded at
  the next start with a warm start (1060 Qwen3-Coder experts in 606 ms); and the multi-expert launch
  of step 5, done without touching the fifteen matmul kernels: the kernel source is transformed at
  load time (the `__global__` entry becomes a `__device__` body, a new entry reads the weight, input
  and output pointers of expert `blockIdx.y` from device tables), so a layer's resident experts cost
  one launch per projection plus one activation launch instead of four launches and a download per
  expert. The dp4a step was then done the same way: the dp4a kernels are transformed into
  multi-expert kernels too (the signature parser now strips comments, since a dp4a signature
  carries `[(cols/32) * 40 bytes]` in one), gate and up read one Q8_1 quantization of the layer
  input and down one quantization of all the activated experts, which efd being a multiple of 32
  keeps block-aligned per expert; the batched-prefill path quantizes the chunk the same way. The
  kernels compile when the cache is built, not in the first token.
- **F6** — `attentionLayerBatch` in both passes (the MLA one for the latent layout only; the expanded
  DeepSeek-V2-Lite layout keeps the per-token prefill), with the projections through a cuBLAS FP16
  GEMM (`GemmF16`, weights read once per chunk) or, without cuBLAS, a per-token dp4a loop; the
  engines' batched prefill takes GPU-resident layers through it. Beyond the plan, the routed experts
  that are resident in the GPU cache run on the GPU over all the chunk's tokens routed to them
  (`launchResidentBatch`, the same multi-expert kernels with one table entry per token) while the CPU
  computes the others: without that, the batched prefill was no faster than the per-token one,
  because it gave up the GPU cache the per-token path uses. Qwen3.5-MoE got `SoftmaxMoe.forwardBatch`
  for its CPU prefill. Completed afterwards: `LayerGpuForwardPass.prefillBatch` in the Qwen3.5 pass
  (full offload; every projection one FP16 GEMM per chunk, the DeltaNet conv and recurrence token by
  token on the chunk's rows through the decode kernels with repointed parameter blocks, attention
  with the batch kernels and one flash launch, the MoE layers' experts through a batched engine
  callback on the CPU), and the expanded MLA layout (DeepSeek-V2-Lite's combined `wkv_b`, or
  `-Dmla.latent=false`) in `MlaAttentionCudaPass`, with the per-head K/V assembly per token because
  the shared k_rope differs per token.
- **F7** — the plan-side corrections (output projection, Q-LoRA and split-KV tensors, FP16 copies of
  the latent MLA pass, no router, every tensor rounded to the 2 MiB page, one helper for the load and
  the hardware plan, a flat reserve instead of 80% of the device), the shared experts and dense
  leading layers uploaded before the cache is sized, the cache's activation buffers allocated in its
  constructor, the flash kernel precompiled. Steps 4-5 became one mechanism, `VramGuard`, after the
  step-4 measurement showed that allocation accounting cannot decide: `cuMemAlloc` kept succeeding
  up to 8 GB on the 6 GB card, copies inside the newest block ran at about 146 GB/s up to 5.6 GB
  allocated and then at 24 and 9 GB/s, and `cuMemGetInfo` reported 0 free bytes about 500 MB
  before the drop. An allocation of at least 1 MiB made while less than 256 MB is free is probed
  instead: a kernel reads it with cache-volatile loads against a reference buffer allocated when
  the context was created (a plain copy of an 8 MiB block is served from the 24 MB L2 and measured
  840 GB/s; a cache-volatile load re-fetches a system-memory line over PCIe on every read), after
  a warm-up that raises the clocks. Below a third of the reference rate the allocation is released:
  a weight falls back to its CPU twin, a MoE attention layer (and the following ones) keeps CPU
  attention, the expert cache stops growing. `CudaFloatTensor.getGpuWeights` failures already
  switched a tensor to its twin, so the weight path needed no new handling.
- **F8** — `PlacementCalibrator` with park switches in the four engines, a process-wide switch that
  sends every GPU tensor to its CPU twin (the true CPU candidate), `MatmulPool.setActiveThreads`,
  the keeper as an axis, stored verdicts, `--auto-tune` calling it on the loaded model, and the report
  on JMX and `/api/metrics`. Steps 7 and 8 were done afterwards: when a first probe decodes faster
  than 200 ms per token, every candidate decodes after a 256-token prefix (`placement.context`),
  and each candidate's prefill of that prefix is timed and printed, weighted into the score by
  `placement.workload` (0 by default, 1 for `balanced`, 8 for `prompt`). The verdict version became
  2, so stored verdicts of the first version are not applied.
- **F10** — per-layer graphs for the Qwen3.5-MoE layers, the routed-expert callback outside them.
- **F12** — rejected by its zero-code test: `-Dmatmul.spin=2000000` against the default 20000 on
  Qwen3-Coder with the cache off, A-B-A, gave 256 ms per token against 249 and 366 ms, inside the
  run-to-run drift and below the 7-13 ms ceiling the plan set. Nothing was built.
- **F11** — done, gain not confirmed. The zero-code test won: `-Dmatmul.chunks.per.thread=16` on
  Qwen3-Coder with the cache on gave 124 ms per token against 134 and 212 ms, CPU experts 80 against
  103 and 123 ms. The compaction was then built in the three CPU expert loops (`Qwen3MoEInferenceEngine`,
  `MoEFFN`, `SoftmaxMoe`): the parallel ranges cover only the CPU slots, and an unfilled routing slot
  is zeroed outside the loops. It is bit-identical by construction. Its own A-B-A against the
  previous build was inconclusive: 124 ms per token against 369 and 106 ms, with the first
  baseline run in a throttled state.

## 9. Validation (2026-09-29)

Every comparison below is an interleaved run of protocol P, but shortened to at most five minutes
per comparison: A-B-A, one process per run, 20-40 decoded tokens, windows of ten tokens, best and
median window compared phase by phase. On this laptop two identical runs a few minutes apart differ
by up to 2×, so a result counts only when B falls outside the range of the two A runs; where it
does not, the entry says inconclusive. GPU-mode output is not bit-reproducible across processes
(section 8.1), so correctness is checked with teacher-forced logit distances against the
run-to-run noise, or with a CPU reference.

### 9.1 Decode

| Fix | Model and placement | A | B | A (again) | Verdict |
|---|---|---|---|---|---|
| F4 pool under the GPU | Qwen3-Coder-30B, first 10 layers on the GPU | 667 ms | 333 ms | 408 ms | pool wins (−18% on the best A) |
| F4 pool under the GPU | Qwen3-8B, 16 of 36 layers on the GPU | 1.2 tok/s | 2.6 tok/s | 1.4 tok/s | pool wins (about 2×) |
| F5 dp4a expert slots | Qwen3-Coder-30B, MoE-optimized, cache on | 252 ms | 174 ms | 198 ms | dp4a wins; expert phase 124 against 147-192 ms |
| F11 chunk count (zero-code) | Qwen3-Coder-30B, cache on | 134 ms | 124 ms | 212 ms | 16 chunks per thread win |
| F11 compaction | Qwen3-Coder-30B, cache on | 369 ms | 124 ms | 106 ms | inconclusive (first A throttled) |
| F12 spin window | Qwen3-Coder-30B, cache off | 249 ms | 256 ms | 366 ms | no gain; rejected |
| F1 keeper | Qwen3-Coder-30B, cache on | 180 ms | 167 ms | 141 ms | no gain; stays opt-in |

Times are the best ten-token window in ms per decoded token. F10 (per-layer graphs for the
Qwen3.5-MoE layers) was validated earlier the same day: Qwen3.5-35B-A3B 122 against 129.5 ms per
token, GPU layers 26.3 against 34.9 ms, text identical.

### 9.2 Prefill

| Model | Path | Per prompt token | Correctness |
|---|---|---|---|
| Qwen3.5-0.8B, 452-token prompt | GPU per token → GPU batched | 9.8 → 1.5 ms (6.5×) | final prompt logits within 0.0125 of the CPU (per-token GPU path: 0.52); argmax equal to the CPU on 16 of 16 steps |
| Qwen3.5-35B-A3B, 452-token prompt | GPU per token → GPU batched | about 130 → 115 ms | 40-token text identical to the per-token reference |
| DeepSeek-Coder-V2-Lite, 120-token prompt | GPU per token → GPU batched (expanded MLA) | 156 → 78 ms (2×, two batched runs) | logit distance 1.86 at the prompt's end against a run-to-run noise of 2.24 |
| Qwen3-Coder-30B, 314-token prompt | GPU batched with the hybrid expert cache | about 27 ms | distance 2.57 against a noise of 2.61 (earlier the same day) |

For Qwen3.5-35B-A3B the routed experts on the CPU bound the prefill, as the plan expected for the
MoE models. The batched path first left the MoE decode about 4× slower in its CPU experts (section
8.1); with `SoftmaxMoe.warmDecodePath` its CPU experts came back to 70-80 ms per token (from
196-212). Four alternating runs then settled the decode after the two prefills: per-token prefill
132 and 137 ms per token (median window), batched prefill 176 and 129 ms, the 176 being drift; the
40 generated tokens were identical in all four runs.

### 9.3 VRAM guard (F7)

| Test | Result |
|---|---|
| 256 MiB blocks up to 8 GB on the 6 GB RTX 4050 (WSL2) | every allocation succeeds; copy rate about 146 GB/s up to 5.6 GB, then 24 and 9 GB/s; `cuMemGetInfo` free at 0 from about 5.1 GB |
| Qwen3-Coder-30B, expert cache margin −1500 MB | the chunk that spilled read at 3% of the reference rate and was released; the cache kept 35 chunks |
| Qwen3-Coder-30B, every allocation of 1 MiB or more probed | 218 probes, 0 rejected, 1 inconclusive; load time within 1.5 s of an unprobed load |

### 9.4 Placement calibration (F8)

On Qwen3.5-0.8B (full offload, before its GPU batched prefill existed) the calibrator timed decode
after a 256-token prefix, since a first probe decoded faster than 200 ms per token: GPU 9.3 ms per
token with a prefill of 9.7 ms per token against CPU 102.1 and 44.2, verdict GPU; the thread stage
found 11 and 15 threads equal. It also exposed the Qwen3.5 CPU attention defect of section 8.1.

### 9.5 Regression checks

- Recurrent-state reset: a second prompt in one interactive process now produces the same text as
  the same prompt run alone, on Qwen3.5-0.8B, Nemotron-3-Nano-4B and Falcon-H1-1.5B.
- GPU smoke on eight architectures (Llama-3.2-1B, Qwen3.5-0.8B, Gemma-3-1B, LFM2-1.2B,
  Falcon-H1-0.5B, Nemotron-3-Nano-4B, OLMo-2-1B, Qwen3-0.6B): correct answers, 20-108 tok/s.
- `--gpu-backend opencl` with the pool on: Llama-3.2-1B completed with correct output on the only
  OpenCL device of this machine, PoCL on the CPU (the signal-handler guard logged its restore).

### 9.6 Follow-up after v1.19.1

F2 step 4 and F9 step 3 were completed (section 8.2). A GPU smoke over the local models, run in
blocks of at most five minutes with a short prompt at temperature 0, gave correct answers from
30 models: Llama-3.2-1B, Qwen2.5-3B, Qwen3-0.6B, Qwen3.5-0.8B, SmolLM3, Gemma-2-2B, Gemma-3-1B,
Gemma-3n-E4B, Gemma-4-E2B, Phi-3-mini, Phi-4-mini, ERNIE-4.5-0.3B, Hy-MT2-1.8B, Spark-X2.5-1.7B,
Nanbeige4.2-3B, Granite-3.3-2B, Granite-4.0-h-micro and h-tiny, Falcon3-3B, Falcon-H1-0.5B,
Nemotron-3-Nano-4B, OLMo-2-1B, LFM2-1.2B, LFM2.5-8B-A1B, Ling-3.0-tiny, Qwen2.5-VL-3B, Qwen3-VL-4B,
Mistral-7B, aya-23-8B and GLM-4-9B (plus Qwen3-Coder-30B, Qwen3.5-35B-A3B and
DeepSeek-Coder-V2-Lite from the measurements above). The tiny GLM4-MoE model, whose weights are
random, ran through the GPU attention pass.

Found and not resolved:

- **Olmo-3-7B answers with a single `<|im_start|>`.** The same happens on the CPU and with the GPU
  build from before this work, so it is an older defect, most likely in its chat template (the
  prompt carries an injected function-calling system message), not in the GPU path.
- **aya-23-8B decodes at 1.7 tok/s** although the plan puts all 32 layers on the GPU (5334 MB of
  6140 estimated). An 8B model on this GPU should be several times faster. The suspicion, not
  verified, is that part of it lands in WSL2's shared memory (the 256K-token output projection is
  large, and the VRAM guard does not cover every allocation of the dense pass).
- **Gemma-3n-E4B decodes at 2.0 tok/s**; whether that is its usual speed was not checked.
- The MiniMax-M2 re-measurement and the remaining large models (GLM-4.7-Flash, GPT-OSS-20B,
  Devstral-24B) were not run in the smoke.
