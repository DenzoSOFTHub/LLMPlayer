# Autotuning & placement heuristics

This document is the single source of truth for how LLMPlayer decides where every tensor of a model lives — GPU VRAM, CPU RAM, or disk — and how it sizes the CPU thread pool. It is written for a developer who needs to understand the exact decision rules: every threshold, formula, flag, and the file:line where it lives. The deep analysis (bandwidth cost model derivation, value-per-VRAM-byte ranking, roadmap) is in `docs/optimization/placement-autotuning.md`; this is the operational reference.

All decision logic is concentrated in three files:

- `src/main/java/it/denzosoft/llmplayer/api/LLMEngine.java` — `load()` (placement), `estimateKvCache()`, `sumNonExpertTensorBytes()`, `getPhysicalMemorySize()`, `buildHardwarePlan()`.
- `src/main/java/it/denzosoft/llmplayer/cli/CLIRunner.java` — thread-pool sizing, NUMA hint, `autoTune()` / `measurePlacement()`.
- `src/main/java/it/denzosoft/llmplayer/cli/CLIOptions.java` — flag parsing and `detectPhysicalCores()` / `detectNumaNodes()`.

---

## 1. The cost model

**Rule.** Single-token decode is memory-bandwidth bound, not compute bound: per token every weight is read once (or, for MoE experts, K-of-N times) and the arithmetic per byte read is trivial. Per-token latency is therefore approximately the sum over all tensors of `bytes / bandwidth(where the tensor lives)`, plus the KV-cache read. Placement is binary per tensor — a tensor lives entirely on GPU (fast HBM read + matmul on GPU) or entirely in CPU RAM (slower DRAM read + matmul on CPU). Weights are **never** streamed across PCIe per token, because streaming a tensor over PCIe gen4 (~16–32 GB/s) is several times slower than reading it from CPU DRAM (~50–90 GB/s) and doing the matmul on the CPU.

**Rationale.** Because the bandwidth difference `(1/BW_cpu − 1/BW_gpu)` is a constant, the value of placing a tensor on the GPU is proportional only to its reads-per-token. This is the whole game: rank tensors by reads-per-token, fit the highest-value bytes into VRAM, and run the rest on CPU. Attention/FFN/output (1 read/token) are high value; MoE routed experts (K/N reads/token, e.g. 6/64 ≈ 0.09×) are low value and stay on CPU; the token-embedding table is a lookup (≈0 per-token matmul value) and is always on CPU.

This model is the justification for every threshold below. The full derivation, bandwidth table, and measured routing-concentration data are in `docs/optimization/placement-autotuning.md` §1–2.

---

## 2. Strategy selection: first-N-layers vs MoE-optimized

The placement strategy is chosen in `LLMEngine.load()` (LLMEngine.java:305–408), only when a GPU was successfully initialized. The deciding inputs are the requested layer count (`gpuConfig.getGpuLayers()`, default `-1`), the GPU memory mode, and whether the model is MoE (`quickConfig.expertCount() > 0`, read from a quick GGUF parse at LLMEngine.java:316–318).

**Decision tree (in evaluation order):**

1. **Shared-memory modes** (LLMEngine.java:334–338). If `--gpu-memory managed` or `--gpu-memory host-mapped`, all `blockCount` layers go on GPU and the budgeting below is skipped — the driver pages VRAM ↔ system RAM.

2. **`--gpu-layers -1` (auto, the default)** — the engine branches on MoE-ness (LLMEngine.java:339–391):
   - **MoE model (`expertCount > 0`)** (LLMEngine.java:342–363): compute `nonExpertBytes = sumNonExpertTensorBytes(...)` and `usableVram = vram * 0.80`. **If `nonExpertBytes <= usableVram`**, select **MoE-optimized**: `moeOptimized = true`, `gpuLayersUsed = blockCount` — every layer's attention, norms, router, and shared experts go on GPU, while the routed experts (the bulk of the model) stay on CPU. **Otherwise**, fall back to first-N-layers using the crude estimate `bytesPerLayer = modelFileSize / blockCount`, `fittableLayers = floor(usableVram / bytesPerLayer)`, capped at `blockCount` (LLMEngine.java:354–362).
   - **Dense model (`expertCount == 0`)** (LLMEngine.java:364–386): use the **first-N-layers** KV-aware budget of §3.
   - **VRAM undetectable** (`vram <= 0`, LLMEngine.java:387–390): put all `blockCount` layers on GPU.

3. **`--gpu-layers 0`** (LLMEngine.java:392–395): all `blockCount` layers on GPU, budget ignored.

4. **`--gpu-layers N` (N > 0)** (LLMEngine.java:396–398): force first-N-layers, `gpuLayersUsed = min(N, blockCount)`. An explicit positive `N` always means first-N-layers — it overrides MoE optimization.

5. **Explicit MoE override** (LLMEngine.java:401–404): after the above, if `gpuConfig.isMoeOptimized()` is set (`--moe-optimized`), `moeOptimized` is forced `true`. The CLI tri-state `Boolean moeOptimized` (CLIOptions.java:41, default `null` = auto) is wired into `GpuConfig` only when non-null (CLIRunner.java:71–73, 82–84), so `--no-moe-optimized` forces first-N even for MoE models.

**`sumNonExpertTensorBytes`** (LLMEngine.java:1701–1791) is what the MoE 80% check sums: for leading dense layers (`i < leadingDenseBlockCount`) it counts all tensors; for MoE layers it counts norms, attention (Q/K/V/O + QK-norms, or the DeepSeek2 MLA tensors `attn_kv_a_mqa` / `attn_kv_a_norm` / `attn_kv_b`), the router (`ffn_gate_inp`), and the shared experts (`ffn_*_shexp`) — and deliberately **excludes** the routed expert weights (`ffn_*_exps`).

**Per-tensor placement during load** is done in `ModelLoader` (ModelLoader.java:86, 314/467 `moeMode = moeOptimizedGpu && hasGpu`): inside the layer loop `TensorFactory.gpuBufferManager` is toggled GPU-ON for attention/norms/router/shared experts and GPU-OFF for the three `ffn_*_exps` tensors (ModelLoader.java:328, 357–361, 363 for Qwen3MoE; mirrored at 477, 494–498, 505 for DeepSeek2).

**Rationale.** Dense layer weights are fungible (all read once per token), so the engine just fills VRAM greedily and contiguously (first N layers, avoiding PCIe round-trips mid-network). For MoE, all the 1×-per-token tensors (attention) have high value and the routed experts have low value, so the KTransformers-style split puts all attention on GPU and keeps experts on CPU. The MoE budget is a more conservative 80% (vs 90% dense) because the expert-on-CPU layout is more heterogeneous and leaves headroom for a future hot-expert VRAM cache.

**Controlling flags:** `--gpu` / `--no-gpu`, `--gpu-device <id>`, `--gpu-layers <N>` (-1 auto / 0 all / N first-N), `--moe-optimized` / `--no-moe-optimized`, `--gpu-memory {device|managed|host-mapped}`, `--gpu-backend {auto|cuda|opencl}`.

---

## 3. KV-aware VRAM budget and FP16-KV auto-enable

This is the dense first-N-layers budget (LLMEngine.java:364–386), and the part the engine refers to as "Phase 1.1": each GPU-resident layer reserves not just its weight bytes but also its slice of the KV cache, which grows linearly with context length.

**Closed-form layer count:**

```
N = min( blockCount, floor( usableVram / (bytesPerLayer + kvPerLayer) ) )
```

with the terms:

- `usableVram = (long)(vram * 0.90) - nonLayerBytes` (LLMEngine.java:370) — 90% of total VRAM, less the non-layer tensors. The 10% is reserved for driver/allocator overhead.
- `nonLayerBytes = estimateNonLayerBytes(...)` (LLMEngine.java:369, defined 1806–1814) — the output projection (`output.weight`) plus output norm (`output_norm.weight`). The token-embedding table is **not** counted: it is loaded on CPU for every architecture (lookup only).
- `bytesPerLayer = (modelFileSize - nonLayerBytes) / blockCount` (LLMEngine.java:371–372) — a uniform average weight size per layer (valid because dense layer weights are fungible).
- `kvPerLayer` — the per-layer KV reserve, derived from `estimateKvCache()` (LLMEngine.java:1077–1088): for the all-layers FP32 estimate, standard / Qwen3MoE / Phi3 / Mistral3 use `2 * blockCount * ctx * kvDim * 4` bytes, while DeepSeek2 MLA uses `blockCount * ctx * (headCount*keyLength + headCount*valueLength) * 4`. Per layer: `kvPerLayerFp32 = kvAllFp32 / blockCount` (LLMEngine.java:326), and `kvPerLayerFp16 = kvPerLayerFp32 / 2` (LLMEngine.java:327). The context used is `ctxForKv = min(maxContextLength, model contextLength)` (LLMEngine.java:324).

**FP16-KV auto-enable** (LLMEngine.java:374–378). The engine computes how many layers fit under each KV element size:

```
fitFp32 = floor( usableVram / (bytesPerLayer + kvPerLayerFp32) )
fitFp16 = floor( usableVram / (bytesPerLayer + kvPerLayerFp16) )
autoFp16 = !fp16KvSet && fitFp16 > fitFp32 && fitFp32 < blockCount
```

All three conditions must hold: (1) the user did not explicitly set `-Dcuda.kv.fp16` (`fp16KvSet`, LLMEngine.java:328), (2) FP16 KV fits strictly more layers, and (3) FP32 KV does not already fit every layer. When they hold, the engine sets `System.setProperty("cuda.kv.fp16", "true")` and uses `kvPerLayer = kvPerLayerFp16` for the final layer count (LLMEngine.java:377–379). The chosen `N` is then `min(max(0, fittableLayers), blockCount)` (LLMEngine.java:380), and the result is logged with the KV MB and an "FP16 KV auto-enabled" note (LLMEngine.java:383–386).

**Rationale.** Reserving KV up front stops long contexts from over-committing VRAM after the weights are already placed (a 3B model at ctx=24576 reserves several GB of KV). Auto-switching to FP16 KV is a free way to reclaim that VRAM for additional weight layers whenever FP32 KV is the binding constraint, at negligible quality cost.

**Controlling flags:** `--context-length <N>` / `-c` (default 2048, CLIOptions.java:27 — drives `kvPerLayer`), `-Dcuda.kv.fp16=true|false` (explicit override disables the auto-enable), `--gpu-layers <N>` (bypasses the formula entirely).

---

## 4. `--auto-tune` calibration

**Rule.** When `--auto-tune` is set and the GPU is not disabled (CLIRunner.java:109–111), the engine empirically measures decode throughput of the heuristic GPU placement and of CPU-only, and keeps the faster (CLIRunner.java:176–208). For each candidate it loads the model, runs a discarded 6-token warm-up, then a measured 24-token decode, and reads `GenerationResponse.tokensPerSecond()` (CLIRunner.java:177–178, 199–203). The calibration prompt is fixed (`"Write one short paragraph about the history of computing."`). The decision (CLIRunner.java:183–191):

- If `tpsGpu` is valid and `tpsGpu >= tpsCpu` → keep the GPU config (**ties go to GPU**).
- Else if `tpsCpu` is valid → return a fresh `new GpuConfig()` (GPU disabled, CPU-only).
- Else (both measurements failed/NaN) → keep the GPU config (fall back to the heuristic choice).

**Rationale.** The file-size-based heuristic can be wrong: on a weak iGPU, a slow bus, or when too little of a partial-fit model lands in VRAM, GPU placement can be slower than running entirely on CPU. Measuring once at load auto-corrects this partial-fit footgun, at the cost of loading the model a couple of extra times. It is opt-in for exactly that reason.

**Controlling flags:** `--auto-tune` (CLIOptions.java:33), gated off by `--no-gpu`.

---

## 5. CPU thread sizing (physical cores + NUMA)

**Rule.** The matmul thread pool defaults to **physical** cores, not logical/hyperthreaded ones: `threads = detectPhysicalCores()` (CLIOptions.java:24). `detectPhysicalCores()` (CLIOptions.java:234–251) reads Linux sysfs `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list`, counts the distinct sibling groups (one per physical core), and falls back to `Runtime.getRuntime().availableProcessors()` on non-Linux or any read failure. The chosen count is applied by setting the ForkJoinPool common-pool parallelism (CLIRunner.java:43–45):

```
System.setProperty("java.util.concurrent.ForkJoinPool.common.parallelism", String.valueOf(threads))
```

If `threads < logical`, the engine logs that it is using physical cores and that `--threads` overrides (CLIRunner.java:46–50). It also calls `detectNumaNodes()` (CLIOptions.java:254–261, counting `/sys/devices/system/node/node*`), and when more than one node is present it prints a `numactl --cpunodebind=0 --membind=0` pinning hint (CLIRunner.java:51–55) — a hint only; the engine does not pin threads itself.

**Rationale.** The matmul hot path is bandwidth bound, so two hyperthreads on one core only contend for that core's load/store ports — physical sizing is the bandwidth-correct default and matches llama.cpp practice. NUMA locality is decisive on multi-socket boxes, hence the hint.

**Honest measurement caveat.** The physical-core default rests on the bandwidth argument and llama.cpp precedent, **not** on a clean measured speedup. On the thermally-constrained reference box the effect is not cleanly measurable: one clean A/B showed physical-11 ≈ 5.0 vs logical-22 ≈ 3.8 tok/s (~24%), but a heat-soaked sweep was contradictory (physical helped some models and hurt others, with absolute numbers throttled into the 1.5–8 tok/s range). No fixed speedup figure is claimed; `--threads N` overrides for users who measure otherwise on their own hardware. See `docs/optimization/placement-autotuning.md` §5 for the data.

**Controlling flags:** `--threads <N>` (override; default physical cores), `-Djava.util.concurrent.ForkJoinPool.common.parallelism=<N>` (the underlying property the flag sets).

---

## 6. Lazy mmap for models > RAM + MADV_RANDOM

**Rule (preload decision, LLMEngine.java:414–430).** The engine warms (preloads) the whole file into RAM only when it comfortably fits. With no explicit override:

```
preload = ramBytes <= 0 || modelBytes < (long)(0.85 * ramBytes)
```

So a model at or above 85% of physical RAM (`getPhysicalMemorySize()`, LLMEngine.java:1394–1404, via the Sun `OperatingSystemMXBean.getTotalPhysicalMemorySize()`; returns -1 if unavailable, in which case preload defaults to true) skips preload and relies on lazy mmap — only the working set pages in on demand, read-only from the model file (never from swap), and for MoE the cold experts stay on disk. Explicit overrides take precedence (LLMEngine.java:415–420): `-Dpreload=true` or `-Dno.preload=false` force preload on; `-Dno.preload=true` or `-Dpreload=false` force it off.

**Rule (MADV_RANDOM, LLMEngine.java:431–434).** When and only when preload is skipped, the engine sets `mmap.advise.random = String.valueOf(!preload)` (i.e. `true`). `MemorySegmentTensorData.mapFile()` reads that property (MemorySegmentTensorData.java:60) and, if true, calls `adviseRandom()` (MemorySegmentTensorData.java:34–48), which invokes the libc `madvise(addr, len, MADV_RANDOM=1)` via Panama FFM — best-effort, silently no-op where the `madvise` symbol is absent (non-Linux).

**Rationale.** Preloading a file larger than RAM is futile — it reads the whole file only for the OS to evict the pages again, turning the warm into a cold read in disguise. The 0.85 cap leaves a 15% safety margin against near-edge OOM. `MADV_RANDOM` then disables OS read-ahead, which would otherwise waste disk bandwidth fetching pages that the sparse, random MoE cold-expert access pattern will never use.

**Related memory-safety check.** Separately, `checkMemory` / `buildHardwarePlan` marks a config unsafe (and `CLIRunner` prompts for confirmation unless `--force`/`-y`) when `estimatedRam = modelFileSize + KV` is **not** strictly below `0.90 * availableRam`, where `availableRam = min(jvmMaxMemory, physicalMemory)` (LLMEngine.java:1350–1366; 10% margin at line 1357).

**Controlling flags:** `-Dpreload=true|false`, `-Dno.preload=true|false`, the internal `mmap.advise.random` property (set automatically, not user-facing), and `--force` / `-y` to skip the unsafe-config prompt.

---

## 7. Summary of controlling flags and `-D` properties

| Flag / property | Default | Where read | Effect |
|---|---|---|---|
| `--gpu` / `--gpu-device <id>` | auto-detect | CLIRunner.java:65–69 | Enable GPU; select device ordinal. |
| `--no-gpu` | off | CLIRunner.java:62–64 | Disable GPU; CPU-only. |
| `--gpu-layers <N>` | `-1` (auto) | LLMEngine.java:339–398 | -1 = KV-aware auto; 0 = all layers; N>0 = first-N (overrides MoE opt). |
| `--moe-optimized` | `null` (auto) | LLMEngine.java:401–404 | Force attention-on-GPU / experts-on-CPU. |
| `--no-moe-optimized` | `null` (auto) | CLIOptions.java:41 → GpuConfig | Force first-N-layers even for MoE. |
| `--gpu-memory {device\|managed\|host-mapped}` | `device` | LLMEngine.java:334–338 | managed/host-mapped → all layers on GPU, driver pages. |
| `--gpu-backend {auto\|cuda\|opencl}` | `auto` | CLIRunner.java:74/85 | Backend selection (CUDA preferred). |
| `--auto-tune` | off | CLIRunner.java:109–111, 176–208 | Measure GPU vs CPU decode tok/s, keep faster. |
| `--context-length <N>` / `-c` | `2048` | LLMEngine.java:324, 1077–1088 | Drives `kvPerLayer` in the VRAM budget. |
| `--threads <N>` | physical cores | CLIRunner.java:43–45 | CPU matmul pool size. |
| `--force` / `-y` | off | CLIRunner.java:95–104 | Skip the not-recommended (RAM) confirmation prompt. |
| `-Dcuda.kv.fp16=true\|false` | unset (FP32) | LLMEngine.java:328–329, 376–378 | Force FP16 KV; setting it disables the auto-enable. |
| `-Dpreload=true\|false` | auto | LLMEngine.java:415–418 | Force full-file preload on/off. |
| `-Dno.preload=true\|false` | auto | LLMEngine.java:416–420 | Force lazy mmap on/off. |
| `mmap.advise.random` (internal) | set by `load` | LLMEngine.java:434; MemorySegmentTensorData.java:60 | `true` ⇔ `!preload` ⇒ `madvise(MADV_RANDOM)`. |
| `-Djava.util.concurrent.ForkJoinPool.common.parallelism=<N>` | physical cores | CLIRunner.java:44 | Underlying property set by `--threads`. |

**Fixed thresholds at a glance:** dense VRAM cap `0.90` (LLMEngine.java:370); MoE VRAM cap `0.80` (LLMEngine.java:345); preload / lazy-mmap cutoff `0.85 × physical RAM` (LLMEngine.java:424); memory-safety margin `0.90 × availableRam` (LLMEngine.java:1357); auto-tune warm-up 6 tokens, measured 24 tokens, GPU wins on tie (CLIRunner.java:178, 183, 199).

---

For the full bandwidth cost-model derivation, the value-per-VRAM-byte ranking, measured MoE routing-concentration data, and the optimization roadmap, see `docs/optimization/placement-autotuning.md`.
