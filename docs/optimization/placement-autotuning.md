# Auto-tuning execution for models that exceed available VRAM

This document analyses how LLMPlayer should place tensors and schedule work when a model does not
fit entirely in the available GPU memory, and proposes heuristics that maximise both throughput
(tokens/s) and response time (latency) by making the best use of the available GPUs, CPU cores, and
memory. It ends with a three-phase implementation roadmap.

## 1. The physics this rests on

Single-stream **decode** (one token at a time) is **memory-bandwidth bound**, not compute bound.
Every weight is read exactly once per token and the arithmetic per byte is tiny (one multiply-add),
so per-token latency is, to first order:

```
t_token ≈ Σ_tensors ( bytes(tensor) / BW(where tensor lives) )  +  KV_bytes/BW  +  overheads
```

Bandwidth hierarchy on a typical machine (the RTX 4050 reference box):

| Path | Bandwidth | Ratio |
|---|---|---|
| GPU HBM (RTX 4050) | ~192 GB/s | 1× |
| GPU HBM (A100/H100) | 1500–3300 GB/s | 8–17× |
| CPU DRAM (DDR5 dual-channel) | ~50–90 GB/s | ~0.3–0.45× |
| **PCIe gen4 ×8 / ×16** | **~16 / 32 GB/s** | **~0.1×** |

**The rule that follows:** never stream a weight across PCIe to compute it on the GPU per token.
Streaming a 100 MB layer over PCIe (16 GB/s ≈ 6 ms) is ~5× slower than reading it from CPU DRAM
(80 GB/s ≈ 1.25 ms) and doing the matmul on the CPU. The decision is therefore binary **per tensor**:
it lives on the GPU (and is matmul'd there) or it lives in CPU RAM (and is matmul'd there). This is
exactly what llama.cpp's `-ngl` and LLMPlayer's `gpu-layers N` do — they partition residency, they do
not offload-and-stream. Every byte moved to VRAM is read ~2–3× faster (192 vs ~80 GB/s), so the whole
game is: **fit the highest-value bytes into VRAM, run the rest on the CPU.**

## 2. Value-per-VRAM-byte: the ranking heuristic

Define the value of placing a tensor on the GPU as the per-token time it saves:

```
value(tensor) = bytes × reads_per_token × (1/BW_cpu − 1/BW_gpu)
```

`(1/BW_cpu − 1/BW_gpu)` is a constant, so **value per VRAM byte is governed by `reads_per_token`:**

| Tensor class | reads/token | Value per VRAM byte | Placement priority |
|---|---|---|---|
| Attention (Q/K/V/O), norms, router, shared-expert | 1 | high | GPU first |
| Dense FFN (gate/up/down) | 1 | high (equal to attention) | GPU |
| Output projection (vocab×dim) | 1 | high | GPU |
| **MoE routed experts** | **K/N** (top-K of N) | **low** (e.g. 6/64 ≈ 0.09×) | **GPU last** |
| Token embedding | a lookup, not a matmul | ~0 | **CPU always** |
| KV cache | read every token; grows with context | high, but competes with weights | GPU if attention is on GPU |

Three consequences are the backbone of every heuristic below:

1. **All dense weights have equal value per VRAM byte** → there is no per-tensor knapsack to solve;
   you just fill VRAM. Only contiguity breaks the tie.
2. **Contiguity matters because of transfer boundaries.** If the GPU-resident layers are contiguous
   (0..N−1), the activation crosses PCIe exactly once (download after layer N−1, then the CPU runs
   N..L−1). Scattered placement pays a PCIe round-trip at every GPU↔CPU transition. This is why
   "first-N-layers" beats arbitrary scattering and why LLMPlayer keeps the GPU set contiguous to
   enable the fused `CudaForwardPass` chain. **Keep the GPU set contiguous.**
3. **Experts are the great exception.** A 30B MoE is ~80–90% expert weight but only top-K experts are
   read per token. Putting all experts in VRAM buys `K/N` of the value per byte at the highest VRAM
   cost — the worst trade. Experts belong on the CPU, attention on the GPU on every layer (the
   KTransformers insight LLMPlayer already implements as `moeOptimized`).

## 3. What LLMPlayer does today, and the gaps

From `LLMEngine.load()` auto-detection:

- **Dense:** `N = floor((0.90·VRAM − nonLayerBytes) / bytesPerLayer)`, contiguous layers 0..N−1 on
  GPU, rest on CPU; `nonLayerBytes` = output projection + final norm; embedding on CPU. Matches theory.
- **MoE:** if `sumNonExpertTensorBytes ≤ 0.80·VRAM` → all-attention-on-GPU + experts-on-CPU
  (`moeOptimized`); else fall back to first-N-layers. Matches theory.

Four concrete gaps:

1. **The VRAM budget ignores the KV cache.** Placement uses `0.90·VRAM − nonLayerBytes` but never
   reserves room for the KV cache, which is allocated per GPU layer and **grows linearly with context
   length**. At long context this over-commits VRAM. `estimateKvCache()` exists but is only used for
   reporting. → reserve KV in the budget; prefer FP16/Q8 KV to reclaim that space for weights.
2. **No multi-GPU.** Placement uses a single `gpuConfig.getDeviceId()`; `enumerateDevices()` sees all
   cards but only one is used. → layer-split across GPUs.
3. **The CPU side is untuned.** `ForkJoinPool.common.parallelism = availableProcessors()` counts
   hyperthreads, which hurt a bandwidth-bound matmul; no NUMA pinning. → physical-core count + NUMA
   locality.
4. **MoE experts are all-or-nothing on the CPU.** No hot-expert GPU cache for Q4_K MoE (only the
   MXFP4 `ExpertGpuCache`; `GraniteExpertGpu` does Q4_K but does not cache by routing frequency).
   Routing has strong temporal locality → an LRU GPU cache of hot experts is the biggest remaining
   MoE lever.

## 4. Auto-tuning the split — three tiers

**Tier 0 — static analytic (no calibration; refine the current path).** Each GPU layer costs
`bytesPerLayer + kvPerLayer` of VRAM, so the KV reserve folds into a closed form:

```
usableVram   = α·VRAM − driverReserve          # α≈0.90–0.92
kvPerLayer   = 2 · ctx · kvDim · kvElemBytes    # kvElemBytes = 2 (FP16) or 4 (FP32)
N_layers     = floor( (usableVram − nonLayerBytes) / (bytesPerLayer + kvPerLayer) )
```

This is the current code **plus** the `+ kvPerLayer` term. It is accurate because dense bytes are
fungible (§2) — there is nothing smarter than "fill it."

**Tier 1 — fractional-layer / attention-first.** When the budget leaves room for a fraction of a
layer, place that layer's **attention before its FFN**: attention tensors are smaller and slightly
more compute-bound, and keeping attention on GPU lets that layer's KV stay on GPU. Free win,
generalises the MoE attention-priority strategy to the dense boundary layer.

**Tier 2 — one-shot calibration (`--auto-tune`).** Measure once at load with a few dummy tokens:
`t_gpu_layer`, `t_cpu_layer`, `t_xfer`. Per-token time for a contiguous split at N is
`N·t_gpu_layer + (L−N)·t_cpu_layer + t_xfer` — monotonic in N, so push N to the VRAM ceiling.
Calibration's real value is confirming the GPU is actually faster per layer (on a weak iGPU over a
slow bus it may not be) and comparing strategies on measured numbers instead of the crude
file-size-÷-layer-count estimate.

**Tier 3 — empirical sweep.** Extend `autosearch.sh` (already coordinate-ascent over kernel flags +
PPL) to sweep `gpu-layers` and the KV-quant mode, keeping the Pareto-best (tok/s, PPL). Slowest, most
reliable; cache the winning config per (model, machine).

## 5. CPU utilisation heuristics

The CPU half is bandwidth-bound, so the wins are about feeding memory, not adding compute:

- **Use physical cores, not logical.** Size the matmul pool to physical cores, not
  `availableProcessors()`. Hyperthreads share one core's load/store ports; the second thread adds
  contention, not throughput. Often a 10–20% win.
- **NUMA locality is decisive on multi-socket / hybrid CPUs.** Pin the matmul threads and the mmap'd
  weights to one NUMA node (`numactl --cpunodebind=0 --membind=0`, or first-touch the pages on the
  owning thread). Up to 2× on a 2-socket box; steer the hot matmul onto P-cores on hybrid CPUs.
- **SIMD width is free.** LLMPlayer's B2I/I2F kernels exploit AVX2; add a 16-lane branch for AVX-512.
- **Quant of CPU-resident tensors is a bandwidth knob.** The CPU half reads `bytes/BW_cpu`; prefer
  keeping the already-low-bit tensors on the CPU and offloading the higher-bit ones to the GPU.
- **No CPU↔GPU overlap within a token.** Layers form a dependency chain, so the GPU and CPU blocks
  run serially in one forward pass. The only real overlap is server-level batching across requests.

## 6. GPU utilisation heuristics (single and multiple GPUs)

**Single GPU — make every VRAM byte count:** quantise the KV cache (`-Dcuda.kv.fp16`, shipped — half
the KV bytes become more weight layers and attention reads halve at long context; Q8/Q4_1 go further
on MLA); keep dp4a/int8 and CUDA graph on (default); never put the embedding table on GPU (a lookup).

**Multiple GPUs.** LLMPlayer is single-device today. Two models:

- **Layer-split / pipeline parallel (recommended for consumer cards over PCIe).** Assign a contiguous
  block of layers to each GPU, sized to each card's VRAM. The activation crosses GPU→GPU once per
  boundary (P2P DMA or a host bounce). Needs no fast interconnect, tolerates **heterogeneous** GPUs,
  and slots onto the existing per-layer residency model: generalise `gpuLayers` (count + one device)
  to a per-layer device map `int deviceOf[layer]`, with one `CudaForwardPass` instance per device
  handing its output to the next device's input buffer.
- **Tensor parallel** (split each matmul) only pays off with NVLink-class interconnect; over PCIe the
  per-layer all-reduce traffic dwarfs the gain. Skip it for consumer multi-GPU.
- **Decision rule:** sort GPUs by free VRAM, greedily assign contiguous layer blocks proportional to
  each card's VRAM (after its KV reserve), spill the remainder to CPU.

## 7. MoE — the biggest lever when the model is huge

Expert weights dominate a 30B MoE and belong on the CPU (§2). The refinement that matters most:

- **Hot-expert GPU cache (LRU by routing frequency).** Token-to-token routing has strong locality —
  a handful of experts fire far more often than the tail. Keep the top-M hottest experts resident on
  the GPU (sized to leftover VRAM after attention + KV), serve cache hits at GPU bandwidth, fall back
  to the CPU on a miss. LLMPlayer has the matmul pieces (`GraniteExpertGpu` offset-pointer for
  Q4_K/Q6_K, `ExpertGpuCache` for MXFP4); the missing piece is the frequency-tracking LRU admission
  policy sized from the VRAM budget.

## 8. Throughput vs response time

- **Prefill (TTFT)** is compute-bound (the prompt is processed as a batch) → maximal GPU residency +
  batching.
- **Decode (per-token latency)** is the bandwidth-bound regime here; for a single stream
  throughput = 1/latency, so the same placement optimises both.
- The conflict only appears in **multi-request serving**: batching concurrent decodes raises
  aggregate tokens/s but raises each request's latency. Expose a latency-first vs throughput-first
  knob.

## 9. The synthesised algorithm

```
inputs: model (per-tensor bytes, K/N), GPUs[] (free VRAM each), CPU (DRAM BW, physical cores, NUMA), ctx
1. budget each GPU:  usable_g = α·VRAM_g − driverReserve
2. reserve KV:       kvPerLayer = 2·ctx·kvDim·kvElemBytes; pick FP16 KV if it buys ≥1 layer
3. classify tensors by reads/token: {attention, dense-FFN, output}=HOT; {experts}=SPARSE; {embedding}=CPU
4. if MoE and Σ HOT ≤ Σ(usable_g − kv_g):          # KTransformers regime
     place all HOT on GPUs (fill largest-VRAM card first, contiguous), experts on CPU,
     then fill leftover VRAM with an LRU hot-expert cache
   else:                                            # dense / partial regime
     greedily assign contiguous layer blocks to GPUs by descending free VRAM (proportional),
     spill remaining layers to CPU; within the boundary layer, attention before FFN
5. CPU side: pool = physical cores; pin threads+pages to local NUMA node; AVX2/AVX-512 path
6. (optional) calibrate t_gpu_layer/t_cpu_layer/t_xfer; (optional) autosearch sweep, cache the best
```

---

## Implementation roadmap — three phases

### Phase 1 — Smarter single-GPU budgeting + CPU tuning (refine the existing path)
Self-contained refinements to the current placement and CPU code; no new subsystems; fixes a real
OOM/under-fill bug. Highest value per effort.

1. **KV-aware VRAM budget — DONE.** Folds `+ kvPerLayer` into the dense first-N-layers budget (closed
   form in §4 Tier 0) and reserves the KV cache before placing layers; auto-enables FP16 KV when FP32
   KV would not fit all layers but FP16 would. MoE-optimised is left unchanged because its KV cache
   lives on the CPU, not the GPU. Validated: Llama-1B reserves 128 MB KV; Llama-3.2-3B at ctx=16384
   accounts for 3584 MB KV (previously ignored → late VRAM over-commit); at ctx=24576 it auto-enables
   FP16 KV and fits all 28 layers.
3. **CPU physical-core + NUMA tuning — DONE.** The matmul thread pool now defaults to *physical* cores
   (`CLIOptions.detectPhysicalCores()` via Linux sysfs thread-sibling groups), not logical, because
   the bandwidth-bound matmul gains nothing from hyperthreads sharing one core's load/store ports;
   `--threads` overrides. A `numactl` pinning suggestion is logged when >1 NUMA node is detected.
   The choice rests on the bandwidth argument and llama.cpp precedent, not on a measured figure: on
   the thermally-constrained reference box the effect is **not cleanly measurable** — a single clean
   A/B showed physical-11 ≈ 5.0 vs logical-22 ≈ 3.8 tok/s, but a heat-soaked sweep was contradictory
   (physical helped some models and hurt others, with absolute numbers throttled into the 1.5–8 tok/s
   range). No single speedup figure is claimed; `--threads N` overrides for users who measure
   otherwise on their hardware.
4. **`autosearch.sh` → KV-quant sweep — DONE (local tool).** `cuda.kv.fp16`, `kv.q8`, `kv.q4` are now
   in the coordinate-ascent flag set. `--gpu-layers` is deliberately left at auto: the budget is now
   KV-aware, so "fill VRAM" is already the optimal contiguous split and there is nothing to sweep.
   (`autosearch.sh` is gitignored like the other bench scripts, so this change lives locally.)
2. **Fractional-layer attention-first placement — DEFERRED.** Splitting one boundary layer into
   attention-on-GPU + FFN-on-CPU needs the `forwardAttentionOnly` infrastructure wired into the
   loader and the GPU-resident chain (which assumes whole layers), for an estimated ~1–2% gain — at
   or below the ±15–30% thermal-noise floor on the reference box. Better revisited together with the
   Phase 2 MoE attention/FFN split, which shares the same `forwardAttentionOnly` substrate.

### Phase 2 — Measured auto-tuning + MoE hot-expert cache
1. **`--auto-tune` calibration — DONE.** `--auto-tune` measures steady-state decode tok/s for the
   heuristic GPU placement and for CPU-only, then keeps the faster — replacing the file-size guess
   with a measurement and auto-correcting the partial-fit footgun (GPU slower than CPU because too
   little fits VRAM or the GPU/bus is weak). One-time, opt-in (it loads the model a couple of extra
   times). Validated on Llama-1B: GPU 93.1 vs CPU 5.8 tok/s → GPU chosen.
2. **Routing-frequency instrumentation (Phase 2.2a) — DONE.** `-Dmoe.routing.stats=true` counts
   per-layer expert selections in `Qwen3MoEInferenceEngine` and prints, at exit, how much routing the
   top-M experts capture per layer. **Measured on Qwen3-Coder-30B (128 experts, top-8, 48 MoE
   layers, 34k selections):** top-8 = 38.5 %, top-16 = 57.3 %, **top-32 = 79.2 %**, top-64 = 96.2 %
   of routing — i.e. routing is **moderately concentrated** (top-32 captures 3.2× its uniform share,
   top-8 captures 6.4×). **Conclusion: a hot-expert GPU cache is justified** — caching the top-32 of
   128 experts per layer would serve ~79 % of expert matmuls at GPU bandwidth.
3. **Hot-expert GPU cache (Phase 2.2b) — PARTIAL.** The existing `ExpertGpuCache` (per-layer LRU,
   wired into `moeFFN`) was made **type-aware** — block geometry + matmul kernel are now passed in,
   so it is no longer MXFP4-hardcoded. Three concrete outcomes:
   - **Real bug fixed (shipped on by default):** building the cache for Q4_K exposed a latent
     Qwen3-Coder-30B crash. On some prompts the router logits go NaN; `selectTopK` then leaves a
     routing slot at `-1` (because `NaN > -inf` is false), and both the cache (negative byte offset)
     and the CPU path (`IndexOutOfBounds` at offset `-884736` = exactly one Q4_K expert slice)
     dereferenced expert `-1`. `selectTopK` now fills any unfilled slot with a valid expert at
     negligible weight, and `cpuExpertCompute` guards `e < 0`. The 30B now generates coherent output
     on prompts that previously aborted.
   - **Q4_K/Q5_K/Q6_K cache path: EXPERIMENTAL, default OFF** (`-Dmoe.expert.cache.experimental`).
     The type-aware plumbing initialises correctly (`Q4_K experts, 5681 slots` on the 30B) but the
     Q4_K path produces incorrect output. A deep investigation (a 6-agent adversarial workflow plus an
     empirical per-expert GPU-vs-CPU diagnostic, `-Dmoe.cache.debug`) **localised it precisely**: the
     GPU Q4_K expert dequant yields huge/NaN values (e.g. 1.5e6, NaN) while the CPU path is correct
     (~0.01). The following were each **ruled out**: cache orchestration / buffer sizing / result
     combination (the MXFP4 cache path is byte-identical and generates coherent output —
     `sonar-oss-20b` → "Paris"); the FP32 `matmul_q4_k` kernel itself (validated in-place via
     `-Dcuda.dp4a=false` on Llama-1B → "Paris"); the per-expert byte offset (matches
     `Q4_KFloatTensor.dot` and the CPU `expertMatmul` exactly); 4-byte/16-byte read alignment; the
     launch geometry (`getMatmulBlockDim`→256, grid `ceil(rows/8)`, smem 0 — identical to the in-place
     path); and weight repacking (`getGpuWeights` uploads raw bytes, no repack). The exact byte-level
     root cause (why the *same raw Q4_K bytes* dequant correctly in-place but as huge values from the
     cache's uploaded slot) remains open and would need GPU-memory dumping to settle. MXFP4 (GPT-OSS)
     stays validated and on. The diagnostic harness is retained behind `-Dmoe.cache.debug`.
   - **Measured neutral on models that fit RAM.** On the 30B (18 GB, fits 31 GB RAM) the cache ran at
     ~0.6–1.0 tok/s — the same as the CPU expert path — because the per-expert GPU matmul launch +
     PCIe-miss overhead roughly equals CPU SIMD for these small, numerous experts (top-8 × 48 layers
     = ~1150 tiny matmuls/token). **Its real value is for models LARGER than RAM:** there the cache
     keeps hot experts in VRAM and avoids paging them from *disk* (~100× slower than RAM), which is
     not observable on the in-RAM 30B. That benefit awaits both the K-quant correctness fix and a
     >RAM test model.

4. **Lazy mmap for models > RAM — DONE.** `LLMEngine.load` now skips the full-file preload when the
   model exceeds 85 % of physical RAM (`getPhysicalMemorySize`) and relies on lazy mmap: only the
   working set pages in on demand (read-only from the model file, never the swap partition); for MoE
   the cold experts stay on disk. `-Dno.preload=true` / `-Dpreload=true` force the choice. This is
   the foundation for running models larger than physical RAM without disk swap — combined with
   MoE-optimised placement (attention in VRAM) and, eventually, a corrected hot-expert cache, the hot
   working set stays in VRAM+RAM while cold weights remain on disk.

### Phase 3 — Multi-GPU layer-split — NOT IMPLEMENTED (no hardware)
**Parked deliberately.** The reference machine has a single GPU, so a multi-GPU layer-split (per-layer
device map, GPU→GPU activation hand-off) cannot be validated here, and shipping an untested
multi-device forward pass would be irresponsible. The design — generalise `gpuLayers` to
`deviceOf[layer]`, contiguous proportional assignment by free VRAM, one `CudaForwardPass` per device,
pipeline (not tensor) parallel — is recorded for when multi-GPU hardware is available.

### A note on benchmarking this box
CPU/throughput measurements here are unreliable for fine comparisons: the machine is **shared with
other tenants' workloads** (observed concurrent JVMs from unrelated projects) and is thermally
constrained. Contention + throttling — not the change under test — dominated the contradictory
physical-vs-logical CPU sweeps. Treat small (<2×) tok/s deltas measured here as noise.
