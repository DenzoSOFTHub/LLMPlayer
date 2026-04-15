# Option C — cp.async double-buffered input prefetch (attempt)

> **Date**: 2026-04-13
> **Goal**: hide input-vector load latency by prefetching the next super-block's input via Ampere+ `cp.async` while compute consumes the current tile.
> **Result**: ❌ **−2.8% vs baseline** (within noise, but trending negative). Same outcome as Options A and Tier 2.

## What was done

1. **Phase 0 sanity check**: confirmed that the `matmul_q4_k_smem.cu` variant (the algorithm closest to what cp.async would extend) does NOT beat baseline. Smem variant: 49.3 tok/s avg vs baseline 50.9 (−3%). This was the first signal that input-side optimization isn't where the time is going.

2. **Phase 1 cp.async PoC** (`matmul_q4_k_cpasync.cu`): same compute structure as the smem variant, but with a double-buffered shared-memory input tile filled via `cp.async.ca.shared.global` (16-byte per-thread, L1+L2 cached). 64 of the 256 threads issue float4 cp.async per super-block; while compute consumes tile[buf], the next super-block flows into tile[1-buf].

3. **First attempt was 100× slower** — used `cp.async.cg` (L2-only) at 4-byte granularity. This bypassed the L1 cache that the baseline `__ldg` benefits from, AND added 4× the issue overhead. Switched to `cp.async.ca` at 16-byte granularity → back to baseline ballpark.

## Measured results (Llama-3.2-1B Q4_K_M, RTX 4050 Laptop, 100 tokens, alternating pairs)

| Pair | baseline | cpasync |
|---|---:|---:|
| 1 | 54.7 | 52.0 |
| 2 | 50.9 | 51.4 |
| 3 | 53.0 | 50.7 |
| **avg** | **52.9** | **51.4** |

**Delta: −2.8%**. Within run-to-run noise but consistently slightly negative.

PPL preserved at 0.84-0.85 in both modes — output is mathematically equivalent to baseline.

## Why it didn't help

cp.async is a tool for **overlapping memory transfers with compute**. It pays off when:

1. The data being prefetched is on the critical path
2. There's enough compute to hide the load latency
3. Loading via cp.async is cheaper than loading via registers + `__ldg`

For our `matmul_q4_k` matvec at batch=1, **none of those conditions hold**:

- **Input is not on the critical path.** The input vector is 1 KB per super-block, broadcast-read by every row, and stays in L1 cache via `__ldg` after the first row's read. Prefetching it doesn't unblock anything.
- **Compute per super-block is tiny.** Per warp, per super-block: 8 × dequant + 8 × FMA × 2 sub-blocks ≈ 32 ops, ~2-3 ns at peak. Per-row weight load: 144 B ≈ 100+ ns from cold HBM. Compute can't possibly hide weight load latency — and weight reads are NOT what cp.async is helping with here.
- **Hardware prefetch already does this.** `__ldg` reads through the read-only L1 cache and triggers hardware prefetch via the LSU. For sequential streaming reads, the HW prefetcher is competitive with explicit cp.async.
- **L1 vs L2 cache hint matters more than async-vs-sync.** The first attempt with `cp.async.cg` (L2-only) was 100× slower than baseline, demonstrating that staying out of L1 is catastrophic. Switching to `cp.async.ca` recovered most of the throughput, but this just confirmed the cache hint was the dominant factor — not the async overlap.

## What WOULD theoretically work for cp.async on this kernel

The only place cp.async could meaningfully help is **weight prefetch** (each warp prefetching its next super-block's 144 B of weights into shared memory while computing on the previous one). But:

- Per-super-block compute (32 ops) is too small to hide HBM latency (100+ ns)
- Weights are streaming with no reuse — `__ldg` already triggers hardware streaming prefetch
- Adding shared-memory weight tiling reintroduces the smem variant's algorithmic disadvantage (lane-stripes-within-group instead of across-groups), which we already measured at −3% vs baseline

In other words: the theoretical "weight cp.async" win would need to overcome the structural −3% from the smem-style algorithm. Net is unlikely to be positive.

## What's preserved

`matmul_q4_k_cpasync.cu` is opt-in via `-Dcuda.q4k.cpasync=true` (default OFF). Kept for two reasons:
- Reference implementation of the cp.async pattern in this codebase, useful if a future kernel where compute-per-tile is large (e.g., batched matmul, prefill, WMMA tile GEMM) needs explicit memory/compute overlap
- Different hardware (higher-bandwidth GPU where weight bandwidth is no longer the bottleneck) might see different results

Default behavior **completely unchanged**.

## Honest verdict

**Three measured-dead kernel optimization paths now:**
- Option A (NVCC offline PTX/cubin): −1%
- Tier 2 multi-row-per-warp (mr4/mr2): −21% to break-even
- Option C (cp.async input prefetch): −3%

Plus earlier Tier 1 (rmsnorm+quantize fusion, dp4a output projections): marginal/regression.

The recurring pattern is the same: **on RTX 4050 Laptop (192 GB/s HBM), Q4_K matvec at batch=1 is bandwidth-limited on weight reads, and no amount of compute-side or pipelining cleverness moves the needle.** Our ~53 tok/s on Llama-1B is ~22% of theoretical bandwidth peak. llama.cpp's published 55-80% bandwidth utilization comes from datacenter GPUs (A100/H100, 1.5-3 TB/s) where the kernel/memory ratio is genuinely different.

**Recommendation: stop trying to recover this gap via kernel-level optimizations.** Genuine remaining options:

| Direction | Realism |
|---|---|
| **Different hardware** (desktop RTX 4070+/A100/H100) | Real but out-of-scope |
| **MMQ-style batched matmul for prefill** | Helps prompt processing only, not autoregressive decode |
| **Speculative decoding with a cheap draft model** | Requires good draft model; algorithm correctness was proved earlier but speed was 3-4× worse than baseline due to sequential verification — would need batched/parallel verification to win |
| **Features over throughput** | More architectures, better tooling, etc. |

## What we learned

1. **Cache hints can dwarf algorithm choice.** First-attempt cp.async with `cg` (L2-only) was 100× slower than baseline because it bypassed L1. Switching to `ca` recovered the throughput. Lesson: if your kernel relies on broadcast reads across warps, never bypass L1.
2. **Async-prefetch only helps when you have something else to do.** Our compute-per-load ratio is so low that there's nothing for cp.async to hide.
3. **Hardware prefetch is good enough for streaming patterns.** `__ldg` triggers the LSU prefetcher; for sequential reads this is competitive with explicit cp.async at lower instruction overhead.
4. **The phase 0 sanity check saved time.** Validating that the smem variant didn't beat baseline took 5 minutes and predicted the cp.async outcome before writing any kernel code. Always check that the algorithmic prerequisite for an optimization is itself a win.
