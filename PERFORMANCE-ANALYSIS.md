# Performance Analysis: Can We Reach 100 tok/s?

**Date**: 2026-03-07
**Hardware**: Intel Core Ultra 7 155H + NVIDIA RTX 4050 Laptop GPU (6140 MB VRAM, 192 GB/s peak bandwidth)
**Model**: Llama-3.2-1B-Instruct Q4_K_M (770 MB, 16 layers, dim=2048, vocab=128256)
**Runtime**: Java 25, Panama FFM (zero external dependencies), CUDA graph mode

## 1. Current Performance

| Configuration | tok/s | ms/tok |
|--------------|-------|--------|
| CUDA graph mode, 200 tokens, sampling | 53-56 | 17.9-18.9 |
| CUDA graph mode, 200 tokens, greedy | 55-56 | 17.9-18.2 |
| With `-Dcuda.profile=true` (sync barriers) | 40-47 | 21-25 |

## 2. Per-Token Time Breakdown

Measured with per-section instrumentation (`-Ddebug.timing=true`).

| Component | ms/tok | % of total |
|-----------|--------|------------|
| **GPU forward (graph launch + download)** | **16.9** | **95.4%** |
| Embedding lookup (CPU, 2048 floats from mmap) | 0.18 | 1.0% |
| Upload embedding (8 KB via cuMemcpyHtoD) | 0.12 | 0.7% |
| Update token params (8 bytes via cuMemcpyHtoD) | 0.10 | 0.6% |
| Sampling (argmax/top-K over 128K logits) | 0.21 | 1.2% |
| Token decode + string ops | 0.14 | 0.8% |
| Sparse logits copy (every 10th token) | 0.02 | 0.1% |
| **Total** | **17.7** | **100%** |

### GPU Forward Sub-Decomposition

The 16.9 ms "graph" call consists of:

| Sub-component | ms (estimated) |
|---------------|----------------|
| GPU kernel execution (all layers + output) | ~13.0 |
| Panama FFM overhead (4 native function calls) | ~2.0 |
| Logits download (512 KB cuMemcpyDtoH, synchronous) | ~0.5 |
| MemorySegment.copy (native to Java float[], 512 KB) | ~0.3 |
| CUDA driver overhead (graph dispatch, DMA setup) | ~1.1 |
| **Total** | **~16.9** |

## 3. GPU Compute Breakdown (Profiled)

Per-token averages across 16 transformer layers, measured with `cudaContext.finish()` sync barriers between sections.

| Section | ms/tok | Weight data read | Eff. bandwidth | % of 192 GB/s |
|---------|--------|------------------|----------------|----------------|
| GateUp (2x Q4_K matmul, 8192x2048 each) | 5.6 | 288 MB | 51.4 GB/s | **27%** |
| siluDown (SiLU + Q4_K matmul, 2048x8192) | 3.5 | 144 MB | 41.1 GB/s | 21% |
| Output (Q6_K matmul, 128256x2048) | 2.7 | 210 MB | 77.8 GB/s | **41%** |
| QKV (3x Q4_K matmul, Q=4096x2048, K/V=512x2048) | 1.8 | 54 MB | 30.0 GB/s | 16% |
| Wo (Q4_K matmul, 2048x4096) | 1.2 | 36 MB | 30.0 GB/s | 16% |
| Attention (RoPE + KV cache + softmax scores) | 2.4 | ~3 MB | (compute-bound) | -- |
| Norms (RMSNorm, 2x per layer) | 1.3 | ~1 MB | (launch overhead) | -- |
| Upload embedding | 0.1 | -- | -- | -- |
| **TOTAL (profiled, with sync barriers)** | **~15** | **736 MB** | **~50 GB/s** | **26%** |

### Bandwidth Utilization Analysis

- **Best section**: Output Q6_K matmul at **41%** -- benefits from massive parallelism (128K output rows)
- **Worst section**: QKV/Wo at **16%** -- K and V projections have only 512 output rows, not enough warps to saturate 20 SMs
- **Largest section**: GateUp at **27%** -- 8192 rows should provide enough parallelism, but the 1-warp-per-row kernel doesn't fully utilize memory bandwidth
- **Overall**: **26-31%** of peak bandwidth (profiled vs graph mode)

## 4. What Would 100 tok/s Require?

| Target | Required value | Current value | Gap |
|--------|---------------|---------------|-----|
| Total per-token time | <= 10.0 ms | 17.7 ms | -44% |
| GPU compute time | <= 7.5 ms | ~13.0 ms | -42% |
| Memory bandwidth utilization | ~55% (105 GB/s) | ~31% (60 GB/s) | +77% |
| Non-GPU overhead | <= 2.5 ms | ~4.7 ms | -47% |

### Theoretical Limits

| Metric | Value |
|--------|-------|
| Total weight data read per token | 736 MB |
| RTX 4050 Laptop peak memory bandwidth | 192 GB/s |
| Theoretical minimum at 100% utilization | 3.83 ms = **261 tok/s** |
| At 55% utilization (100 tok/s target) | 7.0 ms GPU + 3.0 ms overhead = 10.0 ms |
| At 31% utilization (current) | 13.0 ms GPU + 4.7 ms overhead = 17.7 ms |

## 5. Optimizations Tested and Results

### 5.1 Coalesced Q4_K Kernel (v2) -- REJECTED

**Hypothesis**: Restructure the Q4_K kernel so all 32 warp threads process the same group simultaneously (like the Q6_K kernel that achieved 3x speedup).

**Implementation**: All 32 lanes process the same group with coalesced input reads (`input[base + lane]`) and broadcast weight reads (`weights[bo + wordIdx*4]` where wordIdx = lane/4). No warp divergence on scale decode (branch is uniform across all threads).

**Result**: GateUp went from 5.0-5.6 ms to **6.3-6.8 ms (-21%)**. Overall 40.5 tok/s vs 46.7 tok/s.

**Why it failed**: Q4_K blocks are 144 bytes (4-byte aligned), so the default kernel uses efficient `uint32` and `float4` vectorized loads. The coalesced approach replaces these with single-float reads, losing instruction-level parallelism. For Q6_K (210-byte blocks, NOT 4-byte aligned), vectorized loads are impossible, so coalescing helps. For Q4_K, ILP wins.

### 5.2 Block Size Tuning -- NO EFFECT

Tested CUDA block sizes of 128, 256 (default), and 512 threads per block.

| Block size | Warps/block | tok/s |
|------------|-------------|-------|
| 128 | 4 | 55.8 |
| 256 | 8 | 55.4 |
| 512 | 16 | 55.2 |

All within noise. The GPU has enough parallelism with 8192 rows (gate/up) to saturate SMs regardless of blocking.

### 5.3 Prior Optimizations (from previous sessions)

| Optimization | Before | After | Impact |
|-------------|--------|-------|--------|
| Coalesced Q6_K kernel | 34.0 tok/s | 47.9 tok/s | **+41%** (output matmul 8.3 -> 2.7 ms) |
| CUDA graph mode | ~45 tok/s | ~55 tok/s | **+22%** (eliminates per-launch Panama overhead) |
| Reflection Method caching | ~45 tok/s | ~53 tok/s | **+18%** (eliminates per-token getMethod) |
| Sampler rewrite (quickselect, sparse softmax) | ~50 tok/s | ~53 tok/s | **+6%** |
| Sparse logits history | ~50 tok/s | ~53 tok/s | **+3%** (every 10th token) |
| Cached kernel CUfunction | measured | measured | eliminates hashmap lookup per matmul |
| Removed redundant finish() | measured | measured | cuMemcpyDtoH is already synchronous |

### 5.4 Prior Optimizations That Didn't Help

| Approach | Result | Reason |
|----------|--------|--------|
| Shared memory cooperative loading | No improvement | `__syncthreads` overhead outweighed coalescing benefit |
| Concurrent CUDA streams (gate+up) | No improvement | `recordEvent`+`streamWaitEvent` sync overhead negated concurrency |
| Per-token Arena reuse | No measurable benefit | `Arena.ofConfined()` allocation is already fast |
| Q4_K coalesced kernel (v1, scalar reads) | -8% | Same issue as v2: lost ILP from vectorized loads |
| CUDA unified/managed memory | Massive regression | PCIe page faults (12 GB/s) vs native VRAM (192 GB/s) |
| CUDA host-mapped/zero-copy memory | Massive regression | Same PCIe bottleneck |

## 6. Root Causes of the Performance Gap

### 6.1 Q4_K Kernel Memory Access Pattern

The 1-warp-per-row Q4_K kernel stripes threads across groups within each Q4_K block. For `cols=2048`, each of 32 lanes processes a different group in a different Q4_K block. Weight addresses are strided by `group*32` bytes (32-byte stride between groups). This is NOT coalesced -- different lanes trigger separate 128-byte cache line fetches.

However, `__ldg` (texture cache) and L1/L2 caching mitigate this because:
- All weight data for one row (1152 bytes across 8 Q4_K blocks) fits in L2
- Input vector (8 KB) stays in L2 across all rows (shared read pattern)
- `float4` vectorized loads provide 4x ILP per read instruction

The net result is ~27% effective bandwidth -- better than un-cached strided access, but far from the coalesced 41% achieved by Q6_K.

### 6.2 Small Matmul Low Parallelism

The K and V projections have only 512 output rows = 64 blocks at 256 threads/block. With 20 SMs, each SM gets only ~3 concurrent blocks (24/64 warp slots = 37% occupancy). Not enough active warps to hide DRAM latency, resulting in only 16% bandwidth utilization.

### 6.3 Panama FFM Per-Call Overhead

Each Panama Foreign Function call (cuMemcpyHtoD, cuGraphLaunch, cuMemcpyDtoH) involves:
- Method handle resolution (cached MethodHandle.invokeExact)
- Argument marshaling (Java types to native stack layout)
- Platform invoke bridge (JVM to native transition)
- Return value unmarshaling

Measured overhead: ~0.5 ms per call. With 4 FFM calls per token: ~2 ms total. This is a hard floor imposed by the pure-Java (zero external dependencies) design -- there's no JNI or native library to bypass Panama.

### 6.4 DRAM Page Mode Inefficiency

GDDR6 memory is organized in pages (~1-2 KB). The 1-warp-per-row pattern means different warps access different rows (different DRAM pages). The memory controller must handle requests to many pages simultaneously, causing frequent page activations. This is fundamental to matrix-vector multiply workloads.

## 7. Roadmap to Higher Performance

### Tier 1: Easy Wins (expected: 53 -> ~60 tok/s)

| Optimization | Expected savings | Effort |
|-------------|-----------------|--------|
| Combined uploadX + updateTokenParams (single cuMemcpyHtoD) | -0.1 ms | Low |
| GPU-side argmax for greedy mode (download 4B instead of 512KB) | -0.3 ms | Medium |
| Fused gate+up kernel (single launch, shared input) | -0.1 ms (graph mode) | Medium |

### Tier 2: Medium Effort (expected: 60 -> ~70-75 tok/s)

| Optimization | Expected savings | Effort |
|-------------|-----------------|--------|
| GPU-side embedding lookup (upload token ID, not 8KB vector) | -0.3 ms | Medium |
| Multi-warp K/V matmul (4 warps per row for 512-row projections) | -0.5 ms | Medium |
| Batch FFM calls (single JNI-like wrapper for upload+launch+download) | -1.0 ms | High |

### Tier 3: Major Rewrites (expected: 75 -> 85-100 tok/s)

| Optimization | Expected savings | Effort |
|-------------|-----------------|--------|
| Tiled Q4_K matmul with shared memory reduction | -2.0 ms | Very High |
| Tensor core matmul (WMMA/mma.sync for INT4 dequant) | -3.0 ms | Very High |
| JNI wrapper for CUDA calls (bypass Panama FFM) | -1.5 ms | High (breaks pure-Java design) |

### Estimated Performance Progression

| Scenario | GPU compute | Overhead | Total | tok/s |
|----------|-----------|----------|-------|-------|
| Current | 13.0 ms | 4.7 ms | 17.7 ms | 56 |
| + Tier 1 optimizations | 13.0 ms | 4.2 ms | 17.2 ms | 58 |
| + Tier 2 optimizations | 11.5 ms | 3.0 ms | 14.5 ms | 69 |
| + Tier 3 optimizations | 8.0 ms | 2.0 ms | 10.0 ms | 100 |
| Theoretical maximum | 3.8 ms | 0.5 ms | 4.3 ms | 233 |

## 8. Comparison with llama.cpp

For reference, llama.cpp on similar hardware typically achieves:
- **50-65% bandwidth utilization** (vs our 26-31%)
- **~80-100 tok/s** for Llama 1B Q4_K_M on RTX 4050

Their advantages:
- Hand-tuned CUDA kernels with years of optimization (tiled matmul, dp4a intrinsics)
- Native C++ with zero FFM/JNI overhead
- Direct CUDA API calls without marshaling
- Optimized memory access patterns (coalesced where possible, texture cache tuned)

Our advantages:
- Pure Java, zero external dependencies
- Cross-platform (CPU/OpenCL/CUDA via Panama FFM)
- No compilation step for native code
- Graceful degradation to CPU-only on any JVM

## 9. Conclusion

**100 tok/s is achievable in theory** but requires:
1. **Tiled matmul kernels** with 2x better bandwidth utilization (from 31% to ~55%)
2. **Reduced FFM overhead** through batched native calls or JNI wrapper

The current 53-56 tok/s represents a **reasonable result for a pure-Java inference engine** with simple 1-warp-per-row CUDA kernels. The architecture is sound -- the bottleneck is kernel-level optimization, not design flaws. Each major optimization step (coalesced Q6_K: +41%, CUDA graphs: +22%, reflection caching: +18%) has delivered measurable improvements, and the remaining gains require increasingly sophisticated GPU programming techniques.

The most cost-effective next step would be **Tier 2 optimizations** (multi-warp small matmuls + batched FFM calls) to reach ~70 tok/s, followed by investigating tiled matmul kernels for the largest matrices (gate, up, down) which dominate GPU compute time.
