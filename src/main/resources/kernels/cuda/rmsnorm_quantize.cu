/**
 * Fused RMSNorm + Q8_1 quantization in a single kernel.
 *
 * Replaces the two-kernel sequence used in the dp4a matmul path:
 *    1. rmsnorm_fused: x → xb (FP32 normalized)
 *    2. quantize_q8:   xb → xbQ8 (Q8_1 quantized for dp4a)
 *
 * Saves ~5-7% of total GPU time on Qwen3.5-4B by:
 *   - Eliminating one HBM round-trip for the normalized FP32 vector
 *   - Eliminating one launch (~46 µs/call × 64 calls/token = ~3 ms/token)
 *
 * Output:
 *   normOut: [size] FP32 normalized values (kept for downstream non-dp4a kernels;
 *            may be NULL to skip writing — saves ~size×4 bytes of HBM writes)
 *   qOut:    [(size/32) * 40] bytes Q8_1 blocks, layout [scale(4) | sum(4) | qs(32)]
 *
 * Launch: 1 block, blockDim.x threads (typically 256). Shared mem: (numWarps + 1) floats.
 * size MUST be a multiple of 32 (Q8_1 block size).
 */
extern "C" __global__ void rmsnorm_quantize_fused(
    float* normOut,                // [size] FP32 normalized output (or NULL to skip)
    unsigned char* qOut,           // [(size/32) * 40] Q8_1 output
    const float* x,                // [size] FP32 input
    const float* w,                // [size] FP32 RMSNorm weights
    const int size,
    const float eps)
{
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warpId = tid >> 5;
    int numWarps = blockDim.x >> 5;

    // === Phase 1: sum of squares (block reduction) ===
    float ss = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float v = x[i];
        ss += v * v;
    }
    for (int off = 16; off > 0; off >>= 1)
        ss += __shfl_down_sync(0xFFFFFFFF, ss, off);
    if (lane == 0) smem[warpId] = ss;
    __syncthreads();

    if (warpId == 0) {
        ss = (lane < numWarps) ? smem[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            ss += __shfl_down_sync(0xFFFFFFFF, ss, off);
        if (lane == 0) smem[numWarps] = rsqrtf(ss / (float)size + eps);
    }
    __syncthreads();

    float invRms = smem[numWarps];

    // === Phase 2 + 3: per Q8_1 block (32 elements) — normalize + quantize ===
    // Each warp owns blocks: {warpId, warpId + numWarps, warpId + 2*numWarps, ...}
    int numBlocks = size / 32;
    for (int b = warpId; b < numBlocks; b += numWarps) {
        int idx = b * 32 + lane;

        // Normalize
        float normed = x[idx] * invRms * w[idx];

        // Optionally write normalized FP32 (for downstream non-dp4a consumers)
        if (normOut != 0) normOut[idx] = normed;

        // === Q8_1 quantization (per warp = per Q8_1 block of 32) ===
        // Find max(|normed|) via warp-shuffle reduction
        float absVal = fabsf(normed);
        float maxAbs = absVal;
        unsigned mask = 0xFFFFFFFF;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float other = __shfl_down_sync(mask, maxAbs, off);
            maxAbs = fmaxf(maxAbs, other);
        }
        maxAbs = __shfl_sync(mask, maxAbs, 0);  // broadcast to all lanes

        // Compute scale
        float qScale = maxAbs / 127.0f;
        float invQScale = (maxAbs > 0.0f) ? (127.0f / maxAbs) : 0.0f;
        int q = __float2int_rn(normed * invQScale);
        q = max(-127, min(127, q));

        // Sum of normalized values for this block (Q8_1 stores this)
        float sum = normed;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum += __shfl_down_sync(mask, sum, off);
        }

        // Write Q8_1 block: [scale(4B)][sum(4B)][qs(32B)] = 40 bytes
        long outOffset = (long)b * 40;
        if (lane == 0) {
            *(float*)(qOut + outOffset) = qScale;
            *(float*)(qOut + outOffset + 4) = sum;
        }
        qOut[outOffset + 8 + lane] = (unsigned char)(q & 0xFF);
    }
}
