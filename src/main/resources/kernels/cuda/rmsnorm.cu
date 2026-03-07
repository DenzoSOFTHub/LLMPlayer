/**
 * Fused single-kernel RMSNorm: sumsq + normalize in one launch.
 * Uses a single block with shared memory for the reduction.
 * Shared memory: numWarps + 1 floats (warp reduction + broadcast).
 */
extern "C" __global__ void rmsnorm_fused(
    float* out,
    const float* x,
    const float* w,
    const int size,
    const float eps)
{
    extern __shared__ float smem[];
    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x / 32;
    int numWarps = blockDim.x / 32;

    // Phase 1: Each thread accumulates sum of squares
    float ss = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float v = x[i];
        ss += v * v;
    }

    // Warp-level reduction
    for (int off = 16; off > 0; off >>= 1)
        ss += __shfl_down_sync(0xFFFFFFFF, ss, off);
    if (lane == 0) smem[warpId] = ss;
    __syncthreads();

    // Final reduction in first warp + broadcast
    if (warpId == 0) {
        ss = (lane < numWarps) ? smem[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            ss += __shfl_down_sync(0xFFFFFFFF, ss, off);
        if (lane == 0) smem[numWarps] = rsqrtf(ss / (float)size + eps);
    }
    __syncthreads();

    float scale = smem[numWarps];

    // Phase 2: Normalize and write output
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        out[i] = x[i] * scale * w[i];
    }
}

/**
 * RMS normalization kernel (legacy 2-phase).
 * Phase 1: Compute sum of squares (warp shuffle + single cross-warp sync).
 * Phase 2: Normalize: out[i] = x[i] * rsqrt(ss/size + eps) * w[i]
 */
extern "C" __global__ void rmsnorm_sumsq(
    const float* x,
    float* partial_sums,
    const int size)
{
    extern __shared__ float scratch[];
    int lid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = lid / 32;
    int lane = lid & 31;

    float sum = 0.0f;
    for (int i = gid; i < size; i += gridDim.x * blockDim.x) {
        sum += x[i] * x[i];
    }

    // Warp-level reduction (no __syncthreads needed)
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    // Cross-warp: each warp writes its result, then first warp reduces
    if (lane == 0) scratch[warpId] = sum;
    __syncthreads();

    if (warpId == 0) {
        int numWarps = blockDim.x / 32;
        sum = (lane < numWarps) ? scratch[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        if (lane == 0) partial_sums[blockIdx.x] = sum;
    }
}

extern "C" __global__ void rmsnorm_normalize(
    float* out,
    const float* x,
    const float* w,
    const float* partial_sums,
    const int size,
    const float eps,
    const int num_groups)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    // Sum partial sums
    float ss = 0.0f;
    for (int g = 0; g < num_groups; g++) {
        ss += partial_sums[g];
    }
    float scale = rsqrtf(ss / size + eps);
    out[i] = x[i] * scale * w[i];
}
