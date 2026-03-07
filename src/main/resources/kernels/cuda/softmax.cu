/**
 * Softmax kernel (three-pass with warp shuffle reduction).
 * Pass 1: Find max value.
 * Pass 2: Compute exp(x - max) and sum. Uses __expf for fast approximation.
 * Pass 3: Normalize by sum.
 */
extern "C" __global__ void softmax_max(
    const float* logits,
    float* partial_max,
    const int offset,
    const int size)
{
    extern __shared__ float scratch[];
    int lid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = lid / 32;
    int lane = lid & 31;

    float maxVal = -1e38f;
    for (int i = gid; i < size; i += gridDim.x * blockDim.x) {
        float v = logits[offset + i];
        if (v > maxVal) maxVal = v;
    }

    // Warp-level max reduction
    for (int off = 16; off > 0; off >>= 1)
        maxVal = fmaxf(maxVal, __shfl_down_sync(0xFFFFFFFF, maxVal, off));

    if (lane == 0) scratch[warpId] = maxVal;
    __syncthreads();

    if (warpId == 0) {
        int numWarps = blockDim.x / 32;
        maxVal = (lane < numWarps) ? scratch[lane] : -1e38f;
        for (int off = 16; off > 0; off >>= 1)
            maxVal = fmaxf(maxVal, __shfl_down_sync(0xFFFFFFFF, maxVal, off));
        if (lane == 0) partial_max[blockIdx.x] = maxVal;
    }
}

extern "C" __global__ void softmax_exp_sum(
    float* logits,
    float* partial_sums,
    const float* partial_max,
    const int offset,
    const int size,
    const int num_max_groups)
{
    extern __shared__ float scratch[];
    int lid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = lid / 32;
    int lane = lid & 31;

    // Find global max
    float maxVal = -1e38f;
    for (int g = 0; g < num_max_groups; g++) {
        if (partial_max[g] > maxVal) maxVal = partial_max[g];
    }

    float sum = 0.0f;
    for (int i = gid; i < size; i += gridDim.x * blockDim.x) {
        float v = __expf(logits[offset + i] - maxVal);
        logits[offset + i] = v;
        sum += v;
    }

    // Warp-level sum reduction
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    if (lane == 0) scratch[warpId] = sum;
    __syncthreads();

    if (warpId == 0) {
        int numWarps = blockDim.x / 32;
        sum = (lane < numWarps) ? scratch[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, off);
        if (lane == 0) partial_sums[blockIdx.x] = sum;
    }
}

extern "C" __global__ void softmax_normalize(
    float* logits,
    const float* partial_sums,
    const int offset,
    const int size,
    const int num_sum_groups)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    float sum = 0.0f;
    for (int g = 0; g < num_sum_groups; g++) {
        sum += partial_sums[g];
    }
    logits[offset + i] *= 1.0f / sum;
}
