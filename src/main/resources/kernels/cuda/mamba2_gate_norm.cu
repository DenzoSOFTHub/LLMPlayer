// Fused gate + grouped RMSNorm for Mamba-2.
// norm_before_gate=False: first y *= SiLU(z), then grouped RMSNorm.
// One block per group. blockDim threads process groupSize elements.
extern "C" __global__ void mamba2_gate_norm(
    float* __restrict__ y,         // [innerSize] SSM output (in-place)
    const float* __restrict__ z,   // [innerSize] gate values
    const float* __restrict__ normW, // [innerSize] norm weights (GGUF [groupSize, ngroups] layout)
    int innerSize, int ngroups, float eps
) {
    int g = blockIdx.x;
    if (g >= ngroups) return;

    int groupSize = innerSize / ngroups;
    int off = g * groupSize;

    // Phase 1: Apply gate y *= SiLU(z) and compute sum of squares
    extern __shared__ float shared[];
    float my_ss = 0.0f;
    for (int i = threadIdx.x; i < groupSize; i += blockDim.x) {
        float zi = z[off + i];
        float gated = y[off + i] * zi / (1.0f + expf(-zi)); // y * SiLU(z)
        y[off + i] = gated;
        my_ss += gated * gated;
    }

    // Warp-reduce sum of squares
    unsigned mask = 0xFFFFFFFF;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        my_ss += __shfl_down_sync(mask, my_ss, offset);

    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    int numWarps = blockDim.x / 32;
    if (laneId == 0) shared[warpId] = my_ss;
    __syncthreads();

    float ss = 0.0f;
    if (threadIdx.x < numWarps) ss = shared[threadIdx.x];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        ss += __shfl_down_sync(mask, ss, offset);
    if (threadIdx.x == 0) shared[0] = rsqrtf(ss / (float)groupSize + eps);
    __syncthreads();

    float scale = shared[0];

    // Phase 2: Normalize with weight
    // GGUF ssm_norm [groupSize, ngroups]: element i of group g = normW[g * groupSize + i]
    for (int i = threadIdx.x; i < groupSize; i += blockDim.x) {
        y[off + i] = y[off + i] * scale * normW[g * groupSize + i];
    }
}
