/**
 * Per-head RMSNorm kernel for QK-norm (Qwen3, Gemma3).
 * Each block processes one head: normalizes vec[h*headSize .. (h+1)*headSize-1].
 * Grid: nHeads blocks, each block handles one head independently.
 * Shared memory: (numWarps + 1) floats for warp reduction + broadcast.
 */
extern "C" __global__ void rmsnorm_per_head(
    float* vec,
    const float* weights,
    const int headSize,
    const float eps)
{
    extern __shared__ float smem[];
    int head = blockIdx.x;
    int offset = head * headSize;
    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x / 32;
    int numWarps = blockDim.x / 32;

    // Phase 1: sum of squares within this head
    float ss = 0.0f;
    for (int i = threadIdx.x; i < headSize; i += blockDim.x) {
        float v = vec[offset + i];
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
        if (lane == 0) smem[numWarps] = rsqrtf(ss / (float)headSize + eps);
    }
    __syncthreads();

    float scale = smem[numWarps];

    // Phase 2: normalize in-place
    for (int i = threadIdx.x; i < headSize; i += blockDim.x) {
        vec[offset + i] = vec[offset + i] * scale * weights[i];
    }
}
