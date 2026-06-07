// Mamba-2 SSM scan kernel — shared-memory variant.
//
// Identical math to matmul_scan.cu but explicitly stages the per-group B and C
// vectors into shared memory before the state scan loop. All headDim threads in
// a block broadcast-read the same Bg[n] and Cg[n] per state index; with smem
// the compiler can serve those broadcasts from a single fast load instead of
// relying on the L1 cache for the repeated reads.
//
// Block layout: 1 block per head, headDim threads per block.
// Shared mem:   2 * stateSize * sizeof(float) bytes.
// Dynamic smem passed by caller (so kernel is sizeSize-agnostic).
extern "C" __global__ void mamba2_scan_smem(
    float* __restrict__ S,         // [nheads * headDim * stateSize]
    const float* __restrict__ x,   // [innerSize]
    const float* __restrict__ B,   // [ngroups * stateSize]
    const float* __restrict__ C,   // [ngroups * stateSize]
    const float* __restrict__ dt,  // [nheads]
    const float* __restrict__ A,   // [nheads] (stored as -exp(A_log))
    const float* __restrict__ D,   // [nheads]
    float* __restrict__ output,    // [innerSize]
    int nheads, int headDim, int stateSize, int ngroups
) {
    extern __shared__ float smem[];
    float* sB = smem;
    float* sC = smem + stateSize;

    int h = blockIdx.x;
    int d = threadIdx.x;
    if (h >= nheads || d >= headDim) return;

    int group = h / (nheads / ngroups);

    // Cooperatively load B[group] and C[group] into shared memory.
    // headDim threads load stateSize values each — loop if stateSize > headDim.
    const float* Bg = B + group * stateSize;
    const float* Cg = C + group * stateSize;
    for (int n = d; n < stateSize; n += headDim) {
        sB[n] = Bg[n];
        sC[n] = Cg[n];
    }
    __syncthreads();

    float dtH = dt[h];
    float dA = expf(dtH * A[h]);
    float dH = D[h];
    float* Shd = S + ((long long)h * headDim + d) * stateSize;
    float x_val = x[h * headDim + d] * dtH;

    float y = 0.0f;
    #pragma unroll 4
    for (int n = 0; n < stateSize; n++) {
        float s_new = dA * Shd[n] + sB[n] * x_val;
        Shd[n] = s_new;
        y += s_new * sC[n];
    }

    output[h * headDim + d] = y + dH * x[h * headDim + d];
}
