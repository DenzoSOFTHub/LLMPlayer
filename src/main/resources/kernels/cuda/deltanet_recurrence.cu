// DeltaNet per-head recurrence kernel (optimized).
// One block per head, dV threads per block.
// Computes: L2-norm Q/K, sK = S^T @ k, S_new = alpha*S + beta*outer(k, v - alpha*sK), output = S_new^T @ q
//
// Optimizations vs naive:
// 1. Parallel L2 norm computation (warp-shuffle reduction instead of serial thread-0 loop)
// 2. Loop-invariant hoisting (betaH * diffj precomputed)
// 3. Partial loop unrolling (#pragma unroll 4)
extern "C" __global__ void deltanet_recurrence(
    float* __restrict__ S,              // [nHeads * dQK * dV] state matrices (in-place update)
    const float* __restrict__ qkv,      // [qkvDim] after conv+SiLU
    const float* __restrict__ alphaArr, // [nHeads] per-head alpha (exp(decay))
    const float* __restrict__ betaArr,  // [nHeads] per-head beta (sigmoid)
    float* __restrict__ output,         // [nHeads * dV] output
    int nHeads, int groupCount, int dQK, int dV
) {
    int head = blockIdx.x;
    int j = threadIdx.x;  // column index in S matrix
    if (head >= nHeads || j >= dV) return;

    int group = head % groupCount;
    float alphaH = alphaArr[head];
    float betaH = betaArr[head];

    int qSize = groupCount * dQK;

    // Shared memory layout: [Q(dQK)] [K(dQK)] [reduction(numWarps*2)]
    extern __shared__ float shared[];
    float* q_sh = shared;
    float* k_sh = shared + dQK;
    int numWarps = blockDim.x / 32;
    float* reduce_q = shared + 2 * dQK;          // [numWarps] for Q norm
    float* reduce_k = shared + 2 * dQK + numWarps; // [numWarps] for K norm

    // Cooperatively load Q and K (threads 0..dQK-1)
    if (j < dQK) {
        q_sh[j] = qkv[group * dQK + j];
        k_sh[j] = qkv[qSize + group * dQK + j];
    }
    __syncthreads();

    // === Parallel L2 norm via warp-shuffle reduction ===
    // Each thread computes partial sum for one element (if j < dQK)
    float my_q2 = (j < dQK) ? q_sh[j] * q_sh[j] : 0.0f;
    float my_k2 = (j < dQK) ? k_sh[j] * k_sh[j] : 0.0f;

    // Intra-warp reduction
    unsigned mask = 0xFFFFFFFF;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        my_q2 += __shfl_down_sync(mask, my_q2, offset);
        my_k2 += __shfl_down_sync(mask, my_k2, offset);
    }

    // Lane 0 of each warp writes to shared
    int warpId = j / 32;
    int laneId = j % 32;
    if (laneId == 0) {
        reduce_q[warpId] = my_q2;
        reduce_k[warpId] = my_k2;
    }
    __syncthreads();

    // Final reduction in first warp
    float qLen = 0.0f, kLen = 0.0f;
    if (j < numWarps) {
        qLen = reduce_q[j];
        kLen = reduce_k[j];
    }
    // Reduce across first numWarps lanes
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        qLen += __shfl_down_sync(mask, qLen, offset);
        kLen += __shfl_down_sync(mask, kLen, offset);
    }

    // Broadcast scales from thread 0 via shared memory
    if (j == 0) {
        reduce_q[0] = rsqrtf(qLen + 1e-12f) * rsqrtf((float)dQK); // Q: normalize + scale
        reduce_k[0] = rsqrtf(kLen + 1e-12f);                       // K: normalize only
    }
    __syncthreads();

    float qScale = reduce_q[0];
    float kScale = reduce_k[0];

    // Apply normalization in-place
    if (j < dQK) {
        q_sh[j] *= qScale;
        k_sh[j] *= kScale;
    }
    __syncthreads();

    // Get V[j] for this head
    int vOffset = qSize + qSize;
    float vj = qkv[vOffset + head * dV + j];

    // Pointer to this head's state matrix S[dQK][dV]
    float* Sh = S + (long long)head * dQK * dV;

    // Step 1: Compute sK[j] = sum_i S[i,j] * k[i]
    float sKj = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < dQK; i++) {
        sKj += Sh[i * dV + j] * k_sh[i];
    }

    // Step 2: Precompute loop-invariant terms
    float diffj = vj - alphaH * sKj;
    float beta_diff = betaH * diffj;  // hoisted from inner loop

    // Step 3: Update S[i,j] and compute output[j] = S_new^T @ q in one pass
    float out_j = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < dQK; i++) {
        float s_new = alphaH * Sh[i * dV + j] + beta_diff * k_sh[i];
        Sh[i * dV + j] = s_new;
        out_j += s_new * q_sh[i];
    }

    output[head * dV + j] = out_j;
}
