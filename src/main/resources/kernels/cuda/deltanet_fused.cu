// Fused DeltaNet kernel: recurrence + per-head RMSNorm + SiLU(gate) * output
// Eliminates 3 intermediate kernel launches and global memory round-trips.
//
// One block per head, dV threads per block.
// Input: QKV (after conv+SiLU), alpha/beta gates, raw gate (pre-SiLU), norm weights
// Output: final gated+normalized output ready for ssmOut projection
//
// S layout: TRANSPOSED [nHeads][dV][dQK] for coalesced access in sK computation.
// Thread j handles column j of output and row j of S_T.
extern "C" __global__ void deltanet_fused(
    float* __restrict__ S,              // [nHeads * dV * dQK] TRANSPOSED state (in-place update)
    const float* __restrict__ qkv,      // [qkvDim] after conv+SiLU
    const float* __restrict__ alphaArr, // [nHeads] per-head alpha
    const float* __restrict__ betaArr,  // [nHeads] per-head beta
    const float* __restrict__ gate,     // [nHeads * dV] raw gate output (SiLU applied inline)
    const float* __restrict__ normW,    // [dV] per-head norm weights
    float* __restrict__ output,         // [nHeads * dV] final output
    float normEps,
    int nHeads, int groupCount, int dQK, int dV
) {
    int head = blockIdx.x;
    int j = threadIdx.x;
    if (head >= nHeads || j >= dV) return;

    int group = head % groupCount;
    float alphaH = alphaArr[head];
    float betaH = betaArr[head];
    int qSize = groupCount * dQK;

    // Shared memory: [Q(dQK)] [K(dQK)] [reduction(numWarps*2)] [normReduce(numWarps)]
    extern __shared__ float shared[];
    float* q_sh = shared;
    float* k_sh = shared + dQK;
    int numWarps = blockDim.x / 32;
    float* reduce_q = shared + 2 * dQK;
    float* reduce_k = reduce_q + numWarps;
    float* reduce_norm = reduce_k + numWarps;

    // Load Q and K cooperatively
    if (j < dQK) {
        q_sh[j] = qkv[group * dQK + j];
        k_sh[j] = qkv[qSize + group * dQK + j];
    }
    __syncthreads();

    // === Parallel L2 norm for Q and K ===
    float my_q2 = (j < dQK) ? q_sh[j] * q_sh[j] : 0.0f;
    float my_k2 = (j < dQK) ? k_sh[j] * k_sh[j] : 0.0f;
    unsigned mask = 0xFFFFFFFF;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        my_q2 += __shfl_down_sync(mask, my_q2, offset);
        my_k2 += __shfl_down_sync(mask, my_k2, offset);
    }
    int warpId = j / 32;
    int laneId = j % 32;
    if (laneId == 0) {
        reduce_q[warpId] = my_q2;
        reduce_k[warpId] = my_k2;
    }
    __syncthreads();
    float qLen = 0.0f, kLen = 0.0f;
    if (j < numWarps) { qLen = reduce_q[j]; kLen = reduce_k[j]; }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        qLen += __shfl_down_sync(mask, qLen, offset);
        kLen += __shfl_down_sync(mask, kLen, offset);
    }
    if (j == 0) {
        reduce_q[0] = rsqrtf(qLen + 1e-12f) * rsqrtf((float)dQK);
        reduce_k[0] = rsqrtf(kLen + 1e-12f);
    }
    __syncthreads();
    if (j < dQK) {
        q_sh[j] *= reduce_q[0];
        k_sh[j] *= reduce_k[0];
    }
    __syncthreads();

    // Get V[j]
    int vOffset = qSize + qSize;
    float vj = qkv[vOffset + head * dV + j];

    // S is TRANSPOSED: S_T[head][j][i] = S[head][i][j]
    // Thread j owns row j of S_T, which is column j of the original S.
    float* Sh = S + (long long)head * dV * dQK + (long long)j * dQK;

    // Step 1: sK[j] = sum_i S_T[j][i] * k[i] — CONTIGUOUS access (stride-1)
    float sKj = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < dQK; i++) {
        sKj += Sh[i] * k_sh[i];
    }

    // Step 2: Precompute loop-invariant
    float beta_diff = betaH * (vj - alphaH * sKj);

    // Step 3: Update S_T[j][i] and compute output[j] in one pass
    float out_j = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < dQK; i++) {
        float s_new = alphaH * Sh[i] + beta_diff * k_sh[i];
        Sh[i] = s_new;
        out_j += s_new * q_sh[i];
    }

    // === Fused per-head RMSNorm on output ===
    // Parallel reduction for sum of squares across dV threads in this block
    float my_ss = out_j * out_j;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        my_ss += __shfl_down_sync(mask, my_ss, offset);
    }
    if (laneId == 0) reduce_norm[warpId] = my_ss;
    __syncthreads();
    float ss = 0.0f;
    if (j < numWarps) ss = reduce_norm[j];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        ss += __shfl_down_sync(mask, ss, offset);
    }
    if (j == 0) reduce_norm[0] = rsqrtf(ss / (float)dV + normEps);
    __syncthreads();
    float normScale = reduce_norm[0];

    // Apply norm + norm weights
    out_j = out_j * normScale * normW[j];

    // === Fused SiLU(gate) * output ===
    float g = gate[head * dV + j];
    g = g / (1.0f + expf(-g));  // SiLU
    out_j *= g;

    output[head * dV + j] = out_j;
}
