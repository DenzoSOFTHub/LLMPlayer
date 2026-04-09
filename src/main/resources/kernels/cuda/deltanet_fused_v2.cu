/**
 * Optimized DeltaNet kernel v2: float4 vectorized state access.
 * Same algorithm as deltanet_fused.cu but with:
 * 1. float4 loads/stores for state array (4x fewer memory transactions)
 * 2. Aggressive unrolling of inner loops
 * 3. FMA (fused multiply-add) friendly code structure
 *
 * Enabled via -Dcuda.deltanet.v2=true
 */
extern "C" __global__ void deltanet_fused_v2(
    float* __restrict__ S,
    const float* __restrict__ qkv,
    const float* __restrict__ alphaArr,
    const float* __restrict__ betaArr,
    const float* __restrict__ gate,
    const float* __restrict__ normW,
    float* __restrict__ output,
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

    // Shared memory: Q, K, reduction buffers
    extern __shared__ float shared[];
    float* q_sh = shared;
    float* k_sh = shared + dQK;
    int numWarps = blockDim.x / 32;
    float* warp_buf = shared + 2 * dQK;

    // Load Q and K cooperatively
    if (j < dQK) {
        q_sh[j] = qkv[group * dQK + j];
        k_sh[j] = qkv[qSize + group * dQK + j];
    }
    __syncthreads();

    // === L2 norm for Q and K (unchanged) ===
    float my_q2 = (j < dQK) ? q_sh[j] * q_sh[j] : 0.0f;
    float my_k2 = (j < dQK) ? k_sh[j] * k_sh[j] : 0.0f;
    unsigned mask = 0xFFFFFFFF;
    for (int off = 16; off > 0; off >>= 1) {
        my_q2 += __shfl_down_sync(mask, my_q2, off);
        my_k2 += __shfl_down_sync(mask, my_k2, off);
    }
    int warpId = j / 32, laneId = j % 32;
    if (laneId == 0) { warp_buf[warpId] = my_q2; warp_buf[numWarps + warpId] = my_k2; }
    __syncthreads();
    float qLen = 0, kLen = 0;
    if (j < numWarps) { qLen = warp_buf[j]; kLen = warp_buf[numWarps + j]; }
    for (int off = 16; off > 0; off >>= 1) {
        qLen += __shfl_down_sync(mask, qLen, off);
        kLen += __shfl_down_sync(mask, kLen, off);
    }
    if (j == 0) {
        warp_buf[0] = rsqrtf(qLen + 1e-12f) * rsqrtf((float)dQK);
        warp_buf[1] = rsqrtf(kLen + 1e-12f);
    }
    __syncthreads();
    if (j < dQK) {
        q_sh[j] *= warp_buf[0];
        k_sh[j] *= warp_buf[1];
    }
    __syncthreads();

    // V value for this thread
    float vj = qkv[2 * qSize + head * dV + j];

    // State pointer (transposed: S_T[head][j][i])
    float* Sh = S + (long long)head * dV * dQK + (long long)j * dQK;

    // === Step 1: sK[j] = dot(S_T[j][:], k[:]) — FLOAT4 VECTORIZED ===
    float sKj = 0.0f;
    int dQK4 = dQK / 4;
    float4* Sh4 = (float4*)Sh;
    float4* k4 = (float4*)k_sh;

    #pragma unroll 8
    for (int i4 = 0; i4 < dQK4; i4++) {
        float4 s = Sh4[i4];
        float4 k = k4[i4];
        sKj += s.x * k.x + s.y * k.y + s.z * k.z + s.w * k.w;
    }

    // Precompute loop-invariant
    float beta_diff = betaH * (vj - alphaH * sKj);

    // === Step 2: Update state + compute output — FLOAT4 VECTORIZED ===
    float out_j = 0.0f;
    float4* q4 = (float4*)q_sh;

    #pragma unroll 8
    for (int i4 = 0; i4 < dQK4; i4++) {
        float4 s = Sh4[i4];
        float4 k = k4[i4];
        float4 q = q4[i4];

        // FMA-friendly: s_new = alpha * s + beta_diff * k
        float4 s_new;
        s_new.x = alphaH * s.x + beta_diff * k.x;
        s_new.y = alphaH * s.y + beta_diff * k.y;
        s_new.z = alphaH * s.z + beta_diff * k.z;
        s_new.w = alphaH * s.w + beta_diff * k.w;

        Sh4[i4] = s_new;

        out_j += s_new.x * q.x + s_new.y * q.y + s_new.z * q.z + s_new.w * q.w;
    }

    // === Fused RMSNorm (unchanged) ===
    float my_ss = out_j * out_j;
    for (int off = 16; off > 0; off >>= 1)
        my_ss += __shfl_down_sync(mask, my_ss, off);
    if (laneId == 0) warp_buf[warpId] = my_ss;
    __syncthreads();
    float ss = (j < numWarps) ? warp_buf[j] : 0.0f;
    for (int off = 16; off > 0; off >>= 1)
        ss += __shfl_down_sync(mask, ss, off);
    if (j == 0) warp_buf[0] = rsqrtf(ss / (float)dV + normEps);
    __syncthreads();

    out_j *= warp_buf[0] * normW[j];

    // === SiLU gate ===
    float g = gate[head * dV + j];
    g = g / (1.0f + expf(-g));
    output[head * dV + j] = out_j * g;
}
