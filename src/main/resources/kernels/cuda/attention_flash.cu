// Flash-decoding attention for one query token (decode step), used by every GPU-resident pass.
//
// attention_full (attention.cu) ran one block per query head, kept every score in shared memory
// ((seqLen + 32) * 4 bytes, so launches failed past ~12K tokens under the 48 KB default), let one
// thread walk each K row serially (a warp's loads were kvDim-strided, i.e. uncoalesced), used only
// headSize of its 256 threads for the weighted V sum, and re-read every K/V row once per query
// head of a GQA group (4x the KV traffic on Llama-3.2).
//
// Here one block handles G query heads that share a KV head, over one slice of the sequence
// (gridDim.y slices). Inside a block each warp takes one timestep at a time: the 32 lanes read
// the K row and the V row coalesced, once for all G heads; each head's score is reduced with warp
// shuffles; an online softmax (running max m, running sum l, rescaled accumulator) avoids storing
// any score. Shared memory is fixed and small, independent of the context length.
// The last block of a head group to finish (atomic ticket per group) merges the per-slice partial
// results in the same launch:
//   M = max_s m_s,  out = sum_s exp(m_s - M) * acc_s / sum_s exp(m_s - M) * l_s
// Slices are at least FA_MIN_CHUNK timesteps long, computed on the device from the current
// sequence length: at short contexts most slices are empty and their blocks exit at once, so the
// launch configuration can stay fixed (CUDA graphs) without paying for idle splits.
//
// Math is the same as attention_full: score = scale * q.k (optionally soft-capped as
// softcap * tanh(score / softcap), Gemma 2); a sliding window masks t < position - window + 1;
// softmax; weighted sum of V. Only the floating-point summation order differs.
//
// gridDim.z tokens can be processed in one launch (batched prefill): token z reads its own
// position from tokenParams[2z], its query from q + z*qStride and writes out + z*outStride; each
// token only sees positions <= its own, so causality holds once every token's K/V is written.
// Decode launches use gridDim.z = 1 and zero strides.
//
// Kernels are instantiated for G (query heads per block) and NJ = ceil(headSize / 32) with
// G * NJ <= 32, so q and the accumulators stay in registers. Name: attention_flash_g<G>_j<NJ>
// (FP32 KV) and attention_flash_g<G>_j<NJ>_f16 (FP16 KV).

#define FA_WARPS 8
#define FA_MIN_CHUNK 64

__device__ __forceinline__ float fa_h2f(unsigned short h) {
    float r;
    asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h));
    return r;
}

__device__ __forceinline__ float fa_load(const float* p, int i) { return __ldg(p + i); }
__device__ __forceinline__ float fa_load(const unsigned short* p, int i) { return fa_h2f(__ldg(p + i)); }

// Merge the first `active` slices of G heads (only those slices hold results) into the output,
// then reset the group's ticket for the next launch (graph replay).
template <int G>
__device__ __forceinline__ void fa_combine(float* __restrict__ output, int* __restrict__ tickets,
    const float* __restrict__ partialAcc, const float* __restrict__ partialML,
    int h0, int active, int nSplits, int headSize, const float* __restrict__ sinks)
{
    for (int g = 0; g < G; g++) {
        const int h = h0 + g;
        const long hb = (long) h * nSplits;
        float M = -1e30f;
        for (int s = 0; s < active; s++) {
            if (__ldcg(&partialML[(hb + s) * 2 + 1]) > 0.0f) M = fmaxf(M, __ldcg(&partialML[(hb + s) * 2]));
        }
        float L = 0.0f;
        for (int s = 0; s < active; s++) {
            float ls = __ldcg(&partialML[(hb + s) * 2 + 1]);
            if (ls > 0.0f) L += __expf(__ldcg(&partialML[(hb + s) * 2]) - M) * ls;
        }
        float invL = (L > 0.0f) ? 1.0f / L : 0.0f;
        if (sinks != 0) {
            // Attention sink (GPT-OSS): one extra logit per head in the softmax denominator only
            float sk = sinks[h];
            float M2 = fmaxf(M, sk);
            float c = __expf(M - M2);
            invL = c / (L * c + __expf(sk - M2));
        }
        for (int d = threadIdx.x; d < headSize; d += blockDim.x) {
            float a = 0.0f;
            for (int s = 0; s < active; s++) {
                float ls = __ldcg(&partialML[(hb + s) * 2 + 1]);
                if (ls > 0.0f) a += __expf(__ldcg(&partialML[(hb + s) * 2]) - M) * __ldcg(&partialAcc[(hb + s) * headSize + d]);
            }
            output[h * headSize + d] = a * invL;
        }
    }
    if (threadIdx.x == 0) tickets[blockIdx.x] = 0;
}

// partialAcc: [headCount][nSplits][headSize] unnormalized weighted V sums
// partialML:  [headCount][nSplits][2] running max and running sum
template <int G, int NJ, typename KV>
__device__ __forceinline__ void fa_body(
    float* __restrict__ output, int* __restrict__ tickets,
    float* __restrict__ partialAcc, float* __restrict__ partialML,
    const float* __restrict__ q, const KV* __restrict__ keyCache, const KV* __restrict__ valueCache,
    int headCount, int headCountKV, int headSize, int kvDim,
    const int* __restrict__ tokenParams, int slidingWindow, int nSplits, float scale, float softcap,
    int qStride, int outStride, const float* __restrict__ sinks)
{
    // token of a batched launch (0 for decode)
    const int z = blockIdx.z;
    q += (long) z * qStride;
    output += (long) z * outStride;
    tokenParams += 2 * z;
    partialAcc += (long) z * headCount * nSplits * headSize;
    partialML += (long) z * headCount * nSplits * 2;
    tickets += z * gridDim.x;
    const int kvMul = headCount / headCountKV;
    const int groupsPerKv = kvMul / G;
    const int kvHead = blockIdx.x / groupsPerKv;
    const int h0 = kvHead * kvMul + (blockIdx.x % groupsPerKv) * G; // first query head of the block
    const int split = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    const int position = tokenParams[0];
    const int seqLen = tokenParams[1];
    const int startPos = (slidingWindow > 0) ? max(0, position - slidingWindow + 1) : 0;
    const int span = seqLen - startPos;
    const int active = max(1, min(nSplits, (span + FA_MIN_CHUNK - 1) / FA_MIN_CHUNK));
    const int chunk = (span + active - 1) / active;
    const int t0 = startPos + split * chunk;
    const int t1 = min(seqLen, t0 + chunk);
    const int kvOff = kvHead * headSize;
    __shared__ int sIsLast;

    if (split >= active) {
        // Idle slice (short context): no work, no partial results — just take a ticket, in case
        // this block is the last of its group and has to merge the active slices.
        __threadfence();
        if (threadIdx.x == 0) sIsLast = (atomicAdd(&tickets[blockIdx.x], 1) == nSplits - 1);
        __syncthreads();
        if (!sIsLast) return;
        __threadfence();
        fa_combine<G>(output, tickets, partialAcc, partialML, h0, active, nSplits, headSize, sinks);
        return;
    }

    float qr[G][NJ];
    float acc[G][NJ];
    float m[G], l[G];
    #pragma unroll
    for (int g = 0; g < G; g++) {
        m[g] = -1e30f;
        l[g] = 0.0f;
        #pragma unroll
        for (int j = 0; j < NJ; j++) {
            int d = lane + (j << 5);
            qr[g][j] = (d < headSize) ? q[(h0 + g) * headSize + d] : 0.0f;
            acc[g][j] = 0.0f;
        }
    }

    for (int t = t0 + warp; t < t1; t += FA_WARPS) {
        const KV* kRow = keyCache + (long) t * kvDim + kvOff;
        const KV* vRow = valueCache + (long) t * kvDim + kvOff;
        float kr[NJ], vr[NJ];
        #pragma unroll
        for (int j = 0; j < NJ; j++) {
            int d = lane + (j << 5);
            kr[j] = (d < headSize) ? fa_load(kRow, d) : 0.0f;
            vr[j] = (d < headSize) ? fa_load(vRow, d) : 0.0f;
        }
        #pragma unroll
        for (int g = 0; g < G; g++) {
            float s = 0.0f;
            #pragma unroll
            for (int j = 0; j < NJ; j++) s += qr[g][j] * kr[j];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xFFFFFFFF, s, off);
            s *= scale;
            if (softcap > 0.0f) s = softcap * tanhf(s / softcap);
            float mNew = fmaxf(m[g], s);
            float corr = __expf(m[g] - mNew);
            float p = __expf(s - mNew);
            l[g] = l[g] * corr + p;
            #pragma unroll
            for (int j = 0; j < NJ; j++) acc[g][j] = acc[g][j] * corr + p * vr[j];
            m[g] = mNew;
        }
    }

    // Merge the FA_WARPS per-warp states through shared memory.
    __shared__ float sM[FA_WARPS * G];
    __shared__ float sL[FA_WARPS * G];
    extern __shared__ float sAcc[]; // [FA_WARPS][G][headSize]
    #pragma unroll
    for (int g = 0; g < G; g++) {
        if (lane == 0) { sM[warp * G + g] = m[g]; sL[warp * G + g] = l[g]; }
        #pragma unroll
        for (int j = 0; j < NJ; j++) {
            int d = lane + (j << 5);
            if (d < headSize) sAcc[(warp * G + g) * headSize + d] = acc[g][j];
        }
    }
    __syncthreads();

    for (int g = 0; g < G; g++) {
        float M = -1e30f;
        #pragma unroll
        for (int w = 0; w < FA_WARPS; w++) M = fmaxf(M, sM[w * G + g]);
        const long base = (long) (h0 + g) * nSplits + split;
        for (int d = threadIdx.x; d < headSize; d += blockDim.x) {
            float a = 0.0f;
            #pragma unroll
            for (int w = 0; w < FA_WARPS; w++) a += __expf(sM[w * G + g] - M) * sAcc[(w * G + g) * headSize + d];
            partialAcc[base * headSize + d] = a;
        }
        if (threadIdx.x == 0) {
            float L = 0.0f;
            #pragma unroll
            for (int w = 0; w < FA_WARPS; w++) L += __expf(sM[w * G + g] - M) * sL[w * G + g];
            partialML[base * 2] = M;
            partialML[base * 2 + 1] = L;
        }
    }

    // Last block of this head group merges the slices (threadfence-reduction pattern).
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        int ticket = atomicAdd(&tickets[blockIdx.x], 1);
        sIsLast = (ticket == nSplits - 1);
    }
    __syncthreads();
    if (!sIsLast) return;
    __threadfence();
    fa_combine<G>(output, tickets, partialAcc, partialML, h0, active, nSplits, headSize, sinks);
}


#define FA_PARAMS_F32 float* output, int* tickets, float* partialAcc, float* partialML, const float* q, \
    const float* keyCache, const float* valueCache, \
    const int headCount, const int headCountKV, const int headSize, const int kvDim, \
    const int* tokenParams, const int slidingWindow, const int nSplits, const float scale, const float softcap, \
    const int qStride, const int outStride, const float* sinks
#define FA_PARAMS_F16 float* output, int* tickets, float* partialAcc, float* partialML, const float* q, \
    const unsigned short* keyCache, const unsigned short* valueCache, \
    const int headCount, const int headCountKV, const int headSize, const int kvDim, \
    const int* tokenParams, const int slidingWindow, const int nSplits, const float scale, const float softcap, \
    const int qStride, const int outStride, const float* sinks
#define FA_ARGS output, tickets, partialAcc, partialML, q, keyCache, valueCache, headCount, headCountKV, headSize, kvDim, \
    tokenParams, slidingWindow, nSplits, scale, softcap, qStride, outStride, sinks

#define FA_INST(G, NJ) \
    extern "C" __global__ void __launch_bounds__(FA_WARPS * 32) attention_flash_g##G##_j##NJ(FA_PARAMS_F32) \
        { fa_body<G, NJ, float>(FA_ARGS); } \
    extern "C" __global__ void __launch_bounds__(FA_WARPS * 32) attention_flash_g##G##_j##NJ##_f16(FA_PARAMS_F16) \
        { fa_body<G, NJ, unsigned short>(FA_ARGS); }

// G * NJ <= 32. NJ: 2 = headSize 64, 3 = 80/96, 4 = 128, 8 = 256, 16 = 512, 18 = 576 (latent MLA).
// The host compiles one instantiation per module by defining FA_G and FA_NJ (FlashAttention.java);
// without them every instantiation is built.
#if defined(FA_G) && defined(FA_NJ)
#define FA_INST_EXPAND(G, NJ) FA_INST(G, NJ)   // expand FA_G/FA_NJ before ## pastes them
FA_INST_EXPAND(FA_G, FA_NJ)
#else
FA_INST(1, 2) FA_INST(1, 3) FA_INST(1, 4) FA_INST(1, 8) FA_INST(1, 16) FA_INST(1, 18)
FA_INST(2, 2) FA_INST(2, 3) FA_INST(2, 4) FA_INST(2, 8) FA_INST(2, 16)
FA_INST(3, 2) FA_INST(3, 3) FA_INST(3, 4) FA_INST(3, 8)
FA_INST(4, 2) FA_INST(4, 3) FA_INST(4, 4) FA_INST(4, 8)
FA_INST(6, 2) FA_INST(6, 3) FA_INST(6, 4)
FA_INST(7, 2) FA_INST(7, 3) FA_INST(7, 4)
FA_INST(8, 2) FA_INST(8, 3) FA_INST(8, 4)
#endif

