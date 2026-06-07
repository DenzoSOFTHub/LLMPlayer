// FP16 KV-cache variants (opt-in via -Dcuda.kv.fp16). K/V stored as 16-bit half (raw unsigned
// short, half the bandwidth); Q, scores and output stay FP32. Halving the KV read traffic helps
// the attention step, which is DRAM-bandwidth bound at longer contexts.
//
// Conversions use inline-PTX hardware cvt instructions (no cuda_fp16.h dependency, so NVRTC needs
// no CUDA-toolkit include path — keeps LLMPlayer driver-only).

__device__ __forceinline__ float h2f(unsigned short h) {
    float r;
    asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h));
    return r;
}
__device__ __forceinline__ unsigned short f2h(float f) {
    unsigned short r;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r) : "f"(f));
    return r;
}

extern "C" __global__ void kv_cache_update_f16(
    unsigned short* keyCache,    // [maxSeqLen * kvDim] FP16
    unsigned short* valueCache,  // [maxSeqLen * kvDim] FP16
    const float* k,              // [kvDim] FP32
    const float* v,              // [kvDim] FP32
    const int kvDim,
    const int* tokenParams)      // tokenParams[0] = position
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kvDim) return;
    int cacheOffset = tokenParams[0] * kvDim + i;
    keyCache[cacheOffset] = f2h(k[i]);
    valueCache[cacheOffset] = f2h(v[i]);
}

extern "C" __global__ void attention_full_f16(
    float* output,                       // [headCount * headSize] FP32
    const float* q,                      // [headCount * headSize] FP32
    const unsigned short* keyCache,      // [maxSeqLen * kvDim] FP16
    const unsigned short* valueCache,    // [maxSeqLen * kvDim] FP16
    const int headCount,
    const int headCountKV,
    const int headSize,
    const int kvDim,
    const int* tokenParams,              // [0]=position, [1]=seqLen
    const int slidingWindow)             // 0 = full attention, >0 = window size
{
    int h = blockIdx.x;
    if (h >= headCount) return;

    int seqLen = tokenParams[1];
    int position = tokenParams[0];
    int startPos = (slidingWindow > 0) ? max(0, position - slidingWindow + 1) : 0;

    int kvMul = headCount / headCountKV;
    int kvHead = h / kvMul;
    int qOffset = h * headSize;
    float scaleFactor = rsqrtf((float)headSize);

    extern __shared__ float sharedMem[];
    float* att = sharedMem;
    float* warpReduce = att + seqLen;

    int warpId = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int numWarps = blockDim.x / 32;

    // Step 1: scores Q·K^T
    for (int t = threadIdx.x; t < seqLen; t += blockDim.x) {
        if (t < startPos) { att[t] = -1e38f; continue; }
        float score = 0.0f;
        int kOffset = t * kvDim + kvHead * headSize;
        for (int i = 0; i < headSize; i++) {
            score += q[qOffset + i] * h2f(keyCache[kOffset + i]);
        }
        att[t] = score * scaleFactor;
    }
    __syncthreads();

    // Step 2: softmax
    float maxVal = -1e38f;
    for (int t = threadIdx.x; t < seqLen; t += blockDim.x) {
        if (att[t] > maxVal) maxVal = att[t];
    }
    for (int off = 16; off > 0; off >>= 1)
        maxVal = fmaxf(maxVal, __shfl_down_sync(0xFFFFFFFF, maxVal, off));
    if (lane == 0) warpReduce[warpId] = maxVal;
    __syncthreads();
    if (warpId == 0) {
        maxVal = (lane < numWarps) ? warpReduce[lane] : -1e38f;
        for (int off = 16; off > 0; off >>= 1)
            maxVal = fmaxf(maxVal, __shfl_down_sync(0xFFFFFFFF, maxVal, off));
    }
    __shared__ float globalMax;
    if (threadIdx.x == 0) globalMax = maxVal;
    __syncthreads();
    maxVal = globalMax;

    float sumExp = 0.0f;
    for (int t = threadIdx.x; t < seqLen; t += blockDim.x) {
        float v = __expf(att[t] - maxVal);
        att[t] = v;
        sumExp += v;
    }
    for (int off = 16; off > 0; off >>= 1)
        sumExp += __shfl_down_sync(0xFFFFFFFF, sumExp, off);
    if (lane == 0) warpReduce[warpId] = sumExp;
    __syncthreads();
    if (warpId == 0) {
        sumExp = (lane < numWarps) ? warpReduce[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            sumExp += __shfl_down_sync(0xFFFFFFFF, sumExp, off);
    }
    __shared__ float globalSum;
    if (threadIdx.x == 0) globalSum = sumExp;
    __syncthreads();

    float invSum = 1.0f / globalSum;
    for (int t = threadIdx.x; t < seqLen; t += blockDim.x) {
        att[t] *= invSum;
    }
    __syncthreads();

    // Step 3: weighted V sum
    int outOffset = h * headSize;
    for (int i = threadIdx.x; i < headSize; i += blockDim.x) {
        float val = 0.0f;
        for (int t = startPos; t < seqLen; t++) {
            val += att[t] * h2f(valueCache[t * kvDim + kvHead * headSize + i]);
        }
        output[outOffset + i] = val;
    }
}
