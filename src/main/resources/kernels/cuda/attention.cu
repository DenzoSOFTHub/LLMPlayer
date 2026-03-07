/**
 * Full attention kernel: scores + softmax + weighted V sum.
 * One block per query head. Threads cooperate on time steps.
 *
 * For GQA: multiple query heads share the same KV head (kvMul = headCount / headCountKV).
 * Uses shared memory for attention scores (requires maxSeqLen + 32 floats).
 *
 * tokenParams[0] = position, tokenParams[1] = seqLen (from GPU global memory for graph compat).
 */
extern "C" __global__ void attention_full(
    float* output,              // [headCount * headSize] — attention output (xb2)
    const float* q,             // [headCount * headSize] — query vectors
    const float* keyCache,      // [maxSeqLen * kvDim] — key cache for this layer
    const float* valueCache,    // [maxSeqLen * kvDim] — value cache for this layer
    const int headCount,
    const int headCountKV,
    const int headSize,
    const int kvDim,
    const int* tokenParams)     // GPU buffer: tokenParams[0] = position, tokenParams[1] = seqLen
{
    int h = blockIdx.x;
    if (h >= headCount) return;

    int seqLen = tokenParams[1];

    int kvMul = headCount / headCountKV;
    int kvHead = h / kvMul;
    int qOffset = h * headSize;
    float scaleFactor = rsqrtf((float)headSize);

    // Shared memory: [seqLen] for att scores, [32] for warp reduction
    extern __shared__ float sharedMem[];
    float* att = sharedMem;
    float* warpReduce = att + seqLen;

    int warpId = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int numWarps = blockDim.x / 32;

    // === Step 1: Compute attention scores Q·K^T ===
    for (int t = threadIdx.x; t < seqLen; t += blockDim.x) {
        float score = 0.0f;
        int kOffset = t * kvDim + kvHead * headSize;
        for (int i = 0; i < headSize; i++) {
            score += q[qOffset + i] * keyCache[kOffset + i];
        }
        att[t] = score * scaleFactor;
    }
    __syncthreads();

    // === Step 2: Softmax ===
    // 2a. Find max
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
    // Broadcast
    __shared__ float globalMax;
    if (threadIdx.x == 0) globalMax = maxVal;
    __syncthreads();
    maxVal = globalMax;

    // 2b. Exp and sum
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

    // Normalize
    float invSum = 1.0f / globalSum;
    for (int t = threadIdx.x; t < seqLen; t += blockDim.x) {
        att[t] *= invSum;
    }
    __syncthreads();

    // === Step 3: Weighted V sum ===
    int outOffset = h * headSize;
    for (int i = threadIdx.x; i < headSize; i += blockDim.x) {
        float val = 0.0f;
        for (int t = 0; t < seqLen; t++) {
            val += att[t] * valueCache[t * kvDim + kvHead * headSize + i];
        }
        output[outOffset + i] = val;
    }
}

/**
 * KV cache update: copy K and V vectors to their cache positions.
 * tokenParams[0] = position (from GPU global memory for graph compat).
 */
extern "C" __global__ void kv_cache_update(
    float* keyCache,        // [maxSeqLen * kvDim]
    float* valueCache,      // [maxSeqLen * kvDim]
    const float* k,         // [kvDim]
    const float* v,         // [kvDim]
    const int kvDim,
    const int* tokenParams) // GPU buffer: tokenParams[0] = position
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kvDim) return;
    int position = tokenParams[0];
    int cacheOffset = position * kvDim + i;
    keyCache[cacheOffset] = k[i];
    valueCache[cacheOffset] = v[i];
}
