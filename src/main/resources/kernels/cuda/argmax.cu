/**
 * GPU-side argmax kernel for greedy token sampling.
 * Finds the index of the maximum value in a float array.
 *
 * Two-phase approach:
 *   Phase 1 (argmax_partial): Each block finds its local max, writes to partial results.
 *   Phase 2 (argmax_final): Single block reduces partial results to global argmax.
 *
 * Parameters (argmax_partial):
 *   data:        input float array
 *   partialVal:  output partial max values (one per block)
 *   partialIdx:  output partial max indices (one per block)
 *   size:        total number of elements
 *
 * Parameters (argmax_final):
 *   partialVal:  input partial max values
 *   partialIdx:  input partial max indices
 *   resultIdx:   output int[1] — the global argmax index
 *   numPartials: number of partial results
 */

extern "C" __global__ void argmax_partial(
    const float* __restrict__ data,
    float* __restrict__ partialVal,
    int* __restrict__ partialIdx,
    const int size)
{
    int tid = threadIdx.x;
    int globalId = blockIdx.x * blockDim.x + tid;

    // Each thread finds its local max across strided elements
    float maxVal = -1e30f;
    int maxIdx = 0;
    for (int i = globalId; i < size; i += blockDim.x * gridDim.x) {
        float v = data[i];
        if (v > maxVal) {
            maxVal = v;
            maxIdx = i;
        }
    }

    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        float otherVal = __shfl_down_sync(0xFFFFFFFF, maxVal, offset);
        int otherIdx = __shfl_down_sync(0xFFFFFFFF, maxIdx, offset);
        if (otherVal > maxVal) {
            maxVal = otherVal;
            maxIdx = otherIdx;
        }
    }

    // Shared memory for cross-warp reduction within the block
    __shared__ float sVal[32];  // max 32 warps per block (1024 threads)
    __shared__ int sIdx[32];

    int warpId = tid / 32;
    int lane = tid & 31;
    int numWarps = blockDim.x / 32;

    if (lane == 0) {
        sVal[warpId] = maxVal;
        sIdx[warpId] = maxIdx;
    }
    __syncthreads();

    // First warp reduces across all warps
    if (warpId == 0) {
        maxVal = (lane < numWarps) ? sVal[lane] : -1e30f;
        maxIdx = (lane < numWarps) ? sIdx[lane] : 0;

        for (int offset = 16; offset > 0; offset >>= 1) {
            float otherVal = __shfl_down_sync(0xFFFFFFFF, maxVal, offset);
            int otherIdx = __shfl_down_sync(0xFFFFFFFF, maxIdx, offset);
            if (otherVal > maxVal) {
                maxVal = otherVal;
                maxIdx = otherIdx;
            }
        }

        if (lane == 0) {
            partialVal[blockIdx.x] = maxVal;
            partialIdx[blockIdx.x] = maxIdx;
        }
    }
}

extern "C" __global__ void argmax_final(
    const float* __restrict__ partialVal,
    const int* __restrict__ partialIdx,
    int* __restrict__ resultIdx,
    const int numPartials)
{
    int tid = threadIdx.x;

    float maxVal = -1e30f;
    int maxIdx = 0;
    for (int i = tid; i < numPartials; i += blockDim.x) {
        float v = partialVal[i];
        if (v > maxVal) {
            maxVal = v;
            maxIdx = partialIdx[i];
        }
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        float otherVal = __shfl_down_sync(0xFFFFFFFF, maxVal, offset);
        int otherIdx = __shfl_down_sync(0xFFFFFFFF, maxIdx, offset);
        if (otherVal > maxVal) {
            maxVal = otherVal;
            maxIdx = otherIdx;
        }
    }

    __shared__ float sVal[32];
    __shared__ int sIdx[32];
    int warpId = tid / 32;
    int lane = tid & 31;
    int numWarps = blockDim.x / 32;

    if (lane == 0) {
        sVal[warpId] = maxVal;
        sIdx[warpId] = maxIdx;
    }
    __syncthreads();

    if (warpId == 0) {
        maxVal = (lane < numWarps) ? sVal[lane] : -1e30f;
        maxIdx = (lane < numWarps) ? sIdx[lane] : 0;

        for (int offset = 16; offset > 0; offset >>= 1) {
            float otherVal = __shfl_down_sync(0xFFFFFFFF, maxVal, offset);
            int otherIdx = __shfl_down_sync(0xFFFFFFFF, maxIdx, offset);
            if (otherVal > maxVal) {
                maxVal = otherVal;
                maxIdx = otherIdx;
            }
        }

        if (lane == 0) {
            resultIdx[0] = maxIdx;
        }
    }
}
