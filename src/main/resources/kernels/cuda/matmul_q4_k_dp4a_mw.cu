/**
 * Q4_K × Q8_1 matmul, multi-warp variant: NWARPS warps cooperate on 1 row per block.
 *
 * Compared to single-warp matmul_q4_k_dp4a (which processes 1 row per warp, 8 rows per
 * block), this layout follows llama.cpp's mmvq pattern: 4 warps × 32 lanes = 128 threads,
 * 1 output row per block, gridDim = rows. Per-row weights are read by 4 warps in parallel,
 * giving 4× more in-flight HBM requests per row — better latency hiding for memory-bound
 * Q4_K matvec at batch=1.
 *
 * Per-warp shared memory partial sums + cross-warp reduction at the end.
 *
 * Layout invariants:
 *   blockDim.x = NWARPS * 32 = 128
 *   gridDim.x  = rows
 *
 * Each warp processes super-blocks strided by NWARPS:
 *   warp w handles super-blocks b = w, w+NWARPS, w+2*NWARPS, ...
 * Within a warp, lane handles groups in stride 32 just like the single-warp kernel.
 */
#define NWARPS 4

__device__ __forceinline__ float half2float_mw(unsigned short h) {
    unsigned int sign = (h >> 15) & 1;
    unsigned int exp = (h >> 10) & 0x1F;
    unsigned int mantissa = h & 0x3FF;
    if (exp == 0) {
        if (mantissa == 0) return sign ? -0.0f : 0.0f;
        while (!(mantissa & 0x400)) { mantissa <<= 1; exp--; }
        exp++; mantissa &= 0x3FF;
    } else if (exp == 31) {
        unsigned int f = (sign << 31) | 0x7F800000 | (mantissa << 13);
        return *(float*)&f;
    }
    unsigned int f = (sign << 31) | ((exp + 112) << 23) | (mantissa << 13);
    return *(float*)&f;
}

extern "C" __global__ void matmul_q4_k_dp4a_mw(
    const unsigned char* __restrict__ weights,
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    const int rows,
    const int cols,
    const int addToOutput)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    int warpId = threadIdx.x / 32;     // 0..NWARPS-1
    int lane   = threadIdx.x & 31;

    int numSuperBlocks = cols / 256;
    long rowOffset = (long)row * numSuperBlocks * 144;
    float sum = 0.0f;

    // Stride super-blocks across warps. Each warp covers strided subset.
    for (int b = warpId; b < numSuperBlocks; b += NWARPS) {
        long bo = rowOffset + (long)b * 144;

        unsigned int dm = __ldg((const unsigned int*)(weights + bo));
        float d = half2float_mw(dm & 0xFFFF);
        float dmin = half2float_mw(dm >> 16);

        unsigned int sc0 = __ldg((const unsigned int*)(weights + bo + 4));
        unsigned int sc1 = __ldg((const unsigned int*)(weights + bo + 8));
        unsigned int sc2 = __ldg((const unsigned int*)(weights + bo + 12));

        // Each lane handles 1 of 4 groups × 8 i positions: lane k → (group=k/8, i=k%8)
        int group = lane >> 3;     // 0..3
        int i = lane & 7;          // 0..7

        // Decode scales/mins WITHOUT byte array (avoids local-memory spill)
        unsigned int shift = (group & 1) ? 16 : 0;
        unsigned int b0 = (sc0 >> shift) & 0xFFFFu;
        unsigned int b1 = (sc1 >> shift) & 0xFFFFu;
        int scale0, min0, scale1, min1;
        if (group < 2) {
            scale0 = (int)( b0        & 0x3F);
            scale1 = (int)((b0 >> 8)  & 0x3F);
            min0   = (int)( b1        & 0x3F);
            min1   = (int)((b1 >> 8)  & 0x3F);
        } else {
            unsigned int b2 = (sc2 >> shift) & 0xFFFFu;
            scale0 = (int)( (b2        & 0x0F) | (((b0 >> 6)  & 0x03) << 4));
            scale1 = (int)(((b2 >> 8)  & 0x0F) | (((b0 >> 14) & 0x03) << 4));
            min0   = (int)(((b2 >> 4)  & 0x0F) | (((b1 >> 6)  & 0x03) << 4));
            min1   = (int)(((b2 >> 12) & 0x0F) | (((b1 >> 14) & 0x03) << 4));
        }

        // Q8_1 blocks for this group: covers elements [b*256+group*64, +64)
        int q8Block0 = (b * 256 + group * 64) / 32;
        int q8Block1 = (b * 256 + group * 64 + 32) / 32;
        long q8off0 = (long)q8Block0 * 40;
        long q8off1 = (long)q8Block1 * 40;
        float inScale0 = *(const float*)(input + q8off0);
        float inSum0   = *(const float*)(input + q8off0 + 4);
        float inScale1 = *(const float*)(input + q8off1);
        float inSum1   = *(const float*)(input + q8off1 + 4);

        // One uint of weight = 4 bytes = 8 nibbles for the 4-byte chunk i within the group
        const unsigned char* qsBase = weights + bo + 16 + group * 32;
        unsigned int qw = __ldg((const unsigned int*)(qsBase + i * 4));
        int q4_sub0 = (qw & 0x0F) | (((qw >> 8) & 0x0F) << 8)
                    | (((qw >> 16) & 0x0F) << 16) | (((qw >> 24) & 0x0F) << 24);
        int q4_sub1 = ((qw >> 4) & 0x0F) | ((((qw >> 12) & 0x0F)) << 8)
                    | (((qw >> 20) & 0x0F) << 16) | (((qw >> 28) & 0x0F) << 24);

        int in_sub0 = *(const int*)(input + q8off0 + 8 + i * 4);
        int in_sub1 = *(const int*)(input + q8off1 + 8 + i * 4);

        int dp0 = __dp4a(q4_sub0, in_sub0, 0);
        int dp1 = __dp4a(q4_sub1, in_sub1, 0);

        // Each lane contributes its 4-element slice to the row's total.
        // Combine with scales: each lane handles only 4 consecutive elements,
        // so we must scale the sum-correction terms (inSum) by 4/32 = 1/8.
        float lanePartial =
              d * (float)scale0 * inScale0 * (float)dp0
            - dmin * (float)min0 * inSum0 * (1.0f / 8.0f)
            + d * (float)scale1 * inScale1 * (float)dp1
            - dmin * (float)min1 * inSum1 * (1.0f / 8.0f);

        sum += lanePartial;
    }

    // Within-warp reduction
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    // Cross-warp reduction via shared memory
    __shared__ float warpSums[NWARPS];
    if (lane == 0) warpSums[warpId] = sum;
    __syncthreads();

    if (warpId == 0) {
        float total = (lane < NWARPS) ? warpSums[lane] : 0.0f;
        for (int off = NWARPS / 2; off > 0; off >>= 1)
            total += __shfl_down_sync(0xFFFFFFFF, total, off);
        if (lane == 0) {
            if (addToOutput) output[row] += total;
            else             output[row]  = total;
        }
    }
}
