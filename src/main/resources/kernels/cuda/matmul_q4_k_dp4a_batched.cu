/**
 * Batched Q4_K × Q8_1 matmul (matmat) using __dp4a — for speculative-decoding verification.
 * Processes K input vectors against the SAME weight matrix, reading each weight row ONCE and
 * computing K dot products. Since the weight read is the DRAM-bandwidth bottleneck at batch=1,
 * amortizing it over K inputs is the speculative-verify speedup (up to ~K× on the matmul).
 *
 * Layout: input = K Q8_1 vectors, vector k at byte offset k*inputStride (inputStride=(cols/32)*40).
 *         output[k*rows + row].  One warp per output row; lane loop over Q4_K groups.
 * K is capped at MAX_BATCH (speculation depth is typically <= 8).
 */
#define MAX_BATCH 8

__device__ __forceinline__ float half2float_b(unsigned short h) {
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

extern "C" __global__ void matmul_q4_k_dp4a_batched(
    const unsigned char* __restrict__ weights,
    const unsigned char* __restrict__ input,    // K Q8_1 vectors, stride = inputStride bytes
    float* __restrict__ output,                 // [K * rows]
    const int rows,
    const int cols,
    const int K,
    const int inputStride,                      // bytes per input vector = (cols/32)*40
    const int addToOutput)
{
    int warpId = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int rowsPerBlock = blockDim.x / 32;
    int row = blockIdx.x * rowsPerBlock + warpId;
    if (row >= rows) return;

    int numSuperBlocks = cols / 256;
    long rowOffset = (long)row * numSuperBlocks * 144;
    int numGroups = numSuperBlocks * 4;

    float sum[MAX_BATCH];
    #pragma unroll
    for (int k = 0; k < MAX_BATCH; k++) sum[k] = 0.0f;

    for (int g = lane; g < numGroups; g += 32) {
        int b = g >> 2;
        int group = g & 3;
        long bo = rowOffset + (long)b * 144;

        // --- decode weight ONCE (shared across all K inputs) ---
        unsigned int dm = __ldg((const unsigned int*)(weights + bo));
        float d = half2float_b(dm & 0xFFFF);
        float dmin = half2float_b(dm >> 16);
        unsigned int sc0 = __ldg((const unsigned int*)(weights + bo + 4));
        unsigned int sc1 = __ldg((const unsigned int*)(weights + bo + 8));
        unsigned int sc2 = __ldg((const unsigned int*)(weights + bo + 12));
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

        int q8Block0 = (b * 256 + group * 64) / 32;
        int q8Block1 = (b * 256 + group * 64 + 32) / 32;
        long q8off0 = (long)q8Block0 * 40;
        long q8off1 = (long)q8Block1 * 40;
        const unsigned char* qsBase = weights + bo + 16 + group * 32;

        // Pre-unpack the 8 weight words into sub0/sub1 int8x4 (reused across K inputs)
        int wsub0[8], wsub1[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            unsigned int qw = __ldg((const unsigned int*)(qsBase + i * 4));
            wsub0[i] = (qw & 0x0F) | (((qw >> 8) & 0x0F) << 8)
                     | (((qw >> 16) & 0x0F) << 16) | (((qw >> 24) & 0x0F) << 24);
            wsub1[i] = ((qw >> 4) & 0x0F) | ((((qw >> 12) & 0x0F)) << 8)
                     | (((qw >> 20) & 0x0F) << 16) | (((qw >> 28) & 0x0F) << 24);
        }

        // --- per-input dp4a (weight already in registers) ---
        for (int k = 0; k < K; k++) {
            const unsigned char* in = input + (long)k * inputStride;
            float inScale0 = *(const float*)(in + q8off0);
            float inSum0   = *(const float*)(in + q8off0 + 4);
            float inScale1 = *(const float*)(in + q8off1);
            float inSum1   = *(const float*)(in + q8off1 + 4);
            int dp0 = 0, dp1 = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int in_sub0 = *(const int*)(in + q8off0 + 8 + i * 4);
                int in_sub1 = *(const int*)(in + q8off1 + 8 + i * 4);
                dp0 = __dp4a(wsub0[i], in_sub0, dp0);
                dp1 = __dp4a(wsub1[i], in_sub1, dp1);
            }
            sum[k] += d * (float)scale0 * inScale0 * (float)dp0 - dmin * (float)min0 * inSum0;
            sum[k] += d * (float)scale1 * inScale1 * (float)dp1 - dmin * (float)min1 * inSum1;
        }
    }

    // Warp-reduce each input's partial sum and write
    for (int k = 0; k < K; k++) {
        float s = sum[k];
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_down_sync(0xFFFFFFFF, s, offset);
        if (lane == 0) {
            int idx = k * rows + row;
            if (addToOutput) output[idx] += s; else output[idx] = s;
        }
    }
}
