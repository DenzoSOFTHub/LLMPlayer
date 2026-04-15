// FP16 → FP32 (same impl as matmul_q4_k_dp4a.cu)
__device__ __forceinline__ float half2float(unsigned short h) {
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

/**
 * Q4_K matmul (dp4a, multi-row variant: 4 rows per warp).
 *
 * Goal vs the baseline `matmul_q4_k_dp4a` (1 row per warp):
 *   - Each warp computes 4 output rows in parallel.
 *   - The Q8_1 input reads (scale, sum, qs) are SHARED across the 4 rows.
 *     Saves 3× input bandwidth (input is read once, used for 4 outputs).
 *   - 4× more independent dp4a operations per lane → better instruction-level
 *     parallelism, hides memory-load latency more effectively.
 *
 * Trade-offs:
 *   - 4× more weight reads per warp (different rows = different weights).
 *   - 4× more registers per lane (4 partial sums + 4 weight slots).
 *   - Lower SM occupancy (fewer concurrent warps).
 *
 * Net win depends on whether the baseline was latency-bound (helps) or
 * occupancy-bound (hurts). Empirical question — must measure.
 *
 * Compute layout:
 *   blockDim.x = 128 (4 warps per block)
 *   gridDim.x  = ceil(rows / (4 * 4))   // 4 warps × 4 rows = 16 rows per block
 *   Each warp owns a contiguous group of 4 rows.
 *
 * Caller must compute gridDim/blockDim accordingly.
 */
extern "C" __global__ void matmul_q4_k_dp4a_mr4(
    const unsigned char* __restrict__ weights,
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    const int rows, const int cols, const int addToOutput)
{
    constexpr int ROWS_PER_WARP = 4;
    int warpId = threadIdx.x >> 5;
    int lane   = threadIdx.x & 31;
    int warpsPerBlock = blockDim.x >> 5;
    int rowBase = blockIdx.x * (warpsPerBlock * ROWS_PER_WARP) + warpId * ROWS_PER_WARP;
    if (rowBase >= rows) return;

    int rowsThisWarp = (rows - rowBase < ROWS_PER_WARP) ? (rows - rowBase) : ROWS_PER_WARP;

    int numSuperBlocks = cols / 256;
    int numGroups = numSuperBlocks * 4;
    float sums[ROWS_PER_WARP] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = lane; g < numGroups; g += 32) {
        int b = g >> 2;
        int group = g & 3;

        // ============ Read Q8_1 input ONCE (shared across all rows) ============
        int q8Block0 = (b * 256 + group * 64) >> 5;
        int q8Block1 = q8Block0 + 1;
        long q8off0 = (long)q8Block0 * 40;
        long q8off1 = (long)q8Block1 * 40;
        float inScale0 = *(const float*)(input + q8off0);
        float inSum0   = *(const float*)(input + q8off0 + 4);
        float inScale1 = *(const float*)(input + q8off1);
        float inSum1   = *(const float*)(input + q8off1 + 4);

        // Pre-read Q8_1 input qs bytes into registers (32 bytes × 2 sub-blocks)
        int in_sub0[8], in_sub1[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            in_sub0[i] = *(const int*)(input + q8off0 + 8 + i * 4);
            in_sub1[i] = *(const int*)(input + q8off1 + 8 + i * 4);
        }

        // ============ Per-row weight reads + dp4a ============
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            if (r >= rowsThisWarp) break;  // tail block protection

            int row = rowBase + r;
            long rowOffset = (long)row * numSuperBlocks * 144;
            long bo = rowOffset + (long)b * 144;

            // Q4_K header (d, dmin)
            unsigned int dm = __ldg((const unsigned int*)(weights + bo));
            float d_w  = half2float(dm & 0xFFFF);
            float dmin = half2float(dm >> 16);

            // Scales (12 bytes from byte 4)
            unsigned int sc0 = __ldg((const unsigned int*)(weights + bo + 4));
            unsigned int sc1 = __ldg((const unsigned int*)(weights + bo + 8));
            unsigned int sc2 = __ldg((const unsigned int*)(weights + bo + 12));
            unsigned char sb[12];
            sb[0] = sc0 & 0xFF; sb[1] = (sc0>>8) & 0xFF; sb[2] = (sc0>>16) & 0xFF; sb[3] = (sc0>>24) & 0xFF;
            sb[4] = sc1 & 0xFF; sb[5] = (sc1>>8) & 0xFF; sb[6] = (sc1>>16) & 0xFF; sb[7] = (sc1>>24) & 0xFF;
            sb[8] = sc2 & 0xFF; sb[9] = (sc2>>8) & 0xFF; sb[10] = (sc2>>16) & 0xFF; sb[11] = (sc2>>24) & 0xFF;

            int sb0_idx = group * 2, sb1_idx = group * 2 + 1;
            int scale0, min0, scale1, min1;
            if (sb0_idx < 4) { scale0 = sb[sb0_idx] & 0x3F; min0 = sb[sb0_idx+4] & 0x3F; }
            else { scale0 = (sb[sb0_idx+4] & 0x0F) | ((sb[sb0_idx-4] >> 6) << 4);
                   min0   = ((sb[sb0_idx+4]>>4) & 0x0F) | ((sb[sb0_idx]>>6) << 4); }
            if (sb1_idx < 4) { scale1 = sb[sb1_idx] & 0x3F; min1 = sb[sb1_idx+4] & 0x3F; }
            else { scale1 = (sb[sb1_idx+4] & 0x0F) | ((sb[sb1_idx-4] >> 6) << 4);
                   min1   = ((sb[sb1_idx+4]>>4) & 0x0F) | ((sb[sb1_idx]>>6) << 4); }

            // Weight nibbles for this group (32 bytes)
            const unsigned char* qsBase = weights + bo + 16 + group * 32;

            int dp0 = 0, dp1 = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                unsigned int qw = __ldg((const unsigned int*)(qsBase + i * 4));
                int q4_sub0 = (qw & 0x0F) | (((qw >> 8) & 0x0F) << 8)
                            | (((qw >> 16) & 0x0F) << 16) | (((qw >> 24) & 0x0F) << 24);
                int q4_sub1 = ((qw >> 4) & 0x0F) | ((((qw >> 12) & 0x0F)) << 8)
                            | (((qw >> 20) & 0x0F) << 16) | (((qw >> 28) & 0x0F) << 24);
                dp0 = __dp4a(q4_sub0, in_sub0[i], dp0);
                dp1 = __dp4a(q4_sub1, in_sub1[i], dp1);
            }

            sums[r] += d_w * (float)scale0 * inScale0 * (float)dp0 - dmin * (float)min0 * inSum0;
            sums[r] += d_w * (float)scale1 * inScale1 * (float)dp1 - dmin * (float)min1 * inSum1;
        }
    }

    // ============ Per-row warp-shuffle reduction + output ============
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        if (r >= rowsThisWarp) break;
        float v = sums[r];
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xFFFFFFFF, v, off);
        if (lane == 0) {
            int row = rowBase + r;
            if (addToOutput) output[row] += v;
            else output[row] = v;
        }
    }
}
