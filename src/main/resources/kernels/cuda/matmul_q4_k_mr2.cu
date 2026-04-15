/**
 * Q4_K dequant + matmul, multi-row variant: 2 rows per warp.
 * Less register pressure than mr4, more parallelism than baseline.
 *
 * Compute layout:
 *   blockDim.x = 256 (8 warps per block)
 *   gridDim.x  = ceil(rows / 16)   // 8 warps × 2 rows = 16 rows per block
 */
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

extern "C" __global__ void matmul_q4_k_mr2(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    const int rows, const int cols, const int addToOutput)
{
    constexpr int ROWS_PER_WARP = 2;
    int warpId = threadIdx.x >> 5;
    int lane   = threadIdx.x & 31;
    int warpsPerBlock = blockDim.x >> 5;
    int rowBase = blockIdx.x * (warpsPerBlock * ROWS_PER_WARP) + warpId * ROWS_PER_WARP;
    if (rowBase >= rows) return;

    int rowsThisWarp = (rows - rowBase < ROWS_PER_WARP) ? (rows - rowBase) : ROWS_PER_WARP;

    int numBlocks = cols / 256;
    int numGroups = numBlocks * 4;
    float sum0 = 0.0f, sum1 = 0.0f;

    for (int g = lane; g < numGroups; g += 32) {
        int b = g >> 2;
        int group = g & 3;
        int inputBase = b * 256 + group * 64;

        // Read FP32 input ONCE
        float4 in0[8], in1[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            in0[i] = __ldg((const float4*)(input + inputBase + i * 4));
            in1[i] = __ldg((const float4*)(input + inputBase + 32 + i * 4));
        }

        // Per-row weight reads + dequant + dot
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            if (r >= rowsThisWarp) break;
            int row = rowBase + r;
            long rowOffset = (long)row * numBlocks * 144;
            long bo = rowOffset + (long)b * 144;

            unsigned int dm = __ldg((const unsigned int*)(weights + bo));
            float d = half2float(dm & 0xFFFF);
            float dmin = half2float(dm >> 16);

            unsigned int sc0 = __ldg((const unsigned int*)(weights + bo + 4));
            unsigned int sc1 = __ldg((const unsigned int*)(weights + bo + 8));
            unsigned int sc2 = __ldg((const unsigned int*)(weights + bo + 12));
            unsigned char sb[12];
            sb[0] = sc0 & 0xFF; sb[1] = (sc0 >> 8) & 0xFF; sb[2] = (sc0 >> 16) & 0xFF; sb[3] = (sc0 >> 24) & 0xFF;
            sb[4] = sc1 & 0xFF; sb[5] = (sc1 >> 8) & 0xFF; sb[6] = (sc1 >> 16) & 0xFF; sb[7] = (sc1 >> 24) & 0xFF;
            sb[8] = sc2 & 0xFF; sb[9] = (sc2 >> 8) & 0xFF; sb[10] = (sc2 >> 16) & 0xFF; sb[11] = (sc2 >> 24) & 0xFF;

            int sb0 = group * 2, sb1 = group * 2 + 1;
            int scale0, min0, scale1, min1;
            if (sb0 < 4) { scale0 = sb[sb0] & 0x3F; min0 = sb[sb0 + 4] & 0x3F; }
            else { scale0 = (sb[sb0 + 4] & 0x0F) | ((sb[sb0 - 4] >> 6) << 4);
                   min0 = ((sb[sb0 + 4] >> 4) & 0x0F) | ((sb[sb0] >> 6) << 4); }
            if (sb1 < 4) { scale1 = sb[sb1] & 0x3F; min1 = sb[sb1 + 4] & 0x3F; }
            else { scale1 = (sb[sb1 + 4] & 0x0F) | ((sb[sb1 - 4] >> 6) << 4);
                   min1 = ((sb[sb1 + 4] >> 4) & 0x0F) | ((sb[sb1] >> 6) << 4); }

            float ds0 = d * (float)scale0;
            float dm0 = dmin * (float)min0;
            float ds1 = d * (float)scale1;
            float dm1 = dmin * (float)min1;

            float s = (r == 0) ? sum0 : sum1;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                unsigned int qw = __ldg((const unsigned int*)(weights + bo + 16 + group * 32 + i * 4));
                s += (ds0 * (float)(qw & 0x0F)        - dm0) * in0[i].x;
                s += (ds1 * (float)((qw >> 4) & 0x0F) - dm1) * in1[i].x;
                s += (ds0 * (float)((qw >> 8) & 0x0F) - dm0) * in0[i].y;
                s += (ds1 * (float)((qw >> 12) & 0x0F) - dm1) * in1[i].y;
                s += (ds0 * (float)((qw >> 16) & 0x0F) - dm0) * in0[i].z;
                s += (ds1 * (float)((qw >> 20) & 0x0F) - dm1) * in1[i].z;
                s += (ds0 * (float)((qw >> 24) & 0x0F) - dm0) * in0[i].w;
                s += (ds1 * (float)((qw >> 28) & 0x0F) - dm1) * in1[i].w;
            }
            if (r == 0) sum0 = s; else sum1 = s;
        }
    }

    // Reduction + output (both rows)
    for (int off = 16; off > 0; off >>= 1)
        sum0 += __shfl_down_sync(0xFFFFFFFF, sum0, off);
    for (int off = 16; off > 0; off >>= 1)
        sum1 += __shfl_down_sync(0xFFFFFFFF, sum1, off);
    if (lane == 0) {
        if (addToOutput) output[rowBase] += sum0;
        else             output[rowBase]  = sum0;
        if (rowsThisWarp > 1) {
            if (addToOutput) output[rowBase + 1] += sum1;
            else             output[rowBase + 1]  = sum1;
        }
    }
}
