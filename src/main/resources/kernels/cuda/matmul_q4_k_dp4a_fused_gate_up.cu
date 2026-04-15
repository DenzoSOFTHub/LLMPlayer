/**
 * Fused gate + up matmul for Q4_K × Q8_1 dp4a path.
 *
 * Reads the Q8_1 input ONCE per group and computes BOTH gate and up output rows in the
 * same thread. This halves the input-vector reads + the per-launch overhead vs running
 * `matmul_q4_k_dp4a` twice.
 *
 * Layout:
 *   blockDim.x = 256 (8 warps, 8 rows per block)
 *   gridDim.x  = ceil(rows / 8)  (rows is the GATE/UP output dim, both same)
 *
 * Each warp computes ROW `row` for both gate and up. Output:
 *   gateOutput[row] = sum_i gate_w[row, i] * input[i]
 *   upOutput[row]   = sum_i up_w[row, i]   * input[i]
 */
__device__ __forceinline__ float half2float_fdp(unsigned short h) {
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

// Process one Q4_K super-block × Q8_1 group: returns the dot-product contribution.
// Inlined into the fused kernel so register/instruction sharing happens between gate/up.
__device__ __forceinline__ float q4k_dp4a_one_group(
    const unsigned char* __restrict__ weights, long bo,
    int group, int q8Block0, int q8Block1,
    const unsigned char* __restrict__ input,
    float inScale0, float inSum0, float inScale1, float inSum1)
{
    unsigned int dm = __ldg((const unsigned int*)(weights + bo));
    float d = half2float_fdp(dm & 0xFFFF);
    float dmin = half2float_fdp(dm >> 16);

    unsigned int sc0 = __ldg((const unsigned int*)(weights + bo + 4));
    unsigned int sc1 = __ldg((const unsigned int*)(weights + bo + 8));
    unsigned int sc2 = __ldg((const unsigned int*)(weights + bo + 12));
    unsigned char sb[12];
    sb[0]=sc0&0xFF; sb[1]=(sc0>>8)&0xFF; sb[2]=(sc0>>16)&0xFF; sb[3]=(sc0>>24)&0xFF;
    sb[4]=sc1&0xFF; sb[5]=(sc1>>8)&0xFF; sb[6]=(sc1>>16)&0xFF; sb[7]=(sc1>>24)&0xFF;
    sb[8]=sc2&0xFF; sb[9]=(sc2>>8)&0xFF; sb[10]=(sc2>>16)&0xFF; sb[11]=(sc2>>24)&0xFF;

    int sb0 = group * 2, sb1 = group * 2 + 1;
    int scale0, min0, scale1, min1;
    if (sb0 < 4) { scale0 = sb[sb0] & 0x3F; min0 = sb[sb0+4] & 0x3F; }
    else { scale0 = (sb[sb0+4] & 0x0F) | ((sb[sb0-4] >> 6) << 4);
           min0 = ((sb[sb0+4]>>4) & 0x0F) | ((sb[sb0]>>6) << 4); }
    if (sb1 < 4) { scale1 = sb[sb1] & 0x3F; min1 = sb[sb1+4] & 0x3F; }
    else { scale1 = (sb[sb1+4] & 0x0F) | ((sb[sb1-4] >> 6) << 4);
           min1 = ((sb[sb1+4]>>4) & 0x0F) | ((sb[sb1]>>6) << 4); }

    long q8off0 = (long)q8Block0 * 40;
    long q8off1 = (long)q8Block1 * 40;
    const unsigned char* qsBase = weights + bo + 16 + group * 32;

    int dp0 = 0, dp1 = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        unsigned int qw = __ldg((const unsigned int*)(qsBase + i * 4));
        int q4_sub0 = (qw & 0x0F) | (((qw >> 8) & 0x0F) << 8)
                    | (((qw >> 16) & 0x0F) << 16) | (((qw >> 24) & 0x0F) << 24);
        int q4_sub1 = ((qw >> 4) & 0x0F) | ((((qw >> 12) & 0x0F)) << 8)
                    | (((qw >> 20) & 0x0F) << 16) | (((qw >> 28) & 0x0F) << 24);
        int in_sub0 = *(const int*)(input + q8off0 + 8 + i * 4);
        int in_sub1 = *(const int*)(input + q8off1 + 8 + i * 4);
        dp0 = __dp4a(q4_sub0, in_sub0, dp0);
        dp1 = __dp4a(q4_sub1, in_sub1, dp1);
    }

    return d * (float)scale0 * inScale0 * (float)dp0 - dmin * (float)min0 * inSum0
         + d * (float)scale1 * inScale1 * (float)dp1 - dmin * (float)min1 * inSum1;
}

extern "C" __global__ void matmul_q4_k_dp4a_fused_gate_up(
    const unsigned char* __restrict__ gateWeights,
    const unsigned char* __restrict__ upWeights,
    const unsigned char* __restrict__ input,    // Q8_1 quantized input
    float* __restrict__ gateOutput,
    float* __restrict__ upOutput,
    const int rows,
    const int cols)
{
    int warpId = threadIdx.x / 32;
    int lane   = threadIdx.x & 31;
    int rowsPerBlock = blockDim.x / 32;
    int row = blockIdx.x * rowsPerBlock + warpId;
    if (row >= rows) return;

    int numSuperBlocks = cols / 256;
    long rowOffset = (long)row * numSuperBlocks * 144;
    int numGroups = numSuperBlocks * 4;
    float sumGate = 0.0f, sumUp = 0.0f;

    for (int g = lane; g < numGroups; g += 32) {
        int b = g >> 2;
        int group = g & 3;
        long bo = rowOffset + (long)b * 144;

        // Load Q8_1 input headers ONCE — used by both gate and up
        int q8Block0 = (b * 256 + group * 64) / 32;
        int q8Block1 = (b * 256 + group * 64 + 32) / 32;
        long q8off0 = (long)q8Block0 * 40;
        long q8off1 = (long)q8Block1 * 40;
        float inScale0 = *(const float*)(input + q8off0);
        float inSum0   = *(const float*)(input + q8off0 + 4);
        float inScale1 = *(const float*)(input + q8off1);
        float inSum1   = *(const float*)(input + q8off1 + 4);

        sumGate += q4k_dp4a_one_group(gateWeights, bo, group, q8Block0, q8Block1,
                                      input, inScale0, inSum0, inScale1, inSum1);
        sumUp   += q4k_dp4a_one_group(upWeights,   bo, group, q8Block0, q8Block1,
                                      input, inScale0, inSum0, inScale1, inSum1);
    }

    // Warp reductions
    for (int off = 16; off > 0; off >>= 1) {
        sumGate += __shfl_down_sync(0xFFFFFFFF, sumGate, off);
        sumUp   += __shfl_down_sync(0xFFFFFFFF, sumUp, off);
    }

    if (lane == 0) {
        gateOutput[row] = sumGate;
        upOutput[row]   = sumUp;
    }
}
