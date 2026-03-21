/**
 * Q4_K × Q8_1 matrix-vector multiply using __dp4a intrinsic.
 * Input is pre-quantized to Q8_1 format: [float scale][float sum][int8 qs[32]] per block of 32.
 * Uses __dp4a for 4x int8 multiply-add in a single instruction (4x fewer instructions than FP32).
 *
 * Q4_K dequant formula: value = d * scale_w * q - dmin * min_w
 * Dot product reformulation:
 *   sum_i(value[i] * input[i]) = d * scale_w * input_scale * dp4a_sum - dmin * min_w * input_sum
 *
 * One warp (32 threads) per output row, threads stripe across groups.
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

extern "C" __global__ void matmul_q4_k_dp4a(
    const unsigned char* __restrict__ weights,  // Q4_K weights
    const unsigned char* __restrict__ input,    // Q8_1 quantized input: [(cols/32) * 40 bytes]
    float* __restrict__ output,
    const int rows,
    const int cols,
    const int addToOutput)
{
    int warpId = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int rowsPerBlock = blockDim.x / 32;
    int row = blockIdx.x * rowsPerBlock + warpId;
    if (row >= rows) return;

    int numSuperBlocks = cols / 256;  // Q4_K super-blocks (256 elements each)
    long rowOffset = (long)row * numSuperBlocks * 144;
    int numGroups = numSuperBlocks * 4;  // 4 groups per super-block
    float sum = 0.0f;

    for (int g = lane; g < numGroups; g += 32) {
        int b = g >> 2;           // super-block index
        int group = g & 3;        // group within super-block (0-3)
        long bo = rowOffset + (long)b * 144;

        // Load Q4_K header: d and dmin
        unsigned int dm = __ldg((const unsigned int*)(weights + bo));
        float d = half2float(dm & 0xFFFF);
        float dmin = half2float(dm >> 16);

        // Load scales
        unsigned int sc0 = __ldg((const unsigned int*)(weights + bo + 4));
        unsigned int sc1 = __ldg((const unsigned int*)(weights + bo + 8));
        unsigned int sc2 = __ldg((const unsigned int*)(weights + bo + 12));
        unsigned char sb[12];
        sb[0] = sc0 & 0xFF; sb[1] = (sc0>>8) & 0xFF; sb[2] = (sc0>>16) & 0xFF; sb[3] = (sc0>>24) & 0xFF;
        sb[4] = sc1 & 0xFF; sb[5] = (sc1>>8) & 0xFF; sb[6] = (sc1>>16) & 0xFF; sb[7] = (sc1>>24) & 0xFF;
        sb[8] = sc2 & 0xFF; sb[9] = (sc2>>8) & 0xFF; sb[10] = (sc2>>16) & 0xFF; sb[11] = (sc2>>24) & 0xFF;

        // Decode scales and mins for this group's two sub-blocks
        int sb0 = group * 2, sb1 = group * 2 + 1;
        int scale0, min0, scale1, min1;
        if (sb0 < 4) { scale0 = sb[sb0] & 0x3F; min0 = sb[sb0+4] & 0x3F; }
        else { scale0 = (sb[sb0+4] & 0x0F) | ((sb[sb0-4] >> 6) << 4); min0 = ((sb[sb0+4]>>4) & 0x0F) | ((sb[sb0]>>6) << 4); }
        if (sb1 < 4) { scale1 = sb[sb1] & 0x3F; min1 = sb[sb1+4] & 0x3F; }
        else { scale1 = (sb[sb1+4] & 0x0F) | ((sb[sb1-4] >> 6) << 4); min1 = ((sb[sb1+4]>>4) & 0x0F) | ((sb[sb1]>>6) << 4); }

        // Group covers elements [b*256 + group*64, b*256 + group*64 + 64)
        // Sub-block 0: elements [0..31], sub-block 1: elements [32..63]
        // Q8_1 blocks: each covers 32 elements → 2 Q8_1 blocks per group
        int q8Block0 = (b * 256 + group * 64) / 32;      // first Q8_1 block
        int q8Block1 = (b * 256 + group * 64 + 32) / 32;  // second Q8_1 block

        // Read Q8_1 block data: [scale(4B)][sum(4B)][qs(32B)]
        long q8off0 = (long)q8Block0 * 40;
        long q8off1 = (long)q8Block1 * 40;
        float inScale0 = *(const float*)(input + q8off0);
        float inSum0   = *(const float*)(input + q8off0 + 4);
        float inScale1 = *(const float*)(input + q8off1);
        float inSum1   = *(const float*)(input + q8off1 + 4);

        // Q4_K weight bytes: 32 bytes per group starting at bo + 16 + group*32
        const unsigned char* qsBase = weights + bo + 16 + group * 32;

        // Compute dp4a dot products for sub-block 0 and sub-block 1
        int dp0 = 0, dp1 = 0;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            // Load 4 Q4_K bytes → extract low nibbles (sub0) and high nibbles (sub1)
            unsigned int qw = __ldg((const unsigned int*)(qsBase + i * 4));

            // Extract 4 low nibbles for sub-block 0, pack as int8x4
            int q4_sub0 = (qw & 0x0F) | (((qw >> 8) & 0x0F) << 8)
                        | (((qw >> 16) & 0x0F) << 16) | (((qw >> 24) & 0x0F) << 24);

            // Extract 4 high nibbles for sub-block 1, pack as int8x4
            int q4_sub1 = ((qw >> 4) & 0x0F) | ((((qw >> 12) & 0x0F)) << 8)
                        | (((qw >> 20) & 0x0F) << 16) | (((qw >> 28) & 0x0F) << 24);

            // Load 4 Q8_1 input values for each sub-block, pack as int8x4
            int in_sub0 = *(const int*)(input + q8off0 + 8 + i * 4);
            int in_sub1 = *(const int*)(input + q8off1 + 8 + i * 4);

            // dp4a: 4 int8 multiply-adds in single instruction
            dp0 = __dp4a(q4_sub0, in_sub0, dp0);
            dp1 = __dp4a(q4_sub1, in_sub1, dp1);
        }

        // Combine: dot = d * scale_w * input_scale * dp4a_result - dmin * min_w * input_sum
        sum += d * (float)scale0 * inScale0 * (float)dp0 - dmin * (float)min0 * inSum0;
        sum += d * (float)scale1 * inScale1 * (float)dp1 - dmin * (float)min1 * inSum1;
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (lane == 0) {
        if (addToOutput) output[row] += sum;
        else output[row] = sum;
    }
}
