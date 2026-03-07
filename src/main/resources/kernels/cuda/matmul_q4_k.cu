/**
 * Q4_K dequantize + matrix-vector multiply kernel.
 * 256 weights per super-block, 144 bytes per block, 4 groups of 64 elements per block.
 * Layout: [d:fp16][dmin:fp16][scales:12B][qs:128B]
 *
 * Each warp (32 threads) computes one output row.
 * Threads stripe across GROUPS (not blocks) for full warp utilization.
 * Uses __restrict__ for read-only cache (LDG) and uint32 vectorized loads.
 * addToOutput: 0 = write (output[row] = sum), 1 = accumulate (output[row] += sum)
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

extern "C" __global__ void matmul_q4_k(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
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

    int numBlocks = cols / 256;
    long rowOffset = (long)row * numBlocks * 144;
    int numGroups = numBlocks * 4;
    float sum = 0.0f;

    for (int g = lane; g < numGroups; g += 32) {
        int b = g >> 2;
        int group = g & 3;
        long bo = rowOffset + (long)b * 144;

        // Load d and dmin as single uint32 (2x fp16 packed)
        unsigned int dm = __ldg((const unsigned int*)(weights + bo));
        float d = half2float(dm & 0xFFFF);
        float dmin = half2float(dm >> 16);

        // Load scale bytes: 3 uint32 reads for 12 bytes at bo+4
        unsigned int sc0 = __ldg((const unsigned int*)(weights + bo + 4));
        unsigned int sc1 = __ldg((const unsigned int*)(weights + bo + 8));
        unsigned int sc2 = __ldg((const unsigned int*)(weights + bo + 12));

        // Extract individual scale bytes
        unsigned char sb[12];
        sb[0] = sc0 & 0xFF; sb[1] = (sc0 >> 8) & 0xFF; sb[2] = (sc0 >> 16) & 0xFF; sb[3] = (sc0 >> 24) & 0xFF;
        sb[4] = sc1 & 0xFF; sb[5] = (sc1 >> 8) & 0xFF; sb[6] = (sc1 >> 16) & 0xFF; sb[7] = (sc1 >> 24) & 0xFF;
        sb[8] = sc2 & 0xFF; sb[9] = (sc2 >> 8) & 0xFF; sb[10] = (sc2 >> 16) & 0xFF; sb[11] = (sc2 >> 24) & 0xFF;

        // Decode scales and mins for this group's two sub-blocks
        int sb0 = group * 2;
        int sb1 = group * 2 + 1;
        int scale0, min0, scale1, min1;
        if (sb0 < 4) {
            scale0 = sb[sb0] & 0x3F;
            min0 = sb[sb0 + 4] & 0x3F;
        } else {
            scale0 = (sb[sb0 + 4] & 0x0F) | ((sb[sb0 - 4] >> 6) << 4);
            min0 = ((sb[sb0 + 4] >> 4) & 0x0F) | ((sb[sb0] >> 6) << 4);
        }
        if (sb1 < 4) {
            scale1 = sb[sb1] & 0x3F;
            min1 = sb[sb1 + 4] & 0x3F;
        } else {
            scale1 = (sb[sb1 + 4] & 0x0F) | ((sb[sb1 - 4] >> 6) << 4);
            min1 = ((sb[sb1 + 4] >> 4) & 0x0F) | ((sb[sb1] >> 6) << 4);
        }

        float ds0 = d * (float)scale0;
        float dm0 = dmin * (float)min0;
        float ds1 = d * (float)scale1;
        float dm1 = dmin * (float)min1;

        int inputBase = b * 256 + group * 64;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float4 in0 = __ldg((const float4*)(input + inputBase + i * 4));
            float4 in1 = __ldg((const float4*)(input + inputBase + 32 + i * 4));

            // Load 4 quantized bytes as single uint32
            unsigned int qw = __ldg((const unsigned int*)(weights + bo + 16 + group * 32 + i * 4));

            sum += (ds0 * (float)(qw & 0x0F) - dm0) * in0.x;
            sum += (ds1 * (float)((qw >> 4) & 0x0F) - dm1) * in1.x;
            sum += (ds0 * (float)((qw >> 8) & 0x0F) - dm0) * in0.y;
            sum += (ds1 * (float)((qw >> 12) & 0x0F) - dm1) * in1.y;
            sum += (ds0 * (float)((qw >> 16) & 0x0F) - dm0) * in0.z;
            sum += (ds1 * (float)((qw >> 20) & 0x0F) - dm1) * in1.z;
            sum += (ds0 * (float)((qw >> 24) & 0x0F) - dm0) * in0.w;
            sum += (ds1 * (float)((qw >> 28) & 0x0F) - dm1) * in1.w;
        }
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (lane == 0) {
        if (addToOutput) output[row] += sum;
        else output[row] = sum;
    }
}
