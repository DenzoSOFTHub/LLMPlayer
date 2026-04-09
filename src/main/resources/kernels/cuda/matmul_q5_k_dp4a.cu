/**
 * Q5_K × Q8_1 matrix-vector multiply using __dp4a intrinsic.
 * Input is pre-quantized to Q8_1: [float scale][float sum][int8 qs[32]] per 32-element block.
 *
 * Q5_K block (176 bytes per 256 elements):
 *   [d:fp16][dmin:fp16][scales:12B][qh:32B][qs:128B]
 *   q5 = (qs_nibble | (qh_bit << 4)), range 0-31
 *   value = d * scale * q5 - dmin * min
 *
 * Dot product: sum((d*sc*q5 - dmin*mn) * x) = d*sc * inScale * dp4a(q5, in_q8) - dmin*mn * inSum
 *
 * One warp per row. Threads stripe across groups (4 per super-block).
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

extern "C" __global__ void matmul_q5_k_dp4a(
    const unsigned char* __restrict__ weights,
    const unsigned char* __restrict__ input,   // Q8_1 quantized
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
    int numGroups = numBlocks * 4;
    int rowStride = numBlocks * 176;
    float sum = 0.0f;

    for (int g = lane; g < numGroups; g += 32) {
        int b = g >> 2;
        int group = g & 3;
        long bo = (long)row * rowStride + b * 176;

        float d = half2float(__ldg((const unsigned short*)(weights + bo)));
        float dmin = half2float(__ldg((const unsigned short*)(weights + bo + 2)));

        // Decode scales and mins (same as Q4_K)
        int sb0 = group * 2, sb1 = group * 2 + 1;
        int scale0, min0, scale1, min1;
        if (sb0 < 4) {
            scale0 = __ldg(&weights[bo + 4 + sb0]) & 0x3F;
            min0 = __ldg(&weights[bo + 4 + sb0 + 4]) & 0x3F;
        } else {
            scale0 = (__ldg(&weights[bo + 4 + sb0 + 4]) & 0x0F) | ((__ldg(&weights[bo + 4 + sb0 - 4]) >> 6) << 4);
            min0 = ((__ldg(&weights[bo + 4 + sb0 + 4]) >> 4) & 0x0F) | ((__ldg(&weights[bo + 4 + sb0]) >> 6) << 4);
        }
        if (sb1 < 4) {
            scale1 = __ldg(&weights[bo + 4 + sb1]) & 0x3F;
            min1 = __ldg(&weights[bo + 4 + sb1 + 4]) & 0x3F;
        } else {
            scale1 = (__ldg(&weights[bo + 4 + sb1 + 4]) & 0x0F) | ((__ldg(&weights[bo + 4 + sb1 - 4]) >> 6) << 4);
            min1 = ((__ldg(&weights[bo + 4 + sb1 + 4]) >> 4) & 0x0F) | ((__ldg(&weights[bo + 4 + sb1]) >> 6) << 4);
        }

        // Q8_1 blocks for this group's two sub-blocks
        int inputBase = b * 256 + group * 64;
        int q8Block0 = inputBase / 32;
        int q8Block1 = (inputBase + 32) / 32;
        long q8off0 = (long)q8Block0 * 40;
        long q8off1 = (long)q8Block1 * 40;
        float inScale0 = *(const float*)(input + q8off0);
        float inSum0   = *(const float*)(input + q8off0 + 4);
        float inScale1 = *(const float*)(input + q8off1);
        float inSum1   = *(const float*)(input + q8off1 + 4);

        int dp0 = 0, dp1 = 0;
        int qhShift0 = group * 2;
        int qhShift1 = group * 2 + 1;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int l = i * 4;  // process 4 consecutive elements

            // Read 4 qs bytes for this group
            unsigned int qs4 = __ldg((const unsigned int*)(weights + bo + 48 + group * 32 + l));

            // Read 4 qh bytes (shared across all groups, at bo+16+l)
            unsigned int qh4 = __ldg((const unsigned int*)(weights + bo + 16 + l));

            // Extract 4 q5 values for sub-block 0 (low nibbles + high bit)
            int q5_0 = (qs4 & 0x0F)         | (((qh4 >> qhShift0) & 1) << 4);
            int q5_1 = ((qs4 >> 8) & 0x0F)  | ((((qh4 >> 8) >> qhShift0) & 1) << 4);
            int q5_2 = ((qs4 >> 16) & 0x0F) | ((((qh4 >> 16) >> qhShift0) & 1) << 4);
            int q5_3 = ((qs4 >> 24) & 0x0F) | ((((qh4 >> 24) >> qhShift0) & 1) << 4);
            int q5_packed_0 = (q5_0 & 0xFF) | ((q5_1 & 0xFF) << 8) | ((q5_2 & 0xFF) << 16) | ((q5_3 & 0xFF) << 24);

            // Extract 4 q5 values for sub-block 1 (high nibbles + high bit)
            int q5_4 = ((qs4 >> 4) & 0x0F)  | (((qh4 >> qhShift1) & 1) << 4);
            int q5_5 = ((qs4 >> 12) & 0x0F) | ((((qh4 >> 8) >> qhShift1) & 1) << 4);
            int q5_6 = ((qs4 >> 20) & 0x0F) | ((((qh4 >> 16) >> qhShift1) & 1) << 4);
            int q5_7 = ((qs4 >> 28) & 0x0F) | ((((qh4 >> 24) >> qhShift1) & 1) << 4);
            int q5_packed_1 = (q5_4 & 0xFF) | ((q5_5 & 0xFF) << 8) | ((q5_6 & 0xFF) << 16) | ((q5_7 & 0xFF) << 24);

            // Load Q8_1 input int8x4
            int in_sub0 = *(const int*)(input + q8off0 + 8 + i * 4);
            int in_sub1 = *(const int*)(input + q8off1 + 8 + i * 4);

            dp0 = __dp4a(q5_packed_0, in_sub0, dp0);
            dp1 = __dp4a(q5_packed_1, in_sub1, dp1);
        }

        // Final: d * scale * inScale * dp4a_sum - dmin * min * inSum
        sum += d * (float)scale0 * inScale0 * (float)dp0 - dmin * (float)min0 * inSum0;
        sum += d * (float)scale1 * inScale1 * (float)dp1 - dmin * (float)min1 * inSum1;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (lane == 0) {
        if (addToOutput) output[row] += sum;
        else output[row] = sum;
    }
}
