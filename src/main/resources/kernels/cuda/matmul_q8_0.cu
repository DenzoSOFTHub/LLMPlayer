/**
 * Q8_0 dequantize + matrix-vector multiply kernel.
 * 32 weights per block, 34 bytes per block.
 * Layout: [scale:fp16 (2B)][32 x int8 quants (32B)]
 * Each warp (32 threads) computes one output row, striping across blocks.
 * Uses float4 vectorized loads for input reads.
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

extern "C" __global__ void matmul_q8_0(
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

    int numBlocks = cols / 32;
    int rowStride = numBlocks * 34;
    float sum = 0.0f;

    for (int b = lane; b < numBlocks; b += 32) {
        int bo = row * rowStride + b * 34;
        float scale = half2float(*(unsigned short*)(weights + bo));
        int inputBase = b * 32;

        #pragma unroll
        for (int l = 0; l < 8; l++) {
            float4 in4 = ((const float4*)(input + inputBase))[l];
            signed char q0 = (signed char)weights[bo + 2 + l * 4];
            signed char q1 = (signed char)weights[bo + 2 + l * 4 + 1];
            signed char q2 = (signed char)weights[bo + 2 + l * 4 + 2];
            signed char q3 = (signed char)weights[bo + 2 + l * 4 + 3];
            sum += scale * ((float)q0 * in4.x + (float)q1 * in4.y +
                            (float)q2 * in4.z + (float)q3 * in4.w);
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
