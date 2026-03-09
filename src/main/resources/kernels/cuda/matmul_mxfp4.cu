/**
 * MXFP4 (Microscaling FP4 E2M1) dequantize + matrix-vector multiply kernel.
 * 32 weights per block, 17 bytes per block: [e8m0_scale:1B][qs:16B]
 *
 * Split nibble layout: low nibbles of qs[0..15] -> positions 0..15,
 *                      high nibbles of qs[0..15] -> positions 16..31
 *
 * FP4 E2M1 values: 0, +/-0.5, +/-1.0, +/-1.5, +/-2.0, +/-3.0, +/-4.0, +/-6.0
 * E8M0 scale: 2^(exponent - 127), special: 0->0, 255->0
 *
 * Each warp (32 threads) computes one output row.
 * Threads stripe across blocks for full warp utilization.
 * addToOutput: 0 = write (output[row] = sum), 1 = accumulate (output[row] += sum)
 */

__device__ __constant__ float FP4_TABLE[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ float e8m0_scale(unsigned char exp) {
    if (exp == 0 || exp == 255) return 0.0f;
    return __int_as_float(((unsigned int)exp) << 23);
}

extern "C" __global__ void matmul_mxfp4(
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
    long rowOffset = (long)row * numBlocks * 17;
    float sum = 0.0f;

    for (int b = lane; b < numBlocks; b += 32) {
        long bo = rowOffset + (long)b * 17;
        float scale = e8m0_scale(__ldg(weights + bo));
        int inputBase = b * 32;

        // Process 16 packed bytes = 32 FP4 values
        // Unroll by 4 for instruction-level parallelism
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            unsigned char p0 = __ldg(weights + bo + 1 + i * 4);
            unsigned char p1 = __ldg(weights + bo + 1 + i * 4 + 1);
            unsigned char p2 = __ldg(weights + bo + 1 + i * 4 + 2);
            unsigned char p3 = __ldg(weights + bo + 1 + i * 4 + 3);

            // Low nibbles -> positions 0..15
            float4 in_lo = __ldg((const float4*)(input + inputBase + i * 4));
            sum += FP4_TABLE[p0 & 0x0F] * scale * in_lo.x;
            sum += FP4_TABLE[p1 & 0x0F] * scale * in_lo.y;
            sum += FP4_TABLE[p2 & 0x0F] * scale * in_lo.z;
            sum += FP4_TABLE[p3 & 0x0F] * scale * in_lo.w;

            // High nibbles -> positions 16..31
            float4 in_hi = __ldg((const float4*)(input + inputBase + 16 + i * 4));
            sum += FP4_TABLE[(p0 >> 4) & 0x0F] * scale * in_hi.x;
            sum += FP4_TABLE[(p1 >> 4) & 0x0F] * scale * in_hi.y;
            sum += FP4_TABLE[(p2 >> 4) & 0x0F] * scale * in_hi.z;
            sum += FP4_TABLE[(p3 >> 4) & 0x0F] * scale * in_hi.w;
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
