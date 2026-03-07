/**
 * Q6_K dequantize + matrix-vector multiply kernel.
 * 256 weights per super-block, 210 bytes per block.
 * Layout: [ql:128B][qh:64B][scales:16B][d:fp16]
 *
 * COALESCED access pattern: all 32 threads in a warp process the same half-block
 * simultaneously, with each thread handling one of 32 element positions.
 * Consecutive threads read consecutive byte addresses → perfectly coalesced
 * memory transactions (vs. the old half-striped pattern where threads accessed
 * scattered addresses from different blocks).
 *
 * addToOutput: 0 = write, 1 = accumulate
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

extern "C" __global__ void matmul_q6_k(
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
    int numHalves = numBlocks * 2;
    int rowStride = numBlocks * 210;
    float sum = 0.0f;

    // Each half-block: 32 threads handle 32 positions in parallel (coalesced)
    for (int h = 0; h < numHalves; h++) {
        int b = h >> 1;
        int hf = h & 1;
        long bo = (long)row * rowStride + b * 210;

        float d = half2float(__ldg((const unsigned short*)(weights + bo + 208)));

        int qlOff = hf * 64;
        int qhOff = hf * 32;
        int scBase = hf * 8;
        int inputBase = b * 256 + hf * 128;

        // Coalesced byte reads: consecutive lanes → consecutive addresses
        unsigned char qlByte0 = __ldg(&weights[bo + qlOff + lane]);
        unsigned char qlByte1 = __ldg(&weights[bo + qlOff + 32 + lane]);
        unsigned char qhByte  = __ldg(&weights[bo + 128 + qhOff + lane]);

        // Decode four 6-bit quantized values from the three bytes
        int q0 = (qlByte0 & 0x0F)        | (((qhByte >> 0) & 3) << 4);
        int q1 = (qlByte1 & 0x0F)        | (((qhByte >> 2) & 3) << 4);
        int q2 = ((qlByte0 >> 4) & 0x0F) | (((qhByte >> 4) & 3) << 4);
        int q3 = ((qlByte1 >> 4) & 0x0F) | (((qhByte >> 6) & 3) << 4);

        // Scale lookup (two groups: lanes 0-15 use scBase, lanes 16-31 use scBase+1)
        int scIdx = scBase + (lane >> 4);
        float ds0 = d * (float)((signed char)__ldg(&weights[bo + 192 + scIdx]));
        float ds1 = d * (float)((signed char)__ldg(&weights[bo + 192 + scIdx + 2]));
        float ds2 = d * (float)((signed char)__ldg(&weights[bo + 192 + scIdx + 4]));
        float ds3 = d * (float)((signed char)__ldg(&weights[bo + 192 + scIdx + 6]));

        // Coalesced input reads: consecutive lanes → consecutive floats
        sum += ds0 * (float)(q0 - 32) * __ldg(&input[inputBase + lane]);
        sum += ds1 * (float)(q1 - 32) * __ldg(&input[inputBase + 32 + lane]);
        sum += ds2 * (float)(q2 - 32) * __ldg(&input[inputBase + 64 + lane]);
        sum += ds3 * (float)(q3 - 32) * __ldg(&input[inputBase + 96 + lane]);
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (lane == 0) {
        if (addToOutput) output[row] += sum;
        else output[row] = sum;
    }
}
