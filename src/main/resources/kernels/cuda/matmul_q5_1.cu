/**
 * Q5_1 dequantize + matrix-vector multiply kernel.
 * 32 weights per block, 24 bytes per block.
 * Layout: [scale:fp16 (2B)][min:fp16 (2B)][uint32 qh (4B)][16 bytes nibbles] = 24B
 * Element mapping (SPLIT, like Q5_0):
 *   Elements  0..15 = LOW nibbles of bytes 0..15, high bits from qh bits 0..15
 *   Elements 16..31 = HIGH nibbles of bytes 0..15, high bits from qh bits 16..31
 * value = (nibble | (high_bit << 4)) * scale + min
 *
 * Block size 24 IS divisible by 4 — aligned uint32 reads are safe.
 */
__device__ __forceinline__ float half2float_q51(unsigned short h) {
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

extern "C" __global__ void matmul_q5_1(
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
    long rowStride = (long)numBlocks * 24;
    float sum = 0.0f;

    for (int b = lane; b < numBlocks; b += 32) {
        long bo = (long)row * rowStride + (long)b * 24;

        // scale + min: aligned uint32 read (4 bytes)
        unsigned int sm = __ldg((const unsigned int*)(weights + bo));
        float scale = half2float_q51(sm & 0xFFFF);
        float min = half2float_q51(sm >> 16);

        // qh: aligned uint32 read at offset 4
        unsigned int qh = __ldg((const unsigned int*)(weights + bo + 4));

        int inputBase = b * 32;

        // 16 packed bytes -> 32 elements (split layout)
        // Read as 4 uint32 chunks for efficiency
        unsigned int packed0 = __ldg((const unsigned int*)(weights + bo + 8));
        unsigned int packed1 = __ldg((const unsigned int*)(weights + bo + 12));
        unsigned int packed2 = __ldg((const unsigned int*)(weights + bo + 16));
        unsigned int packed3 = __ldg((const unsigned int*)(weights + bo + 20));

        #pragma unroll
        for (int j = 0; j < 16; j++) {
            unsigned int packed;
            int shift = (j & 3) * 8;
            if (j < 4) packed = (packed0 >> shift) & 0xFF;
            else if (j < 8) packed = (packed1 >> shift) & 0xFF;
            else if (j < 12) packed = (packed2 >> shift) & 0xFF;
            else packed = (packed3 >> shift) & 0xFF;

            int lo4 = packed & 0x0F;
            int hi4 = (packed >> 4) & 0x0F;

            int q0 = lo4 | (((qh >> j) & 1) << 4);
            int q1 = hi4 | (((qh >> (j + 16)) & 1) << 4);

            sum += (scale * (float)q0 + min) * input[inputBase + j];
            sum += (scale * (float)q1 + min) * input[inputBase + 16 + j];
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
