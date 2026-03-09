/**
 * IQ4_NL dequantize + matrix-vector multiply kernel.
 * 32 weights per block, 18 bytes per block.
 * Layout: [scale:fp16 (2B)][16 bytes nibbles]
 * Uses non-linear lookup table: value = scale * KVALUES_IQ4NL[nibble]
 * Split nibble layout: low nibbles → positions 0-15, high nibbles → positions 16-31
 * Each warp (32 threads) computes one output row, striping across blocks.
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

__device__ __constant__ float KVALUES_IQ4NL[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
       1.f,   13.f,  25.f,  38.f,  53.f,  69.f,  89.f, 113.f
};

extern "C" __global__ void matmul_iq4_nl(
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
    int rowStride = numBlocks * 18;
    float sum = 0.0f;

    for (int b = lane; b < numBlocks; b += 32) {
        int bo = row * rowStride + b * 18;
        float scale = half2float(*(unsigned short*)(weights + bo));
        int inputBase = b * 32;

        // Split nibble layout: low nibbles → positions 0-15, high → positions 16-31
        #pragma unroll
        for (int l = 0; l < 16; l++) {
            unsigned char nibbles = __ldg(weights + bo + 2 + l);
            int lo = nibbles & 0x0F;
            int hi = (nibbles >> 4) & 0x0F;
            sum += scale * KVALUES_IQ4NL[lo] * input[inputBase + l];
            sum += scale * KVALUES_IQ4NL[hi] * input[inputBase + l + 16];
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
