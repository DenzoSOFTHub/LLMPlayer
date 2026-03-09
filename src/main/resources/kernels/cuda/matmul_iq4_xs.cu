/**
 * IQ4_XS dequantize + matrix-vector multiply kernel.
 * 256 weights per super-block, 136 bytes per block.
 * Layout: [d:fp16 (2B)][scales_h:u16 (2B)][scales_l (4B)][qs (128B)]
 * Sub-block scale: ls = low4 | (high2 << 4), effective = d * (ls - 32)
 * Nibble lookup via KVALUES_IQ4NL table. Nibble layout: low=even, high=odd per byte.
 * Each warp (32 threads) computes one output row, striping across super-blocks.
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

__device__ __constant__ float KVALUES_IQ4NL_XS[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
       1.f,   13.f,  25.f,  38.f,  53.f,  69.f,  89.f, 113.f
};

extern "C" __global__ void matmul_iq4_xs(
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
    int rowStride = numBlocks * 136;
    float sum = 0.0f;

    for (int b = lane; b < numBlocks; b += 32) {
        int bo = row * rowStride + b * 136;
        float d = half2float(*(unsigned short*)(weights + bo));
        unsigned short scalesH = *(unsigned short*)(weights + bo + 2);
        int inputBase = b * 256;

        // Process 8 sub-blocks of 32 elements each
        for (int ib = 0; ib < 8; ib++) {
            // Decode 6-bit sub-block scale
            unsigned char scalesLByte = __ldg(weights + bo + 4 + ib / 2);
            int low4 = (ib & 1) ? ((scalesLByte >> 4) & 0x0F) : (scalesLByte & 0x0F);
            int high2 = (scalesH >> (2 * ib)) & 3;
            int ls = low4 | (high2 << 4);
            float dl = d * (float)(ls - 32);

            int qsBase = bo + 8 + ib * 16;
            int subBase = inputBase + ib * 32;

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                unsigned char nibbles = __ldg(weights + qsBase + i);
                int lo = nibbles & 0x0F;
                int hi = (nibbles >> 4) & 0x0F;
                sum += dl * KVALUES_IQ4NL_XS[lo] * input[subBase + i];
                sum += dl * KVALUES_IQ4NL_XS[hi] * input[subBase + 16 + i];
            }
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
