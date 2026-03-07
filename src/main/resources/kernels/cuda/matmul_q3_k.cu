/**
 * Q3_K dequantize + matrix-vector multiply kernel.
 * 256 weights per super-block, 110 bytes per block.
 * Layout: [hmask:32B][qs:64B][scales:12B][d:fp16]
 *
 * Threads stripe across sub-blocks (8 per block = 2 halves x 4 pairs).
 * Optimized scale decoding: only 2 scales computed per sub-block (not all 16).
 * Uses __ldg for read-only texture cache access.
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

/**
 * Decode one Q3_K 6-bit scale from the 12 raw scale bytes.
 * Scale index s ranges 0..15. Reads only 2 bytes instead of all 12.
 */
__device__ __forceinline__ int decodeQ3KScale(const unsigned char* __restrict__ scaleBytes, int s) {
    int lo = (s < 8) ? (__ldg(&scaleBytes[s & 7]) & 0x0F) : (__ldg(&scaleBytes[s & 7]) >> 4);
    int hi = (__ldg(&scaleBytes[8 + (s & 3)]) >> ((s >> 2) * 2)) & 0x03;
    return lo | (hi << 4);
}

extern "C" __global__ void matmul_q3_k(
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
    int numUnits = numBlocks * 8;   // 8 sub-blocks per block (2 halves x 4 pairs)
    int rowStride = numBlocks * 110;
    float sum = 0.0f;

    for (int u = lane; u < numUnits; u += 32) {
        int b = u >> 3;          // block index
        int subIdx = u & 7;     // sub-block index (0-7)
        int hf = subIdx >> 2;   // half (0 or 1)
        int pair = subIdx & 3;  // pair within half (0-3)

        long bo = (long)row * rowStride + b * 110;

        float d = half2float(__ldg((const unsigned short*)(weights + bo + 108)));

        // Efficient scale decode: only the 2 needed scales
        int scaleIdx = hf * 8 + pair * 2;
        float dl0 = d * (float)(decodeQ3KScale(weights + bo + 96, scaleIdx) - 32);
        float dl1 = d * (float)(decodeQ3KScale(weights + bo + 96, scaleIdx + 1) - 32);

        int shift = pair * 2;
        int qBase = hf * 32;
        int inputBase = b * 256 + hf * 128 + pair * 32;
        int hmBitPos = hf * 4 + pair;

        // First 16 elements of this sub-block
        #pragma unroll 4
        for (int l = 0; l < 16; l++) {
            unsigned char qsByte = __ldg(&weights[bo + 32 + qBase + l]);
            unsigned char hmByte = __ldg(&weights[bo + l]);

            int lowBits = (qsByte >> shift) & 3;
            int hbit = (hmByte >> hmBitPos) & 1;
            int q = (lowBits | (hbit << 2)) - 4;

            sum += dl0 * (float)q * __ldg(&input[inputBase + l]);
        }

        // Second 16 elements of this sub-block
        #pragma unroll 4
        for (int l = 0; l < 16; l++) {
            unsigned char qsByte = __ldg(&weights[bo + 32 + qBase + 16 + l]);
            unsigned char hmByte = __ldg(&weights[bo + 16 + l]);

            int lowBits = (qsByte >> shift) & 3;
            int hbit = (hmByte >> hmBitPos) & 1;
            int q = (lowBits | (hbit << 2)) - 4;

            sum += dl1 * (float)q * __ldg(&input[inputBase + 16 + l]);
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
