/**
 * Q6_K matmul with TILED shared memory input.
 * Loads input in 256-element tiles (1KB) into shared memory, matching Q6_K super-block size.
 * All warps in the block cooperatively load each tile, then process their rows.
 * Eliminates redundant global memory reads for the input vector without requiring
 * cols*4 bytes of shared memory (which fails for large cols like 9216).
 *
 * Block: 256 threads (8 warps = 8 rows per block)
 * Shared mem: 256 * 4 = 1024 bytes (one Q6_K super-block worth of input)
 *
 * Enabled via -Dcuda.q6k.tiled=true
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

extern "C" __global__ void matmul_q6_k_tiled(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    const int rows,
    const int cols,
    const int addToOutput)
{
    __shared__ float sInput[256];  // One tile = 256 floats = 1 Q6_K super-block

    int warpId = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int rowsPerBlock = blockDim.x / 32;
    int row = blockIdx.x * rowsPerBlock + warpId;

    int numBlocks = cols / 256;
    int rowStride = numBlocks * 210;
    float sum = 0.0f;

    // Process one Q6_K super-block at a time
    for (int b = 0; b < numBlocks; b++) {
        // Cooperatively load 256 input floats for this super-block
        int inputBase = b * 256;
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            sInput[i] = input[inputBase + i];
        }
        __syncthreads();

        if (row < rows) {
            long bo = (long)row * rowStride + b * 210;
            float d = half2float(__ldg((const unsigned short*)(weights + bo + 208)));

            // Process both halves of this super-block
            for (int hf = 0; hf < 2; hf++) {
                int qlOff = hf * 64;
                int qhOff = hf * 32;
                int scBase = hf * 8;
                int sInputBase = hf * 128;

                unsigned char qlByte0 = __ldg(&weights[bo + qlOff + lane]);
                unsigned char qlByte1 = __ldg(&weights[bo + qlOff + 32 + lane]);
                unsigned char qhByte  = __ldg(&weights[bo + 128 + qhOff + lane]);

                int q0 = (qlByte0 & 0x0F)        | (((qhByte >> 0) & 3) << 4);
                int q1 = (qlByte1 & 0x0F)        | (((qhByte >> 2) & 3) << 4);
                int q2 = ((qlByte0 >> 4) & 0x0F) | (((qhByte >> 4) & 3) << 4);
                int q3 = ((qlByte1 >> 4) & 0x0F) | (((qhByte >> 6) & 3) << 4);

                int scIdx = scBase + (lane >> 4);
                float ds0 = d * (float)((signed char)__ldg(&weights[bo + 192 + scIdx]));
                float ds1 = d * (float)((signed char)__ldg(&weights[bo + 192 + scIdx + 2]));
                float ds2 = d * (float)((signed char)__ldg(&weights[bo + 192 + scIdx + 4]));
                float ds3 = d * (float)((signed char)__ldg(&weights[bo + 192 + scIdx + 6]));

                // Read from shared memory
                sum += ds0 * (float)(q0 - 32) * sInput[sInputBase + lane];
                sum += ds1 * (float)(q1 - 32) * sInput[sInputBase + 32 + lane];
                sum += ds2 * (float)(q2 - 32) * sInput[sInputBase + 64 + lane];
                sum += ds3 * (float)(q3 - 32) * sInput[sInputBase + 96 + lane];
            }
        }
        __syncthreads();  // Sync before loading next tile
    }

    if (row >= rows) return;

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (lane == 0) {
        if (addToOutput) output[row] += sum;
        else output[row] = sum;
    }
}
