/**
 * Q5_K matmul with shared memory input caching.
 * Loads full input vector into shared memory once per block.
 * Enabled via -Dcuda.q5k.smem=true
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

extern "C" __global__ void matmul_q5_k_smem(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    const int rows,
    const int cols,
    const int addToOutput)
{
    extern __shared__ float sInput[];

    int warpId = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int rowsPerBlock = blockDim.x / 32;
    int row = blockIdx.x * rowsPerBlock + warpId;

    // Cooperatively load input into shared memory
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sInput[i] = input[i];
    }
    __syncthreads();

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

        float ds0 = d * (float)scale0, dm0 = dmin * (float)min0;
        float ds1 = d * (float)scale1, dm1 = dmin * (float)min1;
        int inputBase = b * 256 + group * 64;

        #pragma unroll 8
        for (int l = 0; l < 32; l++) {
            unsigned char qsByte = __ldg(&weights[bo + 48 + group * 32 + l]);
            unsigned char qhByte = __ldg(&weights[bo + 16 + l]);

            int ql0 = qsByte & 0x0F;
            int ql1 = (qsByte >> 4) & 0x0F;
            int qh0 = (qhByte >> (group * 2)) & 1;
            int qh1 = (qhByte >> (group * 2 + 1)) & 1;
            int q0 = ql0 | (qh0 << 4);
            int q1 = ql1 | (qh1 << 4);

            // Read from shared memory
            sum += (ds0 * (float)q0 - dm0) * sInput[inputBase + l];
            sum += (ds1 * (float)q1 - dm1) * sInput[inputBase + 32 + l];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (lane == 0) {
        if (addToOutput) output[row] += sum;
        else output[row] = sum;
    }
}
