/**
 * Q4_K matmul with shared-memory input tiling.
 * Combines coalesced weight reads (100% cache line utilization) with
 * shared-memory input caching (immune to L2 eviction from weight reads).
 *
 * Strategy: iterate over Q4_K super-blocks, caching 256 input floats in shared memory.
 * All 32 warp threads process the SAME group for coalesced weight reads,
 * with all 4 groups unrolled per super-block for compute density.
 *
 * Block: 256 threads (8 warps, 8 rows). Shared: 256 floats (1 KB) per super-block tile.
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

extern "C" __global__ void matmul_q4_k_smem(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    const int rows,
    const int cols,
    const int addToOutput)
{
    __shared__ float tile[256];  // input tile (1 super-block = 256 floats)

    int warpId = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int rowsPerBlock = blockDim.x / 32;
    int row = blockIdx.x * rowsPerBlock + warpId;

    int numBlocks = cols / 256;
    long rowOffset = (row < rows) ? (long)row * numBlocks * 144 : 0;
    float sum = 0.0f;

    for (int b = 0; b < numBlocks; b++) {
        // Phase 1: Cooperatively load 256 input floats into shared memory
        // 256 threads load 256 floats — one per thread, perfectly coalesced
        tile[threadIdx.x] = input[b * 256 + threadIdx.x];
        __syncthreads();

        if (row >= rows) { __syncthreads(); continue; }

        // Phase 2: Each warp processes its row's super-block b
        long bo = rowOffset + (long)b * 144;

        // Load block header — all threads read same address (L1 broadcast)
        unsigned int dm = __ldg((const unsigned int*)(weights + bo));
        float d = half2float(dm & 0xFFFF);
        float dmin = half2float(dm >> 16);

        // Load 12 scale bytes — broadcast
        unsigned int sc0 = __ldg((const unsigned int*)(weights + bo + 4));
        unsigned int sc1 = __ldg((const unsigned int*)(weights + bo + 8));
        unsigned int sc2 = __ldg((const unsigned int*)(weights + bo + 12));

        unsigned char sb[12];
        sb[0] = sc0 & 0xFF; sb[1] = (sc0 >> 8) & 0xFF;
        sb[2] = (sc0 >> 16) & 0xFF; sb[3] = (sc0 >> 24) & 0xFF;
        sb[4] = sc1 & 0xFF; sb[5] = (sc1 >> 8) & 0xFF;
        sb[6] = (sc1 >> 16) & 0xFF; sb[7] = (sc1 >> 24) & 0xFF;
        sb[8] = sc2 & 0xFF; sb[9] = (sc2 >> 8) & 0xFF;
        sb[10] = (sc2 >> 16) & 0xFF; sb[11] = (sc2 >> 24) & 0xFF;

        // Process all 4 groups with coalesced weight reads + shared memory input
        #pragma unroll
        for (int group = 0; group < 4; group++) {
            int sb0 = group * 2;
            int sb1 = group * 2 + 1;
            int scale0, min0, scale1, min1;
            if (sb0 < 4) {
                scale0 = sb[sb0] & 0x3F;
                min0 = sb[sb0 + 4] & 0x3F;
            } else {
                scale0 = (sb[sb0 + 4] & 0x0F) | ((sb[sb0 - 4] >> 6) << 4);
                min0 = ((sb[sb0 + 4] >> 4) & 0x0F) | ((sb[sb0] >> 6) << 4);
            }
            if (sb1 < 4) {
                scale1 = sb[sb1] & 0x3F;
                min1 = sb[sb1 + 4] & 0x3F;
            } else {
                scale1 = (sb[sb1 + 4] & 0x0F) | ((sb[sb1 - 4] >> 6) << 4);
                min1 = ((sb[sb1 + 4] >> 4) & 0x0F) | ((sb[sb1] >> 6) << 4);
            }

            float ds0 = d * (float)scale0;
            float dm0 = dmin * (float)min0;
            float ds1 = d * (float)scale1;
            float dm1 = dmin * (float)min1;

            // COALESCED: 32 threads read 32 consecutive weight bytes → 1 cache line
            unsigned char qByte = __ldg(&weights[bo + 16 + group * 32 + lane]);
            float q0 = (float)(qByte & 0x0F);
            float q1 = (float)((qByte >> 4) & 0x0F);

            // SHARED MEMORY: guaranteed fast, no L2 contention
            float in0 = tile[group * 64 + lane];
            float in1 = tile[group * 64 + 32 + lane];

            sum += (ds0 * q0 - dm0) * in0;
            sum += (ds1 * q1 - dm1) * in1;
        }
        __syncthreads();
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (lane == 0 && row < rows) {
        if (addToOutput) output[row] += sum;
        else output[row] = sum;
    }
}
