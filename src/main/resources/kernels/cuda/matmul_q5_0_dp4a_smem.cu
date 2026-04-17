/**
 * Q5_0 × Q8_1 matmul using __dp4a + shared-memory input cache.
 *
 * Target: Gemma-3-1B Q4_K_M where Q5_0 is used for Q/K/gate/up (per CLAUDE.md
 * note on Gemma-3 quant layout). Current matmul_q5_0_dp4a.cu reads Q8_1 bytes
 * directly from HBM; with 8 warps per block sharing the same input vector,
 * 7 of 8 reads are redundant. Caching Q8_1 in shared memory eliminates them.
 *
 * Q8_1 input layout: per 32-elem block, 40 bytes = [float scale][float sum][int8 qs[32]].
 * Shared-mem footprint per block: (cols / 32) × 40 bytes.
 *   Gemma-3-1B dim=1152 → 1440 B.  ffn=6912 → 8640 B.  Both under 48KB default.
 *
 * Q5_0 weight layout (22 B per 32-elem block): see matmul_q5_0_dp4a.cu.
 * Block is NOT 4-byte aligned; weight loads use byte __ldg only.
 *
 * 1 warp per row, lanes stride across blocks by 32.
 */
extern "C" __global__ void matmul_q5_0_dp4a_smem(
    const unsigned char* __restrict__ weights,
    const unsigned char* __restrict__ input,    // Q8_1 (40-byte blocks)
    float* __restrict__ output,
    const int rows, const int cols, const int addToOutput)
{
    extern __shared__ unsigned char sInput[];

    int warpId = threadIdx.x / 32;
    int lane   = threadIdx.x & 31;
    int rowsPerBlock = blockDim.x / 32;
    int row = blockIdx.x * rowsPerBlock + warpId;

    int numBlocks = cols / 32;
    int q8Bytes   = numBlocks * 40;

    // Cooperative 4-byte load into shared memory.
    // Q8_1 is naturally 4-aligned (40 mult of 4).
    for (int i = (int)(threadIdx.x * 4); i < q8Bytes; i += (int)(blockDim.x * 4)) {
        *(int*)(sInput + i) = *(const int*)(input + i);
    }
    __syncthreads();

    if (row >= rows) return;

    long rowOffset = (long)row * numBlocks * 22;
    float sum = 0.0f;

    for (int b = lane; b < numBlocks; b += 32) {
        long bo = rowOffset + (long)b * 22;

        // FP16 scale (byte-load, 22B block is not 4-aligned)
        unsigned short scaleBits = (unsigned short)__ldg(weights + bo)
                                 | ((unsigned short)__ldg(weights + bo + 1) << 8);
        unsigned int sign = (scaleBits >> 15) & 1;
        unsigned int exp = (scaleBits >> 10) & 0x1F;
        unsigned int mant = scaleBits & 0x3FF;
        float scale;
        if (exp == 0) {
            if (mant == 0) scale = sign ? -0.0f : 0.0f;
            else {
                while (!(mant & 0x400)) { mant <<= 1; exp--; }
                exp++; mant &= 0x3FF;
                unsigned int f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                scale = *(float*)&f;
            }
        } else if (exp == 31) {
            unsigned int f = (sign << 31) | 0x7F800000 | (mant << 13);
            scale = *(float*)&f;
        } else {
            unsigned int f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
            scale = *(float*)&f;
        }

        unsigned int qh = ((unsigned int)__ldg(weights + bo + 2))
                        | ((unsigned int)__ldg(weights + bo + 3) <<  8)
                        | ((unsigned int)__ldg(weights + bo + 4) << 16)
                        | ((unsigned int)__ldg(weights + bo + 5) << 24);

        int q8Off = b * 40;
        float inScale = *(const float*)(sInput + q8Off);

        int dpAccum = 0;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int ql0 = (int)__ldg(weights + bo + 6 + j*4);
            int ql1 = (int)__ldg(weights + bo + 6 + j*4 + 1);
            int ql2 = (int)__ldg(weights + bo + 6 + j*4 + 2);
            int ql3 = (int)__ldg(weights + bo + 6 + j*4 + 3);

            unsigned int qhLo = (qh >> (j*4)) & 0x0F;
            unsigned int qhHi = (qh >> (16 + j*4)) & 0x0F;

            int q0_lo = ((ql0 & 0x0F) | (((qhLo >> 0) & 1) << 4)) - 16;
            int q1_lo = ((ql1 & 0x0F) | (((qhLo >> 1) & 1) << 4)) - 16;
            int q2_lo = ((ql2 & 0x0F) | (((qhLo >> 2) & 1) << 4)) - 16;
            int q3_lo = ((ql3 & 0x0F) | (((qhLo >> 3) & 1) << 4)) - 16;
            int qPackLo = (q0_lo & 0xFF) | ((q1_lo & 0xFF) << 8)
                        | ((q2_lo & 0xFF) << 16) | ((q3_lo & 0xFF) << 24);
            int inLo = *(const int*)(sInput + q8Off + 8 + j*4);
            dpAccum = __dp4a(qPackLo, inLo, dpAccum);

            int q0_hi = (((ql0 >> 4) & 0x0F) | (((qhHi >> 0) & 1) << 4)) - 16;
            int q1_hi = (((ql1 >> 4) & 0x0F) | (((qhHi >> 1) & 1) << 4)) - 16;
            int q2_hi = (((ql2 >> 4) & 0x0F) | (((qhHi >> 2) & 1) << 4)) - 16;
            int q3_hi = (((ql3 >> 4) & 0x0F) | (((qhHi >> 3) & 1) << 4)) - 16;
            int qPackHi = (q0_hi & 0xFF) | ((q1_hi & 0xFF) << 8)
                        | ((q2_hi & 0xFF) << 16) | ((q3_hi & 0xFF) << 24);
            int inHi = *(const int*)(sInput + q8Off + 8 + 16 + j*4);
            dpAccum = __dp4a(qPackHi, inHi, dpAccum);
        }

        sum += scale * inScale * (float)dpAccum;
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    if (lane == 0) {
        if (addToOutput) output[row] += sum;
        else             output[row]  = sum;
    }
}
