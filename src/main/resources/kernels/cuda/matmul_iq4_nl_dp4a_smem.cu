/**
 * IQ4_NL × Q8_1 matmul using __dp4a + shared-memory input cache.
 *
 * Target: Phi-3-mini IQ4_NL where QKV/GateUp/siluDown take 78 ms/tok combined
 * (the single biggest contributor to the 19% of llama.cpp gap). The underlying
 * matmul_iq4_nl_dp4a.cu reads Q8_1 from HBM per warp; with rowsPerBlock warps
 * all operating on the same input vector, 7 of 8 reads are wasted.
 *
 * IQ4_NL layout (18 B per 32-elem block):
 *   bo+0..1  : FP16 scale d
 *   bo+2..17 : 16 ql bytes (SPLIT nibble: low→0..15, high→16..31)
 * Non-linear K-means codebook: KVALUES_IQ4NL[16] signed int8.
 *
 * Smem footprint: (cols / 32) × 40 bytes of Q8_1 per block.
 *   Phi-3-mini dim=3072 → 3840 B.  ffn=8192 → 10240 B.  Under 48KB default.
 */
__device__ __constant__ signed char KVALUES_IQ4NL_SMEM[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};

extern "C" __global__ void matmul_iq4_nl_dp4a_smem(
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
    int q8Bytes = numBlocks * 40;

    // Cooperative 4-byte load Q8_1 into shared memory.
    for (int i = (int)(threadIdx.x * 4); i < q8Bytes; i += (int)(blockDim.x * 4)) {
        *(int*)(sInput + i) = *(const int*)(input + i);
    }
    __syncthreads();

    if (row >= rows) return;

    long rowOffset = (long)row * numBlocks * 18;
    float sum = 0.0f;

    for (int b = lane; b < numBlocks; b += 32) {
        long bo = rowOffset + (long)b * 18;

        // FP16 scale (18B block not 4-aligned → byte loads only)
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

        int q8Off = b * 40;
        float inScale = *(const float*)(sInput + q8Off);

        int dpAccum = 0;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int ql0 = (int)__ldg(weights + bo + 2 + j*4);
            int ql1 = (int)__ldg(weights + bo + 2 + j*4 + 1);
            int ql2 = (int)__ldg(weights + bo + 2 + j*4 + 2);
            int ql3 = (int)__ldg(weights + bo + 2 + j*4 + 3);

            int q0_lo = (int)KVALUES_IQ4NL_SMEM[ql0 & 0x0F];
            int q1_lo = (int)KVALUES_IQ4NL_SMEM[ql1 & 0x0F];
            int q2_lo = (int)KVALUES_IQ4NL_SMEM[ql2 & 0x0F];
            int q3_lo = (int)KVALUES_IQ4NL_SMEM[ql3 & 0x0F];
            int qPackLo = (q0_lo & 0xFF) | ((q1_lo & 0xFF) << 8)
                        | ((q2_lo & 0xFF) << 16) | ((q3_lo & 0xFF) << 24);
            int inLo = *(const int*)(sInput + q8Off + 8 + j*4);
            dpAccum = __dp4a(qPackLo, inLo, dpAccum);

            int q0_hi = (int)KVALUES_IQ4NL_SMEM[(ql0 >> 4) & 0x0F];
            int q1_hi = (int)KVALUES_IQ4NL_SMEM[(ql1 >> 4) & 0x0F];
            int q2_hi = (int)KVALUES_IQ4NL_SMEM[(ql2 >> 4) & 0x0F];
            int q3_hi = (int)KVALUES_IQ4NL_SMEM[(ql3 >> 4) & 0x0F];
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
