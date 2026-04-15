/**
 * IQ4_NL × Q8_1 matrix-vector multiply using __dp4a int8 dot product.
 *
 * IQ4_NL layout (18 bytes per 32-element block):
 *   bo+0..1   : FP16 scale d
 *   bo+2..17  : 16 ql bytes (low+high nibbles)
 *
 * Element mapping (SPLIT, like Q5_0):
 *   element  i (0..15)  = KVALUES_IQ4NL[ ql[i].LOW  ]
 *   element 16+i (0..15) = KVALUES_IQ4NL[ ql[i].HIGH ]
 *
 * KVALUES_IQ4NL: 16-entry signed int8 lookup table (range -127..113).
 *
 * dp4a: pack 4 K-mean values into int32 (each is int8, fits naturally), dp4a against
 * 4 Q8_1 input bytes. Per block: 8 dp4a calls (4 low + 4 high).
 *
 * IQ4_NL block size 18 is NOT 4-byte aligned; use byte loads only.
 */
__device__ __constant__ signed char KVALUES_IQ4NL_DP4A[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};

extern "C" __global__ void matmul_iq4_nl_dp4a(
    const unsigned char* __restrict__ weights,
    const unsigned char* __restrict__ input,    // Q8_1 (40-byte blocks)
    float* __restrict__ output,
    const int rows, const int cols, const int addToOutput)
{
    int warpId = threadIdx.x / 32;
    int lane   = threadIdx.x & 31;
    int rowsPerBlock = blockDim.x / 32;
    int row = blockIdx.x * rowsPerBlock + warpId;
    if (row >= rows) return;

    int numBlocks = cols / 32;
    long rowOffset = (long)row * numBlocks * 18;
    float sum = 0.0f;

    for (int b = lane; b < numBlocks; b += 32) {
        long bo = rowOffset + (long)b * 18;

        // FP16 scale (byte-load, 18 not aligned)
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

        long q8Off = (long)b * 40;
        float inScale = *(const float*)(input + q8Off);

        int dpAccum = 0;

        // 4 chunks of 4 ql bytes = 4 elements low + 4 elements high per chunk
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int ql0 = (int)__ldg(weights + bo + 2 + j*4);
            int ql1 = (int)__ldg(weights + bo + 2 + j*4 + 1);
            int ql2 = (int)__ldg(weights + bo + 2 + j*4 + 2);
            int ql3 = (int)__ldg(weights + bo + 2 + j*4 + 3);

            // Low nibbles → elements j*4 .. j*4+3
            int q0_lo = (int)KVALUES_IQ4NL_DP4A[ql0 & 0x0F];
            int q1_lo = (int)KVALUES_IQ4NL_DP4A[ql1 & 0x0F];
            int q2_lo = (int)KVALUES_IQ4NL_DP4A[ql2 & 0x0F];
            int q3_lo = (int)KVALUES_IQ4NL_DP4A[ql3 & 0x0F];
            int qPackLo = (q0_lo & 0xFF) | ((q1_lo & 0xFF) << 8)
                        | ((q2_lo & 0xFF) << 16) | ((q3_lo & 0xFF) << 24);
            int inLo = *(const int*)(input + q8Off + 8 + j*4);
            dpAccum = __dp4a(qPackLo, inLo, dpAccum);

            // High nibbles → elements 16+j*4 .. 16+j*4+3
            int q0_hi = (int)KVALUES_IQ4NL_DP4A[(ql0 >> 4) & 0x0F];
            int q1_hi = (int)KVALUES_IQ4NL_DP4A[(ql1 >> 4) & 0x0F];
            int q2_hi = (int)KVALUES_IQ4NL_DP4A[(ql2 >> 4) & 0x0F];
            int q3_hi = (int)KVALUES_IQ4NL_DP4A[(ql3 >> 4) & 0x0F];
            int qPackHi = (q0_hi & 0xFF) | ((q1_hi & 0xFF) << 8)
                        | ((q2_hi & 0xFF) << 16) | ((q3_hi & 0xFF) << 24);
            int inHi = *(const int*)(input + q8Off + 8 + 16 + j*4);
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
