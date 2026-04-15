/**
 * IQ4_XS × Q8_1 matrix-vector multiply using __dp4a int8 dot product.
 *
 * IQ4_XS layout (136 bytes per 256-element super-block — 4-byte aligned):
 *   bo+0..1   : FP16 d (super-block scale)
 *   bo+2..3   : scalesH (uint16, 2 high-bits per sub-block × 8 sub-blocks)
 *   bo+4..7   : scalesL (4 low-bits per sub-block × 8 sub-blocks = 4 bytes)
 *   bo+8..135 : 128 bytes qs (8 sub-blocks × 16 ql bytes each)
 *
 * Per sub-block:
 *   ls = (scalesL_4bits) | (scalesH_2bits << 4)   (6-bit, range 0..63)
 *   dl = d * (ls - 32)
 *   32 elements, SPLIT layout (low nibbles → 0..15, high → 16..31)
 *
 * Each Q8_1 input block covers 32 elements = 1 sub-block. So 8 Q8_1 blocks per super-block.
 *
 * Per super-block: 8 sub-blocks × 8 dp4a (4 low + 4 high) = 64 dp4a per lane.
 */
__device__ __constant__ signed char KVALUES_IQ4NL_XS_DP4A[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};

extern "C" __global__ void matmul_iq4_xs_dp4a(
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

    int numSuperBlocks = cols / 256;
    long rowOffset = (long)row * numSuperBlocks * 136;
    float sum = 0.0f;

    for (int b = lane; b < numSuperBlocks; b += 32) {
        long bo = rowOffset + (long)b * 136;

        // Header reads (136 is 4-byte aligned, so all uint reads are safe)
        unsigned short dBits = __ldg((const unsigned short*)(weights + bo));
        unsigned short scalesH = __ldg((const unsigned short*)(weights + bo + 2));
        unsigned int scalesL = __ldg((const unsigned int*)(weights + bo + 4));

        // FP16 → FP32 d
        unsigned int sign = (dBits >> 15) & 1;
        unsigned int exp = (dBits >> 10) & 0x1F;
        unsigned int mant = dBits & 0x3FF;
        float d;
        if (exp == 0) {
            if (mant == 0) d = sign ? -0.0f : 0.0f;
            else {
                while (!(mant & 0x400)) { mant <<= 1; exp--; }
                exp++; mant &= 0x3FF;
                unsigned int f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                d = *(float*)&f;
            }
        } else if (exp == 31) {
            unsigned int f = (sign << 31) | 0x7F800000 | (mant << 13);
            d = *(float*)&f;
        } else {
            unsigned int f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
            d = *(float*)&f;
        }

        // Process 8 sub-blocks per super-block
        #pragma unroll
        for (int ib = 0; ib < 8; ib++) {
            // Decode 6-bit sub-block scale
            int low4  = (int)((scalesL >> (ib * 4)) & 0x0F);
            int high2 = (int)((scalesH >> (ib * 2)) & 0x03);
            int ls = low4 | (high2 << 4);
            float dl = d * (float)(ls - 32);

            // Q8_1 block index: b * 8 + ib (8 Q8_1 blocks per super-block)
            long q8Off = (long)(b * 8 + ib) * 40;
            float inScale = *(const float*)(input + q8Off);

            // ql for this sub-block at bo + 8 + ib*16, length 16 bytes
            long qsBo = bo + 8 + (long)ib * 16;

            int dp = 0;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int ql0 = (int)__ldg(weights + qsBo + j*4);
                int ql1 = (int)__ldg(weights + qsBo + j*4 + 1);
                int ql2 = (int)__ldg(weights + qsBo + j*4 + 2);
                int ql3 = (int)__ldg(weights + qsBo + j*4 + 3);

                // Low nibbles → elements j*4 .. j*4+3 of sub-block
                int q0_lo = (int)KVALUES_IQ4NL_XS_DP4A[ql0 & 0x0F];
                int q1_lo = (int)KVALUES_IQ4NL_XS_DP4A[ql1 & 0x0F];
                int q2_lo = (int)KVALUES_IQ4NL_XS_DP4A[ql2 & 0x0F];
                int q3_lo = (int)KVALUES_IQ4NL_XS_DP4A[ql3 & 0x0F];
                int qPackLo = (q0_lo & 0xFF) | ((q1_lo & 0xFF) << 8)
                            | ((q2_lo & 0xFF) << 16) | ((q3_lo & 0xFF) << 24);
                int inLo = *(const int*)(input + q8Off + 8 + j*4);
                dp = __dp4a(qPackLo, inLo, dp);

                // High nibbles → elements 16+j*4 .. 16+j*4+3
                int q0_hi = (int)KVALUES_IQ4NL_XS_DP4A[(ql0 >> 4) & 0x0F];
                int q1_hi = (int)KVALUES_IQ4NL_XS_DP4A[(ql1 >> 4) & 0x0F];
                int q2_hi = (int)KVALUES_IQ4NL_XS_DP4A[(ql2 >> 4) & 0x0F];
                int q3_hi = (int)KVALUES_IQ4NL_XS_DP4A[(ql3 >> 4) & 0x0F];
                int qPackHi = (q0_hi & 0xFF) | ((q1_hi & 0xFF) << 8)
                            | ((q2_hi & 0xFF) << 16) | ((q3_hi & 0xFF) << 24);
                int inHi = *(const int*)(input + q8Off + 8 + 16 + j*4);
                dp = __dp4a(qPackHi, inHi, dp);
            }

            sum += dl * inScale * (float)dp;
        }
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    if (lane == 0) {
        if (addToOutput) output[row] += sum;
        else             output[row]  = sum;
    }
}
