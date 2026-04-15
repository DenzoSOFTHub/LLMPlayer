/**
 * IQ4_XS × Q8_1 matmul — MULTI-WARP version.
 *
 * IQ4_XS block = 136 bytes, 256 elements, 8 sub-blocks of 32.
 * Single-warp version only had (numSuperBlocks) × 1 lane working per super-block.
 * For Gemma-2-2B cols=2304 → 9 super-blocks → 9 of 32 lanes working. Catastrophic.
 *
 * Multi-warp version: 4 warps × 32 lanes = 128 threads per block, 1 row per block.
 * Each thread handles exactly one (super-block, sub-block) work unit (up to 128).
 *
 * For cols=2304: 9 × 8 = 72 work units → 72/128 = 56% utilization (much better than 9/32=28%).
 * For cols=4096: 16 × 8 = 128 work units → 100% utilization.
 * For cols=8192: 32 × 8 = 256 work units → 2 iters per thread, fully utilized.
 *
 * Cross-warp reduction in shared memory.
 */
#define NWARPS_IQXS 4
#define WARP_SIZE   32

__device__ __constant__ signed char KVALUES_IQ4NL_MW[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};

extern "C" __global__ void matmul_iq4_xs_dp4a_mw(
    const unsigned char* __restrict__ weights,
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    const int rows, const int cols, const int addToOutput)
{
    const int row = blockIdx.x;
    if (row >= rows) return;

    const int warpId = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int tid = threadIdx.x;                       // 0..127

    const int numSuperBlocks = cols / 256;
    const long rowOffset = (long)row * numSuperBlocks * 136;

    // Total work units = numSB * 8 sub-blocks
    const int totalWork = numSuperBlocks * 8;

    float tmp = 0.0f;

    for (int work = tid; work < totalWork; work += NWARPS_IQXS * WARP_SIZE) {
        const int b = work / 8;        // super-block index
        const int ib = work & 7;       // sub-block within super-block
        const long bo = rowOffset + (long)b * 136;

        // Header reads (always 4-byte aligned — 136 is mult of 4)
        unsigned short dBits = __ldg((const unsigned short*)(weights + bo));
        unsigned short scalesH = __ldg((const unsigned short*)(weights + bo + 2));
        unsigned int scalesL = __ldg((const unsigned int*)(weights + bo + 4));

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

        // Decode 6-bit sub-block scale
        int low4  = (int)((scalesL >> (ib * 4)) & 0x0F);
        int high2 = (int)((scalesH >> (ib * 2)) & 0x03);
        int ls = low4 | (high2 << 4);
        float dl = d * (float)(ls - 32);

        // Q8_1 block index: b * 8 + ib
        long q8Off = (long)(b * 8 + ib) * 40;
        float inScale = *(const float*)(input + q8Off);

        long qsBo = bo + 8 + (long)ib * 16;

        int dp = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int ql0 = (int)__ldg(weights + qsBo + j*4);
            int ql1 = (int)__ldg(weights + qsBo + j*4 + 1);
            int ql2 = (int)__ldg(weights + qsBo + j*4 + 2);
            int ql3 = (int)__ldg(weights + qsBo + j*4 + 3);

            int q0_lo = (int)KVALUES_IQ4NL_MW[ql0 & 0x0F];
            int q1_lo = (int)KVALUES_IQ4NL_MW[ql1 & 0x0F];
            int q2_lo = (int)KVALUES_IQ4NL_MW[ql2 & 0x0F];
            int q3_lo = (int)KVALUES_IQ4NL_MW[ql3 & 0x0F];
            int qPackLo = (q0_lo & 0xFF) | ((q1_lo & 0xFF) << 8)
                        | ((q2_lo & 0xFF) << 16) | ((q3_lo & 0xFF) << 24);
            int inLo = *(const int*)(input + q8Off + 8 + j*4);
            dp = __dp4a(qPackLo, inLo, dp);

            int q0_hi = (int)KVALUES_IQ4NL_MW[(ql0 >> 4) & 0x0F];
            int q1_hi = (int)KVALUES_IQ4NL_MW[(ql1 >> 4) & 0x0F];
            int q2_hi = (int)KVALUES_IQ4NL_MW[(ql2 >> 4) & 0x0F];
            int q3_hi = (int)KVALUES_IQ4NL_MW[(ql3 >> 4) & 0x0F];
            int qPackHi = (q0_hi & 0xFF) | ((q1_hi & 0xFF) << 8)
                        | ((q2_hi & 0xFF) << 16) | ((q3_hi & 0xFF) << 24);
            int inHi = *(const int*)(input + q8Off + 8 + 16 + j*4);
            dp = __dp4a(qPackHi, inHi, dp);
        }

        tmp += dl * inScale * (float)dp;
    }

    // Within-warp reduction
    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        tmp += __shfl_down_sync(0xFFFFFFFF, tmp, off);
    }

    // Cross-warp via shared memory
    __shared__ float warp_sums[NWARPS_IQXS];
    if (lane == 0) warp_sums[warpId] = tmp;
    __syncthreads();

    if (warpId == 0) {
        float acc = (lane < NWARPS_IQXS) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int off = NWARPS_IQXS / 2; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFF, acc, off);
        }
        if (lane == 0) {
            if (addToOutput) output[row] += acc;
            else             output[row]  = acc;
        }
    }
}
