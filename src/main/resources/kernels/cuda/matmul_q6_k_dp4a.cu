/**
 * Q6_K × Q8_1 matrix-vector multiply using __dp4a intrinsic.
 * Q6_K: [ql:128B][qh:64B][scales:16B][d:fp16] per 256-element super-block.
 * q6 = (ql_nibble | (qh_2bits << 4)), range 0-63; value = d * scale * (q6 - 32)
 *
 * Each half-block (128 elements) has 4 sub-groups of 32. Each sub-group has TWO 16-element
 * scale zones. We split dp4a into two halves per sub-group to apply correct scales.
 *
 * Dot: d * sc * (inScale * dp4a(q6, in_q8) - 32 * inSum)
 */
__device__ __forceinline__ float half2float(unsigned short h) {
    unsigned int sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, mantissa = h & 0x3FF;
    if (exp == 0) { if (mantissa == 0) return sign ? -0.0f : 0.0f; while (!(mantissa & 0x400)) { mantissa <<= 1; exp--; } exp++; mantissa &= 0x3FF; }
    else if (exp == 31) { unsigned int f = (sign << 31) | 0x7F800000 | (mantissa << 13); return *(float*)&f; }
    unsigned int f = (sign << 31) | ((exp + 112) << 23) | (mantissa << 13); return *(float*)&f;
}

extern "C" __global__ void matmul_q6_k_dp4a(
    const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    const int rows, const int cols, const int addToOutput)
{
    int warpId = threadIdx.x / 32, lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x / 32) + warpId;
    if (row >= rows) return;

    int nSB = cols / 256;
    long rowOff = (long)row * nSB * 210;
    int nHalves = nSB * 2;
    float sum = 0.0f;

    // Process half-blocks. Each half = 128 elements = 4 sub-groups of 32.
    // Each lane handles one half-block at a time (striped across lanes).
    for (int h = lane; h < nHalves; h += 32) {
        int sb = h >> 1;
        int hf = h & 1;
        long bo = rowOff + (long)sb * 210;

        float d = half2float(__ldg((const unsigned short*)(W + bo + 208)));

        // For each of 4 sub-groups in this half:
        for (int sg = 0; sg < 4; sg++) {
            int scBase = hf * 8 + sg * 2;
            float sc_lo = d * (float)((signed char)__ldg(&W[bo + 192 + scBase]));
            float sc_hi = d * (float)((signed char)__ldg(&W[bo + 192 + scBase + 1]));

            // Elements: [sb*256 + hf*128 + sg*32 .. +31]
            // These align with one Q8_1 block
            int elemStart = sb * 256 + hf * 128 + sg * 32;
            long q8off = (long)(elemStart / 32) * 40;
            float inScale = *(const float*)(input + q8off);
            float inSum   = *(const float*)(input + q8off + 4);

            // First 16 elements (scale = sc_lo): dp4a 4 groups of 4
            int dp_lo = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                int p = k * 4; // position 0-15 within sub-group
                // Extract 4 q6 values
                unsigned char ql0 = W[bo + hf*64 + sg*16 + p];
                unsigned char ql1 = W[bo + hf*64 + sg*16 + p+1];
                unsigned char ql2 = W[bo + hf*64 + sg*16 + p+2];
                unsigned char ql3 = W[bo + hf*64 + sg*16 + p+3];
                unsigned char qh0 = W[bo + 128 + hf*32 + p];
                unsigned char qh1 = W[bo + 128 + hf*32 + p+1];
                unsigned char qh2 = W[bo + 128 + hf*32 + p+2];
                unsigned char qh3 = W[bo + 128 + hf*32 + p+3];

                int q0, q1, q2, q3;
                if (sg < 2) {
                    q0 = (ql0 & 0x0F) | (((qh0 >> (sg*2)) & 3) << 4);
                    q1 = (ql1 & 0x0F) | (((qh1 >> (sg*2)) & 3) << 4);
                    q2 = (ql2 & 0x0F) | (((qh2 >> (sg*2)) & 3) << 4);
                    q3 = (ql3 & 0x0F) | (((qh3 >> (sg*2)) & 3) << 4);
                } else {
                    q0 = (ql0 & 0x0F) | (((qh0 >> (4+(sg-2)*2)) & 3) << 4);
                    q1 = (ql1 & 0x0F) | (((qh1 >> (4+(sg-2)*2)) & 3) << 4);
                    q2 = (ql2 & 0x0F) | (((qh2 >> (4+(sg-2)*2)) & 3) << 4);
                    q3 = (ql3 & 0x0F) | (((qh3 >> (4+(sg-2)*2)) & 3) << 4);
                }

                int q_packed = (q0&0xFF) | ((q1&0xFF)<<8) | ((q2&0xFF)<<16) | ((q3&0xFF)<<24);
                int in4 = *(const int*)(input + q8off + 8 + k*4);
                dp_lo = __dp4a(q_packed, in4, dp_lo);
            }

            // Last 16 elements (scale = sc_hi): dp4a 4 groups of 4
            int dp_hi = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                int p = k * 4; // position 0-15, but these are elements 16-31 in sub-group
                unsigned char ql0 = W[bo + hf*64 + sg*16 + p]; // same ql byte, HIGH nibble
                unsigned char ql1 = W[bo + hf*64 + sg*16 + p+1];
                unsigned char ql2 = W[bo + hf*64 + sg*16 + p+2];
                unsigned char ql3 = W[bo + hf*64 + sg*16 + p+3];
                unsigned char qh0 = W[bo + 128 + hf*32 + p];
                unsigned char qh1 = W[bo + 128 + hf*32 + p+1];
                unsigned char qh2 = W[bo + 128 + hf*32 + p+2];
                unsigned char qh3 = W[bo + 128 + hf*32 + p+3];

                int q0, q1, q2, q3;
                if (sg < 2) {
                    q0 = ((ql0>>4)&0x0F) | (((qh0 >> (sg*2+2)) & 3) << 4);
                    q1 = ((ql1>>4)&0x0F) | (((qh1 >> (sg*2+2)) & 3) << 4);
                    q2 = ((ql2>>4)&0x0F) | (((qh2 >> (sg*2+2)) & 3) << 4);
                    q3 = ((ql3>>4)&0x0F) | (((qh3 >> (sg*2+2)) & 3) << 4);
                } else {
                    q0 = ((ql0>>4)&0x0F) | (((qh0 >> (4+(sg-2)*2+2)) & 3) << 4);
                    q1 = ((ql1>>4)&0x0F) | (((qh1 >> (4+(sg-2)*2+2)) & 3) << 4);
                    q2 = ((ql2>>4)&0x0F) | (((qh2 >> (4+(sg-2)*2+2)) & 3) << 4);
                    q3 = ((ql3>>4)&0x0F) | (((qh3 >> (4+(sg-2)*2+2)) & 3) << 4);
                }

                int q_packed = (q0&0xFF) | ((q1&0xFF)<<8) | ((q2&0xFF)<<16) | ((q3&0xFF)<<24);
                int in4 = *(const int*)(input + q8off + 8 + 16 + k*4); // offset by 16 for second half
                dp_hi = __dp4a(q_packed, in4, dp_hi);
            }

            // Accumulate: sc * (inScale * dp_sum - 32 * inSum)
            // Split inSum equally between the two halves
            float halfInSum = inSum * 0.5f;
            sum += sc_lo * (inScale * (float)dp_lo - 32.0f * halfInSum);
            sum += sc_hi * (inScale * (float)dp_hi - 32.0f * halfInSum);
        }
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    if (lane == 0) {
        if (addToOutput) output[row] += sum;
        else output[row] = sum;
    }
}
