/**
 * Q8_0 × Q8_1 matrix-vector multiply using __dp4a int8 dot product.
 *
 * Q8_0 layout (34 bytes per 32-element block):
 *   bo+0..1  : FP16 scale d
 *   bo+2..33 : 32 int8 quants (already signed int8 — no unpacking needed!)
 *
 * Q8_0 weights are already int8, so dp4a is direct — no nibble unpacking, no offsets.
 * Per block: 8 dp4a calls (4 weights per dp4a × 8 = 32 elements per block).
 *
 * Q8_0 block size 34 is NOT 4-byte aligned; use byte-loads OR 4-byte loads only when
 * we know offsets are aligned. Within-block qs starts at bo+2 (could be 4-aligned only
 * for even blocks). Safer: byte-load the qs into packed ints.
 */
extern "C" __global__ void matmul_q8_0_dp4a(
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
    long rowOffset = (long)row * numBlocks * 34;
    float sum = 0.0f;

    for (int b = lane; b < numBlocks; b += 32) {
        long bo = rowOffset + (long)b * 34;

        // FP16 scale d (byte-load — 34 not aligned)
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
        // 8 chunks of 4 weights each = 32 elements
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            // 4 weight bytes (signed int8), packed via byte-loads
            int w0 = (int)(signed char)__ldg(weights + bo + 2 + j*4);
            int w1 = (int)(signed char)__ldg(weights + bo + 2 + j*4 + 1);
            int w2 = (int)(signed char)__ldg(weights + bo + 2 + j*4 + 2);
            int w3 = (int)(signed char)__ldg(weights + bo + 2 + j*4 + 3);
            int wPack = (w0 & 0xFF) | ((w1 & 0xFF) << 8) | ((w2 & 0xFF) << 16) | ((w3 & 0xFF) << 24);

            // 4 Q8_1 input bytes at offset 8 + j*4
            int in4 = *(const int*)(input + q8Off + 8 + j*4);

            dpAccum = __dp4a(wPack, in4, dpAccum);
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
