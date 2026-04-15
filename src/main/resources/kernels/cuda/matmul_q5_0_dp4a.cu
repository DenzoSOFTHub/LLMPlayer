/**
 * Q5_0 × Q8_1 matrix-vector multiply using __dp4a int8 dot product.
 *
 * Q5_0 layout (22 bytes per 32-element block):
 *   bo+0..1  : FP16 scale d
 *   bo+2..5  : 4-byte qh (32 high bits — 1 bit per element)
 *   bo+6..21 : 16 bytes ql (low+high nibbles, SPLIT layout per existing memory note)
 *
 * Element mapping (SPLIT, NOT interleaved):
 *   element  i (0..15)  = (ql[i].LOW  | (qh bit i      << 4)) - 16
 *   element 16+i (0..15) = (ql[i].HIGH | (qh bit (16+i) << 4)) - 16
 *
 * dp4a: pack 4 consecutive (q-16) signed int8 weights into int32, dp4a against 4 Q8_1
 * input bytes. Per block: 8 dp4a calls (4 for low-half elements 0..15, 4 for high-half).
 *
 * Q5_0 block size 22 is NOT 4-byte aligned; use byte loads only.
 *
 * Threading: 1 warp per row, blockDim=256 (8 rows/block), each lane handles 1 Q5_0 block
 * stride 32 (matches our existing dp4a Q4_K orchestration).
 */
extern "C" __global__ void matmul_q5_0_dp4a(
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
    long rowOffset = (long)row * numBlocks * 22;
    float sum = 0.0f;

    for (int b = lane; b < numBlocks; b += 32) {
        long bo = rowOffset + (long)b * 22;

        // FP16 scale d (byte-load, no alignment guarantee)
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

        // qh as 4 bytes packed into uint
        unsigned int qh = ((unsigned int)__ldg(weights + bo + 2))
                        | ((unsigned int)__ldg(weights + bo + 3) <<  8)
                        | ((unsigned int)__ldg(weights + bo + 4) << 16)
                        | ((unsigned int)__ldg(weights + bo + 5) << 24);

        // Q8_1 input block index = b (Q8_1 has 32 elems per block, same as Q5_0)
        long q8Off = (long)b * 40;
        float inScale = *(const float*)(input + q8Off);

        int dpAccum = 0;

        // Process 16 ql bytes (positions 0..15). Each j chunk = 4 ql bytes = 4 elements
        // for the low half (elems 0..15) AND 4 elements for the high half (elems 16..31).
        // We do 4 chunks × (1 low dp4a + 1 high dp4a) = 8 dp4a per block.
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            // Read 4 ql bytes
            int ql0 = (int)__ldg(weights + bo + 6 + j*4);
            int ql1 = (int)__ldg(weights + bo + 6 + j*4 + 1);
            int ql2 = (int)__ldg(weights + bo + 6 + j*4 + 2);
            int ql3 = (int)__ldg(weights + bo + 6 + j*4 + 3);

            // Read 8 qh bits: 4 for low half (bits j*4..j*4+3), 4 for high half (bits 16+j*4..)
            unsigned int qhLo = (qh >> (j*4)) & 0x0F;          // 4 bits
            unsigned int qhHi = (qh >> (16 + j*4)) & 0x0F;     // 4 bits

            // === Low half: 4 elements at positions j*4 .. j*4+3 ===
            int q0_lo = ((ql0 & 0x0F) | (((qhLo >> 0) & 1) << 4)) - 16;
            int q1_lo = ((ql1 & 0x0F) | (((qhLo >> 1) & 1) << 4)) - 16;
            int q2_lo = ((ql2 & 0x0F) | (((qhLo >> 2) & 1) << 4)) - 16;
            int q3_lo = ((ql3 & 0x0F) | (((qhLo >> 3) & 1) << 4)) - 16;
            int qPackLo = (q0_lo & 0xFF) | ((q1_lo & 0xFF) << 8)
                        | ((q2_lo & 0xFF) << 16) | ((q3_lo & 0xFF) << 24);

            // 4 Q8_1 input bytes for low half at offset 8 + j*4
            int inLo = *(const int*)(input + q8Off + 8 + j*4);
            dpAccum = __dp4a(qPackLo, inLo, dpAccum);

            // === High half: 4 elements at positions 16+j*4 .. 16+j*4+3 ===
            int q0_hi = (((ql0 >> 4) & 0x0F) | (((qhHi >> 0) & 1) << 4)) - 16;
            int q1_hi = (((ql1 >> 4) & 0x0F) | (((qhHi >> 1) & 1) << 4)) - 16;
            int q2_hi = (((ql2 >> 4) & 0x0F) | (((qhHi >> 2) & 1) << 4)) - 16;
            int q3_hi = (((ql3 >> 4) & 0x0F) | (((qhHi >> 3) & 1) << 4)) - 16;
            int qPackHi = (q0_hi & 0xFF) | ((q1_hi & 0xFF) << 8)
                        | ((q2_hi & 0xFF) << 16) | ((q3_hi & 0xFF) << 24);

            // 4 Q8_1 input bytes for high half at offset 8 + 16 + j*4
            int inHi = *(const int*)(input + q8Off + 8 + 16 + j*4);
            dpAccum = __dp4a(qPackHi, inHi, dpAccum);
        }

        sum += scale * inScale * (float)dpAccum;
    }

    // Warp reduction
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    if (lane == 0) {
        if (addToOutput) output[row] += sum;
        else             output[row]  = sum;
    }
}
