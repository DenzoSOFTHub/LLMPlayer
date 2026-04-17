/**
 * Q3_K × Q8_1 matrix-vector multiply using __dp4a intrinsic.
 *
 * Q3_K layout (110 bytes per 256-element super-block):
 *   bo+0..31   : hmask   — 32 bytes, 1 high-bit per element (bit hmBitPos in byte i)
 *   bo+32..95  : qs      — 64 bytes, 2 low-bits per element (bit-packed, shift = pair*2)
 *   bo+96..107 : scales  — 12 bytes, 16 6-bit scales packed (3*4 low + 3*2 hi)
 *   bo+108..109: d       — FP16 global scale
 *
 * Quantized value per element: q = (lowBits | (hbit << 2)) - 4  in [-4, 3]
 * Dequant formula: value = d * (scale - 32) * q  (NO dmin term — asymmetric around 0 already).
 *
 * 8 sub-blocks per super-block (2 halves × 4 pairs). Each sub-block = 32 elements = 1 Q8_1 block.
 * Each sub-block uses TWO scales: dl0 for first 16 elements, dl1 for second 16.
 *
 * Dot product reformulation per sub-block:
 *   dot_first16  = d * dl0 * input_scale * __dp4a(pack(q[0..15]),   int8(input[0..15]))
 *   dot_second16 = d * dl1 * input_scale * __dp4a(pack(q[16..31]),  int8(input[16..31]))
 *
 * Q3_K block size 110 is NOT 4-byte aligned; byte loads via __ldg only.
 * 1 warp (32 threads) per output row, lanes stripe across sub-blocks with stride 32.
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

/**
 * Decode one Q3_K 6-bit scale from the 12 raw scale bytes (byte loads via __ldg).
 * Scale index s in [0, 16). Reads only 2 bytes per scale.
 */
__device__ __forceinline__ int decodeQ3KScale(const unsigned char* __restrict__ scaleBytes, int s) {
    int lo = (s < 8) ? (__ldg(&scaleBytes[s & 7]) & 0x0F)
                     : (__ldg(&scaleBytes[s & 7]) >> 4);
    int hi = (__ldg(&scaleBytes[8 + (s & 3)]) >> ((s >> 2) * 2)) & 0x03;
    return lo | (hi << 4);
}

extern "C" __global__ void matmul_q3_k_dp4a(
    const unsigned char* __restrict__ weights,   // Q3_K weights
    const unsigned char* __restrict__ input,     // Q8_1 (40-byte blocks)
    float* __restrict__ output,
    const int rows, const int cols, const int addToOutput)
{
    int warpId = threadIdx.x / 32;
    int lane   = threadIdx.x & 31;
    int rowsPerBlock = blockDim.x / 32;
    int row = blockIdx.x * rowsPerBlock + warpId;
    if (row >= rows) return;

    int numBlocks = cols / 256;
    int numUnits  = numBlocks * 8;            // 8 sub-blocks per super-block
    long rowStride = (long)numBlocks * 110;
    float sum = 0.0f;

    for (int u = lane; u < numUnits; u += 32) {
        int b       = u >> 3;                 // super-block
        int subIdx  = u & 7;                  // sub-block (0..7)
        int hf      = subIdx >> 2;            // half (0 or 1)
        int pair    = subIdx & 3;             // pair (0..3)

        long bo = (long)row * rowStride + (long)b * 110;

        float d = half2float((unsigned short)(
              (unsigned int)__ldg(weights + bo + 108)
            | ((unsigned int)__ldg(weights + bo + 109) << 8)));

        int scaleIdx = hf * 8 + pair * 2;
        float dl0 = d * (float)(decodeQ3KScale(weights + bo + 96, scaleIdx)     - 32);
        float dl1 = d * (float)(decodeQ3KScale(weights + bo + 96, scaleIdx + 1) - 32);

        int shift     = pair * 2;
        int qBase     = hf * 32;
        int hmBitPos  = hf * 4 + pair;

        // Each sub-block = 32 elements = 1 Q8_1 block.
        int q8Block = b * 8 + hf * 4 + pair;
        long q8off = (long)q8Block * 40;
        float inScale = *(const float*)(input + q8off);

        int dp0 = 0, dp1 = 0;

        // First 16 elements: qs offsets [qBase, qBase+16), hm offsets [0, 16).
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int l = k * 4;
            unsigned char qs0 = __ldg(&weights[bo + 32 + qBase + l]);
            unsigned char qs1 = __ldg(&weights[bo + 32 + qBase + l + 1]);
            unsigned char qs2 = __ldg(&weights[bo + 32 + qBase + l + 2]);
            unsigned char qs3 = __ldg(&weights[bo + 32 + qBase + l + 3]);
            unsigned char hm0 = __ldg(&weights[bo + l]);
            unsigned char hm1 = __ldg(&weights[bo + l + 1]);
            unsigned char hm2 = __ldg(&weights[bo + l + 2]);
            unsigned char hm3 = __ldg(&weights[bo + l + 3]);

            int q0 = (int)(((qs0 >> shift) & 3) | (((hm0 >> hmBitPos) & 1) << 2)) - 4;
            int q1 = (int)(((qs1 >> shift) & 3) | (((hm1 >> hmBitPos) & 1) << 2)) - 4;
            int q2 = (int)(((qs2 >> shift) & 3) | (((hm2 >> hmBitPos) & 1) << 2)) - 4;
            int q3 = (int)(((qs3 >> shift) & 3) | (((hm3 >> hmBitPos) & 1) << 2)) - 4;

            int qPack = (q0 & 0xFF) | ((q1 & 0xFF) << 8)
                      | ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
            int inPack = *(const int*)(input + q8off + 8 + l);
            dp0 = __dp4a(qPack, inPack, dp0);
        }

        // Second 16 elements: qs offsets [qBase+16, qBase+32), hm offsets [16, 32).
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int l = k * 4;
            unsigned char qs0 = __ldg(&weights[bo + 32 + qBase + 16 + l]);
            unsigned char qs1 = __ldg(&weights[bo + 32 + qBase + 16 + l + 1]);
            unsigned char qs2 = __ldg(&weights[bo + 32 + qBase + 16 + l + 2]);
            unsigned char qs3 = __ldg(&weights[bo + 32 + qBase + 16 + l + 3]);
            unsigned char hm0 = __ldg(&weights[bo + 16 + l]);
            unsigned char hm1 = __ldg(&weights[bo + 16 + l + 1]);
            unsigned char hm2 = __ldg(&weights[bo + 16 + l + 2]);
            unsigned char hm3 = __ldg(&weights[bo + 16 + l + 3]);

            int q0 = (int)(((qs0 >> shift) & 3) | (((hm0 >> hmBitPos) & 1) << 2)) - 4;
            int q1 = (int)(((qs1 >> shift) & 3) | (((hm1 >> hmBitPos) & 1) << 2)) - 4;
            int q2 = (int)(((qs2 >> shift) & 3) | (((hm2 >> hmBitPos) & 1) << 2)) - 4;
            int q3 = (int)(((qs3 >> shift) & 3) | (((hm3 >> hmBitPos) & 1) << 2)) - 4;

            int qPack = (q0 & 0xFF) | ((q1 & 0xFF) << 8)
                      | ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
            int inPack = *(const int*)(input + q8off + 8 + 16 + l);
            dp1 = __dp4a(qPack, inPack, dp1);
        }

        sum += dl0 * inScale * (float)dp0 + dl1 * inScale * (float)dp1;
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (lane == 0) {
        if (addToOutput) output[row] += sum;
        else             output[row]  = sum;
    }
}
