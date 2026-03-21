// Dequantize Q4_K super-block to FP32.
// Used for pre-dequantizing weights at load time for cuBLAS gemv.
// Each thread handles one element. Grid covers all elements in the tensor.
// Output is row-major FP32: output[row * cols + col] = dequantized value.
//
// Q4_K layout per super-block (256 elements, 144 bytes):
//   [d:fp16][dmin:fp16][scales:12B][qs:128B]
//   4 groups of 64 elements, each with 2 sub-blocks of 32.
__device__ __forceinline__ float half2float_dq(unsigned short h) {
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

extern "C" __global__ void dequant_q4_k_f32(
    const unsigned char* __restrict__ weights,  // Q4_K data
    float* __restrict__ output,                 // FP32 output [rows * cols]
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = rows * cols;
    if (idx >= totalElements) return;

    int row = idx / cols;
    int col = idx % cols;

    // Find which super-block and position within it
    int numBlocks = cols / 256;
    int superBlock = col / 256;
    int posInBlock = col % 256;

    long bo = (long)row * numBlocks * 144 + (long)superBlock * 144;

    // Read header
    unsigned int dm = *(const unsigned int*)(weights + bo);
    float d = half2float_dq(dm & 0xFFFF);
    float dmin = half2float_dq(dm >> 16);

    // Read scales
    unsigned char sb[12];
    unsigned int sc0 = *(const unsigned int*)(weights + bo + 4);
    unsigned int sc1 = *(const unsigned int*)(weights + bo + 8);
    unsigned int sc2 = *(const unsigned int*)(weights + bo + 12);
    sb[0] = sc0 & 0xFF; sb[1] = (sc0>>8) & 0xFF; sb[2] = (sc0>>16) & 0xFF; sb[3] = (sc0>>24) & 0xFF;
    sb[4] = sc1 & 0xFF; sb[5] = (sc1>>8) & 0xFF; sb[6] = (sc1>>16) & 0xFF; sb[7] = (sc1>>24) & 0xFF;
    sb[8] = sc2 & 0xFF; sb[9] = (sc2>>8) & 0xFF; sb[10] = (sc2>>16) & 0xFF; sb[11] = (sc2>>24) & 0xFF;

    // Determine group and sub-block
    int group = posInBlock / 64;
    int posInGroup = posInBlock % 64;
    int subBlock = posInGroup / 32;  // 0 or 1
    int posInSub = posInGroup % 32;

    // Decode scale and min for this sub-block
    int sbIdx = group * 2 + subBlock;
    int scale, mn;
    if (sbIdx < 4) {
        scale = sb[sbIdx] & 0x3F;
        mn = sb[sbIdx + 4] & 0x3F;
    } else {
        scale = (sb[sbIdx + 4] & 0x0F) | ((sb[sbIdx - 4] >> 6) << 4);
        mn = ((sb[sbIdx + 4] >> 4) & 0x0F) | ((sb[sbIdx] >> 6) << 4);
    }

    // Read quantized nibble
    int qsByteIdx = group * 32 + posInSub;
    unsigned char qsByte = weights[bo + 16 + qsByteIdx];
    int q = (subBlock == 0) ? (qsByte & 0x0F) : ((qsByte >> 4) & 0x0F);

    // Dequantize
    output[idx] = d * (float)scale * (float)q - dmin * (float)mn;
}
