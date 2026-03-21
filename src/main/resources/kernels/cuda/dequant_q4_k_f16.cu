// Dequantize Q4_K super-block to FP16 (__half).
// One thread per element. Output is row-major FP16.
// Used for pre-dequantizing weights at load time for cuBLAS mixed-precision gemv.
__device__ __forceinline__ unsigned short float2half_rn(float f) {
    unsigned int x = *(unsigned int*)&f;
    unsigned int sign = (x >> 16) & 0x8000;
    int exp = ((x >> 23) & 0xFF) - 127;
    unsigned int mantissa = x & 0x7FFFFF;
    if (exp > 15) return sign | 0x7C00;
    if (exp < -14) return sign;
    return sign | ((exp + 15) << 10) | (mantissa >> 13);
}

__device__ __forceinline__ float half2float_dq16(unsigned short h) {
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

extern "C" __global__ void dequant_q4_k_f16(
    const unsigned char* __restrict__ weights,
    unsigned short* __restrict__ output,    // FP16 output [rows * cols]
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = rows * cols;
    if (idx >= totalElements) return;

    int row = idx / cols;
    int col = idx % cols;
    int numBlocks = cols / 256;
    int superBlock = col / 256;
    int posInBlock = col % 256;
    long bo = (long)row * numBlocks * 144 + (long)superBlock * 144;

    unsigned int dm = *(const unsigned int*)(weights + bo);
    float d = half2float_dq16(dm & 0xFFFF);
    float dmin = half2float_dq16(dm >> 16);

    unsigned char sb[12];
    unsigned int sc0 = *(const unsigned int*)(weights + bo + 4);
    unsigned int sc1 = *(const unsigned int*)(weights + bo + 8);
    unsigned int sc2 = *(const unsigned int*)(weights + bo + 12);
    sb[0] = sc0 & 0xFF; sb[1] = (sc0>>8) & 0xFF; sb[2] = (sc0>>16) & 0xFF; sb[3] = (sc0>>24) & 0xFF;
    sb[4] = sc1 & 0xFF; sb[5] = (sc1>>8) & 0xFF; sb[6] = (sc1>>16) & 0xFF; sb[7] = (sc1>>24) & 0xFF;
    sb[8] = sc2 & 0xFF; sb[9] = (sc2>>8) & 0xFF; sb[10] = (sc2>>16) & 0xFF; sb[11] = (sc2>>24) & 0xFF;

    int group = posInBlock / 64;
    int posInGroup = posInBlock % 64;
    int subBlock = posInGroup / 32;
    int posInSub = posInGroup % 32;

    int sbIdx = group * 2 + subBlock;
    int scale, mn;
    if (sbIdx < 4) { scale = sb[sbIdx] & 0x3F; mn = sb[sbIdx + 4] & 0x3F; }
    else { scale = (sb[sbIdx + 4] & 0x0F) | ((sb[sbIdx - 4] >> 6) << 4); mn = ((sb[sbIdx + 4] >> 4) & 0x0F) | ((sb[sbIdx] >> 6) << 4); }

    int qsByteIdx = group * 32 + posInSub;
    unsigned char qsByte = weights[bo + 16 + qsByteIdx];
    int q = (subBlock == 0) ? (qsByte & 0x0F) : ((qsByte >> 4) & 0x0F);

    float val = d * (float)scale * (float)q - dmin * (float)mn;
    output[idx] = float2half_rn(val);
}
