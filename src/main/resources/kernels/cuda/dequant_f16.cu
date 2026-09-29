// Dequantize a row range of a quantized weight matrix to FP16, row-major, for the batched
// prefill GEMM (cuBLAS, FP16 inputs, FP32 accumulate). One thread per output element.
//
// Signature (all kernels): weights = base of the whole [rows][cols] tensor, out = FP16 tile,
// rowStart = first row of the tile, nRows = tile rows, cols = row length. out[(r - rowStart) * cols + c].
//
// Block layouts follow ggml, matching the validated matvec kernels in this directory.

__device__ __forceinline__ float dq_h2f(unsigned short h) {
    float r;
    asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h));
    return r;
}
__device__ __forceinline__ unsigned short dq_f2h(float f) {
    unsigned short r;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r) : "f"(f));
    return r;
}
__device__ __forceinline__ float dq_bf2f(unsigned short b) {
    return __int_as_float(((unsigned int) b) << 16);
}
__device__ __forceinline__ unsigned short dq_ld16(const unsigned char* p) {
    return (unsigned short) p[0] | ((unsigned short) p[1] << 8);
}

#define DQ_PROLOGUE \
    long idx = (long) blockIdx.x * blockDim.x + threadIdx.x; \
    long total = (long) nRows * cols; \
    if (idx >= total) return; \
    int rr = (int) (idx / cols); \
    int col = (int) (idx - (long) rr * cols); \
    long row = (long) rowStart + rr;

// ---- Q4_K: 256 elements, 144 bytes: d, dmin (fp16), scales[12], qs[128] ----
extern "C" __global__ void dequant_q4_k_f16t(const unsigned char* __restrict__ w, unsigned short* __restrict__ out,
                                             int rowStart, int nRows, int cols) {
    DQ_PROLOGUE
    const unsigned char* b = w + row * (cols / 256) * 144 + (col / 256) * 144;
    int j = col & 255;
    float d = dq_h2f(dq_ld16(b)), dmin = dq_h2f(dq_ld16(b + 2));
    const unsigned char* sc = b + 4;
    int group = j / 64, sub = (j % 64) / 32, l = j % 32;
    int is = group * 2 + sub;
    int s, m;
    if (is < 4) { s = sc[is] & 63; m = sc[is + 4] & 63; }
    else { s = (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4); m = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4); }
    unsigned char q = b[16 + group * 32 + l];
    int v = sub == 0 ? (q & 0xF) : (q >> 4);
    out[idx] = dq_f2h(d * s * v - dmin * m);
}

// ---- Q5_K: 256 elements, 176 bytes: d, dmin, scales[12], qh[32], qs[128] ----
extern "C" __global__ void dequant_q5_k_f16t(const unsigned char* __restrict__ w, unsigned short* __restrict__ out,
                                             int rowStart, int nRows, int cols) {
    DQ_PROLOGUE
    const unsigned char* b = w + row * (cols / 256) * 176 + (col / 256) * 176;
    int j = col & 255;
    float d = dq_h2f(dq_ld16(b)), dmin = dq_h2f(dq_ld16(b + 2));
    const unsigned char* sc = b + 4;
    int group = j / 64, high = (j % 64) >= 32, l = j % 32;
    int is = group * 2 + high;
    int s, m;
    if (is < 4) { s = sc[is] & 63; m = sc[is + 4] & 63; }
    else { s = (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4); m = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4); }
    unsigned char qs = b[48 + group * 32 + l];
    int ql = high ? (qs >> 4) : (qs & 0xF);
    int qh = (b[16 + l] >> (group * 2 + high)) & 1;
    out[idx] = dq_f2h(d * s * (ql | (qh << 4)) - dmin * m);
}

// ---- Q6_K: 256 elements, 210 bytes: ql[128], qh[64], scales[16] (int8), d ----
extern "C" __global__ void dequant_q6_k_f16t(const unsigned char* __restrict__ w, unsigned short* __restrict__ out,
                                             int rowStart, int nRows, int cols) {
    DQ_PROLOGUE
    const unsigned char* b = w + row * (cols / 256) * 210 + (col / 256) * 210;
    int j = col & 255;
    float d = dq_h2f(dq_ld16(b + 208));
    int half = j / 128, jl = j % 128, quad = jl / 32, l = jl % 32;
    const unsigned char* ql = b + half * 64;
    const unsigned char* qh = b + 128 + half * 32;
    int q4 = (quad == 0) ? (ql[l] & 0xF) : (quad == 1) ? (ql[32 + l] & 0xF) : (quad == 2) ? (ql[l] >> 4) : (ql[32 + l] >> 4);
    int q2 = (qh[l] >> (quad * 2)) & 3;
    int q = (q4 | (q2 << 4)) - 32;
    int s = (signed char) b[192 + j / 16];
    out[idx] = dq_f2h(d * s * q);
}

// ---- Q3_K: 256 elements, 110 bytes: hmask[32], qs[64], scales[12], d ----
extern "C" __global__ void dequant_q3_k_f16t(const unsigned char* __restrict__ w, unsigned short* __restrict__ out,
                                             int rowStart, int nRows, int cols) {
    DQ_PROLOGUE
    const unsigned char* b = w + row * (cols / 256) * 110 + (col / 256) * 110;
    int j = col & 255;
    float d = dq_h2f(dq_ld16(b + 108));
    int hf = j / 128, jj = j % 128, pair = jj / 32, l = jj % 32;
    int s = hf * 8 + pair * 2 + (l >= 16 ? 1 : 0);
    const unsigned char* scb = b + 96;
    int lo = (s < 8) ? (scb[s & 7] & 0xF) : (scb[s & 7] >> 4);
    int hi = (scb[8 + (s & 3)] >> ((s >> 2) * 2)) & 3;
    int scale = (lo | (hi << 4)) - 32;
    int low = (b[32 + hf * 32 + l] >> (pair * 2)) & 3;
    int hbit = (b[l] >> (hf * 4 + pair)) & 1;
    int q = (low | (hbit << 2)) - 4;
    out[idx] = dq_f2h(d * scale * q);
}

// ---- Q8_0: 32 elements, 34 bytes: d, qs[32] (int8) ----
extern "C" __global__ void dequant_q8_0_f16t(const unsigned char* __restrict__ w, unsigned short* __restrict__ out,
                                             int rowStart, int nRows, int cols) {
    DQ_PROLOGUE
    const unsigned char* b = w + row * (cols / 32) * 34 + (col / 32) * 34;
    out[idx] = dq_f2h(dq_h2f(dq_ld16(b)) * (float) (signed char) b[2 + (col & 31)]);
}

// ---- Q5_0: 32 elements, 22 bytes: d, qh (u32), qs[16]; split nibbles ----
extern "C" __global__ void dequant_q5_0_f16t(const unsigned char* __restrict__ w, unsigned short* __restrict__ out,
                                             int rowStart, int nRows, int cols) {
    DQ_PROLOGUE
    const unsigned char* b = w + row * (cols / 32) * 22 + (col / 32) * 22;
    int j = col & 31;
    unsigned int qh = (unsigned int) b[2] | ((unsigned int) b[3] << 8) | ((unsigned int) b[4] << 16) | ((unsigned int) b[5] << 24);
    int lo = j < 16 ? (b[6 + j] & 0xF) : (b[6 + j - 16] >> 4);
    int q = (lo | (((qh >> j) & 1) << 4)) - 16;
    out[idx] = dq_f2h(dq_h2f(dq_ld16(b)) * q);
}

__constant__ float DQ_KVALUES_IQ4NL[16] = {-127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
                                             1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f};

// ---- IQ4_NL: 32 elements, 18 bytes: d, qs[16]; split nibbles, non-linear codebook ----
extern "C" __global__ void dequant_iq4_nl_f16t(const unsigned char* __restrict__ w, unsigned short* __restrict__ out,
                                               int rowStart, int nRows, int cols) {
    DQ_PROLOGUE
    const unsigned char* b = w + row * (cols / 32) * 18 + (col / 32) * 18;
    int j = col & 31;
    int nib = j < 16 ? (b[2 + j] & 0xF) : (b[2 + j - 16] >> 4);
    out[idx] = dq_f2h(dq_h2f(dq_ld16(b)) * DQ_KVALUES_IQ4NL[nib]);
}

// ---- IQ4_XS: 256 elements, 136 bytes: d, scales_h (u16), scales_l[4], qs[128]; split per 32 ----
extern "C" __global__ void dequant_iq4_xs_f16t(const unsigned char* __restrict__ w, unsigned short* __restrict__ out,
                                               int rowStart, int nRows, int cols) {
    DQ_PROLOGUE
    const unsigned char* b = w + row * (cols / 256) * 136 + (col / 256) * 136;
    int j = col & 255;
    float d = dq_h2f(dq_ld16(b));
    int ib = j / 32, in = j % 32;
    unsigned int sh = dq_ld16(b + 2);
    unsigned char sl = b[4 + ib / 2];
    int ls = ((ib & 1) ? (sl >> 4) : (sl & 0xF)) | (((sh >> (2 * ib)) & 3) << 4);
    unsigned char q = b[8 + ib * 16 + (in & 15)];
    int nib = in < 16 ? (q & 0xF) : (q >> 4);
    out[idx] = dq_f2h(d * (float) (ls - 32) * DQ_KVALUES_IQ4NL[nib]);
}

// ---- F16 / BF16 / F32 ----
extern "C" __global__ void dequant_f16_f16t(const unsigned char* __restrict__ w, unsigned short* __restrict__ out,
                                            int rowStart, int nRows, int cols) {
    DQ_PROLOGUE
    out[idx] = ((const unsigned short*) w)[row * cols + col];
}
extern "C" __global__ void dequant_bf16_f16t(const unsigned char* __restrict__ w, unsigned short* __restrict__ out,
                                             int rowStart, int nRows, int cols) {
    DQ_PROLOGUE
    out[idx] = dq_f2h(dq_bf2f(((const unsigned short*) w)[row * cols + col]));
}
extern "C" __global__ void dequant_f32_f16t(const unsigned char* __restrict__ w, unsigned short* __restrict__ out,
                                            int rowStart, int nRows, int cols) {
    DQ_PROLOGUE
    out[idx] = dq_f2h(((const float*) w)[row * cols + col]);
}
