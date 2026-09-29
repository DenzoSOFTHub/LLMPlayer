// Q2_K matmul: 256 weights per 84-byte super-block — scales[16] (4-bit scale | 4-bit min per 16
// weights), qs[64] 2-bit quants, fp16 d, fp16 dmin. Element order (ggml dequantize_row_q2_K):
// weight 128n + 32k + l reads bits 2k of qs[32n + l], with scale byte 8n + 2k + l/16.
// One warp per row; each lane takes groups of 32 weights (one (n, k) pair).
__device__ __forceinline__ float q2k_h2f(unsigned short h) {
    float r;
    asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h));
    return r;
}

extern "C" __global__ void matmul_q2_k(
    const unsigned char* __restrict__ weights, const float* __restrict__ input, float* __restrict__ output,
    const int rows, const int cols, const int addToOutput)
{
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    int nsb = cols / 256;
    long rowBase = (long) row * nsb * 84;
    float sum = 0.0f;
    for (int g = lane; g < nsb * 8; g += 32) {
        int sb = g >> 3, n = (g >> 2) & 1, k = g & 3;
        long bo = rowBase + (long) sb * 84;
        const unsigned short* dd = (const unsigned short*) (weights + bo + 80);
        float d = q2k_h2f(__ldg(dd)), dmin = q2k_h2f(__ldg(dd + 1));
        const float* x = input + sb * 256 + n * 128 + k * 32;
        const unsigned int* q = (const unsigned int*) (weights + bo + 16 + 32 * n);
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            int sc = __ldg(weights + bo + 8 * n + 2 * k + half);
            float qx = 0.0f, sx = 0.0f;
            #pragma unroll
            for (int w4 = 0; w4 < 4; w4++) {
                unsigned int word = __ldg(q + half * 4 + w4);
                #pragma unroll
                for (int b = 0; b < 4; b++) {
                    float xv = x[half * 16 + w4 * 4 + b];
                    qx += (float) ((word >> (8 * b + 2 * k)) & 3) * xv;
                    sx += xv;
                }
            }
            sum += d * (float) (sc & 0xF) * qx - dmin * (float) (sc >> 4) * sx;
        }
    }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, off);
    if (lane == 0) output[row] = addToOutput ? output[row] + sum : sum;
}

// FP16 row-range dequantization for the batched prefill GEMM (see dequant_f16.cu for the contract).
extern "C" __global__ void dequant_q2_k_f16t(const unsigned char* __restrict__ w, unsigned short* __restrict__ out,
                                             int rowStart, int nRows, int cols) {
    long idx = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long) nRows * cols) return;
    int rr = (int) (idx / cols), col = (int) (idx - (long) rr * cols);
    long bo = ((long) rowStart + rr) * (cols / 256) * 84 + (long) (col / 256) * 84;
    int j = col & 255, n = j >> 7, k = (j >> 5) & 3, l = j & 31;
    float d = q2k_h2f((unsigned short) (w[bo + 80] | (w[bo + 81] << 8)));
    float dmin = q2k_h2f((unsigned short) (w[bo + 82] | (w[bo + 83] << 8)));
    int sc = w[bo + 8 * n + 2 * k + (l >> 4)];
    int q = (w[bo + 16 + 32 * n + l] >> (2 * k)) & 3;
    float v = d * (float) (sc & 0xF) * (float) q - dmin * (float) (sc >> 4);
    unsigned short o; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(o) : "f"(v));
    out[idx] = o;
}
