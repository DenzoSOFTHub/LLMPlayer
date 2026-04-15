/**
 * Q4_K × Q8_1 matmul — port of llama.cpp's `mul_mat_vec_q` template specialized for Q4_K
 * with ncols_dst=1 (batch=1 inference).
 *
 * Source: https://github.com/ggml-org/llama.cpp ggml/src/ggml-cuda/mmvq.cu (MIT license).
 * Adapted to our 40-byte Q8_1 buffer format (FP32 scale + FP32 sum + 32 byte qs) — we read
 * inScale only (inSum is unused; we use llama.cpp's "sum-via-dp4a-with-0x01010101 constant"
 * trick that computes the sum at dot-product time for free).
 *
 * Key algorithmic differences vs our `matmul_q4_k_dp4a.cu`:
 *  - 4 warps × 32 lanes = 128 threads per block (vs ours: 8 warps × 1 row = 256 threads)
 *  - 1 row per block (vs ours: 8 rows per block)  → 4× more parallelism per row
 *  - vdr=2: each thread processes 2 adjacent Q8_1 sub-blocks per call (16 weights)
 *  - dp4a-with-constant trick computes inSum on-the-fly inside the dp4a hardware
 *
 * Constants (Q4_K specific):
 *   QK_K = 256, QR4_K = 2, QI4_K = QK_K/(4*QR4_K) = 32, QI8_1 = 8
 *   nwarps = 4, vdr = 2
 *   blocks_per_iter = vdr * nwarps * warp_size / QI4_K = 2*4*32/32 = 8
 */
#define NWARPS 4
#define WARP_SIZE 32
#define QR4_K 2
#define QI4_K 32
#define QI8_1 8
#define QK_K 256

// Per-thread vec_dot for Q4_K × Q8_1 — reads 8 bytes of weights (16 nibbles) + 16 bytes of input
// (2 Q8_1 sub-blocks worth) and accumulates into one float.
//
// Inlined and adapted from llama.cpp's vec_dot_q4_K_q8_1 + vec_dot_q4_K_q8_1_impl_vmmq.
// Our Q8_1 buffer has 40-byte stride (vs theirs 36); we read inScale at offset 0 and skip
// the inSum at offset 4 (using the on-the-fly dp4a-sum trick instead).
__device__ __forceinline__ float vec_dot_q4_K_q8_1_ours(
    const unsigned char* __restrict__ weights,  // start of Q4_K row (already offset by row*numSB*144)
    const unsigned char* __restrict__ input,    // start of Q8_1 input vector (40-byte blocks)
    int kbx,                                    // super-block index
    int iqs)                                    // position within super-block (0,2,4,...,30)
{
    // === Q4_K weight reads ===
    const long bo = (long)kbx * 144;
    const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));   // 0,0,0,0, 2,2,2,2, 4,4,4,4, 6,6,6,6
    const int qs_offset = 16 * bq8_offset + 4 * ((iqs / 2) % 4); // bytes 0..127 of qs (= bo+16..bo+143)

    // v[0] = qs[qs_offset .. +3], v[1] = qs[qs_offset+16 .. +19]  (4-byte uint loads)
    int v0, v1;
    v0 = __ldg((const int*)(weights + bo + 16 + qs_offset));
    v1 = __ldg((const int*)(weights + bo + 16 + qs_offset + 16));

    // === Scale/min decode ===
    // Q4_K scales layout: 12 bytes at bo+4. For sub-block i (0..7): scale,min are 6-bit packed
    // following the standard Q4_K convention. iqs = 0,2,...,30 → bq8_offset + 0,1 picks 2 sub-blocks.
    // We need sc[0], sc[1], m[0], m[1] for sub-blocks (bq8_offset/2)*2 and (bq8_offset/2)*2+1.
    // Actually for bq8_offset = 2*j (j=0..3), the relevant sub-blocks are j*2 and j*2+1.
    // — wait re-check: bq8_offset is the Q8_1 block offset (each = 32 elements).
    // For Q4_K: 8 sub-blocks of 32 elements each. So sub_idx for bq8_offset+i is just bq8_offset+i.
    const int sc_idx0 = bq8_offset;       // sub-block index for v[0]'s scale
    const int sc_idx1 = bq8_offset + 1;   // sub-block index for v[1]'s scale

    unsigned int sc0u = __ldg((const unsigned int*)(weights + bo + 4));
    unsigned int sc1u = __ldg((const unsigned int*)(weights + bo + 8));
    unsigned int sc2u = __ldg((const unsigned int*)(weights + bo + 12));

    int sc[2], m[2];
    // Inline decoder per sub-index. Avoids C++ lambdas (NVRTC quirks).
    {
        int sub = sc_idx0;
        if (sub < 4) {
            unsigned int sh = (sub & 3) * 8;
            sc[0] = (int)((sc0u >> sh) & 0x3F);
            m[0]  = (int)((sc1u >> sh) & 0x3F);
        } else {
            unsigned int sh = ((sub - 4) & 3) * 8;
            unsigned int s4   = (sc2u >> sh) & 0xFF;
            unsigned int sLow = (sc0u >> sh) & 0xFF;
            unsigned int sMin = (sc1u >> sh) & 0xFF;
            sc[0] = (int)((s4 & 0x0F) | ((sLow >> 6) << 4));
            m[0]  = (int)(((s4 >> 4) & 0x0F) | ((sMin >> 6) << 4));
        }
    }
    {
        int sub = sc_idx1;
        if (sub < 4) {
            unsigned int sh = (sub & 3) * 8;
            sc[1] = (int)((sc0u >> sh) & 0x3F);
            m[1]  = (int)((sc1u >> sh) & 0x3F);
        } else {
            unsigned int sh = ((sub - 4) & 3) * 8;
            unsigned int s4   = (sc2u >> sh) & 0xFF;
            unsigned int sLow = (sc0u >> sh) & 0xFF;
            unsigned int sMin = (sc1u >> sh) & 0xFF;
            sc[1] = (int)((s4 & 0x0F) | ((sLow >> 6) << 4));
            m[1]  = (int)(((s4 >> 4) & 0x0F) | ((sMin >> 6) << 4));
        }
    }

    // === Q8_1 input reads (our 40-byte format) ===
    // BUG-FIX 2026-04-14: the Q8_1 block index must include kbx*8 (super-block stride),
    // not just bq8_offset. Each super-block contains 8 Q8_1 sub-blocks (256 elems / 32).
    // llama.cpp does this via `kby = kbx * (qk/QK8_1)` and passes `&y[kby]` to vec_dot.
    const int kby = kbx * (QK_K / 32);   // = kbx * 8: Q8_1 block index of super-block start
    float d8[2];
    int u[2 * QR4_K];
    const int u_off = ((iqs / 2) % 4) * 4;  // byte offset within Q8_1 qs

    #pragma unroll
    for (int i = 0; i < QR4_K; i++) {
        const long blockOff = (long)(kby + bq8_offset + i) * 40;  // 40-byte stride per Q8_1 block
        d8[i] = *(const float*)(input + blockOff);                // inScale (we ignore inSum at +4)
        u[2*i + 0] = *(const int*)(input + blockOff + 8 + u_off);
        u[2*i + 1] = *(const int*)(input + blockOff + 8 + u_off + 16);
    }

    // === Q4_K dm (super-block scale/min) ===
    unsigned int dm = __ldg((const unsigned int*)(weights + bo));
    // d (low 16 bits) and dmin (high 16 bits) as fp16 packed — manual conversion
    float dval, dminval;
    {
        unsigned int h = dm & 0xFFFF;
        unsigned int sign = (h >> 15) & 1;
        unsigned int exp = (h >> 10) & 0x1F;
        unsigned int mantissa = h & 0x3FF;
        if (exp == 0) {
            if (mantissa == 0) { dval = sign ? -0.0f : 0.0f; }
            else {
                while (!(mantissa & 0x400)) { mantissa <<= 1; exp--; }
                exp++; mantissa &= 0x3FF;
                unsigned int f = (sign << 31) | ((exp + 112) << 23) | (mantissa << 13);
                dval = *(float*)&f;
            }
        } else if (exp == 31) {
            unsigned int f = (sign << 31) | 0x7F800000 | (mantissa << 13);
            dval = *(float*)&f;
        } else {
            unsigned int f = (sign << 31) | ((exp + 112) << 23) | (mantissa << 13);
            dval = *(float*)&f;
        }
    }
    {
        unsigned int h = dm >> 16;
        unsigned int sign = (h >> 15) & 1;
        unsigned int exp = (h >> 10) & 0x1F;
        unsigned int mantissa = h & 0x3FF;
        if (exp == 0) {
            if (mantissa == 0) { dminval = sign ? -0.0f : 0.0f; }
            else {
                while (!(mantissa & 0x400)) { mantissa <<= 1; exp--; }
                exp++; mantissa &= 0x3FF;
                unsigned int f = (sign << 31) | ((exp + 112) << 23) | (mantissa << 13);
                dminval = *(float*)&f;
            }
        } else if (exp == 31) {
            unsigned int f = (sign << 31) | 0x7F800000 | (mantissa << 13);
            dminval = *(float*)&f;
        } else {
            unsigned int f = (sign << 31) | ((exp + 112) << 23) | (mantissa << 13);
            dminval = *(float*)&f;
        }
    }

    // === vec_dot_q4_K_q8_1_impl_vmmq core loop ===
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
    #pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        // Extract low/high nibbles per i iteration
        const int v0i = (v0 >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v1 >> (4*i)) & 0x0F0F0F0F;

        // dot1 = sum(weights[low/high nibbles] * u_quants) — actual matmul
        const int dot1 = __dp4a(v1i, u[2*i+1], __dp4a(v0i, u[2*i+0], 0));
        // dot2 = sum(u_quants) computed for free via dp4a-with-constant trick
        const int dot2 = __dp4a(0x01010101, u[2*i+1], __dp4a(0x01010101, u[2*i+0], 0));

        sumf_d += d8[i] * (float)(dot1 * sc[i]);
        sumf_m += d8[i] * (float)(dot2 * m[i]);
    }

    return dval * sumf_d - dminval * sumf_m;
}

// Multi-warp orchestrator (llama.cpp's mmvq pattern):
//   blockDim = 128 (4 warps × 32 lanes), gridDim = rows, 1 row per block.
//   Each thread handles 1 vec_dot call distributed via tid -> (kbx, kqs) per llama.cpp's
//   `kbx = tid / (qi/vdr)`, `kqs = vdr * (tid % (qi/vdr))` for Q4_K (qi=32, vdr=2).
//   Cross-warp reduction in shared memory.
extern "C" __global__ void matmul_q4_k_mmvq(
    const unsigned char* __restrict__ weights,
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    const int rows,
    const int cols,
    const int addToOutput)
{
    const int row = blockIdx.x;
    if (row >= rows) return;

    const int warpId = threadIdx.x / WARP_SIZE;     // 0..NWARPS-1
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int tid = threadIdx.x;                    // 0..NWARPS*WARP_SIZE-1

    const int blocks_per_row_x = cols / QK_K;
    constexpr int blocks_per_iter = 2 * NWARPS * WARP_SIZE / QI4_K; // = 8 for NWARPS=4
    constexpr int qi_div_vdr = QI4_K / 2;                           // = 16

    const long row_offset = (long)row * blocks_per_row_x * 144;
    const unsigned char* weights_row = weights + row_offset;

    float tmp = 0.0f;

    for (int kbx = tid / qi_div_vdr; kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kqs = 2 * (tid % qi_div_vdr);   // 0, 2, ..., 30
        tmp += vec_dot_q4_K_q8_1_ours(weights_row, input, kbx, kqs);
    }

    // Within-warp reduction
    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        tmp += __shfl_down_sync(0xFFFFFFFF, tmp, off);
    }

    // Cross-warp reduction via shared memory
    __shared__ float warp_sums[NWARPS];
    if (lane == 0) warp_sums[warpId] = tmp;
    __syncthreads();

    if (warpId == 0) {
        float acc = (lane < NWARPS) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int off = NWARPS / 2; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFF, acc, off);
        }
        if (lane == 0) {
            if (addToOutput) output[row] += acc;
            else             output[row]  = acc;
        }
    }
}
