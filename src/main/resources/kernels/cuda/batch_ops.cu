// Multi-token variants of the per-token kernels, for the batched GPU prefill (CudaForwardPass).
// Activations are row-major [n][size]; token t's position is tokenParams[2t] (seqLen at 2t+1).
// Same math as rmsnorm_fused / rope_apply / kv_cache_update(_f16) per row.

// One block per row: out[t] = x[t] * rsqrt(mean(x[t]^2) + eps) * w. In place allowed.
extern "C" __global__ void rmsnorm_batch(float* out, const float* x, const float* w,
                                         const int size, const float eps)
{
    extern __shared__ float smem[];
    const float* xr = x + (long) blockIdx.x * size;
    float* orow = out + (long) blockIdx.x * size;
    int lane = threadIdx.x & 31, warpId = threadIdx.x / 32, numWarps = blockDim.x / 32;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) { float v = xr[i]; ss += v * v; }
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xFFFFFFFF, ss, off);
    if (lane == 0) smem[warpId] = ss;
    __syncthreads();
    if (warpId == 0) {
        ss = (lane < numWarps) ? smem[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xFFFFFFFF, ss, off);
        if (lane == 0) smem[numWarps] = rsqrtf(ss / (float) size + eps);
    }
    __syncthreads();
    float scale = smem[numWarps];
    for (int i = threadIdx.x; i < size; i += blockDim.x) orow[i] = xr[i] * scale * w[i];
}

// RoPE for n tokens: blockIdx.y = token; vec row stride = vecStride floats.
extern "C" __global__ void rope_apply_batch(float* vec, const float* cosTable, const float* sinTable,
                                            const int nHeads, const int headSize, const int halfRope,
                                            const int* tokenParams, const int ropeType, const int vecStride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nHeads * halfRope) return;
    int t = blockIdx.y;
    int position = tokenParams[2 * t];
    float* v = vec + (long) t * vecStride;
    int h = idx / halfRope, d = idx % halfRope;
    float c = cosTable[position * halfRope + d];
    float s = sinTable[position * halfRope + d];
    int base = h * headSize;
    int i0 = (ropeType == 0) ? base + 2 * d : base + d;
    int i1 = (ropeType == 0) ? base + 2 * d + 1 : base + halfRope + d;
    float v0 = v[i0], v1 = v[i1];
    v[i0] = v0 * c - v1 * s;
    v[i1] = v0 * s + v1 * c;
}

// KV cache write for n tokens: blockIdx.y = token.
extern "C" __global__ void kv_cache_update_batch(float* keyCache, float* valueCache,
                                                 const float* k, const float* v, const int kvDim,
                                                 const int* tokenParams)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kvDim) return;
    int t = blockIdx.y;
    long dst = (long) tokenParams[2 * t] * kvDim + i;
    keyCache[dst] = k[(long) t * kvDim + i];
    valueCache[dst] = v[(long) t * kvDim + i];
}

__device__ __forceinline__ unsigned short bo_f2h(float f) {
    unsigned short r;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r) : "f"(f));
    return r;
}

extern "C" __global__ void kv_cache_update_batch_f16(unsigned short* keyCache, unsigned short* valueCache,
                                                     const float* k, const float* v, const int kvDim,
                                                     const int* tokenParams)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kvDim) return;
    int t = blockIdx.y;
    long dst = (long) tokenParams[2 * t] * kvDim + i;
    keyCache[dst] = bo_f2h(k[(long) t * kvDim + i]);
    valueCache[dst] = bo_f2h(v[(long) t * kvDim + i]);
}

// y[i] += bias[i % size] over total = n * size elements.
extern "C" __global__ void add_bias_batch(float* y, const float* bias, const int size, const int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    y[i] += bias[i % size];
}

// y[i] += a * x[i]
extern "C" __global__ void axpy(float* y, const float* x, const float a, const int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    y[i] += a * x[i];
}

// FP32 -> FP16 (round to nearest), saturated to the FP16 range so that an outlier activation
// becomes 65504 rather than infinity in the GEMM input.
extern "C" __global__ void f32_to_f16_sat(const float* __restrict__ in, unsigned short* __restrict__ out,
                                          const int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    out[i] = bo_f2h(fminf(65504.0f, fmaxf(-65504.0f, in[i])));
}

// GeGLU: a[i] = gelu(a[i]) * b[i] (tanh approximation, as SwiGLUFFN.gelu on the CPU) — Gemma 2/3/4,
// Spark2.5. Works on one token or n contiguous tokens (size = n * ffnDim).
extern "C" __global__ void gelu_mul(float* a, const float* b, const int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float v = a[i];
    a[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v))) * b[i];
}

// Spark2.5 head-wise output gate: out[i] *= sigmoid(gate[i / headSize]) over total = n * headCount *
// headSize elements (gate is [n][headCount], contiguous like out).
extern "C" __global__ void head_sigmoid_gate(float* out, const float* gate, const int headSize, const int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    out[i] *= 1.0f / (1.0f + __expf(-gate[i / headSize]));
}

// MLA (DeepSeek2): build per-head K = [K_nope | k_rope] and V padded to keyLength (zeros past
// valueLength) from the combined wkv_b output kvd[h][keyNope + valueLength] and the shared, already
// rotated k_rope[ropeDim]. Padding V lets the flash kernel use one head size for K and V.
extern "C" __global__ void mla_split_kv(const float* kvd, const float* krope, float* k, float* v,
                                        const int headCount, const int keyNope, const int ropeDim,
                                        const int valueLength)
{
    int keyLength = keyNope + ropeDim;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= headCount * keyLength) return;
    int h = i / keyLength, j = i % keyLength;
    const float* src = kvd + (long) h * (keyNope + valueLength);
    k[i] = (j < keyNope) ? src[j] : krope[j - keyNope];
    v[i] = (j < valueLength) ? src[keyNope + j] : 0.0f;
}

// Drop the padding: out[h][0..valueLength) = in[h][0..valueLength) with in's head stride keyLength.
extern "C" __global__ void mla_compact(const float* in, float* out, const int headCount,
                                       const int keyLength, const int valueLength)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= headCount * valueLength) return;
    int h = i / valueLength, j = i % valueLength;
    out[i] = in[(long) h * keyLength + j];
}

// MLA with separate K_b / V_b (GLM-4.7-Flash, DeepSeek-V3): K = [K_nope | k_rope] per head from
// knope[h][keyNope] and the shared rotated k_rope; V[h] = vsrc[h][valueLength] padded to keyLength.
extern "C" __global__ void mla_assemble_kv(const float* knope, const float* vsrc, const float* krope,
                                           float* k, float* v, const int headCount, const int keyNope,
                                           const int ropeDim, const int valueLength)
{
    int keyLength = keyNope + ropeDim;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= headCount * keyLength) return;
    int h = i / keyLength, j = i % keyLength;
    k[i] = (j < keyNope) ? knope[h * keyNope + j] : krope[j - keyNope];
    v[i] = (j < valueLength) ? vsrc[h * valueLength + j] : 0.0f;
}

// Latent MLA (MlaAttentionCudaPass, GGUFs with separate attn_k_b / attn_v_b): the rope part of
// each head's query goes after its latent query, dst[h][kvLoraRank + j] = q[h][keyNope + j].
extern "C" __global__ void mla_q_rope_copy(const float* q, float* dst, const int headCount,
                                           const int keyLength, const int keyNope, const int entry,
                                           const int kvLoraRank, const int ropeDim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= headCount * ropeDim) return;
    int h = i / ropeDim, j = i % ropeDim;
    dst[h * entry + kvLoraRank + j] = q[h * keyLength + keyNope + j];
}

__device__ __forceinline__ float bo_h2f(unsigned short h) {
    float r;
    asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h));
    return r;
}

// Per-head matrix-vector product with FP16 weights W[heads][rows][cols] (blockIdx.y = head):
// y[h * yStride + r] = W[h][r] . x[h * xStride ...]. One warp per row, 8 rows per block.
extern "C" __global__ void matmul_f16_heads(const unsigned short* __restrict__ w, const float* __restrict__ x,
                                            float* __restrict__ y, const int rows, const int cols,
                                            const int xStride, const int yStride)
{
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;
    int h = blockIdx.y;
    const unsigned short* wr = w + ((long) h * rows + row) * cols;
    const float* xh = x + (long) h * xStride;
    float s = 0.0f;
    if ((cols & 1) == 0) {
        const unsigned int* w2 = (const unsigned int*) wr;
        for (int i = lane; i < (cols >> 1); i += 32) {
            unsigned int p = __ldg(w2 + i);
            s += bo_h2f((unsigned short) (p & 0xFFFF)) * xh[2 * i] + bo_h2f((unsigned short) (p >> 16)) * xh[2 * i + 1];
        }
    } else {
        for (int i = lane; i < cols; i += 32) s += bo_h2f(__ldg(wr + i)) * xh[i];
    }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xFFFFFFFF, s, off);
    if (lane == 0) y[(long) h * yStride + row] = s;
}

// y[i] += g * x[i] with g = sigmoid(logit[0]), or 1 when logit is null: the Qwen3.5-MoE shared
// expert, scaled by its one-logit gate.
extern "C" __global__ void axpy_sigmoid_gate(float* y, const float* x, const float* logit, const int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = logit != 0 ? 1.0f / (1.0f + __expf(-logit[0])) : 1.0f;
    y[i] += g * x[i];
}
