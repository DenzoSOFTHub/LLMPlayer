/**
 * RoPE (Rotary Position Embedding) kernel.
 * Applies rotation to Q or K vectors in-place.
 * Supports NORMAL mode (consecutive pairs) and NEOX mode (split-half pairs).
 *
 * Each thread handles one (cos, sin) pair for one head.
 * Total threads needed: nHeads * halfRope.
 *
 * tokenParams[0] = position (read from GPU global memory for CUDA graph compatibility).
 */
extern "C" __global__ void rope_apply(
    float* vec,              // [nHeads * headSize] — Q or K vector
    const float* cosTable,   // [maxSeqLen * halfRope] — pre-computed cos values
    const float* sinTable,   // [maxSeqLen * halfRope] — pre-computed sin values
    const int nHeads,
    const int headSize,
    const int halfRope,
    const int* tokenParams,  // GPU buffer: tokenParams[0] = position
    const int ropeType)      // 0 = NORMAL (consecutive), 2 = NEOX (split-half)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nHeads * halfRope;
    if (idx >= total) return;

    int position = tokenParams[0];

    int h = idx / halfRope;
    int d = idx % halfRope;
    int tableIdx = position * halfRope + d;

    float cos_val = cosTable[tableIdx];
    float sin_val = sinTable[tableIdx];

    int base = h * headSize;
    float v0, v1;
    int i0, i1;

    if (ropeType == 0) {
        // NORMAL: consecutive pairs (vec[2d], vec[2d+1])
        i0 = base + 2 * d;
        i1 = base + 2 * d + 1;
    } else {
        // NEOX: split-half pairs (vec[d], vec[halfRope+d])
        i0 = base + d;
        i1 = base + halfRope + d;
    }

    v0 = vec[i0];
    v1 = vec[i1];
    vec[i0] = v0 * cos_val - v1 * sin_val;
    vec[i1] = v0 * sin_val + v1 * cos_val;
}
