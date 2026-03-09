/**
 * Split a packed gate+up buffer into separate gate and up arrays.
 * Used for Phi-3/4 models where wUp produces a concatenated [gate|up] output
 * of size 2*ffnDim, which must be split before SiLU+multiply.
 *
 * packed[0..ffnDim-1] → gate[0..ffnDim-1]
 * packed[ffnDim..2*ffnDim-1] → up[0..ffnDim-1]
 */
extern "C" __global__ void split_gate_up(
    const float* __restrict__ packed,
    float* __restrict__ gate,
    float* __restrict__ up,
    const int ffnDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ffnDim) return;
    gate[idx] = packed[idx];
    up[idx] = packed[ffnDim + idx];
}
