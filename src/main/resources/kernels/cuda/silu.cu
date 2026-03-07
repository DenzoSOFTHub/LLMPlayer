/**
 * SiLU (Swish) activation: x[i] = x[i] / (1 + exp(-x[i]))
 * Uses __expf for fast approximation (~2 ULP).
 */
extern "C" __global__ void silu(
    float* x,
    const int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float v = x[i];
    x[i] = v / (1.0f + __expf(-v));
}
