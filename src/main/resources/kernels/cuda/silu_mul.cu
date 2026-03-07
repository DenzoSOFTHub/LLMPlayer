/**
 * Fused SiLU + element-wise multiply: a[i] = (a[i] / (1 + exp(-a[i]))) * b[i]
 * Saves one kernel launch + one global memory round-trip vs separate silu + elementwise_mul.
 * Uses __expf for fast approximation (~2 ULP).
 */
extern "C" __global__ void silu_mul(
    float* a,
    const float* b,
    const int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float v = a[i];
    a[i] = (v / (1.0f + __expf(-v))) * b[i];
}
