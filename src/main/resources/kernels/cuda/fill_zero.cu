/**
 * Zero-fill: x[i] = 0.0f
 */
extern "C" __global__ void fill_zero(
    float* x,
    const int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    x[i] = 0.0f;
}
