/**
 * Accumulate: y[i] += x[i]
 */
extern "C" __global__ void accumulate(
    float* y,
    const float* x,
    const int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    y[i] += x[i];
}
