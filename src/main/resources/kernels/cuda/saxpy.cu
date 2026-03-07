/**
 * SAXPY: y[i] += a * x[i]
 */
extern "C" __global__ void saxpy(
    float* y,
    const float* x,
    const float a,
    const int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    y[i] += a * x[i];
}
