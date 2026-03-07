/**
 * Element-wise multiply: a[i] *= b[i]
 */
extern "C" __global__ void elementwise_mul(
    float* a,
    const float* b,
    const int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    a[i] *= b[i];
}
