// In-place scalar multiply: x[i] *= scale
extern "C" __global__ void scale_inplace(float* __restrict__ x, float scale, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) x[i] *= scale;
}
