extern "C" __global__ void split_qkv(
    const float* __restrict__ qkv,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    const int qDim,
    const int kvDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalDim = qDim + kvDim + kvDim;
    if (idx >= totalDim) return;
    float val = qkv[idx];
    if (idx < qDim) {
        q[idx] = val;
    } else if (idx < qDim + kvDim) {
        k[idx - qDim] = val;
    } else {
        v[idx - qDim - kvDim] = val;
    }
}
