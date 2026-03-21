// a[i] *= sigmoid(b[i])
// Used for attention output gating: xb2 *= sigmoid(attnGate)
extern "C" __global__ void sigmoid_elementwise_mul(
    float* __restrict__ a,         // [size] in-place: a[i] *= sigmoid(b[i])
    const float* __restrict__ b,   // [size] gate values (sigmoid applied inline)
    int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        a[i] *= 1.0f / (1.0f + expf(-b[i]));
    }
}
