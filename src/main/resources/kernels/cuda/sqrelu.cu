// Squared ReLU activation: x = max(0, x)^2
// Used by Nemotron-H FFN layers (no SwiGLU gate).
extern "C" __global__ void sqrelu(float* __restrict__ x, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float v = x[i];
        x[i] = (v > 0.0f) ? v * v : 0.0f;
    }
}
