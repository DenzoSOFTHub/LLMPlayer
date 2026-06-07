// tanh-approximation GELU, in-place. Used by Gemma GeGLU FFN (gelu(gate) then elementwise_mul
// with up) and by the Gemma PLE input gate. Matches the CPU formula in Gemma4InferenceEngine:
//   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
extern "C" __global__ void gelu(float* __restrict__ x, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    }
}
