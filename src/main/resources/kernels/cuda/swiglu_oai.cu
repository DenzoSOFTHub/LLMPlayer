/**
 * GPT-OSS SwiGLU variant (llama.cpp swiglu_oai, alpha = 1.702, limit = 7):
 *   x = min(gate, 7); y = clamp(up, -7, 7); gate = x * sigmoid(1.702 x) * (y + 1)
 * In place on gate. Same math as ExpertGpuCache.swigluOai on the CPU.
 */
extern "C" __global__ void swiglu_oai(
    float* gate,
    const float* up,
    const int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float x = fminf(gate[i], 7.0f);
    float y = fmaxf(-7.0f, fminf(up[i], 7.0f));
    float glu = x / (1.0f + __expf(-1.702f * x));
    gate[i] = glu * (y + 1.0f);
}
