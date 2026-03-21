// Convert FP32 vector to FP16. One thread per element.
extern "C" __global__ void convert_f32_to_f16(
    const float* __restrict__ input,
    unsigned short* __restrict__ output,
    int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float f = input[i];
    unsigned int x = *(unsigned int*)&f;
    unsigned int sign = (x >> 16) & 0x8000;
    int exp = ((x >> 23) & 0xFF) - 127;
    unsigned int mantissa = x & 0x7FFFFF;
    if (exp > 15) { output[i] = sign | 0x7C00; return; }
    if (exp < -14) { output[i] = sign; return; }
    output[i] = sign | ((exp + 15) << 10) | (mantissa >> 13);
}
