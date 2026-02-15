/**
 * Q4_0 dequantize + matrix-vector multiply kernel.
 * 32 weights per block, 18 bytes per block.
 * Layout: [scale:fp16 (2B)][16 bytes nibbles (32 x 4-bit)]
 * value = (nibble - 8) * scale
 */

__kernel void matmul_q4_0(
    __global const uchar* weights,
    __global const float* input,
    __global float* output,
    const int rows,
    const int cols)
{
    int row = get_global_id(0);
    if (row >= rows) return;

    int numBlocks = cols / 32;
    int rowStride = numBlocks * 18;
    float sum = 0.0f;

    for (int b = 0; b < numBlocks; b++) {
        int bo = row * rowStride + b * 18;

        float scale = vload_half(0, (__global const half*)(weights + bo));

        int inputBase = b * 32;
        for (int i = 0; i < 16; i++) {
            uchar packed = weights[bo + 2 + i];
            int lo = (packed & 0x0F) - 8;
            int hi = ((packed >> 4) & 0x0F) - 8;

            sum += scale * (float)lo * input[inputBase + i * 2];
            sum += scale * (float)hi * input[inputBase + i * 2 + 1];
        }
    }
    output[row] += sum;
}
