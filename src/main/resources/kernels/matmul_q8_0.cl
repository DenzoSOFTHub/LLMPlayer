/**
 * Q8_0 dequantize + matrix-vector multiply kernel.
 * 32 weights per block, 34 bytes per block.
 * Layout: [scale:fp16 (2B)][32 x int8 quants (32B)]
 */

__kernel void matmul_q8_0(
    __global const uchar* weights,
    __global const float* input,
    __global float* output,
    const int rows,
    const int cols)
{
    int row = get_global_id(0);
    if (row >= rows) return;

    int numBlocks = cols / 32;
    int rowStride = numBlocks * 34;
    float sum = 0.0f;

    for (int b = 0; b < numBlocks; b++) {
        int bo = row * rowStride + b * 34;

        float scale = vload_half(0, (__global const half*)(weights + bo));

        int inputBase = b * 32;
        #pragma unroll 8
        for (int i = 0; i < 32; i++) {
            // int8 quant (signed)
            char q = (char)weights[bo + 2 + i];
            sum += scale * (float)q * input[inputBase + i];
        }
    }
    output[row] += sum;
}
