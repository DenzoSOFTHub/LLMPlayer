/**
 * Q6_K dequantize + matrix-vector multiply kernel.
 * 256 weights per super-block, 210 bytes per block.
 * Layout: [ql:128B][qh:64B][scales:16B][d:fp16]
 */

__kernel void matmul_q6_k(
    __global const uchar* weights,
    __global const float* input,
    __global float* output,
    const int rows,
    const int cols)
{
    int row = get_global_id(0);
    if (row >= rows) return;

    int numBlocks = cols / 256;
    int rowStride = numBlocks * 210;
    float sum = 0.0f;

    for (int b = 0; b < numBlocks; b++) {
        int bo = row * rowStride + b * 210;

        // d is at offset 208 (after ql:128 + qh:64 + scales:16)
        float d = vload_half(0, (__global const half*)(weights + bo + 208));

        int inputBase = b * 256;

        for (int hf = 0; hf < 2; hf++) {
            int qlOff = hf * 64;
            int qhOff = hf * 32;
            int scBase = hf * 8;
            int elemBase = hf * 128;

            for (int l = 0; l < 32; l++) {
                uchar qlByte0 = weights[bo + qlOff + l];
                uchar qlByte1 = weights[bo + qlOff + 32 + l];
                uchar qhByte  = weights[bo + 128 + qhOff + l];

                int q0 = (qlByte0 & 0x0F)        | (((qhByte >> 0) & 3) << 4);
                int q1 = (qlByte1 & 0x0F)        | (((qhByte >> 2) & 3) << 4);
                int q2 = ((qlByte0 >> 4) & 0x0F) | (((qhByte >> 4) & 3) << 4);
                int q3 = ((qlByte1 >> 4) & 0x0F) | (((qhByte >> 6) & 3) << 4);

                int scIdx = scBase + (l / 16);
                // scales are int8 (signed)
                char sc0 = (char)weights[bo + 192 + scIdx];
                char sc1 = (char)weights[bo + 192 + scIdx + 2];
                char sc2 = (char)weights[bo + 192 + scIdx + 4];
                char sc3 = (char)weights[bo + 192 + scIdx + 6];

                sum += d * (float)sc0 * (float)(q0 - 32) * input[inputBase + elemBase + l];
                sum += d * (float)sc1 * (float)(q1 - 32) * input[inputBase + elemBase + 32 + l];
                sum += d * (float)sc2 * (float)(q2 - 32) * input[inputBase + elemBase + 64 + l];
                sum += d * (float)sc3 * (float)(q3 - 32) * input[inputBase + elemBase + 96 + l];
            }
        }
    }
    output[row] += sum;
}
