/**
 * Q5_K dequantize + matrix-vector multiply kernel.
 * 256 weights per super-block, 176 bytes per block.
 * Layout: [d:fp16][dmin:fp16][scales:12B][qh:32B][qs:128B]
 *
 * CRITICAL: sub-blocks 4-7 nibble swap:
 *   LOW nibble of scaleBytes[sb+4] = scale
 *   HIGH nibble of scaleBytes[sb+4] = min
 */

__kernel void matmul_q5_k(
    __global const uchar* weights,
    __global const float* input,
    __global float* output,
    const int rows,
    const int cols)
{
    int row = get_global_id(0);
    if (row >= rows) return;

    int numBlocks = cols / 256;
    int rowStride = numBlocks * 176;
    float sum = 0.0f;

    for (int b = 0; b < numBlocks; b++) {
        int bo = row * rowStride + b * 176;

        float d = vload_half(0, (__global const half*)(weights + bo));
        float dmin = vload_half(0, (__global const half*)(weights + bo + 2));

        int scales[8], mins[8];
        for (int sb = 0; sb < 8; sb++) {
            if (sb < 4) {
                scales[sb] = weights[bo + 4 + sb] & 0x3F;
                mins[sb] = weights[bo + 4 + sb + 4] & 0x3F;
            } else {
                scales[sb] = (weights[bo + 4 + sb + 4] & 0x0F)
                           | ((weights[bo + 4 + sb - 4] >> 6) << 4);
                mins[sb] = ((weights[bo + 4 + sb + 4] >> 4) & 0x0F)
                         | ((weights[bo + 4 + sb] >> 6) << 4);
            }
        }

        int inputBase = b * 256;
        for (int group = 0; group < 4; group++) {
            float ds0 = d * (float)scales[group * 2];
            float dm0 = dmin * (float)mins[group * 2];
            float ds1 = d * (float)scales[group * 2 + 1];
            float dm1 = dmin * (float)mins[group * 2 + 1];

            for (int l = 0; l < 32; l++) {
                uchar qsByte = weights[bo + 48 + group * 32 + l];
                uchar qhByte = weights[bo + 16 + l];

                int ql0 = qsByte & 0x0F;
                int ql1 = (qsByte >> 4) & 0x0F;
                int qh0 = (qhByte >> (group * 2)) & 1;
                int qh1 = (qhByte >> (group * 2 + 1)) & 1;

                int q0 = ql0 | (qh0 << 4);
                int q1 = ql1 | (qh1 << 4);

                float w0 = ds0 * (float)q0 - dm0;
                float w1 = ds1 * (float)q1 - dm1;

                sum += w0 * input[inputBase + group * 64 + l];
                sum += w1 * input[inputBase + group * 64 + 32 + l];
            }
        }
    }
    output[row] += sum;
}
