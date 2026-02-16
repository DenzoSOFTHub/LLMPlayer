/**
 * Q3_K dequantize + matrix-vector multiply kernel.
 * 256 weights per super-block, 110 bytes per block.
 * Layout: [hmask:32B][qs:64B][scales:12B][d:fp16]
 */

__kernel void matmul_q3_k(
    __global const uchar* weights,
    __global const float* input,
    __global float* output,
    const int rows,
    const int cols)
{
    int row = get_global_id(0);
    if (row >= rows) return;

    int numBlocks = cols / 256;
    int rowStride = numBlocks * 110;
    float sum = 0.0f;

    for (int b = 0; b < numBlocks; b++) {
        int bo = row * rowStride + b * 110;

        // d is at offset 108
        float d = vload_half(0, (__global const half*)(weights + bo + 108));

        // Decode 16 scales from 12 packed bytes at offset 96
        uchar raw[12];
        for (int i = 0; i < 12; i++) raw[i] = weights[bo + 96 + i];

        int sc[16];
        for (int i = 0; i < 8; i++) {
            sc[i] = raw[i] & 0x0F;
            sc[i + 8] = raw[i] >> 4;
        }
        for (int i = 0; i < 4; i++) {
            sc[i]      |= (raw[8 + i] & 0x03) << 4;
            sc[i + 4]  |= ((raw[8 + i] >> 2) & 0x03) << 4;
            sc[i + 8]  |= ((raw[8 + i] >> 4) & 0x03) << 4;
            sc[i + 12] |= ((raw[8 + i] >> 6) & 0x03) << 4;
        }

        int inputBase = b * 256;
        int scaleIdx = 0;
        int hmBitPos = 0;

        for (int hf = 0; hf < 2; hf++) {
            int qBase = hf * 32;

            for (int pair = 0; pair < 4; pair++) {
                int shift = pair * 2;
                float dl0 = d * (float)(sc[scaleIdx] - 32);
                scaleIdx++;
                float dl1 = d * (float)(sc[scaleIdx] - 32);
                scaleIdx++;

                int elemBase = hf * 128 + pair * 32;

                #pragma unroll 4
                for (int l = 0; l < 16; l++) {
                    uchar qsByte = weights[bo + 32 + qBase + l];
                    uchar hmByte = weights[bo + l];
                    int lowBits = (qsByte >> shift) & 3;
                    int hbit = (hmByte >> hmBitPos) & 1;
                    int q = (lowBits | (hbit << 2)) - 4;
                    sum += dl0 * (float)q * input[inputBase + elemBase + l];
                }

                #pragma unroll 4
                for (int l = 0; l < 16; l++) {
                    uchar qsByte = weights[bo + 32 + qBase + 16 + l];
                    uchar hmByte = weights[bo + 16 + l];
                    int lowBits = (qsByte >> shift) & 3;
                    int hbit = (hmByte >> hmBitPos) & 1;
                    int q = (lowBits | (hbit << 2)) - 4;
                    sum += dl1 * (float)q * input[inputBase + elemBase + 16 + l];
                }

                hmBitPos++;
            }
        }
    }
    output[row] += sum;
}
