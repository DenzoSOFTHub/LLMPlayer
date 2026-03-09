/**
 * IQ3_S dequantize + matrix-vector multiply kernel.
 * 256 weights per super-block, 110 bytes per block.
 * Layout: [d:fp16 (2B)][qs:64B grid index low][qh:8B grid index high bit][signs:32B][scales:4B]
 *
 * 8 groups of 32 weights, processed in pairs (64 weights at a time).
 * Each pair shares a scale byte (low nibble for first 32, high nibble for second 32).
 * Grid index (9 bits): 8 low bits from qs + 1 high bit from qh.
 * Grid lookup: IQ3S_GRID (512 uint32 entries), each encoding 4 unsigned byte values.
 * Scale formula: d * (1 + 2 * scale_nibble)
 *
 * Each warp (32 threads) computes one output row, striping across super-blocks.
 */
__device__ __forceinline__ float half2float(unsigned short h) {
    unsigned int sign = (h >> 15) & 1;
    unsigned int exp = (h >> 10) & 0x1F;
    unsigned int mantissa = h & 0x3FF;
    if (exp == 0) {
        if (mantissa == 0) return sign ? -0.0f : 0.0f;
        while (!(mantissa & 0x400)) { mantissa <<= 1; exp--; }
        exp++; mantissa &= 0x3FF;
    } else if (exp == 31) {
        unsigned int f = (sign << 31) | 0x7F800000 | (mantissa << 13);
        return *(float*)&f;
    }
    unsigned int f = (sign << 31) | ((exp + 112) << 23) | (mantissa << 13);
    return *(float*)&f;
}

__device__ __constant__ unsigned int IQ3S_GRID[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x01010107, 0x01010301, 0x01010303, 0x01010305, 0x01010307,
    0x01010501, 0x01010503, 0x01010505, 0x01010507, 0x01010701, 0x01010703, 0x01010705, 0x01010707,
    0x01030101, 0x01030103, 0x01030105, 0x01030107, 0x01030301, 0x01030303, 0x01030305, 0x01030307,
    0x01030501, 0x01030503, 0x01030505, 0x01030507, 0x01030701, 0x01030703, 0x01030705, 0x01030707,
    0x01050101, 0x01050103, 0x01050105, 0x01050107, 0x01050301, 0x01050303, 0x01050305, 0x01050307,
    0x01050501, 0x01050503, 0x01050505, 0x01050507, 0x01050701, 0x01050703, 0x01050705, 0x01050707,
    0x01070101, 0x01070103, 0x01070105, 0x01070107, 0x01070301, 0x01070303, 0x01070305, 0x01070307,
    0x01070501, 0x01070503, 0x01070505, 0x01070507, 0x01070701, 0x01070703, 0x01070705, 0x01070707,
    0x03010101, 0x03010103, 0x03010105, 0x03010107, 0x03010301, 0x03010303, 0x03010305, 0x03010307,
    0x03010501, 0x03010503, 0x03010505, 0x03010507, 0x03010701, 0x03010703, 0x03010705, 0x03010707,
    0x03030101, 0x03030103, 0x03030105, 0x03030107, 0x03030301, 0x03030303, 0x03030305, 0x03030307,
    0x03030501, 0x03030503, 0x03030505, 0x03030507, 0x03030701, 0x03030703, 0x03030705, 0x03030707,
    0x03050101, 0x03050103, 0x03050105, 0x03050107, 0x03050301, 0x03050303, 0x03050305, 0x03050307,
    0x03050501, 0x03050503, 0x03050505, 0x03050507, 0x03050701, 0x03050703, 0x03050705, 0x03050707,
    0x03070101, 0x03070103, 0x03070105, 0x03070107, 0x03070301, 0x03070303, 0x03070305, 0x03070307,
    0x03070501, 0x03070503, 0x03070505, 0x03070507, 0x03070701, 0x03070703, 0x03070705, 0x03070707,
    0x05010101, 0x05010103, 0x05010105, 0x05010107, 0x05010301, 0x05010303, 0x05010305, 0x05010307,
    0x05010501, 0x05010503, 0x05010505, 0x05010507, 0x05010701, 0x05010703, 0x05010705, 0x05010707,
    0x05030101, 0x05030103, 0x05030105, 0x05030107, 0x05030301, 0x05030303, 0x05030305, 0x05030307,
    0x05030501, 0x05030503, 0x05030505, 0x05030507, 0x05030701, 0x05030703, 0x05030705, 0x05030707,
    0x05050101, 0x05050103, 0x05050105, 0x05050107, 0x05050301, 0x05050303, 0x05050305, 0x05050307,
    0x05050501, 0x05050503, 0x05050505, 0x05050507, 0x05050701, 0x05050703, 0x05050705, 0x05050707,
    0x05070101, 0x05070103, 0x05070105, 0x05070107, 0x05070301, 0x05070303, 0x05070305, 0x05070307,
    0x05070501, 0x05070503, 0x05070505, 0x05070507, 0x05070701, 0x05070703, 0x05070705, 0x05070707,
    0x07010101, 0x07010103, 0x07010105, 0x07010107, 0x07010301, 0x07010303, 0x07010305, 0x07010307,
    0x07010501, 0x07010503, 0x07010505, 0x07010507, 0x07010701, 0x07010703, 0x07010705, 0x07010707,
    0x07030101, 0x07030103, 0x07030105, 0x07030107, 0x07030301, 0x07030303, 0x07030305, 0x07030307,
    0x07030501, 0x07030503, 0x07030505, 0x07030507, 0x07030701, 0x07030703, 0x07030705, 0x07030707,
    0x07050101, 0x07050103, 0x07050105, 0x07050107, 0x07050301, 0x07050303, 0x07050305, 0x07050307,
    0x07050501, 0x07050503, 0x07050505, 0x07050507, 0x07050701, 0x07050703, 0x07050705, 0x07050707,
    0x07070101, 0x07070103, 0x07070105, 0x07070107, 0x07070301, 0x07070303, 0x07070305, 0x07070307,
    0x07070501, 0x07070503, 0x07070505, 0x07070507, 0x07070701, 0x07070703, 0x07070705, 0x07070707,
    0x01010109, 0x0101010b, 0x01010309, 0x0101030b, 0x01010509, 0x0101050b, 0x01010709, 0x0101070b,
    0x01030109, 0x0103010b, 0x01030309, 0x0103030b, 0x01030509, 0x0103050b, 0x01030709, 0x0103070b,
    0x01050109, 0x0105010b, 0x01050309, 0x0105030b, 0x01050509, 0x0105050b, 0x01050709, 0x0105070b,
    0x01070109, 0x0107010b, 0x01070309, 0x0107030b, 0x01070509, 0x0107050b, 0x01070709, 0x0107070b,
    0x03010109, 0x0301010b, 0x03010309, 0x0301030b, 0x03010509, 0x0301050b, 0x03010709, 0x0301070b,
    0x03030109, 0x0303010b, 0x03030309, 0x0303030b, 0x03030509, 0x0303050b, 0x03030709, 0x0303070b,
    0x03050109, 0x0305010b, 0x03050309, 0x0305030b, 0x03050509, 0x0305050b, 0x03050709, 0x0305070b,
    0x03070109, 0x0307010b, 0x03070309, 0x0307030b, 0x03070509, 0x0307050b, 0x03070709, 0x0307070b,
    0x05010109, 0x0501010b, 0x05010309, 0x0501030b, 0x05010509, 0x0501050b, 0x05010709, 0x0501070b,
    0x05030109, 0x0503010b, 0x05030309, 0x0503030b, 0x05030509, 0x0503050b, 0x05030709, 0x0503070b,
    0x05050109, 0x0505010b, 0x05050309, 0x0505030b, 0x05050509, 0x0505050b, 0x05050709, 0x0505070b,
    0x05070109, 0x0507010b, 0x05070309, 0x0507030b, 0x05070509, 0x0507050b, 0x05070709, 0x0507070b,
    0x07010109, 0x0701010b, 0x07010309, 0x0701030b, 0x07010509, 0x0701050b, 0x07010709, 0x0701070b,
    0x07030109, 0x0703010b, 0x07030309, 0x0703030b, 0x07030509, 0x0703050b, 0x07030709, 0x0703070b,
    0x07050109, 0x0705010b, 0x07050309, 0x0705030b, 0x07050509, 0x0705050b, 0x07050709, 0x0705070b,
    0x07070109, 0x0707010b, 0x07070309, 0x0707030b, 0x07070509, 0x0707050b, 0x07070709, 0x0707070b,
    0x01010901, 0x01010903, 0x01010905, 0x01010907, 0x01010b01, 0x01010b03, 0x01010b05, 0x01010b07,
    0x01030901, 0x01030903, 0x01030905, 0x01030907, 0x01030b01, 0x01030b03, 0x01030b05, 0x01030b07,
    0x01050901, 0x01050903, 0x01050905, 0x01050907, 0x01050b01, 0x01050b03, 0x01050b05, 0x01050b07,
    0x01070901, 0x01070903, 0x01070905, 0x01070907, 0x01070b01, 0x01070b03, 0x01070b05, 0x01070b07,
    0x03010901, 0x03010903, 0x03010905, 0x03010907, 0x03010b01, 0x03010b03, 0x03010b05, 0x03010b07,
    0x03030901, 0x03030903, 0x03030905, 0x03030907, 0x03030b01, 0x03030b03, 0x03030b05, 0x03030b07,
    0x03050901, 0x03050903, 0x03050905, 0x03050907, 0x03050b01, 0x03050b03, 0x03050b05, 0x03050b07,
    0x03070901, 0x03070903, 0x03070905, 0x03070907, 0x03070b01, 0x03070b03, 0x03070b05, 0x03070b07,
    0x05010901, 0x05010903, 0x05010905, 0x05010907, 0x05010b01, 0x05010b03, 0x05010b05, 0x05010b07,
    0x05030901, 0x05030903, 0x05030905, 0x05030907, 0x05030b01, 0x05030b03, 0x05030b05, 0x05030b07,
    0x05050901, 0x05050903, 0x05050905, 0x05050907, 0x05050b01, 0x05050b03, 0x05050b05, 0x05050b07,
    0x05070901, 0x05070903, 0x05070905, 0x05070907, 0x05070b01, 0x05070b03, 0x05070b05, 0x05070b07,
    0x07010901, 0x07010903, 0x07010905, 0x07010907, 0x07010b01, 0x07010b03, 0x07010b05, 0x07010b07,
    0x07030901, 0x07030903, 0x07030905, 0x07030907, 0x07030b01, 0x07030b03, 0x07030b05, 0x07030b07,
    0x07050901, 0x07050903, 0x07050905, 0x07050907, 0x07050b01, 0x07050b03, 0x07050b05, 0x07050b07,
    0x07070901, 0x07070903, 0x07070905, 0x07070907, 0x07070b01, 0x07070b03, 0x07070b05, 0x07070b07
};

extern "C" __global__ void matmul_iq3_s(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    const int rows,
    const int cols,
    const int addToOutput)
{
    int warpId = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int rowsPerBlock = blockDim.x / 32;
    int row = blockIdx.x * rowsPerBlock + warpId;
    if (row >= rows) return;

    int numSuperBlocks = cols / 256;
    int rowStride = numSuperBlocks * 110;
    float sum = 0.0f;

    for (int sb = lane; sb < numSuperBlocks; sb += 32) {
        int bo = row * rowStride + sb * 110;
        float d = half2float(*(unsigned short*)(weights + bo));

        int inputBase = sb * 256;

        // Process in pairs of 32-weight groups (64 weights at a time)
        #pragma unroll
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            int scaleByte = __ldg(weights + bo + 106 + ib32 / 2); // OFF_SCALES
            float db1 = d * (float)(1 + 2 * (scaleByte & 0x0F));
            float db2 = d * (float)(1 + 2 * ((scaleByte >> 4) & 0x0F));

            // First 32 weights of pair
            int qhByte0 = __ldg(weights + bo + 66 + ib32 / 4); // OFF_QH
            int qsBase1 = ib32 * 8;
            int signBase1 = ib32 * 4;

            #pragma unroll
            for (int l = 0; l < 4; l++) {
                int qs0 = __ldg(weights + bo + 2 + qsBase1 + 2 * l);
                int qs1 = __ldg(weights + bo + 2 + qsBase1 + 2 * l + 1);
                unsigned int grid1 = IQ3S_GRID[qs0 | ((qhByte0 << (8 - 2 * l)) & 256)];
                unsigned int grid2 = IQ3S_GRID[qs1 | ((qhByte0 << (7 - 2 * l)) & 256)];
                int signs = __ldg(weights + bo + 74 + signBase1 + l); // OFF_SIGNS

                int weightIdx = inputBase + ib32 * 32 + l * 8;

                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int gv = (grid1 >> (8 * j)) & 0xFF;
                    float sign = (signs & (1 << j)) ? -1.0f : 1.0f;
                    sum += db1 * (float)gv * sign * input[weightIdx + j];
                }
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int gv = (grid2 >> (8 * j)) & 0xFF;
                    float sign = (signs & (1 << (j + 4))) ? -1.0f : 1.0f;
                    sum += db1 * (float)gv * sign * input[weightIdx + 4 + j];
                }
            }

            // Second 32 weights of pair
            int qhByte1 = __ldg(weights + bo + 66 + ib32 / 4 + 1); // OFF_QH
            int qsBase2 = (ib32 + 1) * 8;
            int signBase2 = (ib32 + 1) * 4;

            #pragma unroll
            for (int l = 0; l < 4; l++) {
                int qs0 = __ldg(weights + bo + 2 + qsBase2 + 2 * l);
                int qs1 = __ldg(weights + bo + 2 + qsBase2 + 2 * l + 1);
                unsigned int grid1 = IQ3S_GRID[qs0 | ((qhByte1 << (8 - 2 * l)) & 256)];
                unsigned int grid2 = IQ3S_GRID[qs1 | ((qhByte1 << (7 - 2 * l)) & 256)];
                int signs = __ldg(weights + bo + 74 + signBase2 + l); // OFF_SIGNS

                int weightIdx = inputBase + (ib32 + 1) * 32 + l * 8;

                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int gv = (grid1 >> (8 * j)) & 0xFF;
                    float sign = (signs & (1 << j)) ? -1.0f : 1.0f;
                    sum += db2 * (float)gv * sign * input[weightIdx + j];
                }
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int gv = (grid2 >> (8 * j)) & 0xFF;
                    float sign = (signs & (1 << (j + 4))) ? -1.0f : 1.0f;
                    sum += db2 * (float)gv * sign * input[weightIdx + 4 + j];
                }
            }
        }
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (lane == 0) {
        if (addToOutput) output[row] += sum;
        else output[row] = sum;
    }
}
