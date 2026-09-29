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
    // Generated from IQGridTables.java (ggml iq3s_grid); the previous
    // hand-copied table was wrong, so every IQ2_S/IQ3_S GPU matmul produced garbage.
    0x01010101U, 0x01010103U, 0x01010105U, 0x0101010bU, 0x0101010fU, 0x01010301U, 0x01010303U, 0x01010305U,
    0x01010309U, 0x0101030dU, 0x01010501U, 0x01010503U, 0x0101050bU, 0x01010707U, 0x01010901U, 0x01010905U,
    0x0101090bU, 0x0101090fU, 0x01010b03U, 0x01010b07U, 0x01010d01U, 0x01010d05U, 0x01010f03U, 0x01010f09U,
    0x01010f0fU, 0x01030101U, 0x01030103U, 0x01030105U, 0x01030109U, 0x01030301U, 0x01030303U, 0x0103030bU,
    0x01030501U, 0x01030507U, 0x0103050fU, 0x01030703U, 0x0103070bU, 0x01030909U, 0x01030d03U, 0x01030d0bU,
    0x01030f05U, 0x01050101U, 0x01050103U, 0x0105010bU, 0x0105010fU, 0x01050301U, 0x01050307U, 0x0105030dU,
    0x01050503U, 0x0105050bU, 0x01050701U, 0x01050709U, 0x01050905U, 0x0105090bU, 0x0105090fU, 0x01050b03U,
    0x01050b07U, 0x01050f01U, 0x01050f07U, 0x01070107U, 0x01070303U, 0x0107030bU, 0x01070501U, 0x01070505U,
    0x01070703U, 0x01070707U, 0x0107070dU, 0x01070909U, 0x01070b01U, 0x01070b05U, 0x01070d0fU, 0x01070f03U,
    0x01070f0bU, 0x01090101U, 0x01090307U, 0x0109030fU, 0x01090503U, 0x01090509U, 0x01090705U, 0x01090901U,
    0x01090907U, 0x01090b03U, 0x01090f01U, 0x010b0105U, 0x010b0109U, 0x010b0501U, 0x010b0505U, 0x010b050dU,
    0x010b0707U, 0x010b0903U, 0x010b090bU, 0x010b090fU, 0x010b0d0dU, 0x010b0f07U, 0x010d010dU, 0x010d0303U,
    0x010d0307U, 0x010d0703U, 0x010d0b05U, 0x010d0f03U, 0x010f0101U, 0x010f0105U, 0x010f0109U, 0x010f0501U,
    0x010f0505U, 0x010f050dU, 0x010f0707U, 0x010f0b01U, 0x010f0b09U, 0x03010101U, 0x03010103U, 0x03010105U,
    0x03010109U, 0x03010301U, 0x03010303U, 0x03010307U, 0x0301030bU, 0x0301030fU, 0x03010501U, 0x03010505U,
    0x03010703U, 0x03010709U, 0x0301070dU, 0x03010b09U, 0x03010b0dU, 0x03010d03U, 0x03010f05U, 0x03030101U,
    0x03030103U, 0x03030107U, 0x0303010dU, 0x03030301U, 0x03030309U, 0x03030503U, 0x03030701U, 0x03030707U,
    0x03030903U, 0x03030b01U, 0x03030b05U, 0x03030f01U, 0x03030f0dU, 0x03050101U, 0x03050305U, 0x0305030bU,
    0x0305030fU, 0x03050501U, 0x03050509U, 0x03050705U, 0x03050901U, 0x03050907U, 0x03050b0bU, 0x03050d01U,
    0x03050f05U, 0x03070103U, 0x03070109U, 0x0307010fU, 0x03070301U, 0x03070307U, 0x03070503U, 0x0307050fU,
    0x03070701U, 0x03070709U, 0x03070903U, 0x03070d05U, 0x03070f01U, 0x03090107U, 0x0309010bU, 0x03090305U,
    0x03090309U, 0x03090703U, 0x03090707U, 0x03090905U, 0x0309090dU, 0x03090b01U, 0x03090b09U, 0x030b0103U,
    0x030b0301U, 0x030b0307U, 0x030b0503U, 0x030b0701U, 0x030b0705U, 0x030b0b03U, 0x030d0501U, 0x030d0509U,
    0x030d050fU, 0x030d0909U, 0x030d090dU, 0x030f0103U, 0x030f0107U, 0x030f0301U, 0x030f0305U, 0x030f0503U,
    0x030f070bU, 0x030f0903U, 0x030f0d05U, 0x030f0f01U, 0x05010101U, 0x05010103U, 0x05010107U, 0x0501010bU,
    0x0501010fU, 0x05010301U, 0x05010305U, 0x05010309U, 0x0501030dU, 0x05010503U, 0x05010507U, 0x0501050fU,
    0x05010701U, 0x05010705U, 0x05010903U, 0x05010907U, 0x0501090bU, 0x05010b01U, 0x05010b05U, 0x05010d0fU,
    0x05010f01U, 0x05010f07U, 0x05010f0bU, 0x05030101U, 0x05030105U, 0x05030301U, 0x05030307U, 0x0503030fU,
    0x05030505U, 0x0503050bU, 0x05030703U, 0x05030709U, 0x05030905U, 0x05030b03U, 0x05050103U, 0x05050109U,
    0x0505010fU, 0x05050503U, 0x05050507U, 0x05050701U, 0x0505070fU, 0x05050903U, 0x05050b07U, 0x05050b0fU,
    0x05050f03U, 0x05050f09U, 0x05070101U, 0x05070105U, 0x0507010bU, 0x05070303U, 0x05070505U, 0x05070509U,
    0x05070703U, 0x05070707U, 0x05070905U, 0x05070b01U, 0x05070d0dU, 0x05090103U, 0x0509010fU, 0x05090501U,
    0x05090507U, 0x05090705U, 0x0509070bU, 0x05090903U, 0x05090f05U, 0x05090f0bU, 0x050b0109U, 0x050b0303U,
    0x050b0505U, 0x050b070fU, 0x050b0901U, 0x050b0b07U, 0x050b0f01U, 0x050d0101U, 0x050d0105U, 0x050d010fU,
    0x050d0503U, 0x050d0b0bU, 0x050d0d03U, 0x050f010bU, 0x050f0303U, 0x050f050dU, 0x050f0701U, 0x050f0907U,
    0x050f0b01U, 0x07010105U, 0x07010303U, 0x07010307U, 0x0701030bU, 0x0701030fU, 0x07010505U, 0x07010703U,
    0x07010707U, 0x0701070bU, 0x07010905U, 0x07010909U, 0x0701090fU, 0x07010b03U, 0x07010d07U, 0x07010f03U,
    0x07030103U, 0x07030107U, 0x0703010bU, 0x07030309U, 0x07030503U, 0x07030507U, 0x07030901U, 0x07030d01U,
    0x07030f05U, 0x07030f0dU, 0x07050101U, 0x07050305U, 0x07050501U, 0x07050705U, 0x07050709U, 0x07050b01U,
    0x07070103U, 0x07070301U, 0x07070309U, 0x07070503U, 0x07070507U, 0x0707050fU, 0x07070701U, 0x07070903U,
    0x07070907U, 0x0707090fU, 0x07070b0bU, 0x07070f07U, 0x07090107U, 0x07090303U, 0x0709030dU, 0x07090505U,
    0x07090703U, 0x07090b05U, 0x07090d01U, 0x07090d09U, 0x070b0103U, 0x070b0301U, 0x070b0305U, 0x070b050bU,
    0x070b0705U, 0x070b0909U, 0x070b0b0dU, 0x070b0f07U, 0x070d030dU, 0x070d0903U, 0x070f0103U, 0x070f0107U,
    0x070f0501U, 0x070f0505U, 0x070f070bU, 0x09010101U, 0x09010109U, 0x09010305U, 0x09010501U, 0x09010509U,
    0x0901050fU, 0x09010705U, 0x09010903U, 0x09010b01U, 0x09010f01U, 0x09030105U, 0x0903010fU, 0x09030303U,
    0x09030307U, 0x09030505U, 0x09030701U, 0x0903070bU, 0x09030907U, 0x09030b03U, 0x09030b0bU, 0x09050103U,
    0x09050107U, 0x09050301U, 0x0905030bU, 0x09050503U, 0x09050707U, 0x09050901U, 0x09050b0fU, 0x09050d05U,
    0x09050f01U, 0x09070109U, 0x09070303U, 0x09070307U, 0x09070501U, 0x09070505U, 0x09070703U, 0x0907070bU,
    0x09090101U, 0x09090105U, 0x09090509U, 0x0909070fU, 0x09090901U, 0x09090f03U, 0x090b010bU, 0x090b010fU,
    0x090b0503U, 0x090b0d05U, 0x090d0307U, 0x090d0709U, 0x090d0d01U, 0x090f0301U, 0x090f030bU, 0x090f0701U,
    0x090f0907U, 0x090f0b03U, 0x0b010105U, 0x0b010301U, 0x0b010309U, 0x0b010505U, 0x0b010901U, 0x0b010909U,
    0x0b01090fU, 0x0b010b05U, 0x0b010d0dU, 0x0b010f09U, 0x0b030103U, 0x0b030107U, 0x0b03010bU, 0x0b030305U,
    0x0b030503U, 0x0b030705U, 0x0b030f05U, 0x0b050101U, 0x0b050303U, 0x0b050507U, 0x0b050701U, 0x0b05070dU,
    0x0b050b07U, 0x0b070105U, 0x0b07010fU, 0x0b070301U, 0x0b07050fU, 0x0b070909U, 0x0b070b03U, 0x0b070d0bU,
    0x0b070f07U, 0x0b090103U, 0x0b090109U, 0x0b090501U, 0x0b090705U, 0x0b09090dU, 0x0b0b0305U, 0x0b0b050dU,
    0x0b0b0b03U, 0x0b0b0b07U, 0x0b0d0905U, 0x0b0f0105U, 0x0b0f0109U, 0x0b0f0505U, 0x0d010303U, 0x0d010307U,
    0x0d01030bU, 0x0d010703U, 0x0d010707U, 0x0d010d01U, 0x0d030101U, 0x0d030501U, 0x0d03050fU, 0x0d030d09U,
    0x0d050305U, 0x0d050709U, 0x0d050905U, 0x0d050b0bU, 0x0d050d05U, 0x0d050f01U, 0x0d070101U, 0x0d070309U,
    0x0d070503U, 0x0d070901U, 0x0d09050bU, 0x0d090907U, 0x0d090d05U, 0x0d0b0101U, 0x0d0b0107U, 0x0d0b0709U,
    0x0d0b0d01U, 0x0d0d010bU, 0x0d0d0901U, 0x0d0f0303U, 0x0d0f0307U, 0x0f010101U, 0x0f010109U, 0x0f01010fU,
    0x0f010501U, 0x0f010505U, 0x0f01070dU, 0x0f010901U, 0x0f010b09U, 0x0f010d05U, 0x0f030105U, 0x0f030303U,
    0x0f030509U, 0x0f030907U, 0x0f03090bU, 0x0f050103U, 0x0f050109U, 0x0f050301U, 0x0f05030dU, 0x0f050503U,
    0x0f050701U, 0x0f050b03U, 0x0f070105U, 0x0f070705U, 0x0f07070bU, 0x0f070b07U, 0x0f090103U, 0x0f09010bU,
    0x0f090307U, 0x0f090501U, 0x0f090b01U, 0x0f0b0505U, 0x0f0b0905U, 0x0f0d0105U, 0x0f0d0703U, 0x0f0f0101U,
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
            int qhByte0 = __ldg(weights + bo + 66 + ib32); // OFF_QH: one qh byte per 32-weight group (ggml)
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
            int qhByte1 = __ldg(weights + bo + 66 + ib32 + 1); // OFF_QH: qh advances by 2 per pair of groups (ggml)
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

// FP16 row-range dequantization for the batched prefill GEMM (see dequant_f16.cu for the contract).
extern "C" __global__ void dequant_iq3_s_f16t(const unsigned char* __restrict__ w, unsigned short* __restrict__ out,
                                              int rowStart, int nRows, int cols) {
    long idx = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long) nRows * cols) return;
    int rr = (int) (idx / cols), col = (int) (idx - (long) rr * cols);
    long bo = ((long) rowStart + rr) * (cols / 256) * 110 + (long) (col / 256) * 110;
    int j = col & 255, ib32 = j / 32, l = (j % 32) / 8, jj = j % 8;
    float d = half2float((unsigned short) w[bo] | ((unsigned short) w[bo + 1] << 8));
    int scaleByte = w[bo + 106 + ib32 / 2];
    float db = d * (float) (1 + 2 * ((ib32 & 1) ? ((scaleByte >> 4) & 0x0F) : (scaleByte & 0x0F)));
    int qhByte = w[bo + 66 + ib32];
    int second = jj >> 2;
    int qs = w[bo + 2 + ib32 * 8 + 2 * l + second];
    unsigned int grid = IQ3S_GRID[qs | ((qhByte << ((second ? 7 : 8) - 2 * l)) & 256)];
    int gv = (grid >> (8 * (jj & 3))) & 0xFF;
    float v = db * (float) gv * ((w[bo + 74 + ib32 * 4 + l] & (1 << jj)) ? -1.0f : 1.0f);
    unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
    out[idx] = h;
}
