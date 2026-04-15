/**
 * Q4_K matmul with double-buffered cp.async input prefetch (Ampere+).
 *
 * Same compute structure as `matmul_q4_k_smem` (32 threads per warp process the same
 * group with coalesced byte reads + shared-memory input). The only difference: while
 * compute consumes the current input tile, the next super-block's input is being copied
 * from global to shared memory via `cp.async.cg.shared.global` — bypassing registers
 * and overlapping with weight reads + arithmetic.
 *
 * Block: 256 threads (8 warps × 8 rows). Shared memory: 2 KB (two 256-float tiles).
 *
 * cp.async is sm_80+ (Ampere). On older devices NVRTC will fail to compile and the
 * kernel falls back to the standard path (caller catches the load failure).
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

// cp.async.ca.shared.global: copy 16 bytes from global to shared, cache in L1+L2.
// 16-byte (float4) granularity = full cache line, lowest per-instruction overhead.
__device__ __forceinline__ void cp_async16(unsigned int smem_addr, const void* gmem_ptr) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_addr), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

extern "C" __global__ void matmul_q4_k_cpasync(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    const int rows,
    const int cols,
    const int addToOutput)
{
    __shared__ float tile[2][256];  // double-buffered input tile (1 super-block each)

    int warpId = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int rowsPerBlock = blockDim.x / 32;
    int row = blockIdx.x * rowsPerBlock + warpId;

    int numBlocks = cols / 256;
    long rowOffset = (row < rows) ? (long)row * numBlocks * 144 : 0;
    float sum = 0.0f;

    // 64 threads per block participate in the input tile copy:
    // each copies 16 bytes (one float4) → 64 × 16 = 1024 B = 256 floats per tile.
    // Other threads do nothing during the copy phase.
    int copyTid = threadIdx.x;  // 0..63 do the copy, others skip
    bool isCopier = copyTid < 64;
    unsigned int tile0_smem = isCopier ? __cvta_generic_to_shared(&tile[0][copyTid * 4]) : 0;
    unsigned int tile1_smem = isCopier ? __cvta_generic_to_shared(&tile[1][copyTid * 4]) : 0;

    // Pre-issue async load of tile 0
    if (isCopier) {
        cp_async16(tile0_smem, &input[0 * 256 + copyTid * 4]);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    int buf = 0;
    for (int b = 0; b < numBlocks; b++) {
        // Issue async load of NEXT super-block (b+1) into the alternate buffer
        if (b + 1 < numBlocks && isCopier) {
            unsigned int dst = (buf == 0) ? tile1_smem : tile0_smem;
            cp_async16(dst, &input[(b + 1) * 256 + copyTid * 4]);
        }
        if (b + 1 < numBlocks) cp_async_commit();

        // Compute on the CURRENT buffer (already filled, fully visible to all threads)
        if (row < rows) {
            long bo = rowOffset + (long)b * 144;

            unsigned int dm = __ldg((const unsigned int*)(weights + bo));
            float d = half2float(dm & 0xFFFF);
            float dmin = half2float(dm >> 16);

            unsigned int sc0 = __ldg((const unsigned int*)(weights + bo + 4));
            unsigned int sc1 = __ldg((const unsigned int*)(weights + bo + 8));
            unsigned int sc2 = __ldg((const unsigned int*)(weights + bo + 12));

            unsigned char sb[12];
            sb[0] = sc0 & 0xFF; sb[1] = (sc0 >> 8) & 0xFF;
            sb[2] = (sc0 >> 16) & 0xFF; sb[3] = (sc0 >> 24) & 0xFF;
            sb[4] = sc1 & 0xFF; sb[5] = (sc1 >> 8) & 0xFF;
            sb[6] = (sc1 >> 16) & 0xFF; sb[7] = (sc1 >> 24) & 0xFF;
            sb[8] = sc2 & 0xFF; sb[9] = (sc2 >> 8) & 0xFF;
            sb[10] = (sc2 >> 16) & 0xFF; sb[11] = (sc2 >> 24) & 0xFF;

            #pragma unroll
            for (int group = 0; group < 4; group++) {
                int sb0 = group * 2;
                int sb1 = group * 2 + 1;
                int scale0, min0, scale1, min1;
                if (sb0 < 4) { scale0 = sb[sb0] & 0x3F; min0 = sb[sb0 + 4] & 0x3F; }
                else {
                    scale0 = (sb[sb0 + 4] & 0x0F) | ((sb[sb0 - 4] >> 6) << 4);
                    min0 = ((sb[sb0 + 4] >> 4) & 0x0F) | ((sb[sb0] >> 6) << 4);
                }
                if (sb1 < 4) { scale1 = sb[sb1] & 0x3F; min1 = sb[sb1 + 4] & 0x3F; }
                else {
                    scale1 = (sb[sb1 + 4] & 0x0F) | ((sb[sb1 - 4] >> 6) << 4);
                    min1 = ((sb[sb1 + 4] >> 4) & 0x0F) | ((sb[sb1] >> 6) << 4);
                }

                float ds0 = d * (float)scale0;
                float dm0 = dmin * (float)min0;
                float ds1 = d * (float)scale1;
                float dm1 = dmin * (float)min1;

                unsigned char qByte = __ldg(&weights[bo + 16 + group * 32 + lane]);
                float q0 = (float)(qByte & 0x0F);
                float q1 = (float)((qByte >> 4) & 0x0F);

                float in0 = tile[buf][group * 64 + lane];
                float in1 = tile[buf][group * 64 + 32 + lane];

                sum += (ds0 * q0 - dm0) * in0;
                sum += (ds1 * q1 - dm1) * in1;
            }
        }

        // Wait for the prefetch of b+1 to complete before swapping buffers
        if (b + 1 < numBlocks) {
            cp_async_wait_all();
            __syncthreads();
            buf = 1 - buf;
        } else {
            __syncthreads();
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (lane == 0 && row < rows) {
        if (addToOutput) output[row] += sum;
        else output[row] = sum;
    }
}
