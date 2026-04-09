/**
 * Q4_K matmul with 2 warps per output row.
 * Each warp handles HALF the columns. Partial sums combined via shared memory.
 * Grid: ceil(rows / (blockDim/64)) blocks, blockDim should be multiple of 64.
 * Enabled via -Dcuda.q4k.2warp=true
 */
__device__ __forceinline__ float h2f(unsigned short h) {
    unsigned int s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    if (e == 0) { if (m == 0) return s ? -0.0f : 0.0f; while (!(m & 0x400)) { m <<= 1; e--; } e++; m &= 0x3FF; }
    else if (e == 31) { unsigned int f = (s << 31) | 0x7F800000 | (m << 13); return *(float*)&f; }
    unsigned int f = (s << 31) | ((e + 112) << 23) | (m << 13);
    return *(float*)&f;
}

extern "C" __global__ void matmul_q4_k_2warp(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    const int rows,
    const int cols,
    const int addToOutput)
{
    // 2 warps = 64 threads per row
    const int WARPS_PER_ROW = 2;
    int globalThread = blockIdx.x * blockDim.x + threadIdx.x;
    int globalWarp = globalThread / 32;
    int row = globalWarp / WARPS_PER_ROW;
    int halfId = globalWarp % WARPS_PER_ROW;  // 0 or 1
    int lane = threadIdx.x & 31;

    if (row >= rows) return;

    int numBlocks = cols / 256;
    int numGroups = numBlocks * 4;
    // Split groups between the two warps
    int halfGroups = (numGroups + 1) / 2;
    int gStart = halfId * halfGroups;
    int gEnd = (halfId == 0) ? halfGroups : numGroups;

    long rowOffset = (long)row * numBlocks * 144;
    float sum = 0.0f;

    for (int g = gStart + lane; g < gEnd; g += 32) {
        int b = g >> 2;
        int group = g & 3;
        long bo = rowOffset + (long)b * 144;

        unsigned int dm = __ldg((const unsigned int*)(weights + bo));
        float d = h2f(dm & 0xFFFF);
        float dmin = h2f(dm >> 16);

        unsigned int sc0 = __ldg((const unsigned int*)(weights + bo + 4));
        unsigned int sc1 = __ldg((const unsigned int*)(weights + bo + 8));
        unsigned int sc2 = __ldg((const unsigned int*)(weights + bo + 12));

        unsigned char sb[12];
        sb[0]=sc0&0xFF; sb[1]=(sc0>>8)&0xFF; sb[2]=(sc0>>16)&0xFF; sb[3]=(sc0>>24)&0xFF;
        sb[4]=sc1&0xFF; sb[5]=(sc1>>8)&0xFF; sb[6]=(sc1>>16)&0xFF; sb[7]=(sc1>>24)&0xFF;
        sb[8]=sc2&0xFF; sb[9]=(sc2>>8)&0xFF; sb[10]=(sc2>>16)&0xFF; sb[11]=(sc2>>24)&0xFF;

        int sb0 = group * 2, sb1 = group * 2 + 1;
        int scale0, min0, scale1, min1;
        if (sb0 < 4) { scale0 = sb[sb0] & 0x3F; min0 = sb[sb0+4] & 0x3F; }
        else { scale0 = (sb[sb0+4]&0x0F)|((sb[sb0-4]>>6)<<4); min0 = ((sb[sb0+4]>>4)&0x0F)|((sb[sb0]>>6)<<4); }
        if (sb1 < 4) { scale1 = sb[sb1] & 0x3F; min1 = sb[sb1+4] & 0x3F; }
        else { scale1 = (sb[sb1+4]&0x0F)|((sb[sb1-4]>>6)<<4); min1 = ((sb[sb1+4]>>4)&0x0F)|((sb[sb1]>>6)<<4); }

        float ds0 = d*(float)scale0, dm0 = dmin*(float)min0;
        float ds1 = d*(float)scale1, dm1 = dmin*(float)min1;
        int inputBase = b * 256 + group * 64;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float4 in0 = __ldg((const float4*)(input + inputBase + i*4));
            float4 in1 = __ldg((const float4*)(input + inputBase + 32 + i*4));
            unsigned int qw = __ldg((const unsigned int*)(weights + bo + 16 + group*32 + i*4));
            sum += (ds0*(float)(qw&0x0F)-dm0)*in0.x + (ds1*(float)((qw>>4)&0x0F)-dm1)*in1.x;
            sum += (ds0*(float)((qw>>8)&0x0F)-dm0)*in0.y + (ds1*(float)((qw>>12)&0x0F)-dm1)*in1.y;
            sum += (ds0*(float)((qw>>16)&0x0F)-dm0)*in0.z + (ds1*(float)((qw>>20)&0x0F)-dm1)*in1.z;
            sum += (ds0*(float)((qw>>24)&0x0F)-dm0)*in0.w + (ds1*(float)((qw>>28)&0x0F)-dm1)*in1.w;
        }
    }

    // Intra-warp reduction
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    // Cross-warp reduction via shared memory
    __shared__ float warpPartials[64]; // max 32 rows * 2 warps
    int localWarp = threadIdx.x / 32;
    if (lane == 0) warpPartials[localWarp] = sum;
    __syncthreads();

    // First warp of each row writes result
    if (lane == 0 && halfId == 0) {
        float total = warpPartials[localWarp] + warpPartials[localWarp + 1];
        if (addToOutput) output[row] += total;
        else output[row] = total;
    }
}
