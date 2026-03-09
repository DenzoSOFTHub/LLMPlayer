/**
 * BF16 matrix-vector multiply kernel.
 * Each element is 2 bytes (bfloat16). Conversion: shift left 16 to get float32.
 * Each warp (32 threads) computes one output row.
 */
extern "C" __global__ void matmul_bf16(
    const unsigned short* __restrict__ weights,
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

    float sum = 0.0f;
    int base = row * cols;
    for (int i = lane; i < cols; i += 32) {
        // BF16 to F32: upper 16 bits of float32
        unsigned int bits = (unsigned int)__ldg(weights + base + i) << 16;
        float w = *(float*)&bits;
        sum += w * input[i];
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (lane == 0) {
        if (addToOutput) output[row] += sum;
        else output[row] = sum;
    }
}
