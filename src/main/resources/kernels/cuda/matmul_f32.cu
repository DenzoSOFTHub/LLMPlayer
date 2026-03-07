/**
 * F32 matrix-vector multiply kernel.
 * Each warp (32 threads) computes one output row: out[row] += dot(weights[row*cols..], input[0..cols])
 * Uses float4 vectorized loads for both weights and input (4x fewer memory transactions).
 */
extern "C" __global__ void matmul_f32(
    const float* __restrict__ weights,
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
    int cols4 = cols / 4;
    const float4* w4 = (const float4*)(weights + base);
    const float4* in4 = (const float4*)input;
    for (int i = lane; i < cols4; i += 32) {
        float4 w = w4[i];
        float4 inp = in4[i];
        sum += w.x * inp.x + w.y * inp.y + w.z * inp.z + w.w * inp.w;
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (lane == 0) {
        if (addToOutput) output[row] += sum;
        else output[row] = sum;
    }
}
