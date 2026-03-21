// Quantize FP32 vector to Q8_1 blocks for dp4a matmul.
// Each block of 32 FP32 values → 40 bytes: [float scale][float sum][int8_t qs[32]]
// One warp per block. Grid = ceil(numBlocks / warpsPerBlock).
extern "C" __global__ void quantize_q8(
    const float* __restrict__ input,   // [size] FP32 input vector
    unsigned char* __restrict__ output, // [(size/32) * 40] Q8_1 output
    int size
) {
    int numBlocks = size / 32;
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warpId >= numBlocks) return;

    int inputOffset = warpId * 32;
    float val = input[inputOffset + lane];

    // Find max absolute value via warp-shuffle reduction
    float absVal = fabsf(val);
    float maxAbs = absVal;
    unsigned mask = 0xFFFFFFFF;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(mask, maxAbs, offset);
        maxAbs = fmaxf(maxAbs, other);
    }
    maxAbs = __shfl_sync(mask, maxAbs, 0);  // broadcast to all lanes

    // Compute scale
    float scale = maxAbs / 127.0f;
    float invScale = (maxAbs > 0.0f) ? (127.0f / maxAbs) : 0.0f;

    // Quantize
    int q = __float2int_rn(val * invScale);
    q = max(-127, min(127, q));

    // Compute sum of original FP32 values via warp reduction
    float sum = val;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Write output: [scale(4B)][sum(4B)][qs(32B)] = 40 bytes per block
    long outOffset = (long)warpId * 40;
    if (lane == 0) {
        *(float*)(output + outOffset) = scale;
        *(float*)(output + outOffset + 4) = sum;  // only lane 0 has the reduced sum
    }
    // All lanes write their quantized byte
    output[outOffset + 8 + lane] = (unsigned char)(q & 0xFF);
}
