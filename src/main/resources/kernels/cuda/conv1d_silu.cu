// Fused depthwise causal conv1d + SiLU activation.
// Combines conv1d_short + silu into a single kernel launch.
// Each thread handles one channel: convolve → store history → apply SiLU in-place.
extern "C" __global__ void conv1d_silu(
    float* __restrict__ qkv,               // [channels] input/output (in-place)
    float* __restrict__ convState,          // [histSize * channels] circular buffer
    const float* __restrict__ convWeights,  // [channels * kernelSize]
    int channels,
    int kernelSize,         // typically 4
    int* __restrict__ tokenParams  // tokenParams[0] = position (= convPos)
) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    int convPos = tokenParams[0];
    int histSize = kernelSize - 1; // 3

    float current = qkv[ch];

    // Causal conv1d: output = sum_k(weight[K-1-k] * input[t-k])
    float sum = convWeights[ch * kernelSize + (kernelSize - 1)] * current;
    for (int k = 1; k < kernelSize; k++) {
        if (convPos - k >= 0) {
            int histIdx = (convPos - k) % histSize;
            sum += convWeights[ch * kernelSize + (kernelSize - 1 - k)]
                 * convState[histIdx * channels + ch];
        }
    }

    // Store current value into circular buffer
    convState[(convPos % histSize) * channels + ch] = current;

    // Apply SiLU in-place: x * sigmoid(x)
    qkv[ch] = sum / (1.0f + expf(-sum));
}
