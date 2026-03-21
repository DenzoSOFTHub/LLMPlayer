// Depthwise causal conv1d with fixed short kernel (width 3-4).
// Each thread handles one channel. Conv state is a circular buffer on GPU.
// convState layout: [histSize][channels] where histSize = kernelSize - 1
// convWeights layout: [channels][kernelSize] (GGUF row-major)
// PyTorch convention: weight[K-1] = current input, weight[0] = oldest
extern "C" __global__ void conv1d_short(
    float* __restrict__ qkv,               // [channels] input/output (in-place)
    float* __restrict__ convState,          // [histSize * channels] circular buffer
    const float* __restrict__ convWeights,  // [channels * kernelSize]
    int channels,
    int kernelSize,         // typically 4
    int* __restrict__ tokenParams  // tokenParams[0] = position (= convPos before this step)
) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    int convPos = tokenParams[0];  // position = number of tokens already stored
    int histSize = kernelSize - 1; // 3

    float current = qkv[ch];

    // output[t] = sum_k(weight[K-1-k] * input[t-k]) for k=0..K-1
    // k=0: weight[K-1] * current_input
    float sum = convWeights[ch * kernelSize + (kernelSize - 1)] * current;

    // k=1..K-1: weight[K-1-k] * history[t-k]
    for (int k = 1; k < kernelSize; k++) {
        if (convPos - k >= 0) {
            int histIdx = (convPos - k) % histSize;
            sum += convWeights[ch * kernelSize + (kernelSize - 1 - k)]
                 * convState[histIdx * channels + ch];
        }
    }

    // Store current value into circular buffer BEFORE overwriting output
    convState[(convPos % histSize) * channels + ch] = current;

    // Write convolved result
    qkv[ch] = sum;
}
