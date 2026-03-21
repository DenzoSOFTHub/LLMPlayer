// Deinterleave packed Q+gate projection output.
// Input layout:  [Q_h0(headSize), gate_h0(headSize), Q_h1(headSize), gate_h1(headSize), ...]
// Output: separate Q[headCount * headSize] and gate[headCount * headSize]
extern "C" __global__ void deinterleave_q_gate(
    const float* __restrict__ packed,  // [headCount * headSize * 2]
    float* __restrict__ q,             // [headCount * headSize]
    float* __restrict__ gate,          // [headCount * headSize]
    int headCount, int headSize
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = headCount * headSize;
    if (i >= total) return;

    int head = i / headSize;
    int pos = i % headSize;
    int srcBase = head * headSize * 2;

    q[i] = packed[srcBase + pos];
    gate[i] = packed[srcBase + headSize + pos];
}
