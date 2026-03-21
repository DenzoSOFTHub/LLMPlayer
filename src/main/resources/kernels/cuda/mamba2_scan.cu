// Mamba-2 SSM scan kernel (recurrent mode, single token).
// One block per head, headDim threads per block.
// Each thread handles one element d of head h: processes all stateSize state elements.
//
// S layout: [nheads][headDim][stateSize] (persistent on GPU)
// x layout: [nheads * headDim] (from xBC split after conv+SiLU)
// B layout: [ngroups * stateSize]
// C layout: [ngroups * stateSize]
// dt layout: [nheads] (already discretized: softplus(dt + dt_bias))
// A layout: [nheads] (stored as -exp(A_log), negative)
// D layout: [nheads] (residual connection)
// output layout: [nheads * headDim]
extern "C" __global__ void mamba2_scan(
    float* __restrict__ S,         // [nheads * headDim * stateSize] persistent state
    const float* __restrict__ x,   // [innerSize] = [nheads * headDim]
    const float* __restrict__ B,   // [ngroups * stateSize]
    const float* __restrict__ C,   // [ngroups * stateSize]
    const float* __restrict__ dt,  // [nheads] discretized timestep
    const float* __restrict__ A,   // [nheads] stored as -exp(A_log)
    const float* __restrict__ D,   // [nheads] residual
    float* __restrict__ output,    // [innerSize]
    int nheads, int headDim, int stateSize, int ngroups
) {
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (h >= nheads || d >= headDim) return;

    int group = h / (nheads / ngroups);
    float dtH = dt[h];
    float dA = expf(dtH * A[h]);  // A is negative → dA ∈ (0,1)
    float dH = D[h];

    // This head's state: S[h][d][0..stateSize-1]
    float* Shd = S + ((long long)h * headDim + d) * stateSize;

    float x_val = x[h * headDim + d] * dtH;

    // B and C for this group
    const float* Bg = B + group * stateSize;
    const float* Cg = C + group * stateSize;

    // SSM scan: S[n] = dA * S[n] + dt * B[n] * x, y = sum(S[n] * C[n])
    float y = 0.0f;
    #pragma unroll 4
    for (int n = 0; n < stateSize; n++) {
        float s_new = dA * Shd[n] + Bg[n] * x_val;
        Shd[n] = s_new;
        y += s_new * Cg[n];
    }

    // D residual
    output[h * headDim + d] = y + dH * x[h * headDim + d];
}
