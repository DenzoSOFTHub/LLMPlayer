// Discretize Mamba-2 dt: dt[h] = softplus(dt[h] + dt_bias[h])
// In-place on dt buffer. nheads elements (typically 96).
extern "C" __global__ void mamba2_dt_softplus(
    float* __restrict__ dt,
    const float* __restrict__ dt_bias,
    int nheads
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= nheads) return;
    float d = dt[h] + dt_bias[h];
    dt[h] = (d > 20.0f) ? d : logf(1.0f + expf(d));
}
