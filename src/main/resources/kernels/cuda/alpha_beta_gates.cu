// Compute alpha and beta gates for DeltaNet recurrence.
// alpha[h] = exp(negExpA[h] * softplus(alphaProj[h] + dtBias[h]))
// beta[h] = sigmoid(betaProj[h])
// Input alpha/beta contain the raw projection outputs; overwritten with gate values.
extern "C" __global__ void alpha_beta_gates(
    float* __restrict__ alpha,             // [timeStepRank] in: projection, out: gate
    float* __restrict__ beta,              // [timeStepRank] in: projection, out: gate
    const float* __restrict__ negExpA,     // [timeStepRank] pre-computed -exp(A_log)
    const float* __restrict__ dtBias,      // [timeStepRank]
    int timeStepRank
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= timeStepRank) return;

    // softplus(x) = log(1 + exp(x)), clamped for numerical stability
    float a = alpha[h] + dtBias[h];
    float sp = (a > 20.0f) ? a : logf(1.0f + expf(a));
    alpha[h] = expf(negExpA[h] * sp);

    beta[h] = 1.0f / (1.0f + expf(-beta[h]));
}
