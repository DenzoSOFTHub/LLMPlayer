/**
 * RMS normalization kernel.
 * Phase 1: Compute sum of squares (reduction).
 * Phase 2: Normalize: out[i] = x[i] * rsqrt(ss/size + eps) * w[i]
 *
 * Uses two-pass approach with local memory reduction.
 */
__kernel void rmsnorm_sumsq(
    __global const float* x,
    __global float* partial_sums,
    const int size,
    __local float* scratch)
{
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int group_size = get_local_size(0);
    int group_id = get_group_id(0);

    float sum = 0.0f;
    for (int i = gid; i < size; i += get_global_size(0)) {
        sum += x[i] * x[i];
    }
    scratch[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction
    for (int s = group_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) partial_sums[group_id] = scratch[0];
}

__kernel void rmsnorm_normalize(
    __global float* out,
    __global const float* x,
    __global const float* w,
    __global const float* partial_sums,
    const int size,
    const float eps,
    const int num_groups)
{
    int i = get_global_id(0);
    if (i >= size) return;

    // Sum partial sums
    float ss = 0.0f;
    for (int g = 0; g < num_groups; g++) {
        ss += partial_sums[g];
    }
    float scale = rsqrt(ss / size + eps);
    out[i] = x[i] * scale * w[i];
}
