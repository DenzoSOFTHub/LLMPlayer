/**
 * Softmax kernel (two-pass with local memory reduction).
 * Pass 1: Find max value.
 * Pass 2: Compute exp(x - max) and sum.
 * Pass 3: Normalize by sum.
 */
__kernel void softmax_max(
    __global const float* logits,
    __global float* partial_max,
    const int offset,
    const int size,
    __local float* scratch)
{
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int group_size = get_local_size(0);
    int group_id = get_group_id(0);

    float maxVal = -INFINITY;
    for (int i = gid; i < size; i += get_global_size(0)) {
        float v = logits[offset + i];
        if (v > maxVal) maxVal = v;
    }
    scratch[lid] = maxVal;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = group_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) partial_max[group_id] = scratch[0];
}

__kernel void softmax_exp_sum(
    __global float* logits,
    __global float* partial_sums,
    __global const float* partial_max,
    const int offset,
    const int size,
    const int num_max_groups,
    __local float* scratch)
{
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int group_size = get_local_size(0);
    int group_id = get_group_id(0);

    // Find global max
    float maxVal = -INFINITY;
    for (int g = 0; g < num_max_groups; g++) {
        if (partial_max[g] > maxVal) maxVal = partial_max[g];
    }

    float sum = 0.0f;
    for (int i = gid; i < size; i += get_global_size(0)) {
        float v = exp(logits[offset + i] - maxVal);
        logits[offset + i] = v;
        sum += v;
    }
    scratch[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = group_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) partial_sums[group_id] = scratch[0];
}

__kernel void softmax_normalize(
    __global float* logits,
    __global const float* partial_sums,
    const int offset,
    const int size,
    const int num_sum_groups)
{
    int i = get_global_id(0);
    if (i >= size) return;

    float sum = 0.0f;
    for (int g = 0; g < num_sum_groups; g++) {
        sum += partial_sums[g];
    }
    logits[offset + i] *= 1.0f / sum;
}
