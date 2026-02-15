/**
 * Accumulate: y[i] += x[i]
 */
__kernel void accumulate(
    __global float* y,
    __global const float* x,
    const int size)
{
    int i = get_global_id(0);
    if (i >= size) return;
    y[i] += x[i];
}
