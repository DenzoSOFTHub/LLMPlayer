/**
 * Zero-fill: x[i] = 0.0f
 */
__kernel void fill_zero(
    __global float* x,
    const int size)
{
    int i = get_global_id(0);
    if (i >= size) return;
    x[i] = 0.0f;
}
