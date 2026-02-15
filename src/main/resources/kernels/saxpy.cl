/**
 * SAXPY: y[i] += a * x[i]
 */
__kernel void saxpy(
    __global float* y,
    __global const float* x,
    const float a,
    const int size)
{
    int i = get_global_id(0);
    if (i >= size) return;
    y[i] += a * x[i];
}
