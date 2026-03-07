/**
 * Element-wise multiply: a[i] *= b[i]
 */
__kernel void elementwise_mul(
    __global float* a,
    __global const float* b,
    const int size)
{
    int i = get_global_id(0);
    if (i >= size) return;
    a[i] *= b[i];
}
