/**
 * SiLU (Swish) activation: x[i] = x[i] / (1 + exp(-x[i]))
 */
__kernel void silu(
    __global float* x,
    const int size)
{
    int i = get_global_id(0);
    if (i >= size) return;
    float v = x[i];
    x[i] = v / (1.0f + exp(-v));
}
