/**
 * F32 matrix-vector multiply kernel.
 * Each work-item computes one output row: out[row] += dot(weights[row*cols..], input[0..cols])
 */
__kernel void matmul_f32(
    __global const float* weights,
    __global const float* input,
    __global float* output,
    const int rows,
    const int cols)
{
    int row = get_global_id(0);
    if (row >= rows) return;

    float sum = 0.0f;
    int base = row * cols;
    for (int i = 0; i < cols; i++) {
        sum += weights[base + i] * input[i];
    }
    output[row] += sum;
}
