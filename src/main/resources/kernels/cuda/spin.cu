// Clock keeper (opt-in, -Dcuda.clockkeeper): a small kernel that keeps one SM busy for about
// `iters` FMA iterations. Launched in a loop on a side stream so that the GPU's boost governor
// does not drop to the idle power state between the short attention bursts of a decode step in
// which most of the work runs on the CPU (measured: P8 at 210 MHz SM / 405 MHz memory for the
// whole generation of MiniMax-M2 with the routed experts on the CPU).
extern "C" __global__ void gpu_spin(float* sink, const int iters)
{
    float a = threadIdx.x * 1e-3f, b = 1.0001f;
    for (int i = 0; i < iters; i++) { a = a * b + 1e-7f; b = b * 0.99999f + 1e-6f; }
    if (a == 12345.678f) sink[threadIdx.x] = a; // never true: keeps the loop alive
}
