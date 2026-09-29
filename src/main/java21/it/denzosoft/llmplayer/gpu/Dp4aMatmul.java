package it.denzosoft.llmplayer.gpu;

import it.denzosoft.llmplayer.tensor.CudaFloatTensor;
import it.denzosoft.llmplayer.tensor.GGMLType;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Shared GPU matmul dispatcher: the dp4a int8 kernel for the weight's quant type when one exists
 * (input quantized to Q8_1 first), otherwise the tensor's own FP32 kernel.
 *
 * <ul>
 *   <li><b>Coverage</b>: Q4_K, Q5_K, Q3_K, Q5_0, Q8_0, IQ4_NL, IQ4_XS, and Q6_K when
 *       {@code -Dcuda.dp4a.q6=true} (measured ~3% slower than FP32 on the RTX 4050, so opt-in, as in
 *       the standard pass).</li>
 *   <li><b>Quantize once</b>: consecutive matmuls on the same input buffer (gate/up, every expert
 *       of an MoE layer) reuse one Q8_1 quantization. The caller must call {@link #invalidate()}
 *       whenever it writes a buffer that may be a matmul input by any other means than
 *       {@link #matmul}.</li>
 *   <li><b>Geometry</b>: every dp4a kernel maps one warp to one row ({@code row = blockIdx.x *
 *       (blockDim.x / 32) + warp}), so it is launched with 256 threads and rows/8 blocks, not with
 *       the FP32 kernel's geometry (which opt-in Q4_K variants change).</li>
 *   <li><b>Offset weights</b>: {@code weightOffset} addresses a 2D slice of a 3D tensor (one MoE
 *       expert) without a separate upload.</li>
 * </ul>
 *
 * Not thread-safe: one instance per stream-owning pass. Allocates nothing per call.
 */
public final class Dp4aMatmul {

    private static final boolean DP4A = !"false".equals(System.getProperty("cuda.dp4a", "true"));
    private static final boolean DP4A_Q6 = "true".equals(System.getProperty("cuda.dp4a.q6", "false"));
    private static final boolean DP4A_Q3 = !"false".equals(System.getProperty("cuda.dp4a.q3", "true"));
    private static final boolean DP4A_Q5 = !"false".equals(System.getProperty("cuda.dp4a.q5", "true"));

    private final CudaContext ctx;
    private final MemorySegment stream;
    private final boolean enabled;
    private final long q8Buf;
    private final MemorySegment quantizeFunc;
    private final MemorySegment q4k, q5k, q6k, q3k, q50, q80, iq4nl, iq4xs;
    private final KernelParams quantPB, dp4aPB, fp32PB;
    private long cachedIn;
    private int cachedCols;

    /**
     * @param maxCols largest input length any matmul will use (sizes the Q8_1 scratch buffer)
     */
    public Dp4aMatmul(CudaContext ctx, CudaBufferManager bm, Arena arena, int maxCols) {
        this.ctx = ctx;
        this.stream = ctx.getStream();
        MemorySegment qf = null, a = null, b = null, c = null, d = null, e = null, f = null, g = null, h = null;
        boolean ok = false;
        if (DP4A) {
            try {
                qf = ctx.compileKernel("kernels/cuda/quantize_q8.cu", "quantize_q8");
                a = ctx.compileKernel("kernels/cuda/matmul_q4_k_dp4a.cu", "matmul_q4_k_dp4a");
                ok = true;
                b = tryCompile("kernels/cuda/matmul_q5_k_dp4a.cu", "matmul_q5_k_dp4a");
                c = DP4A_Q6 ? tryCompile("kernels/cuda/matmul_q6_k_dp4a.cu", "matmul_q6_k_dp4a") : null;
                d = tryCompile("kernels/cuda/matmul_q3_k_dp4a.cu", "matmul_q3_k_dp4a");
                e = tryCompile("kernels/cuda/matmul_q5_0_dp4a.cu", "matmul_q5_0_dp4a");
                f = tryCompile("kernels/cuda/matmul_q8_0_dp4a.cu", "matmul_q8_0_dp4a");
                g = tryCompile("kernels/cuda/matmul_iq4_nl_dp4a.cu", "matmul_iq4_nl_dp4a");
                h = tryCompile("kernels/cuda/matmul_iq4_xs_dp4a.cu", "matmul_iq4_xs_dp4a");
            } catch (Exception ex) {
                ok = false;
            }
        }
        this.enabled = ok;
        this.quantizeFunc = qf;
        this.q4k = a; this.q5k = b; this.q6k = c; this.q3k = d;
        this.q50 = e; this.q80 = f; this.iq4nl = g; this.iq4xs = h;
        this.q8Buf = ok ? bm.createBuffer((long) ((maxCols + 31) / 32) * 40) : 0;
        this.quantPB = new KernelParams(arena, 3);
        this.dp4aPB = new KernelParams(arena, 6);
        this.fp32PB = new KernelParams(arena, 6);
    }

    private MemorySegment tryCompile(String res, String name) {
        try { return ctx.compileKernel(res, name); } catch (Exception e) { return null; }
    }

    /** The dp4a kernel for {@code type}, or null when the FP32 kernel must be used. */
    private MemorySegment kernelFor(GGMLType type) {
        if (!enabled) return null;
        switch (type) {
            case Q4_K:   return q4k;
            case Q5_K:   return DP4A_Q5 ? q5k : null;
            case Q6_K:   return q6k;
            case Q3_K:   return DP4A_Q3 ? q3k : null;
            case Q5_0:   return q50;
            case Q8_0:   return q80;
            case IQ4_NL: return iq4nl;
            case IQ4_XS: return iq4xs;
            default:     return null;
        }
    }

    /** True when matmuls on {@code t} take the int8 path. */
    public boolean isDp4a(CudaFloatTensor t) {
        return kernelFor(t.type()) != null;
    }

    /** Forget the cached quantization (an input buffer was rewritten outside {@link #matmul}). */
    public void invalidate() {
        cachedIn = 0;
    }

    /**
     * {@code out[r] (+)= W[r] · in} for rows [0, rows), W = the weights of {@code t} starting at
     * byte {@code weightOffset}.
     */
    public void matmul(CudaFloatTensor t, long weightOffset, long in, long out, int rows, int cols,
                       boolean accumulate) {
        MemorySegment k = kernelFor(t.type());
        if (k != null) {
            if (cachedIn != in || cachedCols != cols) {
                quantPB.setLong(0, in).setLong(1, q8Buf).setInt(2, cols);
                launch(quantizeFunc, (((cols + 31) / 32) + 7) / 8, 256, 0, quantPB);
                cachedIn = in;
                cachedCols = cols;
            }
            dp4aPB.setLong(0, t.getGpuWeights() + weightOffset).setLong(1, q8Buf).setLong(2, out)
                  .setInt(3, rows).setInt(4, cols).setInt(5, accumulate ? 1 : 0);
            launch(k, (rows + 7) / 8, 256, 0, dp4aPB);
        } else {
            fp32PB.setLong(0, t.getGpuWeights() + weightOffset).setLong(1, in).setLong(2, out)
                  .setInt(3, rows).setInt(4, cols).setInt(5, accumulate ? 1 : 0);
            launch(t.getCudaFunction(), t.getMatmulGridDim(rows, cols), t.getMatmulBlockDim(cols),
                t.getMatmulSharedMem(cols), fp32PB);
        }
        if (out == cachedIn) cachedIn = 0;
    }

    private void launch(MemorySegment fn, int grid, int block, int sm, KernelParams p) {
        int err = CudaBindings.launchKernel(fn, grid, 1, 1, block, 1, 1, sm, stream, p.ptrs(), MemorySegment.NULL);
        if (err != CudaBindings.CUDA_SUCCESS) throw new RuntimeException("CUDA matmul launch error: " + err);
    }
}
