package it.denzosoft.llmplayer.gpu;

import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.tensor.CudaFloatTensor;
import it.denzosoft.llmplayer.tensor.FloatTensor;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * GPU acceleration for Granite Hybrid MoE expert FFN (e.g. granite-4.0-h-tiny).
 *
 * The router top-K runs on the CPU engine; this helper computes the routed-expert + shared-expert
 * SwiGLU on the GPU. Each routed expert's 2D weight slice inside the 3D {@code ffn_*_exps} tensor is
 * addressed by an OFFSET into the tensor's GPU buffer (no new kernel — reuses each tensor's own FP32
 * matmul kernel via its public accessors), so Q4_K/Q6_K experts work unchanged.
 *
 * Contained: does NOT touch {@code NemotronHCudaForwardPass}; the dense Nemotron-H / Granite-dense
 * GPU forward pass is unaffected. If anything fails the caller falls back to the CPU expert path.
 */
public final class GraniteExpertGpu implements AutoCloseable {

    private final CudaContext ctx;
    private final CudaBufferManager bm;
    private final Arena arena;
    private final MemorySegment stream;

    private final int dim, expertCount, eFfn, shFfn;
    private final long gpuIn, gpuGate, gpuUp, gpuExpertOut, gpuOut;
    private final MemorySegment hostIn, hostOut;

    private final MemorySegment siluMulFunc, saxpyFunc, accumFunc, fillZeroFunc;
    private final long blockSize;

    // dp4a for expert matmuls (Q4_K gate/up are the bottleneck; Q6_K down -> FP32 fallback).
    private final boolean useDp4a = !"false".equals(System.getProperty("cuda.dp4a", "true"));
    private final MemorySegment quantizeFunc, dp4aQ4kFunc, dp4aQ5kFunc;
    private final long gpuQ8In;
    private final PB quantPB, dp4aPB;

    private static final class PB {
        final MemorySegment args, ptrs;
        PB(Arena a, int n) {
            args = a.allocate(n * 8L, 8);
            ptrs = a.allocate(ValueLayout.ADDRESS, n);
            for (int i = 0; i < n; i++) ptrs.setAtIndex(ValueLayout.ADDRESS, i, args.asSlice(i * 8L, 8));
        }
        void setLong(int i, long v) { args.set(ValueLayout.JAVA_LONG, i * 8L, v); }
        void setInt(int i, int v) { args.set(ValueLayout.JAVA_INT, i * 8L, v); }
        void setFloat(int i, float v) { args.set(ValueLayout.JAVA_FLOAT, i * 8L, v); }
    }
    private final PB matmulPB, siluMulPB, saxpyPB, accumPB, fillPB;

    public GraniteExpertGpu(ModelConfig config, CudaBufferManager bufferManager) {
        this.bm = bufferManager;
        this.ctx = bufferManager.getCudaContext();
        this.arena = Arena.ofShared();
        this.stream = ctx.getStream();
        this.dim = config.embeddingLength();
        this.expertCount = config.expertCount();
        this.eFfn = config.expertFfnLength() > 0 ? config.expertFfnLength() : config.intermediateSize();
        this.shFfn = Math.max(config.expertSharedFeedForwardLength(), 1);
        int maxFfn = Math.max(eFfn, shFfn);
        long fb = Float.BYTES;
        long maxWg = ctx.getDeviceInfo().maxWorkGroupSize();
        this.blockSize = Math.min(256, maxWg);

        gpuIn = bm.createBuffer((long) dim * fb);
        gpuGate = bm.createBuffer((long) maxFfn * fb);
        gpuUp = bm.createBuffer((long) maxFfn * fb);
        gpuExpertOut = bm.createBuffer((long) dim * fb);
        gpuOut = bm.createBuffer((long) dim * fb);
        hostIn = arena.allocate(ValueLayout.JAVA_FLOAT, dim);
        hostOut = arena.allocate(ValueLayout.JAVA_FLOAT, dim);

        int maxIn = Math.max(dim, Math.max(eFfn, shFfn));
        gpuQ8In = useDp4a ? bm.createBuffer((long) ((maxIn + 31) / 32) * 40) : 0;
        if (useDp4a) {
            quantizeFunc = ctx.compileKernel("kernels/cuda/quantize_q8.cu", "quantize_q8");
            dp4aQ4kFunc  = ctx.compileKernel("kernels/cuda/matmul_q4_k_dp4a.cu", "matmul_q4_k_dp4a");
            dp4aQ5kFunc  = ctx.compileKernel("kernels/cuda/matmul_q5_k_dp4a.cu", "matmul_q5_k_dp4a");
        } else { quantizeFunc = dp4aQ4kFunc = dp4aQ5kFunc = null; }
        siluMulFunc  = ctx.compileKernel("kernels/cuda/silu_mul.cu", "silu_mul");
        saxpyFunc    = ctx.compileKernel("kernels/cuda/saxpy.cu", "saxpy");
        accumFunc    = ctx.compileKernel("kernels/cuda/accumulate.cu", "accumulate");
        fillZeroFunc = ctx.compileKernel("kernels/cuda/fill_zero.cu", "fill_zero");

        matmulPB  = new PB(arena, 6);
        quantPB   = new PB(arena, 3);
        dp4aPB    = new PB(arena, 6);
        siluMulPB = new PB(arena, 3);
        saxpyPB   = new PB(arena, 4);
        accumPB   = new PB(arena, 3);
        fillPB    = new PB(arena, 2);
    }

    /**
     * Compute the MoE FFN output for one token on the GPU.
     * @param input        normed input (FFN norm output), length dim
     * @param sel          selected expert indices, length >= used
     * @param weights      renormalized routing weights, length >= used
     * @param used         number of active experts (top-K)
     * @param out          output buffer (length dim) — overwritten with the MoE result (routed + shared)
     */
    public void computeMoE(FloatTensor gateExps, FloatTensor upExps, FloatTensor downExps,
                           FloatTensor gateShexp, FloatTensor upShexp, FloatTensor downShexp,
                           float[] input, int[] sel, float[] weights, int used, float[] out) {
        MemorySegment.copy(input, 0, hostIn, ValueLayout.JAVA_FLOAT, 0, dim);
        ctx.writeBuffer(gpuIn, hostIn, (long) dim * Float.BYTES);

        // gpuOut = 0
        fillPB.setLong(0, gpuOut); fillPB.setInt(1, dim);
        launch(fillZeroFunc, grid(dim), (int) blockSize, fillPB);

        long gateBpe = ((CudaFloatTensor) gateExps).getWeightsBytes() / expertCount;
        long upBpe   = ((CudaFloatTensor) upExps).getWeightsBytes() / expertCount;
        long downBpe = ((CudaFloatTensor) downExps).getWeightsBytes() / expertCount;

        for (int k = 0; k < used; k++) {
            int e = sel[k];
            matmul(gateExps, gpuIn, gpuGate, eFfn, dim, (long) e * gateBpe);
            matmul(upExps,   gpuIn, gpuUp,   eFfn, dim, (long) e * upBpe);
            siluMul(gpuGate, gpuUp, eFfn);                 // gpuGate = silu(gpuGate) * gpuUp
            matmul(downExps, gpuGate, gpuExpertOut, dim, eFfn, (long) e * downBpe);
            saxpy(gpuOut, gpuExpertOut, weights[k], dim);  // gpuOut += w_k * expertOut
        }

        if (gateShexp != null) {
            matmul(gateShexp, gpuIn, gpuGate, shFfn, dim, 0);
            matmul(upShexp,   gpuIn, gpuUp,   shFfn, dim, 0);
            siluMul(gpuGate, gpuUp, shFfn);
            matmul(downShexp, gpuGate, gpuExpertOut, dim, shFfn, 0);
            accum(gpuOut, gpuExpertOut, dim);
        }

        ctx.readBuffer(gpuOut, hostOut, (long) dim * Float.BYTES);
        MemorySegment.copy(hostOut, ValueLayout.JAVA_FLOAT, 0, out, 0, dim);
    }

    private void matmul(FloatTensor t, long in, long out, int rows, int cols, long woff) {
        CudaFloatTensor ct = (CudaFloatTensor) t;
        MemorySegment dp4a = useDp4a ? dp4aFunc(ct) : null;
        if (dp4a != null) {
            quantPB.setLong(0, in); quantPB.setLong(1, gpuQ8In); quantPB.setInt(2, cols);
            launch(quantizeFunc, (((cols + 31) / 32) + 7) / 8, 256, 0, quantPB);
            dp4aPB.setLong(0, ct.getGpuWeights() + woff); dp4aPB.setLong(1, gpuQ8In); dp4aPB.setLong(2, out);
            dp4aPB.setInt(3, rows); dp4aPB.setInt(4, cols); dp4aPB.setInt(5, 0);
            launch(dp4a, ct.getMatmulGridDim(rows, cols), ct.getMatmulBlockDim(cols), 0, dp4aPB);
            return;
        }
        matmulPB.setLong(0, ct.getGpuWeights() + woff); matmulPB.setLong(1, in); matmulPB.setLong(2, out);
        matmulPB.setInt(3, rows); matmulPB.setInt(4, cols); matmulPB.setInt(5, 0);
        launch(ct.getCudaFunction(), ct.getMatmulGridDim(rows, cols), ct.getMatmulBlockDim(cols),
               ct.getMatmulSharedMem(cols), matmulPB);
    }

    private MemorySegment dp4aFunc(CudaFloatTensor t) {
        switch (t.type()) {
            case Q4_K: return dp4aQ4kFunc;
            case Q5_K: return dp4aQ5kFunc;
            default:   return null;   // Q6_K (down) / others -> FP32
        }
    }

    private void siluMul(long a, long b, int n) {
        siluMulPB.setLong(0, a); siluMulPB.setLong(1, b); siluMulPB.setInt(2, n);
        launch(siluMulFunc, grid(n), (int) blockSize, siluMulPB);
    }
    private void saxpy(long y, long x, float a, int n) {
        saxpyPB.setLong(0, y); saxpyPB.setLong(1, x); saxpyPB.setFloat(2, a); saxpyPB.setInt(3, n);
        launch(saxpyFunc, grid(n), (int) blockSize, saxpyPB);
    }
    private void accum(long y, long x, int n) {
        accumPB.setLong(0, y); accumPB.setLong(1, x); accumPB.setInt(2, n);
        launch(accumFunc, grid(n), (int) blockSize, accumPB);
    }
    private int grid(int n) { return (int) ((n + blockSize - 1) / blockSize); }
    private void launch(MemorySegment fn, int grid, int block, PB p) { launch(fn, grid, block, 0, p); }
    private void launch(MemorySegment fn, int grid, int block, int sm, PB p) {
        int err = CudaBindings.launchKernel(fn, grid, 1, 1, block, 1, 1, sm, stream, p.ptrs, MemorySegment.NULL);
        if (err != CudaBindings.CUDA_SUCCESS) throw new RuntimeException("GraniteExpertGpu CUDA error: " + err);
    }

    @Override public void close() { arena.close(); }
}
