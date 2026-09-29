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
 * addressed by an OFFSET into the tensor's GPU buffer, through the shared {@link Dp4aMatmul}
 * dispatcher: dp4a for every quant type that has an int8 kernel, the tensor's FP32 kernel otherwise.
 * The layer input is quantized to Q8_1 once for all gate/up matmuls of the layer (routed and shared
 * experts), and each expert's SwiGLU output once for its down projection.
 *
 * Contained: does NOT touch {@code NemotronHCudaForwardPass}; the dense Nemotron-H / Granite-dense
 * GPU forward pass is unaffected. If anything fails the caller falls back to the CPU expert path.
 */
public final class GraniteExpertGpu implements it.denzosoft.llmplayer.inference.GpuMoeExperts {

    private final CudaContext ctx;
    private final CudaBufferManager bm;
    private final Arena arena;
    private final MemorySegment stream;

    private final int dim, expertCount, eFfn, shFfn;
    private final long gpuIn, gpuGate, gpuUp, gpuExpertOut, gpuOut;
    private final MemorySegment hostIn, hostOut;

    private final MemorySegment siluMulFunc, saxpyFunc, accumFunc, fillZeroFunc;
    private final long blockSize;

    // Two dispatchers, each with its own Q8_1 scratch: gate/up read gpuIn (quantized once per
    // layer), down reads gpuGate (quantized once per expert).
    private final Dp4aMatmul gateUpMm, downMm;

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
    private final PB siluMulPB, saxpyPB, accumPB, fillPB;

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

        gateUpMm = new Dp4aMatmul(ctx, bm, arena, dim);
        downMm = new Dp4aMatmul(ctx, bm, arena, maxFfn);
        siluMulFunc  = ctx.compileKernel("kernels/cuda/silu_mul.cu", "silu_mul");
        saxpyFunc    = ctx.compileKernel("kernels/cuda/saxpy.cu", "saxpy");
        accumFunc    = ctx.compileKernel("kernels/cuda/accumulate.cu", "accumulate");
        fillZeroFunc = ctx.compileKernel("kernels/cuda/fill_zero.cu", "fill_zero");

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
        gateUpMm.invalidate(); // new layer input

        // gpuOut = 0
        fillPB.setLong(0, gpuOut); fillPB.setInt(1, dim);
        launch(fillZeroFunc, grid(dim), (int) blockSize, fillPB);

        long gateBpe = ((CudaFloatTensor) gateExps).getWeightsBytes() / expertCount;
        long upBpe   = ((CudaFloatTensor) upExps).getWeightsBytes() / expertCount;
        long downBpe = ((CudaFloatTensor) downExps).getWeightsBytes() / expertCount;

        for (int k = 0; k < used; k++) {
            int e = sel[k];
            gateUpMm.matmul((CudaFloatTensor) gateExps, (long) e * gateBpe, gpuIn, gpuGate, eFfn, dim, false);
            gateUpMm.matmul((CudaFloatTensor) upExps, (long) e * upBpe, gpuIn, gpuUp, eFfn, dim, false);
            siluMul(gpuGate, gpuUp, eFfn);                 // gpuGate = silu(gpuGate) * gpuUp
            downMm.invalidate();
            downMm.matmul((CudaFloatTensor) downExps, (long) e * downBpe, gpuGate, gpuExpertOut, dim, eFfn, false);
            saxpy(gpuOut, gpuExpertOut, weights[k], dim);  // gpuOut += w_k * expertOut
        }

        if (gateShexp != null) {
            gateUpMm.matmul((CudaFloatTensor) gateShexp, 0, gpuIn, gpuGate, shFfn, dim, false);
            gateUpMm.matmul((CudaFloatTensor) upShexp, 0, gpuIn, gpuUp, shFfn, dim, false);
            siluMul(gpuGate, gpuUp, shFfn);
            downMm.invalidate();
            downMm.matmul((CudaFloatTensor) downShexp, 0, gpuGate, gpuExpertOut, dim, shFfn, false);
            accum(gpuOut, gpuExpertOut, dim);
        }

        ctx.readBuffer(gpuOut, hostOut, (long) dim * Float.BYTES);
        MemorySegment.copy(hostOut, ValueLayout.JAVA_FLOAT, 0, out, 0, dim);
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
