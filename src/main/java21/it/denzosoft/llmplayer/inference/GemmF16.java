package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.gpu.CublasBindings;
import it.denzosoft.llmplayer.gpu.CudaBufferManager;
import it.denzosoft.llmplayer.gpu.CudaContext;
import it.denzosoft.llmplayer.gpu.KernelParams;
import it.denzosoft.llmplayer.tensor.CudaFloatTensor;
import it.denzosoft.llmplayer.tensor.GGMLType;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.EnumMap;
import java.util.Map;

/**
 * Multi-token projection for the batched prefill of the MoE attention passes (F6): the weight is
 * dequantized to FP16 in row tiles and multiplied with the FP16-rounded activations of all the
 * chunk's tokens by one cuBLAS GEMM per tile (FP32 accumulation), so each layer's weights are read
 * once per chunk instead of once per token. The same scheme as {@code CudaForwardPass.prefillBatch}
 * (activations saturated to the FP16 range, as llama.cpp's cuBLAS prefill does), which is why the
 * generated text can differ slightly from the per-token path. Needs {@code libcublas.so}.
 */
final class GemmF16 {

    private final CudaContext ctx;
    private final MemorySegment stream;
    private final MemorySegment handle, alpha, beta0, beta1;
    private final long scratch16, in16;
    private final long tileBytes;
    private final MemorySegment f2hFunc;
    private final Map<GGMLType, MemorySegment> dequant = new EnumMap<>(GGMLType.class);
    private final KernelParams dqPB, f2hPB;

    /** Whether cuBLAS is available (the helper cannot be built otherwise). */
    static boolean available() {
        return CublasBindings.isAvailable() && !"false".equals(System.getProperty("prefill.gpu.gemm", "true"));
    }

    /** Dequant kernel name for {@code t}, or null when the type has none. */
    static String dequantKernel(GGMLType t) {
        switch (t) {
            case Q4_K: return "dequant_q4_k_f16t";
            case Q5_K: return "dequant_q5_k_f16t";
            case Q6_K: return "dequant_q6_k_f16t";
            case Q3_K: return "dequant_q3_k_f16t";
            case Q8_0: return "dequant_q8_0_f16t";
            case Q5_0: return "dequant_q5_0_f16t";
            case IQ4_NL: return "dequant_iq4_nl_f16t";
            case IQ4_XS: return "dequant_iq4_xs_f16t";
            case F16: return "dequant_f16_f16t";
            case BF16: return "dequant_bf16_f16t";
            case F32: return "dequant_f32_f16t";
            case IQ3_XXS: return "dequant_iq3_xxs_f16t";
            case IQ2_S: return "dequant_iq2_s_f16t";
            case IQ3_S: return "dequant_iq3_s_f16t";
            case IQ1_M: return "dequant_iq1_m_f16t";
            case IQ2_XXS: return "dequant_iq2_xxs_f16t";
            case Q2_K: return "dequant_q2_k_f16t";
            default: return null;
        }
    }

    private static String dequantResource(GGMLType t) {
        switch (t) {
            case IQ3_XXS: return "kernels/cuda/matmul_iq3_xxs.cu";
            case IQ2_S: return "kernels/cuda/matmul_iq2_s.cu";
            case IQ3_S: return "kernels/cuda/matmul_iq3_s.cu";
            case IQ1_M: return "kernels/cuda/matmul_iq1_m.cu";
            case IQ2_XXS: return "kernels/cuda/matmul_iq2_xxs.cu";
            case Q2_K: return "kernels/cuda/matmul_q2_k.cu";
            default: return "kernels/cuda/dequant_f16.cu";
        }
    }

    /**
     * @param maxN     largest token count
     * @param maxCols  largest input width of any projection
     * @param tensors  every weight the helper will be asked to multiply (their dequant kernels are
     *                 compiled now; construction fails when one has none)
     */
    GemmF16(CudaContext ctx, CudaBufferManager bm, Arena arena, int maxN, int maxCols, long tileBytes,
            Iterable<CudaFloatTensor> tensors) {
        this.ctx = ctx;
        this.stream = ctx.getStream();
        this.tileBytes = tileBytes;
        for (CudaFloatTensor t : tensors) {
            if (t == null || dequant.containsKey(t.type())) continue;
            String k = dequantKernel(t.type());
            if (k == null) throw new IllegalStateException("no FP16 dequant kernel for " + t.type());
            dequant.put(t.type(), ctx.compileKernel(dequantResource(t.type()), k));
        }
        this.f2hFunc = ctx.compileKernel("kernels/cuda/batch_ops.cu", "f32_to_f16_sat");
        this.scratch16 = bm.createBuffer(tileBytes);
        this.in16 = bm.createBuffer((long) maxN * maxCols * 2);
        this.handle = CublasBindings.create(arena);
        CublasBindings.setStream(handle, stream);
        this.alpha = arena.allocateFrom(ValueLayout.JAVA_FLOAT, 1.0f);
        this.beta0 = arena.allocateFrom(ValueLayout.JAVA_FLOAT, 0.0f);
        this.beta1 = arena.allocateFrom(ValueLayout.JAVA_FLOAT, 1.0f);
        this.dqPB = new KernelParams(arena, 5);
        this.f2hPB = new KernelParams(arena, 3);
    }

    /** Round {@code total} floats at {@code in32} to FP16 into the activation buffer. */
    void toF16(long in32, int total) {
        f2hPB.setLong(0, in32).setLong(1, in16).setInt(2, total);
        launch(f2hFunc, (total + 255) / 256, f2hPB);
    }

    /**
     * {@code out[t][0..rows) (+)= W[0..rows) . in16[t]} for the {@code n} tokens of the last
     * {@link #toF16} (row stride {@code cols}); the output rows are {@code ldc} floats apart.
     */
    void gemm(CudaFloatTensor w, int rows, int cols, long out32, int ldc, int n, boolean accumulate) {
        MemorySegment dq = dequant.get(w.type());
        long wBase = w.getGpuWeights();
        int tile = (int) Math.max(64, Math.min(rows, tileBytes / ((long) cols * 2)));
        tile = Math.max(1, tile / 64 * 64);
        for (int r0 = 0; r0 < rows; r0 += tile) {
            int tr = Math.min(tile, rows - r0);
            long total = (long) tr * cols;
            dqPB.setLong(0, wBase).setLong(1, scratch16).setInt(2, r0).setInt(3, tr).setInt(4, cols);
            launch(dq, (int) ((total + 255) / 256), dqPB);
            CublasBindings.gemmEx(handle, CublasBindings.CUBLAS_OP_T, CublasBindings.CUBLAS_OP_N,
                tr, n, cols, alpha,
                scratch16, CublasBindings.CUDA_R_16F, cols,
                in16, CublasBindings.CUDA_R_16F, cols,
                accumulate ? beta1 : beta0,
                out32 + (long) r0 * Float.BYTES, CublasBindings.CUDA_R_32F, ldc,
                CublasBindings.CUBLAS_COMPUTE_32F, CublasBindings.CUBLAS_GEMM_DEFAULT);
        }
    }

    private void launch(MemorySegment fn, int grid, KernelParams p) {
        int err = it.denzosoft.llmplayer.gpu.CudaBindings.launchKernel(fn, grid, 1, 1, 256, 1, 1, 0, stream,
            p.ptrs(), MemorySegment.NULL);
        if (err != it.denzosoft.llmplayer.gpu.CudaBindings.CUDA_SUCCESS) throw new RuntimeException("GEMM prefill CUDA error: " + err);
    }

    void close() {
        try { CublasBindings.destroy(handle); } catch (Exception ignored) { }
        try { ctx.freeBuffer(scratch16); } catch (Exception ignored) { }
        try { ctx.freeBuffer(in16); } catch (Exception ignored) { }
    }
}
