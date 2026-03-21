package it.denzosoft.llmplayer.gpu;

import it.denzosoft.llmplayer.tensor.CudaFloatTensor;
import it.denzosoft.llmplayer.tensor.GGMLType;

import java.lang.foreign.*;

/**
 * cuBLAS-accelerated matrix-vector multiply for quantized weights.
 * Pre-dequantizes Q4_K weights to FP32 at load time, then uses cublasSgemv for inference.
 *
 * Enabled via -Dcuda.cublas=true. Requires libcublas.so and enough VRAM for FP32 weights.
 * Best for small models (1-3B) where FP32 weights fit in VRAM alongside Q4_K.
 *
 * VRAM overhead: rows × cols × 4 bytes per dequantized tensor (7.1x the Q4_K size).
 */
public class CublasMatmul implements AutoCloseable {

    private final MemorySegment cublasHandle;
    private final CudaContext cudaContext;
    private final Arena arena;

    // Host-side alpha/beta constants
    private final MemorySegment alphaOne;   // float 1.0
    private final MemorySegment betaZero;   // float 0.0
    private final MemorySegment betaOne;    // float 1.0

    // FP16 input conversion buffer
    private long gpuInputF16;  // device pointer to FP16 input buffer
    private int inputF16Dim;   // current allocated dim

    public CublasMatmul(CudaContext cudaContext) {
        this.cudaContext = cudaContext;
        this.arena = Arena.ofShared();

        // Create cuBLAS handle
        cublasHandle = CublasBindings.create(arena);
        CublasBindings.setStream(cublasHandle, cudaContext.getStream());

        // Allocate host-side constants
        alphaOne = arena.allocate(ValueLayout.JAVA_FLOAT);
        alphaOne.set(ValueLayout.JAVA_FLOAT, 0, 1.0f);
        betaZero = arena.allocate(ValueLayout.JAVA_FLOAT);
        betaZero.set(ValueLayout.JAVA_FLOAT, 0, 0.0f);
        betaOne = arena.allocate(ValueLayout.JAVA_FLOAT);
        betaOne.set(ValueLayout.JAVA_FLOAT, 0, 1.0f);

        System.err.println("cuBLAS: initialized (handle=" + cublasHandle + ")");
    }

    /**
     * Pre-dequantize a Q4_K weight tensor to FP32 on GPU.
     * Returns the device pointer to the FP32 buffer. Caller owns the memory.
     *
     * @param tensor  Q4_K CUDA tensor
     * @param rows    number of output rows (matmul M dimension)
     * @param cols    number of input columns (matmul K dimension)
     * @return device pointer to FP32 [rows × cols] buffer, or 0 if not Q4_K
     */
    public long dequantizeToF32(CudaFloatTensor tensor, int rows, int cols,
                                 CudaBufferManager bufferManager, MemorySegment dequantFunc) {
        if (tensor.type() != GGMLType.Q4_K) return 0;

        long totalElements = (long) rows * cols;
        long f32Bytes = totalElements * Float.BYTES;
        long f32Ptr = bufferManager.createBuffer(f32Bytes);

        // Launch dequantization kernel
        int blockDim = 256;
        int gridDim = (int) ((totalElements + blockDim - 1) / blockDim);

        try (Arena temp = Arena.ofConfined()) {
            MemorySegment params = cudaContext.buildKernelParams(temp,
                tensor.getGpuWeights(), f32Ptr, rows, cols);
            cudaContext.launchKernel1D(dequantFunc, totalElements, blockDim, 0, params);
            cudaContext.finish(); // ensure dequant completes
        }

        return f32Ptr;
    }

    /**
     * FP32 matrix-vector multiply using cuBLAS sgemv.
     * Computes: output[row] = sum_col A[row][col] * input[col]  (or += if accumulate)
     *
     * cuBLAS uses column-major, so for our row-major A[rows×cols]:
     *   cublasSgemv(CUBLAS_OP_T, cols, rows, alpha, A, cols, x, 1, beta, y, 1)
     *   This transposes column-major A (treating row-major as transposed column-major).
     *
     * @param f32Weights  device pointer to FP32 [rows × cols] weight matrix
     * @param input       device pointer to FP32 [cols] input vector
     * @param output      device pointer to FP32 [rows] output vector
     * @param rows        number of output rows
     * @param cols        number of input columns
     * @param accumulate  if true, output += A*x; if false, output = A*x
     */
    public void sgemv(long f32Weights, long input, long output,
                       int rows, int cols, boolean accumulate) {
        // Row-major A[rows×cols] in memory = column-major A^T[cols×rows]
        // To compute y = A * x (row-major), use cublas: y = A^T_colmajor^T * x = cublasSgemv(OP_T, cols, rows, ...)
        CublasBindings.sgemv(cublasHandle, CublasBindings.CUBLAS_OP_T,
            cols, rows,             // m=cols, n=rows (of the column-major storage)
            alphaOne,               // alpha = 1.0
            f32Weights, cols,       // A, lda=cols
            input, 1,               // x, incx=1
            accumulate ? betaOne : betaZero,  // beta
            output, 1);            // y, incy=1
    }

    /**
     * Pre-dequantize a Q4_K weight tensor to FP16 on GPU.
     */
    public long dequantizeToF16(CudaFloatTensor tensor, int rows, int cols,
                                 CudaBufferManager bufferManager, MemorySegment dequantF16Func) {
        if (tensor.type() != GGMLType.Q4_K) return 0;
        long totalElements = (long) rows * cols;
        long f16Bytes = totalElements * 2;  // 2 bytes per FP16
        long f16Ptr = bufferManager.createBuffer(f16Bytes);
        int blockDim = 256;
        int gridDim = (int) ((totalElements + blockDim - 1) / blockDim);
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment params = cudaContext.buildKernelParams(temp,
                tensor.getGpuWeights(), f16Ptr, rows, cols);
            cudaContext.launchKernel1D(dequantF16Func, totalElements, blockDim, 0, params);
            cudaContext.finish();
        }
        return f16Ptr;
    }

    /**
     * Allocate FP16 input buffer for a given dimension.
     */
    public void ensureInputF16(int dim, CudaBufferManager bufferManager) {
        if (inputF16Dim < dim) {
            if (gpuInputF16 != 0) {
                // Can't free old buffer easily, just allocate new
            }
            gpuInputF16 = bufferManager.createBuffer((long) dim * 2);
            inputF16Dim = dim;
        }
    }

    /**
     * Mixed-precision gemv: output(FP32) = A(FP16) * x(FP32) + beta * output(FP32)
     * Converts input to FP16 on-the-fly, then uses cublasGemmEx for FP16×FP16→FP32.
     *
     * For row-major A[rows×cols]: use transa=OP_T, m=cols, n=1, k=rows.
     * Wait — gemv convention: y[m] = A[m×n] * x[n]. With OP_T on col-major A[n×m]:
     *   y = A^T * x, where A is stored as [cols × rows] (our row-major).
     *   m=rows (output dim), n=1 (vector), k=cols (inner dim).
     *   transa=OP_T: A_stored[cols×rows] transposed = [rows×cols].
     */
    public void gemmExF16(long f16Weights, long inputF32, long outputF32,
                           int rows, int cols, boolean accumulate,
                           MemorySegment convertF16Func, CudaContext ctx) {
        // Convert FP32 input to FP16 (tiny: cols floats)
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment params = ctx.buildKernelParams(temp, inputF32, gpuInputF16, cols);
            ctx.launchKernel1D(convertF16Func, cols, 256, 0, params);
        }

        // cublasGemmEx: C[m×n] = alpha * op(A)[m×k] * op(B)[k×n] + beta * C[m×n]
        // Our row-major weights A[rows×cols] = col-major A_stored[cols×rows]
        // With transa=OP_T: op(A) = A_stored^T = [rows×cols], so m=rows, k=cols
        // B is input vector [cols×1] (col-major = same as row-major for vector), n=1
        CublasBindings.gemmEx(cublasHandle,
            CublasBindings.CUBLAS_OP_T,  // transa: transpose stored matrix
            CublasBindings.CUBLAS_OP_N,  // transb: no transpose on input
            rows, 1, cols,               // m, n, k
            alphaOne,                    // alpha = 1.0f
            f16Weights, CublasBindings.CUDA_R_16F, cols,  // A (FP16), lda=cols
            gpuInputF16, CublasBindings.CUDA_R_16F, cols, // B (FP16), ldb=cols
            accumulate ? betaOne : betaZero,               // beta
            outputF32, CublasBindings.CUDA_R_32F, rows,   // C (FP32), ldc=rows
            CublasBindings.CUBLAS_COMPUTE_32F,
            CublasBindings.CUBLAS_GEMM_DEFAULT);
    }

    public long getInputF16() { return gpuInputF16; }

    public static boolean isAvailable() {
        return CublasBindings.isAvailable();
    }

    @Override
    public void close() {
        CublasBindings.destroy(cublasHandle);
        arena.close();
    }
}
