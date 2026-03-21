package it.denzosoft.llmplayer.gpu;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

/**
 * Panama FFM bindings for libcublas.so (cuBLAS v2 API).
 * Zero external dependencies — loads the system-installed cuBLAS library directly.
 * Provides FP16/FP32 matrix-vector multiply via cublasGemmEx for mixed-precision inference.
 */
public class CublasBindings {

    public static final int CUBLAS_STATUS_SUCCESS = 0;
    public static final int CUBLAS_OP_N = 0;  // no transpose
    public static final int CUBLAS_OP_T = 1;  // transpose

    // CUDA data types for cublasGemmEx
    public static final int CUDA_R_16F = 2;   // FP16
    public static final int CUDA_R_32F = 0;   // FP32

    // Compute types
    public static final int CUBLAS_COMPUTE_32F = 68;  // FP32 compute

    // Gemm algorithm
    public static final int CUBLAS_GEMM_DEFAULT = -1;

    private static final SymbolLookup CUBLAS_LIB;
    private static final Linker LINKER = Linker.nativeLinker();

    static {
        SymbolLookup lib = null;
        String[] names = {"libcublas.so.12", "libcublas.so.11", "libcublas.so",
                          "/usr/local/cuda/lib64/libcublas.so",
                          "/usr/lib/x86_64-linux-gnu/libcublas.so"};
        for (String name : names) {
            try {
                lib = SymbolLookup.libraryLookup(name, Arena.global());
                break;
            } catch (Exception ignored) {}
        }
        CUBLAS_LIB = lib;
    }

    public static boolean isAvailable() {
        return CUBLAS_LIB != null;
    }

    // --- Function handles ---

    private static MethodHandle find(String name, FunctionDescriptor desc) {
        return CUBLAS_LIB.find(name)
            .map(addr -> LINKER.downcallHandle(addr, desc))
            .orElseThrow(() -> new UnsupportedOperationException("cuBLAS function not found: " + name));
    }

    // cublasCreate_v2(cublasHandle_t *handle) -> int
    private static final MethodHandle cublasCreate = find("cublasCreate_v2",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

    // cublasDestroy_v2(cublasHandle_t handle) -> int
    private static final MethodHandle cublasDestroy = find("cublasDestroy_v2",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

    // cublasSetStream_v2(cublasHandle_t handle, cudaStream_t stream) -> int
    private static final MethodHandle cublasSetStream = find("cublasSetStream_v2",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    // cublasSgemv_v2(handle, trans, m, n, alpha*, A, lda, x, incx, beta*, y, incy) -> int
    private static final MethodHandle cublasSgemv = find("cublasSgemv_v2",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,  // handle
            ValueLayout.JAVA_INT, // trans
            ValueLayout.JAVA_INT, // m
            ValueLayout.JAVA_INT, // n
            ValueLayout.ADDRESS,  // alpha (device/host pointer to float)
            ValueLayout.JAVA_LONG, // A (device pointer)
            ValueLayout.JAVA_INT, // lda
            ValueLayout.JAVA_LONG, // x (device pointer)
            ValueLayout.JAVA_INT, // incx
            ValueLayout.ADDRESS,  // beta (device/host pointer to float)
            ValueLayout.JAVA_LONG, // y (device pointer)
            ValueLayout.JAVA_INT  // incy
        ));

    // cublasGemmEx(handle, transa, transb, m, n, k, alpha*, A, Atype, lda, B, Btype, ldb,
    //              beta*, C, Ctype, ldc, computeType, algo) -> int
    private static final MethodHandle cublasGemmEx = find("cublasGemmEx",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,   // handle
            ValueLayout.JAVA_INT,  // transa
            ValueLayout.JAVA_INT,  // transb
            ValueLayout.JAVA_INT,  // m
            ValueLayout.JAVA_INT,  // n
            ValueLayout.JAVA_INT,  // k
            ValueLayout.ADDRESS,   // alpha (host pointer)
            ValueLayout.JAVA_LONG, // A (device pointer)
            ValueLayout.JAVA_INT,  // Atype
            ValueLayout.JAVA_INT,  // lda
            ValueLayout.JAVA_LONG, // B (device pointer)
            ValueLayout.JAVA_INT,  // Btype
            ValueLayout.JAVA_INT,  // ldb
            ValueLayout.ADDRESS,   // beta (host pointer)
            ValueLayout.JAVA_LONG, // C (device pointer)
            ValueLayout.JAVA_INT,  // Ctype
            ValueLayout.JAVA_INT,  // ldc
            ValueLayout.JAVA_INT,  // computeType
            ValueLayout.JAVA_INT   // algo
        ));

    // --- Public API ---

    public static MemorySegment create(Arena arena) {
        MemorySegment handleBuf = arena.allocate(ValueLayout.ADDRESS);
        int err;
        try { err = (int) cublasCreate.invokeExact(handleBuf); }
        catch (Throwable t) { throw new RuntimeException("cublasCreate failed", t); }
        if (err != CUBLAS_STATUS_SUCCESS) throw new RuntimeException("cublasCreate error: " + err);
        return handleBuf.get(ValueLayout.ADDRESS, 0);
    }

    public static void destroy(MemorySegment handle) {
        try { cublasDestroy.invokeExact(handle); }
        catch (Throwable t) { throw new RuntimeException("cublasDestroy failed", t); }
    }

    public static void setStream(MemorySegment handle, MemorySegment stream) {
        int err;
        try { err = (int) cublasSetStream.invokeExact(handle, stream); }
        catch (Throwable t) { throw new RuntimeException("cublasSetStream failed", t); }
        if (err != CUBLAS_STATUS_SUCCESS) throw new RuntimeException("cublasSetStream error: " + err);
    }

    /**
     * FP32 matrix-vector multiply: y = alpha * A * x + beta * y
     * A is column-major [m × n], x is [n], y is [m].
     * For row-major weights [rows × cols]: use trans=CUBLAS_OP_T, m=cols, n=rows.
     */
    public static void sgemv(MemorySegment handle, int trans, int m, int n,
                              MemorySegment alpha, long A, int lda,
                              long x, int incx,
                              MemorySegment beta, long y, int incy) {
        int err;
        try { err = (int) cublasSgemv.invokeExact(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy); }
        catch (Throwable t) { throw new RuntimeException("cublasSgemv failed", t); }
        if (err != CUBLAS_STATUS_SUCCESS) throw new RuntimeException("cublasSgemv error: " + err);
    }

    /**
     * Mixed-precision GEMM: C = alpha * op(A) * op(B) + beta * C
     * Supports FP16 inputs with FP32 compute and FP32 output.
     * For matrix-vector: set n=1.
     */
    public static void gemmEx(MemorySegment handle, int transa, int transb,
                               int m, int n, int k,
                               MemorySegment alpha,
                               long A, int Atype, int lda,
                               long B, int Btype, int ldb,
                               MemorySegment beta,
                               long C, int Ctype, int ldc,
                               int computeType, int algo) {
        int err;
        try {
            err = (int) cublasGemmEx.invokeExact(handle, transa, transb, m, n, k,
                alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
        } catch (Throwable t) { throw new RuntimeException("cublasGemmEx failed", t); }
        if (err != CUBLAS_STATUS_SUCCESS) throw new RuntimeException("cublasGemmEx error: " + err);
    }
}
