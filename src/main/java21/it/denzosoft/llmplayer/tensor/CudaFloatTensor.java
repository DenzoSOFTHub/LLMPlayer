package it.denzosoft.llmplayer.tensor;

import it.denzosoft.llmplayer.gpu.CudaBufferManager;
import it.denzosoft.llmplayer.gpu.CudaContext;

import java.lang.foreign.*;

/**
 * Base class for CUDA GPU-accelerated tensors.
 * Provides matmulParallel() override that dispatches to CUDA GPU,
 * with automatic fallback to CPU on error.
 * Mirrors GpuFloatTensor but uses long device pointers + CudaContext.
 */
public abstract class CudaFloatTensor extends FloatTensor {

    protected final CudaBufferManager bufferManager;
    protected final CudaContext cudaContext;
    private volatile long gpuWeights; // CUdeviceptr, 0 = not uploaded
    private volatile MemorySegment cachedFunction; // compiled kernel function

    protected CudaFloatTensor(TensorData data, long size, CudaBufferManager bufferManager) {
        super(data, size);
        this.bufferManager = bufferManager;
        this.cudaContext = bufferManager.getCudaContext();
    }

    /**
     * Return the CUDA kernel resource path (e.g. "kernels/cuda/matmul_f32.cu").
     */
    protected abstract String kernelResourcePath();

    /**
     * Return the kernel function name (e.g. "matmul_f32").
     */
    protected abstract String kernelName();

    /**
     * Return the number of raw bytes per block in the quantized format.
     */
    protected abstract int blockBytes();

    /**
     * Return the number of float elements per block.
     */
    protected abstract int blockSize();

    /**
     * Get or lazily upload the weight data to GPU.
     */
    public long getGpuWeights() {
        long cached = gpuWeights;
        if (cached != 0) return cached;
        synchronized (this) {
            cached = gpuWeights;
            if (cached != 0) return cached;
            long totalBytes = (size / blockSize()) * blockBytes();
            cached = bufferManager.getOrUploadWeights(data, 0, totalBytes);
            gpuWeights = cached;
            return cached;
        }
    }

    @Override
    public void matmulParallel(float[] input, float[] out, int rows, int cols) {
        try {
            gpuMatmul(input, out, rows, cols);
        } catch (Exception e) {
            // Fallback to CPU
            super.matmulParallel(input, out, rows, cols);
        }
    }

    /**
     * Execute matmul on CUDA GPU (accumulate mode for backward compat).
     */
    protected void gpuMatmul(float[] input, float[] out, int rows, int cols) {
        MemorySegment function = getFunction();
        long weightPtr = getGpuWeights();

        try (Arena tempArena = Arena.ofConfined()) {
            long inputBytes = (long) cols * Float.BYTES;
            long outputBytes = (long) rows * Float.BYTES;
            long inputPtr = bufferManager.getPooledInputBuffer(inputBytes);
            long outputPtr = bufferManager.getPooledOutputBuffer(outputBytes);

            // Upload input
            MemorySegment inputHost = tempArena.allocate(ValueLayout.JAVA_FLOAT, cols);
            MemorySegment.copy(input, 0, inputHost, ValueLayout.JAVA_FLOAT, 0, cols);
            cudaContext.writeBuffer(inputPtr, inputHost, inputBytes);

            // Upload existing output
            MemorySegment outputHost = tempArena.allocate(ValueLayout.JAVA_FLOAT, rows);
            MemorySegment.copy(out, 0, outputHost, ValueLayout.JAVA_FLOAT, 0, rows);
            cudaContext.writeBuffer(outputPtr, outputHost, outputBytes);

            // Build kernel params and launch
            MemorySegment params = buildKernelParams(tempArena, weightPtr, inputPtr, outputPtr, rows, cols, 1);
            long blockSize = getMatmulBlockDim(cols);
            int gridDim = getMatmulGridDim(rows, cols);
            long globalSize = (long) gridDim * blockSize;
            int smBytes = computeSharedMemBytes(cols, blockSize);
            cudaContext.launchKernel1D(function, globalSize, blockSize, smBytes, params);
            cudaContext.finish();

            // Read back
            cudaContext.readBuffer(outputPtr, outputHost, outputBytes);
            MemorySegment.copy(outputHost, ValueLayout.JAVA_FLOAT, 0, out, 0, rows);
        }
    }

    /**
     * Buffer-to-buffer GPU matmul with accumulate mode (output[row] += sum).
     * Does NOT synchronize — the result stays GPU-resident.
     * Used by CudaForwardPass for residual accumulation (Wo, Down projections).
     */
    public void gpuMatmulBuffered(long gpuInput, long gpuOutput, int rows, int cols, Arena tempArena) {
        gpuMatmulBuffered(gpuInput, gpuOutput, rows, cols, tempArena, true);
    }

    /**
     * Buffer-to-buffer GPU matmul with write mode (output[row] = sum).
     * Does NOT synchronize — the result stays GPU-resident.
     * Used by CudaForwardPass for fresh outputs (Q, K, V, Gate, Up, logits).
     */
    public void gpuMatmulBufferedWrite(long gpuInput, long gpuOutput, int rows, int cols, Arena tempArena) {
        gpuMatmulBuffered(gpuInput, gpuOutput, rows, cols, tempArena, false);
    }

    private void gpuMatmulBuffered(long gpuInput, long gpuOutput, int rows, int cols, Arena tempArena, boolean addToOutput) {
        gpuMatmulOnStream(gpuInput, gpuOutput, rows, cols, tempArena, addToOutput, null);
    }

    /**
     * Buffer-to-buffer GPU matmul on a specific stream (write mode).
     */
    public void gpuMatmulBufferedWriteOnStream(long gpuInput, long gpuOutput, int rows, int cols,
                                                 Arena tempArena, MemorySegment onStream) {
        gpuMatmulOnStream(gpuInput, gpuOutput, rows, cols, tempArena, false, onStream);
    }

    /**
     * Buffer-to-buffer GPU matmul on a specific stream (accumulate mode).
     */
    public void gpuMatmulBufferedOnStream(long gpuInput, long gpuOutput, int rows, int cols,
                                            Arena tempArena, MemorySegment onStream) {
        gpuMatmulOnStream(gpuInput, gpuOutput, rows, cols, tempArena, true, onStream);
    }

    private MemorySegment getFunction() {
        MemorySegment f = cachedFunction;
        if (f != null) return f;
        f = cudaContext.compileKernel(kernelResourcePath(), kernelName());
        cachedFunction = f;
        return f;
    }

    private void gpuMatmulOnStream(long gpuInput, long gpuOutput, int rows, int cols,
                                     Arena tempArena, boolean addToOutput, MemorySegment onStream) {
        MemorySegment function = getFunction();
        long weightPtr = getGpuWeights();

        MemorySegment params = buildKernelParams(tempArena, weightPtr, gpuInput, gpuOutput, rows, cols, addToOutput ? 1 : 0);
        long blockSize = getMatmulBlockDim(cols);
        // gridDim count via override (allows multi-warp-per-row kernels like Q4_K 2warp)
        int gridDim = getMatmulGridDim(rows, cols);
        long globalSize = (long) gridDim * blockSize;
        int smBytes = computeSharedMemBytes(cols, blockSize);
        if (onStream != null) {
            cudaContext.launchKernel1DOnStream(function, globalSize, blockSize, smBytes, params, onStream);
        } else {
            cudaContext.launchKernel1D(function, globalSize, blockSize, smBytes, params);
        }
    }

    /**
     * Compute the effective CUDA block size, possibly reduced to fit shared memory limits.
     * Default returns the max block size. Subclasses override for shared-memory kernels.
     */
    protected long computeEffectiveCudaBlockSize(int cols, long maxBlockSize) {
        return maxBlockSize;
    }

    /**
     * Compute dynamic shared memory bytes for the kernel launch.
     * Default is 0 (no shared memory). Subclasses override for shared-memory kernels.
     */
    protected int computeSharedMemBytes(int cols, long cudaBlockSize) {
        return 0;
    }

    /**
     * Build kernel params: weights, input, output, rows, cols, addToOutput.
     */
    protected MemorySegment buildKernelParams(Arena tempArena, long weightPtr, long inputPtr,
                                               long outputPtr, int rows, int cols, int addToOutput) {
        return cudaContext.buildKernelParams(tempArena, weightPtr, inputPtr, outputPtr, rows, cols, addToOutput);
    }

    // === Public accessors for CudaForwardPass pre-allocated params ===

    /**
     * Get the compiled CUDA kernel function for matmul. Triggers lazy compilation.
     */
    public MemorySegment getCudaFunction() {
        return getFunction();
    }

    /**
     * Pre-compute matmul CUDA block dim for given cols.
     */
    public int getMatmulBlockDim(int cols) {
        long maxBlockSize = Math.min(256, cudaContext.getDeviceInfo().maxWorkGroupSize());
        return (int) computeEffectiveCudaBlockSize(cols, maxBlockSize);
    }

    /**
     * Pre-compute matmul grid dim for given rows and cols.
     */
    public int getMatmulGridDim(int rows, int cols) {
        int blockDim = getMatmulBlockDim(cols);
        long rowsPerBlock = blockDim / 32;
        return (int) ((rows + rowsPerBlock - 1) / rowsPerBlock);
    }

    /**
     * Pre-compute matmul shared memory bytes for given cols.
     */
    public int getMatmulSharedMem(int cols) {
        int blockDim = getMatmulBlockDim(cols);
        return computeSharedMemBytes(cols, blockDim);
    }
}
