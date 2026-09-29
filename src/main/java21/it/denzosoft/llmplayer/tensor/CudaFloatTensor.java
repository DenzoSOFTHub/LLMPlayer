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

    /** Total weight bytes on GPU (element count / blockSize × blockBytes). */
    public long getWeightsBytes() {
        return (size / blockSize()) * blockBytes();
    }

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

    // CPU twin (SIMD when available) over the same weight data: used for dot() and for the CPU
    // fallback, both of which would otherwise run this class's scalar kernel.
    private volatile FloatTensor cpuTwin;
    private volatile boolean cpuTwinTried;
    // Set once the GPU path failed for this tensor; later calls go straight to the CPU twin
    // instead of retrying (and failing) the GPU on every token.
    private volatile boolean gpuFailed;
    private static final java.util.concurrent.atomic.AtomicBoolean FALLBACK_LOGGED =
        new java.util.concurrent.atomic.AtomicBoolean();

    /** The CPU (SIMD when available) variant of this tensor, or null if none exists. */
    protected FloatTensor cpuTwin() {
        if (cpuTwinTried) return cpuTwin;
        synchronized (this) {
            if (!cpuTwinTried) {
                try {
                    cpuTwin = TensorFactory.createCpu(type(), data, size);
                } catch (Exception e) {
                    cpuTwin = null;
                }
                cpuTwinTried = true;
            }
        }
        return cpuTwin;
    }

    /**
     * CPU dot product. Delegates to the CPU twin (SIMD) when one exists: MoE expert loops and
     * other CPU-side consumers call dot() on GPU-backed weights, and the scalar kernel in each
     * subclass ({@link #dotScalar}) is several times slower.
     */
    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        FloatTensor twin = cpuTwin();
        if (twin != null) return twin.dot(thisOffset, other, otherOffset, length);
        return dotScalar(thisOffset, other, otherOffset, length);
    }

    /** Scalar reference dot product (subclasses implement the dequantization inline). */
    protected float dotScalar(long thisOffset, float[] other, int otherOffset, int length) {
        return super.dot(thisOffset, other, otherOffset, length);
    }

    @Override
    public boolean isGpuResident() { return true; }

    /**
     * Smallest weight size sent through the per-tensor GPU path. Below it one upload, launch,
     * blocking download and accumulate cost more than the CPU twin's matmul (0.6-1.5 MB measured
     * at full clocks; at the idle clocks of an MoE decode nothing small wins on the GPU). The
     * GPU-resident passes read weights directly and are not affected.
     * {@code -Dgpu.tensor.min.bytes} (default 1 MiB; 0 sends everything to the GPU).
     */
    private static final long MIN_GPU_BYTES = Long.getLong("gpu.tensor.min.bytes", 1L << 20);

    @Override
    public void matmulParallelCpu(float[] input, float[] out, int rows, int cols) {
        FloatTensor twin = cpuTwin();
        if (twin != null) twin.matmulParallel(input, out, rows, cols);
        else super.matmulParallel(input, out, rows, cols);
    }

    /** CPU row kernels go to the SIMD twin (the default loops over this class's scalar dot). */
    @Override
    public void matmulRows(float[] input, float[] out, int rowFrom, int rowTo, int cols) {
        FloatTensor twin = cpuTwin();
        if (twin != null) twin.matmulRows(input, out, rowFrom, rowTo, cols);
        else super.matmulRows(input, out, rowFrom, rowTo, cols);
    }

    @Override
    public void matmulRowsBatch(float[][] in, float[][] out, int n, int rowFrom, int rowTo, int cols) {
        FloatTensor twin = cpuTwin();
        if (twin != null) twin.matmulRowsBatch(in, out, n, rowFrom, rowTo, cols);
        else super.matmulRowsBatch(in, out, n, rowFrom, rowTo, cols);
    }

    @Override
    public void matmulParallel(float[] input, float[] out, int rows, int cols) {
        if ((!FloatTensor.gpuMatmulEnabled() || MIN_GPU_BYTES > 0 && getWeightsBytes() < MIN_GPU_BYTES)
                && cpuTwin() != null) {
            cpuTwin().matmulParallel(input, out, rows, cols);
            return;
        }
        if (!gpuFailed) {
            try {
                gpuMatmul(input, out, rows, cols);
                return;
            } catch (Exception e) {
                gpuFailed = true;
                if (FALLBACK_LOGGED.compareAndSet(false, true)) {
                    System.err.println("CUDA: per-tensor matmul failed (" + type() + " " + rows + "x" + cols
                        + ": " + e.getMessage() + ") — this tensor now runs on the CPU");
                }
            }
        }
        FloatTensor twin = cpuTwin();
        if (twin != null) twin.matmulParallel(input, out, rows, cols);
        else super.matmulParallel(input, out, rows, cols);
    }

    /**
     * Execute matmul on CUDA GPU with accumulate semantics ({@code out[r] += W[r]·input}).
     * One input upload, a write-mode launch, one output download, then the add on the CPU: the
     * previous version also uploaded {@code out} so that the kernel could accumulate, costing a
     * second transfer. The staging block is page-locked, so both copies are direct DMA, and the
     * launch reuses a pre-built parameter block (no per-call Arena).
     */
    protected void gpuMatmul(float[] input, float[] out, int rows, int cols) {
        MemorySegment function = getFunction();
        long weightPtr = getGpuWeights();
        long inputBytes = (long) cols * Float.BYTES;
        long outputBytes = (long) rows * Float.BYTES;
        long outOffset = (inputBytes + 63) & ~63L;

        synchronized (bufferManager.perTensorLock()) {
            long inputPtr = bufferManager.getPooledInputBuffer(inputBytes);
            long outputPtr = bufferManager.getPooledOutputBuffer(outputBytes);
            MemorySegment staging = bufferManager.staging(outOffset + outputBytes);

            MemorySegment.copy(input, 0, staging, ValueLayout.JAVA_FLOAT, 0, cols);
            cudaContext.writeBufferAsync(inputPtr, staging, inputBytes);

            it.denzosoft.llmplayer.gpu.KernelParams p = bufferManager.perTensorParams();
            p.setLong(0, weightPtr).setLong(1, inputPtr).setLong(2, outputPtr)
             .setInt(3, rows).setInt(4, cols).setInt(5, 0);
            long blockSize = getMatmulBlockDim(cols);
            int gridDim = getMatmulGridDim(rows, cols);
            int smBytes = computeSharedMemBytes(cols, blockSize);
            cudaContext.launchKernel1D(function, (long) gridDim * blockSize, blockSize, smBytes, p.ptrs());

            MemorySegment outHost = staging.asSlice(outOffset, outputBytes);
            cudaContext.readBuffer(outputPtr, outHost, outputBytes); // waits for the launch
            // Bulk copy + vector add instead of a scalar loop of segment reads (150-200k rows for
            // an output projection)
            float[] tmp = bufferManager.accumulateScratch(rows);
            MemorySegment.copy(outHost, ValueLayout.JAVA_FLOAT, 0, tmp, 0, rows);
            it.denzosoft.llmplayer.tensor.VectorOpsFactory.get().accumulate(out, tmp, rows);
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
