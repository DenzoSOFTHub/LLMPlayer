package it.denzosoft.llmplayer.tensor;

import it.denzosoft.llmplayer.gpu.GpuBufferManager;
import it.denzosoft.llmplayer.gpu.OpenCLBindings;
import it.denzosoft.llmplayer.gpu.OpenCLContext;

import java.lang.foreign.*;

/**
 * Base class for GPU-accelerated tensors.
 * Provides matmulParallel() override that dispatches to GPU,
 * with automatic fallback to CPU on error.
 */
public abstract class GpuFloatTensor extends FloatTensor {

    protected final GpuBufferManager bufferManager;
    protected final OpenCLContext clContext;
    private volatile MemorySegment gpuWeights;

    protected GpuFloatTensor(TensorData data, long size, GpuBufferManager bufferManager) {
        super(data, size);
        this.bufferManager = bufferManager;
        this.clContext = bufferManager.getClContext();
    }

    /**
     * Return the kernel resource path (e.g. "kernels/matmul_f32.cl").
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
    protected MemorySegment getGpuWeights() {
        MemorySegment cached = gpuWeights;
        if (cached != null) return cached;
        synchronized (this) {
            cached = gpuWeights;
            if (cached != null) return cached;
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
     * Execute matmul on GPU. Subclasses can override for custom kernel arg setup.
     */
    protected void gpuMatmul(float[] input, float[] out, int rows, int cols) {
        MemorySegment kernel = clContext.compileKernel(kernelResourcePath(), kernelName());
        MemorySegment weightBuf = getGpuWeights();

        try (Arena tempArena = Arena.ofConfined()) {
            // Get pooled GPU buffers (reused across matmul calls, not created/destroyed each time)
            long inputBytes = (long) cols * Float.BYTES;
            long outputBytes = (long) rows * Float.BYTES;
            MemorySegment inputBuf = bufferManager.getPooledInputBuffer(inputBytes);
            MemorySegment outputBuf = bufferManager.getPooledOutputBuffer(outputBytes);

            // Upload input to GPU via writeBuffer
            MemorySegment inputHost = tempArena.allocate(ValueLayout.JAVA_FLOAT, cols);
            MemorySegment.copy(input, 0, inputHost, ValueLayout.JAVA_FLOAT, 0, cols);
            clContext.writeBuffer(inputBuf, inputHost, inputBytes);

            // Upload existing output values
            MemorySegment outputHost = tempArena.allocate(ValueLayout.JAVA_FLOAT, rows);
            MemorySegment.copy(out, 0, outputHost, ValueLayout.JAVA_FLOAT, 0, rows);
            clContext.writeBuffer(outputBuf, outputHost, outputBytes);

            setKernelArgs(kernel, weightBuf, inputBuf, outputBuf, rows, cols, tempArena);
            long globalWorkSize = ((rows + 63L) / 64) * 64;
            clContext.enqueueKernel1D(kernel, globalWorkSize, 64, tempArena);
            clContext.finish();

            clContext.readBuffer(outputBuf, outputHost, outputBytes);
            MemorySegment.copy(outputHost, ValueLayout.JAVA_FLOAT, 0, out, 0, rows);
        }
    }

    /**
     * Set kernel arguments. Default: weights, input, output, rows, cols.
     * Subclasses for quantized formats override to add block info.
     */
    protected void setKernelArgs(MemorySegment kernel, MemorySegment weightBuf,
                                  MemorySegment inputBuf, MemorySegment outputBuf,
                                  int rows, int cols, Arena tempArena) {
        clContext.setKernelArgMem(kernel, 0, weightBuf, tempArena);
        clContext.setKernelArgMem(kernel, 1, inputBuf, tempArena);
        clContext.setKernelArgMem(kernel, 2, outputBuf, tempArena);
        clContext.setKernelArgInt(kernel, 3, rows, tempArena);
        clContext.setKernelArgInt(kernel, 4, cols, tempArena);
    }
}
