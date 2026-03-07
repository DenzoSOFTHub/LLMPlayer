package it.denzosoft.llmplayer.gpu;

import it.denzosoft.llmplayer.tensor.MemorySegmentTensorData;
import it.denzosoft.llmplayer.tensor.TensorData;

import java.lang.foreign.*;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Manages CUDA GPU buffer caching for tensor weights.
 * Weights are uploaded once and reused across forward passes.
 * Uses long device pointers (CUdeviceptr) instead of MemorySegment cl_mem.
 *
 * Supports three memory modes:
 * - DEVICE (default): standard cuMemAlloc + cuMemcpyHtoD
 * - MANAGED: cuMemAllocManaged (unified memory, automatic page migration)
 * - HOST_MAPPED: cuMemHostAlloc (zero-copy, GPU reads from host RAM via PCIe)
 */
public class CudaBufferManager implements AutoCloseable {

    public enum MemoryMode { DEVICE, MANAGED, HOST_MAPPED }

    private final CudaContext cudaContext;
    private final MemoryMode memoryMode;
    // Two-level cache: TensorData identity → (byteOffset → GPU CUdeviceptr)
    private final Map<TensorData, Map<Long, Long>> weightCache = new IdentityHashMap<>();
    // Buffer pool: size → reusable GPU CUdeviceptr
    private final Map<Long, Long> inputBufferPool = new ConcurrentHashMap<>();
    private final Map<Long, Long> outputBufferPool = new ConcurrentHashMap<>();
    // Track host pointers for HOST_MAPPED mode (need to free with cuMemFreeHost)
    private final List<Long> hostMappedPointers = new ArrayList<>();
    private volatile boolean closed = false;

    public CudaBufferManager(CudaContext cudaContext) {
        this(cudaContext, MemoryMode.DEVICE);
    }

    public CudaBufferManager(CudaContext cudaContext, MemoryMode memoryMode) {
        this.cudaContext = cudaContext;
        this.memoryMode = memoryMode;
    }

    public MemoryMode getMemoryMode() { return memoryMode; }

    /**
     * Get or upload tensor weight data to GPU.
     * Uses TensorData identity + byteOffset as cache key.
     */
    public long getOrUploadWeights(TensorData data, long byteOffset, long sizeBytes) {
        synchronized (this) {
            Map<Long, Long> offsetMap = weightCache.get(data);
            if (offsetMap != null) {
                Long cached = offsetMap.get(byteOffset);
                if (cached != null) return cached;
            }

            long gpuPtr;
            if (memoryMode == MemoryMode.MANAGED) {
                // Unified memory: allocate managed, then write data (driver handles placement)
                gpuPtr = cudaContext.allocManagedBuffer(sizeBytes);
                copyDataToPtr(data, byteOffset, sizeBytes, gpuPtr);
            } else if (memoryMode == MemoryMode.HOST_MAPPED) {
                // Zero-copy: allocate pinned host memory, get device pointer
                long[] ptrs = cudaContext.allocHostMappedBuffer(sizeBytes);
                long hostPtr = ptrs[0];
                gpuPtr = ptrs[1];
                hostMappedPointers.add(hostPtr);
                // Write data directly to host-mapped memory (GPU reads via PCIe)
                MemorySegment hostSeg = MemorySegment.ofAddress(hostPtr).reinterpret(sizeBytes);
                copyDataToSegment(data, byteOffset, sizeBytes, hostSeg);
            } else {
                // Standard device memory: allocate + copy
                gpuPtr = cudaContext.allocBuffer(sizeBytes);
                copyDataToPtr(data, byteOffset, sizeBytes, gpuPtr);
            }

            if (offsetMap == null) {
                offsetMap = new ConcurrentHashMap<>();
                weightCache.put(data, offsetMap);
            }
            offsetMap.put(byteOffset, gpuPtr);
            return gpuPtr;
        }
    }

    private void copyDataToPtr(TensorData data, long byteOffset, long sizeBytes, long gpuPtr) {
        if (data instanceof MemorySegmentTensorData) {
            MemorySegment seg = ((MemorySegmentTensorData) data).segment();
            MemorySegment slice = seg.asSlice(byteOffset, sizeBytes);
            cudaContext.writeBuffer(gpuPtr, slice, sizeBytes);
        } else {
            try (Arena stagingArena = Arena.ofConfined()) {
                MemorySegment hostBuf = stagingArena.allocate(sizeBytes);
                copyDataToSegment(data, byteOffset, sizeBytes, hostBuf);
                cudaContext.writeBuffer(gpuPtr, hostBuf, sizeBytes);
            }
        }
    }

    private void copyDataToSegment(TensorData data, long byteOffset, long sizeBytes, MemorySegment dest) {
        if (data instanceof MemorySegmentTensorData) {
            MemorySegment seg = ((MemorySegmentTensorData) data).segment();
            MemorySegment.copy(seg, byteOffset, dest, 0, sizeBytes);
        } else {
            byte[] chunk = new byte[(int) Math.min(sizeBytes, 65536)];
            long remaining = sizeBytes;
            long offset = 0;
            while (remaining > 0) {
                int toRead = (int) Math.min(remaining, chunk.length);
                data.copyBytes(byteOffset + offset, chunk, 0, toRead);
                MemorySegment.copy(chunk, 0, dest, ValueLayout.JAVA_BYTE, offset, toRead);
                offset += toRead;
                remaining -= toRead;
            }
        }
    }

    /**
     * Get a pooled input buffer of at least the given size.
     */
    public long getPooledInputBuffer(long sizeBytes) {
        Long buf = inputBufferPool.get(sizeBytes);
        if (buf != null) return buf;
        long ptr = cudaContext.allocBuffer(sizeBytes);
        inputBufferPool.put(sizeBytes, ptr);
        return ptr;
    }

    /**
     * Get a pooled output buffer of at least the given size.
     */
    public long getPooledOutputBuffer(long sizeBytes) {
        Long buf = outputBufferPool.get(sizeBytes);
        if (buf != null) return buf;
        long ptr = cudaContext.allocBuffer(sizeBytes);
        outputBufferPool.put(sizeBytes, ptr);
        return ptr;
    }

    /**
     * Create a new GPU buffer (always device memory for intermediates).
     */
    public long createBuffer(long sizeBytes) {
        return cudaContext.allocBuffer(sizeBytes);
    }

    /**
     * Allocate a weight buffer using the configured memory mode.
     */
    public long allocWeightBuffer(long sizeBytes) {
        if (memoryMode == MemoryMode.MANAGED) {
            return cudaContext.allocManagedBuffer(sizeBytes);
        } else if (memoryMode == MemoryMode.HOST_MAPPED) {
            long[] ptrs = cudaContext.allocHostMappedBuffer(sizeBytes);
            hostMappedPointers.add(ptrs[0]);
            return ptrs[1]; // return device-accessible pointer
        } else {
            return cudaContext.allocBuffer(sizeBytes);
        }
    }

    /**
     * Upload RMS norm weights (float array) to a persistent GPU buffer.
     */
    public long uploadNormWeights(float[] weights) {
        try (Arena staging = Arena.ofConfined()) {
            long sizeBytes = (long) weights.length * Float.BYTES;
            MemorySegment hostBuf = staging.allocate(ValueLayout.JAVA_FLOAT, weights.length);
            MemorySegment.copy(weights, 0, hostBuf, ValueLayout.JAVA_FLOAT, 0, weights.length);
            long gpuPtr = cudaContext.allocBuffer(sizeBytes);
            cudaContext.writeBuffer(gpuPtr, hostBuf, sizeBytes);
            return gpuPtr;
        }
    }

    public CudaContext getCudaContext() { return cudaContext; }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        if (memoryMode == MemoryMode.HOST_MAPPED) {
            // Free host-mapped memory via cuMemFreeHost
            for (long hostPtr : hostMappedPointers) {
                try { cudaContext.freeHostMappedBuffer(hostPtr); } catch (Exception ignored) {}
            }
            hostMappedPointers.clear();
        } else {
            // Free device or managed memory via cuMemFree
            for (Map<Long, Long> offsetMap : weightCache.values()) {
                for (long ptr : offsetMap.values()) {
                    try { cudaContext.freeBuffer(ptr); } catch (Exception ignored) {}
                }
            }
        }
        weightCache.clear();
        for (long ptr : inputBufferPool.values()) {
            try { cudaContext.freeBuffer(ptr); } catch (Exception ignored) {}
        }
        inputBufferPool.clear();
        for (long ptr : outputBufferPool.values()) {
            try { cudaContext.freeBuffer(ptr); } catch (Exception ignored) {}
        }
        outputBufferPool.clear();
    }
}
