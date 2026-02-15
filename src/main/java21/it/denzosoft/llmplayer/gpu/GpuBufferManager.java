package it.denzosoft.llmplayer.gpu;

import it.denzosoft.llmplayer.tensor.TensorData;

import java.lang.foreign.*;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import static it.denzosoft.llmplayer.gpu.OpenCLBindings.*;

/**
 * Manages GPU buffer caching for tensor weights.
 * Weights are uploaded once and reused across forward passes.
 * Thread-safe: uses ConcurrentHashMap for weight cache.
 */
public class GpuBufferManager implements AutoCloseable {

    private final OpenCLContext clContext;
    // Two-level cache: TensorData identity → (byteOffset → GPU cl_mem)
    // IdentityHashMap avoids hash collisions across distinct TensorData objects.
    private final Map<TensorData, Map<Long, MemorySegment>> weightCache = new IdentityHashMap<>();
    // Buffer pool: size → reusable GPU cl_mem (avoids create/destroy per matmul)
    private final Map<Long, MemorySegment> inputBufferPool = new ConcurrentHashMap<>();
    private final Map<Long, MemorySegment> outputBufferPool = new ConcurrentHashMap<>();
    private volatile boolean closed = false;

    public GpuBufferManager(OpenCLContext clContext) {
        this.clContext = clContext;
    }

    /**
     * Get or upload tensor weight data to GPU.
     * Uses TensorData identity + byteOffset as cache key.
     */
    public MemorySegment getOrUploadWeights(TensorData data, long byteOffset, long sizeBytes) {
        synchronized (this) {
            Map<Long, MemorySegment> offsetMap = weightCache.get(data);
            if (offsetMap != null) {
                MemorySegment cached = offsetMap.get(byteOffset);
                if (cached != null) return cached;
            }

            // Copy tensor data to a contiguous host buffer using a confined arena
            // that is freed immediately after GPU upload completes.
            MemorySegment gpuBuf;
            try (Arena stagingArena = Arena.ofConfined()) {
                MemorySegment hostBuf = stagingArena.allocate(sizeBytes);
                byte[] chunk = new byte[(int) Math.min(sizeBytes, 65536)];
                long remaining = sizeBytes;
                long offset = 0;
                while (remaining > 0) {
                    int toRead = (int) Math.min(remaining, chunk.length);
                    data.copyBytes(byteOffset + offset, chunk, 0, toRead);
                    MemorySegment.copy(chunk, 0, hostBuf, ValueLayout.JAVA_BYTE, offset, toRead);
                    offset += toRead;
                    remaining -= toRead;
                }

                // Upload to GPU (CL_MEM_COPY_HOST_PTR copies data, so hostBuf can be freed)
                gpuBuf = clContext.createGpuBuffer(sizeBytes, CL_MEM_READ_ONLY, hostBuf);
            }

            if (offsetMap == null) {
                offsetMap = new ConcurrentHashMap<>();
                weightCache.put(data, offsetMap);
            }
            offsetMap.put(byteOffset, gpuBuf);
            return gpuBuf;
        }
    }

    /**
     * Get a pooled input buffer of at least the given size.
     * The buffer is reused across matmul calls to avoid GPU alloc/free overhead.
     */
    public MemorySegment getPooledInputBuffer(long sizeBytes) {
        MemorySegment buf = inputBufferPool.get(sizeBytes);
        if (buf != null) return buf;
        buf = clContext.createGpuBuffer(sizeBytes, CL_MEM_READ_ONLY);
        inputBufferPool.put(sizeBytes, buf);
        return buf;
    }

    /**
     * Get a pooled output buffer of at least the given size.
     * The buffer is reused across matmul calls to avoid GPU alloc/free overhead.
     */
    public MemorySegment getPooledOutputBuffer(long sizeBytes) {
        MemorySegment buf = outputBufferPool.get(sizeBytes);
        if (buf != null) return buf;
        buf = clContext.createGpuBuffer(sizeBytes, CL_MEM_READ_WRITE);
        outputBufferPool.put(sizeBytes, buf);
        return buf;
    }

    /**
     * Create a temporary input buffer (written each call).
     */
    public MemorySegment createInputBuffer(long sizeBytes) {
        return clContext.createGpuBuffer(sizeBytes, CL_MEM_READ_ONLY);
    }

    /**
     * Create a temporary output buffer (read each call).
     */
    public MemorySegment createOutputBuffer(long sizeBytes) {
        return clContext.createGpuBuffer(sizeBytes, CL_MEM_READ_WRITE);
    }

    public OpenCLContext getClContext() { return clContext; }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        for (Map<Long, MemorySegment> offsetMap : weightCache.values()) {
            for (MemorySegment buf : offsetMap.values()) {
                try { releaseMemObject(buf); } catch (Exception ignored) {}
            }
        }
        weightCache.clear();
        for (MemorySegment buf : inputBufferPool.values()) {
            try { releaseMemObject(buf); } catch (Exception ignored) {}
        }
        inputBufferPool.clear();
        for (MemorySegment buf : outputBufferPool.values()) {
            try { releaseMemObject(buf); } catch (Exception ignored) {}
        }
        outputBufferPool.clear();
    }
}
