package it.denzosoft.llmplayer.gpu;

import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.TensorData;

import java.lang.foreign.*;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * LRU GPU cache for MoE expert weight slices with batched kernel execution.
 *
 * Key optimization: all gate+up kernel launches are batched (no sync between them),
 * followed by a single cuStreamSynchronize + bulk download. Same for down phase.
 * This reduces sync overhead from 12 per MoE layer to just 2.
 *
 * Each expert gets its own GPU output buffer to allow concurrent in-flight kernels.
 * Input is shared (read-only) across all experts in the same phase.
 *
 * Designed for MXFP4-quantized expert weights (17 bytes per 32-element block).
 */
public class ExpertGpuCache {

    private static final int BLOCK_SIZE = 32;
    private static final int BLOCK_BYTES = 17;
    private static final int MAX_EXPERTS_PER_TOKEN = 8; // Max top-K (GPT-OSS uses 4)

    private final CudaContext cudaContext;
    private final int maxSlots;
    private final long sliceBytes;

    // Weight cache pool
    private final long[] gpuBuffers;
    private final long[] slotKeys;
    private final long[] accessTimes;
    private long accessCounter;
    private final Map<Long, Integer> keyToSlot = new HashMap<>();

    // Compiled kernel
    private final MemorySegment mxfp4Kernel;
    private final long cudaBlockSize;

    // Shared GPU input buffer (read-only during kernel execution)
    private long gpuInputBuf;
    private int inputBufDim;

    // Per-expert GPU output buffers (each expert writes to its own buffer)
    // Two sets: one for gate/up phase (expertFfnDim), one for down phase (dim)
    private final long[] gpuGateOutBufs = new long[MAX_EXPERTS_PER_TOKEN];
    private final long[] gpuUpOutBufs = new long[MAX_EXPERTS_PER_TOKEN];
    private final long[] gpuDownOutBufs = new long[MAX_EXPERTS_PER_TOKEN];
    private int gateBufDim;
    private int downBufDim;

    // Stats
    private long hits;
    private long misses;

    public ExpertGpuCache(CudaContext cudaContext, int maxSlots, long expertElements) {
        this.cudaContext = cudaContext;
        this.maxSlots = maxSlots;
        this.sliceBytes = (expertElements / BLOCK_SIZE) * BLOCK_BYTES;
        this.cudaBlockSize = Math.min(256, cudaContext.getDeviceInfo().maxWorkGroupSize());

        this.mxfp4Kernel = cudaContext.compileKernel("kernels/cuda/matmul_mxfp4.cu", "matmul_mxfp4");

        this.gpuBuffers = new long[maxSlots];
        this.slotKeys = new long[maxSlots];
        this.accessTimes = new long[maxSlots];
        Arrays.fill(slotKeys, -1L);

        for (int i = 0; i < maxSlots; i++) {
            gpuBuffers[i] = cudaContext.allocBuffer(sliceBytes);
        }

        System.out.println("  Expert GPU cache: " + maxSlots + " slots, " +
            (maxSlots * sliceBytes / 1024 / 1024) + " MB VRAM");
    }

    /**
     * Batch expert computation with minimal GPU sync points.
     *
     * Phase 1: Upload input → launch all gate+up kernels (no sync between them) → single sync → bulk download
     * Phase 2: CPU activation (SiLU/SwigluOAI + elementwise mul)
     * Phase 3: Upload activated → launch all down kernels → single sync → bulk download
     *
     * Total: 2 cuStreamSynchronize calls per MoE layer (was 12 in naive implementation).
     */
    public synchronized void computeExperts(
            FloatTensor gateExps, FloatTensor upExps, FloatTensor downExps,
            float[] input, int[] selectedExperts, float[] selectedWeights,
            int expertUsedCount, int layer, int dim, int expertFfnDim,
            float[][] gatePerExpert, float[][] upPerExpert, float[][] outPerExpert,
            boolean useSwigluOai,
            FloatTensor gateExpsBias, FloatTensor upExpsBias, FloatTensor downExpsBias) {

        long gateUpElements = (long) expertFfnDim * dim;
        long downElements = (long) dim * expertFfnDim;

        ensureInputBuffer(Math.max(dim, expertFfnDim));
        ensurePerExpertBuffers(expertUsedCount, expertFfnDim, dim);

        try (Arena temp = Arena.ofConfined()) {
            long ffnBytes = (long) expertFfnDim * Float.BYTES;
            long dimBytes = (long) dim * Float.BYTES;
            MemorySegment hostBuf = temp.allocate(ValueLayout.JAVA_FLOAT, Math.max(dim, expertFfnDim));

            // === Phase 1: Gate + Up matmuls (batched, 2 sync points) ===

            // Upload input once
            MemorySegment.copy(input, 0, hostBuf, ValueLayout.JAVA_FLOAT, 0, dim);
            cudaContext.writeBuffer(gpuInputBuf, hostBuf, (long) dim * Float.BYTES);

            // Launch all gate+up kernels without sync
            for (int k = 0; k < expertUsedCount; k++) {
                int e = selectedExperts[k];

                // Zero output buffers on GPU
                cudaContext.fillBufferZero(gpuGateOutBufs[k], ffnBytes);
                cudaContext.fillBufferZero(gpuUpOutBufs[k], ffnBytes);

                // Gate kernel
                long gatePtr = getOrUploadSlice(gateExps, e, gateUpElements, layer, 0);
                launchKernelNoSync(temp, gatePtr, gpuInputBuf, gpuGateOutBufs[k], expertFfnDim, dim);

                // Up kernel
                long upPtr = getOrUploadSlice(upExps, e, gateUpElements, layer, 1);
                launchKernelNoSync(temp, upPtr, gpuInputBuf, gpuUpOutBufs[k], expertFfnDim, dim);
            }

            // Single sync for all gate+up kernels
            cudaContext.finish();

            // Bulk download gate+up results
            for (int k = 0; k < expertUsedCount; k++) {
                MemorySegment.copy(gatePerExpert[k], 0, hostBuf, ValueLayout.JAVA_FLOAT, 0, expertFfnDim);
                cudaContext.readBuffer(gpuGateOutBufs[k], hostBuf, ffnBytes);
                MemorySegment.copy(hostBuf, ValueLayout.JAVA_FLOAT, 0, gatePerExpert[k], 0, expertFfnDim);

                cudaContext.readBuffer(gpuUpOutBufs[k], hostBuf, ffnBytes);
                MemorySegment.copy(hostBuf, ValueLayout.JAVA_FLOAT, 0, upPerExpert[k], 0, expertFfnDim);

                // Apply biases (GPT-OSS)
                int e = selectedExperts[k];
                if (gateExpsBias != null) addExpertBias(gatePerExpert[k], gateExpsBias, e, expertFfnDim);
                if (upExpsBias != null) addExpertBias(upPerExpert[k], upExpsBias, e, expertFfnDim);
            }

            // === Phase 2: CPU activation ===
            for (int k = 0; k < expertUsedCount; k++) {
                if (useSwigluOai) {
                    swigluOai(gatePerExpert[k], upPerExpert[k], expertFfnDim);
                } else {
                    silu(gatePerExpert[k], expertFfnDim);
                    elementwiseMul(gatePerExpert[k], upPerExpert[k], expertFfnDim);
                }
            }

            // === Phase 3: Down matmuls (batched) ===
            for (int k = 0; k < expertUsedCount; k++) {
                int e = selectedExperts[k];

                // Upload activated gate as input
                MemorySegment.copy(gatePerExpert[k], 0, hostBuf, ValueLayout.JAVA_FLOAT, 0, expertFfnDim);
                cudaContext.writeBuffer(gpuInputBuf, hostBuf, ffnBytes);

                // Zero output
                cudaContext.fillBufferZero(gpuDownOutBufs[k], dimBytes);

                // Down kernel
                long downPtr = getOrUploadSlice(downExps, e, downElements, layer, 2);
                launchKernelNoSync(temp, downPtr, gpuInputBuf, gpuDownOutBufs[k], dim, expertFfnDim);
            }

            // Single sync for all down kernels
            cudaContext.finish();

            // Bulk download down results
            for (int k = 0; k < expertUsedCount; k++) {
                cudaContext.readBuffer(gpuDownOutBufs[k], hostBuf, dimBytes);
                MemorySegment.copy(hostBuf, ValueLayout.JAVA_FLOAT, 0, outPerExpert[k], 0, dim);

                int e = selectedExperts[k];
                if (downExpsBias != null) addExpertBias(outPerExpert[k], downExpsBias, e, dim);
            }
        }
    }

    public String getStats() {
        long total = hits + misses;
        double hitRate = total > 0 ? (100.0 * hits / total) : 0;
        return String.format("Expert GPU cache: %d hits, %d misses (%.1f%% hit rate)",
            hits, misses, hitRate);
    }

    public long getHits() { return hits; }
    public long getMisses() { return misses; }

    public void close() {
        for (int i = 0; i < maxSlots; i++) {
            if (gpuBuffers[i] != 0) {
                cudaContext.freeBuffer(gpuBuffers[i]);
                gpuBuffers[i] = 0;
            }
        }
        if (gpuInputBuf != 0) cudaContext.freeBuffer(gpuInputBuf);
        freePerExpertBuffers();
    }

    // --- Private helpers ---

    private long getOrUploadSlice(FloatTensor tensor, int expert, long elements, int layer, int proj) {
        long key = encodeKey(layer, proj, expert);
        Integer slotIdx = keyToSlot.get(key);

        if (slotIdx != null) {
            accessTimes[slotIdx] = ++accessCounter;
            hits++;
            return gpuBuffers[slotIdx];
        }

        int slot = findLruSlot();
        long oldKey = slotKeys[slot];
        if (oldKey != -1L) keyToSlot.remove(oldKey);

        uploadExpertSlice(tensor, expert, elements, gpuBuffers[slot]);
        slotKeys[slot] = key;
        accessTimes[slot] = ++accessCounter;
        keyToSlot.put(key, slot);
        misses++;

        return gpuBuffers[slot];
    }

    private void launchKernelNoSync(Arena temp, long gpuWeights, long gpuInput, long gpuOutput,
                                     int rows, int cols) {
        MemorySegment params = cudaContext.buildKernelParams(temp,
            gpuWeights, gpuInput, gpuOutput, rows, cols, 0);
        long rowsPerBlock = cudaBlockSize / 32;
        long globalSize = ((rows + rowsPerBlock - 1) / rowsPerBlock) * cudaBlockSize;
        cudaContext.launchKernel1D(mxfp4Kernel, globalSize, cudaBlockSize, 0, params);
        // NO finish() — caller batches multiple launches then syncs once
    }

    private void uploadExpertSlice(FloatTensor tensor, int expert, long elements, long gpuPtr) {
        long elementOffset = expert * elements;
        long numBlocks = elements / BLOCK_SIZE;
        long byteOffset = (elementOffset / BLOCK_SIZE) * BLOCK_BYTES;
        long byteSize = numBlocks * BLOCK_BYTES;

        TensorData data = tensor.data();
        try (Arena staging = Arena.ofConfined()) {
            MemorySegment hostBuf = staging.allocate(byteSize);
            if (data instanceof it.denzosoft.llmplayer.tensor.MemorySegmentTensorData msData) {
                MemorySegment seg = msData.segment();
                MemorySegment.copy(seg, byteOffset, hostBuf, 0, byteSize);
            } else {
                byte[] chunk = new byte[(int) Math.min(byteSize, 65536)];
                long remaining = byteSize;
                long off = 0;
                while (remaining > 0) {
                    int toRead = (int) Math.min(remaining, chunk.length);
                    data.copyBytes(byteOffset + off, chunk, 0, toRead);
                    MemorySegment.copy(chunk, 0, hostBuf, ValueLayout.JAVA_BYTE, off, toRead);
                    off += toRead;
                    remaining -= toRead;
                }
            }
            cudaContext.writeBuffer(gpuPtr, hostBuf, byteSize);
        }
    }

    private int findLruSlot() {
        for (int i = 0; i < maxSlots; i++) {
            if (slotKeys[i] == -1L) return i;
        }
        int lruSlot = 0;
        long minTime = accessTimes[0];
        for (int i = 1; i < maxSlots; i++) {
            if (accessTimes[i] < minTime) {
                minTime = accessTimes[i];
                lruSlot = i;
            }
        }
        return lruSlot;
    }

    private static long encodeKey(int layer, int projection, int expert) {
        return ((long) layer << 32) | ((long) projection << 16) | expert;
    }

    private void ensureInputBuffer(int dim) {
        if (dim > inputBufDim) {
            if (gpuInputBuf != 0) cudaContext.freeBuffer(gpuInputBuf);
            gpuInputBuf = cudaContext.allocBuffer((long) dim * Float.BYTES);
            inputBufDim = dim;
        }
    }

    private void ensurePerExpertBuffers(int expertCount, int expertFfnDim, int dim) {
        if (expertFfnDim > gateBufDim) {
            for (int i = 0; i < MAX_EXPERTS_PER_TOKEN; i++) {
                if (gpuGateOutBufs[i] != 0) cudaContext.freeBuffer(gpuGateOutBufs[i]);
                if (gpuUpOutBufs[i] != 0) cudaContext.freeBuffer(gpuUpOutBufs[i]);
                gpuGateOutBufs[i] = cudaContext.allocBuffer((long) expertFfnDim * Float.BYTES);
                gpuUpOutBufs[i] = cudaContext.allocBuffer((long) expertFfnDim * Float.BYTES);
            }
            gateBufDim = expertFfnDim;
        }
        if (dim > downBufDim) {
            for (int i = 0; i < MAX_EXPERTS_PER_TOKEN; i++) {
                if (gpuDownOutBufs[i] != 0) cudaContext.freeBuffer(gpuDownOutBufs[i]);
                gpuDownOutBufs[i] = cudaContext.allocBuffer((long) dim * Float.BYTES);
            }
            downBufDim = dim;
        }
    }

    private void freePerExpertBuffers() {
        for (int i = 0; i < MAX_EXPERTS_PER_TOKEN; i++) {
            if (gpuGateOutBufs[i] != 0) cudaContext.freeBuffer(gpuGateOutBufs[i]);
            if (gpuUpOutBufs[i] != 0) cudaContext.freeBuffer(gpuUpOutBufs[i]);
            if (gpuDownOutBufs[i] != 0) cudaContext.freeBuffer(gpuDownOutBufs[i]);
        }
    }

    private static void silu(float[] x, int size) {
        for (int i = 0; i < size; i++) {
            x[i] = x[i] / (1.0f + (float) Math.exp(-x[i]));
        }
    }

    private static void elementwiseMul(float[] a, float[] b, int size) {
        for (int i = 0; i < size; i++) {
            a[i] *= b[i];
        }
    }

    private static void swigluOai(float[] gate, float[] up, int size) {
        for (int i = 0; i < size; i++) {
            float x = Math.min(gate[i], 7.0f);
            float y = Math.max(-7.0f, Math.min(up[i], 7.0f));
            float glu = x / (1.0f + (float) Math.exp(-1.702f * x));
            gate[i] = glu * (y + 1.0f);
        }
    }

    private static void addExpertBias(float[] output, FloatTensor bias2D, int expert, int size) {
        long offset = (long) expert * size;
        for (int i = 0; i < size; i++) {
            output[i] += bias2D.getFloat(offset + i);
        }
    }
}
