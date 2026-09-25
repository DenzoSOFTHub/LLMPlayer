package it.denzosoft.llmplayer.tensor;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * {@link ExpertCache} backed by off-heap slots filled with positional reads from the model file.
 *
 * One slot holds one expert of one layer — all three projections together, because the router never
 * needs gate without up and down. That makes the slot the natural eviction unit and keeps each
 * sub-slice at a single quantization type even on mixes like Q4_K_M, where {@code ffn_down_exps} is
 * Q6_K on some layers and Q4_K on others.
 *
 * Reads go through {@link FileChannel#read(ByteBuffer, long)} into a {@code ByteBuffer} view of the
 * slot segment, so the bytes land straight in the cache with no intermediate copy, and the call
 * lowers to {@code pread} on Linux. Positional reads do not touch the channel position and are safe
 * to issue concurrently, so the misses of one layer are filled in parallel.
 *
 * Retention is least-frequently-used with a least-recently-used tiebreak. That is the hot set: on
 * Qwen3-Coder-30B the top 32 of 128 experts carry ~79 % of routing, so counting selections keeps
 * exactly those resident and lets the long cold tail cycle through the remaining slots, instead of a
 * plain LRU letting the tail evict the hot experts.
 */
public final class MappedExpertCache implements ExpertCache {

    private final Arena arena;
    private final FileChannel channel;
    private final long baseAddress;      // address of the whole-file mapping, to turn addresses into file offsets
    private final int maxSlots;
    private final long slotBytes;

    private final MemorySegment[] slotSegments;
    private final FloatTensor[][] slotTensors;   // [slot][projection], rebuilt when a slot is refilled
    private final long[] slotKeys;               // -1 = free
    private final long[] slotAccess;             // LRU clock stamp
    private final int[] slotEpoch;               // prepare() generation that last used the slot
    private final int[] slotFreq;                // selection count of the key each slot holds
    private final Map<Long, Integer> keyToSlot = new HashMap<>();
    private final Map<Long, Integer> freq = new HashMap<>();

    private long accessClock;
    private int epoch;
    private long hits, misses, bytesRead, readNanos;
    private boolean degraded;

    public MappedExpertCache(Path modelPath, TensorData mappedFile, int maxSlots, long slotBytes)
            throws Exception {
        this.arena = Arena.ofShared();
        this.channel = FileChannel.open(modelPath, StandardOpenOption.READ);
        this.baseAddress = ((MemorySegmentTensorData) mappedFile).segment().address();
        this.maxSlots = maxSlots;
        this.slotBytes = slotBytes;
        this.slotSegments = new MemorySegment[maxSlots];
        this.slotTensors = new FloatTensor[maxSlots][3];
        this.slotKeys = new long[maxSlots];
        this.slotAccess = new long[maxSlots];
        this.slotEpoch = new int[maxSlots];
        this.slotFreq = new int[maxSlots];
        java.util.Arrays.fill(slotKeys, -1L);
        for (int i = 0; i < maxSlots; i++) {
            slotSegments[i] = arena.allocate(slotBytes);
        }
        System.out.println("  Expert RAM cache: " + maxSlots + " experts x "
            + (slotBytes / 1024) + " KB = " + (maxSlots * slotBytes / (1024 * 1024))
            + " MB, filled by positional reads (SSD streaming)");
    }

    private static long key(int layer, int expert) {
        return ((long) layer << 32) | (expert & 0xFFFFFFFFL);
    }

    @Override
    public synchronized boolean prepare(int layer, int[] selectedExperts, int usedCount,
                                        FloatTensor gateExps, FloatTensor upExps, FloatTensor downExps,
                                        long elementsPerSlice) {
        if (degraded) return false;
        FloatTensor[] projections = { gateExps, upExps, downExps };
        long[] byteLen = new long[3];
        long total = 0;
        for (int p = 0; p < 3; p++) {
            if (projections[p] == null) return false;
            byteLen[p] = sliceBytes(projections[p], elementsPerSlice);
            if (byteLen[p] <= 0) return false;
            total += byteLen[p];
        }
        if (total > slotBytes) return false; // slot sizing was computed from a smaller layer

        epoch++;
        int[] fillSlots = new int[usedCount];
        int[] fillExperts = new int[usedCount];
        int fillCount = 0;

        for (int k = 0; k < usedCount; k++) {
            int expert = selectedExperts[k];
            if (expert < 0) continue; // backfilled slot from a NaN router row
            long id = key(layer, expert);
            freq.merge(id, 1, Integer::sum);
            Integer existing = keyToSlot.get(id);
            if (existing != null) {
                hits++;
                slotAccess[existing] = ++accessClock;
                slotEpoch[existing] = epoch;
                slotFreq[existing] = freq.get(id);
                continue;
            }
            int slot = findVictim();
            if (slot < 0) return false; // every slot is in use by this same layer — cache too small
            misses++;
            long old = slotKeys[slot];
            if (old != -1L) keyToSlot.remove(old);
            slotKeys[slot] = id;
            slotAccess[slot] = ++accessClock;
            slotEpoch[slot] = epoch;
            slotFreq[slot] = freq.get(id);
            keyToSlot.put(id, slot);
            fillSlots[fillCount] = slot;
            fillExperts[fillCount] = expert;
            fillCount++;
        }

        if (fillCount > 0) {
            final int n = fillCount;
            long t0 = System.nanoTime();
            try {
                // Positional reads are independent and the slot segments are disjoint, so the misses
                // of this layer overlap. Nothing shared is mutated here — no locking, and in
                // particular no re-entry into this monitor, which the worker threads do not hold.
                IntStream.range(0, n).parallel().forEach(i ->
                    readSlot(fillSlots[i], fillExperts[i], projections, byteLen, elementsPerSlice));
            } catch (RuntimeException e) {
                degrade(e.getCause() != null ? e.getCause() : e);
                return false;
            }
            readNanos += System.nanoTime() - t0;
            // Tensor construction is serial: TensorFactory's GPU toggle is a static, so it must not
            // be flipped from several threads at once. Building three wrappers per miss is cheap.
            Object savedGpu = TensorFactory.getGpuBufferManager();
            try {
                TensorFactory.setGpuBufferManager(null); // this is a CPU path — want CPU/SIMD tensors
                for (int i = 0; i < n; i++) {
                    int slot = fillSlots[i];
                    long subOffset = 0;
                    for (int p = 0; p < 3; p++) {
                        slotTensors[slot][p] = TensorFactory.create(projections[p].type(),
                            new MemorySegmentTensorData(slotSegments[slot].asSlice(subOffset, byteLen[p])),
                            elementsPerSlice);
                        subOffset += byteLen[p];
                    }
                    bytesRead += subOffset;
                }
            } finally {
                TensorFactory.setGpuBufferManager(savedGpu);
            }
        }
        return true;
    }

    /** Read one expert's three projection slices into its slot. I/O only — no shared state. */
    private void readSlot(int slot, int expert, FloatTensor[] projections, long[] byteLen,
                          long elementsPerSlice) {
        MemorySegment slotSeg = slotSegments[slot];
        long subOffset = 0;
        try {
            for (int p = 0; p < 3; p++) {
                FloatTensor source = projections[p];
                long len = byteLen[p];
                long fileOffset = fileOffsetOf(source) + expertByteOffset(source, expert, elementsPerSlice);
                readFully(slotSeg.asSlice(subOffset, len).asByteBuffer(), fileOffset, len);
                subOffset += len;
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private void readFully(ByteBuffer dst, long fileOffset, long len) throws java.io.IOException {
        dst.limit((int) len);
        long pos = fileOffset;
        while (dst.hasRemaining()) {
            int n = channel.read(dst, pos);
            if (n < 0) throw new java.io.EOFException("short read at " + pos);
            pos += n;
        }
    }

    /**
     * File offset of a tensor, recovered from its mapped address. The GGUF file is mapped as one
     * segment starting at file offset 0, so the distance from the mapping base is the file offset —
     * which avoids threading the GGUF tensor directory through the inference engines.
     */
    private long fileOffsetOf(FloatTensor tensor) {
        return ((MemorySegmentTensorData) tensor.data()).segment().address() - baseAddress;
    }

    private static long sliceBytes(FloatTensor tensor, long elementsPerSlice) {
        GGMLType type = tensor.type();
        if (type == null || type.getBlockSize() <= 0) return -1;
        return (elementsPerSlice / type.getBlockSize()) * (long) type.getTypeSize();
    }

    private static long expertByteOffset(FloatTensor tensor, int expert, long elementsPerSlice) {
        GGMLType type = tensor.type();
        return (expert * elementsPerSlice / type.getBlockSize()) * (long) type.getTypeSize();
    }

    /**
     * Least-frequently-used victim, breaking ties by least-recently-used. Slots claimed by the
     * current prepare() or by the one before it are never evicted: one layer cannot evict its own
     * experts, and batched prefill can prepare the next group of experts while the previous group
     * is still being computed (see {@link ExpertCache#tensorFor}).
     *
     * The scan reads {@code slotFreq}, a plain int array kept alongside the slots, rather than
     * looking each key up in {@link #freq}. The map lookup boxed a Long per slot per miss — a JFR
     * allocation profile attributed 37 MB to this method alone — and the scan is already O(maxSlots),
     * which is 701 slots at a 2 GB budget.
     */
    private int findVictim() {
        int best = -1;
        int bestFreq = Integer.MAX_VALUE;
        long bestAccess = Long.MAX_VALUE;
        for (int i = 0; i < maxSlots; i++) {
            if (slotEpoch[i] >= epoch - 1) continue; // in use by this prepare() or the previous one
            if (slotKeys[i] == -1L) return i;    // free slot
            int fv = slotFreq[i];
            if (fv < bestFreq || (fv == bestFreq && slotAccess[i] < bestAccess)) {
                best = i;
                bestFreq = fv;
                bestAccess = slotAccess[i];
            }
        }
        return best;
    }

    @Override
    public FloatTensor tensorFor(int layer, int expert, int projection) {
        Integer slot = keyToSlot.get(key(layer, expert));
        return slot == null ? null : slotTensors[slot][projection];
    }

    private void degrade(Throwable cause) {
        if (degraded) return;
        degraded = true;
        System.err.println("Expert RAM cache error: " + cause + " — falling back to the mmap path");
        if ("true".equals(System.getProperty("cuda.debug", "false"))) cause.printStackTrace();
    }

    @Override public synchronized long hits() { return hits; }
    @Override public synchronized long misses() { return misses; }
    @Override public synchronized long bytesRead() { return bytesRead; }
    @Override public synchronized long readNanos() { return readNanos; }
    @Override public int slotCount() { return maxSlots; }
    @Override public long slotBytes() { return slotBytes; }

    @Override
    public synchronized String stats() {
        long total = hits + misses;
        if (total == 0) return null;
        return String.format(
            "Expert RAM cache: %.1f%% hit rate (%d hits, %d misses), %.2f GB read from disk in %.1f s",
            100.0 * hits / total, hits, misses,
            bytesRead / 1073741824.0, readNanos / 1e9);
    }

    @Override
    public synchronized void close() {
        try {
            channel.close();
        } catch (Exception ignore) {
            // best-effort
        }
        try {
            arena.close();
        } catch (Exception ignore) {
            // best-effort
        }
    }
}
