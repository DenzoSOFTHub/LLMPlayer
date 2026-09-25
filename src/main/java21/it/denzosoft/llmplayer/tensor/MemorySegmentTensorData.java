package it.denzosoft.llmplayer.tensor;

import java.io.Closeable;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * TensorData implementation backed by Panama FFM MemorySegment.
 * Requires Java 21+. Provides zero-copy access to memory-mapped files.
 */
public class MemorySegmentTensorData implements TensorData {

    private static final ValueLayout.OfByte BYTE_LE = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt INT_LE = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LONG_LE = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfFloat FLOAT_LE = ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfDouble DOUBLE_LE = ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private final MemorySegment segment;

    public MemorySegmentTensorData(MemorySegment segment) {
        this.segment = segment;
    }

    public MemorySegment segment() { return segment; }

    private static final int MADV_RANDOM = 1;
    private static final int MADV_SEQUENTIAL = 2;
    private static final int MADV_WILLNEED = 3;

    /** Cached madvise(2) downcall handle, or null when the platform has no madvise. Resolved once:
     *  {@link #adviseWillNeed} is on the per-token MoE path (experts x projections x layers calls),
     *  so building a downcall handle per call would be pure overhead. */
    private static final java.lang.invoke.MethodHandle MADVISE = resolveMadvise();

    /** madvise(2) requires a page-aligned start address, so ranges are rounded down to this. */
    private static final long PAGE_SIZE = resolvePageSize();

    private static java.lang.invoke.MethodHandle resolveMadvise() {
        try {
            java.lang.foreign.Linker linker = java.lang.foreign.Linker.nativeLinker();
            MemorySegment fn = linker.defaultLookup().find("madvise").orElse(null);
            if (fn == null) return null; // not Linux / no madvise
            return linker.downcallHandle(fn,
                java.lang.foreign.FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT));
        } catch (Throwable ignore) {
            return null;
        }
    }

    private static long resolvePageSize() {
        try {
            java.lang.foreign.Linker linker = java.lang.foreign.Linker.nativeLinker();
            MemorySegment fn = linker.defaultLookup().find("getpagesize").orElse(null);
            if (fn != null) {
                java.lang.invoke.MethodHandle h = linker.downcallHandle(fn,
                    java.lang.foreign.FunctionDescriptor.of(ValueLayout.JAVA_INT));
                int ps = (int) h.invoke();
                if (ps > 0) return ps;
            }
        } catch (Throwable ignore) {
            // fall through
        }
        return 4096L;
    }

    /** Best-effort madvise() on the mmap. Linux: MADV_RANDOM=1 (sparse access, e.g. MoE cold
     *  experts — disables read-ahead), MADV_SEQUENTIAL=2 (cyclic layer walk of a dense >RAM
     *  model — aggressive read-ahead, pages behind the walk dropped first). */
    private static void advise(MemorySegment seg, long size, int advice, String label) {
        if (MADVISE == null) return;
        try {
            int rc = (int) MADVISE.invoke(seg, size, advice);
            if (rc == 0) System.out.println("  mmap: " + label + " advised (lazy >RAM load)");
        } catch (Throwable ignore) {
            // best-effort; platforms without madvise just keep default read-ahead
        }
    }

    public static TensorDataFactory.MappedFile mapFile(Path path) throws IOException {
        Arena arena = Arena.ofShared();
        try {
            FileChannel channel = FileChannel.open(path, StandardOpenOption.READ);
            long fileSize = channel.size();
            MemorySegment mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize, arena);
            channel.close();
            // For a lazy (no-preload) load of a model larger than RAM, LLMEngine.load picks the
            // advice by access pattern: "random" for MoE (sparse expert touches — read-ahead off),
            // "sequential" for dense (cyclic layer walk — read-ahead up, used pages dropped first).
            String mmapAdvise = System.getProperty("mmap.advise", "none");
            if ("random".equals(mmapAdvise)) {
                advise(mapped, fileSize, MADV_RANDOM, "MADV_RANDOM (read-ahead off, sparse MoE access)");
            } else if ("sequential".equals(mmapAdvise)) {
                advise(mapped, fileSize, MADV_SEQUENTIAL, "MADV_SEQUENTIAL (read-ahead up, cyclic dense layer walk)");
            }
            MemorySegmentTensorData data = new MemorySegmentTensorData(mapped);
            // Wrap Arena as Closeable
            Closeable closer = new Closeable() {
                @Override
                public void close() {
                    arena.close();
                }
            };
            return new TensorDataFactory.MappedFile(data, fileSize, closer);
        } catch (IOException e) {
            arena.close();
            throw e;
        }
    }

    @Override
    public byte getByte(long offset) {
        return segment.get(BYTE_LE, offset);
    }

    @Override
    public short getShortLE(long offset) {
        return segment.get(SHORT_LE, offset);
    }

    @Override
    public int getIntLE(long offset) {
        return segment.get(INT_LE, offset);
    }

    @Override
    public long getLongLE(long offset) {
        return segment.get(LONG_LE, offset);
    }

    @Override
    public float getFloatLE(long offset) {
        return segment.get(FLOAT_LE, offset);
    }

    @Override
    public double getDoubleLE(long offset) {
        return segment.get(DOUBLE_LE, offset);
    }

    @Override
    public void copyBytes(long srcOffset, byte[] dst, int dstOffset, int length) {
        MemorySegment.copy(segment, BYTE_LE, srcOffset, dst, dstOffset, length);
    }

    @Override
    public TensorData slice(long offset, long size) {
        return new MemorySegmentTensorData(segment.asSlice(offset, size));
    }

    @Override
    public long byteSize() {
        return segment.byteSize();
    }

    @Override
    public void preload() {
        segment.load();
    }

    /**
     * Start an asynchronous read-ahead of {@code [offset, offset+length)} via MADV_WILLNEED.
     *
     * MADV_WILLNEED overrides the MADV_RANDOM hint that the lazy >RAM MoE path sets over the whole
     * mapping, which is exactly what is wanted: expert access is random *between* experts but
     * strictly sequential *within* one multi-megabyte expert slice. Without this, faulting a
     * ~2.6 MB expert costs ~650 individual page faults with read-ahead disabled.
     *
     * The call does not block, so several ranges can be queued before the compute loop starts.
     */
    @Override
    public void adviseWillNeed(long offset, long length) {
        if (MADVISE == null || length <= 0 || offset < 0) return;
        if (offset + length > segment.byteSize()) return;
        try {
            // madvise() needs a page-aligned start; round down and extend the length to match.
            long addr = segment.address() + offset;
            long aligned = addr & -PAGE_SIZE;
            MADVISE.invoke(MemorySegment.ofAddress(aligned), length + (addr - aligned), MADV_WILLNEED);
        } catch (Throwable ignore) {
            // best-effort: a failed hint just means the compute thread faults the pages itself
        }
    }
}
