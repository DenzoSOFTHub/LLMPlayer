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

    /** Best-effort madvise(MADV_RANDOM) on the mmap to disable read-ahead for random-access (>RAM) loads. */
    private static void adviseRandom(MemorySegment seg, long size) {
        try {
            java.lang.foreign.Linker linker = java.lang.foreign.Linker.nativeLinker();
            MemorySegment fn = linker.defaultLookup().find("madvise").orElse(null);
            if (fn == null) return; // not Linux / no madvise
            java.lang.invoke.MethodHandle h = linker.downcallHandle(fn,
                java.lang.foreign.FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT));
            int rc = (int) h.invoke(seg, size, 1); // MADV_RANDOM = 1 on Linux
            if (rc == 0) System.out.println("  mmap: MADV_RANDOM advised (read-ahead off for lazy >RAM load)");
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
            // For a lazy (no-preload) load of a model larger than RAM, advise the OS that access is
            // random so it does not waste disk bandwidth reading ahead pages that will not be used
            // (MoE cold experts are touched sparsely). Set by LLMEngine.load when it skips preload.
            if ("true".equals(System.getProperty("mmap.advise.random", "false"))) {
                adviseRandom(mapped, fileSize);
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
}
