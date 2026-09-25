package it.denzosoft.llmplayer.tensor;

/**
 * Abstraction over memory-mapped tensor data, compatible with Java 8+.
 * On Java 21+, backed by MemorySegment (Panama FFM).
 * On older JVMs, backed by MappedByteBuffer.
 */
public interface TensorData {

    byte getByte(long offset);

    short getShortLE(long offset);

    int getIntLE(long offset);

    long getLongLE(long offset);

    float getFloatLE(long offset);

    double getDoubleLE(long offset);

    void copyBytes(long srcOffset, byte[] dst, int dstOffset, int length);

    TensorData slice(long offset, long size);

    long byteSize();

    /**
     * Hint to preload this data into physical memory.
     * Default implementation is a no-op.
     */
    default void preload() {}

    /**
     * Hint that the given byte range will be needed soon, so the OS should start reading it
     * asynchronously. Unlike {@link #preload()} this does not block and does not touch the pages —
     * it only starts the read-ahead, so the caller can issue several ranges up front and overlap
     * the I/O with compute.
     *
     * Used by the MoE expert-granular read-ahead path (see {@code ExpertPrefetch}): a routed
     * expert is a contiguous multi-megabyte slice, and faulting it in one range instead of page by
     * page is what makes a model larger than RAM usable.
     *
     * Default implementation is a no-op.
     */
    default void adviseWillNeed(long offset, long length) {}
}
