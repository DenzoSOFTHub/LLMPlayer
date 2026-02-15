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
}
