package it.denzosoft.llmplayer.tensor;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Path;

/**
 * Factory for creating TensorData from memory-mapped files.
 * Uses MemorySegment (Panama FFM) on Java 21+, falls back to MappedByteBuffer.
 */
public final class TensorDataFactory {

    private static final boolean HAS_FFM;

    static {
        boolean found;
        try {
            Class.forName("java.lang.foreign.MemorySegment");
            found = true;
        } catch (ClassNotFoundException e) {
            found = false;
        }
        HAS_FFM = found;
    }

    private TensorDataFactory() {}

    public static boolean hasFFM() {
        return HAS_FFM;
    }

    public static MappedFile mapFile(Path path) throws IOException {
        if (HAS_FFM) {
            try {
                Class<?> cls = Class.forName("it.denzosoft.llmplayer.tensor.MemorySegmentTensorData");
                java.lang.reflect.Method m = cls.getMethod("mapFile", Path.class);
                return (MappedFile) m.invoke(null, path);
            } catch (Exception e) {
                System.err.println("Warning: FFM mapping failed, falling back to ByteBuffer: " + e.getMessage());
            }
        }
        return ByteBufferTensorData.mapFile(path);
    }

    /**
     * Wraps a memory-mapped TensorData with lifecycle management.
     */
    public static class MappedFile implements Closeable {
        private final TensorData data;
        private final long fileSize;
        private final Closeable closer;

        public MappedFile(TensorData data, long fileSize, Closeable closer) {
            this.data = data;
            this.fileSize = fileSize;
            this.closer = closer;
        }

        public TensorData data() { return data; }
        public long fileSize() { return fileSize; }

        @Override
        public void close() throws IOException {
            closer.close();
        }
    }
}
