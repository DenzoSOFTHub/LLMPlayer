package it.denzosoft.llmplayer.tensor;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * TensorData implementation backed by segmented MappedByteBuffers.
 * Handles files larger than 2GB by splitting into 1GB segments.
 * Compatible with Java 8+.
 */
public class ByteBufferTensorData implements TensorData {

    private static final int SEGMENT_BITS = 30; // 1 GB per segment
    private static final long SEGMENT_SIZE = 1L << SEGMENT_BITS;
    private static final long SEGMENT_MASK = SEGMENT_SIZE - 1;

    private final MappedByteBuffer[] segments;
    private final long baseOffset;
    private final long size;

    ByteBufferTensorData(MappedByteBuffer[] segments, long baseOffset, long size) {
        this.segments = segments;
        this.baseOffset = baseOffset;
        this.size = size;
    }

    public static TensorDataFactory.MappedFile mapFile(java.nio.file.Path path) throws IOException {
        FileChannel channel = FileChannel.open(path, java.nio.file.StandardOpenOption.READ);
        long fileSize = channel.size();
        int numSegments = (int) ((fileSize + SEGMENT_SIZE - 1) / SEGMENT_SIZE);
        MappedByteBuffer[] segments = new MappedByteBuffer[numSegments];
        for (int i = 0; i < numSegments; i++) {
            long offset = (long) i * SEGMENT_SIZE;
            long segSize = Math.min(SEGMENT_SIZE, fileSize - offset);
            segments[i] = channel.map(FileChannel.MapMode.READ_ONLY, offset, segSize);
            segments[i].order(ByteOrder.LITTLE_ENDIAN);
        }
        ByteBufferTensorData data = new ByteBufferTensorData(segments, 0, fileSize);
        return new TensorDataFactory.MappedFile(data, fileSize, channel);
    }

    @Override
    public byte getByte(long offset) {
        long abs = baseOffset + offset;
        return segments[(int) (abs >>> SEGMENT_BITS)].get((int) (abs & SEGMENT_MASK));
    }

    @Override
    public short getShortLE(long offset) {
        long abs = baseOffset + offset;
        int segIdx = (int) (abs >>> SEGMENT_BITS);
        int segOff = (int) (abs & SEGMENT_MASK);
        if (segOff + 2 <= SEGMENT_SIZE) {
            return segments[segIdx].getShort(segOff);
        }
        // Cross-boundary: read byte by byte (LE)
        int b0 = Byte.toUnsignedInt(segments[segIdx].get(segOff));
        abs++;
        int b1 = Byte.toUnsignedInt(segments[(int) (abs >>> SEGMENT_BITS)].get((int) (abs & SEGMENT_MASK)));
        return (short) (b0 | (b1 << 8));
    }

    @Override
    public int getIntLE(long offset) {
        long abs = baseOffset + offset;
        int segIdx = (int) (abs >>> SEGMENT_BITS);
        int segOff = (int) (abs & SEGMENT_MASK);
        if (segOff + 4 <= SEGMENT_SIZE) {
            return segments[segIdx].getInt(segOff);
        }
        // Cross-boundary
        int result = 0;
        for (int i = 0; i < 4; i++) {
            long a = abs + i;
            result |= (Byte.toUnsignedInt(segments[(int) (a >>> SEGMENT_BITS)].get((int) (a & SEGMENT_MASK)))) << (i * 8);
        }
        return result;
    }

    @Override
    public long getLongLE(long offset) {
        long abs = baseOffset + offset;
        int segIdx = (int) (abs >>> SEGMENT_BITS);
        int segOff = (int) (abs & SEGMENT_MASK);
        if (segOff + 8 <= SEGMENT_SIZE) {
            return segments[segIdx].getLong(segOff);
        }
        // Cross-boundary
        long result = 0;
        for (int i = 0; i < 8; i++) {
            long a = abs + i;
            result |= ((long) Byte.toUnsignedInt(segments[(int) (a >>> SEGMENT_BITS)].get((int) (a & SEGMENT_MASK)))) << (i * 8);
        }
        return result;
    }

    @Override
    public float getFloatLE(long offset) {
        return Float.intBitsToFloat(getIntLE(offset));
    }

    @Override
    public double getDoubleLE(long offset) {
        return Double.longBitsToDouble(getLongLE(offset));
    }

    @Override
    public void copyBytes(long srcOffset, byte[] dst, int dstOffset, int length) {
        long abs = baseOffset + srcOffset;
        int remaining = length;
        int dstPos = dstOffset;
        while (remaining > 0) {
            int segIdx = (int) (abs >>> SEGMENT_BITS);
            int segOff = (int) (abs & SEGMENT_MASK);
            int available = (int) Math.min(remaining, SEGMENT_SIZE - segOff);
            ByteBuffer dup = segments[segIdx].duplicate();
            dup.position(segOff);
            dup.get(dst, dstPos, available);
            abs += available;
            dstPos += available;
            remaining -= available;
        }
    }

    @Override
    public TensorData slice(long offset, long size) {
        return new ByteBufferTensorData(segments, baseOffset + offset, size);
    }

    @Override
    public long byteSize() {
        return size;
    }

    @Override
    public void preload() {
        // Load all segments that overlap our range
        long start = baseOffset;
        long end = baseOffset + size;
        int startSeg = (int) (start >>> SEGMENT_BITS);
        int endSeg = (int) ((end - 1) >>> SEGMENT_BITS);
        for (int s = startSeg; s <= Math.min(endSeg, segments.length - 1); s++) {
            segments[s].load();
        }
    }
}
