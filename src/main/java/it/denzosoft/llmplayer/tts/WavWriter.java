package it.denzosoft.llmplayer.tts;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

/** Mono 16-bit PCM WAV encoding of float samples in [-1, 1]. */
public final class WavWriter {

    private WavWriter() {}

    public static byte[] encode(float[] pcm, int sampleRate) {
        int dataBytes = pcm.length * 2;
        ByteBuffer bb = ByteBuffer.allocate(44 + dataBytes).order(ByteOrder.LITTLE_ENDIAN);
        bb.put(new byte[] {'R', 'I', 'F', 'F'}).putInt(36 + dataBytes).put(new byte[] {'W', 'A', 'V', 'E'});
        bb.put(new byte[] {'f', 'm', 't', ' '}).putInt(16)
          .putShort((short) 1)             // PCM
          .putShort((short) 1)             // mono
          .putInt(sampleRate)
          .putInt(sampleRate * 2)          // byte rate
          .putShort((short) 2)             // block align
          .putShort((short) 16);           // bits per sample
        bb.put(new byte[] {'d', 'a', 't', 'a'}).putInt(dataBytes);
        for (float v : pcm) {
            float c = Math.max(-1f, Math.min(1f, v));
            bb.putShort((short) (c * 32767f));
        }
        return bb.array();
    }

    public static void write(float[] pcm, int sampleRate, Path path) throws IOException {
        Files.write(path, encode(pcm, sampleRate));
    }
}
