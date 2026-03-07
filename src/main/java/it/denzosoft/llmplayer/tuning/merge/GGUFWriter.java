package it.denzosoft.llmplayer.tuning.merge;

import it.denzosoft.llmplayer.gguf.GGUFFile;
import it.denzosoft.llmplayer.gguf.GGUFTensorInfo;
import it.denzosoft.llmplayer.tensor.GGMLType;
import it.denzosoft.llmplayer.tensor.TensorData;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Stage 6: Write a new GGUF file with merged LoRA tensors.
 *
 * Strategy: copy the original GGUF file verbatim, then overwrite modified
 * tensor data regions in-place. Since merged tensors are re-quantized back
 * to their original format, byte sizes match exactly and no structural
 * changes to the file are needed.
 *
 * Supported quantization formats for write-back: F32, F16, BF16, Q8_0, Q4_0.
 * Tensors with unsupported formats are skipped with a warning.
 */
public class GGUFWriter {

    /**
     * Write a new GGUF file that is a copy of the original with modified tensors
     * re-quantized back to their original format.
     *
     * @param originalPath  path to the original GGUF file
     * @param original      the parsed original GGUF file (memory-mapped)
     * @param mergedTensors merged tensors from LoRAMerger
     * @param outputPath    output file path
     */
    public void write(Path originalPath, GGUFFile original, List<LoRAMerger.MergedTensor> mergedTensors,
                      Path outputPath) throws IOException {
        // Build lookup from tensor name to merged tensor
        Map<String, LoRAMerger.MergedTensor> mergedMap = new HashMap<String, LoRAMerger.MergedTensor>();
        for (LoRAMerger.MergedTensor mt : mergedTensors) {
            mergedMap.put(mt.name(), mt);
        }

        // Step 1: Copy the entire original file
        System.out.println("[GGUFWriter] Copying original GGUF to " + outputPath);
        Files.copy(originalPath, outputPath, StandardCopyOption.REPLACE_EXISTING);

        long tensorDataOffset = original.getTensorDataOffset();
        int written = 0;
        int skipped = 0;

        // Step 2: Overwrite modified tensor regions in place
        RandomAccessFile raf = new RandomAccessFile(outputPath.toFile(), "rw");
        try {
            for (GGUFTensorInfo tensorInfo : original.getTensorInfos()) {
                LoRAMerger.MergedTensor merged = mergedMap.get(tensorInfo.name());
                if (merged == null) {
                    continue;
                }

                GGMLType type = tensorInfo.type();
                if (!isSupportedType(type)) {
                    System.out.println("[GGUFWriter] WARNING: unsupported quant type " + type
                            + " for tensor " + tensorInfo.name() + ", skipping write-back");
                    skipped++;
                    continue;
                }

                // Re-quantize merged F32 data back to original format
                byte[] quantized = quantize(merged.data(), type);

                // Verify size matches
                long expectedSize = tensorInfo.byteSize();
                if (quantized.length != expectedSize) {
                    System.out.println("[GGUFWriter] WARNING: size mismatch for " + tensorInfo.name()
                            + " (expected " + expectedSize + ", got " + quantized.length + "), skipping");
                    skipped++;
                    continue;
                }

                // Seek to tensor data position and write
                long absoluteOffset = tensorDataOffset + tensorInfo.offset();
                raf.seek(absoluteOffset);
                raf.write(quantized);
                written++;

                System.out.println("[GGUFWriter] Wrote " + tensorInfo.name()
                        + " | " + type + " | " + quantized.length + " bytes");
            }
        } finally {
            raf.close();
        }

        long fileSize = Files.size(outputPath);
        System.out.println("[GGUFWriter] Complete: " + written + " tensors written, "
                + skipped + " skipped, output " + (fileSize / (1024 * 1024)) + " MB");
    }

    private boolean isSupportedType(GGMLType type) {
        return type == GGMLType.F32
                || type == GGMLType.F16
                || type == GGMLType.BF16
                || type == GGMLType.Q8_0
                || type == GGMLType.Q4_0;
    }

    /** Quantize F32 data back to the specified GGML format. */
    private byte[] quantize(float[] data, GGMLType type) {
        if (type == GGMLType.F32) {
            return quantizeF32(data);
        } else if (type == GGMLType.F16) {
            return quantizeF16(data);
        } else if (type == GGMLType.BF16) {
            return quantizeBF16(data);
        } else if (type == GGMLType.Q8_0) {
            return quantizeQ8_0(data);
        } else if (type == GGMLType.Q4_0) {
            return quantizeQ4_0(data);
        }
        throw new IllegalArgumentException("Unsupported quantization type: " + type);
    }

    /** F32: 4 bytes per element, little-endian. */
    private byte[] quantizeF32(float[] data) {
        byte[] out = new byte[data.length * 4];
        for (int i = 0; i < data.length; i++) {
            int bits = Float.floatToRawIntBits(data[i]);
            int off = i * 4;
            out[off] = (byte) bits;
            out[off + 1] = (byte) (bits >>> 8);
            out[off + 2] = (byte) (bits >>> 16);
            out[off + 3] = (byte) (bits >>> 24);
        }
        return out;
    }

    /** F16: IEEE 754 half-precision, 2 bytes per element, little-endian. */
    private byte[] quantizeF16(float[] data) {
        byte[] out = new byte[data.length * 2];
        for (int i = 0; i < data.length; i++) {
            short h = floatToHalf(data[i]);
            int off = i * 2;
            out[off] = (byte) h;
            out[off + 1] = (byte) (h >>> 8);
        }
        return out;
    }

    /** BF16: upper 16 bits of float32, 2 bytes per element, little-endian. */
    private byte[] quantizeBF16(float[] data) {
        byte[] out = new byte[data.length * 2];
        for (int i = 0; i < data.length; i++) {
            // Round-to-nearest-even: add rounding bias before truncating
            int bits = Float.floatToRawIntBits(data[i]);
            int lsb = (bits >>> 16) & 1;
            bits += 0x7FFF + lsb;
            short bf = (short) (bits >>> 16);
            int off = i * 2;
            out[off] = (byte) bf;
            out[off + 1] = (byte) (bf >>> 8);
        }
        return out;
    }

    /**
     * Q8_0: blocks of 32 values.
     * Layout per block: 2 bytes F16 scale (d), 32 bytes int8 quants.
     * Total: 34 bytes per block.
     */
    private byte[] quantizeQ8_0(float[] data) {
        int blockCount = data.length / 32;
        byte[] out = new byte[blockCount * 34];
        for (int b = 0; b < blockCount; b++) {
            int dataOffset = b * 32;
            int outOffset = b * 34;

            // Find max absolute value and compute scale
            float amax = 0f;
            for (int i = 0; i < 32; i++) {
                float abs = Math.abs(data[dataOffset + i]);
                if (abs > amax) amax = abs;
            }
            float d = amax / 127f;
            float id = (d != 0f) ? 1f / d : 0f;
            short hd = floatToHalf(d);
            out[outOffset] = (byte) hd;
            out[outOffset + 1] = (byte) (hd >>> 8);
            for (int i = 0; i < 32; i++) {
                int q = Math.round(data[dataOffset + i] * id);
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                out[outOffset + 2 + i] = (byte) q;
            }
        }
        return out;
    }

    /**
     * Q4_0: blocks of 32 values.
     * Layout per block: 2 bytes F16 scale (d), 16 bytes packed nibbles.
     * Low nibble = quants[2*j], high nibble = quants[2*j+1]. Each value + 8 to store unsigned.
     * Total: 18 bytes per block.
     */
    private byte[] quantizeQ4_0(float[] data) {
        int blockCount = data.length / 32;
        byte[] out = new byte[blockCount * 18];
        for (int b = 0; b < blockCount; b++) {
            int dataOffset = b * 32;
            int outOffset = b * 18;

            // Find max absolute value and compute scale (range [-8d, 7d])
            float amax = 0f;
            for (int i = 0; i < 32; i++) {
                float abs = Math.abs(data[dataOffset + i]);
                if (abs > amax) amax = abs;
            }
            float d = amax / 7f;
            float id = (d != 0f) ? 1f / d : 0f;
            short hd = floatToHalf(d);
            out[outOffset] = (byte) hd;
            out[outOffset + 1] = (byte) (hd >>> 8);
            // Quantize to 4-bit and pack two per byte
            for (int j = 0; j < 16; j++) {
                float v0 = data[dataOffset + 2 * j];
                float v1 = data[dataOffset + 2 * j + 1];
                // Quantize to signed [-8, 7] range, then add 8 for unsigned [0, 15] storage
                int q0 = Math.round(v0 * id) + 8;
                int q1 = Math.round(v1 * id) + 8;
                if (q0 < 0) q0 = 0;
                if (q0 > 15) q0 = 15;
                if (q1 < 0) q1 = 0;
                if (q1 > 15) q1 = 15;
                out[outOffset + 2 + j] = (byte) (q0 | (q1 << 4));
            }
        }
        return out;
    }

    /**
     * Convert float32 to IEEE 754 half-precision (float16).
     * Handles normals, subnormals, infinity, NaN, and zero.
     */
    private static short floatToHalf(float v) {
        int bits = Float.floatToRawIntBits(v);
        int sign = (bits >>> 16) & 0x8000;
        int exp = (bits >>> 23) & 0xFF;
        int mantissa = bits & 0x007FFFFF;

        if (exp == 0xFF) {
            // Inf or NaN
            return (short) (sign | 0x7C00 | (mantissa != 0 ? 0x0200 : 0));
        }

        // Add rounding bias
        int roundBit = 0x00001000;
        bits = (bits & 0x7FFFFFFF) + roundBit;
        exp = (bits >>> 23) & 0xFF;
        mantissa = bits & 0x007FFFFF;

        if (exp == 0xFF) {
            // Overflow to infinity after rounding
            return (short) (sign | 0x7C00);
        }

        // Re-bias exponent: float32 bias=127, float16 bias=15
        int expF16 = exp - 127 + 15;

        if (expF16 >= 0x1F) {
            // Overflow: clamp to infinity
            return (short) (sign | 0x7C00);
        }

        if (expF16 <= 0) {
            // Underflow: try subnormal
            if (expF16 < -10) {
                // Too small: flush to zero
                return (short) sign;
            }
            // Subnormal: shift mantissa (with implicit leading 1)
            mantissa |= 0x00800000;
            int shift = 1 - expF16 + 13;
            return (short) (sign | (mantissa >>> shift));
        }

        // Normal case
        return (short) (sign | (expF16 << 10) | (mantissa >>> 13));
    }
}
