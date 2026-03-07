package it.denzosoft.llmplayer.tuning.merge;

import it.denzosoft.llmplayer.gguf.GGUFFile;
import it.denzosoft.llmplayer.gguf.GGUFTensorInfo;
import it.denzosoft.llmplayer.tensor.GGMLType;
import it.denzosoft.llmplayer.tensor.TensorData;
import it.denzosoft.llmplayer.tuning.train.LoRAAdapter;
import it.denzosoft.llmplayer.tuning.train.TrainableTensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Stage 5: Merge trained LoRA adapter weights back into the base model.
 *
 * For each adapter, the corresponding base tensor is dequantized to F32,
 * the LoRA delta (B * A * alpha/rank) is added element-by-element, and
 * the result is stored as a MergedTensor.
 */
public class LoRAMerger {

    /** Holds the merged F32 data for a single tensor. */
    public static class MergedTensor {
        private final String name;
        private final float[] data;
        private final int rows;
        private final int cols;
        private final GGMLType originalType;

        public MergedTensor(String name, float[] data, int rows, int cols, GGMLType originalType) {
            this.name = name;
            this.data = data;
            this.rows = rows;
            this.cols = cols;
            this.originalType = originalType;
        }

        public String name() { return name; }
        public float[] data() { return data; }
        public int rows() { return rows; }
        public int cols() { return cols; }
        public GGMLType originalType() { return originalType; }
    }

    /**
     * Merge all LoRA adapters into the base model tensors.
     *
     * @param gguf     the base GGUF model file
     * @param adapters trained LoRA adapters to merge
     * @return list of merged tensors with F32 data
     */
    public List<MergedTensor> merge(GGUFFile gguf, List<LoRAAdapter> adapters) {
        List<MergedTensor> results = new ArrayList<MergedTensor>();

        for (LoRAAdapter adapter : adapters) {
            String tensorName = adapter.name() + ".weight";
            GGUFTensorInfo tensorInfo = gguf.findTensor(tensorName);
            if (tensorInfo == null) {
                System.out.println("[LoRAMerger] WARNING: tensor not found in GGUF: " + tensorName
                        + ", skipping adapter " + adapter.name());
                continue;
            }

            GGMLType type = tensorInfo.type();
            long elementCount = tensorInfo.elementCount();

            if (!isSupportedType(type)) {
                System.out.println("[LoRAMerger] WARNING: unsupported quant type " + type
                        + " for tensor " + tensorName + ", skipping adapter " + adapter.name());
                continue;
            }

            // Dequantize base tensor to F32
            TensorData tensorData = gguf.getTensorData(tensorInfo);
            float[] base = dequantize(tensorData, type, (int) elementCount);

            // Compute LoRA delta: (B * A) * (alpha / rank)
            TrainableTensor delta = adapter.computeDelta();
            float[] deltaData = delta.data();

            // Add delta element-by-element: W_new = W_base + delta
            for (int i = 0; i < base.length; i++) {
                base[i] += deltaData[i];
            }

            int rows = adapter.outputDim();
            int cols = adapter.inputDim();

            System.out.println("[LoRAMerger] Merged adapter " + adapter.name()
                    + " | type=" + type + " | elements=" + elementCount
                    + " | shape=" + rows + "x" + cols);

            results.add(new MergedTensor(tensorName, base, rows, cols, type));
        }

        return results;
    }

    private boolean isSupportedType(GGMLType type) {
        return type == GGMLType.F32
                || type == GGMLType.F16
                || type == GGMLType.BF16
                || type == GGMLType.Q8_0
                || type == GGMLType.Q4_0;
    }

    private float[] dequantize(TensorData data, GGMLType type, int elementCount) {
        float[] result = new float[elementCount];
        if (type == GGMLType.F32) {
            dequantizeF32(data, result, elementCount);
        } else if (type == GGMLType.F16) {
            dequantizeF16(data, result, elementCount);
        } else if (type == GGMLType.BF16) {
            dequantizeBF16(data, result, elementCount);
        } else if (type == GGMLType.Q8_0) {
            dequantizeQ8_0(data, result, elementCount);
        } else if (type == GGMLType.Q4_0) {
            dequantizeQ4_0(data, result, elementCount);
        }
        return result;
    }

    /** F32: direct read, 4 bytes per element. */
    private void dequantizeF32(TensorData data, float[] result, int elementCount) {
        for (int i = 0; i < elementCount; i++) {
            result[i] = data.getFloatLE(i * 4L);
        }
    }

    /** F16: IEEE 754 half-precision float, 2 bytes per element. */
    private void dequantizeF16(TensorData data, float[] result, int elementCount) {
        for (int i = 0; i < elementCount; i++) {
            int h = data.getShortLE(i * 2L) & 0xFFFF;
            result[i] = halfToFloat(h);
        }
    }

    /** BF16: brain float16, upper 16 bits of IEEE 754 float32. */
    private void dequantizeBF16(TensorData data, float[] result, int elementCount) {
        for (int i = 0; i < elementCount; i++) {
            int bits = (data.getShortLE(i * 2L) & 0xFFFF) << 16;
            result[i] = Float.intBitsToFloat(bits);
        }
    }

    /**
     * Q8_0: blocks of 32 values.
     * Layout: 2 bytes F16 scale (d), then 32 signed int8 quants.
     * Total: 34 bytes per block.
     */
    private void dequantizeQ8_0(TensorData data, float[] result, int elementCount) {
        int blockCount = elementCount / 32;
        for (int b = 0; b < blockCount; b++) {
            long blockOffset = b * 34L;
            float d = halfToFloat(data.getShortLE(blockOffset) & 0xFFFF);
            for (int i = 0; i < 32; i++) {
                byte q = data.getByte(blockOffset + 2 + i);
                result[b * 32 + i] = q * d;
            }
        }
    }

    /**
     * Q4_0: blocks of 32 values.
     * Layout: 2 bytes F16 scale (d), then 16 bytes (2 nibbles each).
     * Low nibble = quants[2*j], high nibble = quants[2*j+1]. Each nibble minus 8.
     * Total: 18 bytes per block.
     */
    private void dequantizeQ4_0(TensorData data, float[] result, int elementCount) {
        int blockCount = elementCount / 32;
        for (int b = 0; b < blockCount; b++) {
            long blockOffset = b * 18L;
            float d = halfToFloat(data.getShortLE(blockOffset) & 0xFFFF);
            for (int j = 0; j < 16; j++) {
                int packed = data.getByte(blockOffset + 2 + j) & 0xFF;
                int lo = packed & 0x0F;
                int hi = (packed >> 4) & 0x0F;
                result[b * 32 + 2 * j] = (lo - 8) * d;
                result[b * 32 + 2 * j + 1] = (hi - 8) * d;
            }
        }
    }

    /**
     * Convert IEEE 754 half-precision (16-bit) to single-precision float.
     * Handles zero exponent (subnormals and zero) as a special case.
     */
    private static float halfToFloat(int h) {
        int exp = (h >> 10) & 0x1F;
        int mantissa = h & 0x03FF;
        if (exp == 0) {
            if (mantissa == 0) {
                // Signed zero
                return Float.intBitsToFloat((h & 0x8000) << 16);
            }
            // Subnormal: normalize
            while ((mantissa & 0x0400) == 0) {
                mantissa <<= 1;
                exp--;
            }
            exp++;
            mantissa &= ~0x0400;
            return Float.intBitsToFloat(
                    ((h & 0x8000) << 16)
                    | ((exp + 0x70) << 23)
                    | (mantissa << 13));
        }
        if (exp == 0x1F) {
            // Inf or NaN
            return Float.intBitsToFloat(
                    ((h & 0x8000) << 16) | 0x7F800000 | (mantissa << 13));
        }
        // Normal case
        return Float.intBitsToFloat(
                ((h & 0x8000) << 16)
                | (((h & 0x7C00) + 0x1C000) << 13)
                | ((h & 0x03FF) << 13));
    }
}
