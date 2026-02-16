package it.denzosoft.llmplayer.tensor;

/**
 * IQ4_NL quantization: 32 weights per block (non-linear 4-bit).
 * Block layout: [float16 scale (2 bytes)] [16 bytes of nibbles] = 18 bytes/block
 * Uses a non-linear lookup table instead of linear (nibble - 8) dequantization.
 *
 * IMPORTANT: Unlike Q4_0 (interleaved nibbles), IQ4_NL uses SPLIT nibble layout:
 *   - Low nibbles of bytes 0-15 → weights at positions 0-15 (first half)
 *   - High nibbles of bytes 0-15 → weights at positions 16-31 (second half)
 * This matches llama.cpp dequantize_row_iq4_nl: y[j+0] and y[j+QK4_NL/2].
 */
public class IQ4_NLFloatTensor extends FloatTensor {

    private static final int BLOCK_SIZE = 32;
    private static final int BLOCK_BYTES = 18;
    private static final ThreadLocal<float[]> DOT_BUFFER = ThreadLocal.withInitial(() -> new float[BLOCK_SIZE]);

    /**
     * Non-linear lookup table from llama.cpp (ggml-common.h).
     * Maps each 4-bit nibble (0-15) to a dequantized integer value.
     * Values are derived from K-means clustering of quantized model weights.
     */
    static final float[] KVALUES_IQ4NL = {
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113
    };

    public IQ4_NLFloatTensor(TensorData data, long size) {
        super(data, size);
    }

    @Override
    public GGMLType type() { return GGMLType.IQ4_NL; }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int inBlockIndex = (int) (index % BLOCK_SIZE);
        long blockOffset = blockIndex * BLOCK_BYTES;

        short scaleBits = data.getShortLE(blockOffset);
        float scale = Float16.toFloat(scaleBits);

        // Split nibble layout: positions 0-15 use low nibbles, 16-31 use high nibbles
        int nibble;
        if (inBlockIndex < 16) {
            byte packed = data.getByte(blockOffset + 2 + inBlockIndex);
            nibble = packed & 0x0F;
        } else {
            byte packed = data.getByte(blockOffset + 2 + inBlockIndex - 16);
            nibble = (packed >> 4) & 0x0F;
        }
        return scale * KVALUES_IQ4NL[nibble];
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        float result = 0f;
        int numBlocks = length / BLOCK_SIZE;
        long blockStart = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        int otherBase = otherOffset;
        float[] tmp = DOT_BUFFER.get();

        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float scale = Float16.toFloat(data.getShortLE(bo));

            // Split nibble layout: low nibbles → first half, high nibbles → second half
            for (int i = 0; i < 16; i++) {
                byte packed = data.getByte(bo + 2 + i);
                int lo = packed & 0x0F;
                int hi = (packed >> 4) & 0x0F;
                tmp[i]      = scale * KVALUES_IQ4NL[lo];
                tmp[i + 16] = scale * KVALUES_IQ4NL[hi];
            }

            result += VectorOpsFactory.get().dot(tmp, 0, other, otherBase, BLOCK_SIZE);
            otherBase += BLOCK_SIZE;
        }
        return result;
    }
}
