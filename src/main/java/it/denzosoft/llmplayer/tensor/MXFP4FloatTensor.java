package it.denzosoft.llmplayer.tensor;

/**
 * MXFP4 (Microscaling FP4 E2M1) quantization: 32 weights per block.
 * Layout (17 bytes) per llama.cpp block_mxfp4:
 *   - e (1 byte): E8M0 shared block exponent
 *   - qs (16 bytes): 32 x 4-bit FP4 E2M1 values (2 per byte)
 *
 * Nibble layout is SPLIT (same as IQ4_NL, NOT interleaved):
 *   - Low nibbles of qs[0..15] → weight positions 0..15 (first half)
 *   - High nibbles of qs[0..15] → weight positions 16..31 (second half)
 *
 * FP4 E2M1 encoding: [sign(1) | exponent(2) | mantissa(1)]
 * Representable values: 0, +/-0.5, +/-1.0, +/-1.5, +/-2.0, +/-3.0, +/-4.0, +/-6.0
 *
 * E8M0 scale: power-of-2 block scale = 2^(exponent - 127)
 * Special: exp=0 → 0 (subnormal), exp=255 → 0 (NaN/invalid block)
 *
 * Dequantization: value = FP4_TABLE[nibble] * 2^(scale_byte - 127)
 */
public class MXFP4FloatTensor extends FloatTensor {

    private static final int BLOCK_SIZE = 32;
    private static final int HALF_BLOCK = BLOCK_SIZE / 2; // 16
    private static final int BLOCK_BYTES = 17;
    private static final ThreadLocal<float[]> DOT_BUFFER = ThreadLocal.withInitial(() -> new float[BLOCK_SIZE]);

    /**
     * FP4 E2M1 lookup table. Maps each 4-bit value to its float representation.
     * Bit layout: [sign(1) | exp(2) | mantissa(1)]
     *
     * Positive values (sign=0):
     *   0b0000 = 0.0, 0b0001 = 0.5, 0b0010 = 1.0, 0b0011 = 1.5,
     *   0b0100 = 2.0, 0b0101 = 3.0, 0b0110 = 4.0, 0b0111 = 6.0
     *
     * Negative values (sign=1): same magnitudes, negated.
     */
    private static final float[] FP4_TABLE = {
         0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };

    public MXFP4FloatTensor(TensorData data, long size) {
        super(data, size);
    }

    @Override
    public GGMLType type() { return GGMLType.MXFP4; }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int inBlockIndex = (int) (index % BLOCK_SIZE);
        long blockOffset = blockIndex * BLOCK_BYTES;

        // E8M0 scale byte is the first byte of the block
        int scaleByte = Byte.toUnsignedInt(data.getByte(blockOffset));
        float scale = e8m0ToFloat(scaleByte);

        // Split nibble layout: positions 0..15 use low nibbles, 16..31 use high nibbles
        int byteIdx;
        int nibble;
        if (inBlockIndex < HALF_BLOCK) {
            byteIdx = inBlockIndex;
            byte packed = data.getByte(blockOffset + 1 + byteIdx);
            nibble = packed & 0x0F;
        } else {
            byteIdx = inBlockIndex - HALF_BLOCK;
            byte packed = data.getByte(blockOffset + 1 + byteIdx);
            nibble = (packed >> 4) & 0x0F;
        }

        return FP4_TABLE[nibble] * scale;
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

            // E8M0 scale is the first byte
            int scaleByte = Byte.toUnsignedInt(data.getByte(bo));
            float scale = e8m0ToFloat(scaleByte);

            // Dequantize 32 FP4 values with SPLIT nibble layout:
            // low nibbles → positions 0..15, high nibbles → positions 16..31
            for (int i = 0; i < HALF_BLOCK; i++) {
                byte packed = data.getByte(bo + 1 + i);
                int lo = packed & 0x0F;
                int hi = (packed >> 4) & 0x0F;
                tmp[i]              = FP4_TABLE[lo] * scale;
                tmp[i + HALF_BLOCK] = FP4_TABLE[hi] * scale;
            }

            result += VectorOpsFactory.get().dot(tmp, 0, other, otherBase, BLOCK_SIZE);
            otherBase += BLOCK_SIZE;
        }
        return result;
    }

    /**
     * Convert E8M0 exponent byte to float scale.
     * E8M0 = unsigned exponent only, no mantissa: value = 2^(exp - 127).
     * Special case: exp=0 → 0 (subnormal/zero block).
     * Special case: exp=255 → 0 (NaN/invalid block, per OCP MX spec).
     */
    private static float e8m0ToFloat(int exp) {
        if (exp == 0 || exp == 255) {
            return 0f;
        }
        // 2^(exp - 127) = build float with biased exponent = exp
        return Float.intBitsToFloat(exp << 23);
    }
}
