package it.denzosoft.llmplayer.tensor;

/**
 * IQ2_XXS quantization: 256 weights per super-block (2.0625 bpw), 66 bytes: fp16 d, then per
 * group of 32 weights two uint32 — four 8-bit grid indices, and four 7-bit sign indices plus a
 * 4-bit scale. Weight = d * (0.5 + scale) * 0.25 * grid[j] * sign (ggml dequantize_row_iq2_xxs).
 */
public class IQ2_XXSFloatTensor extends FloatTensor {

    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 66;
    private static final ThreadLocal<float[]> BLOCK = ThreadLocal.withInitial(() -> new float[BLOCK_SIZE]);

    public IQ2_XXSFloatTensor(TensorData data, long size) {
        super(data, size);
    }

    @Override
    public GGMLType type() { return GGMLType.IQ2_XXS; }

    @Override
    public float getFloat(long index) {
        float[] b = BLOCK.get();
        dequantizeBlock(index / BLOCK_SIZE, b, 0);
        return b[(int) (index % BLOCK_SIZE)];
    }

    /** One super-block into {@code out[off .. off + 256)}. */
    void dequantizeBlock(long block, float[] out, int off) {
        long bo = block * BLOCK_BYTES;
        float d = Float16.toFloat(data.getShortLE(bo));
        int y = off;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            long q = bo + 2 + 8L * ib32;
            int aux1 = data.getIntLE(q + 4);
            float db = d * (0.5f + (aux1 >>> 28)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                long g = IQ1GridTables.IQ2XXS_GRID[Byte.toUnsignedInt(data.getByte(q + l))];
                int signs = IQGridTables.KSIGNS_IQ2XS[(aux1 >>> (7 * l)) & 127];
                for (int j = 0; j < 8; j++) {
                    float v = db * ((g >>> (8 * j)) & 0xFF);
                    out[y++] = (signs & (1 << j)) != 0 ? -v : v;
                }
            }
        }
    }

    @Override
    public void dequantize(float[] out, int outOffset, long srcOffset, int length) {
        if (srcOffset % BLOCK_SIZE != 0 || length % BLOCK_SIZE != 0) {
            super.dequantize(out, outOffset, srcOffset, length);
            return;
        }
        long first = srcOffset / BLOCK_SIZE;
        for (int b = 0; b < length / BLOCK_SIZE; b++) dequantizeBlock(first + b, out, outOffset + b * BLOCK_SIZE);
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        float[] b = BLOCK.get();
        VectorOps ops = VectorOpsFactory.get();
        long first = thisOffset / BLOCK_SIZE;
        float sum = 0f;
        for (int k = 0; k < length / BLOCK_SIZE; k++) {
            dequantizeBlock(first + k, b, 0);
            sum += ops.dot(b, 0, other, otherOffset + k * BLOCK_SIZE, BLOCK_SIZE);
        }
        return sum;
    }
}
