package it.denzosoft.llmplayer.tensor;

/**
 * Q2_K quantization: 256 weights per super-block (84 bytes).
 * Layout:
 *   - scales (16 bytes): 16 x uint8 (4-bit scale + 4-bit min), one per 16 weights
 *   - qs (64 bytes): 256 x 2-bit quants
 *   - d (fp16, 2 bytes): super-block scale
 *   - dmin (fp16, 2 bytes): super-block minimum
 * Total: 16 + 64 + 2 + 2 = 84 bytes
 *
 * Element order (ggml dequantize_row_q2_K): the block is two halves of 128 weights, each using 32
 * qs bytes; inside a half, shift s = 0, 2, 4, 6 selects 32 consecutive weights, the first 16 from
 * bytes 0..15 and the next 16 from bytes 16..31 of the half, each 16 with its own scale byte. So
 * weight j = 128n + 32k + l reads bits 2k of qs[32n + l] with scale byte 8n + 2k + l/16.
 * The previous implementation read the 2-bit fields in plain byte order, which returned the
 * right values in the wrong positions for every Q2_K tensor.
 */
public class Q2_KFloatTensor extends FloatTensor {

    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 84;
    private static final ThreadLocal<float[]> BLOCK = ThreadLocal.withInitial(() -> new float[BLOCK_SIZE]);

    public Q2_KFloatTensor(TensorData data, long size) {
        super(data, size);
    }

    @Override
    public GGMLType type() { return GGMLType.Q2_K; }

    @Override
    public float getFloat(long index) {
        long bo = (index / BLOCK_SIZE) * BLOCK_BYTES;
        int j = (int) (index % BLOCK_SIZE);
        int n = j >> 7, k = (j >> 5) & 3, l = j & 31;
        float d = Float16.toFloat(data.getShortLE(bo + 80));
        float dmin = Float16.toFloat(data.getShortLE(bo + 82));
        int sc = Byte.toUnsignedInt(data.getByte(bo + 8 * n + 2 * k + (l >> 4)));
        int q = (Byte.toUnsignedInt(data.getByte(bo + 16 + 32 * n + l)) >> (2 * k)) & 3;
        return d * (sc & 0xF) * q - dmin * (sc >> 4);
    }

    /** One super-block into {@code out[off .. off + 256)}. */
    void dequantizeBlock(long block, float[] out, int off) {
        long bo = block * BLOCK_BYTES;
        float d = Float16.toFloat(data.getShortLE(bo + 80));
        float dmin = Float16.toFloat(data.getShortLE(bo + 82));
        int y = off;
        for (int n = 0; n < 2; n++) {
            long q = bo + 16 + 32L * n;
            for (int k = 0; k < 4; k++) {
                int shift = 2 * k;
                for (int half = 0; half < 2; half++) {
                    int sc = Byte.toUnsignedInt(data.getByte(bo + 8 * n + 2 * k + half));
                    float dl = d * (sc & 0xF), ml = dmin * (sc >> 4);
                    long qb = q + 16L * half;
                    for (int l = 0; l < 16; l++) {
                        out[y++] = dl * ((Byte.toUnsignedInt(data.getByte(qb + l)) >> shift) & 3) - ml;
                    }
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
