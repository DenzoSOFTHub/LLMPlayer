package it.denzosoft.llmplayer.tensor;

/**
 * IQ1_M quantization: 256 weights per super-block (1.75 bpw), 56 bytes:
 * <ul>
 * <li>qs[32]: low 8 bits of the 11-bit grid index of each group of 8 weights</li>
 * <li>qh[16]: per pair of groups, the high 3 bits of each index and the sign of its delta</li>
 * <li>scales[8]: four uint16 holding a 3-bit scale per 16 weights, and in their top nibbles the
 *     fp16 super-block scale</li>
 * </ul>
 * Weight = d * (2 * scale3 + 1) * (grid[j] + delta), grid values in {-1, 0, 1} from
 * {@code iq1s_grid}, delta = ±0.125 (ggml dequantize_row_iq1_m).
 */
public class IQ1_MFloatTensor extends FloatTensor {

    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 56;
    private static final ThreadLocal<float[]> BLOCK = ThreadLocal.withInitial(() -> new float[BLOCK_SIZE]);

    public IQ1_MFloatTensor(TensorData data, long size) {
        super(data, size);
    }

    @Override
    public GGMLType type() { return GGMLType.IQ1_M; }

    @Override
    public float getFloat(long index) {
        float[] b = BLOCK.get();
        dequantizeBlock(index / BLOCK_SIZE, b, 0);
        return b[(int) (index % BLOCK_SIZE)];
    }

    /** One super-block into {@code out[off .. off + 256)}. */
    void dequantizeBlock(long block, float[] out, int off) {
        long bo = block * BLOCK_BYTES;
        int sc0 = data.getShortLE(bo + 48) & 0xFFFF, sc1 = data.getShortLE(bo + 50) & 0xFFFF;
        int sc2 = data.getShortLE(bo + 52) & 0xFFFF, sc3 = data.getShortLE(bo + 54) & 0xFFFF;
        int u16 = (sc0 >> 12) | ((sc1 >> 8) & 0x00F0) | ((sc2 >> 4) & 0x0F00) | (sc3 & 0xF000);
        float d = Float16.toFloat((short) u16);
        int[] sc = {sc0, sc1, sc2, sc3};
        int y = off;
        for (int ib = 0; ib < 8; ib++) {
            int s = sc[ib >> 1];
            int shift = 6 * (ib & 1);
            float dl1 = d * (2 * ((s >> shift) & 7) + 1);
            float dl2 = d * (2 * ((s >> (shift + 3)) & 7) + 1);
            int qh0 = Byte.toUnsignedInt(data.getByte(bo + 32 + 2 * ib));
            int qh1 = Byte.toUnsignedInt(data.getByte(bo + 32 + 2 * ib + 1));
            long qsBase = bo + 4L * ib;
            for (int l = 0; l < 4; l++) {
                int qh = l < 2 ? qh0 : qh1;
                int idx = Byte.toUnsignedInt(data.getByte(qsBase + l))
                    | (((l & 1) == 0 ? (qh << 8) : (qh << 4)) & 0x700);
                float delta = (qh & ((l & 1) == 0 ? 0x08 : 0x80)) != 0 ? -IQ1GridTables.IQ1S_DELTA : IQ1GridTables.IQ1S_DELTA;
                float dl = l < 2 ? dl1 : dl2;
                long g = IQ1GridTables.IQ1S_GRID[idx];
                for (int j = 0; j < 8; j++) {
                    out[y++] = dl * ((byte) (g >>> (8 * j)) + delta);
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
