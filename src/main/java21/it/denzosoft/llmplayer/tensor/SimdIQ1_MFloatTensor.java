package it.denzosoft.llmplayer.tensor;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * SIMD IQ1_M dot product. Every group of 8 weights is one codebook entry, so the codebook is
 * expanded once into floats ({@code GRID_F}, 2048 x 8) and a group costs one vector load and
 * three vector operations: {@code acc += (grid + delta) * dl * x}. No per-call allocation.
 * Layout and math as {@link IQ1_MFloatTensor}.
 */
public class SimdIQ1_MFloatTensor extends IQ1_MFloatTensor {

    private static final VectorSpecies<Float> F8 = FloatVector.SPECIES_256;
    private static final ValueLayout.OfLong LONG_LE = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 56;
    private static final float DELTA = 0.125f;

    private static final float[] GRID_F = new float[2048 * 8];
    static {
        for (int i = 0; i < 2048; i++) {
            long g = IQ1GridTables.IQ1S_GRID[i];
            for (int j = 0; j < 8; j++) GRID_F[i * 8 + j] = (byte) (g >>> (8 * j));
        }
    }

    private final MemorySegment segment;

    public SimdIQ1_MFloatTensor(TensorData data, long size) {
        super(data, size);
        this.segment = ((MemorySegmentTensorData) data).segment();
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        final float[] grid = GRID_F;
        FloatVector acc0 = FloatVector.zero(F8), acc1 = FloatVector.zero(F8);
        long bo = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        int x = otherOffset;
        for (int b = 0; b < length / BLOCK_SIZE; b++, bo += BLOCK_BYTES) {
            long qs0 = segment.get(LONG_LE, bo), qs1 = segment.get(LONG_LE, bo + 8);
            long qs2 = segment.get(LONG_LE, bo + 16), qs3 = segment.get(LONG_LE, bo + 24);
            long qhLo = segment.get(LONG_LE, bo + 32), qhHi = segment.get(LONG_LE, bo + 40);
            long scales = segment.get(LONG_LE, bo + 48);
            int sc0 = (int) (scales & 0xFFFF), sc1 = (int) ((scales >>> 16) & 0xFFFF);
            int sc2 = (int) ((scales >>> 32) & 0xFFFF), sc3 = (int) ((scales >>> 48) & 0xFFFF);
            int u16 = (sc0 >> 12) | ((sc1 >> 8) & 0x00F0) | ((sc2 >> 4) & 0x0F00) | (sc3 & 0xF000);
            float d = Float.float16ToFloat((short) u16);
            for (int ib = 0; ib < 8; ib++) {
                int s = ib < 2 ? sc0 : ib < 4 ? sc1 : ib < 6 ? sc2 : sc3;
                int shift = 6 * (ib & 1);
                float dl1 = d * (2 * ((s >> shift) & 7) + 1);
                float dl2 = d * (2 * ((s >> (shift + 3)) & 7) + 1);
                long qhWord = ib < 4 ? qhLo : qhHi;
                int qh0 = (int) ((qhWord >>> (16 * (ib & 3))) & 0xFF);
                int qh1 = (int) ((qhWord >>> (16 * (ib & 3) + 8)) & 0xFF);
                long qsWord = ib < 2 ? qs0 : ib < 4 ? qs1 : ib < 6 ? qs2 : qs3;
                int qsBits = (int) (qsWord >>> (32 * (ib & 1)));
                int i0 = (qsBits & 0xFF) | ((qh0 << 8) & 0x700);
                int i1 = ((qsBits >>> 8) & 0xFF) | ((qh0 << 4) & 0x700);
                int i2 = ((qsBits >>> 16) & 0xFF) | ((qh1 << 8) & 0x700);
                int i3 = ((qsBits >>> 24) & 0xFF) | ((qh1 << 4) & 0x700);
                float d0 = (qh0 & 0x08) != 0 ? -DELTA : DELTA, d1 = (qh0 & 0x80) != 0 ? -DELTA : DELTA;
                float d2 = (qh1 & 0x08) != 0 ? -DELTA : DELTA, d3 = (qh1 & 0x80) != 0 ? -DELTA : DELTA;
                int xo = x + ib * 32;
                acc0 = FloatVector.fromArray(F8, grid, i0 * 8).add(d0).mul(dl1)
                    .fma(FloatVector.fromArray(F8, other, xo), acc0);
                acc1 = FloatVector.fromArray(F8, grid, i1 * 8).add(d1).mul(dl1)
                    .fma(FloatVector.fromArray(F8, other, xo + 8), acc1);
                acc0 = FloatVector.fromArray(F8, grid, i2 * 8).add(d2).mul(dl2)
                    .fma(FloatVector.fromArray(F8, other, xo + 16), acc0);
                acc1 = FloatVector.fromArray(F8, grid, i3 * 8).add(d3).mul(dl2)
                    .fma(FloatVector.fromArray(F8, other, xo + 24), acc1);
            }
            x += BLOCK_SIZE;
        }
        return acc0.add(acc1).reduceLanes(VectorOperators.ADD);
    }
}
