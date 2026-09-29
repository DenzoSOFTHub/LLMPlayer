package it.denzosoft.llmplayer.tensor;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * SIMD IQ2_XXS dot product: the 256-entry codebook and the 128 sign patterns are expanded once
 * into floats, so each group of 8 weights is {@code acc += grid * signs * db * x}. No per-call
 * allocation. Layout and math as {@link IQ2_XXSFloatTensor}.
 */
public class SimdIQ2_XXSFloatTensor extends IQ2_XXSFloatTensor {

    private static final VectorSpecies<Float> F8 = FloatVector.SPECIES_256;
    private static final ValueLayout.OfLong LONG_LE = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 66;

    private static final float[] GRID_F = new float[256 * 8];
    private static final float[] SIGN_F = new float[128 * 8];
    static {
        for (int i = 0; i < 256; i++) {
            long g = IQ1GridTables.IQ2XXS_GRID[i];
            for (int j = 0; j < 8; j++) GRID_F[i * 8 + j] = (g >>> (8 * j)) & 0xFF;
        }
        for (int i = 0; i < 128; i++) {
            int s = IQGridTables.KSIGNS_IQ2XS[i];
            for (int j = 0; j < 8; j++) SIGN_F[i * 8 + j] = (s & (1 << j)) != 0 ? -1f : 1f;
        }
    }

    private final MemorySegment segment;

    public SimdIQ2_XXSFloatTensor(TensorData data, long size) {
        super(data, size);
        this.segment = ((MemorySegmentTensorData) data).segment();
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        final float[] grid = GRID_F, signs = SIGN_F;
        FloatVector acc0 = FloatVector.zero(F8), acc1 = FloatVector.zero(F8);
        long bo = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        int x = otherOffset;
        for (int b = 0; b < length / BLOCK_SIZE; b++, bo += BLOCK_BYTES) {
            float d = Float.float16ToFloat(segment.get(SHORT_LE, bo));
            for (int ib32 = 0; ib32 < 8; ib32++) {
                long w = segment.get(LONG_LE, bo + 2 + 8L * ib32);
                int aux0 = (int) w, aux1 = (int) (w >>> 32);
                float db = d * (0.5f + (aux1 >>> 28)) * 0.25f;
                int xo = x + ib32 * 32;
                acc0 = FloatVector.fromArray(F8, grid, (aux0 & 0xFF) * 8)
                    .mul(FloatVector.fromArray(F8, signs, (aux1 & 127) * 8)).mul(db)
                    .fma(FloatVector.fromArray(F8, other, xo), acc0);
                acc1 = FloatVector.fromArray(F8, grid, ((aux0 >>> 8) & 0xFF) * 8)
                    .mul(FloatVector.fromArray(F8, signs, ((aux1 >>> 7) & 127) * 8)).mul(db)
                    .fma(FloatVector.fromArray(F8, other, xo + 8), acc1);
                acc0 = FloatVector.fromArray(F8, grid, ((aux0 >>> 16) & 0xFF) * 8)
                    .mul(FloatVector.fromArray(F8, signs, ((aux1 >>> 14) & 127) * 8)).mul(db)
                    .fma(FloatVector.fromArray(F8, other, xo + 16), acc0);
                acc1 = FloatVector.fromArray(F8, grid, ((aux0 >>> 24) & 0xFF) * 8)
                    .mul(FloatVector.fromArray(F8, signs, ((aux1 >>> 21) & 127) * 8)).mul(db)
                    .fma(FloatVector.fromArray(F8, other, xo + 24), acc1);
            }
            x += BLOCK_SIZE;
        }
        return acc0.add(acc1).reduceLanes(VectorOperators.ADD);
    }
}
