package it.denzosoft.llmplayer.tensor;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * SIMD-optimized Q8_0 tensor using lane-parallel B2I/I2F path.
 *
 * Rewritten 2026-04-15 after JFR flagged {@code SimdQ8_0FloatTensor.dot} as THE
 * hotspot on Qwen3-4B-Thinking Q8_0 (15391 samples, 7× any other method). Old
 * version had a scalar {@code for j in F_LEN} inner loop converting int8 quants
 * to float via a {@code float[F_LEN]} scratch, then {@code FloatVector.fromArray}.
 * New version: {@code ByteVector.fromMemorySegment} → {@code B2I} → {@code I2F}
 * → FMA, zero scratch allocations, zero byte[] copies.
 *
 * Q8_0 block layout (34 bytes, 32 elements):
 *   scale (fp16, 2 bytes)
 *   qs[32]: signed int8 quants
 */
public class SimdQ8_0FloatTensor extends Q8_0FloatTensor {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_256;   // 8 floats
    private static final VectorSpecies<Integer> I_SPECIES = IntVector.SPECIES_256;   // 8 ints
    private static final VectorSpecies<Byte> B_SPECIES = ByteVector.SPECIES_64;      // 8 bytes
    private static final int F_LEN = 8;
    private static final int BLOCK_SIZE = 32;
    private static final int BLOCK_BYTES = 34;
    private static final ValueLayout.OfByte BYTE_LE = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED;
    private static final ByteOrder BYTE_ORDER = ByteOrder.LITTLE_ENDIAN;

    private final MemorySegment segment;

    public SimdQ8_0FloatTensor(TensorData data, long size) {
        super(data, size);
        this.segment = ((MemorySegmentTensorData) data).segment();
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        if (FloatVector.SPECIES_PREFERRED.length() != 8 || length % BLOCK_SIZE != 0) {
            return super.dot(thisOffset, other, otherOffset, length);
        }

        int numBlocks = length / BLOCK_SIZE;
        long blockStart = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        int otherBase = otherOffset;

        FloatVector acc = FloatVector.zero(F_SPECIES);

        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float scale = Float.float16ToFloat(segment.get(SHORT_LE, bo));
            FloatVector vScale = FloatVector.broadcast(F_SPECIES, scale);
            long qsBase = bo + 2;

            // 4 × 8-element sub-chunks per 32-element block
            for (int i = 0; i < BLOCK_SIZE; i += F_LEN) {
                ByteVector vqb = ByteVector.fromMemorySegment(B_SPECIES, segment, qsBase + i, BYTE_ORDER);
                IntVector vqI = (IntVector) vqb.convertShape(VectorOperators.B2I, I_SPECIES, 0);
                FloatVector vqF = (FloatVector) vqI.convertShape(VectorOperators.I2F, F_SPECIES, 0);
                FloatVector in = FloatVector.fromArray(F_SPECIES, other, otherBase + i);
                acc = vqF.fma(vScale.mul(in), acc);
            }
            otherBase += BLOCK_SIZE;
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    @Override
    public float dot(long thisOffset, FloatTensor other, long otherOffset, int length) {
        if (other instanceof SimdQ8_0FloatTensor) {
            return dotQ8Q8Simd(thisOffset, (SimdQ8_0FloatTensor) other, otherOffset, length);
        }
        return super.dot(thisOffset, other, otherOffset, length);
    }

    /**
     * Q8_0 × Q8_0 with lane-parallel integer accumulation.
     */
    private float dotQ8Q8Simd(long thisOffset, SimdQ8_0FloatTensor other, long otherOffset, int length) {
        if (FloatVector.SPECIES_PREFERRED.length() != 8 || length % BLOCK_SIZE != 0) {
            return super.dot(thisOffset, (FloatTensor) other, otherOffset, length);
        }

        int numBlocks = length / BLOCK_SIZE;
        long thisBlock = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        long otherBlock = (otherOffset / BLOCK_SIZE) * BLOCK_BYTES;
        MemorySegment otherSeg = other.segment;

        float result = 0f;

        for (int b = 0; b < numBlocks; b++) {
            float d0 = Float.float16ToFloat(segment.get(SHORT_LE, thisBlock));
            float d1 = Float.float16ToFloat(otherSeg.get(SHORT_LE, otherBlock));
            long qs0 = thisBlock + 2;
            long qs1 = otherBlock + 2;

            IntVector isum = IntVector.zero(I_SPECIES);
            for (int i = 0; i < BLOCK_SIZE; i += F_LEN) {
                ByteVector v0 = ByteVector.fromMemorySegment(B_SPECIES, segment, qs0 + i, BYTE_ORDER);
                ByteVector v1 = ByteVector.fromMemorySegment(B_SPECIES, otherSeg, qs1 + i, BYTE_ORDER);
                IntVector i0 = (IntVector) v0.convertShape(VectorOperators.B2I, I_SPECIES, 0);
                IntVector i1 = (IntVector) v1.convertShape(VectorOperators.B2I, I_SPECIES, 0);
                isum = isum.add(i0.mul(i1));
            }
            result += d0 * d1 * isum.reduceLanes(VectorOperators.ADD);

            thisBlock += BLOCK_BYTES;
            otherBlock += BLOCK_BYTES;
        }
        return result;
    }
}
