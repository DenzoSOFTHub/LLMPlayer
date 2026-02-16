package it.denzosoft.llmplayer.tensor;

import it.denzosoft.llmplayer.gpu.GpuBufferManager;

/**
 * GPU-accelerated Q4_K tensor.
 * Delegates matmulParallel to GPU kernel, falls back to CPU on error.
 */
public class Q4_KGpuTensor extends GpuFloatTensor {

    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 144;
    private static final ThreadLocal<float[]> DOT_BUFFER = ThreadLocal.withInitial(() -> new float[BLOCK_SIZE]);
    private static final ThreadLocal<byte[]> TL_SCALE_BYTES = ThreadLocal.withInitial(() -> new byte[12]);
    private static final ThreadLocal<byte[]> TL_QS = ThreadLocal.withInitial(() -> new byte[128]);
    private static final ThreadLocal<int[]> TL_SCALES = ThreadLocal.withInitial(() -> new int[8]);
    private static final ThreadLocal<int[]> TL_MINS = ThreadLocal.withInitial(() -> new int[8]);

    public Q4_KGpuTensor(TensorData data, long size, GpuBufferManager bufferManager) {
        super(data, size, bufferManager);
    }

    @Override
    public GGMLType type() { return GGMLType.Q4_K; }

    @Override
    protected String kernelResourcePath() { return "kernels/matmul_q4_k.cl"; }

    @Override
    protected String kernelName() { return "matmul_q4_k"; }

    @Override
    protected int blockBytes() { return BLOCK_BYTES; }

    @Override
    protected int blockSize() { return BLOCK_SIZE; }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int j = (int) (index % BLOCK_SIZE);
        long bo = blockIndex * BLOCK_BYTES;

        float d = Float16.toFloat(data.getShortLE(bo));
        float dmin = Float16.toFloat(data.getShortLE(bo + 2));

        int group = j / 64;
        int jLocal = j % 64;
        int l = jLocal % 32;
        boolean isHigh = jLocal >= 32;

        int scaleIdx = group * 2 + (isHigh ? 1 : 0);

        byte[] scaleBytes = TL_SCALE_BYTES.get();
        data.copyBytes(bo + 4, scaleBytes, 0, 12);
        int sc, m;
        if (scaleIdx < 4) {
            sc = Byte.toUnsignedInt(scaleBytes[scaleIdx]) & 0x3F;
            m = Byte.toUnsignedInt(scaleBytes[scaleIdx + 4]) & 0x3F;
        } else {
            sc = (Byte.toUnsignedInt(scaleBytes[scaleIdx + 4]) & 0x0F)
               | ((Byte.toUnsignedInt(scaleBytes[scaleIdx - 4]) >> 6) << 4);
            m = ((Byte.toUnsignedInt(scaleBytes[scaleIdx + 4]) >> 4) & 0x0F)
              | ((Byte.toUnsignedInt(scaleBytes[scaleIdx]) >> 6) << 4);
        }

        int qsOffset = group * 32 + l;
        int qsByte = Byte.toUnsignedInt(data.getByte(bo + 16 + qsOffset));
        int q = isHigh ? ((qsByte >> 4) & 0x0F) : (qsByte & 0x0F);

        return d * sc * q - dmin * m;
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        float result = 0f;
        int numBlocks = length / BLOCK_SIZE;
        long blockStart = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        int otherBase = otherOffset;
        float[] tmp = DOT_BUFFER.get();
        byte[] scaleBytes = TL_SCALE_BYTES.get();
        byte[] qs = TL_QS.get();
        int[] scales = TL_SCALES.get();
        int[] mins = TL_MINS.get();

        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float d = Float16.toFloat(data.getShortLE(bo));
            float dmin = Float16.toFloat(data.getShortLE(bo + 2));

            data.copyBytes(bo + 4, scaleBytes, 0, 12);
            data.copyBytes(bo + 16, qs, 0, 128);
            for (int sb = 0; sb < 8; sb++) {
                if (sb < 4) {
                    scales[sb] = Byte.toUnsignedInt(scaleBytes[sb]) & 0x3F;
                    mins[sb] = Byte.toUnsignedInt(scaleBytes[sb + 4]) & 0x3F;
                } else {
                    scales[sb] = (Byte.toUnsignedInt(scaleBytes[sb + 4]) & 0x0F)
                               | ((Byte.toUnsignedInt(scaleBytes[sb - 4]) >> 6) << 4);
                    mins[sb] = ((Byte.toUnsignedInt(scaleBytes[sb + 4]) >> 4) & 0x0F)
                             | ((Byte.toUnsignedInt(scaleBytes[sb]) >> 6) << 4);
                }
            }

            for (int group = 0; group < 4; group++) {
                float ds0 = d * scales[group * 2];
                float dm0 = dmin * mins[group * 2];
                float ds1 = d * scales[group * 2 + 1];
                float dm1 = dmin * mins[group * 2 + 1];

                for (int l = 0; l < 32; l++) {
                    int qsByte2 = Byte.toUnsignedInt(qs[group * 32 + l]);
                    int q0 = qsByte2 & 0x0F;
                    int q1 = (qsByte2 >> 4) & 0x0F;

                    tmp[group * 64 + l]      = ds0 * q0 - dm0;
                    tmp[group * 64 + 32 + l] = ds1 * q1 - dm1;
                }
            }

            result += VectorOpsFactory.get().dot(tmp, 0, other, otherBase, BLOCK_SIZE);
            otherBase += BLOCK_SIZE;
        }
        return result;
    }
}
