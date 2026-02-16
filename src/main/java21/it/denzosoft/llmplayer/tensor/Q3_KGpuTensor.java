package it.denzosoft.llmplayer.tensor;

import it.denzosoft.llmplayer.gpu.GpuBufferManager;

/**
 * GPU-accelerated Q3_K tensor.
 * Delegates matmulParallel to GPU kernel, falls back to CPU on error.
 */
public class Q3_KGpuTensor extends GpuFloatTensor {

    private static final int BLOCK_SIZE = 256;
    private static final int BLOCK_BYTES = 110;
    private static final ThreadLocal<float[]> DOT_BUFFER = ThreadLocal.withInitial(() -> new float[BLOCK_SIZE]);
    private static final ThreadLocal<byte[]> TL_RAW = ThreadLocal.withInitial(() -> new byte[12]);
    private static final ThreadLocal<byte[]> TL_HM = ThreadLocal.withInitial(() -> new byte[32]);
    private static final ThreadLocal<byte[]> TL_QS = ThreadLocal.withInitial(() -> new byte[64]);
    private static final ThreadLocal<int[]> TL_SC = ThreadLocal.withInitial(() -> new int[16]);

    public Q3_KGpuTensor(TensorData data, long size, GpuBufferManager bufferManager) {
        super(data, size, bufferManager);
    }

    @Override
    public GGMLType type() { return GGMLType.Q3_K; }

    @Override
    protected String kernelResourcePath() { return "kernels/matmul_q3_k.cl"; }

    @Override
    protected String kernelName() { return "matmul_q3_k"; }

    @Override
    protected int blockBytes() { return BLOCK_BYTES; }

    @Override
    protected int blockSize() { return BLOCK_SIZE; }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int j = (int) (index % BLOCK_SIZE);
        long bo = blockIndex * BLOCK_BYTES;

        float d = Float16.toFloat(data.getShortLE(bo + 108));

        int half = j / 128;
        int jInHalf = j % 128;
        int pair = jInHalf / 32;
        int jInPair = jInHalf % 32;
        int which16 = jInPair / 16;
        int l = jInPair % 16;

        int qsByteIdx = half * 32 + which16 * 16 + l;
        int qsShift = pair * 2;
        int qsByte = Byte.toUnsignedInt(data.getByte(bo + 32 + qsByteIdx));
        int lowBits = (qsByte >> qsShift) & 0x03;

        int hmByteIdx = which16 * 16 + l;
        int hmBit = half * 4 + pair;
        int hmByte = Byte.toUnsignedInt(data.getByte(bo + hmByteIdx));
        int hbit = (hmByte >> hmBit) & 1;

        int q = (lowBits | (hbit << 2)) - 4;

        int subBlock = j / 16;
        int[] sc = TL_SC.get();
        decodeAllScales(bo, sc);

        return d * (sc[subBlock] - 32) * q;
    }

    @Override
    public float dot(long thisOffset, float[] other, int otherOffset, int length) {
        float result = 0f;
        int numBlocks = length / BLOCK_SIZE;
        long blockStart = (thisOffset / BLOCK_SIZE) * BLOCK_BYTES;
        int otherIdx = otherOffset;
        float[] tmp = DOT_BUFFER.get();
        byte[] hm = TL_HM.get();
        byte[] qs = TL_QS.get();
        int[] sc = TL_SC.get();

        for (int b = 0; b < numBlocks; b++) {
            long bo = blockStart + (long) b * BLOCK_BYTES;
            float d = Float16.toFloat(data.getShortLE(bo + 108));

            data.copyBytes(bo, hm, 0, 32);
            data.copyBytes(bo + 32, qs, 0, 64);
            decodeAllScales(bo, sc);

            int scaleIdx = 0;
            int hmBitPos = 0;

            for (int half = 0; half < 2; half++) {
                int qBase = half * 32;

                for (int pair = 0; pair < 4; pair++) {
                    int shift = pair * 2;
                    float dl0 = d * (sc[scaleIdx] - 32);
                    scaleIdx++;
                    float dl1 = d * (sc[scaleIdx] - 32);
                    scaleIdx++;

                    int elemBase = half * 128 + pair * 32;

                    for (int l = 0; l < 16; l++) {
                        int qsByte = Byte.toUnsignedInt(qs[qBase + l]);
                        int hmByte = Byte.toUnsignedInt(hm[l]);
                        int lowBits = (qsByte >> shift) & 3;
                        int hbit = (hmByte >> hmBitPos) & 1;
                        int qVal = (lowBits | (hbit << 2)) - 4;
                        tmp[elemBase + l] = dl0 * qVal;
                    }

                    for (int l = 0; l < 16; l++) {
                        int qsByte = Byte.toUnsignedInt(qs[qBase + 16 + l]);
                        int hmByte = Byte.toUnsignedInt(hm[16 + l]);
                        int lowBits = (qsByte >> shift) & 3;
                        int hbit = (hmByte >> hmBitPos) & 1;
                        int qVal = (lowBits | (hbit << 2)) - 4;
                        tmp[elemBase + 16 + l] = dl1 * qVal;
                    }

                    hmBitPos++;
                }
            }

            result += VectorOpsFactory.get().dot(tmp, 0, other, otherIdx, BLOCK_SIZE);
            otherIdx += BLOCK_SIZE;
        }
        return result;
    }

    private void decodeAllScales(long bo, int[] sc) {
        byte[] raw = TL_RAW.get();
        data.copyBytes(bo + 96, raw, 0, 12);

        for (int i = 0; i < 8; i++) {
            int v = Byte.toUnsignedInt(raw[i]);
            sc[i] = v & 0x0F;
            sc[i + 8] = v >> 4;
        }
        for (int i = 0; i < 4; i++) {
            int v = Byte.toUnsignedInt(raw[8 + i]);
            sc[i]     |= (v & 0x03) << 4;
            sc[i + 4] |= ((v >> 2) & 0x03) << 4;
            sc[i + 8] |= ((v >> 4) & 0x03) << 4;
            sc[i + 12]|= ((v >> 6) & 0x03) << 4;
        }
    }
}
