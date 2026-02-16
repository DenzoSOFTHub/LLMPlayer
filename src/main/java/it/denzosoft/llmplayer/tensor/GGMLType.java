package it.denzosoft.llmplayer.tensor;

public enum GGMLType {
    F32(0, 1, 4),
    F16(1, 1, 2),
    Q4_0(2, 32, 18),      // 32 weights per block, 2 bytes scale + 16 bytes quants
    Q4_1(3, 32, 20),
    // 4, 5 deprecated
    Q5_0(6, 32, 22),
    Q5_1(7, 32, 24),
    Q8_0(8, 32, 34),      // 32 weights per block, 2 bytes scale + 32 bytes quants
    Q8_1(9, 32, 36),
    Q2_K(10, 256, 84),    // super-block 256
    Q3_K(11, 256, 110),   // super-block 256
    Q4_K(12, 256, 144),   // super-block 256
    Q5_K(13, 256, 176),   // super-block 256
    Q6_K(14, 256, 210),   // super-block 256
    Q8_K(15, 256, 292),
    IQ2_XXS(16, 256, 66),
    IQ2_XS(17, 256, 74),
    IQ3_XXS(18, 256, 98),
    IQ1_S(19, 256, 50),
    IQ4_NL(20, 32, 18),
    IQ3_S(21, 256, 110),
    IQ2_S(22, 256, 82),
    IQ4_XS(23, 256, 136),
    I8(24, 1, 1),
    I16(25, 1, 2),
    I32(26, 1, 4),
    I64(27, 1, 8),
    F64(28, 1, 8),
    IQ1_M(29, 256, 56),
    BF16(30, 1, 2),
    // Microscaling FP4: 32 weights per block, 16 bytes FP4 data + 1 byte E8M0 scale = 17 bytes
    MXFP4(39, 32, 17);

    private final int id;
    private final int blockSize;
    private final int typeSize;

    GGMLType(int id, int blockSize, int typeSize) {
        this.id = id;
        this.blockSize = blockSize;
        this.typeSize = typeSize;
    }

    public int getId() { return id; }
    public int getBlockSize() { return blockSize; }
    public int getTypeSize() { return typeSize; }

    public long bytesForElements(long nElements) {
        return (nElements / blockSize) * typeSize;
    }

    private static final GGMLType[] BY_ID = new GGMLType[40];
    static {
        for (GGMLType t : values()) {
            BY_ID[t.id] = t;
        }
    }

    public static GGMLType fromId(int id) {
        if (id < 0 || id >= BY_ID.length || BY_ID[id] == null) {
            throw new IllegalArgumentException("Unknown GGML type id: " + id);
        }
        return BY_ID[id];
    }
}
