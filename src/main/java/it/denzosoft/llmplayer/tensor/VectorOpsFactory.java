package it.denzosoft.llmplayer.tensor;

/**
 * Factory that provides the best available VectorOps implementation.
 * On Java 21+ with Vector API, returns SimdVectorOps.
 * Otherwise returns ScalarOps.
 */
public final class VectorOpsFactory {

    private static final VectorOps INSTANCE;

    static {
        VectorOps ops = null;
        try {
            Class<?> cls = Class.forName("it.denzosoft.llmplayer.tensor.SimdVectorOps");
            ops = (VectorOps) cls.getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            // Vector API not available
        }
        if (ops == null) {
            ops = new ScalarOps();
        }
        INSTANCE = ops;
        System.out.println("  VectorOps: " + INSTANCE.getClass().getSimpleName());
    }

    private VectorOpsFactory() {}

    public static VectorOps get() {
        return INSTANCE;
    }
}
