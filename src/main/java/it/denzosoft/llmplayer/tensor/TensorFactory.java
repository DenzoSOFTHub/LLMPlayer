package it.denzosoft.llmplayer.tensor;

import java.lang.reflect.Constructor;

public final class TensorFactory {

    private static volatile Object gpuBufferManager; // GpuBufferManager or CudaBufferManager, set via reflection
    private static volatile String gpuBackend = "opencl"; // "cuda" or "opencl"

    private TensorFactory() {}

    /**
     * Called by LLMEngine via reflection to enable/disable GPU tensor creation.
     */
    public static void setGpuBufferManager(Object manager) {
        gpuBufferManager = manager;
    }

    /**
     * Returns the current GPU buffer manager, or null if GPU is not active.
     */
    public static Object getGpuBufferManager() {
        return gpuBufferManager;
    }

    /**
     * Set the GPU backend type: "cuda" or "opencl".
     */
    public static void setGpuBackend(String backend) {
        gpuBackend = backend;
    }

    public static String getGpuBackend() {
        return gpuBackend;
    }

    // Cached SIMD constructors (null = not probed, absent = probed and not available)
    private static volatile Boolean simdAvailable;
    private static volatile Constructor<?> simdQ4KCtor;
    private static volatile Constructor<?> simdQ8_0Ctor;
    private static volatile Constructor<?> simdQ6KCtor;
    private static volatile Constructor<?> simdQ5_0Ctor;
    private static volatile Constructor<?> simdQ5KCtor;
    private static volatile Constructor<?> simdQ3KCtor;
    private static volatile Constructor<?> simdIQ4NLCtor;
    private static volatile Constructor<?> simdIQ4XSCtor;
    private static volatile Constructor<?> simdIQ3XXSCtor;
    private static volatile Constructor<?> simdIQ3SCtor;
    private static volatile Constructor<?> simdIQ2SCtor;

    public static FloatTensor create(GGMLType type, TensorData data, long elementCount) {
        // Try GPU tensor first
        if (gpuBufferManager != null) {
            FloatTensor gpuTensor = tryCreateGpuTensor(type, data, elementCount);
            if (gpuTensor != null) return gpuTensor;
        }

        // Try SIMD-optimized tensors (Java 21+ with Vector API and MemorySegment)
        FloatTensor simdTensor = tryCreateSimdTensor(type, data, elementCount);
        if (simdTensor != null) return simdTensor;

        // CPU fallback
        if (type == GGMLType.F32) return new F32FloatTensor(data, elementCount);
        if (type == GGMLType.F16) return new F16FloatTensor(data, elementCount);
        if (type == GGMLType.BF16) return new BF16FloatTensor(data, elementCount);
        if (type == GGMLType.Q4_0) return new Q4_0FloatTensor(data, elementCount);
        if (type == GGMLType.Q5_0) return new Q5_0FloatTensor(data, elementCount);
        if (type == GGMLType.Q8_0) return new Q8_0FloatTensor(data, elementCount);
        if (type == GGMLType.Q4_K) return new Q4_KFloatTensor(data, elementCount);
        if (type == GGMLType.Q3_K) return new Q3_KFloatTensor(data, elementCount);
        if (type == GGMLType.Q6_K) return new Q6_KFloatTensor(data, elementCount);
        if (type == GGMLType.Q5_K) return new Q5_KFloatTensor(data, elementCount);
        if (type == GGMLType.Q2_K) return new Q2_KFloatTensor(data, elementCount);
        if (type == GGMLType.IQ4_NL) return new IQ4_NLFloatTensor(data, elementCount);
        if (type == GGMLType.IQ4_XS) return new IQ4_XSFloatTensor(data, elementCount);
        if (type == GGMLType.IQ3_XXS) return new IQ3_XXSFloatTensor(data, elementCount);
        if (type == GGMLType.IQ3_S) return new IQ3_SFloatTensor(data, elementCount);
        if (type == GGMLType.IQ2_S) return new IQ2_SFloatTensor(data, elementCount);
        if (type == GGMLType.MXFP4) return new MXFP4FloatTensor(data, elementCount);
        throw new UnsupportedOperationException("Unsupported tensor type: " + type);
    }

    private static FloatTensor tryCreateSimdTensor(GGMLType type, TensorData data, long elementCount) {
        Boolean avail = simdAvailable;
        if (avail != null && !avail) return null;

        // Only these types have SIMD variants
        if (type != GGMLType.Q4_K && type != GGMLType.Q8_0
            && type != GGMLType.Q6_K && type != GGMLType.Q5_0
            && type != GGMLType.Q5_K && type != GGMLType.Q3_K
            && type != GGMLType.IQ4_NL && type != GGMLType.IQ4_XS
            && type != GGMLType.IQ3_XXS && type != GGMLType.IQ3_S && type != GGMLType.IQ2_S) return null;

        // Probe SIMD availability on first call
        if (avail == null) {
            probeSimdTensors();
            avail = simdAvailable;
            if (!avail) return null;
        }

        try {
            Constructor<?> ctor = null;
            if (type == GGMLType.Q4_K) ctor = simdQ4KCtor;
            else if (type == GGMLType.Q8_0) ctor = simdQ8_0Ctor;
            else if (type == GGMLType.Q6_K) ctor = simdQ6KCtor;
            else if (type == GGMLType.Q5_0) ctor = simdQ5_0Ctor;
            else if (type == GGMLType.Q5_K) ctor = simdQ5KCtor;
            else if (type == GGMLType.Q3_K) ctor = simdQ3KCtor;
            else if (type == GGMLType.IQ4_NL) ctor = simdIQ4NLCtor;
            else if (type == GGMLType.IQ4_XS) ctor = simdIQ4XSCtor;
            else if (type == GGMLType.IQ3_XXS) ctor = simdIQ3XXSCtor;
            else if (type == GGMLType.IQ3_S) ctor = simdIQ3SCtor;
            else if (type == GGMLType.IQ2_S) ctor = simdIQ2SCtor;
            if (ctor == null) return null;
            return (FloatTensor) ctor.newInstance(data, elementCount);
        } catch (Exception e) {
            return null;
        }
    }

    private static synchronized void probeSimdTensors() {
        if (simdAvailable != null) return;
        String base = "it.denzosoft.llmplayer.tensor.";
        try {
            // Check that MemorySegmentTensorData is available (Java 21+)
            Class.forName("it.denzosoft.llmplayer.tensor.MemorySegmentTensorData");

            simdQ4KCtor = probeCtor(base + "SimdQ4_KFloatTensor");
            simdQ8_0Ctor = probeCtor(base + "SimdQ8_0FloatTensor");
            simdQ6KCtor = probeCtor(base + "SimdQ6_KFloatTensor");
            simdQ5_0Ctor = probeCtor(base + "SimdQ5_0FloatTensor");
            simdQ5KCtor = probeCtor(base + "SimdQ5_KFloatTensor");
            simdQ3KCtor = probeCtor(base + "SimdQ3_KFloatTensor");
            simdIQ4NLCtor = probeCtor(base + "SimdIQ4_NLFloatTensor");
            simdIQ4XSCtor = probeCtor(base + "SimdIQ4_XSFloatTensor");
            simdIQ3XXSCtor = probeCtor(base + "SimdIQ3_XXSFloatTensor");
            simdIQ3SCtor = probeCtor(base + "SimdIQ3_SFloatTensor");
            simdIQ2SCtor = probeCtor(base + "SimdIQ2_SFloatTensor");

            simdAvailable = (simdQ4KCtor != null || simdQ8_0Ctor != null || simdQ6KCtor != null
                || simdQ5_0Ctor != null || simdQ5KCtor != null || simdQ3KCtor != null
                || simdIQ4NLCtor != null || simdIQ4XSCtor != null
                || simdIQ3XXSCtor != null || simdIQ3SCtor != null || simdIQ2SCtor != null);
            if (simdAvailable) {
                StringBuilder sb = new StringBuilder("  SIMD tensors:");
                if (simdQ4KCtor != null) sb.append(" Q4_K");
                if (simdQ8_0Ctor != null) sb.append(" Q8_0");
                if (simdQ6KCtor != null) sb.append(" Q6_K");
                if (simdQ5_0Ctor != null) sb.append(" Q5_0");
                if (simdQ5KCtor != null) sb.append(" Q5_K");
                if (simdQ3KCtor != null) sb.append(" Q3_K");
                if (simdIQ4NLCtor != null) sb.append(" IQ4_NL");
                if (simdIQ4XSCtor != null) sb.append(" IQ4_XS");
                if (simdIQ3XXSCtor != null) sb.append(" IQ3_XXS");
                if (simdIQ3SCtor != null) sb.append(" IQ3_S");
                if (simdIQ2SCtor != null) sb.append(" IQ2_S");
                System.out.println(sb);
            }
        } catch (ClassNotFoundException e) {
            simdAvailable = Boolean.FALSE;
        }
    }

    private static Constructor<?> probeCtor(String className) {
        try {
            Class<?> cls = Class.forName(className);
            return cls.getConstructor(TensorData.class, long.class);
        } catch (Exception e) {
            return null;
        }
    }

    private static FloatTensor tryCreateGpuTensor(GGMLType type, TensorData data, long elementCount) {
        String className = gpuTensorClassName(type);
        if (className == null) return null;

        try {
            Class<?> gpuClass = Class.forName(className);
            String bufMgrClassName = "cuda".equals(gpuBackend)
                ? "it.denzosoft.llmplayer.gpu.CudaBufferManager"
                : "it.denzosoft.llmplayer.gpu.GpuBufferManager";
            Class<?> bufMgrClass = Class.forName(bufMgrClassName);
            Constructor<?> ctor = gpuClass.getConstructor(TensorData.class, long.class, bufMgrClass);
            return (FloatTensor) ctor.newInstance(data, elementCount, gpuBufferManager);
        } catch (Exception e) {
            // GPU tensor class not available, fall through to CPU
            return null;
        }
    }

    private static String gpuTensorClassName(GGMLType type) {
        String base = "it.denzosoft.llmplayer.tensor.";
        boolean cuda = "cuda".equals(gpuBackend);
        if (type == GGMLType.F32) return base + (cuda ? "F32CudaTensor" : "F32GpuTensor");
        if (type == GGMLType.BF16 && cuda) return base + "BF16CudaTensor";
        if (type == GGMLType.F16 && cuda) return base + "F16CudaTensor";
        if (type == GGMLType.Q3_K) return base + (cuda ? "Q3_KCudaTensor" : "Q3_KGpuTensor");
        if (type == GGMLType.Q4_K) return base + (cuda ? "Q4_KCudaTensor" : "Q4_KGpuTensor");
        if (type == GGMLType.Q5_K) return base + (cuda ? "Q5_KCudaTensor" : "Q5_KGpuTensor");
        if (type == GGMLType.Q6_K) return base + (cuda ? "Q6_KCudaTensor" : "Q6_KGpuTensor");
        if (type == GGMLType.Q8_0) return base + (cuda ? "Q8_0CudaTensor" : "Q8_0GpuTensor");
        if (type == GGMLType.Q4_0) return base + (cuda ? "Q4_0CudaTensor" : "Q4_0GpuTensor");
        if (type == GGMLType.Q5_0 && cuda) return base + "Q5_0CudaTensor";
        if (type == GGMLType.IQ4_NL && cuda) return base + "IQ4_NLCudaTensor";
        if (type == GGMLType.IQ4_XS && cuda) return base + "IQ4_XSCudaTensor";
        if (type == GGMLType.IQ3_XXS && cuda) return base + "IQ3_XXSCudaTensor";
        if (type == GGMLType.IQ3_S && cuda) return base + "IQ3_SCudaTensor";
        if (type == GGMLType.IQ2_S && cuda) return base + "IQ2_SCudaTensor";
        return null; // No GPU version for this type
    }
}
