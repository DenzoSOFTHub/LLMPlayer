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

        // Only Q4_K and Q8_0 have SIMD variants
        if (type != GGMLType.Q4_K && type != GGMLType.Q8_0) return null;

        // Probe SIMD availability on first call
        if (avail == null) {
            probeSimdTensors();
            avail = simdAvailable;
            if (!avail) return null;
        }

        try {
            Constructor<?> ctor = (type == GGMLType.Q4_K) ? simdQ4KCtor : simdQ8_0Ctor;
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

            try {
                Class<?> cls = Class.forName(base + "SimdQ4_KFloatTensor");
                simdQ4KCtor = cls.getConstructor(TensorData.class, long.class);
            } catch (Exception ignored) {}

            try {
                Class<?> cls = Class.forName(base + "SimdQ8_0FloatTensor");
                simdQ8_0Ctor = cls.getConstructor(TensorData.class, long.class);
            } catch (Exception ignored) {}

            simdAvailable = (simdQ4KCtor != null || simdQ8_0Ctor != null);
            if (simdAvailable) {
                System.out.println("  SIMD tensors: " +
                    (simdQ4KCtor != null ? "Q4_K " : "") +
                    (simdQ8_0Ctor != null ? "Q8_0" : ""));
            }
        } catch (ClassNotFoundException e) {
            simdAvailable = Boolean.FALSE;
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
        return null; // No GPU version for this type
    }
}
