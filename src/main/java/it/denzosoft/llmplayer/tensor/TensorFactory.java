package it.denzosoft.llmplayer.tensor;

import java.lang.reflect.Constructor;

public final class TensorFactory {

    private static volatile Object gpuBufferManager; // GpuBufferManager instance, set via reflection

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

    public static FloatTensor create(GGMLType type, TensorData data, long elementCount) {
        // Try GPU tensor first
        if (gpuBufferManager != null) {
            FloatTensor gpuTensor = tryCreateGpuTensor(type, data, elementCount);
            if (gpuTensor != null) return gpuTensor;
        }

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
        throw new UnsupportedOperationException("Unsupported tensor type: " + type);
    }

    private static FloatTensor tryCreateGpuTensor(GGMLType type, TensorData data, long elementCount) {
        String className = gpuTensorClassName(type);
        if (className == null) return null;

        try {
            Class<?> gpuClass = Class.forName(className);
            Class<?> bufMgrClass = Class.forName("it.denzosoft.llmplayer.gpu.GpuBufferManager");
            Constructor<?> ctor = gpuClass.getConstructor(TensorData.class, long.class, bufMgrClass);
            return (FloatTensor) ctor.newInstance(data, elementCount, gpuBufferManager);
        } catch (Exception e) {
            // GPU tensor class not available, fall through to CPU
            return null;
        }
    }

    private static String gpuTensorClassName(GGMLType type) {
        String base = "it.denzosoft.llmplayer.tensor.";
        if (type == GGMLType.F32) return base + "F32GpuTensor";
        if (type == GGMLType.Q3_K) return base + "Q3_KGpuTensor";
        if (type == GGMLType.Q4_K) return base + "Q4_KGpuTensor";
        if (type == GGMLType.Q5_K) return base + "Q5_KGpuTensor";
        if (type == GGMLType.Q6_K) return base + "Q6_KGpuTensor";
        if (type == GGMLType.Q8_0) return base + "Q8_0GpuTensor";
        if (type == GGMLType.Q4_0) return base + "Q4_0GpuTensor";
        return null; // No GPU version for this type
    }
}
