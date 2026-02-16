package it.denzosoft.llmplayer.gpu;

/**
 * Information about an OpenCL device (GPU, NPU, or CPU).
 */
public record DeviceInfo(
    int index,
    long deviceId,
    String name,
    String vendor,
    long globalMemory,
    int computeUnits,
    String deviceType,
    long maxWorkGroupSize
) {
    @Override
    public String toString() {
        long memMB = globalMemory / (1024 * 1024);
        return String.format("[%d] %s (%s) - %s, %d MB, %d CUs",
            index, name, vendor, deviceType, memMB, computeUnits);
    }
}
