package it.denzosoft.llmplayer.gpu;

/**
 * GPU configuration (Java 8 compatible).
 * Actual GPU operations are in java21/ sources loaded via reflection.
 * Supports CUDA and OpenCL backends.
 */
public class GpuConfig {

    public enum GpuBackend { AUTO, CUDA, OPENCL }

    private boolean enabled;
    private int deviceId;
    private boolean listDevices;
    private int gpuLayers = -1; // -1 = auto-detect, 0 = all layers on GPU, N = first N layers on GPU
    private boolean moeOptimized = false; // MoE: attention on GPU, experts on CPU
    private GpuBackend backend = GpuBackend.AUTO;
    private String memoryMode = "device"; // device, managed, host-mapped

    public GpuConfig() {
        this.enabled = false;
        this.deviceId = 0;
        this.listDevices = false;
        this.gpuLayers = -1;
        this.moeOptimized = false;
        this.backend = GpuBackend.AUTO;
    }

    public boolean isEnabled() { return enabled; }
    public void setEnabled(boolean enabled) { this.enabled = enabled; }

    public int getDeviceId() { return deviceId; }
    public void setDeviceId(int deviceId) { this.deviceId = deviceId; }

    public boolean isListDevices() { return listDevices; }
    public void setListDevices(boolean listDevices) { this.listDevices = listDevices; }

    public int getGpuLayers() { return gpuLayers; }
    public void setGpuLayers(int gpuLayers) { this.gpuLayers = gpuLayers; }

    public boolean isMoeOptimized() { return moeOptimized; }
    public void setMoeOptimized(boolean moeOptimized) { this.moeOptimized = moeOptimized; }

    public GpuBackend getBackend() { return backend; }
    public void setBackend(GpuBackend backend) { this.backend = backend; }

    public String getMemoryMode() { return memoryMode; }
    public void setMemoryMode(String memoryMode) { this.memoryMode = memoryMode; }
}
