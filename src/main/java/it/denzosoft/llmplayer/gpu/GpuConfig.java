package it.denzosoft.llmplayer.gpu;

/**
 * GPU configuration (Java 8 compatible).
 * Actual OpenCL operations are in java21/ sources loaded via reflection.
 */
public class GpuConfig {

    private boolean enabled;
    private int deviceId;
    private boolean listDevices;
    private int gpuLayers = -1; // -1 = auto-detect, 0 = all layers on GPU, N = first N layers on GPU
    private boolean moeOptimized = false; // MoE: attention on GPU, experts on CPU

    public GpuConfig() {
        this.enabled = false;
        this.deviceId = 0;
        this.listDevices = false;
        this.gpuLayers = -1;
        this.moeOptimized = false;
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
}
