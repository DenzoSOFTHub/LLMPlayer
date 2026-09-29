package it.denzosoft.llmplayer.inference;

/**
 * Implemented by GPU forward passes to say whether they keep the KV cache (and any recurrent
 * state) on the device. Passes that do run the attention against the CPU-side cache of the
 * {@code InferenceState} (the OpenCL pass) return false, so any state can resume on them.
 */
public interface GpuKvOwner {
    boolean ownsKvCache();
}
