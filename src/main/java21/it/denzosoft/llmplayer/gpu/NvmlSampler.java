package it.denzosoft.llmplayer.gpu;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

/**
 * Minimal NVML binding (Panama FFM, {@code libnvidia-ml.so.1}) for reading the GPU's performance
 * state, clocks and power from inside the process. Used by {@link CudaClockKeeper} to report the
 * P-state histogram of a generation: sampling nvidia-smi from a script at 2 s cannot resolve a
 * decode's bursts, and spawning it from the JVM costs far more than an NVML call.
 *
 * <p>Every method fails soft: {@link #open} returns null when the library or the device is not
 * available (no NVIDIA driver, a container without NVML), and a failed sample returns false.
 */
final class NvmlSampler {

    static final int NVML_CLOCK_SM = 1, NVML_CLOCK_MEM = 2;

    private final Arena arena = Arena.ofShared();
    private final MemorySegment device;
    private final MethodHandle getPState, getClock, getPower;
    private final MemorySegment intBuf;

    private NvmlSampler(MemorySegment device, MethodHandle getPState, MethodHandle getClock, MethodHandle getPower) {
        this.device = device;
        this.getPState = getPState;
        this.getClock = getClock;
        this.getPower = getPower;
        this.intBuf = arena.allocate(ValueLayout.JAVA_INT);
    }

    /** NVML handle for device {@code index}, or null when NVML is not usable. */
    static NvmlSampler open(int index) {
        try {
            SymbolLookup lib;
            try {
                lib = SymbolLookup.libraryLookup("libnvidia-ml.so.1", Arena.global());
            } catch (IllegalArgumentException e) {
                lib = SymbolLookup.libraryLookup("libnvidia-ml.so", Arena.global());
            }
            Linker linker = Linker.nativeLinker();
            MethodHandle init = linker.downcallHandle(lib.find("nvmlInit_v2").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT));
            MethodHandle byIndex = linker.downcallHandle(lib.find("nvmlDeviceGetHandleByIndex_v2").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            MethodHandle pstate = linker.downcallHandle(lib.find("nvmlDeviceGetPerformanceState").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            MethodHandle clock = linker.downcallHandle(lib.find("nvmlDeviceGetClockInfo").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            MethodHandle power = linker.downcallHandle(lib.find("nvmlDeviceGetPowerUsage").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            if ((int) init.invokeExact() != 0) return null;
            MemorySegment h = Arena.global().allocate(ValueLayout.ADDRESS);
            if ((int) byIndex.invokeExact(index, h) != 0) return null;
            return new NvmlSampler(h.get(ValueLayout.ADDRESS, 0), pstate, clock, power);
        } catch (Throwable e) {
            return null;
        }
    }

    /**
     * One sample into {@code out}: [P-state (0..15, -1 unknown), SM MHz, memory MHz, power mW].
     * Returns false when any query fails.
     */
    synchronized boolean sample(int[] out) {
        try {
            if ((int) getPState.invokeExact(device, intBuf) != 0) return false;
            int ps = intBuf.get(ValueLayout.JAVA_INT, 0);
            out[0] = ps >= 0 && ps < 16 ? ps : -1;
            out[1] = (int) getClock.invokeExact(device, NVML_CLOCK_SM, intBuf) == 0 ? intBuf.get(ValueLayout.JAVA_INT, 0) : -1;
            out[2] = (int) getClock.invokeExact(device, NVML_CLOCK_MEM, intBuf) == 0 ? intBuf.get(ValueLayout.JAVA_INT, 0) : -1;
            out[3] = (int) getPower.invokeExact(device, intBuf) == 0 ? intBuf.get(ValueLayout.JAVA_INT, 0) : -1;
            return true;
        } catch (Throwable e) {
            return false;
        }
    }
}
