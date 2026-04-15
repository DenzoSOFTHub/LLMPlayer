package it.denzosoft.llmplayer.gpu;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.foreign.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Manages CUDA context lifecycle: device enumeration, context/stream creation,
 * NVRTC kernel compilation, and buffer management.
 * Mirrors OpenCLContext for the CUDA backend.
 */
public class CudaContext implements AutoCloseable {

    private final Arena arena;
    private final int device;         // CUdevice (int ordinal)
    private final MemorySegment ctx;  // CUcontext pointer
    private final MemorySegment stream; // CUstream pointer
    private final DeviceInfo deviceInfo;
    private final int computeCapabilityMajor;
    private final int computeCapabilityMinor;
    private final Map<String, MemorySegment> functionCache = new ConcurrentHashMap<>(); // CUfunction
    private final Map<String, MemorySegment> moduleCache = new ConcurrentHashMap<>();   // CUmodule
    private volatile boolean closed = false;

    // Option A: prefer pre-compiled cubin/PTX over NVRTC JIT when available.
    // Opt-in so default behavior is unchanged; toggled via -Dcuda.prebuilt=true.
    private static final boolean USE_PREBUILT =
        "true".equals(System.getProperty("cuda.prebuilt", "false"));
    private static final boolean PREBUILT_VERBOSE =
        "true".equals(System.getProperty("cuda.prebuilt.verbose", "false"));

    private CudaContext(Arena arena, int device, MemorySegment ctx, MemorySegment stream,
                        DeviceInfo deviceInfo, int ccMajor, int ccMinor) {
        this.arena = arena;
        this.device = device;
        this.ctx = ctx;
        this.stream = stream;
        this.deviceInfo = deviceInfo;
        this.computeCapabilityMajor = ccMajor;
        this.computeCapabilityMinor = ccMinor;
    }

    /**
     * Enumerate all CUDA devices.
     */
    public static List<DeviceInfo> enumerateDevices() {
        List<DeviceInfo> result = new ArrayList<>();
        if (!CudaBindings.isCudaAvailable()) return result;

        try (Arena temp = Arena.ofConfined()) {
            int err = CudaBindings.init(0);
            if (err != CudaBindings.CUDA_SUCCESS) return result;

            MemorySegment countBuf = temp.allocate(ValueLayout.JAVA_INT);
            err = CudaBindings.deviceGetCount(countBuf);
            if (err != CudaBindings.CUDA_SUCCESS) return result;
            int count = countBuf.get(ValueLayout.JAVA_INT, 0);

            for (int i = 0; i < count; i++) {
                MemorySegment devBuf = temp.allocate(ValueLayout.JAVA_INT);
                err = CudaBindings.deviceGet(devBuf, i);
                if (err != CudaBindings.CUDA_SUCCESS) continue;
                int dev = devBuf.get(ValueLayout.JAVA_INT, 0);

                // Get name
                MemorySegment nameBuf = temp.allocate(256);
                CudaBindings.deviceGetName(nameBuf, 256, dev);
                String name = nameBuf.getString(0).trim();

                // Get total memory
                MemorySegment memBuf = temp.allocate(ValueLayout.JAVA_LONG);
                CudaBindings.deviceTotalMem(memBuf, dev);
                long totalMem = memBuf.get(ValueLayout.JAVA_LONG, 0);

                // Get SM count
                MemorySegment attrBuf = temp.allocate(ValueLayout.JAVA_INT);
                CudaBindings.deviceGetAttribute(attrBuf, CudaBindings.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
                int smCount = attrBuf.get(ValueLayout.JAVA_INT, 0);

                // Get max threads per block
                CudaBindings.deviceGetAttribute(attrBuf, CudaBindings.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev);
                long maxThreadsPerBlock = attrBuf.get(ValueLayout.JAVA_INT, 0);

                result.add(new DeviceInfo(i, dev, name, "NVIDIA", totalMem, smCount, "CUDA GPU", maxThreadsPerBlock));
            }
        } catch (Exception e) {
            // CUDA not functional
        }
        return result;
    }

    /**
     * Create a CUDA context for the device at the given index.
     */
    public static CudaContext create(int deviceIndex) {
        if (!CudaBindings.isAvailable()) {
            throw new RuntimeException("CUDA library not available");
        }

        Arena arena = Arena.ofShared();
        try {
            checkError(CudaBindings.init(0), "cuInit");

            MemorySegment devBuf = arena.allocate(ValueLayout.JAVA_INT);
            checkError(CudaBindings.deviceGet(devBuf, deviceIndex), "cuDeviceGet");
            int dev = devBuf.get(ValueLayout.JAVA_INT, 0);

            // Get device name
            MemorySegment nameBuf = arena.allocate(256);
            CudaBindings.deviceGetName(nameBuf, 256, dev);
            String name = nameBuf.getString(0).trim();

            // Get total memory
            MemorySegment memBuf = arena.allocate(ValueLayout.JAVA_LONG);
            CudaBindings.deviceTotalMem(memBuf, dev);
            long totalMem = memBuf.get(ValueLayout.JAVA_LONG, 0);

            // Get SM count
            MemorySegment attrBuf = arena.allocate(ValueLayout.JAVA_INT);
            CudaBindings.deviceGetAttribute(attrBuf, CudaBindings.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
            int smCount = attrBuf.get(ValueLayout.JAVA_INT, 0);

            // Get max threads per block
            CudaBindings.deviceGetAttribute(attrBuf, CudaBindings.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev);
            long maxThreadsPerBlock = attrBuf.get(ValueLayout.JAVA_INT, 0);

            // Get compute capability
            CudaBindings.deviceGetAttribute(attrBuf, CudaBindings.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
            int ccMajor = attrBuf.get(ValueLayout.JAVA_INT, 0);
            CudaBindings.deviceGetAttribute(attrBuf, CudaBindings.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
            int ccMinor = attrBuf.get(ValueLayout.JAVA_INT, 0);

            DeviceInfo info = new DeviceInfo(deviceIndex, dev, name, "NVIDIA", totalMem, smCount, "CUDA GPU", maxThreadsPerBlock);

            // Create context
            MemorySegment ctxBuf = arena.allocate(ValueLayout.ADDRESS);
            checkError(CudaBindings.ctxCreate(ctxBuf, 0, dev), "cuCtxCreate");
            MemorySegment ctx = ctxBuf.get(ValueLayout.ADDRESS, 0);

            // Create stream
            MemorySegment streamBuf = arena.allocate(ValueLayout.ADDRESS);
            checkError(CudaBindings.streamCreate(streamBuf, 0), "cuStreamCreate");
            MemorySegment strm = streamBuf.get(ValueLayout.ADDRESS, 0);

            return new CudaContext(arena, dev, ctx, strm, info, ccMajor, ccMinor);
        } catch (Exception e) {
            arena.close();
            throw e;
        }
    }

    /**
     * Compile a kernel from a .cu resource file using NVRTC, caching the result.
     * Returns the CUfunction handle.
     */
    public MemorySegment compileKernel(String resourcePath, String kernelName) {
        String cacheKey = resourcePath + ":" + kernelName;
        MemorySegment cached = functionCache.get(cacheKey);
        if (cached != null) return cached;

        synchronized (this) {
            cached = functionCache.get(cacheKey);
            if (cached != null) return cached;

            // Check if module already compiled for this resource
            MemorySegment module = moduleCache.get(resourcePath);
            if (module == null) {
                // Option A (PTX pre-compiled): try prebuilt cubin/ptx before NVRTC.
                // Opt-in via -Dcuda.prebuilt=true; default falls back to NVRTC.
                if (USE_PREBUILT) {
                    module = tryLoadPrebuiltModule(resourcePath);
                }
                if (module == null) {
                    String source = loadResource(resourcePath);
                    if (source == null) {
                        throw new RuntimeException("Kernel resource not found: " + resourcePath);
                    }
                    module = compileSourceToModule(source, resourcePath);
                }
                moduleCache.put(resourcePath, module);
            }

            // Get function from module
            MemorySegment funcBuf = arena.allocate(ValueLayout.ADDRESS);
            MemorySegment nameStr = arena.allocateFrom(kernelName);
            checkError(CudaBindings.moduleGetFunction(funcBuf, module, nameStr), "cuModuleGetFunction: " + kernelName);
            MemorySegment func = funcBuf.get(ValueLayout.ADDRESS, 0);

            functionCache.put(cacheKey, func);
            return func;
        }
    }

    private MemorySegment compileSourceToModule(String source, String resourcePath) {
        try (Arena temp = Arena.ofConfined()) {
            // Create NVRTC program
            MemorySegment progBuf = temp.allocate(ValueLayout.ADDRESS);
            MemorySegment srcStr = temp.allocateFrom(source);
            MemorySegment progName = temp.allocateFrom(resourcePath);
            checkError(CudaBindings.createProgram(progBuf, srcStr, progName,
                0, MemorySegment.NULL, MemorySegment.NULL), "nvrtcCreateProgram");
            MemorySegment prog = progBuf.get(ValueLayout.ADDRESS, 0);

            try {
                // Compile with architecture flag + fast math (enables FMA, fast div/sqrt, flush-to-zero)
                String archOpt = "--gpu-architecture=compute_" + computeCapabilityMajor + computeCapabilityMinor;
                String fastMathOpt = "--use_fast_math";
                MemorySegment archOptStr = temp.allocateFrom(archOpt);
                MemorySegment fastMathStr = temp.allocateFrom(fastMathOpt);
                MemorySegment optionsArray = temp.allocate(ValueLayout.ADDRESS, 2);
                optionsArray.setAtIndex(ValueLayout.ADDRESS, 0, archOptStr);
                optionsArray.setAtIndex(ValueLayout.ADDRESS, 1, fastMathStr);

                int compileResult = CudaBindings.compileProgram(prog, 2, optionsArray);
                if (compileResult != CudaBindings.CUDA_SUCCESS) {
                    String log = getNvrtcLog(prog, temp);
                    throw new RuntimeException("NVRTC compile failed (" + compileResult + "):\n" + log);
                }

                // Get PTX
                MemorySegment ptxSizeBuf = temp.allocate(ValueLayout.JAVA_LONG);
                checkError(CudaBindings.getPTXSize(prog, ptxSizeBuf), "nvrtcGetPTXSize");
                long ptxSize = ptxSizeBuf.get(ValueLayout.JAVA_LONG, 0);

                MemorySegment ptxBuf = temp.allocate(ptxSize);
                checkError(CudaBindings.getPTX(prog, ptxBuf), "nvrtcGetPTX");

                // Load module from PTX
                MemorySegment moduleBuf = arena.allocate(ValueLayout.ADDRESS);
                checkError(CudaBindings.moduleLoadDataEx(moduleBuf, ptxBuf, 0,
                    MemorySegment.NULL, MemorySegment.NULL), "cuModuleLoadDataEx");
                return moduleBuf.get(ValueLayout.ADDRESS, 0);
            } finally {
                CudaBindings.destroyProgram(progBuf);
            }
        }
    }

    /**
     * Try to load a pre-compiled cubin or PTX binary from resources, avoiding NVRTC entirely.
     * Lookup order for resource `kernels/cuda/foo.cu` on CC=8.9:
     *   1. kernels/cuda/prebuilt/foo.sm89.cubin  (sm-matched SASS, no driver JIT)
     *   2. kernels/cuda/prebuilt/foo.sm89.ptx    (virtual PTX, driver JITs)
     * Returns the CUmodule pointer, or null if no prebuilt artifact is available.
     */
    private MemorySegment tryLoadPrebuiltModule(String resourcePath) {
        // Derive "<dir>/prebuilt/<base>.sm<CC>.<ext>" from "<dir>/<base>.cu"
        int slash = resourcePath.lastIndexOf('/');
        int dot = resourcePath.lastIndexOf('.');
        if (slash < 0 || dot < 0 || dot <= slash) return null;
        String dir = resourcePath.substring(0, slash);
        String base = resourcePath.substring(slash + 1, dot);
        String cc = "sm" + computeCapabilityMajor + computeCapabilityMinor;
        String cubinPath = dir + "/prebuilt/" + base + "." + cc + ".cubin";
        String ptxPath   = dir + "/prebuilt/" + base + "." + cc + ".ptx";

        byte[] bin = loadResourceBytes(cubinPath);
        String chosen = cubinPath;
        if (bin == null) {
            bin = loadResourceBytes(ptxPath);
            chosen = ptxPath;
        }
        if (bin == null) return null;

        try (Arena temp = Arena.ofConfined()) {
            // Copy bytes into a native segment. PTX is NUL-terminated text; cubin is ELF binary.
            // Allocate +1 so PTX path is guaranteed NUL-terminated (driver expects C string for PTX).
            MemorySegment buf = temp.allocate(bin.length + 1L);
            MemorySegment.copy(bin, 0, buf, ValueLayout.JAVA_BYTE, 0, bin.length);
            buf.set(ValueLayout.JAVA_BYTE, bin.length, (byte) 0);

            MemorySegment moduleBuf = arena.allocate(ValueLayout.ADDRESS);
            int err = CudaBindings.moduleLoadDataEx(moduleBuf, buf, 0,
                MemorySegment.NULL, MemorySegment.NULL);
            if (err != CudaBindings.CUDA_SUCCESS) {
                if (PREBUILT_VERBOSE) {
                    System.err.println("[cuda.prebuilt] cuModuleLoadDataEx failed (" + err
                        + ") for " + chosen + " — falling back to NVRTC");
                }
                return null;
            }
            if (PREBUILT_VERBOSE) {
                System.err.println("[cuda.prebuilt] loaded " + chosen + " (" + bin.length + " bytes)");
            }
            return moduleBuf.get(ValueLayout.ADDRESS, 0);
        } catch (Exception e) {
            if (PREBUILT_VERBOSE) {
                System.err.println("[cuda.prebuilt] exception loading " + chosen + ": " + e);
            }
            return null;
        }
    }

    private static byte[] loadResourceBytes(String path) {
        try (InputStream is = CudaContext.class.getClassLoader().getResourceAsStream(path)) {
            if (is == null) return null;
            return is.readAllBytes();
        } catch (IOException e) {
            return null;
        }
    }

    private String getNvrtcLog(MemorySegment prog, Arena temp) {
        try {
            MemorySegment logSizeBuf = temp.allocate(ValueLayout.JAVA_LONG);
            CudaBindings.getProgramLogSize(prog, logSizeBuf);
            long logSize = logSizeBuf.get(ValueLayout.JAVA_LONG, 0);
            if (logSize <= 1) return "(no log)";
            MemorySegment logBuf = temp.allocate(logSize);
            CudaBindings.getProgramLog(prog, logBuf);
            return logBuf.getString(0);
        } catch (Exception e) {
            return "(failed to get log: " + e.getMessage() + ")";
        }
    }

    /**
     * Pre-compile all known CUDA kernels to avoid compilation stalls during inference.
     */
    public void precompileKernels() {
        String[][] kernels = {
            {"kernels/cuda/matmul_f32.cu", "matmul_f32"},
            {"kernels/cuda/matmul_q4_0.cu", "matmul_q4_0"},
            {"kernels/cuda/matmul_q4_k.cu", "matmul_q4_k"},
            {"kernels/cuda/matmul_q5_k.cu", "matmul_q5_k"},
            {"kernels/cuda/matmul_q6_k.cu", "matmul_q6_k"},
            {"kernels/cuda/matmul_q8_0.cu", "matmul_q8_0"},
            {"kernels/cuda/matmul_q3_k.cu", "matmul_q3_k"},
            {"kernels/cuda/rmsnorm.cu", "rmsnorm_fused"},
            {"kernels/cuda/rmsnorm.cu", "rmsnorm_sumsq"},
            {"kernels/cuda/rmsnorm.cu", "rmsnorm_normalize"},
            {"kernels/cuda/softmax.cu", "softmax_max"},
            {"kernels/cuda/softmax.cu", "softmax_exp_sum"},
            {"kernels/cuda/softmax.cu", "softmax_normalize"},
            {"kernels/cuda/silu.cu", "silu"},
            {"kernels/cuda/silu_mul.cu", "silu_mul"},
            {"kernels/cuda/rope.cu", "rope_apply"},
            {"kernels/cuda/attention.cu", "attention_full"},
            {"kernels/cuda/attention.cu", "kv_cache_update"},
            {"kernels/cuda/saxpy.cu", "saxpy"},
            {"kernels/cuda/accumulate.cu", "accumulate"},
            {"kernels/cuda/elementwise_mul.cu", "elementwise_mul"},
            {"kernels/cuda/fill_zero.cu", "fill_zero"},
            {"kernels/cuda/matmul_q4_k_fused_gate_up.cu", "matmul_q4_k_fused_gate_up"},
            {"kernels/cuda/argmax.cu", "argmax_partial"},
            {"kernels/cuda/argmax.cu", "argmax_final"},
            {"kernels/cuda/matmul_mxfp4.cu", "matmul_mxfp4"},
        };
        for (String[] kv : kernels) {
            try {
                compileKernel(kv[0], kv[1]);
            } catch (Exception ignored) {
                // Some kernels may not exist — that's fine
            }
        }
    }

    /**
     * Allocate GPU buffer. Returns CUdeviceptr as long.
     */
    public long allocBuffer(long bytes) {
        MemorySegment dptrBuf = arena.allocate(ValueLayout.JAVA_LONG);
        checkError(CudaBindings.memAlloc(dptrBuf, bytes), "cuMemAlloc");
        return dptrBuf.get(ValueLayout.JAVA_LONG, 0);
    }

    /**
     * Free a GPU buffer.
     */
    public void freeBuffer(long dptr) {
        CudaBindings.memFree(dptr);
    }

    /**
     * Allocate unified/managed memory accessible from both CPU and GPU.
     * The CUDA driver automatically migrates pages between host RAM and VRAM on demand.
     * Returns a CUdeviceptr that can be used in kernels just like device memory.
     */
    public long allocManagedBuffer(long bytes) {
        MemorySegment dptrBuf = arena.allocate(ValueLayout.JAVA_LONG);
        checkError(CudaBindings.memAllocManaged(dptrBuf, bytes, CudaBindings.CU_MEM_ATTACH_GLOBAL),
            "cuMemAllocManaged");
        return dptrBuf.get(ValueLayout.JAVA_LONG, 0);
    }

    /**
     * Allocate host-mapped (zero-copy) memory: pinned host memory that the GPU
     * can access directly via PCIe without explicit copies.
     * Returns [hostPtr, devicePtr] where hostPtr is used for CPU writes
     * and devicePtr is used in GPU kernels.
     */
    public long[] allocHostMappedBuffer(long bytes) {
        MemorySegment ppBuf = arena.allocate(ValueLayout.ADDRESS);
        checkError(CudaBindings.memHostAlloc(ppBuf, bytes,
            CudaBindings.CU_MEMHOSTALLOC_DEVICEMAP | CudaBindings.CU_MEMHOSTALLOC_PORTABLE),
            "cuMemHostAlloc");
        MemorySegment hostPtr = ppBuf.get(ValueLayout.ADDRESS, 0);

        MemorySegment dptrBuf = arena.allocate(ValueLayout.JAVA_LONG);
        checkError(CudaBindings.memHostGetDevicePointer(dptrBuf, hostPtr, 0),
            "cuMemHostGetDevicePointer");
        long devicePtr = dptrBuf.get(ValueLayout.JAVA_LONG, 0);

        return new long[] { hostPtr.address(), devicePtr };
    }

    /**
     * Free host-mapped memory.
     */
    public void freeHostMappedBuffer(long hostPtrAddress) {
        CudaBindings.memFreeHost(MemorySegment.ofAddress(hostPtrAddress));
    }

    /**
     * Check if managed memory API is available.
     */
    public boolean isManagedMemoryAvailable() {
        return CudaBindings.isManagedMemoryAvailable();
    }

    /**
     * Check if host-mapped memory API is available.
     */
    public boolean isHostMappedMemoryAvailable() {
        return CudaBindings.isHostMappedMemoryAvailable();
    }

    /**
     * Get free and total GPU memory in bytes. Returns [free, total].
     */
    public long[] getMemoryInfo() {
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment freeBuf = temp.allocate(ValueLayout.JAVA_LONG);
            MemorySegment totalBuf = temp.allocate(ValueLayout.JAVA_LONG);
            checkError(CudaBindings.memGetInfo(freeBuf, totalBuf), "cuMemGetInfo");
            return new long[] { freeBuf.get(ValueLayout.JAVA_LONG, 0), totalBuf.get(ValueLayout.JAVA_LONG, 0) };
        }
    }

    /**
     * Copy host data to device (blocking).
     */
    public void writeBuffer(long dptr, MemorySegment hostData, long size) {
        checkError(CudaBindings.memcpyHtoD(dptr, hostData, size), "cuMemcpyHtoD");
    }

    /**
     * Copy host data to device (async on stream).
     */
    public void writeBufferAsync(long dptr, MemorySegment hostData, long size) {
        checkError(CudaBindings.memcpyHtoDAsync(dptr, hostData, size, stream), "cuMemcpyHtoDAsync");
    }

    /**
     * Copy device data to host (blocking).
     */
    public void readBuffer(long dptr, MemorySegment hostData, long size) {
        checkError(CudaBindings.memcpyDtoH(hostData, dptr, size), "cuMemcpyDtoH");
    }

    /**
     * Copy device data to device (async on stream).
     */
    public void copyBufferDtoD(long dst, long src, long sizeBytes) {
        checkError(CudaBindings.memcpyDtoDAsync(dst, src, sizeBytes, stream), "cuMemcpyDtoDAsync");
    }

    /**
     * Fill GPU buffer with zero (float 0.0f).
     */
    public void fillBufferZero(long dptr, long sizeBytes) {
        long numFloats = sizeBytes / Float.BYTES;
        checkError(CudaBindings.memsetD32(dptr, 0, numFloats), "cuMemsetD32");
    }

    /**
     * Synchronize the stream (wait for all pending operations).
     */
    public void finish() {
        checkError(CudaBindings.streamSynchronize(stream), "cuStreamSynchronize");
    }

    /**
     * Launch a 1D CUDA kernel on the default stream.
     */
    public void launchKernel1D(MemorySegment function, long globalSize, long blockSize,
                                int sharedMemBytes, MemorySegment kernelParams) {
        launchKernel1DOnStream(function, globalSize, blockSize, sharedMemBytes, kernelParams, stream);
    }

    /**
     * Launch a 1D CUDA kernel on a specific stream.
     */
    public void launchKernel1DOnStream(MemorySegment function, long globalSize, long blockSize,
                                        int sharedMemBytes, MemorySegment kernelParams, MemorySegment onStream) {
        int blockDim = (int) blockSize;
        int gridDim = (int) ((globalSize + blockDim - 1) / blockDim);
        checkError(CudaBindings.launchKernel(function,
            gridDim, 1, 1,
            blockDim, 1, 1,
            sharedMemBytes, onStream,
            kernelParams, MemorySegment.NULL), "cuLaunchKernel");
    }

    /**
     * Create an additional CUDA stream.
     */
    public MemorySegment createExtraStream() {
        MemorySegment streamBuf = arena.allocate(ValueLayout.ADDRESS);
        checkError(CudaBindings.streamCreate(streamBuf, 0), "cuStreamCreate");
        return streamBuf.get(ValueLayout.ADDRESS, 0);
    }

    /**
     * Synchronize a specific stream.
     */
    public void syncStream(MemorySegment targetStream) {
        checkError(CudaBindings.streamSynchronize(targetStream), "cuStreamSynchronize");
    }

    /**
     * Create a lightweight CUDA event (no timing).
     */
    public MemorySegment createEvent() {
        MemorySegment eventBuf = arena.allocate(ValueLayout.ADDRESS);
        checkError(CudaBindings.eventCreate(eventBuf, CudaBindings.CU_EVENT_DISABLE_TIMING), "cuEventCreate");
        return eventBuf.get(ValueLayout.ADDRESS, 0);
    }

    /**
     * Record an event on a stream: marks the point in the stream's queue.
     */
    public void recordEvent(MemorySegment event, MemorySegment onStream) {
        checkError(CudaBindings.eventRecord(event, onStream), "cuEventRecord");
    }

    /**
     * Make a stream wait for an event recorded on another stream.
     */
    public void streamWaitEvent(MemorySegment waitingStream, MemorySegment event) {
        checkError(CudaBindings.streamWaitEvent(waitingStream, event, 0), "cuStreamWaitEvent");
    }

    /**
     * Build a kernel params array (void** array of pointers to arguments).
     */
    public MemorySegment buildKernelParams(Arena tempArena, Object... args) {
        MemorySegment params = tempArena.allocate(ValueLayout.ADDRESS, args.length);
        for (int i = 0; i < args.length; i++) {
            Object arg = args[i];
            MemorySegment argMem;
            if (arg instanceof Long) {
                argMem = tempArena.allocateFrom(ValueLayout.JAVA_LONG, (Long) arg);
            } else if (arg instanceof Integer) {
                argMem = tempArena.allocateFrom(ValueLayout.JAVA_INT, (Integer) arg);
            } else if (arg instanceof Float) {
                argMem = tempArena.allocateFrom(ValueLayout.JAVA_FLOAT, (Float) arg);
            } else {
                throw new IllegalArgumentException("Unsupported kernel param type: " + arg.getClass());
            }
            params.setAtIndex(ValueLayout.ADDRESS, i, argMem);
        }
        return params;
    }

    // --- CUDA Graph API ---

    /**
     * Check if CUDA graph capture/replay is available.
     */
    public boolean isGraphApiAvailable() {
        return CudaBindings.isGraphApiAvailable();
    }

    /**
     * Begin capturing kernel launches on the default stream into a graph.
     */
    public void beginCapture() {
        checkError(CudaBindings.streamBeginCapture(stream, CudaBindings.CU_STREAM_CAPTURE_MODE_GLOBAL),
            "cuStreamBeginCapture");
    }

    /**
     * End stream capture and return the CUgraph handle.
     */
    public MemorySegment endCapture() {
        MemorySegment graphBuf = arena.allocate(ValueLayout.ADDRESS);
        checkError(CudaBindings.streamEndCapture(stream, graphBuf), "cuStreamEndCapture");
        return graphBuf.get(ValueLayout.ADDRESS, 0);
    }

    /**
     * Instantiate a graph into an executable graph for fast replay.
     */
    public MemorySegment instantiateGraph(MemorySegment graph) {
        MemorySegment graphExecBuf = arena.allocate(ValueLayout.ADDRESS);
        checkError(CudaBindings.graphInstantiateWithFlags(graphExecBuf, graph, 0L), "cuGraphInstantiateWithFlags");
        return graphExecBuf.get(ValueLayout.ADDRESS, 0);
    }

    /**
     * Launch an instantiated graph on the default stream.
     * All captured kernel launches are replayed in a single API call.
     */
    public void launchGraph(MemorySegment graphExec) {
        checkError(CudaBindings.graphLaunch(graphExec, stream), "cuGraphLaunch");
    }

    /**
     * Destroy an instantiated graph executable.
     */
    public void destroyGraphExec(MemorySegment graphExec) {
        CudaBindings.graphExecDestroy(graphExec);
    }

    /**
     * Destroy a graph.
     */
    public void destroyGraph(MemorySegment graph) {
        CudaBindings.graphDestroy(graph);
    }

    public DeviceInfo getDeviceInfo() { return deviceInfo; }
    public MemorySegment getStream() { return stream; }
    public Arena getArena() { return arena; }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        functionCache.clear();
        for (MemorySegment module : moduleCache.values()) {
            try { CudaBindings.moduleUnload(module); } catch (Exception ignored) {}
        }
        moduleCache.clear();
        try { CudaBindings.streamDestroy(stream); } catch (Exception ignored) {}
        try { CudaBindings.ctxDestroy(ctx); } catch (Exception ignored) {}
        arena.close();
    }

    private static String loadResource(String path) {
        try (InputStream is = CudaContext.class.getClassLoader().getResourceAsStream(path)) {
            if (is == null) return null;
            StringBuilder sb = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    sb.append(line).append('\n');
                }
            }
            return sb.toString();
        } catch (IOException e) {
            return null;
        }
    }

    private static void checkError(int err, String op) {
        if (err != CudaBindings.CUDA_SUCCESS) {
            throw new RuntimeException("CUDA error in " + op + ": " + err);
        }
    }
}
