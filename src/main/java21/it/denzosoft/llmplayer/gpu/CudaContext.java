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

    // On-disk cache of NVRTC output (cubin, or PTX when NVRTC cannot target the device's SM).
    // Keyed by SHA-256 of source + options + NVRTC version, so an edited kernel, a new NVRTC or
    // another GPU never reuses a stale image. Disable with -Dcuda.kernel.cache=false.
    private static final boolean KERNEL_CACHE =
        !"false".equals(System.getProperty("cuda.kernel.cache", "true"));
    private static final String KERNEL_CACHE_DIR = System.getProperty("cuda.kernel.cache.dir",
        System.getProperty("user.home", ".") + "/.cache/llmplayer/cuda");

    /**
     * The context last made current on each thread by this class. A CUDA context is current per
     * thread; cuCtxCreate makes it current only on the creating thread, so a model loaded on one
     * thread (an HTTP worker, say) and used on another would fail every driver call with
     * CUDA_ERROR_INVALID_CONTEXT. Every entry point below calls {@link #ensureCurrent()}.
     */
    private static final ThreadLocal<CudaContext> CURRENT = new ThreadLocal<>();

    // NVRTC target, resolved once per context: "sm_XX" (cubin) or "compute_XX" (PTX).
    private volatile String nvrtcArch;
    private volatile boolean nvrtcCubin;
    // Largest dynamic shared memory a block may opt into (bytes); 0 until queried.
    private volatile int maxSharedOptin;

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

            // Create context. -Dcuda.sched=auto|spin|yield|blocking chooses how a blocking call
            // (stream/event synchronize, synchronous copies) waits: AUTO spins while the GPU is
            // expected to finish soon, BLOCKING_SYNC sleeps on an OS primitive (no core burned).
            MemorySegment ctxBuf = arena.allocate(ValueLayout.ADDRESS);
            checkError(CudaBindings.ctxCreate(ctxBuf, schedFlags(), dev), "cuCtxCreate");
            MemorySegment ctx = ctxBuf.get(ValueLayout.ADDRESS, 0);

            // Stream priorities (lower number = higher priority). The work stream gets the greatest
            // priority, so a lower-priority side stream (the clock keeper) can never delay the
            // model's kernels: the block scheduler dispatches pending blocks of the higher-priority
            // stream first. With plain cuStreamCreate both would be at 0, the least priority.
            int least = 0, greatest = 0;
            if (CudaBindings.isStreamPriorityAvailable()) {
                MemorySegment lb = arena.allocate(ValueLayout.JAVA_INT), gb = arena.allocate(ValueLayout.JAVA_INT);
                if (CudaBindings.streamPriorityRange(lb, gb) == CudaBindings.CUDA_SUCCESS) {
                    least = lb.get(ValueLayout.JAVA_INT, 0);
                    greatest = gb.get(ValueLayout.JAVA_INT, 0);
                }
            }

            // Create the work stream. NON_BLOCKING: it never synchronizes with the legacy NULL
            // stream, so an unrelated legacy-stream call can neither stall it nor invalidate a
            // graph capture on it (CUDA_ERROR_STREAM_CAPTURE_IMPLICIT, error 906). Every copy and
            // memset in this class is therefore issued on this stream, never on the NULL stream.
            MemorySegment streamBuf = arena.allocate(ValueLayout.ADDRESS);
            if (CudaBindings.isStreamPriorityAvailable() && greatest != least) {
                checkError(CudaBindings.streamCreateWithPriority(streamBuf, CudaBindings.CU_STREAM_NON_BLOCKING, greatest),
                    "cuStreamCreateWithPriority");
            } else {
                checkError(CudaBindings.streamCreate(streamBuf, CudaBindings.CU_STREAM_NON_BLOCKING), "cuStreamCreate");
            }
            MemorySegment strm = streamBuf.get(ValueLayout.ADDRESS, 0);

            CudaContext created = new CudaContext(arena, dev, ctx, strm, info, ccMajor, ccMinor);
            created.leastPriority = least;
            created.greatestPriority = greatest;
            CURRENT.set(created);
            // Reference buffers of the allocation guard first, while the device is empty (F7).
            created.vramGuard = VramGuard.create(created);
            return created;
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
        return compileKernel(resourcePath, kernelName, null);
    }

    /**
     * Like {@link #compileKernel(String, String)}, with a preamble of {@code #define} lines
     * prepended to the source (e.g. {@code "#define FA_G 4\n"}). Each distinct preamble is its
     * own module: a source file with many template instantiations can then be compiled one
     * instantiation at a time instead of all at once.
     */
    public MemorySegment compileKernel(String resourcePath, String kernelName, String defines) {
        String moduleKey = defines == null ? resourcePath : resourcePath + "|" + defines;
        String cacheKey = moduleKey + ":" + kernelName;
        MemorySegment cached = functionCache.get(cacheKey);
        if (cached != null) return cached;

        synchronized (this) {
            ensureCurrent();
            cached = functionCache.get(cacheKey);
            if (cached != null) return cached;

            // Check if module already compiled for this resource
            MemorySegment module = moduleCache.get(moduleKey);
            if (module == null) {
                // Option A (PTX pre-compiled): try prebuilt cubin/ptx before NVRTC.
                // Opt-in via -Dcuda.prebuilt=true; default falls back to NVRTC.
                if (USE_PREBUILT && defines == null) {
                    module = tryLoadPrebuiltModule(resourcePath);
                }
                if (module == null) {
                    String source = loadResource(resourcePath);
                    if (source == null) {
                        throw new RuntimeException("Kernel resource not found: " + resourcePath);
                    }
                    if (defines != null) source = defines + source;
                    module = compileSourceToModule(source, resourcePath);
                }
                moduleCache.put(moduleKey, module);
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

    /**
     * Compile {@code source} (generated at run time, e.g. a transformed kernel resource) under
     * {@code moduleKey} and return {@code kernelName}; cached like {@link #compileKernel}.
     */
    public MemorySegment compileKernelSource(String moduleKey, String source, String kernelName) {
        String cacheKey = moduleKey + ":" + kernelName;
        MemorySegment cached = functionCache.get(cacheKey);
        if (cached != null) return cached;
        synchronized (this) {
            ensureCurrent();
            cached = functionCache.get(cacheKey);
            if (cached != null) return cached;
            MemorySegment module = moduleCache.get(moduleKey);
            if (module == null) {
                module = compileSourceToModule(source, moduleKey);
                moduleCache.put(moduleKey, module);
            }
            MemorySegment funcBuf = arena.allocate(ValueLayout.ADDRESS);
            MemorySegment nameStr = arena.allocateFrom(kernelName);
            checkError(CudaBindings.moduleGetFunction(funcBuf, module, nameStr), "cuModuleGetFunction: " + kernelName);
            MemorySegment func = funcBuf.get(ValueLayout.ADDRESS, 0);
            functionCache.put(cacheKey, func);
            return func;
        }
    }

    /** The text of a kernel resource, or null. */
    public static String kernelSource(String resourcePath) {
        return loadResource(resourcePath);
    }

    private MemorySegment compileSourceToModule(String source, String resourcePath) {
        resolveNvrtcTarget();
        String archOpt = "--gpu-architecture=" + nvrtcArch;
        String fastMathOpt = "--use_fast_math";
        byte[] image = null;
        java.nio.file.Path cacheFile = null;
        if (KERNEL_CACHE) {
            cacheFile = kernelCachePath(source, archOpt + " " + fastMathOpt);
            if (cacheFile != null) {
                try {
                    if (java.nio.file.Files.isRegularFile(cacheFile)) image = java.nio.file.Files.readAllBytes(cacheFile);
                } catch (IOException ignored) {
                    image = null;
                }
            }
        }
        if (image != null) {
            MemorySegment module = loadModuleImage(image);
            if (module != null) return module;
            image = null; // corrupt or unloadable cache entry: recompile and overwrite it
        }

        try (Arena temp = Arena.ofConfined()) {
            // Create NVRTC program
            MemorySegment progBuf = temp.allocate(ValueLayout.ADDRESS);
            MemorySegment srcStr = temp.allocateFrom(source);
            MemorySegment progName = temp.allocateFrom(resourcePath);
            checkError(CudaBindings.createProgram(progBuf, srcStr, progName,
                0, MemorySegment.NULL, MemorySegment.NULL), "nvrtcCreateProgram");
            MemorySegment prog = progBuf.get(ValueLayout.ADDRESS, 0);

            try {
                // Architecture flag + fast math (enables FMA, fast div/sqrt, flush-to-zero)
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

                if (nvrtcCubin) {
                    // Real-architecture target: NVRTC already ran ptxas, so the driver loads the
                    // SASS directly instead of JIT-compiling PTX at every start-up.
                    MemorySegment sizeBuf = temp.allocate(ValueLayout.JAVA_LONG);
                    checkError(CudaBindings.getCUBINSize(prog, sizeBuf), "nvrtcGetCUBINSize");
                    long size = sizeBuf.get(ValueLayout.JAVA_LONG, 0);
                    MemorySegment buf = temp.allocate(size);
                    checkError(CudaBindings.getCUBIN(prog, buf), "nvrtcGetCUBIN");
                    image = buf.toArray(ValueLayout.JAVA_BYTE);
                } else {
                    MemorySegment ptxSizeBuf = temp.allocate(ValueLayout.JAVA_LONG);
                    checkError(CudaBindings.getPTXSize(prog, ptxSizeBuf), "nvrtcGetPTXSize");
                    long ptxSize = ptxSizeBuf.get(ValueLayout.JAVA_LONG, 0);
                    MemorySegment ptxBuf = temp.allocate(ptxSize);
                    checkError(CudaBindings.getPTX(prog, ptxBuf), "nvrtcGetPTX");
                    image = ptxBuf.toArray(ValueLayout.JAVA_BYTE); // includes the NUL terminator
                }
            } finally {
                CudaBindings.destroyProgram(progBuf);
            }
        }

        MemorySegment module = loadModuleImage(image);
        if (module == null) throw new RuntimeException("cuModuleLoadDataEx failed for " + resourcePath);
        if (cacheFile != null) writeCacheAtomically(cacheFile, image);
        return module;
    }

    /** Load a cubin or NUL-terminated PTX image; null when the driver rejects it. */
    private MemorySegment loadModuleImage(byte[] image) {
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment buf = temp.allocate(image.length + 1L);
            MemorySegment.copy(image, 0, buf, ValueLayout.JAVA_BYTE, 0, image.length);
            buf.set(ValueLayout.JAVA_BYTE, image.length, (byte) 0);
            MemorySegment moduleBuf = arena.allocate(ValueLayout.ADDRESS);
            int err = CudaBindings.moduleLoadDataEx(moduleBuf, buf, 0, MemorySegment.NULL, MemorySegment.NULL);
            if (err != CudaBindings.CUDA_SUCCESS) return null;
            return moduleBuf.get(ValueLayout.ADDRESS, 0);
        }
    }

    /**
     * Pick the NVRTC target once. The device's own SM when this NVRTC supports it (cubin output,
     * no driver JIT); otherwise the newest virtual architecture NVRTC supports that is not above
     * the device (PTX, which the driver JIT-compiles forward). Passing an unsupported
     * compute_XX verbatim — a device newer than the installed NVRTC — would fail every compile.
     */
    private synchronized void resolveNvrtcTarget() {
        if (nvrtcArch != null) return;
        int cc = computeCapabilityMajor * 10 + computeCapabilityMinor;
        int[] supported = CudaBindings.nvrtcSupportedArchs();
        if (supported == null) {
            nvrtcArch = "compute_" + cc;
            nvrtcCubin = false;
            return;
        }
        boolean exact = false;
        int best = -1;
        for (int a : supported) {
            if (a == cc) exact = true;
            if (a <= cc && a > best) best = a;
        }
        if (exact && CudaBindings.isCubinAvailable()
                && !"false".equals(System.getProperty("cuda.nvrtc.cubin", "true"))) {
            nvrtcArch = "sm_" + cc;
            nvrtcCubin = true;
        } else {
            nvrtcArch = "compute_" + (best > 0 ? best : cc);
            nvrtcCubin = false;
        }
    }

    private java.nio.file.Path kernelCachePath(String source, String options) {
        try {
            java.security.MessageDigest md = java.security.MessageDigest.getInstance("SHA-256");
            int[] v = CudaBindings.nvrtcVersion();
            String header = "nvrtc=" + (v == null ? "?" : v[0] + "." + v[1]) + "\nopts=" + options
                + "\ncubin=" + nvrtcCubin + "\n";
            md.update(header.getBytes(StandardCharsets.UTF_8));
            md.update(source.getBytes(StandardCharsets.UTF_8));
            StringBuilder hex = new StringBuilder();
            for (byte b : md.digest()) hex.append(String.format("%02x", b));
            return java.nio.file.Paths.get(KERNEL_CACHE_DIR, hex + (nvrtcCubin ? ".cubin" : ".ptx"));
        } catch (Exception e) {
            return null;
        }
    }

    private static void writeCacheAtomically(java.nio.file.Path file, byte[] image) {
        try {
            java.nio.file.Files.createDirectories(file.getParent());
            java.nio.file.Path tmp = java.nio.file.Files.createTempFile(file.getParent(), "k", ".tmp");
            java.nio.file.Files.write(tmp, image);
            try {
                java.nio.file.Files.move(tmp, file, java.nio.file.StandardCopyOption.ATOMIC_MOVE,
                    java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            } catch (IOException e) {
                java.nio.file.Files.move(tmp, file, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            }
        } catch (Exception ignored) {
            // The cache is an optimisation only; a read-only home directory just disables it.
        }
    }

    // ---- Stream priorities, scheduling mode, clock keeper ----

    private int leastPriority, greatestPriority;

    /** Numeric priority of the least-priority stream (the largest number). */
    public int leastStreamPriority() { return leastPriority; }

    /** Numeric priority the work stream was created with (the greatest priority). */
    public int workStreamPriority() { return greatestPriority; }

    private static int schedFlags() {
        String v = System.getProperty("cuda.sched", "auto");
        switch (v) {
            case "spin": return CudaBindings.CU_CTX_SCHED_SPIN;
            case "yield": return CudaBindings.CU_CTX_SCHED_YIELD;
            case "blocking": return CudaBindings.CU_CTX_SCHED_BLOCKING_SYNC;
            default: return CudaBindings.CU_CTX_SCHED_AUTO;
        }
    }

    /** Create a non-blocking stream at the least priority (below the work stream). */
    public MemorySegment createLowPriorityStream() {
        ensureCurrent();
        MemorySegment sb = arena.allocate(ValueLayout.ADDRESS);
        if (CudaBindings.isStreamPriorityAvailable()) {
            checkError(CudaBindings.streamCreateWithPriority(sb, CudaBindings.CU_STREAM_NON_BLOCKING, leastPriority),
                "cuStreamCreateWithPriority");
        } else {
            checkError(CudaBindings.streamCreate(sb, CudaBindings.CU_STREAM_NON_BLOCKING), "cuStreamCreate");
        }
        return sb.get(ValueLayout.ADDRESS, 0);
    }

    private volatile CudaClockKeeper clockKeeper;

    /**
     * Install the clock keeper ({@code -Dcuda.clockkeeper}, opt-in) as the process's
     * {@link GpuActivity} listener. Call after every weight upload: the keeper only runs while a
     * generation is in progress, never during loading. No-op when already installed.
     */
    public synchronized void installClockKeeper() {
        if (clockKeeper != null || closed) return;
        clockKeeper = new CudaClockKeeper(this);
        GpuActivity.setListener(clockKeeper);
    }

    /** Legacy name of {@link #installClockKeeper()}. */
    public void maybeStartClockKeeper() { installClockKeeper(); }

    public void launchKernelOnStream(MemorySegment fn, int grid, int block, int shared, MemorySegment params,
                                     MemorySegment onStream) {
        launchKernel1DOnStream(fn, grid, block, shared, params, onStream);
    }

    /** Create an event; with {@code timing} it can be used with {@link #elapsedMs}. */
    public MemorySegment createEvent(boolean timing) {
        ensureCurrent();
        MemorySegment eventBuf = arena.allocate(ValueLayout.ADDRESS);
        checkError(CudaBindings.eventCreate(eventBuf, timing ? CudaBindings.CU_EVENT_DEFAULT : CudaBindings.CU_EVENT_DISABLE_TIMING),
            "cuEventCreate");
        return eventBuf.get(ValueLayout.ADDRESS, 0);
    }

    /** True when every operation before {@code event} has completed (never blocks). */
    public boolean eventDone(MemorySegment event) {
        int err = CudaBindings.eventQuery(event);
        if (err == CudaBindings.CUDA_SUCCESS) return true;
        if (err == CudaBindings.CUDA_ERROR_NOT_READY) return false;
        throw new RuntimeException("CUDA error in cuEventQuery: " + err);
    }

    /**
     * Wait for {@code event} without spinning a core: poll with {@code cuEventQuery} and park
     * between polls ({@code parkNanos}, about 50-60 us granularity on Linux).
     */
    public void waitEventParked(MemorySegment event, long parkNs) {
        while (!eventDone(event)) java.util.concurrent.locks.LockSupport.parkNanos(parkNs);
    }

    /** Milliseconds between two completed timing events. */
    public float elapsedMs(MemorySegment start, MemorySegment end) {
        try (Arena t = Arena.ofConfined()) {
            MemorySegment ms = t.allocate(ValueLayout.JAVA_FLOAT);
            checkError(CudaBindings.eventElapsedTime(ms, start, end), "cuEventElapsedTime");
            return ms.get(ValueLayout.JAVA_FLOAT, 0);
        }
    }

    public void destroyEvent(MemorySegment event) {
        try { CudaBindings.eventDestroy(event); } catch (Exception ignored) { }
    }

    public void destroyStream(MemorySegment s) {
        try { CudaBindings.streamDestroy(s); } catch (Exception ignored) { }
    }

    private void launchKernel1DOnStream(MemorySegment fn, int grid, int block, int shared, MemorySegment params, MemorySegment onStream) {
        checkError(CudaBindings.launchKernel(fn, grid, 1, 1, block, 1, 1, shared, onStream, params, MemorySegment.NULL), "cuLaunchKernel");
    }

    /** Make this context current on the calling thread if it is not already. */
    public void ensureCurrent() {
        if (CURRENT.get() != this) {
            checkError(CudaBindings.ctxSetCurrent(ctx), "cuCtxSetCurrent");
            CURRENT.set(this);
        }
    }

    /**
     * Allow {@code function} to be launched with up to {@code bytes} of dynamic shared memory.
     * Above 48 KB a kernel must opt in explicitly; returns false when the device cannot provide
     * that much (the caller must then use a kernel that does not need it).
     */
    public boolean setMaxDynamicSharedMem(MemorySegment function, int bytes) {
        if (bytes <= 48 * 1024) return true;
        if (bytes > getMaxSharedMemOptin()) return false;
        ensureCurrent();
        return CudaBindings.funcSetAttribute(function,
            CudaBindings.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes) == CudaBindings.CUDA_SUCCESS;
    }

    /** Largest dynamic shared memory per block the device allows with opt-in (bytes). */
    public int getMaxSharedMemOptin() {
        int v = maxSharedOptin;
        if (v != 0) return v;
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment buf = temp.allocate(ValueLayout.JAVA_INT);
            int err = CudaBindings.deviceGetAttribute(buf,
                CudaBindings.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device);
            v = err == CudaBindings.CUDA_SUCCESS ? buf.get(ValueLayout.JAVA_INT, 0) : 48 * 1024;
        }
        if (v < 48 * 1024) v = 48 * 1024;
        maxSharedOptin = v;
        return v;
    }

    /**
     * Allocate page-locked host memory (not device-mapped). Copies from/to it are true DMA
     * transfers that need no intermediate staging. Free with {@link #freeHostMappedBuffer}.
     */
    public MemorySegment allocPinnedHost(long bytes) {
        ensureCurrent();
        MemorySegment ppBuf = arena.allocate(ValueLayout.ADDRESS);
        checkError(CudaBindings.memHostAlloc(ppBuf, bytes, CudaBindings.CU_MEMHOSTALLOC_PORTABLE), "cuMemHostAlloc");
        MemorySegment seg = ppBuf.get(ValueLayout.ADDRESS, 0).reinterpret(bytes);
        pinnedBlocks.add(new long[] { seg.address(), bytes });
        return seg;
    }

    /** Free a block returned by {@link #allocPinnedHost}. */
    public void freePinnedHost(MemorySegment seg) {
        long a = seg.address();
        pinnedBlocks.removeIf(b -> b[0] == a);
        CudaBindings.memFreeHost(MemorySegment.ofAddress(a));
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
        ensureCurrent();
        MemorySegment dptrBuf = arena.allocate(ValueLayout.JAVA_LONG);
        checkError(CudaBindings.memAlloc(dptrBuf, bytes), "cuMemAlloc");
        return dptrBuf.get(ValueLayout.JAVA_LONG, 0);
    }

    /**
     * Free a GPU buffer.
     */
    public void freeBuffer(long dptr) {
        ensureCurrent();
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
        ensureCurrent();
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment freeBuf = temp.allocate(ValueLayout.JAVA_LONG);
            MemorySegment totalBuf = temp.allocate(ValueLayout.JAVA_LONG);
            checkError(CudaBindings.memGetInfo(freeBuf, totalBuf), "cuMemGetInfo");
            return new long[] { freeBuf.get(ValueLayout.JAVA_LONG, 0), totalBuf.get(ValueLayout.JAVA_LONG, 0) };
        }
    }

    /**
     * Copy host data to device, ordered on the work stream after every earlier launch. The host
     * buffer may be reused as soon as this returns: from pageable memory the driver has staged it
     * by then; from pinned memory the call waits for the copy.
     */
    public void writeBuffer(long dptr, MemorySegment hostData, long size) {
        ensureCurrent();
        checkError(CudaBindings.memcpyHtoDAsync(dptr, hostData, size, stream), "cuMemcpyHtoDAsync");
        if (isPinned(hostData)) {
            checkError(CudaBindings.streamSynchronize(stream), "cuStreamSynchronize");
        }
    }

    /**
     * Copy host data to device (async on stream). With pinned host memory the host buffer must
     * not be modified until the stream has been synchronized.
     */
    public void writeBufferAsync(long dptr, MemorySegment hostData, long size) {
        ensureCurrent();
        checkError(CudaBindings.memcpyHtoDAsync(dptr, hostData, size, stream), "cuMemcpyHtoDAsync");
    }

    /**
     * Copy device data to host after every earlier launch on the work stream has completed
     * (blocking).
     */
    public void readBuffer(long dptr, MemorySegment hostData, long size) {
        ensureCurrent();
        if (CudaBindings.isMemcpyDtoHAsyncAvailable()) {
            checkError(CudaBindings.memcpyDtoHAsync(hostData, dptr, size, stream), "cuMemcpyDtoHAsync");
            checkError(CudaBindings.streamSynchronize(stream), "cuStreamSynchronize");
        } else {
            checkError(CudaBindings.streamSynchronize(stream), "cuStreamSynchronize");
            checkError(CudaBindings.memcpyDtoH(hostData, dptr, size), "cuMemcpyDtoH");
        }
    }

    /**
     * Queue a device-to-host copy on the work stream; the host data is valid after {@link #finish()}.
     * Use pinned host memory, or the copy is synchronous.
     */
    public void readBufferAsync(long dptr, MemorySegment hostData, long size) {
        ensureCurrent();
        if (CudaBindings.isMemcpyDtoHAsyncAvailable()) {
            checkError(CudaBindings.memcpyDtoHAsync(hostData, dptr, size, stream), "cuMemcpyDtoHAsync");
        } else {
            checkError(CudaBindings.streamSynchronize(stream), "cuStreamSynchronize");
            checkError(CudaBindings.memcpyDtoH(hostData, dptr, size), "cuMemcpyDtoH");
        }
    }

    /**
     * Copy device data to device (async on stream).
     */
    public void copyBufferDtoD(long dst, long src, long sizeBytes) {
        ensureCurrent();
        checkError(CudaBindings.memcpyDtoDAsync(dst, src, sizeBytes, stream), "cuMemcpyDtoDAsync");
    }

    /**
     * Fill GPU buffer with zero (float 0.0f), ordered on the work stream.
     */
    public void fillBufferZero(long dptr, long sizeBytes) {
        ensureCurrent();
        long numFloats = sizeBytes / Float.BYTES;
        if (CudaBindings.isMemsetAsyncAvailable()) {
            checkError(CudaBindings.memsetD32Async(dptr, 0, numFloats, stream), "cuMemsetD32Async");
        } else {
            checkError(CudaBindings.streamSynchronize(stream), "cuStreamSynchronize");
            checkError(CudaBindings.memsetD32(dptr, 0, numFloats), "cuMemsetD32");
        }
    }

    /**
     * Synchronize the stream (wait for all pending operations).
     */
    public void finish() {
        ensureCurrent();
        checkError(CudaBindings.streamSynchronize(stream), "cuStreamSynchronize");
    }

    // [address, length] of the pinned host blocks handed out by allocPinnedHost (for writeBuffer).
    private final java.util.List<long[]> pinnedBlocks = new java.util.concurrent.CopyOnWriteArrayList<>();

    private boolean isPinned(MemorySegment seg) {
        if (pinnedBlocks.isEmpty() || !seg.isNative()) return false;
        long a = seg.address();
        for (long[] b : pinnedBlocks) {
            if (a >= b[0] && a < b[0] + b[1]) return true;
        }
        return false;
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
        ensureCurrent();
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
        checkError(CudaBindings.streamCreate(streamBuf, CudaBindings.CU_STREAM_NON_BLOCKING), "cuStreamCreate");
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
        ensureCurrent();
        // THREAD_LOCAL: only unsafe calls from the capturing thread invalidate the capture;
        // GLOBAL also failed it on a legacy-stream call made by any other thread (error 906).
        checkError(CudaBindings.streamBeginCapture(stream, CudaBindings.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
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
        ensureCurrent();
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

    private volatile VramGuard vramGuard;

    /** The allocation guard of this context (F7), or null when disabled. */
    public VramGuard vramGuard() { return vramGuard; }

    /**
     * Allocate device memory that must be real VRAM: verified by {@link VramGuard} when free
     * memory is low (throws {@link VramGuard.VramExhaustedException} otherwise). The buffer is
     * zero-filled when it is large enough to be checked.
     */
    public long allocBufferChecked(long bytes, String what) {
        VramGuard g = vramGuard;
        return g != null ? g.allocChecked(bytes, what) : allocBuffer(bytes);
    }
    public Arena getArena() { return arena; }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        if (vramGuard != null) {
            try { vramGuard.close(); } catch (Exception ignored) { }
        }
        CudaClockKeeper keeper = clockKeeper;
        if (keeper != null) {
            if (GpuActivity.listener() == keeper) GpuActivity.setListener(null);
            keeper.shutdown();
        }
        try { ensureCurrent(); } catch (Exception ignored) {}
        functionCache.clear();
        for (MemorySegment module : moduleCache.values()) {
            try { CudaBindings.moduleUnload(module); } catch (Exception ignored) {}
        }
        moduleCache.clear();
        try { CudaBindings.streamDestroy(stream); } catch (Exception ignored) {}
        try { CudaBindings.ctxDestroy(ctx); } catch (Exception ignored) {}
        if (CURRENT.get() == this) CURRENT.remove();
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
