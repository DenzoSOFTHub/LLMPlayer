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
 * Manages OpenCL context lifecycle: device enumeration, context/queue creation,
 * kernel compilation, and buffer management.
 */
public class OpenCLContext implements AutoCloseable {

    private final Arena arena;
    private final MemorySegment device;
    private final MemorySegment context;
    private final MemorySegment queue;
    private final DeviceInfo deviceInfo;
    private final Map<String, MemorySegment> kernelCache = new ConcurrentHashMap<>();
    private final Map<String, MemorySegment> programCache = new ConcurrentHashMap<>();
    private volatile boolean closed = false;

    private OpenCLContext(Arena arena, MemorySegment device, MemorySegment context,
                          MemorySegment queue, DeviceInfo deviceInfo) {
        this.arena = arena;
        this.device = device;
        this.context = context;
        this.queue = queue;
        this.deviceInfo = deviceInfo;
    }

    /**
     * Enumerate all OpenCL devices across all platforms.
     */
    public static List<DeviceInfo> enumerateDevices() {
        List<DeviceInfo> result = new ArrayList<>();
        if (!OpenCLBindings.isAvailable()) return result;

        try (Arena temp = Arena.ofConfined()) {
            // Get platform count
            MemorySegment numPlatforms = temp.allocate(ValueLayout.JAVA_INT);
            int err = OpenCLBindings.getPlatformIDs(0, MemorySegment.NULL, numPlatforms);
            if (err != OpenCLBindings.CL_SUCCESS) return result;
            int platCount = numPlatforms.get(ValueLayout.JAVA_INT, 0);
            if (platCount == 0) return result;

            // Get platform IDs
            MemorySegment platforms = temp.allocate(ValueLayout.ADDRESS, platCount);
            err = OpenCLBindings.getPlatformIDs(platCount, platforms, MemorySegment.NULL);
            if (err != OpenCLBindings.CL_SUCCESS) return result;

            int globalIndex = 0;
            for (int p = 0; p < platCount; p++) {
                MemorySegment platform = platforms.getAtIndex(ValueLayout.ADDRESS, p);

                // Get device count for this platform (all types)
                MemorySegment numDevices = temp.allocate(ValueLayout.JAVA_INT);
                err = OpenCLBindings.getDeviceIDs(platform, OpenCLBindings.CL_DEVICE_TYPE_ALL,
                    0, MemorySegment.NULL, numDevices);
                if (err != OpenCLBindings.CL_SUCCESS) continue;
                int devCount = numDevices.get(ValueLayout.JAVA_INT, 0);
                if (devCount == 0) continue;

                // Get device IDs
                MemorySegment devices = temp.allocate(ValueLayout.ADDRESS, devCount);
                err = OpenCLBindings.getDeviceIDs(platform, OpenCLBindings.CL_DEVICE_TYPE_ALL,
                    devCount, devices, MemorySegment.NULL);
                if (err != OpenCLBindings.CL_SUCCESS) continue;

                for (int d = 0; d < devCount; d++) {
                    MemorySegment dev = devices.getAtIndex(ValueLayout.ADDRESS, d);
                    DeviceInfo info = queryDeviceInfo(temp, dev, globalIndex);
                    if (info != null) {
                        result.add(info);
                        globalIndex++;
                    }
                }
            }
        }
        return result;
    }

    /**
     * Create an OpenCL context for the device at the given index.
     */
    public static OpenCLContext create(int deviceIndex) {
        if (!OpenCLBindings.isAvailable()) {
            throw new RuntimeException("OpenCL library not available");
        }

        Arena arena = Arena.ofShared();
        try {
            // Find the device
            MemorySegment numPlatforms = arena.allocate(ValueLayout.JAVA_INT);
            checkError(OpenCLBindings.getPlatformIDs(0, MemorySegment.NULL, numPlatforms), "getPlatformIDs");
            int platCount = numPlatforms.get(ValueLayout.JAVA_INT, 0);
            MemorySegment platforms = arena.allocate(ValueLayout.ADDRESS, platCount);
            checkError(OpenCLBindings.getPlatformIDs(platCount, platforms, MemorySegment.NULL), "getPlatformIDs");

            int globalIndex = 0;
            for (int p = 0; p < platCount; p++) {
                MemorySegment platform = platforms.getAtIndex(ValueLayout.ADDRESS, p);
                MemorySegment numDevices = arena.allocate(ValueLayout.JAVA_INT);
                int err = OpenCLBindings.getDeviceIDs(platform, OpenCLBindings.CL_DEVICE_TYPE_ALL,
                    0, MemorySegment.NULL, numDevices);
                if (err != OpenCLBindings.CL_SUCCESS) continue;
                int devCount = numDevices.get(ValueLayout.JAVA_INT, 0);
                if (devCount == 0) continue;

                MemorySegment devices = arena.allocate(ValueLayout.ADDRESS, devCount);
                err = OpenCLBindings.getDeviceIDs(platform, OpenCLBindings.CL_DEVICE_TYPE_ALL,
                    devCount, devices, MemorySegment.NULL);
                if (err != OpenCLBindings.CL_SUCCESS) continue;

                for (int d = 0; d < devCount; d++) {
                    if (globalIndex == deviceIndex) {
                        MemorySegment dev = devices.getAtIndex(ValueLayout.ADDRESS, d);
                        DeviceInfo info = queryDeviceInfo(arena, dev, globalIndex);

                        // Create context
                        MemorySegment errcodeRet = arena.allocate(ValueLayout.JAVA_INT);
                        MemorySegment deviceArray = arena.allocate(ValueLayout.ADDRESS, 1);
                        deviceArray.setAtIndex(ValueLayout.ADDRESS, 0, dev);
                        MemorySegment ctx = OpenCLBindings.createContext(
                            MemorySegment.NULL, 1, deviceArray,
                            MemorySegment.NULL, MemorySegment.NULL, errcodeRet);
                        checkError(errcodeRet.get(ValueLayout.JAVA_INT, 0), "createContext");

                        // Create command queue
                        MemorySegment q = OpenCLBindings.createCommandQueue(ctx, dev, 0L, errcodeRet);
                        checkError(errcodeRet.get(ValueLayout.JAVA_INT, 0), "createCommandQueue");

                        return new OpenCLContext(arena, dev, ctx, q, info);
                    }
                    globalIndex++;
                }
            }
            throw new RuntimeException("OpenCL device index " + deviceIndex + " not found");
        } catch (Exception e) {
            arena.close();
            throw e;
        }
    }

    /**
     * Compile a kernel from a .cl resource file, caching the result.
     */
    public MemorySegment compileKernel(String resourcePath, String kernelName) {
        String cacheKey = resourcePath + ":" + kernelName;
        MemorySegment cached = kernelCache.get(cacheKey);
        if (cached != null) return cached;

        synchronized (this) {
            cached = kernelCache.get(cacheKey);
            if (cached != null) return cached;

            // Load source from classpath
            String source = loadResource(resourcePath);
            if (source == null) {
                throw new RuntimeException("Kernel resource not found: " + resourcePath);
            }

            // Create program
            MemorySegment sourceStr = arena.allocateFrom(source);
            MemorySegment strings = arena.allocate(ValueLayout.ADDRESS, 1);
            strings.setAtIndex(ValueLayout.ADDRESS, 0, sourceStr);
            MemorySegment lengths = arena.allocateFrom(ValueLayout.JAVA_LONG, (long) source.length());

            MemorySegment errcodeRet = arena.allocate(ValueLayout.JAVA_INT);
            MemorySegment program = OpenCLBindings.createProgramWithSource(context, 1, strings, lengths, errcodeRet);
            checkError(errcodeRet.get(ValueLayout.JAVA_INT, 0), "createProgramWithSource");

            // Build program
            MemorySegment deviceArray = arena.allocate(ValueLayout.ADDRESS, 1);
            deviceArray.setAtIndex(ValueLayout.ADDRESS, 0, device);
            int buildErr = OpenCLBindings.buildProgram(program, 1, deviceArray,
                MemorySegment.NULL, MemorySegment.NULL, MemorySegment.NULL);
            if (buildErr != OpenCLBindings.CL_SUCCESS) {
                String buildLog = getBuildLog(program, device);
                OpenCLBindings.releaseProgram(program);
                throw new RuntimeException("OpenCL kernel build failed (" + buildErr + "):\n" + buildLog);
            }

            programCache.put(resourcePath, program);

            // Create kernel
            MemorySegment kernelNameStr = arena.allocateFrom(kernelName);
            MemorySegment kernel = OpenCLBindings.createKernel(program, kernelNameStr, errcodeRet);
            checkError(errcodeRet.get(ValueLayout.JAVA_INT, 0), "createKernel: " + kernelName);

            kernelCache.put(cacheKey, kernel);
            return kernel;
        }
    }

    /**
     * Create a GPU buffer with given flags and size.
     */
    public MemorySegment createGpuBuffer(long size, long flags) {
        MemorySegment errcodeRet = arena.allocate(ValueLayout.JAVA_INT);
        MemorySegment buf = OpenCLBindings.createBuffer(context, flags, size, MemorySegment.NULL, errcodeRet);
        checkError(errcodeRet.get(ValueLayout.JAVA_INT, 0), "createBuffer");
        return buf;
    }

    /**
     * Create a GPU buffer and upload host data to it.
     */
    public MemorySegment createGpuBuffer(long size, long flags, MemorySegment hostData) {
        MemorySegment errcodeRet = arena.allocate(ValueLayout.JAVA_INT);
        MemorySegment buf = OpenCLBindings.createBuffer(context,
            flags | OpenCLBindings.CL_MEM_COPY_HOST_PTR, size, hostData, errcodeRet);
        checkError(errcodeRet.get(ValueLayout.JAVA_INT, 0), "createBuffer with host data");
        return buf;
    }

    /**
     * Write host data to a GPU buffer (blocking).
     */
    public void writeBuffer(MemorySegment gpuBuffer, MemorySegment hostData, long size) {
        checkError(OpenCLBindings.enqueueWriteBuffer(queue, gpuBuffer, 1, 0L, size,
            hostData, 0, MemorySegment.NULL, MemorySegment.NULL), "enqueueWriteBuffer");
    }

    /**
     * Read GPU buffer to host memory (blocking).
     */
    public void readBuffer(MemorySegment gpuBuffer, MemorySegment hostData, long size) {
        checkError(OpenCLBindings.enqueueReadBuffer(queue, gpuBuffer, 1, 0L, size,
            hostData, 0, MemorySegment.NULL, MemorySegment.NULL), "enqueueReadBuffer");
    }

    /**
     * Set a kernel argument (pointer to cl_mem).
     */
    public void setKernelArgMem(MemorySegment kernel, int argIndex, MemorySegment clMemArg) {
        setKernelArgMem(kernel, argIndex, clMemArg, arena);
    }

    public void setKernelArgMem(MemorySegment kernel, int argIndex, MemorySegment clMemArg, Arena tempArena) {
        MemorySegment argPtr = tempArena.allocateFrom(ValueLayout.ADDRESS, clMemArg);
        checkError(OpenCLBindings.setKernelArg(kernel, argIndex, ValueLayout.ADDRESS.byteSize(), argPtr),
            "setKernelArg");
    }

    public void setKernelArgInt(MemorySegment kernel, int argIndex, int value) {
        setKernelArgInt(kernel, argIndex, value, arena);
    }

    public void setKernelArgInt(MemorySegment kernel, int argIndex, int value, Arena tempArena) {
        MemorySegment argPtr = tempArena.allocateFrom(ValueLayout.JAVA_INT, value);
        checkError(OpenCLBindings.setKernelArg(kernel, argIndex, ValueLayout.JAVA_INT.byteSize(), argPtr),
            "setKernelArg");
    }

    public void setKernelArgLong(MemorySegment kernel, int argIndex, long value) {
        MemorySegment argPtr = arena.allocateFrom(ValueLayout.JAVA_LONG, value);
        checkError(OpenCLBindings.setKernelArg(kernel, argIndex, ValueLayout.JAVA_LONG.byteSize(), argPtr),
            "setKernelArg");
    }

    public void setKernelArgFloat(MemorySegment kernel, int argIndex, float value) {
        MemorySegment argPtr = arena.allocateFrom(ValueLayout.JAVA_FLOAT, value);
        checkError(OpenCLBindings.setKernelArg(kernel, argIndex, ValueLayout.JAVA_FLOAT.byteSize(), argPtr),
            "setKernelArg");
    }

    /**
     * Enqueue a 1D NDRange kernel execution.
     */
    public void enqueueKernel1D(MemorySegment kernel, long globalWorkSize, long localWorkSize) {
        enqueueKernel1D(kernel, globalWorkSize, localWorkSize, arena);
    }

    public void enqueueKernel1D(MemorySegment kernel, long globalWorkSize, long localWorkSize, Arena tempArena) {
        MemorySegment gws = tempArena.allocateFrom(ValueLayout.JAVA_LONG, globalWorkSize);
        MemorySegment lws;
        if (localWorkSize > 0) {
            lws = tempArena.allocateFrom(ValueLayout.JAVA_LONG, localWorkSize);
        } else {
            lws = MemorySegment.NULL;
        }
        checkError(OpenCLBindings.enqueueNDRangeKernel(queue, kernel, 1,
            MemorySegment.NULL, gws, lws, 0, MemorySegment.NULL, MemorySegment.NULL),
            "enqueueNDRangeKernel");
    }

    /**
     * Wait for all enqueued commands to complete.
     */
    public void finish() {
        checkError(OpenCLBindings.finish(queue), "clFinish");
    }

    /**
     * Pre-compile all known matmul and utility kernels to avoid compilation stalls during inference.
     */
    public void precompileKernels() {
        String[][] kernels = {
            {"kernels/matmul_f32.cl", "matmul_f32"},
            {"kernels/matmul_q4_0.cl", "matmul_q4_0"},
            {"kernels/matmul_q4_k.cl", "matmul_q4_k"},
            {"kernels/matmul_q5_k.cl", "matmul_q5_k"},
            {"kernels/matmul_q6_k.cl", "matmul_q6_k"},
            {"kernels/matmul_q8_0.cl", "matmul_q8_0"},
            {"kernels/matmul_q3_k.cl", "matmul_q3_k"},
            {"kernels/rmsnorm.cl", "rmsnorm_sumsq"},
            {"kernels/rmsnorm.cl", "rmsnorm_normalize"},
            {"kernels/softmax.cl", "softmax_max"},
            {"kernels/softmax.cl", "softmax_exp_sum"},
            {"kernels/softmax.cl", "softmax_normalize"},
            {"kernels/silu.cl", "silu"},
            {"kernels/saxpy.cl", "saxpy"},
            {"kernels/accumulate.cl", "accumulate"},
        };
        for (String[] kv : kernels) {
            try {
                compileKernel(kv[0], kv[1]);
            } catch (Exception ignored) {
                // Some kernels may not exist — that's fine
            }
        }
    }

    public DeviceInfo getDeviceInfo() { return deviceInfo; }
    public MemorySegment getClContext() { return context; }
    public MemorySegment getQueue() { return queue; }
    public MemorySegment getDevice() { return device; }
    public Arena getArena() { return arena; }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        for (MemorySegment kernel : kernelCache.values()) {
            try { OpenCLBindings.releaseKernel(kernel); } catch (Exception ignored) {}
        }
        kernelCache.clear();
        for (MemorySegment program : programCache.values()) {
            try { OpenCLBindings.releaseProgram(program); } catch (Exception ignored) {}
        }
        programCache.clear();
        try { OpenCLBindings.releaseCommandQueue(queue); } catch (Exception ignored) {}
        try { OpenCLBindings.releaseContext(context); } catch (Exception ignored) {}
        arena.close();
    }

    // --- Internal helpers ---

    private static DeviceInfo queryDeviceInfo(Arena arena, MemorySegment device, int index) {
        try {
            String name = queryDeviceString(arena, device, OpenCLBindings.CL_DEVICE_NAME);
            String vendor = queryDeviceString(arena, device, OpenCLBindings.CL_DEVICE_VENDOR);

            MemorySegment memBuf = arena.allocate(ValueLayout.JAVA_LONG);
            OpenCLBindings.getDeviceInfo(device, OpenCLBindings.CL_DEVICE_GLOBAL_MEM_SIZE,
                ValueLayout.JAVA_LONG.byteSize(), memBuf, MemorySegment.NULL);
            long globalMem = memBuf.get(ValueLayout.JAVA_LONG, 0);

            MemorySegment cuBuf = arena.allocate(ValueLayout.JAVA_INT);
            OpenCLBindings.getDeviceInfo(device, OpenCLBindings.CL_DEVICE_MAX_COMPUTE_UNITS,
                ValueLayout.JAVA_INT.byteSize(), cuBuf, MemorySegment.NULL);
            int computeUnits = cuBuf.get(ValueLayout.JAVA_INT, 0);

            MemorySegment typeBuf = arena.allocate(ValueLayout.JAVA_LONG);
            OpenCLBindings.getDeviceInfo(device, OpenCLBindings.CL_DEVICE_TYPE,
                ValueLayout.JAVA_LONG.byteSize(), typeBuf, MemorySegment.NULL);
            long devType = typeBuf.get(ValueLayout.JAVA_LONG, 0);
            String typeStr = (devType & OpenCLBindings.CL_DEVICE_TYPE_GPU) != 0 ? "GPU" :
                             (devType & OpenCLBindings.CL_DEVICE_TYPE_ACCELERATOR) != 0 ? "NPU/Accelerator" : "CPU";

            MemorySegment wgBuf = arena.allocate(ValueLayout.JAVA_LONG);
            OpenCLBindings.getDeviceInfo(device, OpenCLBindings.CL_DEVICE_MAX_WORK_GROUP_SIZE,
                ValueLayout.JAVA_LONG.byteSize(), wgBuf, MemorySegment.NULL);
            long maxWorkGroupSize = wgBuf.get(ValueLayout.JAVA_LONG, 0);
            if (maxWorkGroupSize <= 0) maxWorkGroupSize = 64;

            return new DeviceInfo(index, device.address(), name, vendor, globalMem, computeUnits, typeStr, maxWorkGroupSize);
        } catch (Exception e) {
            return null;
        }
    }

    private static String queryDeviceString(Arena arena, MemorySegment device, int paramName) {
        MemorySegment sizeRet = arena.allocate(ValueLayout.JAVA_LONG);
        OpenCLBindings.getDeviceInfo(device, paramName, 0, MemorySegment.NULL, sizeRet);
        long size = sizeRet.get(ValueLayout.JAVA_LONG, 0);
        if (size <= 0) return "Unknown";
        MemorySegment buf = arena.allocate(size);
        OpenCLBindings.getDeviceInfo(device, paramName, size, buf, MemorySegment.NULL);
        return buf.getString(0).trim();
    }

    private String getBuildLog(MemorySegment program, MemorySegment dev) {
        try {
            MemorySegment sizeRet = arena.allocate(ValueLayout.JAVA_LONG);
            OpenCLBindings.getProgramBuildInfo(program, dev,
                OpenCLBindings.CL_PROGRAM_BUILD_LOG, 0, MemorySegment.NULL, sizeRet);
            long size = sizeRet.get(ValueLayout.JAVA_LONG, 0);
            if (size <= 0) return "(no build log)";
            MemorySegment buf = arena.allocate(size);
            OpenCLBindings.getProgramBuildInfo(program, dev,
                OpenCLBindings.CL_PROGRAM_BUILD_LOG, size, buf, MemorySegment.NULL);
            return buf.getString(0);
        } catch (Exception e) {
            return "(failed to get build log: " + e.getMessage() + ")";
        }
    }

    private static String loadResource(String path) {
        try (InputStream is = OpenCLContext.class.getClassLoader().getResourceAsStream(path)) {
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
        if (err != OpenCLBindings.CL_SUCCESS) {
            throw new RuntimeException("OpenCL error in " + op + ": " + err);
        }
    }
}
