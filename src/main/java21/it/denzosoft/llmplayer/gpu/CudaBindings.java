package it.denzosoft.llmplayer.gpu;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

/**
 * Panama FFM bindings for CUDA Driver API + NVRTC.
 * Loads libcuda and libnvrtc dynamically on Linux.
 */
public final class CudaBindings {

    private CudaBindings() {}

    // CUDA error codes
    public static final int CUDA_SUCCESS = 0;

    // Device attributes
    public static final int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
    public static final int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;
    public static final int CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1;
    public static final int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup CUDA_LIB;
    private static final SymbolLookup NVRTC_LIB;

    static {
        SymbolLookup cudaLib = null;
        SymbolLookup nvrtcLib = null;
        try {
            // Try libcuda.so.1 first, then libcuda.so
            try {
                cudaLib = SymbolLookup.libraryLookup("libcuda.so.1", Arena.global());
            } catch (IllegalArgumentException e) {
                try {
                    cudaLib = SymbolLookup.libraryLookup("libcuda.so", Arena.global());
                } catch (IllegalArgumentException e2) {
                    // CUDA not available
                }
            }
        } catch (Exception e) {
            // CUDA not available
        }

        if (cudaLib != null) {
            try {
                try {
                    nvrtcLib = SymbolLookup.libraryLookup("libnvrtc.so", Arena.global());
                } catch (IllegalArgumentException e) {
                    try {
                        nvrtcLib = SymbolLookup.libraryLookup("/usr/local/cuda/lib64/libnvrtc.so", Arena.global());
                    } catch (IllegalArgumentException e2) {
                        // NVRTC not available
                    }
                }
            } catch (Exception e) {
                // NVRTC not available
            }
        }
        CUDA_LIB = cudaLib;
        NVRTC_LIB = nvrtcLib;
    }

    public static boolean isAvailable() {
        return CUDA_LIB != null && NVRTC_LIB != null;
    }

    public static boolean isCudaAvailable() {
        return CUDA_LIB != null;
    }

    private static MethodHandle findCuda(String name, FunctionDescriptor desc) {
        return CUDA_LIB.find(name)
            .map(addr -> LINKER.downcallHandle(addr, desc))
            .orElseThrow(() -> new UnsupportedOperationException("CUDA function not found: " + name));
    }

    private static MethodHandle findNvrtc(String name, FunctionDescriptor desc) {
        return NVRTC_LIB.find(name)
            .map(addr -> LINKER.downcallHandle(addr, desc))
            .orElseThrow(() -> new UnsupportedOperationException("NVRTC function not found: " + name));
    }

    private static MethodHandle findCudaIfAvailable(String name, FunctionDescriptor desc) {
        if (CUDA_LIB == null) return null;
        try { return findCuda(name, desc); } catch (Exception e) { return null; }
    }

    private static MethodHandle findNvrtcIfAvailable(String name, FunctionDescriptor desc) {
        if (NVRTC_LIB == null) return null;
        try { return findNvrtc(name, desc); } catch (Exception e) { return null; }
    }

    // --- CUDA Driver API MethodHandles ---

    // CUresult cuInit(unsigned int Flags)
    private static final MethodHandle cuInit = findCudaIfAvailable("cuInit",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));

    // CUresult cuDeviceGetCount(int* count)
    private static final MethodHandle cuDeviceGetCount = findCudaIfAvailable("cuDeviceGetCount",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

    // CUresult cuDeviceGet(CUdevice* device, int ordinal)
    private static final MethodHandle cuDeviceGet = findCudaIfAvailable("cuDeviceGet",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

    // CUresult cuDeviceGetName(char* name, int len, CUdevice dev)
    private static final MethodHandle cuDeviceGetName = findCudaIfAvailable("cuDeviceGetName",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));

    // CUresult cuDeviceTotalMem_v2(size_t* bytes, CUdevice dev)
    private static final MethodHandle cuDeviceTotalMem_v2 = findCudaIfAvailable("cuDeviceTotalMem_v2",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

    // CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev)
    private static final MethodHandle cuDeviceGetAttribute = findCudaIfAvailable("cuDeviceGetAttribute",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));

    // CUresult cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev)
    private static final MethodHandle cuCtxCreate_v2 = findCudaIfAvailable("cuCtxCreate_v2",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));

    // CUresult cuCtxDestroy_v2(CUcontext ctx)
    private static final MethodHandle cuCtxDestroy_v2 = findCudaIfAvailable("cuCtxDestroy_v2",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

    // CUresult cuCtxSetCurrent(CUcontext ctx)
    private static final MethodHandle cuCtxSetCurrent = findCudaIfAvailable("cuCtxSetCurrent",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

    // CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize)
    private static final MethodHandle cuMemAlloc_v2 = findCudaIfAvailable("cuMemAlloc_v2",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));

    // CUresult cuMemFree_v2(CUdeviceptr dptr)
    private static final MethodHandle cuMemFree_v2 = findCudaIfAvailable("cuMemFree_v2",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));

    // CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount)
    private static final MethodHandle cuMemcpyHtoD_v2 = findCudaIfAvailable("cuMemcpyHtoD_v2",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));

    // CUresult cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount)
    private static final MethodHandle cuMemcpyDtoH_v2 = findCudaIfAvailable("cuMemcpyDtoH_v2",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));

    // CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream)
    private static final MethodHandle cuMemcpyHtoDAsync_v2 = findCudaIfAvailable("cuMemcpyHtoDAsync_v2",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));

    // CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N)
    private static final MethodHandle cuMemsetD32_v2 = findCudaIfAvailable("cuMemsetD32_v2",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));

    // CUresult cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions, ...)
    private static final MethodHandle cuModuleLoadDataEx = findCudaIfAvailable("cuModuleLoadDataEx",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    // CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name)
    private static final MethodHandle cuModuleGetFunction = findCudaIfAvailable("cuModuleGetFunction",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    // CUresult cuModuleUnload(CUmodule hmod)
    private static final MethodHandle cuModuleUnload = findCudaIfAvailable("cuModuleUnload",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

    // CUresult cuLaunchKernel(CUfunction f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
    //     sharedMemBytes, hStream, kernelParams, extra)
    private static final MethodHandle cuLaunchKernel = findCudaIfAvailable("cuLaunchKernel",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    // CUresult cuStreamCreate(CUstream* phStream, unsigned int Flags)
    private static final MethodHandle cuStreamCreate = findCudaIfAvailable("cuStreamCreate",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

    // CUresult cuStreamSynchronize(CUstream hStream)
    private static final MethodHandle cuStreamSynchronize = findCudaIfAvailable("cuStreamSynchronize",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

    // CUresult cuStreamDestroy_v2(CUstream hStream)
    private static final MethodHandle cuStreamDestroy_v2 = findCudaIfAvailable("cuStreamDestroy_v2",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

    // CUresult cuEventCreate(CUevent* phEvent, unsigned int Flags)
    private static final MethodHandle cuEventCreate = findCudaIfAvailable("cuEventCreate",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

    // CUresult cuEventRecord(CUevent hEvent, CUstream hStream)
    private static final MethodHandle cuEventRecord = findCudaIfAvailable("cuEventRecord",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    // CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags)
    private static final MethodHandle cuStreamWaitEvent = findCudaIfAvailable("cuStreamWaitEvent",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

    // CUresult cuEventDestroy_v2(CUevent hEvent)
    private static final MethodHandle cuEventDestroy_v2 = findCudaIfAvailable("cuEventDestroy_v2",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

    // --- Unified/Managed Memory ---

    // CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags)
    // flags: CU_MEM_ATTACH_GLOBAL=1, CU_MEM_ATTACH_HOST=2
    private static final MethodHandle cuMemAllocManaged = findCudaIfAvailable("cuMemAllocManaged",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT));

    // --- Host-Mapped (Zero-Copy) Memory ---

    // CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags)
    // Flags: CU_MEMHOSTALLOC_DEVICEMAP=0x02, CU_MEMHOSTALLOC_PORTABLE=0x01
    private static final MethodHandle cuMemHostAlloc = findCudaIfAvailable("cuMemHostAlloc",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT));

    // CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr* pdptr, void* p, unsigned int Flags)
    private static final MethodHandle cuMemHostGetDevicePointer_v2 = findCudaIfAvailable("cuMemHostGetDevicePointer_v2",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

    // CUresult cuMemFreeHost(void* p)
    private static final MethodHandle cuMemFreeHost = findCudaIfAvailable("cuMemFreeHost",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

    // CUresult cuMemGetInfo_v2(size_t* free, size_t* total)
    private static final MethodHandle cuMemGetInfo_v2 = findCudaIfAvailable("cuMemGetInfo_v2",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    // --- CUDA Graph API MethodHandles ---

    // CUresult cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode)
    private static final MethodHandle cuStreamBeginCapture_v2 = findCudaIfAvailable("cuStreamBeginCapture_v2",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

    // CUresult cuStreamEndCapture(CUstream hStream, CUgraph* phGraph)
    private static final MethodHandle cuStreamEndCapture = findCudaIfAvailable("cuStreamEndCapture",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    // CUresult cuGraphInstantiateWithFlags(CUgraphExec* phGraphExec, CUgraph hGraph, unsigned long long flags)
    private static final MethodHandle cuGraphInstantiateWithFlags = findCudaIfAvailable("cuGraphInstantiateWithFlags",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));

    // CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream)
    private static final MethodHandle cuGraphLaunch = findCudaIfAvailable("cuGraphLaunch",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    // CUresult cuGraphExecDestroy(CUgraphExec hGraphExec)
    private static final MethodHandle cuGraphExecDestroy = findCudaIfAvailable("cuGraphExecDestroy",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

    // CUresult cuGraphDestroy(CUgraph hGraph)
    private static final MethodHandle cuGraphDestroy = findCudaIfAvailable("cuGraphDestroy",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

    // --- NVRTC MethodHandles ---

    // nvrtcResult nvrtcCreateProgram(nvrtcProgram* prog, const char* src, const char* name, int numHeaders, ...)
    private static final MethodHandle nvrtcCreateProgram = findNvrtcIfAvailable("nvrtcCreateProgram",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    // nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char** options)
    private static final MethodHandle nvrtcCompileProgram = findNvrtcIfAvailable("nvrtcCompileProgram",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

    // nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet)
    private static final MethodHandle nvrtcGetPTXSize = findNvrtcIfAvailable("nvrtcGetPTXSize",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    // nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char* ptx)
    private static final MethodHandle nvrtcGetPTX = findNvrtcIfAvailable("nvrtcGetPTX",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    // nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* logSizeRet)
    private static final MethodHandle nvrtcGetProgramLogSize = findNvrtcIfAvailable("nvrtcGetProgramLogSize",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    // nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char* log)
    private static final MethodHandle nvrtcGetProgramLog = findNvrtcIfAvailable("nvrtcGetProgramLog",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    // nvrtcResult nvrtcDestroyProgram(nvrtcProgram* prog)
    private static final MethodHandle nvrtcDestroyProgram = findNvrtcIfAvailable("nvrtcDestroyProgram",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

    // --- Public API wrappers ---

    public static int init(int flags) {
        try { return (int) cuInit.invokeExact(flags); }
        catch (Throwable t) { throw new RuntimeException("cuInit failed", t); }
    }

    public static int deviceGetCount(MemorySegment count) {
        try { return (int) cuDeviceGetCount.invokeExact(count); }
        catch (Throwable t) { throw new RuntimeException("cuDeviceGetCount failed", t); }
    }

    public static int deviceGet(MemorySegment device, int ordinal) {
        try { return (int) cuDeviceGet.invokeExact(device, ordinal); }
        catch (Throwable t) { throw new RuntimeException("cuDeviceGet failed", t); }
    }

    public static int deviceGetName(MemorySegment name, int len, int dev) {
        try { return (int) cuDeviceGetName.invokeExact(name, len, dev); }
        catch (Throwable t) { throw new RuntimeException("cuDeviceGetName failed", t); }
    }

    public static int deviceTotalMem(MemorySegment bytes, int dev) {
        try { return (int) cuDeviceTotalMem_v2.invokeExact(bytes, dev); }
        catch (Throwable t) { throw new RuntimeException("cuDeviceTotalMem failed", t); }
    }

    public static int deviceGetAttribute(MemorySegment pi, int attrib, int dev) {
        try { return (int) cuDeviceGetAttribute.invokeExact(pi, attrib, dev); }
        catch (Throwable t) { throw new RuntimeException("cuDeviceGetAttribute failed", t); }
    }

    public static int ctxCreate(MemorySegment pctx, int flags, int dev) {
        try { return (int) cuCtxCreate_v2.invokeExact(pctx, flags, dev); }
        catch (Throwable t) { throw new RuntimeException("cuCtxCreate failed", t); }
    }

    public static int ctxDestroy(MemorySegment ctx) {
        try { return (int) cuCtxDestroy_v2.invokeExact(ctx); }
        catch (Throwable t) { throw new RuntimeException("cuCtxDestroy failed", t); }
    }

    public static int ctxSetCurrent(MemorySegment ctx) {
        try { return (int) cuCtxSetCurrent.invokeExact(ctx); }
        catch (Throwable t) { throw new RuntimeException("cuCtxSetCurrent failed", t); }
    }

    public static int memAlloc(MemorySegment dptr, long bytesize) {
        try { return (int) cuMemAlloc_v2.invokeExact(dptr, bytesize); }
        catch (Throwable t) { throw new RuntimeException("cuMemAlloc failed", t); }
    }

    public static int memFree(long dptr) {
        try { return (int) cuMemFree_v2.invokeExact(dptr); }
        catch (Throwable t) { throw new RuntimeException("cuMemFree failed", t); }
    }

    public static int memcpyHtoD(long dstDevice, MemorySegment srcHost, long byteCount) {
        try { return (int) cuMemcpyHtoD_v2.invokeExact(dstDevice, srcHost, byteCount); }
        catch (Throwable t) { throw new RuntimeException("cuMemcpyHtoD failed", t); }
    }

    public static int memcpyDtoH(MemorySegment dstHost, long srcDevice, long byteCount) {
        try { return (int) cuMemcpyDtoH_v2.invokeExact(dstHost, srcDevice, byteCount); }
        catch (Throwable t) { throw new RuntimeException("cuMemcpyDtoH failed", t); }
    }

    public static int memcpyHtoDAsync(long dstDevice, MemorySegment srcHost, long byteCount, MemorySegment stream) {
        try { return (int) cuMemcpyHtoDAsync_v2.invokeExact(dstDevice, srcHost, byteCount, stream); }
        catch (Throwable t) { throw new RuntimeException("cuMemcpyHtoDAsync failed", t); }
    }

    public static int memsetD32(long dstDevice, int ui, long n) {
        try { return (int) cuMemsetD32_v2.invokeExact(dstDevice, ui, n); }
        catch (Throwable t) { throw new RuntimeException("cuMemsetD32 failed", t); }
    }

    public static int moduleLoadDataEx(MemorySegment module, MemorySegment image, int numOptions,
                                        MemorySegment options, MemorySegment optionValues) {
        try { return (int) cuModuleLoadDataEx.invokeExact(module, image, numOptions, options, optionValues); }
        catch (Throwable t) { throw new RuntimeException("cuModuleLoadDataEx failed", t); }
    }

    public static int moduleGetFunction(MemorySegment hfunc, MemorySegment hmod, MemorySegment name) {
        try { return (int) cuModuleGetFunction.invokeExact(hfunc, hmod, name); }
        catch (Throwable t) { throw new RuntimeException("cuModuleGetFunction failed", t); }
    }

    public static int moduleUnload(MemorySegment hmod) {
        try { return (int) cuModuleUnload.invokeExact(hmod); }
        catch (Throwable t) { throw new RuntimeException("cuModuleUnload failed", t); }
    }

    public static int launchKernel(MemorySegment f,
                                    int gridDimX, int gridDimY, int gridDimZ,
                                    int blockDimX, int blockDimY, int blockDimZ,
                                    int sharedMemBytes, MemorySegment hStream,
                                    MemorySegment kernelParams, MemorySegment extra) {
        try {
            return (int) cuLaunchKernel.invokeExact(f,
                gridDimX, gridDimY, gridDimZ,
                blockDimX, blockDimY, blockDimZ,
                sharedMemBytes, hStream,
                kernelParams, extra);
        } catch (Throwable t) { throw new RuntimeException("cuLaunchKernel failed", t); }
    }

    public static int streamCreate(MemorySegment phStream, int flags) {
        try { return (int) cuStreamCreate.invokeExact(phStream, flags); }
        catch (Throwable t) { throw new RuntimeException("cuStreamCreate failed", t); }
    }

    public static int streamSynchronize(MemorySegment hStream) {
        try { return (int) cuStreamSynchronize.invokeExact(hStream); }
        catch (Throwable t) { throw new RuntimeException("cuStreamSynchronize failed", t); }
    }

    public static int streamDestroy(MemorySegment hStream) {
        try { return (int) cuStreamDestroy_v2.invokeExact(hStream); }
        catch (Throwable t) { throw new RuntimeException("cuStreamDestroy failed", t); }
    }

    public static final int CU_EVENT_DISABLE_TIMING = 0x02;

    public static int eventCreate(MemorySegment phEvent, int flags) {
        try { return (int) cuEventCreate.invokeExact(phEvent, flags); }
        catch (Throwable t) { throw new RuntimeException("cuEventCreate failed", t); }
    }

    public static int eventRecord(MemorySegment hEvent, MemorySegment hStream) {
        try { return (int) cuEventRecord.invokeExact(hEvent, hStream); }
        catch (Throwable t) { throw new RuntimeException("cuEventRecord failed", t); }
    }

    public static int streamWaitEvent(MemorySegment hStream, MemorySegment hEvent, int flags) {
        try { return (int) cuStreamWaitEvent.invokeExact(hStream, hEvent, flags); }
        catch (Throwable t) { throw new RuntimeException("cuStreamWaitEvent failed", t); }
    }

    public static int eventDestroy(MemorySegment hEvent) {
        try { return (int) cuEventDestroy_v2.invokeExact(hEvent); }
        catch (Throwable t) { throw new RuntimeException("cuEventDestroy failed", t); }
    }

    // --- Unified/Managed Memory wrappers ---

    public static final int CU_MEM_ATTACH_GLOBAL = 1;

    public static boolean isManagedMemoryAvailable() {
        return cuMemAllocManaged != null;
    }

    public static int memAllocManaged(MemorySegment dptr, long bytesize, int flags) {
        try { return (int) cuMemAllocManaged.invokeExact(dptr, bytesize, flags); }
        catch (Throwable t) { throw new RuntimeException("cuMemAllocManaged failed", t); }
    }

    // --- Host-Mapped (Zero-Copy) Memory wrappers ---

    public static final int CU_MEMHOSTALLOC_DEVICEMAP = 0x02;
    public static final int CU_MEMHOSTALLOC_PORTABLE = 0x01;

    public static boolean isHostMappedMemoryAvailable() {
        return cuMemHostAlloc != null && cuMemHostGetDevicePointer_v2 != null;
    }

    public static int memHostAlloc(MemorySegment pp, long bytesize, int flags) {
        try { return (int) cuMemHostAlloc.invokeExact(pp, bytesize, flags); }
        catch (Throwable t) { throw new RuntimeException("cuMemHostAlloc failed", t); }
    }

    public static int memHostGetDevicePointer(MemorySegment pdptr, MemorySegment p, int flags) {
        try { return (int) cuMemHostGetDevicePointer_v2.invokeExact(pdptr, p, flags); }
        catch (Throwable t) { throw new RuntimeException("cuMemHostGetDevicePointer failed", t); }
    }

    public static int memFreeHost(MemorySegment p) {
        try { return (int) cuMemFreeHost.invokeExact(p); }
        catch (Throwable t) { throw new RuntimeException("cuMemFreeHost failed", t); }
    }

    public static int memGetInfo(MemorySegment free, MemorySegment total) {
        try { return (int) cuMemGetInfo_v2.invokeExact(free, total); }
        catch (Throwable t) { throw new RuntimeException("cuMemGetInfo failed", t); }
    }

    // --- CUDA Graph API wrappers ---

    // CU_STREAM_CAPTURE_MODE_GLOBAL = 0
    public static final int CU_STREAM_CAPTURE_MODE_GLOBAL = 0;

    public static boolean isGraphApiAvailable() {
        return cuStreamBeginCapture_v2 != null && cuGraphLaunch != null;
    }

    public static int streamBeginCapture(MemorySegment hStream, int mode) {
        try { return (int) cuStreamBeginCapture_v2.invokeExact(hStream, mode); }
        catch (Throwable t) { throw new RuntimeException("cuStreamBeginCapture failed", t); }
    }

    public static int streamEndCapture(MemorySegment hStream, MemorySegment phGraph) {
        try { return (int) cuStreamEndCapture.invokeExact(hStream, phGraph); }
        catch (Throwable t) { throw new RuntimeException("cuStreamEndCapture failed", t); }
    }

    public static int graphInstantiateWithFlags(MemorySegment phGraphExec, MemorySegment hGraph, long flags) {
        try { return (int) cuGraphInstantiateWithFlags.invokeExact(phGraphExec, hGraph, flags); }
        catch (Throwable t) { throw new RuntimeException("cuGraphInstantiateWithFlags failed", t); }
    }

    public static int graphLaunch(MemorySegment hGraphExec, MemorySegment hStream) {
        try { return (int) cuGraphLaunch.invokeExact(hGraphExec, hStream); }
        catch (Throwable t) { throw new RuntimeException("cuGraphLaunch failed", t); }
    }

    public static int graphExecDestroy(MemorySegment hGraphExec) {
        try { return (int) cuGraphExecDestroy.invokeExact(hGraphExec); }
        catch (Throwable t) { throw new RuntimeException("cuGraphExecDestroy failed", t); }
    }

    public static int graphDestroy(MemorySegment hGraph) {
        try { return (int) cuGraphDestroy.invokeExact(hGraph); }
        catch (Throwable t) { throw new RuntimeException("cuGraphDestroy failed", t); }
    }

    // --- NVRTC wrappers ---

    public static int createProgram(MemorySegment prog, MemorySegment src, MemorySegment name,
                                     int numHeaders, MemorySegment headers, MemorySegment headerNames) {
        try { return (int) nvrtcCreateProgram.invokeExact(prog, src, name, numHeaders, headers, headerNames); }
        catch (Throwable t) { throw new RuntimeException("nvrtcCreateProgram failed", t); }
    }

    public static int compileProgram(MemorySegment prog, int numOptions, MemorySegment options) {
        try { return (int) nvrtcCompileProgram.invokeExact(prog, numOptions, options); }
        catch (Throwable t) { throw new RuntimeException("nvrtcCompileProgram failed", t); }
    }

    public static int getPTXSize(MemorySegment prog, MemorySegment ptxSizeRet) {
        try { return (int) nvrtcGetPTXSize.invokeExact(prog, ptxSizeRet); }
        catch (Throwable t) { throw new RuntimeException("nvrtcGetPTXSize failed", t); }
    }

    public static int getPTX(MemorySegment prog, MemorySegment ptx) {
        try { return (int) nvrtcGetPTX.invokeExact(prog, ptx); }
        catch (Throwable t) { throw new RuntimeException("nvrtcGetPTX failed", t); }
    }

    public static int getProgramLogSize(MemorySegment prog, MemorySegment logSizeRet) {
        try { return (int) nvrtcGetProgramLogSize.invokeExact(prog, logSizeRet); }
        catch (Throwable t) { throw new RuntimeException("nvrtcGetProgramLogSize failed", t); }
    }

    public static int getProgramLog(MemorySegment prog, MemorySegment log) {
        try { return (int) nvrtcGetProgramLog.invokeExact(prog, log); }
        catch (Throwable t) { throw new RuntimeException("nvrtcGetProgramLog failed", t); }
    }

    public static int destroyProgram(MemorySegment prog) {
        try { return (int) nvrtcDestroyProgram.invokeExact(prog); }
        catch (Throwable t) { throw new RuntimeException("nvrtcDestroyProgram failed", t); }
    }
}
