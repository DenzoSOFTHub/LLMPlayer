package it.denzosoft.llmplayer.gpu;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

/**
 * Panama FFM bindings for OpenCL functions.
 * Loads libOpenCL dynamically (Linux: libOpenCL.so, macOS: OpenCL.framework, Windows: OpenCL.dll).
 */
public final class OpenCLBindings {

    private OpenCLBindings() {}

    // OpenCL constants
    public static final int CL_SUCCESS = 0;
    public static final int CL_DEVICE_TYPE_GPU = (1 << 2);
    public static final int CL_DEVICE_TYPE_ACCELERATOR = (1 << 3);
    public static final int CL_DEVICE_TYPE_ALL = 0xFFFFFFFF;

    // cl_device_info
    public static final int CL_DEVICE_NAME = 0x102B;
    public static final int CL_DEVICE_VENDOR = 0x102C;
    public static final int CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F;
    public static final int CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002;
    public static final int CL_DEVICE_TYPE = 0x1000;
    public static final int CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004;

    // cl_mem_flags
    public static final long CL_MEM_READ_ONLY = (1L << 2);
    public static final long CL_MEM_WRITE_ONLY = (1L << 1);
    public static final long CL_MEM_READ_WRITE = (1L << 0);
    public static final long CL_MEM_COPY_HOST_PTR = (1L << 5);

    // cl_command_queue_properties
    public static final long CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE = (1L << 0);

    // cl_program_build_info
    public static final int CL_PROGRAM_BUILD_LOG = 0x1183;

    private static final SymbolLookup LIB;
    private static final Linker LINKER = Linker.nativeLinker();

    static {
        SymbolLookup lib = null;
        String os = System.getProperty("os.name", "").toLowerCase();
        try {
            if (os.contains("mac")) {
                lib = SymbolLookup.libraryLookup("/System/Library/Frameworks/OpenCL.framework/OpenCL", Arena.global());
            } else if (os.contains("win")) {
                lib = SymbolLookup.libraryLookup("OpenCL", Arena.global());
            } else {
                // Linux: try common paths
                try {
                    lib = SymbolLookup.libraryLookup("libOpenCL.so.1", Arena.global());
                } catch (IllegalArgumentException e) {
                    try {
                        lib = SymbolLookup.libraryLookup("libOpenCL.so", Arena.global());
                    } catch (IllegalArgumentException e2) {
                        lib = SymbolLookup.libraryLookup("/usr/lib/x86_64-linux-gnu/libOpenCL.so.1", Arena.global());
                    }
                }
            }
        } catch (Exception e) {
            // OpenCL not available - lib stays null
        }
        LIB = lib;
    }

    public static boolean isAvailable() {
        return LIB != null;
    }

    private static MethodHandle findFunction(String name, FunctionDescriptor desc) {
        return LIB.find(name)
            .map(addr -> LINKER.downcallHandle(addr, desc))
            .orElseThrow(() -> new UnsupportedOperationException("OpenCL function not found: " + name));
    }

    // --- Platform / Device ---

    // cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms)
    private static final MethodHandle clGetPlatformIDs = findIfAvailable("clGetPlatformIDs",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    // cl_int clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type,
    //     cl_uint num_entries, cl_device_id* devices, cl_uint* num_devices)
    private static final MethodHandle clGetDeviceIDs = findIfAvailable("clGetDeviceIDs",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.JAVA_LONG,
            ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    // cl_int clGetDeviceInfo(cl_device_id device, cl_device_info param_name,
    //     size_t param_value_size, void* param_value, size_t* param_value_size_ret)
    private static final MethodHandle clGetDeviceInfo = findIfAvailable("clGetDeviceInfo",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
            ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    // --- Context / Queue ---

    // cl_context clCreateContext(cl_context_properties* properties, cl_uint num_devices,
    //     cl_device_id* devices, pfn_notify, void* user_data, cl_int* errcode_ret)
    private static final MethodHandle clCreateContext = findIfAvailable("clCreateContext",
        FunctionDescriptor.of(ValueLayout.ADDRESS,
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    // cl_command_queue clCreateCommandQueue(cl_context context, cl_device_id device,
    //     cl_command_queue_properties properties, cl_int* errcode_ret)
    private static final MethodHandle clCreateCommandQueue = findIfAvailable("clCreateCommandQueue",
        FunctionDescriptor.of(ValueLayout.ADDRESS,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS,
            ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));

    // --- Program / Kernel ---

    // cl_program clCreateProgramWithSource(cl_context context, cl_uint count,
    //     const char** strings, const size_t* lengths, cl_int* errcode_ret)
    private static final MethodHandle clCreateProgramWithSource = findIfAvailable("clCreateProgramWithSource",
        FunctionDescriptor.of(ValueLayout.ADDRESS,
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    // cl_int clBuildProgram(cl_program program, cl_uint num_devices, cl_device_id* device_list,
    //     const char* options, pfn_notify, void* user_data)
    private static final MethodHandle clBuildProgram = findIfAvailable("clBuildProgram",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    // cl_int clGetProgramBuildInfo(cl_program program, cl_device_id device,
    //     cl_program_build_info param_name, size_t param_value_size,
    //     void* param_value, size_t* param_value_size_ret)
    private static final MethodHandle clGetProgramBuildInfo = findIfAvailable("clGetProgramBuildInfo",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
            ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    // cl_kernel clCreateKernel(cl_program program, const char* kernel_name, cl_int* errcode_ret)
    private static final MethodHandle clCreateKernel = findIfAvailable("clCreateKernel",
        FunctionDescriptor.of(ValueLayout.ADDRESS,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    // --- Buffer ---

    // cl_mem clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size,
    //     void* host_ptr, cl_int* errcode_ret)
    private static final MethodHandle clCreateBuffer = findIfAvailable("clCreateBuffer",
        FunctionDescriptor.of(ValueLayout.ADDRESS,
            ValueLayout.ADDRESS, ValueLayout.JAVA_LONG,
            ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    // cl_int clEnqueueWriteBuffer(cl_command_queue queue, cl_mem buffer,
    //     cl_bool blocking_write, size_t offset, size_t size,
    //     const void* ptr, cl_uint num_events, cl_event* wait_list, cl_event* event)
    private static final MethodHandle clEnqueueWriteBuffer = findIfAvailable("clEnqueueWriteBuffer",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG,
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    // cl_int clEnqueueReadBuffer(cl_command_queue queue, cl_mem buffer,
    //     cl_bool blocking_read, size_t offset, size_t size,
    //     void* ptr, cl_uint num_events, cl_event* wait_list, cl_event* event)
    private static final MethodHandle clEnqueueReadBuffer = findIfAvailable("clEnqueueReadBuffer",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG,
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    // --- Kernel Execution ---

    // cl_int clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void* arg_value)
    private static final MethodHandle clSetKernelArg = findIfAvailable("clSetKernelArg",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
            ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));

    // cl_int clEnqueueNDRangeKernel(cl_command_queue queue, cl_kernel kernel,
    //     cl_uint work_dim, const size_t* global_work_offset,
    //     const size_t* global_work_size, const size_t* local_work_size,
    //     cl_uint num_events, cl_event* wait_list, cl_event* event)
    private static final MethodHandle clEnqueueNDRangeKernel = findIfAvailable("clEnqueueNDRangeKernel",
        FunctionDescriptor.of(ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    // cl_int clFinish(cl_command_queue queue)
    private static final MethodHandle clFinish = findIfAvailable("clFinish",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

    // --- Release ---

    // cl_int clReleaseMemObject(cl_mem memobj)
    private static final MethodHandle clReleaseMemObject = findIfAvailable("clReleaseMemObject",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

    // cl_int clReleaseKernel(cl_kernel kernel)
    private static final MethodHandle clReleaseKernel = findIfAvailable("clReleaseKernel",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

    // cl_int clReleaseProgram(cl_program program)
    private static final MethodHandle clReleaseProgram = findIfAvailable("clReleaseProgram",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

    // cl_int clReleaseCommandQueue(cl_command_queue queue)
    private static final MethodHandle clReleaseCommandQueue = findIfAvailable("clReleaseCommandQueue",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

    // cl_int clReleaseContext(cl_context context)
    private static final MethodHandle clReleaseContext = findIfAvailable("clReleaseContext",
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

    private static MethodHandle findIfAvailable(String name, FunctionDescriptor desc) {
        if (LIB == null) return null;
        try {
            return findFunction(name, desc);
        } catch (Exception e) {
            return null;
        }
    }

    // --- Public API wrappers ---

    public static int getPlatformIDs(int numEntries, MemorySegment platforms, MemorySegment numPlatforms) {
        try {
            return (int) clGetPlatformIDs.invokeExact(numEntries, platforms, numPlatforms);
        } catch (Throwable t) { throw new RuntimeException("clGetPlatformIDs failed", t); }
    }

    public static int getDeviceIDs(MemorySegment platform, long deviceType,
                                    int numEntries, MemorySegment devices, MemorySegment numDevices) {
        try {
            return (int) clGetDeviceIDs.invokeExact(platform, deviceType, numEntries, devices, numDevices);
        } catch (Throwable t) { throw new RuntimeException("clGetDeviceIDs failed", t); }
    }

    public static int getDeviceInfo(MemorySegment device, int paramName,
                                     long paramValueSize, MemorySegment paramValue,
                                     MemorySegment paramValueSizeRet) {
        try {
            return (int) clGetDeviceInfo.invokeExact(device, paramName,
                paramValueSize, paramValue, paramValueSizeRet);
        } catch (Throwable t) { throw new RuntimeException("clGetDeviceInfo failed", t); }
    }

    public static MemorySegment createContext(MemorySegment properties, int numDevices,
                                               MemorySegment devices, MemorySegment pfnNotify,
                                               MemorySegment userData, MemorySegment errcodeRet) {
        try {
            return (MemorySegment) clCreateContext.invokeExact(
                properties, numDevices, devices, pfnNotify, userData, errcodeRet);
        } catch (Throwable t) { throw new RuntimeException("clCreateContext failed", t); }
    }

    public static MemorySegment createCommandQueue(MemorySegment context, MemorySegment device,
                                                     long properties, MemorySegment errcodeRet) {
        try {
            return (MemorySegment) clCreateCommandQueue.invokeExact(
                context, device, properties, errcodeRet);
        } catch (Throwable t) { throw new RuntimeException("clCreateCommandQueue failed", t); }
    }

    public static MemorySegment createProgramWithSource(MemorySegment context, int count,
                                                          MemorySegment strings, MemorySegment lengths,
                                                          MemorySegment errcodeRet) {
        try {
            return (MemorySegment) clCreateProgramWithSource.invokeExact(
                context, count, strings, lengths, errcodeRet);
        } catch (Throwable t) { throw new RuntimeException("clCreateProgramWithSource failed", t); }
    }

    public static int buildProgram(MemorySegment program, int numDevices,
                                    MemorySegment deviceList, MemorySegment options,
                                    MemorySegment pfnNotify, MemorySegment userData) {
        try {
            return (int) clBuildProgram.invokeExact(
                program, numDevices, deviceList, options, pfnNotify, userData);
        } catch (Throwable t) { throw new RuntimeException("clBuildProgram failed", t); }
    }

    public static int getProgramBuildInfo(MemorySegment program, MemorySegment device,
                                           int paramName, long paramValueSize,
                                           MemorySegment paramValue, MemorySegment paramValueSizeRet) {
        try {
            return (int) clGetProgramBuildInfo.invokeExact(
                program, device, paramName, paramValueSize, paramValue, paramValueSizeRet);
        } catch (Throwable t) { throw new RuntimeException("clGetProgramBuildInfo failed", t); }
    }

    public static MemorySegment createKernel(MemorySegment program, MemorySegment kernelName,
                                               MemorySegment errcodeRet) {
        try {
            return (MemorySegment) clCreateKernel.invokeExact(program, kernelName, errcodeRet);
        } catch (Throwable t) { throw new RuntimeException("clCreateKernel failed", t); }
    }

    public static MemorySegment createBuffer(MemorySegment context, long flags,
                                               long size, MemorySegment hostPtr,
                                               MemorySegment errcodeRet) {
        try {
            return (MemorySegment) clCreateBuffer.invokeExact(context, flags, size, hostPtr, errcodeRet);
        } catch (Throwable t) { throw new RuntimeException("clCreateBuffer failed", t); }
    }

    public static int enqueueWriteBuffer(MemorySegment queue, MemorySegment buffer,
                                          int blockingWrite, long offset, long size,
                                          MemorySegment ptr, int numEvents,
                                          MemorySegment waitList, MemorySegment event) {
        try {
            return (int) clEnqueueWriteBuffer.invokeExact(
                queue, buffer, blockingWrite, offset, size, ptr, numEvents, waitList, event);
        } catch (Throwable t) { throw new RuntimeException("clEnqueueWriteBuffer failed", t); }
    }

    public static int enqueueReadBuffer(MemorySegment queue, MemorySegment buffer,
                                         int blockingRead, long offset, long size,
                                         MemorySegment ptr, int numEvents,
                                         MemorySegment waitList, MemorySegment event) {
        try {
            return (int) clEnqueueReadBuffer.invokeExact(
                queue, buffer, blockingRead, offset, size, ptr, numEvents, waitList, event);
        } catch (Throwable t) { throw new RuntimeException("clEnqueueReadBuffer failed", t); }
    }

    public static int setKernelArg(MemorySegment kernel, int argIndex,
                                    long argSize, MemorySegment argValue) {
        try {
            return (int) clSetKernelArg.invokeExact(kernel, argIndex, argSize, argValue);
        } catch (Throwable t) { throw new RuntimeException("clSetKernelArg failed", t); }
    }

    public static int enqueueNDRangeKernel(MemorySegment queue, MemorySegment kernel,
                                            int workDim, MemorySegment globalWorkOffset,
                                            MemorySegment globalWorkSize, MemorySegment localWorkSize,
                                            int numEvents, MemorySegment waitList, MemorySegment event) {
        try {
            return (int) clEnqueueNDRangeKernel.invokeExact(
                queue, kernel, workDim, globalWorkOffset, globalWorkSize, localWorkSize,
                numEvents, waitList, event);
        } catch (Throwable t) { throw new RuntimeException("clEnqueueNDRangeKernel failed", t); }
    }

    public static int finish(MemorySegment queue) {
        try {
            return (int) clFinish.invokeExact(queue);
        } catch (Throwable t) { throw new RuntimeException("clFinish failed", t); }
    }

    public static int releaseMemObject(MemorySegment memobj) {
        try {
            return (int) clReleaseMemObject.invokeExact(memobj);
        } catch (Throwable t) { throw new RuntimeException("clReleaseMemObject failed", t); }
    }

    public static int releaseKernel(MemorySegment kernel) {
        try {
            return (int) clReleaseKernel.invokeExact(kernel);
        } catch (Throwable t) { throw new RuntimeException("clReleaseKernel failed", t); }
    }

    public static int releaseProgram(MemorySegment program) {
        try {
            return (int) clReleaseProgram.invokeExact(program);
        } catch (Throwable t) { throw new RuntimeException("clReleaseProgram failed", t); }
    }

    public static int releaseCommandQueue(MemorySegment queue) {
        try {
            return (int) clReleaseCommandQueue.invokeExact(queue);
        } catch (Throwable t) { throw new RuntimeException("clReleaseCommandQueue failed", t); }
    }

    public static int releaseContext(MemorySegment context) {
        try {
            return (int) clReleaseContext.invokeExact(context);
        } catch (Throwable t) { throw new RuntimeException("clReleaseContext failed", t); }
    }
}
