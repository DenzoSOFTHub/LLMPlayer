package it.denzosoft.llmplayer.gpu;

import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.GGMLType;
import it.denzosoft.llmplayer.tensor.TensorData;

import java.io.*;
import java.lang.foreign.*;
import java.nio.file.*;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.Map;

/**
 * GPU cache of MoE experts (hybrid CPU/GPU expert computation).
 *
 * <p>A unit holds one expert's gate, up and down slices; the most frequently routed
 * (layer, expert) pairs stay resident. Two ways to use it:
 * <ul>
 * <li>Hybrid ({@link #launchResident} / {@link #finishResident}, the default for the engines):
 *     the resident experts of a layer are queued on the GPU, their outputs' downloads right behind
 *     them, while the caller computes the other experts on the CPU; a miss never waits for PCIe.
 *     With the old upload-on-miss path the main thread spent most of its time copying slices
 *     while the CPU cores idled (Qwen3-Coder-30B: slower than CPU-only).</li>
 * <li>{@link #computeExperts}: every selected expert on the GPU, uploading the missing ones.</li>
 * </ul>
 *
 * <p>Design (docs/optimization/gpu-slower-than-cpu.md, fixes F2 and F5):
 * <ul>
 * <li><b>Allocation</b>: units are packed into chunks of about 128 MiB, each slot sized for the
 *     largest slice of its projection over every layer. One allocation per slot paid the 2 MiB
 *     page rounding per slot and sized every slot for the largest slice of any projection (26-39%
 *     of the VRAM lost, and Qwen3-Coder spilling into shared memory).</li>
 * <li><b>Asynchronous promotion</b>: a promoted expert is copied from the mapped model into a
 *     page-locked staging ring and uploaded on a separate copy stream; it becomes resident (is
 *     used by a launch) only once its copy event has completed. The victim is evicted at enqueue,
 *     so nothing launched later reads it. At most one promotion per layer call, at most
 *     {@code moe.expert.cache.replace} replacements of resident experts per token (hysteresis:
 *     the newcomer must be routed twice as often), free units are filled as they come.</li>
 * <li><b>Speed-aware split</b>: per layer, when the wait for the resident experts exceeds 20% of
 *     the CPU experts' time for 8 calls in a row, one expert fewer is launched on the GPU (the
 *     least routed resident one runs on the CPU instead); when the wait stays under 5% the cap
 *     grows back. At idle GPU clocks a resident expert can be slower than the CPU cores.</li>
 * <li><b>Routing profile</b>: the routing counts are saved at close (see
 *     {@link #setRoutingProfile}) and loaded at the next start, which then uploads the top experts
 *     before the first token instead of learning them during decode.</li>
 * </ul>
 *
 * <p>Works for every expert quant type with an FP32-input matmul kernel; gate, up and down may use
 * different types.
 */
public class ExpertGpuCache implements it.denzosoft.llmplayer.inference.GpuExpertCache {

    private static final int MAX_EXPERTS_PER_TOKEN = 8; // Max top-K (GPT-OSS uses 4)
    /** Target chunk size (the 2 MiB rounding is paid once per chunk). */
    private static final long CHUNK_TARGET = Long.getLong("moe.expert.gpu.chunk.mb", 128L) << 20;
    private static final long PAGE = 2L << 20;
    private static final int RING = Math.max(1, Integer.getInteger("moe.expert.gpu.ring", 4));
    private static final int REPLACEMENTS_PER_TOKEN = Integer.getInteger("moe.expert.cache.replace", 4);
    private static final boolean SPLIT = !"false".equals(System.getProperty("moe.expert.gpu.split", "true"));
    private static final int DECAY_INTERVAL = 1 << 16;   // halve all counts every 64K selections
    private static final int ABSENT = -1, PENDING = -2;
    private static final int PROFILE_MAGIC = 0x4C4C5052; // "LLPR"
    private static final int PROFILE_VERSION = 1;

    private final CudaContext cudaContext;
    private final MemorySegment stream;
    private final long cudaBlockSize;
    private final int layers, experts, dim, efd;

    // Allocation
    private final long[] projBytes;          // capacity of the gate / up / down slot
    private final long unitBytes;
    private final int units;
    private final long[] slotPtr;            // 3 per unit
    private final long[] chunkPtrs;
    private final long deviceBytes;

    // Residency and routing counts, indexed by layer * experts + expert
    private final int[] count;
    private final int[] unitOf;              // unit, ABSENT or PENDING
    private final int[] unitKey;             // key held by a unit, or -1
    private int unitsUsed;
    private int selections;

    // Asynchronous promotion
    private final MemorySegment copyStream;
    private final MemorySegment[] ring, ringDone;
    private final int[] ringUnit, ringKey;
    private final boolean[] ringBusy;
    private final MemorySegment computeMark;
    private int lastPosition = Integer.MIN_VALUE, replacementsThisToken;

    // Pending hybrid call (launchResident -> finishResident)
    private FloatTensor pGate, pUp, pDown;
    private int[] pSelected;
    private int pCount, pLayer;
    private long tLaunchEnd;

    // Speed-aware split, per layer
    private final int[] cap, slowStreak, fastStreak;

    // Kernels
    private final Map<GGMLType, MemorySegment> kernels = new EnumMap<>(GGMLType.class);
    private final MemorySegment siluMulFunc, swigluOaiFunc, accumFunc;
    private final KernelParams matmulPB, actPB, accumPB;
    private final Arena arena = Arena.ofShared();
    private final Map<Long, Long> biasPtrs = new HashMap<>(); // GPT-OSS expert biases

    // Activation buffers, allocated up front (before the caller measures the free VRAM again)
    private final long gpuInputBuf;
    private final MemorySegment hostBuf;     // page-locked input staging
    private final MemorySegment outStaging;  // page-locked output staging, one dim-vector per slot k
    private final long[] gpuGateOutBufs = new long[MAX_EXPERTS_PER_TOKEN];
    private final long[] gpuUpOutBufs = new long[MAX_EXPERTS_PER_TOKEN];
    private final long[] gpuDownOutBufs = new long[MAX_EXPERTS_PER_TOKEN];

    // Multi-expert launch (F5 step 5): one launch per projection for all of a layer's resident
    // experts (blockIdx.y = expert), weight / input / output pointers read from a device table.
    // Per expert the host used to issue gate, up, activation and down launches plus a download.
    private static final boolean MULTI = !"false".equals(System.getProperty("moe.expert.gpu.multi", "true"));
    private final Map<GGMLType, MemorySegment> multiKernels = new EnumMap<>(GGMLType.class);
    private boolean multiBroken = !MULTI;
    private final long gateBlock, upBlock, downBlock;   // [MAX][efd], [MAX][efd], [MAX][dim]
    private final long ptrTable;                        // [Wgate | Wup | Wdown | input | outGate | outUp | outDown] x MAX
    private final MemorySegment ptrHost;                // page-locked, the table's host image
    private final KernelParams multiPB;
    // dp4a for cached K-quant slots (F5 step 5): the multi kernels generated from the dp4a sources.
    // Gate and up read one Q8_1 quantization of the layer input; down reads one quantization of
    // all the activated experts (efd is a multiple of 32, so expert j's blocks start at j * efd / 32).
    private static final boolean DP4A = !"false".equals(System.getProperty("moe.expert.gpu.dp4a", "true"))
        && !"false".equals(System.getProperty("cuda.dp4a", "true"));
    private static final boolean DP4A_Q6 = "true".equals(System.getProperty("cuda.dp4a.q6", "false"));
    private static final boolean DP4A_Q3 = !"false".equals(System.getProperty("cuda.dp4a.q3", "true"));
    private static final boolean DP4A_Q5 = !"false".equals(System.getProperty("cuda.dp4a.q5", "true"));
    private final boolean dp4aOk;
    private final Map<GGMLType, MemorySegment> dp4aMultiKernels = new EnumMap<>(GGMLType.class);
    private final java.util.Set<GGMLType> dp4aBroken = java.util.EnumSet.noneOf(GGMLType.class);
    private MemorySegment quantizeFunc;
    private final long q8In, q8Down, bQ8In, bQ8Down;   // Q8_1 blocks, 40 bytes per 32 values
    private final KernelParams quantPB;
    private final int[] slotOfJ = new int[MAX_EXPERTS_PER_TOKEN];
    private int pJ;
    private boolean pMulti;

    // Batched prefill (F6): resident experts over the chunk's tokens. Buffers allocated up front.
    private static final int BATCH_TOKENS = Math.max(1, Integer.getInteger("prefill.batch", 64));
    private final int batchSlots = BATCH_TOKENS * MAX_EXPERTS_PER_TOKEN;
    private final long bIn, bGate, bUp, bDown, bTable;
    private final MemorySegment bInHost, bOutHost, bTableHost;
    private int[] bSegExpert = new int[0], bSegStart = new int[0], bSegLen = new int[0];
    private int bSegs, bSlotsTotal;
    private int[] bGpuSlotArr = new int[64];
    private int bGpuSlotCount;

    // Stats
    private long hits, misses, promotions, promotionsSkipped, capped;
    private Path profileFile;
    private boolean closed;

    /**
     * @param maxCacheBytes budget for the expert weights (device bytes, including page rounding)
     * @param projBytes     slot capacity of the gate, up and down slice (256-byte aligned)
     * @param layers        layer count (routing tables)
     * @param experts       experts per layer
     * @param dim           model dimension
     * @param efd           expert FFN dimension
     * @param types         expert quant types (their kernels are compiled now)
     */
    public ExpertGpuCache(CudaContext cudaContext, long maxCacheBytes, long[] projBytes,
                          int layers, int experts, int dim, int efd, GGMLType[] types) {
        this.cudaContext = cudaContext;
        this.stream = cudaContext.getStream();
        this.layers = layers;
        this.experts = experts;
        this.dim = dim;
        this.efd = efd;
        this.projBytes = projBytes.clone();
        this.unitBytes = projBytes[0] + projBytes[1] + projBytes[2];
        this.cudaBlockSize = Math.min(256, cudaContext.getDeviceInfo().maxWorkGroupSize());
        this.siluMulFunc = cudaContext.compileKernel("kernels/cuda/silu_mul.cu", "silu_mul");
        this.swigluOaiFunc = cudaContext.compileKernel("kernels/cuda/swiglu_oai.cu", "swiglu_oai");
        this.accumFunc = cudaContext.compileKernel("kernels/cuda/accumulate.cu", "accumulate");
        for (GGMLType t : types) if (t != null) kernel(t);
        this.matmulPB = new KernelParams(arena, 6);
        this.actPB = new KernelParams(arena, 3);
        this.accumPB = new KernelParams(arena, 3);

        // Activation buffers and staging first: they are small, and allocating them lazily later
        // would take memory from beyond the budget the caller measured.
        this.gpuInputBuf = cudaContext.allocBuffer((long) Math.max(dim, efd) * Float.BYTES);
        long f = Float.BYTES;
        this.gateBlock = cudaContext.allocBuffer((long) MAX_EXPERTS_PER_TOKEN * efd * f);
        this.upBlock = cudaContext.allocBuffer((long) MAX_EXPERTS_PER_TOKEN * efd * f);
        this.downBlock = cudaContext.allocBuffer((long) MAX_EXPERTS_PER_TOKEN * dim * f);
        for (int i = 0; i < MAX_EXPERTS_PER_TOKEN; i++) {
            gpuGateOutBufs[i] = gateBlock + (long) i * efd * f;
            gpuUpOutBufs[i] = upBlock + (long) i * efd * f;
            gpuDownOutBufs[i] = downBlock + (long) i * dim * f;
        }
        int m = MAX_EXPERTS_PER_TOKEN;
        this.dp4aOk = DP4A && cudaBlockSize == 256 && dim % 32 == 0 && efd % 32 == 0;
        long dimQ8 = (long) (dim / 32) * 40, efdQ8 = (long) (efd / 32) * 40;
        this.q8In = dp4aOk ? cudaContext.allocBuffer(dimQ8) : 0;
        this.q8Down = dp4aOk ? cudaContext.allocBuffer(m * efdQ8) : 0;
        this.quantPB = new KernelParams(arena, 3);
        this.ptrTable = cudaContext.allocBuffer(9L * m * 8);
        this.ptrHost = cudaContext.allocPinnedHost(9L * m * 8);
        for (int j = 0; j < m; j++) {
            ptrHost.set(ValueLayout.JAVA_LONG, (3L * m + j) * 8, gpuInputBuf);
            ptrHost.set(ValueLayout.JAVA_LONG, (4L * m + j) * 8, gpuGateOutBufs[j]);
            ptrHost.set(ValueLayout.JAVA_LONG, (5L * m + j) * 8, gpuUpOutBufs[j]);
            ptrHost.set(ValueLayout.JAVA_LONG, (6L * m + j) * 8, gpuDownOutBufs[j]);
            ptrHost.set(ValueLayout.JAVA_LONG, (7L * m + j) * 8, q8In);
            ptrHost.set(ValueLayout.JAVA_LONG, (8L * m + j) * 8, q8Down + j * efdQ8);
        }
        cudaContext.writeBuffer(ptrTable, ptrHost, 9L * m * 8); // pinned: waits for the copy
        this.multiPB = new KernelParams(arena, 6);
        long bs = batchSlots;
        this.bIn = cudaContext.allocBuffer((long) BATCH_TOKENS * dim * f);
        this.bGate = cudaContext.allocBuffer(bs * efd * f);
        this.bUp = cudaContext.allocBuffer(bs * efd * f);
        this.bDown = cudaContext.allocBuffer(bs * dim * f);
        this.bTable = cudaContext.allocBuffer(9L * bs * 8);
        this.bQ8In = dp4aOk ? cudaContext.allocBuffer(BATCH_TOKENS * dimQ8) : 0;
        this.bQ8Down = dp4aOk ? cudaContext.allocBuffer(bs * efdQ8) : 0;
        this.bInHost = cudaContext.allocPinnedHost((long) BATCH_TOKENS * dim * f);
        this.bOutHost = cudaContext.allocPinnedHost(bs * dim * f);
        this.bTableHost = cudaContext.allocPinnedHost(9L * bs * 8);
        this.hostBuf = cudaContext.allocPinnedHost((long) Math.max(dim, efd) * Float.BYTES);
        this.outStaging = cudaContext.allocPinnedHost((long) MAX_EXPERTS_PER_TOKEN * dim * Float.BYTES);
        this.copyStream = cudaContext.createExtraStream();
        this.computeMark = cudaContext.createEvent(false);
        this.ring = new MemorySegment[RING];
        this.ringDone = new MemorySegment[RING];
        this.ringUnit = new int[RING];
        this.ringKey = new int[RING];
        this.ringBusy = new boolean[RING];
        for (int r = 0; r < RING; r++) {
            ring[r] = cudaContext.allocPinnedHost(unitBytes);
            ringDone[r] = cudaContext.createEvent(false);
        }

        // Expert weight chunks, each a whole number of units rounded up to the 2 MiB page
        int perChunk = (int) Math.max(1, CHUNK_TARGET / unitBytes);
        long chunkBytes = roundUp(perChunk * unitBytes, PAGE);
        java.util.List<long[]> chunks = new java.util.ArrayList<>(); // [ptr, units]
        long budget = maxCacheBytes, total = 0;
        int totalUnits = 0;
        long[] before = cudaContext.getMemoryInfo();
        while (budget >= unitBytes) {
            int n = budget >= chunkBytes ? perChunk : (int) (budget / unitBytes);
            long bytes = roundUp(n * unitBytes, PAGE);
            while (n > 0 && bytes > budget) { n--; bytes = roundUp(n * unitBytes, PAGE); }
            if (n <= 0) break;
            long ptr;
            try {
                // Verified allocation: a chunk that lands in shared system memory (WSL2
                // overcommit) is released and ends the cache there (F7).
                ptr = cudaContext.allocBufferChecked(bytes, "expert cache chunk");
            } catch (RuntimeException e) {
                break; // out of device memory: keep what we have
            }
            chunks.add(new long[] { ptr, n });
            budget -= bytes;
            total += bytes;
            totalUnits += n;
        }
        long[] after = cudaContext.getMemoryInfo();
        this.units = totalUnits;
        this.deviceBytes = total;
        this.chunkPtrs = new long[chunks.size()];
        this.slotPtr = new long[3 * units];
        int u = 0;
        for (int c = 0; c < chunks.size(); c++) {
            long base = chunks.get(c)[0];
            chunkPtrs[c] = base;
            for (int i = 0; i < chunks.get(c)[1]; i++, u++) {
                long off = base + i * unitBytes;
                slotPtr[3 * u] = off;
                slotPtr[3 * u + 1] = off + projBytes[0];
                slotPtr[3 * u + 2] = off + projBytes[0] + projBytes[1];
            }
        }
        this.count = new int[layers * experts];
        this.unitOf = new int[layers * experts];
        Arrays.fill(unitOf, ABSENT);
        this.unitKey = new int[units];
        Arrays.fill(unitKey, -1);
        this.cap = new int[layers];
        Arrays.fill(cap, MAX_EXPERTS_PER_TOKEN);
        this.slowStreak = new int[layers];
        this.fastStreak = new int[layers];

        System.out.println("  Expert GPU cache: " + units + " experts in " + chunks.size() + " chunks, "
            + (total >> 20) + " MiB (cuMemGetInfo delta " + ((before[0] - after[0]) >> 20) + " MiB; slots "
            + projBytes[0] + "/" + projBytes[1] + "/" + projBytes[2] + " B"
            + (dp4aOk ? "; dp4a for " + dp4aTypes(types) : "") + ")");
        // Compile the multi-expert kernels now: lazily they would compile inside the first token
        for (GGMLType t : types) {
            if (t == null) continue;
            if (!multiBroken) {
                try { multiKernel(t); } catch (RuntimeException e) { multiBroken = true; }
            }
            dp4aMulti(t);
        }
    }

    private static long roundUp(long v, long a) { return (v + a - 1) / a * a; }

    private static String dp4aTypes(GGMLType[] types) {
        java.util.Set<GGMLType> d = java.util.EnumSet.noneOf(GGMLType.class);
        for (GGMLType t : types) if (t != null && dp4aKernelFor(t) != null) d.add(t);
        return d.isEmpty() ? "none of the slot types" : d.toString();
    }

    @Override public int residentExperts() { return unitsUsed; }
    @Override public int capacityExperts() { return units; }
    @Override public long deviceBytes() { return deviceBytes; }
    @Override public long hits() { return hits; }
    @Override public long misses() { return misses; }

    /** Resource and kernel name of the FP32-input matmul for {@code type}, or null if none. */
    public static String[] kernelFor(GGMLType type) {
        switch (type) {
            case MXFP4: return new String[] {"kernels/cuda/matmul_mxfp4.cu", "matmul_mxfp4"};
            case Q4_K:  return new String[] {"kernels/cuda/matmul_q4_k.cu", "matmul_q4_k"};
            case Q5_K:  return new String[] {"kernels/cuda/matmul_q5_k.cu", "matmul_q5_k"};
            case Q6_K:  return new String[] {"kernels/cuda/matmul_q6_k.cu", "matmul_q6_k"};
            case Q3_K:  return new String[] {"kernels/cuda/matmul_q3_k.cu", "matmul_q3_k"};
            case Q8_0:  return new String[] {"kernels/cuda/matmul_q8_0.cu", "matmul_q8_0"};
            case Q4_0:  return new String[] {"kernels/cuda/matmul_q4_0.cu", "matmul_q4_0"};
            case Q5_0:  return new String[] {"kernels/cuda/matmul_q5_0.cu", "matmul_q5_0"};
            case Q5_1:  return new String[] {"kernels/cuda/matmul_q5_1.cu", "matmul_q5_1"};
            case IQ4_NL: return new String[] {"kernels/cuda/matmul_iq4_nl.cu", "matmul_iq4_nl"};
            case IQ4_XS: return new String[] {"kernels/cuda/matmul_iq4_xs.cu", "matmul_iq4_xs"};
            case F16:   return new String[] {"kernels/cuda/matmul_f16.cu", "matmul_f16"};
            case IQ1_M: return new String[] {"kernels/cuda/matmul_iq1_m.cu", "matmul_iq1_m"};
            case IQ2_XXS: return new String[] {"kernels/cuda/matmul_iq2_xxs.cu", "matmul_iq2_xxs"};
            case Q2_K:  return new String[] {"kernels/cuda/matmul_q2_k.cu", "matmul_q2_k"};
            default:    return null;
        }
    }

    private MemorySegment kernel(GGMLType type) {
        MemorySegment f = kernels.get(type);
        if (f == null) {
            String[] k = kernelFor(type);
            if (k == null) throw new IllegalStateException("no GPU expert kernel for " + type);
            f = cudaContext.compileKernel(k[0], k[1]);
            kernels.put(type, f);
        }
        return f;
    }

    // ------------------------------------------------------------------ hybrid path

    @Override
    public synchronized void noteToken(int position) {
        if (position != lastPosition) {
            lastPosition = position;
            replacementsThisToken = 0;
        }
    }

    @Override
    public synchronized int launchResident(
            FloatTensor gateExps, FloatTensor upExps, FloatTensor downExps, float[] input,
            int[] selectedExperts, int expertUsedCount, int layer, int dim, int expertFfnDim,
            boolean useSwigluOai, FloatTensor gateExpsBias, FloatTensor upExpsBias, FloatTensor downExpsBias) {
        publishCompleted();
        pGate = gateExps; pUp = upExps; pDown = downExps;
        pSelected = selectedExperts; pCount = expertUsedCount; pLayer = layer;
        int base = layer * experts;
        int mask = 0, resident = 0, valid = 0;
        for (int k = 0; k < expertUsedCount; k++) {
            int e = selectedExperts[k];
            if (e < 0) continue;
            valid++;
            count(base + e);
            if (unitOf[base + e] >= 0) { mask |= 1 << k; resident++; }
        }
        // Speed-aware split: at most cap[layer] experts on the GPU, the most routed ones
        while (SPLIT && resident > cap[layer]) {
            int worst = -1, worstCount = Integer.MAX_VALUE;
            for (int k = 0; k < expertUsedCount; k++) {
                if ((mask & (1 << k)) == 0) continue;
                int c = count[base + selectedExperts[k]];
                if (c < worstCount) { worstCount = c; worst = k; }
            }
            mask &= ~(1 << worst);
            resident--;
            capped++;
        }
        hits += resident;
        misses += valid - resident;
        if (mask == 0) { tLaunchEnd = System.nanoTime(); return 0; }

        MemorySegment.copy(input, 0, hostBuf, ValueLayout.JAVA_FLOAT, 0, dim);
        cudaContext.writeBufferAsync(gpuInputBuf, hostBuf, (long) dim * Float.BYTES);
        long dimBytes = (long) dim * Float.BYTES;
        pMulti = !multiBroken && gateExpsBias == null && upExpsBias == null && downExpsBias == null
            && launchMulti(gateExps, upExps, downExps, selectedExperts, expertUsedCount, mask, base, useSwigluOai);
        if (pMulti) {
            cudaContext.readBufferAsync(downBlock, outStaging, pJ * dimBytes);
            tLaunchEnd = System.nanoTime();
            return mask;
        }
        MemorySegment gateK = kernel(gateExps.type()), upK = kernel(upExps.type()), downK = kernel(downExps.type());
        long gateBias = gateExpsBias != null ? biasPtr(gateExpsBias, layer, 0) : 0;
        long upBias = upExpsBias != null ? biasPtr(upExpsBias, layer, 1) : 0;
        long downBias = downExpsBias != null ? biasPtr(downExpsBias, layer, 2) : 0;
        for (int k = 0; k < expertUsedCount; k++) {
            if ((mask & (1 << k)) == 0) continue;
            int e = selectedExperts[k];
            launchExpert(k, unitOf[base + e], e, dim, expertFfnDim, gateK, upK, downK,
                useSwigluOai, gateBias, upBias, downBias);
        }
        // Queue the downloads behind the kernels: finishResident then only synchronises
        for (int k = 0; k < expertUsedCount; k++) {
            if ((mask & (1 << k)) != 0) {
                cudaContext.readBufferAsync(gpuDownOutBufs[k], outStaging.asSlice(k * dimBytes, dimBytes), dimBytes);
            }
        }
        tLaunchEnd = System.nanoTime();
        return mask;
    }

    @Override
    public synchronized void finishResident(int mask, float[][] outPerExpert) {
        int layer = pLayer;
        if (mask != 0) {
            long t0 = System.nanoTime();
            long cpuNs = t0 - tLaunchEnd;
            cudaContext.finish();
            long waitNs = System.nanoTime() - t0;
            long dimBytes = (long) dim * Float.BYTES;
            if (pMulti) {
                for (int j = 0; j < pJ; j++) {
                    MemorySegment.copy(outStaging, ValueLayout.JAVA_FLOAT, j * dimBytes, outPerExpert[slotOfJ[j]], 0, dim);
                }
            } else {
                for (int k = 0; k < pCount; k++) {
                    if ((mask & (1 << k)) != 0) MemorySegment.copy(outStaging, ValueLayout.JAVA_FLOAT, k * dimBytes, outPerExpert[k], 0, dim);
                }
            }
            if (SPLIT) adjustCap(layer, waitNs, cpuNs, Integer.bitCount(mask));
        }
        promoteFromLastCall(mask);
        pGate = pUp = pDown = null;
    }

    /** One step of the per-layer GPU expert cap from the last call's wait and CPU time. */
    private void adjustCap(int layer, long waitNs, long cpuNs, int launched) {
        if (waitNs > 0.2 * cpuNs) {
            fastStreak[layer] = 0;
            if (++slowStreak[layer] >= 8) {
                slowStreak[layer] = 0;
                if (cap[layer] > 1) cap[layer] = Math.min(cap[layer], launched) - 1;
            }
        } else {
            slowStreak[layer] = 0;
            if (waitNs < 0.05 * cpuNs && ++fastStreak[layer] >= 8) {
                fastStreak[layer] = 0;
                if (cap[layer] < MAX_EXPERTS_PER_TOKEN) cap[layer]++;
            }
        }
    }

    /**
     * Promote the most frequently routed missing expert of the last call: into a free unit, or,
     * within the per-token replacement budget, over the least routed resident expert when the
     * newcomer is routed at least twice as often (hysteresis against churn).
     */
    private void promoteFromLastCall(int mask) {
        int base = pLayer * experts;
        int best = -1, bestCount = -1;
        for (int k = 0; k < pCount; k++) {
            int e = pSelected[k];
            if (e < 0 || unitOf[base + e] != ABSENT) continue; // resident, capped or in flight
            int c = count[base + e];
            if (c > bestCount) { bestCount = c; best = e; }
        }
        if (best < 0) return;
        boolean free = unitsUsed < units;
        int unit;
        if (free) {
            unit = victimUnit(Integer.MAX_VALUE, pSelected, pCount, pLayer);
        } else {
            if (replacementsThisToken >= REPLACEMENTS_PER_TOKEN) return;
            unit = victimUnit(bestCount / 2, pSelected, pCount, pLayer);
        }
        if (unit < 0) return;
        if (enqueue(unit, base + best, pGate, pUp, pDown, best, false) && !free) replacementsThisToken++;
    }

    /**
     * Queue the {@code mask} experts with one launch per projection (and one activation launch):
     * the resident experts are compacted to j = 0..h-1, their slot pointers written to the device
     * table, outputs in the contiguous blocks. Returns false (per-expert launches instead) when a
     * multi kernel cannot be built for a projection's type.
     */
    private boolean launchMulti(FloatTensor gateExps, FloatTensor upExps, FloatTensor downExps,
                                int[] selected, int n, int mask, int base, boolean useSwigluOai) {
        MemorySegment gK, uK, dK;
        try {
            gK = multiKernel(gateExps.type());
            uK = multiKernel(upExps.type());
            dK = multiKernel(downExps.type());
        } catch (RuntimeException e) {
            multiBroken = true;
            System.err.println("Expert GPU cache: multi-expert kernels unavailable (" + e.getMessage()
                + ") — launching per expert");
            return false;
        }
        int m = MAX_EXPERTS_PER_TOKEN, h = 0;
        for (int k = 0; k < n; k++) {
            if ((mask & (1 << k)) == 0) continue;
            int unit = unitOf[base + selected[k]];
            slotOfJ[h] = k;
            ptrHost.set(ValueLayout.JAVA_LONG, (long) h * 8, slotPtr[3 * unit]);
            ptrHost.set(ValueLayout.JAVA_LONG, ((long) m + h) * 8, slotPtr[3 * unit + 1]);
            ptrHost.set(ValueLayout.JAVA_LONG, (2L * m + h) * 8, slotPtr[3 * unit + 2]);
            h++;
        }
        pJ = h;
        cudaContext.writeBufferAsync(ptrTable, ptrHost, 3L * m * 8);
        long t = ptrTable;
        long rowsPerBlock = cudaBlockSize / 32;
        MemorySegment gD = dp4aMulti(gateExps.type()), uD = dp4aMulti(upExps.type()), dD = dp4aMulti(downExps.type());
        if (gD != null || uD != null) quantize(gpuInputBuf, q8In, dim);
        multiLaunch(gD != null ? gD : gK, t, t + (gD != null ? 7L : 3L) * m * 8, t + 4L * m * 8, efd, dim, h, rowsPerBlock);
        multiLaunch(uD != null ? uD : uK, t + (long) m * 8, t + (uD != null ? 7L : 3L) * m * 8, t + 5L * m * 8,
            efd, dim, h, rowsPerBlock);
        actPB.setLong(0, gateBlock).setLong(1, upBlock).setInt(2, h * efd);
        launch(useSwigluOai ? swigluOaiFunc : siluMulFunc, grid(h * efd), (int) cudaBlockSize, actPB);
        if (dD != null) quantize(gateBlock, q8Down, h * efd);
        multiLaunch(dD != null ? dD : dK, t + 2L * m * 8, t + (dD != null ? 8L : 4L) * m * 8, t + 6L * m * 8,
            dim, efd, h, rowsPerBlock);
        return true;
    }

    /** Quantize {@code n} floats (a multiple of 32) to Q8_1 blocks, one warp per block. */
    private void quantize(long in, long out, int n) {
        quantPB.setLong(0, in).setLong(1, out).setInt(2, n);
        launch(quantizeFunc, ((n / 32) + 7) / 8, 256, quantPB);
    }

    /** dp4a (Q8_1 input) kernel of {@code type}, as for {@code Dp4aMatmul}, or null. */
    private static String[] dp4aKernelFor(GGMLType type) {
        switch (type) {
            case Q4_K:   return new String[] {"kernels/cuda/matmul_q4_k_dp4a.cu", "matmul_q4_k_dp4a"};
            case Q5_K:   return DP4A_Q5 ? new String[] {"kernels/cuda/matmul_q5_k_dp4a.cu", "matmul_q5_k_dp4a"} : null;
            case Q6_K:   return DP4A_Q6 ? new String[] {"kernels/cuda/matmul_q6_k_dp4a.cu", "matmul_q6_k_dp4a"} : null;
            case Q3_K:   return DP4A_Q3 ? new String[] {"kernels/cuda/matmul_q3_k_dp4a.cu", "matmul_q3_k_dp4a"} : null;
            case Q5_0:   return new String[] {"kernels/cuda/matmul_q5_0_dp4a.cu", "matmul_q5_0_dp4a"};
            case Q8_0:   return new String[] {"kernels/cuda/matmul_q8_0_dp4a.cu", "matmul_q8_0_dp4a"};
            case IQ4_NL: return new String[] {"kernels/cuda/matmul_iq4_nl_dp4a.cu", "matmul_iq4_nl_dp4a"};
            case IQ4_XS: return new String[] {"kernels/cuda/matmul_iq4_xs_dp4a.cu", "matmul_iq4_xs_dp4a"};
            default:     return null;
        }
    }

    /** The multi-expert dp4a kernel for {@code type}, or null (FP32 multi kernel instead). */
    private MemorySegment dp4aMulti(GGMLType type) {
        if (!dp4aOk || dp4aBroken.contains(type)) return null;
        MemorySegment f = dp4aMultiKernels.get(type);
        if (f != null) return f;
        String[] k = dp4aKernelFor(type);
        if (k == null) { dp4aBroken.add(type); return null; }
        try {
            if (quantizeFunc == null) quantizeFunc = cudaContext.compileKernel("kernels/cuda/quantize_q8.cu", "quantize_q8");
            f = multiFrom(k[0], k[1]);
        } catch (RuntimeException e) {
            dp4aBroken.add(type);
            System.err.println("Expert GPU cache: dp4a kernel for " + type + " unavailable (" + e.getMessage() + ")");
            return null;
        }
        dp4aMultiKernels.put(type, f);
        return f;
    }

    private void multiLaunch(MemorySegment fn, long wTable, long inTable, long outTable, int rows, int cols, int h,
                             long rowsPerBlock) {
        multiPB.setLong(0, wTable).setLong(1, inTable).setLong(2, outTable).setInt(3, rows).setInt(4, cols).setInt(5, 0);
        int gx = (int) ((rows + rowsPerBlock - 1) / rowsPerBlock);
        int err = CudaBindings.launchKernel(fn, gx, h, 1, (int) cudaBlockSize, 1, 1, 0, stream, multiPB.ptrs(), MemorySegment.NULL);
        if (err != CudaBindings.CUDA_SUCCESS) throw new RuntimeException("ExpertGpuCache multi launch error: " + err);
    }

    /**
     * The multi-expert variant of {@code type}'s matmul kernel, generated from its source: the
     * {@code __global__} entry becomes a {@code __device__} body, and a new entry point reads the
     * weight, input and output pointers of expert {@code blockIdx.y} from device tables.
     */
    private MemorySegment multiKernel(GGMLType type) {
        MemorySegment f = multiKernels.get(type);
        if (f != null) return f;
        String[] k = kernelFor(type);
        if (k == null) throw new IllegalStateException("no GPU expert kernel for " + type);
        f = multiFrom(k[0], k[1]);
        multiKernels.put(type, f);
        return f;
    }

    /**
     * Build {@code name}'s multi-expert variant from its source: the {@code __global__} entry
     * becomes a {@code __device__} body, and a new entry point reads the weight, input and output
     * pointers of expert {@code blockIdx.y} from device tables. The kernel must take
     * {@code (weights, input, output, rows, cols, addToOutput)}; the weight and input types are
     * read from the signature with its comments removed (a dp4a signature carries
     * {@code "[(cols/32) * 40 bytes]"} in a comment).
     */
    private MemorySegment multiFrom(String res, String name) {
        String src = CudaContext.kernelSource(res);
        String sig = "extern \"C\" __global__ void " + name + "(";
        int i = src == null ? -1 : src.indexOf(sig);
        if (i < 0) throw new IllegalStateException("cannot find " + name + " in " + res);
        int open = i + sig.length();
        StringBuilder params = new StringBuilder();
        int depth = 1, j = open;
        while (j < src.length() && depth > 0) {
            char c = src.charAt(j);
            if (c == '/' && j + 1 < src.length() && src.charAt(j + 1) == '/') {
                int nl = src.indexOf('\n', j);
                j = nl < 0 ? src.length() : nl;
                continue;
            }
            if (c == '/' && j + 1 < src.length() && src.charAt(j + 1) == '*') {
                int end = src.indexOf("*/", j + 2);
                j = end < 0 ? src.length() : end + 2;
                continue;
            }
            if (c == '(') depth++;
            else if (c == ')') { if (--depth == 0) break; }
            params.append(c);
            j++;
        }
        String[] ps = params.toString().split(",");
        if (ps.length != 6) throw new IllegalStateException(name + ": expected 6 parameters, found " + ps.length);
        String wType = paramType(ps[0]), inType = paramType(ps[1]);
        String body = src.substring(0, i) + "__device__ __forceinline__ void " + name + "_body("
            + src.substring(open);
        String wrapper = "\nextern \"C\" __global__ void " + name + "_multi(\n"
            + "    const unsigned long long* __restrict__ wPtrs, const unsigned long long* __restrict__ inPtrs,\n"
            + "    const unsigned long long* __restrict__ outPtrs, const int rows, const int cols, const int addToOutput)\n"
            + "{\n    const int e = blockIdx.y;\n    " + name + "_body((" + wType + ") wPtrs[e], (" + inType + ") inPtrs[e], "
            + "(float*) outPtrs[e], rows, cols, addToOutput);\n}\n";
        return cudaContext.compileKernelSource(res + "#multi", body + wrapper, name + "_multi");
    }

    /** The type of one parameter declaration ({@code __restrict__} and the name dropped). */
    private static String paramType(String decl) {
        return decl.replace("__restrict__", " ").trim().replaceAll("\\s*\\w+$", "").replaceAll("\\s+", " ").trim();
    }

    // ------------------------------------------------------------------ batched prefill

    @Override
    public synchronized int launchResidentBatch(int layer, FloatTensor gateExps, FloatTensor upExps, FloatTensor downExps,
                                                float[][] xn, int nTokens, int[] used, int nUsed, int[] groupStart,
                                                int[] groupedSlots, int k, boolean[] onGpu, boolean useSwigluOai) {
        for (int i = 0; i < nUsed; i++) onGpu[used[i]] = false;
        bGpuSlotCount = 0;
        if (multiBroken || nTokens > BATCH_TOKENS || nTokens * k > batchSlots) return 0;
        publishCompleted();
        MemorySegment gK, uK, dK;
        try {
            gK = multiKernel(gateExps.type()); uK = multiKernel(upExps.type()); dK = multiKernel(downExps.type());
        } catch (RuntimeException e) {
            multiBroken = true;
            return 0;
        }
        int base = layer * experts;
        long f = Float.BYTES, bs = batchSlots;
        if (bSegExpert.length < nUsed) { bSegExpert = new int[nUsed]; bSegStart = new int[nUsed]; bSegLen = new int[nUsed]; }
        int entries = 0, segs = 0;
        for (int i = 0; i < nUsed; i++) {
            int e = used[i];
            int unit = unitOf[base + e];
            if (unit < 0) continue;
            int from = groupStart[e], m = groupStart[e + 1] - from;
            if (m <= 0) continue;
            bSegExpert[segs] = e; bSegStart[segs] = entries; bSegLen[segs] = m; segs++;
            for (int j = 0; j < m; j++) {
                int s = groupedSlots[from + j];
                long idx = entries + j;
                bTableHost.set(ValueLayout.JAVA_LONG, idx * 8, slotPtr[3 * unit]);
                bTableHost.set(ValueLayout.JAVA_LONG, (bs + idx) * 8, slotPtr[3 * unit + 1]);
                bTableHost.set(ValueLayout.JAVA_LONG, (2 * bs + idx) * 8, slotPtr[3 * unit + 2]);
                bTableHost.set(ValueLayout.JAVA_LONG, (3 * bs + idx) * 8, bIn + (long) (s / k) * dim * f);
                bTableHost.set(ValueLayout.JAVA_LONG, (4 * bs + idx) * 8, bGate + (long) s * efd * f);
                bTableHost.set(ValueLayout.JAVA_LONG, (5 * bs + idx) * 8, bUp + (long) s * efd * f);
                bTableHost.set(ValueLayout.JAVA_LONG, (6 * bs + idx) * 8, bDown + (long) s * dim * f);
                if (dp4aOk) {
                    bTableHost.set(ValueLayout.JAVA_LONG, (7 * bs + idx) * 8, bQ8In + (long) (s / k) * (dim / 32) * 40);
                    bTableHost.set(ValueLayout.JAVA_LONG, (8 * bs + idx) * 8, bQ8Down + (long) s * (efd / 32) * 40);
                }
                if (bGpuSlotCount == bGpuSlotArr.length) bGpuSlotArr = java.util.Arrays.copyOf(bGpuSlotArr, bGpuSlotCount * 2);
                bGpuSlotArr[bGpuSlotCount++] = s;
            }
            entries += m;
            onGpu[e] = true;
            hits += m;
        }
        bSegs = segs;
        if (segs == 0) return 0;
        for (int t = 0; t < nTokens; t++) MemorySegment.copy(xn[t], 0, bInHost, ValueLayout.JAVA_FLOAT, (long) t * dim * f, dim);
        cudaContext.writeBufferAsync(bIn, bInHost, (long) nTokens * dim * f);
        cudaContext.writeBufferAsync(bTable, bTableHost, (dp4aOk ? 9L : 7L) * bs * 8);
        long rowsPerBlock = cudaBlockSize / 32;
        MemorySegment gD = dp4aMulti(gateExps.type()), uD = dp4aMulti(upExps.type()), dD = dp4aMulti(downExps.type());
        if (gD != null || uD != null) quantize(bIn, bQ8In, nTokens * dim);
        long gIn = (gD != null ? 7 : 3) * bs * 8, uIn = (uD != null ? 7 : 3) * bs * 8;
        for (int g = 0; g < segs; g++) {
            long o = (long) bSegStart[g] * 8;
            multiLaunch(gD != null ? gD : gK, bTable + o, bTable + gIn + o, bTable + 4 * bs * 8 + o, efd, dim, bSegLen[g], rowsPerBlock);
            multiLaunch(uD != null ? uD : uK, bTable + bs * 8 + o, bTable + uIn + o, bTable + 5 * bs * 8 + o, efd, dim, bSegLen[g], rowsPerBlock);
        }
        // activation over every slot of the chunk (the CPU slots' rows are never read back)
        int nSlots = nTokens * k;
        actPB.setLong(0, bGate).setLong(1, bUp).setInt(2, nSlots * efd);
        launch(useSwigluOai ? swigluOaiFunc : siluMulFunc, grid(nSlots * efd), (int) cudaBlockSize, actPB);
        if (dD != null) quantize(bGate, bQ8Down, nSlots * efd);
        long dIn = (dD != null ? 8 : 4) * bs * 8;
        for (int g = 0; g < segs; g++) {
            long o = (long) bSegStart[g] * 8;
            multiLaunch(dD != null ? dD : dK, bTable + 2 * bs * 8 + o, bTable + dIn + o, bTable + 6 * bs * 8 + o, dim, efd, bSegLen[g], rowsPerBlock);
        }
        cudaContext.readBufferAsync(bDown, bOutHost, (long) nSlots * dim * f);
        bSlotsTotal = nSlots;
        return segs;
    }

    @Override
    public synchronized void finishResidentBatch(float[][] out) {
        if (bSegs == 0) return;
        cudaContext.finish();
        long f = Float.BYTES;
        for (int i = 0; i < bGpuSlotCount; i++) {
            int s = bGpuSlotArr[i];
            MemorySegment.copy(bOutHost, ValueLayout.JAVA_FLOAT, (long) s * dim * f, out[s], 0, dim);
        }
        bSegs = 0;
        bGpuSlotCount = 0;
    }

    // ------------------------------------------------------------------ promotion

    /**
     * Stage (layer, expert) {@code key}'s slices into a free ring slot and upload them into
     * {@code unit} on the copy stream. The victim is evicted now; the newcomer becomes resident
     * in {@link #publishCompleted()} once its copies have finished. Returns false when every ring
     * slot is still in flight (the promotion is skipped, not waited for) unless {@code wait}.
     */
    private boolean enqueue(int unit, int key, FloatTensor gate, FloatTensor up, FloatTensor down, int expert,
                            boolean wait) {
        int r = freeRingSlot(wait);
        if (r < 0) { promotionsSkipped++; return false; }
        int old = unitKey[unit];
        if (old >= 0) unitOf[old] = ABSENT; else unitsUsed++;
        unitKey[unit] = key;
        unitOf[key] = PENDING;
        MemorySegment st = ring[r];
        long gateUp = (long) efd * dim;
        long g = stageSlice(gate, expert, gateUp, st, 0, projBytes[0]);
        long p = stageSlice(up, expert, gateUp, st, projBytes[0], projBytes[1]);
        long d = stageSlice(down, expert, gateUp, st, projBytes[0] + projBytes[1], projBytes[2]);
        // The copy must not overtake kernels already queued on the compute stream that read the
        // victim unit (none today — a promotion follows a synchronisation — but keep it ordered).
        cudaContext.recordEvent(computeMark, stream);
        cudaContext.streamWaitEvent(copyStream, computeMark);
        copy(slotPtr[3 * unit], st, g);
        copy(slotPtr[3 * unit + 1], st.asSlice(projBytes[0]), p);
        copy(slotPtr[3 * unit + 2], st.asSlice(projBytes[0] + projBytes[1]), d);
        cudaContext.recordEvent(ringDone[r], copyStream);
        ringBusy[r] = true;
        ringUnit[r] = unit;
        ringKey[r] = key;
        return true;
    }

    private void copy(long dst, MemorySegment src, long bytes) {
        int err = CudaBindings.memcpyHtoDAsync(dst, src, bytes, copyStream);
        if (err != CudaBindings.CUDA_SUCCESS) throw new RuntimeException("ExpertGpuCache upload error: " + err);
    }

    private int freeRingSlot(boolean wait) {
        for (int attempt = 0; ; attempt++) {
            for (int r = 0; r < RING; r++) if (!ringBusy[r]) return r;
            publishCompleted();
            for (int r = 0; r < RING; r++) if (!ringBusy[r]) return r;
            if (!wait) return -1;
            cudaContext.waitEventParked(ringDone[attempt % RING], 20_000L);
        }
    }

    /** Make every completed promotion resident. */
    private void publishCompleted() {
        for (int r = 0; r < RING; r++) {
            if (ringBusy[r] && cudaContext.eventDone(ringDone[r])) {
                ringBusy[r] = false;
                if (unitOf[ringKey[r]] == PENDING && unitKey[ringUnit[r]] == ringKey[r]) {
                    unitOf[ringKey[r]] = ringUnit[r];
                    promotions++;
                }
            }
        }
    }

    private void waitAllPromotions() {
        for (int r = 0; r < RING; r++) if (ringBusy[r]) cudaContext.waitEventParked(ringDone[r], 20_000L);
        publishCompleted();
    }

    private void count(int key) {
        count[key]++;
        if (++selections >= DECAY_INTERVAL) {
            selections = 0;
            for (int i = 0; i < count.length; i++) count[i] >>= 1;
        }
    }

    /**
     * A free unit, else the resident unit with the lowest routing count below {@code maxCount}
     * that is neither in flight nor one of this call's experts; -1 when none qualifies.
     */
    private int victimUnit(int maxCount, int[] selected, int n, int layer) {
        int best = -1, bestCount = Integer.MAX_VALUE;
        for (int u = 0; u < units; u++) {
            int key = unitKey[u];
            if (key < 0) return u;
            if (unitOf[key] == PENDING) continue;
            if (key / experts == layer) {
                int e = key - layer * experts;
                boolean inUse = false;
                for (int k = 0; k < n; k++) if (selected[k] == e) { inUse = true; break; }
                if (inUse) continue;
            }
            int c = count[key];
            if (c < bestCount) { bestCount = c; best = u; }
        }
        return bestCount < maxCount ? best : -1;
    }

    // ------------------------------------------------------------------ full-GPU path

    /**
     * All selected experts of one layer on the GPU with a single synchronisation, uploading the
     * missing ones first (the compute stream waits for their copies). {@code gatePerExpert} /
     * {@code upPerExpert} are not written (kept for the signature).
     */
    public synchronized void computeExperts(
            FloatTensor gateExps, FloatTensor upExps, FloatTensor downExps,
            float[] input, int[] selectedExperts, float[] selectedWeights,
            int expertUsedCount, int layer, int dim, int expertFfnDim,
            float[][] gatePerExpert, float[][] upPerExpert, float[][] outPerExpert,
            boolean useSwigluOai,
            FloatTensor gateExpsBias, FloatTensor upExpsBias, FloatTensor downExpsBias) {
        publishCompleted();
        int base = layer * experts;
        long dimBytes = (long) dim * Float.BYTES;
        MemorySegment.copy(input, 0, hostBuf, ValueLayout.JAVA_FLOAT, 0, dim);
        cudaContext.writeBufferAsync(gpuInputBuf, hostBuf, dimBytes);
        MemorySegment gateK = kernel(gateExps.type()), upK = kernel(upExps.type()), downK = kernel(downExps.type());
        long gateBias = gateExpsBias != null ? biasPtr(gateExpsBias, layer, 0) : 0;
        long upBias = upExpsBias != null ? biasPtr(upExpsBias, layer, 1) : 0;
        long downBias = downExpsBias != null ? biasPtr(downExpsBias, layer, 2) : 0;
        for (int k = 0; k < expertUsedCount; k++) {
            int e = selectedExperts[k];
            if (e < 0) { Arrays.fill(outPerExpert[k], 0, dim, 0f); continue; }
            count(base + e);
            if (unitOf[base + e] == PENDING) waitAllPromotions();
            int unit = unitOf[base + e];
            if (unit >= 0) {
                hits++;
            } else {
                unit = victimUnit(Integer.MAX_VALUE, selectedExperts, expertUsedCount, layer);
                if (unit < 0) throw new IllegalStateException("expert GPU cache: no unit to evict");
                enqueue(unit, base + e, gateExps, upExps, downExps, e, true);
                waitAllPromotions();
                misses++;
            }
            launchExpert(k, unit, e, dim, expertFfnDim, gateK, upK, downK, useSwigluOai, gateBias, upBias, downBias);
            cudaContext.readBufferAsync(gpuDownOutBufs[k], outStaging.asSlice(k * dimBytes, dimBytes), dimBytes);
        }
        cudaContext.finish();
        for (int k = 0; k < expertUsedCount; k++) {
            if (selectedExperts[k] >= 0) MemorySegment.copy(outStaging, ValueLayout.JAVA_FLOAT, k * dimBytes, outPerExpert[k], 0, dim);
        }
    }

    /** Queue one resident expert (slots of {@code unit}) into the per-slot k buffers. */
    private void launchExpert(int k, int unit, int e, int dim, int expertFfnDim,
                              MemorySegment gateK, MemorySegment upK, MemorySegment downK, boolean useSwigluOai,
                              long gateBias, long upBias, long downBias) {
        launchMatmul(gateK, slotPtr[3 * unit], gpuInputBuf, gpuGateOutBufs[k], expertFfnDim, dim);
        launchMatmul(upK, slotPtr[3 * unit + 1], gpuInputBuf, gpuUpOutBufs[k], expertFfnDim, dim);
        if (gateBias != 0) accumulate(gpuGateOutBufs[k], gateBias + (long) e * expertFfnDim * Float.BYTES, expertFfnDim);
        if (upBias != 0) accumulate(gpuUpOutBufs[k], upBias + (long) e * expertFfnDim * Float.BYTES, expertFfnDim);
        actPB.setLong(0, gpuGateOutBufs[k]).setLong(1, gpuUpOutBufs[k]).setInt(2, expertFfnDim);
        launch(useSwigluOai ? swigluOaiFunc : siluMulFunc, grid(expertFfnDim), (int) cudaBlockSize, actPB);
        launchMatmul(downK, slotPtr[3 * unit + 2], gpuGateOutBufs[k], gpuDownOutBufs[k], dim, expertFfnDim);
        if (downBias != 0) accumulate(gpuDownOutBufs[k], downBias + (long) e * dim * Float.BYTES, dim);
    }

    // ------------------------------------------------------------------ routing profile

    @Override
    public synchronized void noteRouting(int layer, int[] selected, int n) {
        int base = layer * experts;
        for (int k = 0; k < n; k++) if (selected[k] >= 0) count(base + selected[k]);
    }

    /**
     * Use {@code file} as this model's routing profile: seed the counts from it when it exists and
     * upload the most routed experts now, and write the counts back at {@link #close()}.
     */
    @Override
    public synchronized void setRoutingProfile(Path file) {
        this.profileFile = file;
        if (file == null || !Files.isRegularFile(file)) return;
        long t0 = System.nanoTime();
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(file)))) {
            if (in.readInt() != PROFILE_MAGIC || in.readInt() != PROFILE_VERSION) return;
            int l = in.readInt(), e = in.readInt();
            if (l != layers || e != experts) return;
            for (int i = 0; i < count.length; i++) count[i] = in.readInt();
        } catch (IOException ex) {
            return;
        }
        profileLoaded = true;
        System.out.printf("  Expert GPU cache: routing profile loaded (%s, %.0f ms)%n",
            file.getFileName(), (System.nanoTime() - t0) / 1e6);
        warmStart();
    }

    private boolean profileLoaded;

    private FloatTensor[][] expertTensors;

    @Override
    public synchronized void bindExperts(FloatTensor[][] experts) {
        this.expertTensors = experts;
    }

    /** Upload the {@code units} most routed experts of the loaded profile. */
    private void warmStart() {
        if (!profileLoaded || expertTensors == null) return;
        profileLoaded = false;
        long t0 = System.nanoTime();
        Integer[] keys = new Integer[count.length];
        for (int i = 0; i < keys.length; i++) keys[i] = i;
        Arrays.sort(keys, (a, b) -> Integer.compare(count[b], count[a]));
        int n = 0;
        for (int i = 0; i < keys.length && n < units; i++) {
            int key = keys[i];
            if (count[key] == 0) break;
            int layer = key / experts, e = key - layer * experts;
            FloatTensor[] t = layer < expertTensors.length ? expertTensors[layer] : null;
            if (t == null || unitOf[key] != ABSENT) continue;
            int unit = victimUnit(Integer.MAX_VALUE, new int[0], 0, -1);
            if (unit < 0 || unitKey[unit] >= 0) break; // full
            enqueue(unit, key, t[0], t[1], t[2], e, true);
            n++;
        }
        waitAllPromotions();
        System.out.printf("  Expert GPU cache: warm start, %d experts uploaded in %.0f ms%n",
            n, (System.nanoTime() - t0) / 1e6);
    }

    private void saveProfile() {
        Path file = profileFile;
        if (file == null) return;
        long sum = 0;
        for (int c : count) sum += c;
        if (sum == 0) return;
        try {
            Files.createDirectories(file.toAbsolutePath().getParent());
            Path tmp = Files.createTempFile(file.toAbsolutePath().getParent(), "routing", ".tmp");
            try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(tmp)))) {
                out.writeInt(PROFILE_MAGIC);
                out.writeInt(PROFILE_VERSION);
                out.writeInt(layers);
                out.writeInt(experts);
                for (int c : count) out.writeInt(c);
            }
            Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException | RuntimeException ignored) {
            // a read-only cache directory only disables the warm start
        }
    }

    // ------------------------------------------------------------------ misc

    /** Device copy of a [experts × size] F32 expert-bias tensor, uploaded on first use. */
    private long biasPtr(FloatTensor bias, int layer, int proj) {
        long key = ((long) layer << 32) | ((long) proj << 16) | 0xFFFF;
        Long ptr = biasPtrs.get(key);
        if (ptr != null) return ptr;
        int n = (int) bias.size();
        float[] w = new float[n];
        for (int i = 0; i < n; i++) w[i] = bias.getFloat(i);
        long bytes = (long) n * Float.BYTES;
        long p = cudaContext.allocBuffer(bytes);
        try (Arena staging = Arena.ofConfined()) {
            MemorySegment h = staging.allocate(ValueLayout.JAVA_FLOAT, n);
            MemorySegment.copy(w, 0, h, ValueLayout.JAVA_FLOAT, 0, n);
            cudaContext.writeBuffer(p, h, bytes);
            cudaContext.finish(); // the confined staging is freed right after
        }
        biasPtrs.put(key, p);
        return p;
    }

    public synchronized String getStats() {
        long total = hits + misses;
        double hitRate = total > 0 ? (100.0 * hits / total) : 0;
        String s = String.format("Expert GPU cache: %d hits, %d misses (%.1f%% hit rate), %d promotions",
            hits, misses, hitRate, promotions);
        if (promotionsSkipped > 0) s += " (" + promotionsSkipped + " deferred)";
        if (capped > 0) {
            int sum = 0;
            for (int c : cap) sum += c;
            s += String.format(", %d experts moved to the CPU by the split (mean cap %.1f)", capped, sum / (double) cap.length);
        }
        return s;
    }

    public long getHits() { return hits; }
    public long getMisses() { return misses; }

    public synchronized void close() {
        if (closed) return;
        closed = true;
        saveProfile();
        try { cudaContext.finish(); cudaContext.syncStream(copyStream); } catch (RuntimeException ignored) { }
        for (int i = 0; i < chunkPtrs.length; i++) {
            if (chunkPtrs[i] != 0) {
                cudaContext.freeBuffer(chunkPtrs[i]);
                chunkPtrs[i] = 0;
            }
        }
        for (int r = 0; r < RING; r++) {
            cudaContext.freePinnedHost(ring[r]);
            cudaContext.destroyEvent(ringDone[r]);
        }
        cudaContext.destroyEvent(computeMark);
        cudaContext.destroyStream(copyStream);
        cudaContext.freeBuffer(gpuInputBuf);
        cudaContext.freeBuffer(gateBlock);
        cudaContext.freeBuffer(upBlock);
        cudaContext.freeBuffer(downBlock);
        cudaContext.freeBuffer(ptrTable);
        cudaContext.freePinnedHost(ptrHost);
        cudaContext.freeBuffer(bIn);
        cudaContext.freeBuffer(bGate);
        cudaContext.freeBuffer(bUp);
        cudaContext.freeBuffer(bDown);
        cudaContext.freeBuffer(bTable);
        if (dp4aOk) {
            cudaContext.freeBuffer(q8In);
            cudaContext.freeBuffer(q8Down);
            cudaContext.freeBuffer(bQ8In);
            cudaContext.freeBuffer(bQ8Down);
        }
        cudaContext.freePinnedHost(bInHost);
        cudaContext.freePinnedHost(bOutHost);
        cudaContext.freePinnedHost(bTableHost);
        for (long p : biasPtrs.values()) cudaContext.freeBuffer(p);
        biasPtrs.clear();
        cudaContext.freePinnedHost(hostBuf);
        cudaContext.freePinnedHost(outStaging);
        arena.close();
    }

    private void launchMatmul(MemorySegment fn, long gpuWeights, long gpuInput, long gpuOutput, int rows, int cols) {
        matmulPB.setLong(0, gpuWeights).setLong(1, gpuInput).setLong(2, gpuOutput)
                .setInt(3, rows).setInt(4, cols).setInt(5, 0);
        long rowsPerBlock = cudaBlockSize / 32;
        launch(fn, (int) ((rows + rowsPerBlock - 1) / rowsPerBlock), (int) cudaBlockSize, matmulPB);
    }

    private void accumulate(long y, long x, int n) {
        accumPB.setLong(0, y).setLong(1, x).setInt(2, n);
        launch(accumFunc, grid(n), (int) cudaBlockSize, accumPB);
    }

    private int grid(int n) { return (int) ((n + cudaBlockSize - 1) / cudaBlockSize); }

    private void launch(MemorySegment fn, int grid, int block, KernelParams p) {
        int err = CudaBindings.launchKernel(fn, grid, 1, 1, block, 1, 1, 0, stream, p.ptrs(), MemorySegment.NULL);
        if (err != CudaBindings.CUDA_SUCCESS) throw new RuntimeException("ExpertGpuCache CUDA error: " + err);
    }

    /**
     * Copy expert {@code expert}'s slice of {@code tensor} into {@code staging} at
     * {@code offset}; returns its byte size (at most {@code capacity}, the slot size).
     */
    private static long stageSlice(FloatTensor tensor, int expert, long elements, MemorySegment staging,
                                   long offset, long capacity) {
        // Geometry of THIS tensor's quant type (gate/up/down may differ)
        int blockSize = tensor.type().getBlockSize();
        int blockBytes = tensor.type().getTypeSize();
        long elementOffset = expert * elements;
        long numBlocks = elements / blockSize;
        long byteOffset = (elementOffset / blockSize) * blockBytes;
        long byteSize = numBlocks * blockBytes;
        if (byteSize > capacity) throw new IllegalStateException("expert slice larger than its cache slot");
        TensorData data = tensor.data();
        if (data instanceof it.denzosoft.llmplayer.tensor.MemorySegmentTensorData msData) {
            MemorySegment.copy(msData.segment(), byteOffset, staging, offset, byteSize);
        } else {
            byte[] chunk = new byte[(int) Math.min(byteSize, 65536)];
            long remaining = byteSize;
            long off = 0;
            while (remaining > 0) {
                int toRead = (int) Math.min(remaining, chunk.length);
                data.copyBytes(byteOffset + off, chunk, 0, toRead);
                MemorySegment.copy(chunk, 0, staging, ValueLayout.JAVA_BYTE, offset + off, toRead);
                off += toRead;
                remaining -= toRead;
            }
        }
        return byteSize;
    }
}
