package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.gpu.CudaBindings;
import it.denzosoft.llmplayer.gpu.CudaBufferManager;
import it.denzosoft.llmplayer.gpu.CudaContext;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.model.ModelWeights;
import it.denzosoft.llmplayer.model.TransformerLayerWeights;
import it.denzosoft.llmplayer.tensor.CudaFloatTensor;
import it.denzosoft.llmplayer.tensor.FloatTensor;

import java.lang.foreign.*;

/**
 * CUDA GPU-resident forward pass for dense pre-norm transformer architectures.
 * Keeps ALL computation on GPU — including attention (RoPE, KV cache, scores, softmax, weighted V).
 * No mid-layer CPU sync points. Only syncs at uploadX/downloadX boundaries.
 *
 * ZERO-ALLOCATION hot path: all kernel param buffers are pre-allocated in the constructor.
 * forwardLayer() only writes param values in-place and launches kernels — no Arena, no buildKernelParams.
 */
public class CudaForwardPass implements AutoCloseable {

    private final CudaContext cudaContext;
    private final CudaBufferManager bufferManager;
    private final Arena arena;

    // GPU-resident activation buffers (CUdeviceptr as long)
    private final long gpuX;
    private final long gpuXb;
    private final long gpuHb;
    private final long gpuHb2;
    private final long gpuQ;
    private final long gpuK;
    private final long gpuV;
    private final long gpuXb2;   // attention output
    private final long gpuPartialSums;

    // Per-layer norm weight buffers on GPU
    private final long[] gpuAttnNormWeights;
    private final long[] gpuFfnNormWeights;

    // GPU KV cache (per-layer)
    private final long[] gpuKeyCache;
    private final long[] gpuValueCache;

    // GPU RoPE tables
    private final long gpuCosTable;
    private final long gpuSinTable;

    // Pre-compiled CUDA functions (auxiliary kernels)
    private final MemorySegment rmsnormFusedFunc;
    private final MemorySegment siluMulFunc;
    private final MemorySegment ropeFunc;
    private final MemorySegment kvCacheUpdateFunc;
    private final MemorySegment attentionFullFunc;
    private final MemorySegment accumulateFunc;

    // Per-layer QKV bias buffers (null arrays if no biases)
    private final long[] gpuQBias;
    private final long[] gpuKBias;
    private final long[] gpuVBias;
    private final boolean hasBias;

    // Default CUDA stream
    private final MemorySegment defaultStream;

    // Output projection on GPU
    private final long gpuOutputNormWeights;
    private final long gpuLogits;           // [vocabSize] — output logits
    private final long gpuLogitsBytes;
    private final MemorySegment hostLogits;   // host staging for logits download

    // Dimensions
    private final int dim;
    private final int qDim;
    private final int kvDim;
    private final int ffnDim;
    private final int vocabSize;
    private final int headCount;
    private final int headCountKV;
    private final int headSize;
    private final int halfRope;
    private final int ropeType;
    private final float normEps;
    private final long blockSize;
    private final int numWorkGroups;
    private final int maxSeqLen;
    private final int blockCount;

    // Host-side staging buffers (only for uploadX/downloadX)
    private final MemorySegment hostX;

    // GPU-stored dynamic params: [position, seqLen] — kernels read via device pointer
    // Enables CUDA graph replay: graph captures the pointer (fixed), data changes per token
    private final long gpuTokenParams;
    private final MemorySegment hostTokenParams;

    // CUDA graph state
    private MemorySegment graphExec;          // null until first graph capture
    private final boolean graphAvailable;     // false if API missing or shared mem too large
    private final int graphAttnSharedMem;     // fixed attention shared mem for graph mode

    // === Pre-allocated kernel param buffers (ZERO-ALLOCATION hot path) ===

    /**
     * Contiguous buffer + ADDRESS pointer array for CUDA kernel params.
     * Each arg gets an 8-byte slot. Pointers are set up once in constructor.
     * Values are updated in-place before each launch.
     */
    private static class ParamBuffer {
        final MemorySegment args;  // contiguous: numArgs × 8 bytes
        final MemorySegment ptrs;  // void** array: numArgs × ADDRESS

        ParamBuffer(Arena arena, int numArgs) {
            args = arena.allocate(numArgs * 8L, 8);
            ptrs = arena.allocate(ValueLayout.ADDRESS, numArgs);
            for (int i = 0; i < numArgs; i++) {
                ptrs.setAtIndex(ValueLayout.ADDRESS, i, args.asSlice(i * 8L, 8));
            }
        }

        void setLong(int argIndex, long value) {
            args.set(ValueLayout.JAVA_LONG, argIndex * 8L, value);
        }

        void setInt(int argIndex, int value) {
            args.set(ValueLayout.JAVA_INT, argIndex * 8L, value);
        }

        void setFloat(int argIndex, float value) {
            args.set(ValueLayout.JAVA_FLOAT, argIndex * 8L, value);
        }
    }

    /**
     * Pre-computed matmul launch descriptor. All fields fixed for the model lifetime.
     */
    private static class MatmulLaunch {
        final MemorySegment function;
        final long weightPtr;
        final int gridDim;
        final int blockDim;
        final int sharedMem;
        final long inputPtr;
        final long outputPtr;
        final int rows;
        final int cols;
        final int addToOutput;

        MatmulLaunch(CudaFloatTensor tensor, long inputPtr, long outputPtr,
                     int rows, int cols, int addToOutput) {
            this.function = tensor.getCudaFunction();
            this.weightPtr = tensor.getGpuWeights();
            this.blockDim = tensor.getMatmulBlockDim(cols);
            this.gridDim = tensor.getMatmulGridDim(rows, cols);
            this.sharedMem = tensor.getMatmulSharedMem(cols);
            this.inputPtr = inputPtr;
            this.outputPtr = outputPtr;
            this.rows = rows;
            this.cols = cols;
            this.addToOutput = addToOutput;
        }
    }

    // Reusable param buffers (one per kernel type, shared across all layers)
    private final ParamBuffer matmulPB;   // 6 args: weights, input, output, rows, cols, addToOutput
    private final ParamBuffer normPB;     // 5 args: out, in, weights, size, eps
    private final ParamBuffer ropePB;     // 8 args: vec, cos, sin, nHeads, headSize, halfRope, tokenParams(ptr), ropeType
    private final ParamBuffer kvPB;       // 6 args: kCache, vCache, k, v, kvDim, tokenParams(ptr)
    private final ParamBuffer attnPB;     // 9 args: out, q, kCache, vCache, headCount, headCountKV, headSize, kvDim, tokenParams(ptr)
    private final ParamBuffer siluPB;     // 3 args: a, b, size
    private final ParamBuffer biasPB;    // 3 args: y, x, size (for QKV bias accumulate)

    // Per-layer matmul launch descriptors: [blockCount][7]
    // Index: 0=wq, 1=wk, 2=wv, 3=wo, 4=gate, 5=up, 6=down
    private final MatmulLaunch[][] layerMatmuls;

    // Output projection launch descriptor
    private final MatmulLaunch outputMatmul; // null if output not on GPU

    // Pre-computed auxiliary kernel grid sizes
    private final int normNumWarps;
    private final int normSharedMem;
    private final int ropeQGridDim;
    private final int ropeKGridDim;
    private final int kvUpdateGridDim;
    private final long attnBlockSize;
    private final int siluGridDim;
    private final int biasQGridDim;
    private final int biasKVGridDim;

    public CudaForwardPass(ModelConfig config, ModelWeights weights, CudaBufferManager bufferManager,
                            Attention attention, int maxSeqLen) {
        this.bufferManager = bufferManager;
        this.cudaContext = bufferManager.getCudaContext();
        this.arena = Arena.ofShared();
        this.maxSeqLen = maxSeqLen;
        this.defaultStream = cudaContext.getStream();

        this.dim = config.embeddingLength();
        this.qDim = config.headCount() * config.headSize();
        this.kvDim = config.kvDim();
        this.ffnDim = config.intermediateSize();
        this.vocabSize = config.vocabSize();
        this.headCount = config.headCount();
        this.headCountKV = config.headCountKV();
        this.headSize = config.headSize();
        this.normEps = config.normEps();
        this.blockCount = config.blockCount();

        RoPE rope = attention.getRope();
        this.halfRope = rope.getRopeDimCount() / 2;
        this.ropeType = rope.getRopeType();

        long maxWg = cudaContext.getDeviceInfo().maxWorkGroupSize();
        this.blockSize = Math.min(256, maxWg);
        this.numWorkGroups = (int) ((dim + blockSize - 1) / blockSize);

        // Allocate GPU activation buffers
        long fb = Float.BYTES;
        gpuX = bufferManager.createBuffer(dim * fb);
        gpuXb = bufferManager.createBuffer(dim * fb);
        gpuQ = bufferManager.createBuffer(qDim * fb);
        gpuK = bufferManager.createBuffer(kvDim * fb);
        gpuV = bufferManager.createBuffer(kvDim * fb);
        gpuXb2 = bufferManager.createBuffer(qDim * fb);
        gpuHb = bufferManager.createBuffer(ffnDim * fb);
        gpuHb2 = bufferManager.createBuffer(ffnDim * fb);
        gpuPartialSums = bufferManager.createBuffer((long) numWorkGroups * fb);

        // Allocate host staging buffer
        hostX = arena.allocate(ValueLayout.JAVA_FLOAT, dim);

        // Allocate GPU tokenParams buffer (2 ints: position + seqLen)
        gpuTokenParams = bufferManager.createBuffer(8);
        hostTokenParams = arena.allocate(ValueLayout.JAVA_INT, 2);

        // Upload RoPE tables to GPU
        float[] cosTable = rope.getCosTable();
        float[] sinTable = rope.getSinTable();
        gpuCosTable = uploadFloatArray(cosTable);
        gpuSinTable = uploadFloatArray(sinTable);

        // Allocate GPU KV cache (per-layer)
        gpuKeyCache = new long[blockCount];
        gpuValueCache = new long[blockCount];
        long kvLayerBytes = (long) maxSeqLen * kvDim * fb;
        for (int i = 0; i < blockCount; i++) {
            gpuKeyCache[i] = bufferManager.createBuffer(kvLayerBytes);
            gpuValueCache[i] = bufferManager.createBuffer(kvLayerBytes);
            cudaContext.fillBufferZero(gpuKeyCache[i], kvLayerBytes);
            cudaContext.fillBufferZero(gpuValueCache[i], kvLayerBytes);
        }

        // Compile auxiliary CUDA kernels
        rmsnormFusedFunc = cudaContext.compileKernel("kernels/cuda/rmsnorm.cu", "rmsnorm_fused");
        siluMulFunc = cudaContext.compileKernel("kernels/cuda/silu_mul.cu", "silu_mul");
        ropeFunc = cudaContext.compileKernel("kernels/cuda/rope.cu", "rope_apply");
        kvCacheUpdateFunc = cudaContext.compileKernel("kernels/cuda/attention.cu", "kv_cache_update");
        attentionFullFunc = cudaContext.compileKernel("kernels/cuda/attention.cu", "attention_full");
        accumulateFunc = cudaContext.compileKernel("kernels/cuda/accumulate.cu", "accumulate");

        // Upload per-layer norm weights to GPU
        gpuAttnNormWeights = new long[blockCount];
        gpuFfnNormWeights = new long[blockCount];
        for (int i = 0; i < blockCount; i++) {
            TransformerLayerWeights layer = weights.layers()[i];
            if (layer.attnNorm() != null) {
                gpuAttnNormWeights[i] = uploadNormWeights(layer.attnNorm(), dim);
            }
            if (layer.ffnNorm() != null) {
                gpuFfnNormWeights[i] = uploadNormWeights(layer.ffnNorm(), dim);
            }
        }

        // Upload per-layer QKV bias tensors to GPU (if present)
        TransformerLayerWeights firstLayer = weights.layers()[0];
        hasBias = firstLayer.qBias() != null;
        if (hasBias) {
            gpuQBias = new long[blockCount];
            gpuKBias = new long[blockCount];
            gpuVBias = new long[blockCount];
            for (int i = 0; i < blockCount; i++) {
                TransformerLayerWeights layer = weights.layers()[i];
                gpuQBias[i] = uploadBiasWeights(layer.qBias(), qDim);
                gpuKBias[i] = uploadBiasWeights(layer.kBias(), kvDim);
                gpuVBias[i] = uploadBiasWeights(layer.vBias(), kvDim);
            }
        } else {
            gpuQBias = null;
            gpuKBias = null;
            gpuVBias = null;
        }

        // Upload output norm weights and prepare output projection on GPU
        gpuOutputNormWeights = uploadNormWeights(weights.outputNorm(), dim);
        FloatTensor outputTensor = weights.output();
        if (outputTensor instanceof CudaFloatTensor) {
            CudaFloatTensor cudaOut = (CudaFloatTensor) outputTensor;
            gpuLogits = bufferManager.createBuffer((long) vocabSize * Float.BYTES);
            gpuLogitsBytes = (long) vocabSize * Float.BYTES;
            hostLogits = arena.allocate(ValueLayout.JAVA_FLOAT, vocabSize);
            outputMatmul = new MatmulLaunch(cudaOut, gpuXb, gpuLogits, vocabSize, dim, 0);
        } else {
            gpuLogits = 0;
            gpuLogitsBytes = 0;
            hostLogits = null;
            outputMatmul = null;
        }

        // === Pre-allocate reusable param buffers ===
        matmulPB = new ParamBuffer(arena, 6);
        normPB = new ParamBuffer(arena, 5);
        ropePB = new ParamBuffer(arena, 8);
        kvPB = new ParamBuffer(arena, 6);
        attnPB = new ParamBuffer(arena, 9);
        siluPB = new ParamBuffer(arena, 3);
        biasPB = new ParamBuffer(arena, 3);

        // Set fixed norm params (out=gpuXb, in=gpuX, size=dim, eps=normEps)
        normPB.setLong(0, gpuXb);
        normPB.setLong(1, gpuX);
        // normPB[2] = weights — set per launch
        normPB.setInt(3, dim);
        normPB.setFloat(4, normEps);

        // Set fixed rope params (cosTable, sinTable, headSize, halfRope, tokenParams, ropeType)
        // ropePB[0] = vec — set per launch
        ropePB.setLong(1, gpuCosTable);
        ropePB.setLong(2, gpuSinTable);
        // ropePB[3] = nHeads — set per launch
        ropePB.setInt(4, headSize);
        ropePB.setInt(5, halfRope);
        ropePB.setLong(6, gpuTokenParams); // device pointer — kernel reads position from tokenParams[0]
        ropePB.setInt(7, ropeType);

        // Set fixed kv update params (k=gpuK, v=gpuV, kvDim, tokenParams)
        // kvPB[0] = kCache — set per launch
        // kvPB[1] = vCache — set per launch
        kvPB.setLong(2, gpuK);
        kvPB.setLong(3, gpuV);
        kvPB.setInt(4, kvDim);
        kvPB.setLong(5, gpuTokenParams); // device pointer — kernel reads position from tokenParams[0]

        // Set fixed attention params (out=gpuXb2, q=gpuQ, headCount, headCountKV, headSize, kvDim, tokenParams)
        attnPB.setLong(0, gpuXb2);
        attnPB.setLong(1, gpuQ);
        // attnPB[2] = kCache — set per launch
        // attnPB[3] = vCache — set per launch
        attnPB.setInt(4, headCount);
        attnPB.setInt(5, headCountKV);
        attnPB.setInt(6, headSize);
        attnPB.setInt(7, kvDim);
        attnPB.setLong(8, gpuTokenParams); // device pointer — kernel reads seqLen from tokenParams[1]

        // Set fixed silu mul params (a=gpuHb, b=gpuHb2, size=ffnDim) — NEVER changes
        siluPB.setLong(0, gpuHb);
        siluPB.setLong(1, gpuHb2);
        siluPB.setInt(2, ffnDim);

        // Pre-compute auxiliary kernel grid sizes
        this.normNumWarps = (int) (blockSize / 32);
        this.normSharedMem = (normNumWarps + 1) * Float.BYTES;

        int ropeQTotal = headCount * halfRope;
        this.ropeQGridDim = (int) ((ropeQTotal + blockSize - 1) / blockSize);
        int ropeKTotal = headCountKV * halfRope;
        this.ropeKGridDim = (int) ((ropeKTotal + blockSize - 1) / blockSize);

        this.kvUpdateGridDim = (int) ((kvDim + blockSize - 1) / blockSize);
        this.attnBlockSize = Math.min(256, maxWg);
        this.siluGridDim = (int) ((ffnDim + blockSize - 1) / blockSize);
        this.biasQGridDim = (int) ((qDim + blockSize - 1) / blockSize);
        this.biasKVGridDim = (int) ((kvDim + blockSize - 1) / blockSize);

        // === Pre-compute per-layer matmul launch descriptors ===
        layerMatmuls = new MatmulLaunch[blockCount][7];
        for (int i = 0; i < blockCount; i++) {
            TransformerLayerWeights layer = weights.layers()[i];
            layerMatmuls[i][0] = new MatmulLaunch((CudaFloatTensor) layer.wq(), gpuXb, gpuQ, qDim, dim, 0);
            layerMatmuls[i][1] = new MatmulLaunch((CudaFloatTensor) layer.wk(), gpuXb, gpuK, kvDim, dim, 0);
            layerMatmuls[i][2] = new MatmulLaunch((CudaFloatTensor) layer.wv(), gpuXb, gpuV, kvDim, dim, 0);
            layerMatmuls[i][3] = new MatmulLaunch((CudaFloatTensor) layer.wo(), gpuXb2, gpuX, dim, qDim, 1);
            layerMatmuls[i][4] = new MatmulLaunch((CudaFloatTensor) layer.wGate(), gpuXb, gpuHb, ffnDim, dim, 0);
            layerMatmuls[i][5] = new MatmulLaunch((CudaFloatTensor) layer.wUp(), gpuXb, gpuHb2, ffnDim, dim, 0);
            layerMatmuls[i][6] = new MatmulLaunch((CudaFloatTensor) layer.wDown(), gpuHb, gpuX, dim, ffnDim, 1);
        }

        // CUDA graph: pre-compute fixed attention shared mem (max seqLen)
        // 48 KB per SM limit → max seqLen ~12256 for graph mode
        this.graphAttnSharedMem = (maxSeqLen + 32) * Float.BYTES;
        this.graphAvailable = !Boolean.getBoolean("cuda.nograph")
                && cudaContext.isGraphApiAvailable()
                && outputMatmul != null
                && graphAttnSharedMem <= 48 * 1024;
        if (graphAvailable) {
            System.err.println("CUDA graph: available (maxSeqLen=" + maxSeqLen
                    + ", attnSharedMem=" + graphAttnSharedMem + " bytes)");
        }
    }

    private long uploadBiasWeights(FloatTensor biasTensor, int size) {
        float[] w = new float[size];
        for (int i = 0; i < size; i++) {
            w[i] = biasTensor.getFloat(i);
        }
        return uploadFloatArray(w);
    }

    private long uploadNormWeights(FloatTensor normTensor, int size) {
        float[] w = new float[size];
        for (int i = 0; i < size; i++) {
            w[i] = normTensor.getFloat(i);
        }
        return bufferManager.uploadNormWeights(w);
    }

    private long uploadFloatArray(float[] data) {
        long bytes = (long) data.length * Float.BYTES;
        long ptr = bufferManager.createBuffer(bytes);
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment host = temp.allocate(ValueLayout.JAVA_FLOAT, data.length);
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_FLOAT, 0, data.length);
            cudaContext.writeBuffer(ptr, host, bytes);
        }
        return ptr;
    }

    /**
     * Check if the CUDA forward pass can handle this model configuration.
     */
    public static boolean isSupported(ModelConfig config, ModelWeights weights) {
        if (config.expertCount() > 0) return false;

        TransformerLayerWeights firstLayer = weights.layers()[0];
        if (firstLayer.attnNorm() == null) return false;
        if (firstLayer.ffnNorm() == null) return false;
        if (firstLayer.wqkv() != null) return false;
        if (firstLayer.postAttnNorm() != null) return false;
        if (firstLayer.postFfnNorm() != null) return false;
        if (firstLayer.wGate() == null) return false;

        // All weight tensors must be CudaFloatTensor
        if (!(firstLayer.wq() instanceof CudaFloatTensor)) return false;
        if (!(firstLayer.wk() instanceof CudaFloatTensor)) return false;
        if (!(firstLayer.wv() instanceof CudaFloatTensor)) return false;
        if (!(firstLayer.wo() instanceof CudaFloatTensor)) return false;
        if (!(firstLayer.wGate() instanceof CudaFloatTensor)) return false;
        if (!(firstLayer.wUp() instanceof CudaFloatTensor)) return false;
        if (!(firstLayer.wDown() instanceof CudaFloatTensor)) return false;

        return true;
    }

    public void uploadX(float[] x) {
        long t0 = 0;
        if (PROFILING) t0 = System.nanoTime();
        MemorySegment.copy(x, 0, hostX, ValueLayout.JAVA_FLOAT, 0, dim);
        cudaContext.writeBuffer(gpuX, hostX, (long) dim * Float.BYTES);
        if (PROFILING) profUpload += System.nanoTime() - t0;
    }

    public void downloadX(float[] x) {
        // cuMemcpyDtoH is synchronous — waits for all preceding GPU ops
        cudaContext.readBuffer(gpuX, hostX, (long) dim * Float.BYTES);
        MemorySegment.copy(hostX, ValueLayout.JAVA_FLOAT, 0, x, 0, dim);
    }

    /**
     * Compute final RMSNorm + output projection on GPU, download logits.
     * Returns true if logits were computed on GPU (caller skips CPU path).
     */
    public boolean forwardFinalLogits(float[] logits) {
        if (PROFILING) {
            profTokens++;
            if (profTokens % 10 == 0) printProfile();
        }
        if (outputMatmul == null) return false;
        long t0 = 0;
        if (PROFILING) t0 = System.nanoTime();

        // Final RMSNorm: gpuX → gpuXb (output norm weights)
        normPB.setLong(2, gpuOutputNormWeights);
        launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);

        // Output projection (write mode)
        launchMatmul(outputMatmul);

        // Download logits (cuMemcpyDtoH is synchronous)
        cudaContext.readBuffer(gpuLogits, hostLogits, gpuLogitsBytes);
        MemorySegment.copy(hostLogits, ValueLayout.JAVA_FLOAT, 0, logits, 0, vocabSize);

        if (PROFILING) profOutputMatmul += System.nanoTime() - t0;
        return true;
    }

    // Profiling accumulators (nanoseconds, across all layers and tokens)
    private long profAttnNorm, profQKV, profRopeKv, profAttn, profWo;
    private long profFfnNorm, profGateUp, profSiluDown, profTotal;
    private long profOutputMatmul, profUpload, profJavaOverhead;
    private int profCount, profTokens;
    private static final boolean PROFILING = Boolean.getBoolean("cuda.profile");

    /**
     * Print profiling results and reset counters.
     */
    public void printProfile() {
        if (profTokens == 0) return;
        System.err.printf("CUDA profile (%d tokens, %d layers):%n", profTokens, blockCount);
        System.err.printf("  attnNorm: %6.2f ms/tok  QKV: %6.2f ms/tok  ropeKv: %6.2f ms/tok%n",
            profAttnNorm / 1e6 / profTokens, profQKV / 1e6 / profTokens, profRopeKv / 1e6 / profTokens);
        System.err.printf("  attn:     %6.2f ms/tok  Wo:  %6.2f ms/tok%n",
            profAttn / 1e6 / profTokens, profWo / 1e6 / profTokens);
        System.err.printf("  ffnNorm:  %6.2f ms/tok  GateUp: %6.2f ms/tok  siluDown: %6.2f ms/tok%n",
            profFfnNorm / 1e6 / profTokens, profGateUp / 1e6 / profTokens, profSiluDown / 1e6 / profTokens);
        System.err.printf("  output:   %6.2f ms/tok  upload: %6.2f ms/tok  java: %6.2f ms/tok%n",
            profOutputMatmul / 1e6 / profTokens, profUpload / 1e6 / profTokens, profJavaOverhead / 1e6 / profTokens);
        System.err.printf("  TOTAL:    %6.2f ms/tok%n", profTotal / 1e6 / profTokens);
        profAttnNorm = profQKV = profRopeKv = profAttn = profWo = 0;
        profFfnNorm = profGateUp = profSiluDown = profTotal = 0;
        profOutputMatmul = profUpload = profJavaOverhead = 0;
        profCount = 0; profTokens = 0;
    }

    /**
     * Execute one transformer layer entirely on CUDA GPU.
     * ZERO-ALLOCATION hot path — all params are pre-allocated and updated in-place.
     */
    public void forwardLayer(InferenceState state, TransformerLayerWeights layerWeights,
                              int layerIdx, int position, Attention attention) {
        if (PROFILING) {
            forwardLayerProfiled(state, layerWeights, layerIdx, position, attention);
            return;
        }

        MatmulLaunch[] ml = layerMatmuls[layerIdx];

        // === Attention Phase ===

        // 1. RMSNorm (attnNorm): gpuX → gpuXb
        normPB.setLong(2, gpuAttnNormWeights[layerIdx]);
        launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);

        // 2. Q/K/V projections (write mode)
        launchMatmul(ml[0]); // wq
        launchMatmul(ml[1]); // wk
        launchMatmul(ml[2]); // wv

        // 2b. Add QKV bias if present (Qwen2)
        if (hasBias) {
            launchBias(gpuQ, gpuQBias[layerIdx], qDim, biasQGridDim);
            launchBias(gpuK, gpuKBias[layerIdx], kvDim, biasKVGridDim);
            launchBias(gpuV, gpuVBias[layerIdx], kvDim, biasKVGridDim);
        }

        // 3. RoPE on Q (position read from gpuTokenParams[0] by kernel)
        ropePB.setLong(0, gpuQ);
        ropePB.setInt(3, headCount);
        launchKernel(ropeFunc, ropeQGridDim, (int) blockSize, 0, ropePB.ptrs);

        // 3b. RoPE on K
        ropePB.setLong(0, gpuK);
        ropePB.setInt(3, headCountKV);
        launchKernel(ropeFunc, ropeKGridDim, (int) blockSize, 0, ropePB.ptrs);

        // 4. KV cache update (position read from gpuTokenParams[0] by kernel)
        kvPB.setLong(0, gpuKeyCache[layerIdx]);
        kvPB.setLong(1, gpuValueCache[layerIdx]);
        launchKernel(kvCacheUpdateFunc, kvUpdateGridDim, (int) blockSize, 0, kvPB.ptrs);

        // 5. Full attention (seqLen read from gpuTokenParams[1] by kernel)
        attnPB.setLong(2, gpuKeyCache[layerIdx]);
        attnPB.setLong(3, gpuValueCache[layerIdx]);
        int attnSharedMem = (position + 1 + 32) * Float.BYTES;
        launchKernel(attentionFullFunc, headCount, (int) attnBlockSize, attnSharedMem, attnPB.ptrs);

        // === FFN Phase ===

        // 6. Wo matmul (accumulate to residual)
        launchMatmul(ml[3]); // wo

        // 7. FFN RMSNorm: gpuX → gpuXb
        normPB.setLong(2, gpuFfnNormWeights[layerIdx]);
        launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);

        // 8. Gate + Up projections (write mode)
        launchMatmul(ml[4]); // gate
        launchMatmul(ml[5]); // up

        // 9. Fused SiLU + element-wise multiply (fully fixed params)
        launchKernel(siluMulFunc, siluGridDim, (int) blockSize, 0, siluPB.ptrs);

        // 10. Down projection (accumulate to residual)
        launchMatmul(ml[6]); // down
    }

    private void forwardLayerProfiled(InferenceState state, TransformerLayerWeights layerWeights,
                                       int layerIdx, int position, Attention attention) {
        MatmulLaunch[] ml = layerMatmuls[layerIdx];
        long t0 = System.nanoTime(), t1;
        long tTotal = t0;

        normPB.setLong(2, gpuAttnNormWeights[layerIdx]);
        launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);
        cudaContext.finish(); t1 = System.nanoTime(); profAttnNorm += t1 - t0; t0 = t1;

        launchMatmul(ml[0]);
        launchMatmul(ml[1]);
        launchMatmul(ml[2]);
        if (hasBias) {
            launchBias(gpuQ, gpuQBias[layerIdx], qDim, biasQGridDim);
            launchBias(gpuK, gpuKBias[layerIdx], kvDim, biasKVGridDim);
            launchBias(gpuV, gpuVBias[layerIdx], kvDim, biasKVGridDim);
        }
        cudaContext.finish(); t1 = System.nanoTime(); profQKV += t1 - t0; t0 = t1;

        ropePB.setLong(0, gpuQ);
        ropePB.setInt(3, headCount);
        launchKernel(ropeFunc, ropeQGridDim, (int) blockSize, 0, ropePB.ptrs);
        ropePB.setLong(0, gpuK);
        ropePB.setInt(3, headCountKV);
        launchKernel(ropeFunc, ropeKGridDim, (int) blockSize, 0, ropePB.ptrs);
        kvPB.setLong(0, gpuKeyCache[layerIdx]);
        kvPB.setLong(1, gpuValueCache[layerIdx]);
        launchKernel(kvCacheUpdateFunc, kvUpdateGridDim, (int) blockSize, 0, kvPB.ptrs);
        cudaContext.finish(); t1 = System.nanoTime(); profRopeKv += t1 - t0; t0 = t1;

        attnPB.setLong(2, gpuKeyCache[layerIdx]);
        attnPB.setLong(3, gpuValueCache[layerIdx]);
        int attnSharedMem = (position + 1 + 32) * Float.BYTES;
        launchKernel(attentionFullFunc, headCount, (int) attnBlockSize, attnSharedMem, attnPB.ptrs);
        cudaContext.finish(); t1 = System.nanoTime(); profAttn += t1 - t0; t0 = t1;

        launchMatmul(ml[3]);
        cudaContext.finish(); t1 = System.nanoTime(); profWo += t1 - t0; t0 = t1;

        normPB.setLong(2, gpuFfnNormWeights[layerIdx]);
        launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);
        cudaContext.finish(); t1 = System.nanoTime(); profFfnNorm += t1 - t0; t0 = t1;

        launchMatmul(ml[4]);
        launchMatmul(ml[5]);
        cudaContext.finish(); t1 = System.nanoTime(); profGateUp += t1 - t0; t0 = t1;

        launchKernel(siluMulFunc, siluGridDim, (int) blockSize, 0, siluPB.ptrs);
        launchMatmul(ml[6]);
        cudaContext.finish(); t1 = System.nanoTime(); profSiluDown += t1 - t0;

        profTotal += System.nanoTime() - tTotal;
        profCount++;
    }

    // --- CUDA Graph methods ---

    /**
     * Update position/seqLen in GPU memory. Must be called before forwardLayer() or forwardGraph().
     * Kernels read these values via device pointer (gpuTokenParams).
     */
    public void updateTokenParams(int position) {
        hostTokenParams.set(ValueLayout.JAVA_INT, 0, position);
        hostTokenParams.set(ValueLayout.JAVA_INT, 4, position + 1);
        cudaContext.writeBuffer(gpuTokenParams, hostTokenParams, 8);
    }

    /**
     * Execute all layers + output projection via CUDA graph.
     * First call: captures all kernel launches into a graph and instantiates it.
     * Subsequent calls: replays the graph with a single API call (~230 kernels).
     * Returns true if logits were computed; false to fall back to per-layer mode.
     *
     * Requires updateTokenParams() to be called first with the current position.
     * uploadX() must also be called first to load the input embedding.
     */
    public boolean forwardGraph(float[] logits) {
        if (!graphAvailable || PROFILING) return false;

        if (graphExec == null) {
            // First call: capture all kernel launches into a CUDA graph
            boolean capturing = false;
            try {
                cudaContext.beginCapture();
                capturing = true;

                for (int layer = 0; layer < blockCount; layer++) {
                    forwardLayerKernels(layer);
                }

                // Final RMSNorm + output projection
                normPB.setLong(2, gpuOutputNormWeights);
                launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);
                launchMatmul(outputMatmul);

                MemorySegment graph = cudaContext.endCapture();
                capturing = false;
                graphExec = cudaContext.instantiateGraph(graph);
                cudaContext.destroyGraph(graph);

                System.err.println("CUDA graph: captured " + blockCount + " layers + output projection");
            } catch (Exception e) {
                if (capturing) {
                    try { cudaContext.endCapture(); } catch (Exception ignored) {}
                }
                System.err.println("CUDA graph: capture failed — " + e.getMessage() + ", falling back to per-layer");
                graphExec = null;
                return false;
            }
        }

        // Launch graph (replays all captured kernels in one API call)
        cudaContext.launchGraph(graphExec);

        // Download logits (cuMemcpyDtoH is synchronous — waits for graph to complete)
        cudaContext.readBuffer(gpuLogits, hostLogits, gpuLogitsBytes);
        MemorySegment.copy(hostLogits, ValueLayout.JAVA_FLOAT, 0, logits, 0, vocabSize);

        return true;
    }

    /**
     * Launch all kernels for one transformer layer (used during graph capture).
     * Same as forwardLayer but without profiling and using fixed graphAttnSharedMem.
     */
    private void forwardLayerKernels(int layerIdx) {
        MatmulLaunch[] ml = layerMatmuls[layerIdx];

        // Attention norm
        normPB.setLong(2, gpuAttnNormWeights[layerIdx]);
        launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);

        // QKV projections
        launchMatmul(ml[0]);
        launchMatmul(ml[1]);
        launchMatmul(ml[2]);

        // QKV bias (Qwen2)
        if (hasBias) {
            launchBias(gpuQ, gpuQBias[layerIdx], qDim, biasQGridDim);
            launchBias(gpuK, gpuKBias[layerIdx], kvDim, biasKVGridDim);
            launchBias(gpuV, gpuVBias[layerIdx], kvDim, biasKVGridDim);
        }

        // RoPE on Q and K
        ropePB.setLong(0, gpuQ);
        ropePB.setInt(3, headCount);
        launchKernel(ropeFunc, ropeQGridDim, (int) blockSize, 0, ropePB.ptrs);
        ropePB.setLong(0, gpuK);
        ropePB.setInt(3, headCountKV);
        launchKernel(ropeFunc, ropeKGridDim, (int) blockSize, 0, ropePB.ptrs);

        // KV cache update
        kvPB.setLong(0, gpuKeyCache[layerIdx]);
        kvPB.setLong(1, gpuValueCache[layerIdx]);
        launchKernel(kvCacheUpdateFunc, kvUpdateGridDim, (int) blockSize, 0, kvPB.ptrs);

        // Full attention (shared mem fixed at maxSeqLen for graph compatibility)
        attnPB.setLong(2, gpuKeyCache[layerIdx]);
        attnPB.setLong(3, gpuValueCache[layerIdx]);
        launchKernel(attentionFullFunc, headCount, (int) attnBlockSize, graphAttnSharedMem, attnPB.ptrs);

        // Wo (accumulate to residual)
        launchMatmul(ml[3]);

        // FFN norm
        normPB.setLong(2, gpuFfnNormWeights[layerIdx]);
        launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);

        // Gate + Up projections
        launchMatmul(ml[4]);
        launchMatmul(ml[5]);

        // SiLU + element-wise multiply
        launchKernel(siluMulFunc, siluGridDim, (int) blockSize, 0, siluPB.ptrs);

        // Down projection (accumulate to residual)
        launchMatmul(ml[6]);
    }

    // --- Launch helpers (zero allocation) ---

    /**
     * Launch a matmul kernel using pre-computed descriptor. Updates shared matmulPB in-place.
     */
    private void launchMatmul(MatmulLaunch ml) {
        matmulPB.setLong(0, ml.weightPtr);
        matmulPB.setLong(1, ml.inputPtr);
        matmulPB.setLong(2, ml.outputPtr);
        matmulPB.setInt(3, ml.rows);
        matmulPB.setInt(4, ml.cols);
        matmulPB.setInt(5, ml.addToOutput);
        launchKernel(ml.function, ml.gridDim, ml.blockDim, ml.sharedMem, matmulPB.ptrs);
    }

    /**
     * Launch accumulate kernel for QKV bias: y[i] += bias[i].
     */
    private void launchBias(long y, long bias, int size, int gridDim) {
        biasPB.setLong(0, y);
        biasPB.setLong(1, bias);
        biasPB.setInt(2, size);
        launchKernel(accumulateFunc, gridDim, (int) blockSize, 0, biasPB.ptrs);
    }

    /**
     * Direct kernel launch bypassing CudaContext wrapper.
     * Pre-computed gridDim avoids division in the hot path.
     */
    private void launchKernel(MemorySegment function, int gridDim, int blockDim,
                               int sharedMem, MemorySegment params) {
        int err = CudaBindings.launchKernel(function,
            gridDim, 1, 1,
            blockDim, 1, 1,
            sharedMem, defaultStream,
            params, MemorySegment.NULL);
        if (err != CudaBindings.CUDA_SUCCESS) {
            throw new RuntimeException("CUDA error in launchKernel: " + err);
        }
    }

    @Override
    public void close() {
        if (graphExec != null) {
            try { cudaContext.destroyGraphExec(graphExec); } catch (Exception ignored) {}
        }
        try { cudaContext.freeBuffer(gpuTokenParams); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuX); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuXb); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuXb2); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuHb); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuHb2); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuQ); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuK); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuV); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuPartialSums); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuCosTable); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuSinTable); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuOutputNormWeights); } catch (Exception ignored) {}
        if (gpuLogits != 0) try { cudaContext.freeBuffer(gpuLogits); } catch (Exception ignored) {}
        if (gpuQBias != null) {
            for (long ptr : gpuQBias) { if (ptr != 0) try { cudaContext.freeBuffer(ptr); } catch (Exception ignored) {} }
            for (long ptr : gpuKBias) { if (ptr != 0) try { cudaContext.freeBuffer(ptr); } catch (Exception ignored) {} }
            for (long ptr : gpuVBias) { if (ptr != 0) try { cudaContext.freeBuffer(ptr); } catch (Exception ignored) {} }
        }
        for (long ptr : gpuAttnNormWeights) {
            if (ptr != 0) try { cudaContext.freeBuffer(ptr); } catch (Exception ignored) {}
        }
        for (long ptr : gpuFfnNormWeights) {
            if (ptr != 0) try { cudaContext.freeBuffer(ptr); } catch (Exception ignored) {}
        }
        for (long ptr : gpuKeyCache) {
            if (ptr != 0) try { cudaContext.freeBuffer(ptr); } catch (Exception ignored) {}
        }
        for (long ptr : gpuValueCache) {
            if (ptr != 0) try { cudaContext.freeBuffer(ptr); } catch (Exception ignored) {}
        }
        arena.close();
    }
}
