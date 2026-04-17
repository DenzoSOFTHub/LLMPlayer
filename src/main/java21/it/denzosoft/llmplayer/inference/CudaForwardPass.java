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
 * CUDA GPU-resident forward pass for dense transformer architectures (pre-norm and post-norm).
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
    private final MemorySegment argmaxPartialFunc;
    private final MemorySegment argmaxFinalFunc;
    private final MemorySegment fusedGateUpFunc; // null if not Q4_K

    // Merged QKV support (Phi-3/4 natively; runtime-fused for other architectures via -Dcuda.fuse.qkv=true)
    private final boolean hasMergedQKV;
    // Runtime QKV fusion: when the model ships separate wq/wk/wv, concatenate them at init
    // into a single GPU buffer and activate the merged-matmul + split_qkv path. Saves 2
    // kernel launches + 2 input-quant reads per layer per token. Opt-in (extra VRAM cost
    // and CPU-fallback disabled for the merged tensor).
    private final boolean runtimeFusedQkv;
    private final it.denzosoft.llmplayer.tensor.CudaFloatTensor[] runtimeMergedQkvTensors; // null if not fused
    private final long[] runtimeMergedQkvPtrs; // null if not fused
    private final long gpuQKV;           // GPU buffer for concatenated QKV output (0 if not merged)
    private final int qkvDim;            // qDim + kvDim + kvDim
    private final MemorySegment splitQkvFunc; // null if not merged
    private final ParamBuffer splitQkvPB;     // null if not merged
    private final int splitQkvGridDim;

    // Per-layer QKV bias buffers (null arrays if no biases)
    private final long[] gpuQBias;
    private final long[] gpuKBias;
    private final long[] gpuVBias;
    private final boolean hasBias;

    // Per-layer QK-norm weight buffers (null if no QK-norm)
    private final long[] gpuQNormWeights;
    private final long[] gpuKNormWeights;
    private final boolean hasQKNorm;
    private final MemorySegment perHeadNormFunc; // null if no QK-norm

    // Per-layer post-attention/FFN norm weight buffers (null if no post-norm, Gemma2/3)
    private final long[] gpuPostAttnNormWeights;
    private final long[] gpuPostFfnNormWeights;
    private final boolean hasPreNorm;
    private final boolean hasPostNorm;
    private final ParamBuffer postNormPB; // null if no post-norm

    // Default CUDA stream
    private final MemorySegment defaultStream;

    // Output projection on GPU
    private final long gpuOutputNormWeights;
    private final long gpuLogits;           // [vocabSize] — output logits
    private final long gpuLogitsBytes;
    private final MemorySegment hostLogits;   // host staging for logits download

    // GPU-side argmax buffers
    private final long gpuArgmaxPartialVal;   // partial max values (one per block)
    private final long gpuArgmaxPartialIdx;   // partial max indices (one per block)
    private final long gpuArgmaxResult;       // int[1] — final argmax index
    private final MemorySegment hostArgmaxResult; // host staging for result download
    private final int argmaxNumBlocks;        // number of blocks for partial argmax

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
    private final int noRopeLayerInterval; // SmolLM3/Llama4 NoPE: 0=all layers use RoPE

    // Host-side staging buffers (only for uploadX/downloadX)
    private final MemorySegment hostX;

    // Combined upload buffer: [dim floats (embedding)] + [2 ints (position, seqLen)]
    // Enables single cuMemcpyHtoD for both embedding and token params
    private final MemorySegment hostCombined;   // dim*4 + 8 bytes
    private final long gpuCombined;             // GPU buffer for combined upload

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
        // dp4a path: type code (0=ineligible, 4=Q4_K, 5=Q5_K, 6=Q6_K) and the
        // Q8_1 input buffer corresponding to inputPtr (0 if not dp4a-eligible).
        final int dp4aType;
        final long q8InputPtr;

        MatmulLaunch(CudaFloatTensor tensor, long inputPtr, long outputPtr,
                     int rows, int cols, int addToOutput) {
            this(tensor, inputPtr, outputPtr, rows, cols, addToOutput, 0L);
        }

        MatmulLaunch(CudaFloatTensor tensor, long inputPtr, long outputPtr,
                     int rows, int cols, int addToOutput, long q8InputPtr) {
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
            // Derive dp4a type from tensor's GGML type
            it.denzosoft.llmplayer.tensor.GGMLType t = tensor.type();
            if (t == it.denzosoft.llmplayer.tensor.GGMLType.Q4_K) this.dp4aType = 4;
            else if (t == it.denzosoft.llmplayer.tensor.GGMLType.Q5_K) this.dp4aType = 5;
            else if (t == it.denzosoft.llmplayer.tensor.GGMLType.Q6_K) this.dp4aType = 6;
            else if (t == it.denzosoft.llmplayer.tensor.GGMLType.Q3_K) this.dp4aType = 3;
            else if (t == it.denzosoft.llmplayer.tensor.GGMLType.Q5_0) this.dp4aType = 50;
            else if (t == it.denzosoft.llmplayer.tensor.GGMLType.Q8_0) this.dp4aType = 80;
            else if (t == it.denzosoft.llmplayer.tensor.GGMLType.IQ4_NL) this.dp4aType = 41;
            else if (t == it.denzosoft.llmplayer.tensor.GGMLType.IQ4_XS) this.dp4aType = 42;
            else this.dp4aType = 0;
            this.q8InputPtr = q8InputPtr;
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
    private final ParamBuffer fusedGateUpPB; // 8 args: gateWeights, upWeights, input, gateOutput, upOutput, gateRows, cols, addToOutput
    private final ParamBuffer argmaxPartialPB; // 4 args: data, partialVal, partialIdx, size
    private final ParamBuffer argmaxFinalPB;   // 4 args: partialVal, partialIdx, resultIdx, numPartials
    private final ParamBuffer perHeadNormPB;   // 4 args: vec, weights, headSize, eps (null if no QK-norm)

    // Per-layer matmul launch descriptors: [blockCount][7]
    // Index: 0=wq, 1=wk, 2=wv, 3=wo, 4=gate, 5=up, 6=down
    private final MatmulLaunch[][] layerMatmuls;

    // Per-layer fused gate+up weight pointers (only if fusedGateUpFunc != null)
    private final long[] fusedGateWeights; // gate weight GPU pointers per layer
    private final long[] fusedUpWeights;   // up weight GPU pointers per layer
    private final int fusedGateUpGridDim;
    private final int fusedGateUpBlockDim;
    private final boolean useFusedGateUp;

    // cuBLAS acceleration: pre-dequantized FP32 weight buffers (opt-in via -Dcuda.cublas=true)
    private it.denzosoft.llmplayer.gpu.CublasMatmul cublasMatmul; // null if disabled
    private long[][] cublasF32Weights;  // [layer][7] FP32 device pointers (0 if not dequantized)
    private long cublasF32Output;       // FP32 output projection weights (0 if not dequantized)
    private boolean useCublas;
    private boolean cublasUseFP16;
    private MemorySegment cublasConvertF16Func;

    // Granite scaling support
    private final float graniteResidualScale;   // 0 = not used
    private final float graniteAttentionScale;  // 0 = use standard 1/sqrt(headSize)
    private final float graniteLogitScale;      // 0 = not used, >0 = divide logits
    private final MemorySegment scaleFunc;      // scale_inplace kernel
    private final ParamBuffer scalePB;          // 3 args: x, scale, size
    private final int scaleDimGridDim;

    // Output projection launch descriptor
    private final MatmulLaunch outputMatmul; // null if output not on GPU

    // dp4a (llama.cpp-style int8 dot product) — opt-in via -Dcuda.dp4a=true.
    // After each FP32 buffer is produced (post attn/ffn norm, post attention, post silu),
    // a quantize_q8 kernel converts it to Q8_1 in a scratch buffer, then matmuls whose
    // weight type is Q4_K/Q5_K/Q6_K read the Q8_1 input and use __dp4a int8x4 dot product.
    private final boolean useDp4a;
    private final long gpuXbQ8;        // Q8_1 buffer for gpuXb input (size: dim/32 * 40)
    private final long gpuXb2Q8;       // Q8_1 buffer for gpuXb2 input (size: qDim/32 * 40)
    private final long gpuHbQ8;        // Q8_1 buffer for gpuHb input (size: ffnDim/32 * 40)
    private final MemorySegment quantizeFunc;
    private final MemorySegment dp4aQ4kFunc;
    private final MemorySegment dp4aQ4kMwFunc;  // multi-warp Q4_K (4 warps/row) — llama.cpp-style
    private final boolean useDp4aMw;
    // T2.1: multi-row Q4_K dp4a (4 rows per warp, 16 rows per block). Opt-in via
    // -Dcuda.q4k.mr4=true. Already present on Qwen35CudaForwardPass; mirrored here.
    private final MemorySegment dp4aQ4kMr4Func;
    private final boolean useDp4aQ4kMr4;
    // Port of llama.cpp's mul_mat_vec_q template specialized for Q4_K (mmvq).
    // Uses 4 warps × 32 lanes per row, vdr=2, sum-via-dp4a-trick. Reads our 40-byte Q8_1 format.
    private final MemorySegment mmvqQ4kFunc;
    private final boolean useMmvqQ4k;
    private final MemorySegment dp4aQ3kFunc;
    private final MemorySegment dp4aQ5kFunc;
    private final MemorySegment dp4aQ6kFunc;
    // Extended dp4a kernels for non-K-quant types (Q5_0, Q8_0, IQ4_NL, IQ4_XS).
    // These types fall through to FP32 in the original wiring; adding dp4a unlocks
    // major models like Gemma-3-1B (Q5_0), Phi-3-mini (IQ4_NL), Gemma-2 (IQ4_XS),
    // Qwen3-0.6B/1.7B (Q8_0).
    private final MemorySegment dp4aQ50Func;
    private final MemorySegment dp4aQ50SmemFunc;     // shared-memory Q8_1 input cache variant
    private final MemorySegment dp4aQ80Func;
    private final MemorySegment dp4aIq4nlFunc;
    private final MemorySegment dp4aIq4nlSmemFunc;   // shared-memory Q8_1 input cache variant
    private final MemorySegment dp4aIq4xsFunc;
    private final MemorySegment dp4aIq4xsMwFunc;  // multi-warp (4 warps × row, 1 row/block)
    // Fused RMSNorm + Q8_1 quantize (single kernel replaces 2-kernel pre-norm sequence).
    // Saves 1 launch + 1 HBM round-trip per layer (~2-3% measured on Llama-1B).
    private final MemorySegment rmsnormQuantizeFunc;
    private final ParamBuffer rmsnormQuantizePB;   // 6 args: normOut, qOut, in, w, size, eps
    private final boolean useFusedNormQuantize;
    // Fused gate+up dp4a: single kernel writes both gate and up outputs from one input read.
    // Halves input-vector reads and per-launch overhead. Q4_K-only.
    private final MemorySegment dp4aFusedGateUpFunc;
    private final ParamBuffer dp4aFusedGateUpPB;   // 7 args: gateW, upW, input, gateOut, upOut, rows, cols
    private final boolean useDp4aFusedGateUp;
    private final ParamBuffer quantizeXbPB;   // 3 args: gpuXb, gpuXbQ8, dim
    private final ParamBuffer quantizeXb2PB;  // 3 args: gpuXb2, gpuXb2Q8, qDim
    private final ParamBuffer quantizeHbPB;   // 3 args: gpuHb, gpuHbQ8, ffnDim
    private final ParamBuffer dp4aPB;         // 6 args: weights, q8Input, output, rows, cols, addToOutput
    private final int quantizeXbGridDim;
    private final int quantizeXb2GridDim;
    private final int quantizeHbGridDim;

    // Packed FFN support (Phi-3/4): wGate is null, wUp outputs 2*ffnDim, split into gate+up
    private final boolean hasPackedFFN;
    private final long gpuHbPacked;          // GPU buffer for packed gate+up output (0 if not packed)
    private final MemorySegment splitGateUpFunc; // null if not packed
    private final ParamBuffer splitGateUpPB;     // null if not packed
    private final int splitGateUpGridDim;

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
    private final int perHeadNormBlockDim;
    private final int perHeadNormSharedMem;

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
        this.noRopeLayerInterval = config.noRopeLayerInterval();

        RoPE rope = attention.getRope();
        this.halfRope = rope.getRopeDimCount() / 2;
        this.ropeType = rope.getRopeType();

        long maxWg = cudaContext.getDeviceInfo().maxWorkGroupSize();
        this.blockSize = Math.min(256, maxWg);
        this.numWorkGroups = (int) ((dim + blockSize - 1) / blockSize);

        // Allocate GPU activation buffers
        long fb = Float.BYTES;

        // Allocate combined contiguous buffer: [embedding (dim*4 bytes)] + [tokenParams (8 bytes)]
        // Enables single cuMemcpyHtoD for both embedding upload and token params update
        long combinedBytes = dim * fb + 8;
        gpuCombined = bufferManager.createBuffer(combinedBytes);
        gpuX = gpuCombined;                      // first dim*4 bytes
        gpuTokenParams = gpuCombined + dim * fb;  // last 8 bytes

        gpuXb = bufferManager.createBuffer(dim * fb);
        gpuQ = bufferManager.createBuffer(qDim * fb);
        gpuK = bufferManager.createBuffer(kvDim * fb);
        gpuV = bufferManager.createBuffer(kvDim * fb);
        gpuXb2 = bufferManager.createBuffer(qDim * fb);
        gpuHb = bufferManager.createBuffer(ffnDim * fb);
        gpuHb2 = bufferManager.createBuffer(ffnDim * fb);
        gpuPartialSums = bufferManager.createBuffer((long) numWorkGroups * fb);

        // Allocate host staging buffers
        hostX = arena.allocate(ValueLayout.JAVA_FLOAT, dim);
        hostCombined = arena.allocate(combinedBytes, 8);
        hostTokenParams = arena.allocate(ValueLayout.JAVA_INT, 2);

        // Upload RoPE tables to GPU
        float[] cosTable = rope.getCosTable();
        float[] sinTable = rope.getSinTable();
        gpuCosTable = uploadFloatArray(cosTable);
        gpuSinTable = uploadFloatArray(sinTable);

        // Determine how many layers are on GPU (partial offload support)
        this.gpuLayerCount = countGpuLayers(weights);

        // Allocate GPU KV cache (only for GPU layers)
        gpuKeyCache = new long[gpuLayerCount];
        gpuValueCache = new long[gpuLayerCount];
        long kvLayerBytes = (long) maxSeqLen * kvDim * fb;
        for (int i = 0; i < gpuLayerCount; i++) {
            gpuKeyCache[i] = bufferManager.createBuffer(kvLayerBytes);
            gpuValueCache[i] = bufferManager.createBuffer(kvLayerBytes);
            cudaContext.fillBufferZero(gpuKeyCache[i], kvLayerBytes);
            cudaContext.fillBufferZero(gpuValueCache[i], kvLayerBytes);
        }

        // Probe QK-norm, bias, and merged QKV presence early (needed for kernel compilation decisions)
        TransformerLayerWeights firstLayer = weights.layers()[0];
        hasBias = firstLayer.qBias() != null;
        hasQKNorm = firstLayer.qNorm() != null;
        hasPreNorm = firstLayer.attnNorm() != null;
        hasPostNorm = firstLayer.postAttnNorm() != null;
        boolean staticMergedQkv = firstLayer.wqkv() != null;

        // Runtime QKV fusion (opt-in via -Dcuda.fuse.qkv=true). For models that ship separate
        // Q/K/V weights (Llama, Qwen, Mistral, Gemma, Granite, etc.), concat byte-level into
        // a single GPU buffer so the existing merged-matmul + split_qkv path can fire. Saves
        // 2 matmul launches and 2 input-quant reads per layer per token.
        boolean fuseQkvReq = !staticMergedQkv
            && "true".equals(System.getProperty("cuda.fuse.qkv", "false"));
        it.denzosoft.llmplayer.tensor.CudaFloatTensor[] localMergedTensors = null;
        long[] localMergedPtrs = null;
        boolean fuseQkvOk = false;
        if (fuseQkvReq) {
            it.denzosoft.llmplayer.tensor.GGMLType qkvType = null;
            boolean eligible = true;
            for (int i = 0; i < gpuLayerCount && eligible; i++) {
                TransformerLayerWeights lw = weights.layers()[i];
                if (!(lw.wq() instanceof it.denzosoft.llmplayer.tensor.CudaFloatTensor)
                    || !(lw.wk() instanceof it.denzosoft.llmplayer.tensor.CudaFloatTensor)
                    || !(lw.wv() instanceof it.denzosoft.llmplayer.tensor.CudaFloatTensor)) {
                    eligible = false; break;
                }
                var q = (it.denzosoft.llmplayer.tensor.CudaFloatTensor) lw.wq();
                var k = (it.denzosoft.llmplayer.tensor.CudaFloatTensor) lw.wk();
                var v = (it.denzosoft.llmplayer.tensor.CudaFloatTensor) lw.wv();
                if (qkvType == null) qkvType = q.type();
                if (q.type() != qkvType || k.type() != qkvType || v.type() != qkvType) {
                    eligible = false; break;
                }
            }
            if (!eligible) {
                System.err.println("CUDA: QKV fusion requested but layers are heterogeneous — disabled");
            } else {
                localMergedTensors = new it.denzosoft.llmplayer.tensor.CudaFloatTensor[gpuLayerCount];
                localMergedPtrs = new long[gpuLayerCount];
                long totalVramAdded = 0L;
                for (int i = 0; i < gpuLayerCount; i++) {
                    TransformerLayerWeights lw = weights.layers()[i];
                    var q = (it.denzosoft.llmplayer.tensor.CudaFloatTensor) lw.wq();
                    var k = (it.denzosoft.llmplayer.tensor.CudaFloatTensor) lw.wk();
                    var v = (it.denzosoft.llmplayer.tensor.CudaFloatTensor) lw.wv();
                    long qB = q.getWeightsBytes();
                    long kB = k.getWeightsBytes();
                    long vB = v.getWeightsBytes();
                    long totalB = qB + kB + vB;
                    long mPtr = bufferManager.createBuffer(totalB);
                    cudaContext.copyBufferDtoD(mPtr, q.getGpuWeights(), qB);
                    cudaContext.copyBufferDtoD(mPtr + qB, k.getGpuWeights(), kB);
                    cudaContext.copyBufferDtoD(mPtr + qB + kB, v.getGpuWeights(), vB);
                    long totalSize = ((long) qDim + 2L * kvDim) * dim;
                    localMergedTensors[i] = new it.denzosoft.llmplayer.tensor.MergedQkvCudaTensor(
                        q, mPtr, totalSize, bufferManager);
                    localMergedPtrs[i] = mPtr;
                    totalVramAdded += totalB;
                }
                cudaContext.finish();
                fuseQkvOk = true;
                System.err.println("CUDA: runtime QKV fusion enabled (" + gpuLayerCount
                    + " layers, type=" + qkvType + ", +" + (totalVramAdded / (1024 * 1024)) + " MB VRAM)");
            }
        }
        this.runtimeFusedQkv = fuseQkvOk;
        this.runtimeMergedQkvTensors = localMergedTensors;
        this.runtimeMergedQkvPtrs = localMergedPtrs;

        hasMergedQKV = staticMergedQkv || fuseQkvOk;

        // Allocate merged QKV buffer if needed
        if (hasMergedQKV) {
            this.qkvDim = qDim + kvDim + kvDim;
            gpuQKV = bufferManager.createBuffer((long) qkvDim * fb);
        } else {
            this.qkvDim = 0;
            gpuQKV = 0;
        }

        // Detect and allocate packed FFN (Phi-3/4): wGate is null, wUp outputs 2*ffnDim
        hasPackedFFN = firstLayer.wGate() == null;
        if (hasPackedFFN) {
            gpuHbPacked = bufferManager.createBuffer(2L * ffnDim * fb);
            System.err.println("CUDA: packed FFN detected (Phi-3/4), allocated 2*ffnDim buffer");
        } else {
            gpuHbPacked = 0;
        }

        // Compile auxiliary CUDA kernels
        rmsnormFusedFunc = cudaContext.compileKernel("kernels/cuda/rmsnorm.cu", "rmsnorm_fused");
        siluMulFunc = cudaContext.compileKernel("kernels/cuda/silu_mul.cu", "silu_mul");
        ropeFunc = cudaContext.compileKernel("kernels/cuda/rope.cu", "rope_apply");
        kvCacheUpdateFunc = cudaContext.compileKernel("kernels/cuda/attention.cu", "kv_cache_update");
        attentionFullFunc = cudaContext.compileKernel("kernels/cuda/attention.cu", "attention_full");
        accumulateFunc = cudaContext.compileKernel("kernels/cuda/accumulate.cu", "accumulate");
        argmaxPartialFunc = cudaContext.compileKernel("kernels/cuda/argmax.cu", "argmax_partial");
        argmaxFinalFunc = cudaContext.compileKernel("kernels/cuda/argmax.cu", "argmax_final");

        // Granite scaling
        graniteResidualScale = config.residualScale();
        graniteAttentionScale = config.attentionScale();
        graniteLogitScale = config.logitScale() > 0 && config.architecture() == it.denzosoft.llmplayer.model.ModelArchitecture.GRANITE
            ? (1.0f / config.logitScale()) : 0;
        if (graniteResidualScale > 0 || graniteLogitScale > 0) {
            MemorySegment sf = cudaContext.compileKernel("kernels/cuda/scale_inplace.cu", "scale_inplace");
            scaleFunc = sf;
            scalePB = new ParamBuffer(arena, 3);
            scaleDimGridDim = (int) ((dim + blockSize - 1) / blockSize);
            System.err.println("CUDA: Granite scaling enabled (resScale=" + graniteResidualScale
                + ", attnScale=" + graniteAttentionScale + ", logitScale=" + graniteLogitScale + ")");
        } else {
            scaleFunc = null;
            scalePB = null;
            scaleDimGridDim = 0;
        }

        // Compile split_qkv kernel (for merged QKV — Phi3/4)
        if (hasMergedQKV) {
            splitQkvFunc = cudaContext.compileKernel("kernels/cuda/split_qkv.cu", "split_qkv");
        } else {
            splitQkvFunc = null;
        }

        // Compile split_gate_up kernel (for packed FFN — Phi-3/4)
        if (hasPackedFFN) {
            splitGateUpFunc = cudaContext.compileKernel("kernels/cuda/split_gate_up.cu", "split_gate_up");
            splitGateUpPB = new ParamBuffer(arena, 4);
            splitGateUpPB.setLong(0, gpuHbPacked);
            splitGateUpPB.setLong(1, gpuHb);
            splitGateUpPB.setLong(2, gpuHb2);
            splitGateUpPB.setInt(3, ffnDim);
            splitGateUpGridDim = (int) ((ffnDim + blockSize - 1) / blockSize);
        } else {
            splitGateUpFunc = null;
            splitGateUpPB = null;
            splitGateUpGridDim = 0;
        }

        // Compile per-head norm kernel (for QK-norm)
        if (hasQKNorm) {
            perHeadNormFunc = cudaContext.compileKernel("kernels/cuda/rmsnorm_per_head.cu", "rmsnorm_per_head");
        } else {
            perHeadNormFunc = null;
        }

        // Try to compile fused gate+up kernel (only works if gate/up are Q4_K)
        MemorySegment fusedFunc = null;
        try {
            fusedFunc = cudaContext.compileKernel("kernels/cuda/matmul_q4_k_fused_gate_up.cu", "matmul_q4_k_fused_gate_up");
        } catch (Exception ignored) {}
        fusedGateUpFunc = fusedFunc;

        // Upload per-layer norm weights to GPU (only GPU layers)
        gpuAttnNormWeights = new long[gpuLayerCount];
        gpuFfnNormWeights = new long[gpuLayerCount];
        for (int i = 0; i < gpuLayerCount; i++) {
            TransformerLayerWeights layer = weights.layers()[i];
            if (layer.attnNorm() != null) {
                gpuAttnNormWeights[i] = uploadNormWeights(layer.attnNorm(), dim);
            }
            if (layer.ffnNorm() != null) {
                gpuFfnNormWeights[i] = uploadNormWeights(layer.ffnNorm(), dim);
            }
        }

        // Upload per-layer QKV bias tensors to GPU (if present)
        if (hasBias) {
            gpuQBias = new long[gpuLayerCount];
            gpuKBias = new long[gpuLayerCount];
            gpuVBias = new long[gpuLayerCount];
            for (int i = 0; i < gpuLayerCount; i++) {
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

        // Upload per-layer QK-norm weights to GPU (if present, Qwen3/Gemma3)
        if (hasQKNorm) {
            gpuQNormWeights = new long[gpuLayerCount];
            gpuKNormWeights = new long[gpuLayerCount];
            for (int i = 0; i < gpuLayerCount; i++) {
                TransformerLayerWeights layer = weights.layers()[i];
                gpuQNormWeights[i] = uploadNormWeights(layer.qNorm(), headSize);
                gpuKNormWeights[i] = uploadNormWeights(layer.kNorm(), headSize);
            }
        } else {
            gpuQNormWeights = null;
            gpuKNormWeights = null;
        }

        // Upload per-layer post-attention/FFN norm weights to GPU (if present, Gemma2/3)
        if (hasPostNorm) {
            gpuPostAttnNormWeights = new long[gpuLayerCount];
            gpuPostFfnNormWeights = new long[gpuLayerCount];
            for (int i = 0; i < gpuLayerCount; i++) {
                TransformerLayerWeights layer = weights.layers()[i];
                gpuPostAttnNormWeights[i] = uploadNormWeights(layer.postAttnNorm(), dim);
                gpuPostFfnNormWeights[i] = uploadNormWeights(layer.postFfnNorm(), dim);
            }
        } else {
            gpuPostAttnNormWeights = null;
            gpuPostFfnNormWeights = null;
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

        // Allocate GPU-side argmax buffers
        argmaxNumBlocks = Math.min(256, (vocabSize + 255) / 256);
        gpuArgmaxPartialVal = bufferManager.createBuffer((long) argmaxNumBlocks * Float.BYTES);
        gpuArgmaxPartialIdx = bufferManager.createBuffer((long) argmaxNumBlocks * Integer.BYTES);
        gpuArgmaxResult = bufferManager.createBuffer(Integer.BYTES);
        hostArgmaxResult = arena.allocate(ValueLayout.JAVA_INT, 1);

        // === Pre-allocate reusable param buffers ===
        matmulPB = new ParamBuffer(arena, 6);
        normPB = new ParamBuffer(arena, 5);
        ropePB = new ParamBuffer(arena, 8);
        kvPB = new ParamBuffer(arena, 6);
        attnPB = new ParamBuffer(arena, 9);
        siluPB = new ParamBuffer(arena, 3);
        biasPB = new ParamBuffer(arena, 3);
        fusedGateUpPB = new ParamBuffer(arena, 8);
        argmaxPartialPB = new ParamBuffer(arena, 4);
        argmaxFinalPB = new ParamBuffer(arena, 4);

        // Per-head norm param buffer (4 args: vec, weights, headSize, eps)
        if (hasQKNorm) {
            perHeadNormPB = new ParamBuffer(arena, 4);
            // perHeadNormPB[0] = vec — set per launch (gpuQ or gpuK)
            // perHeadNormPB[1] = weights — set per launch
            perHeadNormPB.setInt(2, headSize);
            perHeadNormPB.setFloat(3, normEps);
        } else {
            perHeadNormPB = null;
        }

        // Post-norm param buffer (5 args: out, in, weights, size, eps) — reuses rmsnormFusedFunc
        // For post-attn norm: in-place on gpuXb (both out and in = gpuXb)
        // For post-FFN norm: in-place on gpuXb (both out and in = gpuXb)
        if (hasPostNorm) {
            postNormPB = new ParamBuffer(arena, 5);
            postNormPB.setLong(0, gpuXb); // out — always gpuXb
            postNormPB.setLong(1, gpuXb); // in — same as out (in-place)
            // postNormPB[2] = weights — set per launch (postAttnNorm or postFfnNorm)
            postNormPB.setInt(3, dim);
            postNormPB.setFloat(4, normEps);
        } else {
            postNormPB = null;
        }

        // Split QKV param buffer (7 args: qkv, q, k, v, qDim, kvDim)
        if (hasMergedQKV) {
            splitQkvPB = new ParamBuffer(arena, 6);
            splitQkvPB.setLong(0, gpuQKV);
            splitQkvPB.setLong(1, gpuQ);
            splitQkvPB.setLong(2, gpuK);
            splitQkvPB.setLong(3, gpuV);
            splitQkvPB.setInt(4, qDim);
            splitQkvPB.setInt(5, kvDim);
            splitQkvGridDim = (int) ((qkvDim + blockSize - 1) / blockSize);
        } else {
            splitQkvPB = null;
            splitQkvGridDim = 0;
        }

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

        // Set fixed argmax partial params (partialVal, partialIdx, size — data set per launch)
        argmaxPartialPB.setLong(1, gpuArgmaxPartialVal);
        argmaxPartialPB.setLong(2, gpuArgmaxPartialIdx);
        argmaxPartialPB.setInt(3, vocabSize);

        // Set fixed argmax final params
        argmaxFinalPB.setLong(0, gpuArgmaxPartialVal);
        argmaxFinalPB.setLong(1, gpuArgmaxPartialIdx);
        argmaxFinalPB.setLong(2, gpuArgmaxResult);
        argmaxFinalPB.setInt(3, argmaxNumBlocks);

        // Set fixed fused gate+up params (input, gateOutput, upOutput, gateRows, cols, addToOutput)
        fusedGateUpPB.setLong(2, gpuXb);       // input
        fusedGateUpPB.setLong(3, gpuHb);       // gateOutput
        fusedGateUpPB.setLong(4, gpuHb2);      // upOutput
        fusedGateUpPB.setInt(5, ffnDim);       // gateRows
        fusedGateUpPB.setInt(6, dim);          // cols
        fusedGateUpPB.setInt(7, 0);            // addToOutput (write mode)

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
        // Per-head norm: one block per head, blockDim covers headSize
        this.perHeadNormBlockDim = hasQKNorm ? (int) Math.min(Math.max(32, ((headSize + 31) / 32) * 32), blockSize) : 0;
        this.perHeadNormSharedMem = hasQKNorm ? ((perHeadNormBlockDim / 32) + 1) * Float.BYTES : 0;

        // === dp4a path setup (llama.cpp-style int8 dot product) ===
        // Opt-in via -Dcuda.dp4a=true (default false to preserve current behavior).
        // Pre-quantize FP32 inputs to Q8_1 once per buffer per layer, then matmuls
        // whose tensor type is Q4_K/Q5_K/Q6_K read int8 from the Q8_1 buffer and use
        // __dp4a (4× int8 muladd in 1 instruction). Typically 1.5-2× faster than FP32 input.
        // dp4a defaults to ON — empirically +33% on Llama-1B (53→70 tok/s) on RTX 4050.
        // Disable with -Dcuda.dp4a=false. Q6_K dp4a is opt-in via cuda.dp4a.q6=true (known buggy).
        boolean dp4aReq = !"false".equals(System.getProperty("cuda.dp4a", "true"));
        boolean dp4aMwReq = "true".equals(System.getProperty("cuda.dp4a.mw", "false"));
        MemorySegment qFunc = null, dQ4kFunc = null, dQ4kMwFunc = null, dQ5kFunc = null, dQ6kFunc = null;
        MemorySegment dQ3kFunc = null;
        MemorySegment dQ50Func = null, dQ80Func = null, dIq4nlFunc = null, dIq4xsFunc = null, dIq4xsMwFunc = null;
        MemorySegment dQ50SmemFunc = null, dIq4nlSmemFunc = null;
        boolean dp4aAvail = false;
        boolean dp4aMwAvail = false;
        if (dp4aReq) {
            try {
                qFunc = cudaContext.compileKernel("kernels/cuda/quantize_q8.cu", "quantize_q8");
                dQ4kFunc = cudaContext.compileKernel("kernels/cuda/matmul_q4_k_dp4a.cu", "matmul_q4_k_dp4a");
                dp4aAvail = true;
                try { dQ5kFunc = cudaContext.compileKernel("kernels/cuda/matmul_q5_k_dp4a.cu", "matmul_q5_k_dp4a"); }
                catch (Exception ignored) {}
                try { dQ6kFunc = cudaContext.compileKernel("kernels/cuda/matmul_q6_k_dp4a.cu", "matmul_q6_k_dp4a"); }
                catch (Exception ignored) {}
                try { dQ3kFunc = cudaContext.compileKernel("kernels/cuda/matmul_q3_k_dp4a.cu", "matmul_q3_k_dp4a"); }
                catch (Exception e) { System.err.println("CUDA: Q3_K dp4a unavailable: " + e.getMessage()); }
                // Extended dp4a kernels (best-effort)
                try { dQ50Func   = cudaContext.compileKernel("kernels/cuda/matmul_q5_0_dp4a.cu",   "matmul_q5_0_dp4a"); }
                catch (Exception e) { System.err.println("CUDA: Q5_0 dp4a unavailable: " + e.getMessage()); }
                // Shared-memory Q5_0 variant: caches Q8_1 input per block (8 rows share it).
                // Main target: Gemma-3-1B Q5_0-heavy gate/up (profile showed 12.7 ms/tok).
                // Default ON; disable with -Dcuda.q5_0.smem=false.
                if (!"false".equals(System.getProperty("cuda.q5_0.smem", "true"))) {
                    try { dQ50SmemFunc = cudaContext.compileKernel("kernels/cuda/matmul_q5_0_dp4a_smem.cu", "matmul_q5_0_dp4a_smem"); }
                    catch (Exception e) { System.err.println("CUDA: Q5_0 dp4a smem unavailable: " + e.getMessage()); }
                }
                try { dQ80Func   = cudaContext.compileKernel("kernels/cuda/matmul_q8_0_dp4a.cu",   "matmul_q8_0_dp4a"); }
                catch (Exception e) { System.err.println("CUDA: Q8_0 dp4a unavailable: " + e.getMessage()); }
                try { dIq4nlFunc = cudaContext.compileKernel("kernels/cuda/matmul_iq4_nl_dp4a.cu", "matmul_iq4_nl_dp4a"); }
                catch (Exception e) { /* not yet written */ }
                // Shared-memory IQ4_NL variant: caches Q8_1 input per block. **Measured neutral**
                // on Phi-3-mini (12.0 smem vs 12.07 baseline tok/s on RTX 4050 in graph mode —
                // graph mode already amortizes the HBM reads). Kept opt-in for future hardware
                // or for non-graph fallback paths. Enable with -Dcuda.iq4nl.smem=true.
                if ("true".equals(System.getProperty("cuda.iq4nl.smem", "false"))) {
                    try { dIq4nlSmemFunc = cudaContext.compileKernel("kernels/cuda/matmul_iq4_nl_dp4a_smem.cu", "matmul_iq4_nl_dp4a_smem"); }
                    catch (Exception e) { System.err.println("CUDA: IQ4_NL dp4a smem unavailable: " + e.getMessage()); }
                }
                try { dIq4xsFunc = cudaContext.compileKernel("kernels/cuda/matmul_iq4_xs_dp4a.cu", "matmul_iq4_xs_dp4a"); }
                catch (Exception e) { /* not yet written */ }
                try { dIq4xsMwFunc = cudaContext.compileKernel("kernels/cuda/matmul_iq4_xs_dp4a_mw.cu", "matmul_iq4_xs_dp4a_mw"); }
                catch (Exception e) { /* optional multi-warp variant */ }
                if (dp4aMwReq) {
                    try {
                        dQ4kMwFunc = cudaContext.compileKernel("kernels/cuda/matmul_q4_k_dp4a_mw.cu", "matmul_q4_k_dp4a_mw");
                        dp4aMwAvail = true;
                    } catch (Exception e) {
                        System.err.println("CUDA: Q4_K multi-warp dp4a unavailable: " + e.getMessage());
                    }
                }
            } catch (Exception e) {
                System.err.println("CUDA: dp4a kernels unavailable — " + e.getMessage());
            }
        }
        useDp4a = dp4aAvail;
        useDp4aMw = dp4aMwAvail;
        dp4aQ4kMwFunc = dQ4kMwFunc;

        // T2.1: multi-row Q4_K dp4a kernel (opt-in)
        boolean mr4Req = "true".equals(System.getProperty("cuda.q4k.mr4", "false"));
        MemorySegment mr4Func = null;
        if (mr4Req && useDp4a) {
            try {
                mr4Func = cudaContext.compileKernel("kernels/cuda/matmul_q4_k_dp4a_mr4.cu", "matmul_q4_k_dp4a_mr4");
                System.err.println("CUDA: Q4_K multi-row (4 rows/warp) enabled");
            } catch (Exception e) {
                System.err.println("CUDA: Q4_K mr4 unavailable: " + e.getMessage());
            }
        }
        useDp4aQ4kMr4 = (mr4Func != null);
        dp4aQ4kMr4Func = mr4Func;

        // Port of llama.cpp's mul_mat_vec_q for Q4_K (opt-in via -Dcuda.dp4a.mmvq=true)
        boolean mmvqReq = "true".equals(System.getProperty("cuda.dp4a.mmvq", "false"));
        MemorySegment dQ4kMmvqFunc = null;
        if (dp4aAvail && mmvqReq) {
            try {
                dQ4kMmvqFunc = cudaContext.compileKernel("kernels/cuda/matmul_q4_k_mmvq.cu", "matmul_q4_k_mmvq");
                System.err.println("CUDA: Q4_K mmvq kernel enabled (llama.cpp port — 4 warps/row, vdr=2)");
            } catch (Exception e) {
                System.err.println("CUDA: Q4_K mmvq kernel unavailable: " + e.getMessage());
            }
        }
        mmvqQ4kFunc = dQ4kMmvqFunc;
        useMmvqQ4k = dQ4kMmvqFunc != null;

        // Fused dp4a gate+up — opt-in via -Dcuda.dp4a.fused_gate_up=true.
        // Measured marginal/regression on Llama-1B (70.9 vs 73.5 separate-dp4a) because
        // input was already L1-cached and the fused kernel adds register pressure.
        // Kept available for future experimentation on architectures with different ratios.
        MemorySegment dFusedFunc = null;
        boolean dFusedReq = "true".equals(System.getProperty("cuda.dp4a.fused_gate_up", "false"));
        if (dp4aAvail && dFusedReq) {
            try {
                dFusedFunc = cudaContext.compileKernel(
                    "kernels/cuda/matmul_q4_k_dp4a_fused_gate_up.cu",
                    "matmul_q4_k_dp4a_fused_gate_up");
                System.err.println("CUDA: dp4a fused gate+up kernel enabled (opt-in)");
            } catch (Exception e) {
                System.err.println("CUDA: dp4a fused gate+up kernel unavailable: " + e.getMessage());
            }
        }
        dp4aFusedGateUpFunc = dFusedFunc;
        useDp4aFusedGateUp = dFusedFunc != null;
        // ParamBuffer constructed here, but slots that need gpuXbQ8 are set AFTER the buffer alloc below.
        dp4aFusedGateUpPB = useDp4aFusedGateUp ? new ParamBuffer(arena, 7) : null;
        quantizeFunc = qFunc;
        dp4aQ4kFunc = dQ4kFunc;
        dp4aQ3kFunc = dQ3kFunc;
        dp4aQ5kFunc = dQ5kFunc;
        dp4aQ6kFunc = dQ6kFunc;
        dp4aQ50Func = dQ50Func;
        dp4aQ50SmemFunc = dQ50SmemFunc;
        dp4aQ80Func = dQ80Func;
        dp4aIq4nlFunc = dIq4nlFunc;
        dp4aIq4nlSmemFunc = dIq4nlSmemFunc;
        dp4aIq4xsFunc = dIq4xsFunc;
        dp4aIq4xsMwFunc = dIq4xsMwFunc;
        if (useDp4a) {
            // Q8_1 layout: per 32-element block, 4-byte scale + 4-byte sum + 32 bytes int8 = 40 bytes.
            int xbQ8Bytes  = ((dim    + 31) / 32) * 40;
            int xb2Q8Bytes = ((qDim   + 31) / 32) * 40;
            int hbQ8Bytes  = ((ffnDim + 31) / 32) * 40;
            gpuXbQ8  = bufferManager.createBuffer(xbQ8Bytes);
            gpuXb2Q8 = bufferManager.createBuffer(xb2Q8Bytes);
            gpuHbQ8  = bufferManager.createBuffer(hbQ8Bytes);

            // 8 warps per block (256 threads); each warp handles one Q8_1 block (32 elems).
            int xbBlocks  = (dim    + 31) / 32;
            int xb2Blocks = (qDim   + 31) / 32;
            int hbBlocks  = (ffnDim + 31) / 32;
            quantizeXbGridDim  = (xbBlocks  + 7) / 8;
            quantizeXb2GridDim = (xb2Blocks + 7) / 8;
            quantizeHbGridDim  = (hbBlocks  + 7) / 8;

            quantizeXbPB  = new ParamBuffer(arena, 3);
            quantizeXbPB.setLong(0, gpuXb);
            quantizeXbPB.setLong(1, gpuXbQ8);
            quantizeXbPB.setInt(2, dim);

            quantizeXb2PB = new ParamBuffer(arena, 3);
            quantizeXb2PB.setLong(0, gpuXb2);
            quantizeXb2PB.setLong(1, gpuXb2Q8);
            quantizeXb2PB.setInt(2, qDim);

            // Wire fused dp4a gate+up params now that gpuXbQ8 exists
            if (useDp4aFusedGateUp) {
                dp4aFusedGateUpPB.setLong(2, gpuXbQ8);   // input (Q8_1 quantized)
                dp4aFusedGateUpPB.setLong(3, gpuHb);     // gateOutput
                dp4aFusedGateUpPB.setLong(4, gpuHb2);    // upOutput
                dp4aFusedGateUpPB.setInt(5, ffnDim);     // rows
                dp4aFusedGateUpPB.setInt(6, dim);        // cols
            }

            quantizeHbPB  = new ParamBuffer(arena, 3);
            quantizeHbPB.setLong(0, gpuHb);
            quantizeHbPB.setLong(1, gpuHbQ8);
            quantizeHbPB.setInt(2, ffnDim);

            dp4aPB = new ParamBuffer(arena, 6);

            String dp4aTypes = "Q4_K";
            if (dp4aQ3kFunc != null) dp4aTypes += "+Q3_K";
            if (dp4aQ5kFunc != null) dp4aTypes += "+Q5_K";
            if (dp4aQ6kFunc != null) dp4aTypes += "+Q6_K";
            System.err.println("CUDA: dp4a enabled (" + dp4aTypes + " × Q8_1, scratch="
                + (xbQ8Bytes + xb2Q8Bytes + hbQ8Bytes) + " bytes)");
        } else {
            gpuXbQ8 = 0; gpuXb2Q8 = 0; gpuHbQ8 = 0;
            quantizeXbGridDim = 0; quantizeXb2GridDim = 0; quantizeHbGridDim = 0;
            quantizeXbPB = null; quantizeXb2PB = null; quantizeHbPB = null; dp4aPB = null;
        }

        // === Fused RMSNorm + Q8_1 quantize (saves 1 launch + 1 HBM round-trip per layer) ===
        // Default ON when dp4a is on. Disable with -Dcuda.fused_norm_quantize=false.
        MemorySegment rnqFunc = null;
        if (useDp4a && !"false".equals(System.getProperty("cuda.fused_norm_quantize", "true"))) {
            try {
                rnqFunc = cudaContext.compileKernel(
                    "kernels/cuda/rmsnorm_quantize.cu", "rmsnorm_quantize_fused");
                System.err.println("CUDA: fused RMSNorm+Quantize enabled");
            } catch (Exception e) {
                System.err.println("CUDA: fused RMSNorm+Quantize unavailable: " + e.getMessage());
            }
        }
        rmsnormQuantizeFunc = rnqFunc;
        useFusedNormQuantize = (rnqFunc != null);
        if (useFusedNormQuantize) {
            // 6 args: normOut, qOut, in, w, size, eps. Slots 0,1,2,4,5 fixed; slot 3 = w (per-layer)
            rmsnormQuantizePB = new ParamBuffer(arena, 6);
            rmsnormQuantizePB.setLong(0, gpuXb);          // normOut
            rmsnormQuantizePB.setLong(1, gpuXbQ8);        // qOut
            rmsnormQuantizePB.setLong(2, gpuX);           // in
            rmsnormQuantizePB.setInt(4, dim);             // size
            rmsnormQuantizePB.setFloat(5, normEps);       // eps
        } else {
            rmsnormQuantizePB = null;
        }

        // === Pre-compute per-layer matmul launch descriptors (only GPU layers) ===
        layerMatmuls = new MatmulLaunch[gpuLayerCount][7];
        boolean canFuseGateUp = fusedGateUpFunc != null;
        long[] fGateWeights = canFuseGateUp ? new long[gpuLayerCount] : null;
        long[] fUpWeights = canFuseGateUp ? new long[gpuLayerCount] : null;
        for (int i = 0; i < gpuLayerCount; i++) {
            TransformerLayerWeights layer = weights.layers()[i];
            if (hasMergedQKV) {
                // Merged QKV: single matmul wqkv → gpuQKV, then split kernel
                CudaFloatTensor mergedTensor = runtimeFusedQkv
                    ? runtimeMergedQkvTensors[i]
                    : (CudaFloatTensor) layer.wqkv();
                layerMatmuls[i][0] = new MatmulLaunch(mergedTensor, gpuXb, gpuQKV, qkvDim, dim, 0, gpuXbQ8);
                layerMatmuls[i][1] = null; // unused — split kernel handles K
                layerMatmuls[i][2] = null; // unused — split kernel handles V
            } else {
                layerMatmuls[i][0] = new MatmulLaunch((CudaFloatTensor) layer.wq(), gpuXb, gpuQ, qDim, dim, 0, gpuXbQ8);
                layerMatmuls[i][1] = new MatmulLaunch((CudaFloatTensor) layer.wk(), gpuXb, gpuK, kvDim, dim, 0, gpuXbQ8);
                layerMatmuls[i][2] = new MatmulLaunch((CudaFloatTensor) layer.wv(), gpuXb, gpuV, kvDim, dim, 0, gpuXbQ8);
            }
            // Post-norm: Wo/Down write to gpuXb, post-norm then accumulate
            // Granite residual: Wo/Down write to gpuXb, scale, then accumulate (saxpy)
            // Pre-norm: Wo/Down accumulate directly to gpuX (addToOutput=1)
            boolean woWriteMode = hasPostNorm || graniteResidualScale > 0;
            layerMatmuls[i][3] = new MatmulLaunch((CudaFloatTensor) layer.wo(), gpuXb2,
                    woWriteMode ? gpuXb : gpuX, dim, qDim, woWriteMode ? 0 : 1, gpuXb2Q8);
            if (hasPackedFFN) {
                // Packed FFN: no gate weight, wUp outputs 2*ffnDim → gpuHbPacked, then split kernel
                layerMatmuls[i][4] = null;
                layerMatmuls[i][5] = new MatmulLaunch((CudaFloatTensor) layer.wUp(), gpuXb, gpuHbPacked, ffnDim * 2, dim, 0, gpuXbQ8);
            } else {
                layerMatmuls[i][4] = new MatmulLaunch((CudaFloatTensor) layer.wGate(), gpuXb, gpuHb, ffnDim, dim, 0, gpuXbQ8);
                layerMatmuls[i][5] = new MatmulLaunch((CudaFloatTensor) layer.wUp(), gpuXb, gpuHb2, ffnDim, dim, 0, gpuXbQ8);
            }
            boolean downWriteMode = hasPostNorm || graniteResidualScale > 0;
            layerMatmuls[i][6] = new MatmulLaunch((CudaFloatTensor) layer.wDown(), gpuHb,
                    downWriteMode ? gpuXb : gpuX, dim, ffnDim, downWriteMode ? 0 : 1, gpuHbQ8);

            // Check if gate and up are both Q4_K for fused kernel (not applicable for packed FFN)
            if (canFuseGateUp && !hasPackedFFN && layer.wGate() instanceof CudaFloatTensor && layer.wUp() instanceof CudaFloatTensor) {
                CudaFloatTensor gate = (CudaFloatTensor) layer.wGate();
                CudaFloatTensor up = (CudaFloatTensor) layer.wUp();
                if (gate.type() == it.denzosoft.llmplayer.tensor.GGMLType.Q4_K
                        && up.type() == it.denzosoft.llmplayer.tensor.GGMLType.Q4_K) {
                    fGateWeights[i] = gate.getGpuWeights();
                    fUpWeights[i] = up.getGpuWeights();
                } else {
                    canFuseGateUp = false;
                }
            } else if (canFuseGateUp) {
                canFuseGateUp = false;
            }
        }
        useFusedGateUp = canFuseGateUp;
        fusedGateWeights = fGateWeights;
        fusedUpWeights = fUpWeights;
        if (canFuseGateUp) {
            // totalRows = ffnDim * 2, 1 warp per row
            int totalRows = ffnDim * 2;
            int fusedBlockDimVal = (int) Math.min(256, maxWg);
            int warpsPerBlock = fusedBlockDimVal / 32;
            fusedGateUpGridDim = (totalRows + warpsPerBlock - 1) / warpsPerBlock;
            fusedGateUpBlockDim = fusedBlockDimVal;
            System.err.println("CUDA: fused gate+up Q4_K kernel enabled (" + gpuLayerCount + " layers)");
        } else {
            fusedGateUpGridDim = 0;
            fusedGateUpBlockDim = 0;
        }

        // === cuBLAS initialization (opt-in via -Dcuda.cublas=true) ===
        boolean cublasEnabled = Boolean.getBoolean("cuda.cublas")
                && it.denzosoft.llmplayer.gpu.CublasBindings.isAvailable();
        if (cublasEnabled) {
            try {
                it.denzosoft.llmplayer.gpu.CublasMatmul cm = new it.denzosoft.llmplayer.gpu.CublasMatmul(cudaContext);
                boolean useFP16 = !Boolean.getBoolean("cuda.cublas.fp32");
                MemorySegment dequantFunc;
                int bytesPerElement;
                if (useFP16) {
                    dequantFunc = cudaContext.compileKernel("kernels/cuda/dequant_q4_k_f16.cu", "dequant_q4_k_f16");
                    bytesPerElement = 2;
                    // Compile FP32→FP16 conversion kernel for input vector
                    MemorySegment convertFunc = cudaContext.compileKernel("kernels/cuda/convert_f32_to_f16.cu", "convert_f32_to_f16");
                    cm.ensureInputF16(dim, bufferManager);
                    // Store the convert function for use in gemmExF16
                    this.cublasConvertF16Func = convertFunc;
                } else {
                    dequantFunc = cudaContext.compileKernel("kernels/cuda/dequant_q4_k_f32.cu", "dequant_q4_k_f32");
                    bytesPerElement = 4;
                    this.cublasConvertF16Func = null;
                }
                this.cublasUseFP16 = useFP16;

                long[][] f32w = new long[gpuLayerCount][7];
                long totalBytes = 0;
                for (int i = 0; i < gpuLayerCount; i++) {
                    for (int j = 0; j < 7; j++) {
                        MatmulLaunch ml = layerMatmuls[i][j];
                        if (ml != null) {
                            TransformerLayerWeights layer = weights.layers()[i];
                            FloatTensor t = getLayerTensor(layer, j);
                            if (t instanceof CudaFloatTensor && ((CudaFloatTensor) t).type() == it.denzosoft.llmplayer.tensor.GGMLType.Q4_K) {
                                long ptr;
                                if (useFP16) {
                                    ptr = cm.dequantizeToF16((CudaFloatTensor) t, ml.rows, ml.cols, bufferManager, dequantFunc);
                                } else {
                                    ptr = cm.dequantizeToF32((CudaFloatTensor) t, ml.rows, ml.cols, bufferManager, dequantFunc);
                                }
                                f32w[i][j] = ptr;
                                totalBytes += (long) ml.rows * ml.cols * bytesPerElement;
                            }
                        }
                    }
                }

                // Dequantize output projection
                long f32Out = 0;
                if (outputMatmul != null && outputTensor instanceof CudaFloatTensor
                        && ((CudaFloatTensor) outputTensor).type() == it.denzosoft.llmplayer.tensor.GGMLType.Q4_K) {
                    if (useFP16) {
                        f32Out = cm.dequantizeToF16((CudaFloatTensor) outputTensor, vocabSize, dim, bufferManager, dequantFunc);
                    } else {
                        f32Out = cm.dequantizeToF32((CudaFloatTensor) outputTensor, vocabSize, dim, bufferManager, dequantFunc);
                    }
                    totalBytes += (long) vocabSize * dim * bytesPerElement;
                }

                cublasMatmul = cm;
                cublasF32Weights = f32w;
                cublasF32Output = f32Out;
                useCublas = true;
                System.err.println("cuBLAS: dequantized " + (totalBytes / 1024 / 1024)
                    + " MB of Q4_K weights to " + (useFP16 ? "FP16" : "FP32"));
            } catch (Exception e) {
                System.err.println("cuBLAS: initialization failed — " + e.getMessage());
                cublasMatmul = null;
                cublasF32Weights = null;
                cublasF32Output = 0;
                useCublas = false;
            }
        } else {
            cublasMatmul = null;
            cublasF32Weights = null;
            cublasF32Output = 0;
            useCublas = false;
        }

        // CUDA graph: pre-compute fixed attention shared mem (max seqLen)
        // 48 KB per SM limit → max seqLen ~12256 for graph mode
        this.graphAttnSharedMem = (maxSeqLen + 32) * Float.BYTES;
        this.graphAvailable = !Boolean.getBoolean("cuda.nograph")
                && !useCublas  // cuBLAS manages its own state, incompatible with graph capture
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
     * Returns true if at least the first layer has CudaFloatTensor weights (partial offload OK).
     */
    public static boolean isSupported(ModelConfig config, ModelWeights weights) {
        if (config.expertCount() > 0) return false;
        // Command-R uses centered LayerNorm, not RMSNorm — CUDA path implements RMSNorm only.
        // Force CPU until a layernorm.cu kernel is added.
        if (config.useLayerNorm()) return false;

        TransformerLayerWeights firstLayer = weights.layers()[0];
        // Require at least pre-norm or post-norm for attention and FFN
        boolean hasPreNorm = firstLayer.attnNorm() != null && firstLayer.ffnNorm() != null;
        boolean hasPostNorm = firstLayer.postAttnNorm() != null && firstLayer.postFfnNorm() != null;
        if (!hasPreNorm && !hasPostNorm) return false;

        // First layer weight tensors must be CudaFloatTensor — merged QKV or separate Q/K/V
        if (firstLayer.wqkv() != null) {
            if (!(firstLayer.wqkv() instanceof CudaFloatTensor)) return false;
        } else {
            if (!(firstLayer.wq() instanceof CudaFloatTensor)) return false;
            if (!(firstLayer.wk() instanceof CudaFloatTensor)) return false;
            if (!(firstLayer.wv() instanceof CudaFloatTensor)) return false;
        }
        if (!(firstLayer.wo() instanceof CudaFloatTensor)) return false;
        // Packed FFN (Phi-3/4): wGate is null, wUp produces 2*ffnDim output
        if (firstLayer.wGate() != null) {
            if (!(firstLayer.wGate() instanceof CudaFloatTensor)) return false;
        }
        if (!(firstLayer.wUp() instanceof CudaFloatTensor)) return false;
        if (!(firstLayer.wDown() instanceof CudaFloatTensor)) return false;

        return true;
    }

    /**
     * Weaker variant of {@link #isSupported}: checks only that the attention half (steps 1-6c)
     * can run on GPU. FFN tensors may be absent or non-Cuda (e.g. MoE architectures where FFN is
     * a router + sparse experts handled by the calling engine via {@link #forwardAttentionOnly}).
     *
     * Used by MoE engines (Qwen3MoE, DeepSeek2) that own the FFN path themselves and only want
     * GPU acceleration for the attention block.
     */
    public static boolean isSupportedForAttention(ModelConfig config, ModelWeights weights) {
        // Command-R uses centered LayerNorm, not RMSNorm — CUDA path implements RMSNorm only.
        if (config.useLayerNorm()) return false;

        TransformerLayerWeights firstLayer = weights.layers()[0];
        boolean hasPreNorm = firstLayer.attnNorm() != null && firstLayer.ffnNorm() != null;
        boolean hasPostNorm = firstLayer.postAttnNorm() != null && firstLayer.postFfnNorm() != null;
        if (!hasPreNorm && !hasPostNorm) return false;

        // Attention weights must be GPU-resident
        if (firstLayer.wqkv() != null) {
            if (!(firstLayer.wqkv() instanceof CudaFloatTensor)) return false;
        } else {
            if (!(firstLayer.wq() instanceof CudaFloatTensor)) return false;
            if (!(firstLayer.wk() instanceof CudaFloatTensor)) return false;
            if (!(firstLayer.wv() instanceof CudaFloatTensor)) return false;
        }
        if (!(firstLayer.wo() instanceof CudaFloatTensor)) return false;

        // FFN tensors NOT required — caller handles FFN path.
        return true;
    }

    /**
     * Count consecutive GPU-offloaded layers starting from layer 0.
     * Used by InferenceEngine to know which layers to forward on GPU vs CPU.
     */
    public static int countGpuLayers(ModelWeights weights) {
        int count = 0;
        for (TransformerLayerWeights layer : weights.layers()) {
            // Check the main weight tensor (wo is always present)
            if (layer.wo() instanceof CudaFloatTensor) {
                count++;
            } else {
                break; // first-N-layers strategy: contiguous from layer 0
            }
        }
        return count;
    }

    /** Number of layers handled by this CudaForwardPass instance. */
    public int getGpuLayerCount() {
        return gpuLayerCount;
    }

    // Number of transformer layers on GPU (may be less than blockCount for partial offload)
    private final int gpuLayerCount;

    public void uploadX(float[] x) {
        long t0 = 0;
        if (PROFILING) t0 = System.nanoTime();
        MemorySegment.copy(x, 0, hostX, ValueLayout.JAVA_FLOAT, 0, dim);
        cudaContext.writeBuffer(gpuX, hostX, (long) dim * Float.BYTES);
        if (PROFILING) profUpload += System.nanoTime() - t0;
    }

    /**
     * Combined upload: embedding vector + token params in a single cuMemcpyHtoD.
     * Saves one Panama FFM call per token (~0.1ms).
     */
    public void uploadXAndUpdateParams(float[] x, int position) {
        long t0 = 0;
        if (PROFILING) t0 = System.nanoTime();
        long embBytes = (long) dim * Float.BYTES;
        MemorySegment.copy(x, 0, hostCombined, ValueLayout.JAVA_FLOAT, 0, dim);
        hostCombined.set(ValueLayout.JAVA_INT, embBytes, position);
        hostCombined.set(ValueLayout.JAVA_INT, embBytes + 4, position + 1);
        cudaContext.writeBuffer(gpuCombined, hostCombined, embBytes + 8);
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
        if (useCublas && cublasF32Output != 0) {
            if (cublasUseFP16) {
                cublasMatmul.gemmExF16(cublasF32Output, gpuXb, gpuLogits, vocabSize, dim, false,
                    cublasConvertF16Func, cudaContext);
            } else {
                cublasMatmul.sgemv(cublasF32Output, gpuXb, gpuLogits, vocabSize, dim, false);
            }
        } else {
            launchOutputMatmul();
        }

        // Granite logit scaling: divide logits by logitScale (on GPU)
        if (graniteLogitScale != 0) {
            scalePB.setLong(0, gpuLogits);
            scalePB.setFloat(1, graniteLogitScale);
            scalePB.setInt(2, vocabSize);
            int logitsGridDim = (int) ((vocabSize + blockSize - 1) / blockSize);
            launchKernel(scaleFunc, logitsGridDim, (int) blockSize, 0, scalePB.ptrs);
        }

        // Download logits (cuMemcpyDtoH is synchronous)
        cudaContext.readBuffer(gpuLogits, hostLogits, gpuLogitsBytes);
        MemorySegment.copy(hostLogits, ValueLayout.JAVA_FLOAT, 0, logits, 0, vocabSize);

        if (PROFILING) profOutputMatmul += System.nanoTime() - t0;
        return true;
    }

    /**
     * Compute final RMSNorm + output projection + argmax on GPU.
     * Returns the token ID with the highest logit, downloading only 4 bytes instead of 512KB.
     * Returns -1 if output is not on GPU (caller falls back to CPU path).
     *
     * Use for greedy sampling (temperature=0) without repetition penalty.
     */
    public int forwardFinalArgmax() {
        if (outputMatmul == null) return -1;

        // Final RMSNorm: gpuX → gpuXb
        normPB.setLong(2, gpuOutputNormWeights);
        launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);

        // Output projection
        launchOutputMatmul();

        // GPU-side argmax (two-phase: partial → final)
        argmaxPartialPB.setLong(0, gpuLogits);
        launchKernel(argmaxPartialFunc, argmaxNumBlocks, 256, 0, argmaxPartialPB.ptrs);
        launchKernel(argmaxFinalFunc, 1, 256, 0, argmaxFinalPB.ptrs);

        // Download only 4 bytes (the token index)
        cudaContext.readBuffer(gpuArgmaxResult, hostArgmaxResult, Integer.BYTES);
        return hostArgmaxResult.get(ValueLayout.JAVA_INT, 0);
    }

    /**
     * Execute all layers + output projection + argmax via CUDA graph.
     * Like forwardGraph() but returns the argmax token ID instead of downloading full logits.
     * Returns -1 to fall back to normal path.
     */
    public int forwardGraphArgmax() {
        // GPU-side argmax can't be included in the CUDA graph because argmax is a 2-phase kernel
        // and would add variable shared memory requirements. Instead, run graph + argmax separately.
        if (!graphAvailable || PROFILING) return -1;

        if (graphExec == null) {
            // First call — capture graph (same as forwardGraph)
            boolean capturing = false;
            try {
                cudaContext.beginCapture();
                capturing = true;

                for (int layer = 0; layer < gpuLayerCount; layer++) {
                    forwardLayerKernels(layer);
                }

                normPB.setLong(2, gpuOutputNormWeights);
                launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);
                launchOutputMatmul();

                MemorySegment graph = cudaContext.endCapture();
                capturing = false;
                graphExec = cudaContext.instantiateGraph(graph);
                cudaContext.destroyGraph(graph);

                System.err.println("CUDA graph: captured " + gpuLayerCount + " layers + output projection (argmax)");
            } catch (Exception e) {
                if (capturing) {
                    try { cudaContext.endCapture(); } catch (Exception ignored) {}
                }
                System.err.println("CUDA graph: capture failed — " + e.getMessage());
                graphExec = null;
                return -1;
            }
        }

        // Launch graph (all layers + output projection)
        cudaContext.launchGraph(graphExec);

        // Sync before argmax (graph might still be running)
        cudaContext.finish();

        // GPU-side argmax on the logits (already computed on GPU by graph)
        argmaxPartialPB.setLong(0, gpuLogits);
        launchKernel(argmaxPartialFunc, argmaxNumBlocks, 256, 0, argmaxPartialPB.ptrs);
        launchKernel(argmaxFinalFunc, 1, 256, 0, argmaxFinalPB.ptrs);

        // Download only 4 bytes
        cudaContext.readBuffer(gpuArgmaxResult, hostArgmaxResult, Integer.BYTES);
        return hostArgmaxResult.get(ValueLayout.JAVA_INT, 0);
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
        System.err.printf("CUDA profile (%d tokens, %d GPU layers):%n", profTokens, gpuLayerCount);
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
        forwardAttentionPart(layerIdx, position);
        forwardFFNPart(layerIdx);
    }

    /**
     * Run the attention half of a layer on GPU (steps 1-6c: RMSNorm → QKV → bias → Granite scale →
     * RoPE → KV cache → attention → Wo → Granite residual → post-attn norm+bias). Leaves the
     * post-attention residual in gpuX.
     *
     * Public entry point used by engines that want GPU attention but a custom FFN path (e.g.
     * Qwen3MoE — the engine calls {@link #forwardAttentionOnly} / {@link #downloadX}, runs the
     * MoE FFN on CPU, then calls {@link #uploadX} before the next layer's attention).
     */
    public void forwardAttentionOnly(InferenceState state, TransformerLayerWeights layerWeights,
                                      int layerIdx, int position, Attention attention) {
        if (PROFILING) {
            forwardLayerProfiled(state, layerWeights, layerIdx, position, attention);
            return;
        }
        forwardAttentionPart(layerIdx, position);
    }

    /**
     * Steps 1-6c of the per-layer pipeline. Zero-allocation, GPU-resident.
     * Post-condition: gpuX holds the post-attention residual (pre-norm model) or the
     * post-post-norm residual (Gemma2/3-style post-norm model).
     */
    private void forwardAttentionPart(int layerIdx, int position) {
        MatmulLaunch[] ml = layerMatmuls[layerIdx];

        // 1. RMSNorm (attnNorm): gpuX → gpuXb  (or plain copy for post-norm-only models like OLMo2)
        // dp4a path: fused norm+quantize when both are needed (1 launch instead of 2).
        if (hasPreNorm) {
            normAndQuantizeXb(gpuAttnNormWeights[layerIdx]);
        } else {
            cudaContext.copyBufferDtoD(gpuXb, gpuX, (long) dim * Float.BYTES);
            quantizeXb();
        }

        // 2. Q/K/V projections (write mode)
        if (hasMergedQKV) {
            launchMatmulCublasOrDp4a(ml[0], layerIdx, 0); // wqkv → gpuQKV
            launchKernel(splitQkvFunc, splitQkvGridDim, (int) blockSize, 0, splitQkvPB.ptrs);
        } else {
            launchMatmulCublasOrDp4a(ml[0], layerIdx, 0); // wq
            launchMatmulCublasOrDp4a(ml[1], layerIdx, 1); // wk
            launchMatmulCublasOrDp4a(ml[2], layerIdx, 2); // wv
        }


        // 2b. Add QKV bias if present (Qwen2)
        if (hasBias) {
            launchBias(gpuQ, gpuQBias[layerIdx], qDim, biasQGridDim);
            launchBias(gpuK, gpuKBias[layerIdx], kvDim, biasKVGridDim);
            launchBias(gpuV, gpuVBias[layerIdx], kvDim, biasKVGridDim);
        }

        // 2b'. Per-head QK-norm (Qwen3, Gemma3, SmolLM3). Pre-existing bug fix: this path
        // was previously only in forwardLayerKernels (graph capture). Non-graph per-layer
        // mode (used for partial offload and attention-only MoE dispatch) silently skipped
        // QK-norm and would generate garbage for affected architectures.
        if (hasQKNorm) {
            perHeadNormPB.setLong(0, gpuQ);
            perHeadNormPB.setLong(1, gpuQNormWeights[layerIdx]);
            launchKernel(perHeadNormFunc, headCount, perHeadNormBlockDim, perHeadNormSharedMem, perHeadNormPB.ptrs);
            perHeadNormPB.setLong(0, gpuK);
            perHeadNormPB.setLong(1, gpuKNormWeights[layerIdx]);
            launchKernel(perHeadNormFunc, headCountKV, perHeadNormBlockDim, perHeadNormSharedMem, perHeadNormPB.ptrs);
        }

        // 2c. Granite attention scale correction: scale Q so that attention kernel's 1/sqrt(headSize)
        // produces the correct custom scale. Q *= attentionScale * sqrt(headSize)
        if (graniteAttentionScale > 0) {
            float correction = graniteAttentionScale * (float) Math.sqrt(headSize);
            scalePB.setLong(0, gpuQ);
            scalePB.setFloat(1, correction);
            scalePB.setInt(2, qDim);
            int qGridDim = (int) ((qDim + blockSize - 1) / blockSize);
            launchKernel(scaleFunc, qGridDim, (int) blockSize, 0, scalePB.ptrs);
        }

        // 3. RoPE on Q and K (skip for NoPE layers in SmolLM3/Llama4 iRoPE)
        if (noRopeLayerInterval == 0 || (layerIdx % noRopeLayerInterval) != (noRopeLayerInterval - 1)) {
            ropePB.setLong(0, gpuQ);
            ropePB.setInt(3, headCount);
            launchKernel(ropeFunc, ropeQGridDim, (int) blockSize, 0, ropePB.ptrs);

            // 3b. RoPE on K
            ropePB.setLong(0, gpuK);
            ropePB.setInt(3, headCountKV);
            launchKernel(ropeFunc, ropeKGridDim, (int) blockSize, 0, ropePB.ptrs);
        }

        // 4. KV cache update (position read from gpuTokenParams[0] by kernel)
        kvPB.setLong(0, gpuKeyCache[layerIdx]);
        kvPB.setLong(1, gpuValueCache[layerIdx]);
        launchKernel(kvCacheUpdateFunc, kvUpdateGridDim, (int) blockSize, 0, kvPB.ptrs);

        // 5. Full attention (seqLen read from gpuTokenParams[1] by kernel)
        attnPB.setLong(2, gpuKeyCache[layerIdx]);
        attnPB.setLong(3, gpuValueCache[layerIdx]);
        int attnSharedMem = (position + 1 + 32) * Float.BYTES;
        launchKernel(attentionFullFunc, headCount, (int) attnBlockSize, attnSharedMem, attnPB.ptrs);

        // dp4a: pre-quantize gpuXb2 → gpuXb2Q8 for the Wo projection
        quantizeXb2();
        // 6. Wo matmul
        launchMatmulCublasOrDp4a(ml[3], layerIdx, 3); // wo

        // 6b. Granite residual scaling: gpuX += residualScale * gpuXb (saxpy)
        if (graniteResidualScale > 0) {
            launchSaxpy(gpuX, gpuXb, graniteResidualScale, dim);
        }

        // 6c. Post-attention norm + accumulate (Gemma2/3)
        if (hasPostNorm) {
            postNormPB.setLong(2, gpuPostAttnNormWeights[layerIdx]);
            launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, postNormPB.ptrs);
            launchBias(gpuX, gpuXb, dim, numWorkGroups);
        }
    }

    /**
     * Steps 7-10c of the per-layer pipeline: FFN RMSNorm → Gate/Up → silu_mul → Down → post-norm.
     * Pre-condition: gpuX holds the post-attention residual.
     * Post-condition: gpuX holds the full post-FFN output (next layer's input).
     */
    private void forwardFFNPart(int layerIdx) {
        MatmulLaunch[] ml = layerMatmuls[layerIdx];

        // 7. FFN RMSNorm: gpuX → gpuXb  (or plain copy for post-norm-only)
        if (hasPreNorm) {
            normAndQuantizeXb(gpuFfnNormWeights[layerIdx]);
        } else {
            cudaContext.copyBufferDtoD(gpuXb, gpuX, (long) dim * Float.BYTES);
            quantizeXb();  // bug-fix 2026-04-14: forwardLayer non-graph was missing this
        }

        // 8. Gate + Up projections — packed FFN, dp4a-fused (preferred when both Q4_K), 2× dp4a, FP32 fused, or separate
        if (hasPackedFFN) {
            launchMatmulCublasOrDp4a(ml[5], layerIdx, 5);
            launchKernel(splitGateUpFunc, splitGateUpGridDim, (int) blockSize, 0, splitGateUpPB.ptrs);
        } else if (useDp4aFusedGateUp && useDp4a && !useCublas
                                       && ml[4] != null && ml[4].dp4aType == 4
                                       && ml[5] != null && ml[5].dp4aType == 4
                                       && fusedGateWeights != null) {
            launchDp4aFusedGateUp(layerIdx);
        } else if (useDp4a && !useCublas && ml[4] != null && ml[4].dp4aType == 4 && ml[5] != null && ml[5].dp4aType == 4) {
            launchMatmulDp4a(ml[4]);
            launchMatmulDp4a(ml[5]);
        } else if (useFusedGateUp && !useCublas) {
            launchFusedGateUp(layerIdx);
        } else {
            launchMatmulCublas(ml[4], layerIdx, 4); // gate
            launchMatmulCublas(ml[5], layerIdx, 5); // up
        }

        // 9. Fused SiLU + element-wise multiply (fully fixed params)
        launchKernel(siluMulFunc, siluGridDim, (int) blockSize, 0, siluPB.ptrs);

        // dp4a: pre-quantize gpuHb (silu out) → gpuHbQ8 for the Down projection
        quantizeHb();
        // 10. Down projection
        launchMatmulCublasOrDp4a(ml[6], layerIdx, 6); // down

        // 10b. Granite residual scaling for FFN
        if (graniteResidualScale > 0) {
            launchSaxpy(gpuX, gpuXb, graniteResidualScale, dim);
        }

        // 10c. Post-FFN norm + accumulate (Gemma2/3)
        if (hasPostNorm) {
            postNormPB.setLong(2, gpuPostFfnNormWeights[layerIdx]);
            launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, postNormPB.ptrs);
            launchBias(gpuX, gpuXb, dim, numWorkGroups);
        }
    }

    private void forwardLayerProfiled(InferenceState state, TransformerLayerWeights layerWeights,
                                       int layerIdx, int position, Attention attention) {
        MatmulLaunch[] ml = layerMatmuls[layerIdx];
        long t0 = System.nanoTime(), t1;
        long tTotal = t0;

        if (hasPreNorm) {
            normAndQuantizeXb(gpuAttnNormWeights[layerIdx]);
        } else {
            cudaContext.copyBufferDtoD(gpuXb, gpuX, (long) dim * Float.BYTES);
            quantizeXb();
        }
        cudaContext.finish(); t1 = System.nanoTime(); profAttnNorm += t1 - t0; t0 = t1;

        if (hasMergedQKV) {
            launchMatmulDp4a(ml[0]);
            launchKernel(splitQkvFunc, splitQkvGridDim, (int) blockSize, 0, splitQkvPB.ptrs);
        } else {
            launchMatmulDp4a(ml[0]);
            launchMatmulDp4a(ml[1]);
            launchMatmulDp4a(ml[2]);
        }
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

        // dp4a: quantize gpuXb2 (attention output) before Wo
        quantizeXb2();
        launchMatmulDp4a(ml[3]);
        if (hasPostNorm) {
            postNormPB.setLong(2, gpuPostAttnNormWeights[layerIdx]);
            launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, postNormPB.ptrs);
            launchBias(gpuX, gpuXb, dim, numWorkGroups);
        }
        cudaContext.finish(); t1 = System.nanoTime(); profWo += t1 - t0; t0 = t1;

        if (hasPreNorm) {
            normAndQuantizeXb(gpuFfnNormWeights[layerIdx]);
        } else {
            cudaContext.copyBufferDtoD(gpuXb, gpuX, (long) dim * Float.BYTES);
            quantizeXb();
        }
        cudaContext.finish(); t1 = System.nanoTime(); profFfnNorm += t1 - t0; t0 = t1;

        if (hasPackedFFN) {
            launchMatmulDp4a(ml[5]);
            launchKernel(splitGateUpFunc, splitGateUpGridDim, (int) blockSize, 0, splitGateUpPB.ptrs);
        } else if (useDp4a && ml[4] != null && ml[4].dp4aType == 4 && ml[5] != null && ml[5].dp4aType == 4) {
            launchMatmulDp4a(ml[4]);
            launchMatmulDp4a(ml[5]);
        } else if (useFusedGateUp) {
            launchFusedGateUp(layerIdx);
        } else {
            launchMatmul(ml[4]);
            launchMatmul(ml[5]);
        }
        cudaContext.finish(); t1 = System.nanoTime(); profGateUp += t1 - t0; t0 = t1;

        launchKernel(siluMulFunc, siluGridDim, (int) blockSize, 0, siluPB.ptrs);
        // dp4a: quantize gpuHb (silu output) before Down
        quantizeHb();
        launchMatmulDp4a(ml[6]);
        if (hasPostNorm) {
            postNormPB.setLong(2, gpuPostFfnNormWeights[layerIdx]);
            launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, postNormPB.ptrs);
            launchBias(gpuX, gpuXb, dim, numWorkGroups);
        }
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

                for (int layer = 0; layer < gpuLayerCount; layer++) {
                    forwardLayerKernels(layer);
                }

                // Final RMSNorm + output projection
                normPB.setLong(2, gpuOutputNormWeights);
                launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);
                launchOutputMatmul();

                MemorySegment graph = cudaContext.endCapture();
                capturing = false;
                graphExec = cudaContext.instantiateGraph(graph);
                cudaContext.destroyGraph(graph);

                System.err.println("CUDA graph: captured " + gpuLayerCount + " layers + output projection");
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

        // Granite logit scaling (not in graph — applied after graph launch)
        if (graniteLogitScale != 0) {
            scalePB.setLong(0, gpuLogits);
            scalePB.setFloat(1, graniteLogitScale);
            scalePB.setInt(2, vocabSize);
            int logitsGridDim = (int) ((vocabSize + blockSize - 1) / blockSize);
            launchKernel(scaleFunc, logitsGridDim, (int) blockSize, 0, scalePB.ptrs);
        }

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

        // Attention norm (or plain copy for post-norm-only) + dp4a quantize fused if possible.
        if (hasPreNorm) {
            normAndQuantizeXb(gpuAttnNormWeights[layerIdx]);
        } else {
            cudaContext.copyBufferDtoD(gpuXb, gpuX, (long) dim * Float.BYTES);
            quantizeXb();
        }

        // QKV projections
        if (hasMergedQKV) {
            launchMatmulDp4a(ml[0]); // wqkv → gpuQKV
            launchKernel(splitQkvFunc, splitQkvGridDim, (int) blockSize, 0, splitQkvPB.ptrs);
        } else {
            launchMatmulDp4a(ml[0]);
            launchMatmulDp4a(ml[1]);
            launchMatmulDp4a(ml[2]);
        }

        // QKV bias (Qwen2)
        if (hasBias) {
            launchBias(gpuQ, gpuQBias[layerIdx], qDim, biasQGridDim);
            launchBias(gpuK, gpuKBias[layerIdx], kvDim, biasKVGridDim);
            launchBias(gpuV, gpuVBias[layerIdx], kvDim, biasKVGridDim);
        }

        // Per-head QK-norm (Qwen3, Gemma3): normalize each head independently
        if (hasQKNorm) {
            perHeadNormPB.setLong(0, gpuQ);
            perHeadNormPB.setLong(1, gpuQNormWeights[layerIdx]);
            launchKernel(perHeadNormFunc, headCount, perHeadNormBlockDim, perHeadNormSharedMem, perHeadNormPB.ptrs);
            perHeadNormPB.setLong(0, gpuK);
            perHeadNormPB.setLong(1, gpuKNormWeights[layerIdx]);
            launchKernel(perHeadNormFunc, headCountKV, perHeadNormBlockDim, perHeadNormSharedMem, perHeadNormPB.ptrs);
        }

        // Granite attention scale correction
        if (graniteAttentionScale > 0) {
            float correction = graniteAttentionScale * (float) Math.sqrt(headSize);
            scalePB.setLong(0, gpuQ);
            scalePB.setFloat(1, correction);
            scalePB.setInt(2, qDim);
            int qGridDim = (int) ((qDim + blockSize - 1) / blockSize);
            launchKernel(scaleFunc, qGridDim, (int) blockSize, 0, scalePB.ptrs);
        }

        // RoPE on Q and K (skip for NoPE layers in SmolLM3/Llama4 iRoPE)
        if (noRopeLayerInterval == 0 || (layerIdx % noRopeLayerInterval) != (noRopeLayerInterval - 1)) {
            ropePB.setLong(0, gpuQ);
            ropePB.setInt(3, headCount);
            launchKernel(ropeFunc, ropeQGridDim, (int) blockSize, 0, ropePB.ptrs);
            ropePB.setLong(0, gpuK);
            ropePB.setInt(3, headCountKV);
            launchKernel(ropeFunc, ropeKGridDim, (int) blockSize, 0, ropePB.ptrs);
        }

        // KV cache update
        kvPB.setLong(0, gpuKeyCache[layerIdx]);
        kvPB.setLong(1, gpuValueCache[layerIdx]);
        launchKernel(kvCacheUpdateFunc, kvUpdateGridDim, (int) blockSize, 0, kvPB.ptrs);

        // Full attention (shared mem fixed at maxSeqLen for graph compatibility)
        attnPB.setLong(2, gpuKeyCache[layerIdx]);
        attnPB.setLong(3, gpuValueCache[layerIdx]);
        launchKernel(attentionFullFunc, headCount, (int) attnBlockSize, graphAttnSharedMem, attnPB.ptrs);

        // dp4a: pre-quantize gpuXb2 (attention output) → gpuXb2Q8 for the Wo projection
        quantizeXb2();
        launchMatmulDp4a(ml[3]);

        // Granite residual scaling for attention output
        if (graniteResidualScale > 0) {
            launchSaxpy(gpuX, gpuXb, graniteResidualScale, dim);
        }

        // Post-attention norm + accumulate (Gemma2/3): rmsnorm(gpuXb) in-place, then gpuX += gpuXb
        if (hasPostNorm) {
            postNormPB.setLong(2, gpuPostAttnNormWeights[layerIdx]);
            launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, postNormPB.ptrs);
            launchBias(gpuX, gpuXb, dim, numWorkGroups);
        }

        // FFN norm (or plain copy for post-norm-only) + dp4a re-quantize fused if possible.
        if (hasPreNorm) {
            normAndQuantizeXb(gpuFfnNormWeights[layerIdx]);
        } else {
            cudaContext.copyBufferDtoD(gpuXb, gpuX, (long) dim * Float.BYTES);
            quantizeXb();
        }

        // Gate + Up projections — preference order:
        //   packed FFN (Phi-3/4) → dp4a-fused (single kernel, halved input reads) →
        //   2× dp4a separate → FP32 fused → 2× FP32
        if (hasPackedFFN) {
            launchMatmulDp4a(ml[5]);
            launchKernel(splitGateUpFunc, splitGateUpGridDim, (int) blockSize, 0, splitGateUpPB.ptrs);
        } else if (useDp4aFusedGateUp && useDp4a && ml[4] != null && ml[4].dp4aType == 4
                                       && ml[5] != null && ml[5].dp4aType == 4
                                       && fusedGateWeights != null) {
            launchDp4aFusedGateUp(layerIdx);
        } else if (useDp4a && ml[4] != null && ml[4].dp4aType == 4 && ml[5] != null && ml[5].dp4aType == 4) {
            launchMatmulDp4a(ml[4]);
            launchMatmulDp4a(ml[5]);
        } else if (useFusedGateUp) {
            launchFusedGateUp(layerIdx);
        } else {
            launchMatmul(ml[4]);
            launchMatmul(ml[5]);
        }

        // SiLU + element-wise multiply
        launchKernel(siluMulFunc, siluGridDim, (int) blockSize, 0, siluPB.ptrs);

        // dp4a: pre-quantize gpuHb (silu output) → gpuHbQ8 for the Down projection
        quantizeHb();
        launchMatmulDp4a(ml[6]);

        // Granite residual scaling for FFN output
        if (graniteResidualScale > 0) {
            launchSaxpy(gpuX, gpuXb, graniteResidualScale, dim);
        }

        // Post-FFN norm + accumulate (Gemma2/3): rmsnorm(gpuXb) in-place, then gpuX += gpuXb
        if (hasPostNorm) {
            postNormPB.setLong(2, gpuPostFfnNormWeights[layerIdx]);
            launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, postNormPB.ptrs);
            launchBias(gpuX, gpuXb, dim, numWorkGroups);
        }
    }

    // --- Launch helpers (zero allocation) ---

    /**
     * Launch fused gate+up Q4_K kernel for a given layer.
     * Single kernel computes both gate and up projections, reading input once.
     */
    private void launchFusedGateUp(int layerIdx) {
        fusedGateUpPB.setLong(0, fusedGateWeights[layerIdx]);
        fusedGateUpPB.setLong(1, fusedUpWeights[layerIdx]);
        launchKernel(fusedGateUpFunc, fusedGateUpGridDim, fusedGateUpBlockDim, 0, fusedGateUpPB.ptrs);
    }

    /**
     * Fused dp4a gate+up: single kernel computes both projections, reading the Q8_1
     * input ONCE (vs two separate dp4a launches that each read it). Halves input
     * bandwidth for the FFN gate/up stage and saves one kernel-launch's worth of
     * dispatch overhead.
     * Uses same grid/block as the standard Q4_K dp4a kernel (ffnDim is the rows,
     * blockDim=256, 8 rows/block).
     */
    private void launchDp4aFusedGateUp(int layerIdx) {
        dp4aFusedGateUpPB.setLong(0, fusedGateWeights[layerIdx]);
        dp4aFusedGateUpPB.setLong(1, fusedUpWeights[layerIdx]);
        // Same grid/block layout as single dp4a Q4_K matmul over (rows=ffnDim, cols=dim)
        int blockDim = 256;
        int gridDim = (ffnDim + 7) / 8;
        launchKernel(dp4aFusedGateUpFunc, gridDim, blockDim, 0, dp4aFusedGateUpPB.ptrs);
    }

    /**
     * Launch a matmul: cuBLAS sgemv if available for this tensor, otherwise custom kernel.
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
     * Launch matmul via dp4a int8 kernel (Q4_K/Q5_K/Q6_K) reading pre-quantized Q8_1 input.
     * Falls back to {@link #launchMatmul} if dp4a is disabled, the tensor type is not eligible,
     * or the dp4a kernel for that type wasn't compiled.
     * Caller must have already populated ml.q8InputPtr via the corresponding quantize* call.
     */
    private void launchMatmulDp4a(MatmulLaunch ml) {
        if (!useDp4a || ml.dp4aType == 0 || ml.q8InputPtr == 0
            || "true".equals(System.getProperty("cuda.dp4a.bypass", "false"))) {
            launchMatmul(ml);
            return;
        }
        MemorySegment func;
        int gridDim = ml.gridDim;
        int blockDim = ml.blockDim;
        switch (ml.dp4aType) {
            case 4:
                if (useMmvqQ4k) {
                    // mmvq: 4 warps × 32 lanes per row (llama.cpp pattern), 1 row per block.
                    func = mmvqQ4kFunc;
                    blockDim = 128;
                    gridDim = ml.rows;
                } else if (useDp4aMw) {
                    func = dp4aQ4kMwFunc;
                    blockDim = 128;       // 4 warps × 32 lanes (llama.cpp pattern)
                    gridDim = ml.rows;    // 1 row per block
                } else if (useDp4aQ4kMr4 && ml.rows % 4 == 0) {
                    // mr4: 4 warps × 32 lanes = 128 threads/block, each warp 4 rows = 16 rows/block
                    func = dp4aQ4kMr4Func;
                    blockDim = 128;
                    gridDim = (ml.rows + 15) / 16;
                } else {
                    func = dp4aQ4kFunc;
                }
                break;
            case 5:
                if ("false".equals(System.getProperty("cuda.dp4a.q5", "true"))) { launchMatmul(ml); return; }
                func = dp4aQ5kFunc; break;
            case 6:
                // Q6_K dp4a kernel — rewritten 2026-04-14 with byte loads (Q6_K block is 210
                // bytes, not 4-byte aligned). Bit-equivalent to FP32 reference but **measured
                // SLOWER than the FP32 Q6_K kernel** on Llama-1B (70.6 vs 72.85 tok/s) because
                // the byte-load overhead overwhelms the dp4a benefit on Q6_K's heavier format.
                // Default OFF; opt-in via -Dcuda.dp4a.q6=true (kept for ground-truth checks).
                if (!"true".equals(System.getProperty("cuda.dp4a.q6", "false"))) { launchMatmul(ml); return; }
                func = dp4aQ6kFunc; break;
            case 3:
                // Q3_K dp4a kernel. Q3_K block = 110 bytes (not 4-byte aligned) → byte __ldg.
                // Risk profile similar to Q6_K, but Q3_K has less scale-decode overhead per byte
                // (2-bit + 1-bit vs Q6_K's 4+2-bit). Default ON, opt-out via -Dcuda.dp4a.q3=false
                // if a bench regression shows up on a specific model.
                if ("false".equals(System.getProperty("cuda.dp4a.q3", "true"))) { launchMatmul(ml); return; }
                func = dp4aQ3kFunc; break;
            case 50:  // Q5_0 — fixes Gemma-3-1B (Q5_0 used for QKV/gate/up)
                if (dp4aQ50SmemFunc != null) {
                    func = dp4aQ50SmemFunc;
                    // Smem: (cols/32) * 40 bytes of Q8_1 input cache
                    // Reuse existing gridDim/blockDim; only smem changes at launch
                    dp4aPB.setLong(0, ml.weightPtr);
                    dp4aPB.setLong(1, ml.q8InputPtr);
                    dp4aPB.setLong(2, ml.outputPtr);
                    dp4aPB.setInt(3, ml.rows);
                    dp4aPB.setInt(4, ml.cols);
                    dp4aPB.setInt(5, ml.addToOutput);
                    launchKernel(func, gridDim, blockDim, (ml.cols / 32) * 40, dp4aPB.ptrs);
                    return;
                }
                func = dp4aQ50Func; break;
            case 80:  // Q8_0 — fixes Qwen3 Q8_0 models
                func = dp4aQ80Func; break;
            case 41:  // IQ4_NL — fixes Phi-3-mini IQ4_NL
                if (dp4aIq4nlSmemFunc != null) {
                    func = dp4aIq4nlSmemFunc;
                    dp4aPB.setLong(0, ml.weightPtr);
                    dp4aPB.setLong(1, ml.q8InputPtr);
                    dp4aPB.setLong(2, ml.outputPtr);
                    dp4aPB.setInt(3, ml.rows);
                    dp4aPB.setInt(4, ml.cols);
                    dp4aPB.setInt(5, ml.addToOutput);
                    launchKernel(func, gridDim, blockDim, (ml.cols / 32) * 40, dp4aPB.ptrs);
                    return;
                }
                func = dp4aIq4nlFunc; break;
            case 42:  // IQ4_XS — fixes Gemma-2 IQ4_XS
                // Multi-warp kernel is much better for small-cols matmuls where single-warp
                // would leave most lanes idle (e.g. cols=2304: 9 of 32 lanes working).
                // Default on; disable with -Dcuda.dp4a.iq4xs.mw=false.
                if (dp4aIq4xsMwFunc != null
                    && !"false".equals(System.getProperty("cuda.dp4a.iq4xs.mw", "true"))) {
                    func = dp4aIq4xsMwFunc;
                    blockDim = 128;
                    gridDim = ml.rows;
                } else {
                    func = dp4aIq4xsFunc;
                }
                break;
            default: launchMatmul(ml); return;
        }
        if (func == null) { launchMatmul(ml); return; }
        dp4aPB.setLong(0, ml.weightPtr);
        dp4aPB.setLong(1, ml.q8InputPtr);
        dp4aPB.setLong(2, ml.outputPtr);
        dp4aPB.setInt(3, ml.rows);
        dp4aPB.setInt(4, ml.cols);
        dp4aPB.setInt(5, ml.addToOutput);
        launchKernel(func, gridDim, blockDim, 0, dp4aPB.ptrs);
    }

    /**
     * Hybrid: prefer cuBLAS for this matmul; if unavailable, use dp4a; if also unavailable, FP32 kernel.
     */
    private void launchMatmulCublasOrDp4a(MatmulLaunch ml, int layerIdx, int matmulIdx) {
        if (useCublas && cublasF32Weights[layerIdx][matmulIdx] != 0) {
            launchMatmulCublas(ml, layerIdx, matmulIdx);
        } else {
            launchMatmulDp4a(ml);
        }
    }

    /** Quantize gpuXb (FP32, dim) → gpuXbQ8 (Q8_1). No-op if dp4a disabled. */
    private void quantizeXb() {
        if (!useDp4a) return;
        launchKernel(quantizeFunc, quantizeXbGridDim, 256, 0, quantizeXbPB.ptrs);
    }

    /**
     * Fused RMSNorm(gpuX → gpuXb) + Quantize(gpuXb → gpuXbQ8) in a single kernel launch.
     * Saves vs separate-kernel: 1 launch overhead + 1 HBM round-trip on gpuXb (size×4 bytes).
     * Falls back to separate norm + quantize when fused kernel disabled.
     */
    private void normAndQuantizeXb(long normWeights) {
        if (useFusedNormQuantize) {
            rmsnormQuantizePB.setLong(3, normWeights);
            launchKernel(rmsnormQuantizeFunc, 1, (int) blockSize, normSharedMem, rmsnormQuantizePB.ptrs);
        } else {
            normPB.setLong(2, normWeights);
            launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);
            quantizeXb();
        }
    }

    /**
     * Output-projection helper: quantize gpuXb → gpuXbQ8 then launch the (large) output
     * matmul as dp4a. Fallback: plain FP32 matmul. The output matmul's static q8InputPtr is
     * 0 (it was constructed before dp4a init), so we wire the buffer in here at launch time.
     */
    private void launchOutputMatmul() {
        if (!useDp4a || outputMatmul.dp4aType == 0) {
            launchMatmul(outputMatmul);
            return;
        }
        // Select kernel by tensor type. Q6_K supported now (kernel rewritten 2026-04-14).
        MemorySegment outFunc;
        switch (outputMatmul.dp4aType) {
            case 4: outFunc = dp4aQ4kFunc; break;
            case 5:
                if ("false".equals(System.getProperty("cuda.dp4a.q5", "true"))) { launchMatmul(outputMatmul); return; }
                outFunc = dp4aQ5kFunc; break;
            case 6:
                if (!"true".equals(System.getProperty("cuda.dp4a.q6", "false"))) { launchMatmul(outputMatmul); return; }
                outFunc = dp4aQ6kFunc; break;
            case 3:
                if ("false".equals(System.getProperty("cuda.dp4a.q3", "true"))) { launchMatmul(outputMatmul); return; }
                outFunc = dp4aQ3kFunc; break;
            // Fill previous coverage gap: output weights quantized as Q5_0/Q8_0/IQ4 fell
            // through to FP32 kernel. Now use the same dp4a kernels already in use for
            // layer matmuls (Qwen35CudaForwardPass has had this wiring all along).
            case 50: outFunc = dp4aQ50Func; break;
            case 80: outFunc = dp4aQ80Func; break;
            case 41: outFunc = dp4aIq4nlFunc; break;
            case 42: outFunc = dp4aIq4xsFunc; break;
            default: launchMatmul(outputMatmul); return;
        }
        if (outFunc == null) { launchMatmul(outputMatmul); return; }
        // Quantize freshly-normalized gpuXb (output of final RMSNorm) for dp4a output projection
        quantizeXb();
        dp4aPB.setLong(0, outputMatmul.weightPtr);
        dp4aPB.setLong(1, gpuXbQ8);
        dp4aPB.setLong(2, outputMatmul.outputPtr);
        dp4aPB.setInt(3, outputMatmul.rows);
        dp4aPB.setInt(4, outputMatmul.cols);
        dp4aPB.setInt(5, outputMatmul.addToOutput);
        launchKernel(outFunc, outputMatmul.gridDim, outputMatmul.blockDim, 0, dp4aPB.ptrs);
    }

    /** Quantize gpuXb2 (FP32, qDim) → gpuXb2Q8 (Q8_1). No-op if dp4a disabled. */
    private void quantizeXb2() {
        if (!useDp4a) return;
        launchKernel(quantizeFunc, quantizeXb2GridDim, 256, 0, quantizeXb2PB.ptrs);
    }

    /** Quantize gpuHb (FP32, ffnDim) → gpuHbQ8 (Q8_1). No-op if dp4a disabled. */
    private void quantizeHb() {
        if (!useDp4a) return;
        launchKernel(quantizeFunc, quantizeHbGridDim, 256, 0, quantizeHbPB.ptrs);
    }

    /**
     * Launch matmul with cuBLAS override for a specific layer and matmul index.
     * Falls back to custom kernel if cuBLAS is not available for this tensor.
     */
    private void launchMatmulCublas(MatmulLaunch ml, int layerIdx, int matmulIdx) {
        if (useCublas && cublasF32Weights[layerIdx][matmulIdx] != 0) {
            if (cublasUseFP16) {
                cublasMatmul.gemmExF16(cublasF32Weights[layerIdx][matmulIdx],
                    ml.inputPtr, ml.outputPtr, ml.rows, ml.cols, ml.addToOutput == 1,
                    cublasConvertF16Func, cudaContext);
            } else {
                cublasMatmul.sgemv(cublasF32Weights[layerIdx][matmulIdx],
                    ml.inputPtr, ml.outputPtr, ml.rows, ml.cols, ml.addToOutput == 1);
            }
        } else {
            launchMatmul(ml);
        }
    }

    /** Launch saxpy: y[i] += a * x[i] */
    private void launchSaxpy(long y, long x, float a, int size) {
        scalePB.setLong(0, x);
        scalePB.setFloat(1, a);
        scalePB.setInt(2, size);
        // Use scale_inplace to compute x *= a in-place, then accumulate y += x
        launchKernel(scaleFunc, (int) ((size + blockSize - 1) / blockSize), (int) blockSize, 0, scalePB.ptrs);
        launchBias(y, x, size, (int) ((size + blockSize - 1) / blockSize));
    }

    private static FloatTensor getLayerTensor(TransformerLayerWeights layer, int idx) {
        switch (idx) {
            case 0: return layer.wqkv() != null ? layer.wqkv() : layer.wq();
            case 1: return layer.wk();
            case 2: return layer.wv();
            case 3: return layer.wo();
            case 4: return layer.wGate();
            case 5: return layer.wUp();
            case 6: return layer.wDown();
            default: return null;
        }
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
        try { cudaContext.freeBuffer(gpuCombined); } catch (Exception ignored) {} // frees gpuX + gpuTokenParams
        try { cudaContext.freeBuffer(gpuXb); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuXb2); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuHb); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuHb2); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuQ); } catch (Exception ignored) {}
        if (gpuQKV != 0) try { cudaContext.freeBuffer(gpuQKV); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuK); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuV); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuPartialSums); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuCosTable); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuSinTable); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuOutputNormWeights); } catch (Exception ignored) {}
        if (gpuLogits != 0) try { cudaContext.freeBuffer(gpuLogits); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuArgmaxPartialVal); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuArgmaxPartialIdx); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuArgmaxResult); } catch (Exception ignored) {}
        if (useDp4a) {
            try { cudaContext.freeBuffer(gpuXbQ8); } catch (Exception ignored) {}
            try { cudaContext.freeBuffer(gpuXb2Q8); } catch (Exception ignored) {}
            try { cudaContext.freeBuffer(gpuHbQ8); } catch (Exception ignored) {}
        }
        if (gpuQBias != null) {
            for (long ptr : gpuQBias) { if (ptr != 0) try { cudaContext.freeBuffer(ptr); } catch (Exception ignored) {} }
            for (long ptr : gpuKBias) { if (ptr != 0) try { cudaContext.freeBuffer(ptr); } catch (Exception ignored) {} }
            for (long ptr : gpuVBias) { if (ptr != 0) try { cudaContext.freeBuffer(ptr); } catch (Exception ignored) {} }
        }
        if (gpuQNormWeights != null) {
            for (long ptr : gpuQNormWeights) { if (ptr != 0) try { cudaContext.freeBuffer(ptr); } catch (Exception ignored) {} }
            for (long ptr : gpuKNormWeights) { if (ptr != 0) try { cudaContext.freeBuffer(ptr); } catch (Exception ignored) {} }
        }
        if (gpuPostAttnNormWeights != null) {
            for (long ptr : gpuPostAttnNormWeights) { if (ptr != 0) try { cudaContext.freeBuffer(ptr); } catch (Exception ignored) {} }
            for (long ptr : gpuPostFfnNormWeights) { if (ptr != 0) try { cudaContext.freeBuffer(ptr); } catch (Exception ignored) {} }
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
