package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.gpu.CudaBindings;
import it.denzosoft.llmplayer.gpu.CudaBufferManager;
import it.denzosoft.llmplayer.gpu.CudaContext;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.model.Qwen35LayerWeights;
import it.denzosoft.llmplayer.model.Qwen35Weights;
import it.denzosoft.llmplayer.tensor.CudaFloatTensor;
import it.denzosoft.llmplayer.tensor.FloatTensor;

import java.lang.foreign.*;

/**
 * CUDA GPU-resident forward pass for Qwen3.5 hybrid DeltaNet + attention architecture.
 * Handles both DeltaNet layers (3/4) and full GQA attention layers (1/4) on GPU.
 *
 * DeltaNet layers: RMSNorm → QKV matmul → conv1d → SiLU → alpha/beta gates → gate matmul →
 *                  recurrence → per-head norm → gate multiply → output matmul → residual → FFN
 * Attention layers: RMSNorm → Q(packed+gate) matmul → deinterleave → K,V matmul → QK-norm →
 *                   RoPE → KV cache → attention → sigmoid gate → output matmul → residual → FFN
 *
 * ZERO-ALLOCATION hot path: all kernel param buffers are pre-allocated in the constructor.
 */
public class Qwen35CudaForwardPass implements LayerGpuForwardPass {

    private final CudaContext cudaContext;
    private final CudaBufferManager bufferManager;
    private final Arena arena;

    // Qwen3.5-MoE layers: norm and the gated shared expert on the GPU, routed experts through the
    // engine's MoeFfn callback (CPU experts, GPU hot-expert cache) — one sync per layer.
    private final boolean[] moeLayer;
    private final boolean hasMoe;
    private final boolean[] sharedOnGpu;
    private final int sharedFfn;
    private LayerGpuForwardPass.MoeFfn moeFfn;
    private final Qwen35Weights weights;
    private long gpuShG, gpuShU, gpuShOut, gpuShLogit, gpuRouted;
    private MemorySegment hostMoe;                 // pinned [xn | routed]
    private float[] moeXn, moeOut;
    private it.denzosoft.llmplayer.gpu.Dp4aMatmul mmShIn, mmShDown;
    private MemorySegment shSiluFunc, gateAxpyFunc;
    private it.denzosoft.llmplayer.gpu.KernelParams shSiluPB, shAxpyPB, routedAxpyPB;

    // GPU activation buffers
    private final long gpuX;           // [dim] main activation
    private final long gpuXb;          // [dim] after norm
    private final long gpuHb;          // [ffnDim] FFN hidden
    private final long gpuHb2;         // [ffnDim] FFN hidden 2

    // DeltaNet buffers
    private final long gpuQkv;         // [deltaQkvDim] QKV after projection
    private final long gpuAlpha;       // [timeStepRank] alpha gate
    private final long gpuBeta;        // [timeStepRank] beta gate
    private final long gpuGate;        // [innerSize] output gate
    private final long gpuDeltaOut;    // [innerSize] DeltaNet output

    // Attention buffers
    private final long gpuQ;           // [qGateDim] packed Q+gate from projection
    private final long gpuQCompact;    // [qDim] Q after deinterleaving
    private final long gpuAttnGate;    // [qDim] gate after deinterleaving
    private final long gpuK;           // [kvDim]
    private final long gpuV;           // [kvDim]
    private final long gpuXb2;         // [qDim] attention output

    // Host staging buffers
    private final MemorySegment hostX;

    // Combined upload: [embedding (dim*4)] + [tokenParams (8 bytes: position, seqLen)]
    private final MemorySegment hostCombined;
    private final long gpuCombined;
    private final long gpuTokenParams; // device pointer to tokenParams within gpuCombined

    // DeltaNet persistent state on GPU
    private final long[] gpuSsmState;      // [blockCount] -> [timeStepRank * dQK * dV] per DeltaNet layer (0 for attn)
    private final long[] gpuConvState;     // [blockCount] -> [histSize * deltaQkvDim] per DeltaNet layer (0 for attn)

    // KV cache for attention layers
    private final long[] gpuKeyCache;      // [blockCount] -> [maxSeqLen * kvDim] (0 for DeltaNet layers)
    private final long[] gpuValueCache;    // [blockCount]

    // RoPE tables
    private final long gpuCosTable;
    private final long gpuSinTable;

    // Per-layer norm weights on GPU
    private final long[] gpuAttnNormWeights;   // [blockCount]
    private final long[] gpuFfnNormWeights;    // [blockCount] (postAttnNorm)

    // Per DeltaNet layer: conv1d weights, ssmA, dtBias, ssmNorm
    private final long[] gpuConvWeights;       // [blockCount]
    private final long[] gpuSsmAWeights;       // [blockCount]
    private final long[] gpuDtBiasWeights;     // [blockCount]
    private final long[] gpuSsmNormWeights;    // [blockCount]

    // Per attention layer: QK-norm weights
    private final long[] gpuQNormWeights;      // [blockCount]
    private final long[] gpuKNormWeights;      // [blockCount]

    // dp4a: Q8_1 quantized input buffer + kernels
    private final boolean useDp4a;
    private final long gpuXbQ8;             // Q8_1 buffer: (dim/32) * 40 bytes
    private final MemorySegment quantizeFunc;
    private final MemorySegment dp4aFunc;
    private final MemorySegment dp4aQ3kFunc;  // Q3_K dp4a kernel
    private final MemorySegment dp4aQ5kFunc;  // Q5_K dp4a kernel
    private final MemorySegment dp4aQ6kFunc;  // Q6_K dp4a kernel
    private static final boolean DP4A_Q6 = "true".equals(System.getProperty("cuda.dp4a.q6", "false"));
    // Extended dp4a kernels for non-K-quant types (mirroring CudaForwardPass extension)
    private final MemorySegment dp4aQ50Func;
    private final MemorySegment dp4aQ80Func;
    private final MemorySegment dp4aIq4nlFunc;
    private final MemorySegment dp4aIq4xsFunc;
    private final ParamBuffer quantizePB;   // 3 args: input, output, size
    private final ParamBuffer dp4aPB;       // 6 args: weights, input_q8, output, rows, cols, addToOutput
    private final int quantizeGridDim;
    private final int q8BufferSize;         // (dim/32) * 40 bytes
    // T1.1: fused rmsnorm+quantize (opt-in via -Dcuda.fused_norm_quantize=true)
    private final boolean useFusedNormQuantize;
    private final MemorySegment rmsnormQuantizeFunc;
    private final ParamBuffer rmsnormQuantizePB;  // 6 args: normOut, qOut, in, w, size, eps
    // T1.2: dp4a for output projections (opt-in via -Dcuda.dp4a.outputs=true)
    private final boolean useDp4aOutputs;
    private final long gpuScratchQ8;             // shared Q8 scratch for output-proj inputs
    private final int scratchQ8Size;             // bytes
    private final ParamBuffer scratchQuantizePB; // 3 args: input, output, size
    // T2.1: multi-row Q4_K dp4a kernel (opt-in via -Dcuda.q4k.mr4=true)
    private final boolean useDp4aQ4kMr4;
    private final MemorySegment dp4aQ4kMr4Func;

    // Output projection
    private final long gpuOutputNormWeights;
    private final long gpuLogits;
    private final long gpuLogitsBytes;
    private final MemorySegment hostLogits;

    // GPU-side argmax buffers
    private final long gpuArgmaxPartialVal;
    private final long gpuArgmaxPartialIdx;
    private final long gpuArgmaxResult;
    private final MemorySegment hostArgmaxResult;
    private final int argmaxNumBlocks;

    // Fused FFN gate+up
    private final boolean useFusedGateUp;
    private final MemorySegment fusedGateUpFunc;
    private final long[] fusedGateWeights;
    private final long[] fusedUpWeights;
    private final int fusedGateUpGridDim;
    private final int fusedGateUpBlockDim;

    // Pre-compiled CUDA functions
    private final MemorySegment rmsnormFusedFunc;
    private final MemorySegment siluFunc;
    private final MemorySegment siluMulFunc;
    private final MemorySegment ropeFunc;
    private final MemorySegment kvCacheUpdateFunc;
    // Single-token attention (flash-decoding, legacy attention_full as opt-out)
    private FlashAttention flashAttn;
    // FP16 KV cache (-Dcuda.kv.fp16, also set by the KV-aware VRAM budget)
    private final boolean useFp16Kv = "true".equals(System.getProperty("cuda.kv.fp16", "false"));
    private final MemorySegment accumulateFunc;
    private final MemorySegment elementwiseMulFunc;
    private final MemorySegment perHeadNormFunc;
    private final MemorySegment conv1dFunc;
    private final MemorySegment deltanetFunc;
    private final MemorySegment alphaBetaFunc;
    private final MemorySegment deinterleaveFunc;
    private final MemorySegment sigmoidMulFunc;

    // Default CUDA stream
    private final MemorySegment defaultStream;

    // Dimensions
    private final int dim;
    private final int qDim;
    private final int qGateDim;
    private final int kvDim;
    private final int ffnDim;
    private final int vocabSize;
    private final int headCount;
    private final int headCountKV;
    private final int headSize;
    private final int halfRope;
    private final int ropeType;
    private final float normEps;
    private final int blockCount;
    private final int maxSeqLen;
    private final int fullAttnInterval;

    // DeltaNet dimensions
    private final int timeStepRank;
    private final int stateSize;  // dQK = dV
    private final int groupCount;
    private final int innerSize;
    private final int convKernel;
    private final int deltaQkvDim;

    private final long blockSize;  // CUDA block size for 1D kernels

    // Layer type lookup
    private final boolean[] isDeltaNet;

    // === Pre-allocated ParamBuffers ===
    private static class ParamBuffer {
        final MemorySegment args;
        final MemorySegment ptrs;

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

        long getLong(int argIndex) {
            return args.get(ValueLayout.JAVA_LONG, argIndex * 8L);
        }
    }

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
        final int dp4aType;  // 0=none, 3=Q3_K, 4=Q4_K, 5=Q5_K, 6=Q6_K, 50=Q5_0, 80=Q8_0, 41=IQ4_NL, 42=IQ4_XS

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
            var t = tensor.type();
            if (t == it.denzosoft.llmplayer.tensor.GGMLType.Q4_K) dp4aType = 4;
            else if (t == it.denzosoft.llmplayer.tensor.GGMLType.Q5_K) dp4aType = 5;
            else if (t == it.denzosoft.llmplayer.tensor.GGMLType.Q6_K) dp4aType = 6;
            else if (t == it.denzosoft.llmplayer.tensor.GGMLType.Q3_K) dp4aType = 3;
            else if (t == it.denzosoft.llmplayer.tensor.GGMLType.Q5_0) dp4aType = 50;
            else if (t == it.denzosoft.llmplayer.tensor.GGMLType.Q8_0) dp4aType = 80;
            else if (t == it.denzosoft.llmplayer.tensor.GGMLType.IQ4_NL) dp4aType = 41;
            else if (t == it.denzosoft.llmplayer.tensor.GGMLType.IQ4_XS) dp4aType = 42;
            else dp4aType = 0;
        }
    }

    // Reusable param buffers
    private final ParamBuffer matmulPB;       // 6 args
    private final ParamBuffer normPB;         // 5 args: out, in, weights, size, eps
    private final ParamBuffer ropePB;         // 8 args
    private final ParamBuffer kvPB;           // 6 args
    private final ParamBuffer siluPB;         // 2 args: x, size
    private final ParamBuffer siluMulPB;      // 3 args: a, b, size
    private final ParamBuffer conv1dPB;       // 6 args: qkv, convState, convWeights, channels, kernelSize, tokenParams
    private final ParamBuffer deltanetPB;     // 6 args: S, qkv, alpha, beta, output, nHeads, groupCount, dQK, dV
    private final ParamBuffer alphaBetaPB;    // 5 args: alpha, beta, negExpA, dtBias, timeStepRank
    private final ParamBuffer deinterleavePB; // 5 args: packed, q, gate, headCount, headSize
    private final ParamBuffer sigmoidMulPB;   // 3 args: a, b, size
    private final ParamBuffer perHeadNormPB;  // 4 args: vec, weights, headSize, eps
    private final ParamBuffer accPB;          // 3 args: y, x, size
    private final ParamBuffer elemMulPB;      // 3 args: a, b, size
    private final ParamBuffer argmaxPartialPBuf;  // 4 args
    private final ParamBuffer argmaxFinalPBuf;    // 4 args
    private final MemorySegment argmaxPartialFunc;
    private final MemorySegment argmaxFinalFunc;
    private final ParamBuffer fusedGateUpPB;  // 8 args (null if not fused)

    // Per-layer matmul descriptors
    // DeltaNet: [0]=attnQkv, [1]=ssmAlpha, [2]=ssmBeta, [3]=attnGate, [4]=ssmOut, [5]=ffnGate, [6]=ffnUp, [7]=ffnDown
    // Attention: [0]=wq, [1]=wk, [2]=wv, [3]=wo, [4]=unused, [5]=ffnGate, [6]=ffnUp, [7]=ffnDown
    private final MatmulLaunch[][] layerMatmuls;

    // Output projection
    private final MatmulLaunch outputMatmul;

    // Pre-computed grid sizes
    private final int normNumWarps;
    private final int normSharedMem;
    private final int ropeQGridDim;
    private final int ropeKGridDim;
    private final int kvUpdateGridDim;
    private final int siluQkvGridDim;
    private final int siluGateGridDim;
    private final int siluFFNGridDim;
    private final int conv1dGridDim;
    private final int deinterleaveGridDim;
    private final int elemMulGridDim;
    private final int accDimGridDim;
    private final int perHeadNormBlockDim;
    private final int perHeadNormSharedMem;
    private final int deltanetSharedMem;

    // CUDA graph state
    private MemorySegment graphExec;        // generation graph (layers + output)
    private MemorySegment prefillGraphExec; // prefill graph (layers only, no output)
    private boolean graphAvailable; // cleared after a failed capture (no retry per token)
    private final MemorySegment[] layerGraphExec;
    private final boolean[] layerRan;
    private boolean layerGraphsBroken;
    private boolean moeDeviceOnly; // forwardMoeFFN stops before its host part (graph capture)

    private final int gpuLayerCount;

    // Batched prefill (F6 step 5.2): per chunk and layer, every projection is one cuBLAS FP16 GEMM
    // (weights read once per chunk instead of once per token); the DeltaNet conv and recurrence run
    // token by token on the chunk's rows (the recurrence is sequential); attention layers use the
    // batch kernels and one flash launch. Full offload with libcublas only; -Dprefill.gpu.gemm=false
    // or -Dprefill.gpu.batched=false keeps the per-token prefill.
    private static final int PREFILL_BATCH = Integer.getInteger("prefill.batch", 64);
    private int maxBatch;
    private GemmF16 gemm;
    private long bX, bXn, bQkv, bAlpha, bBeta, bGate, bOut, bQ, bQc, bAg, bK, bV, bAtt, bHb, bHb2, bRouted, bTP;
    private MemorySegment bHost;                  // pinned: [maxBatch][dim] x | [maxBatch][dim] moe | tokenParams
    private MemorySegment rmsBatchFunc, ropeBatchFunc, kvBatchFunc;
    private it.denzosoft.llmplayer.gpu.KernelParams bNormPB, bRopePB, bKvPB;
    private LayerGpuForwardPass.MoeFfnBatch moeFfnBatch;
    private float[][] bMoeIn, bMoeOut;

    public Qwen35CudaForwardPass(ModelConfig config, Qwen35Weights weights,
                                  CudaBufferManager bufferManager, int maxSeqLen) {
        this.bufferManager = bufferManager;
        this.cudaContext = bufferManager.getCudaContext();
        this.arena = Arena.ofShared();
        this.weights = weights;
        this.maxSeqLen = maxSeqLen;
        this.defaultStream = cudaContext.getStream();

        // Extract dimensions
        this.dim = config.embeddingLength();
        this.headCount = config.headCount();
        this.headCountKV = config.headCountKV();
        this.headSize = config.headSize();
        this.qDim = headCount * headSize;
        this.qGateDim = qDim * 2;
        this.kvDim = config.kvDim();
        this.ffnDim = config.intermediateSize();
        this.vocabSize = config.vocabSize();
        this.normEps = config.normEps();
        this.blockCount = config.blockCount();
        this.fullAttnInterval = config.fullAttentionInterval();

        this.timeStepRank = config.ssmTimeStepRank();
        this.stateSize = config.ssmStateSize();
        this.groupCount = config.ssmGroupCount();
        this.innerSize = config.ssmInnerSize();
        this.convKernel = config.ssmConvKernel();
        this.deltaQkvDim = groupCount * stateSize * 2 + timeStepRank * stateSize;

        RoPE rope = new RoPE(headSize, config.ropeDimensionCount(), maxSeqLen,
            config.ropeFreqBase(), config.ropeType(), weights.ropeFreqFactors());
        this.halfRope = rope.getRopeDimCount() / 2;
        this.ropeType = rope.getRopeType();

        long maxWg = cudaContext.getDeviceInfo().maxWorkGroupSize();
        this.blockSize = Math.min(256, maxWg);

        // Build layer type map
        isDeltaNet = new boolean[blockCount];
        for (int i = 0; i < blockCount; i++) {
            isDeltaNet[i] = weights.layers()[i].isDeltaNet();
        }

        // Count GPU layers
        this.gpuLayerCount = countGpuLayers(weights);

        long fb = Float.BYTES;

        // Combined upload: embedding + tokenParams (position, seqLen)
        long combinedBytes = dim * fb + 8;
        gpuCombined = bufferManager.createBuffer(combinedBytes);
        gpuX = gpuCombined;
        gpuTokenParams = gpuCombined + dim * fb;
        hostCombined = arena.allocate(combinedBytes, 8);
        hostX = arena.allocate(ValueLayout.JAVA_FLOAT, dim);

        // Activation buffers
        gpuXb = bufferManager.createBuffer(dim * fb);
        gpuHb = bufferManager.createBuffer(ffnDim * fb);
        gpuHb2 = bufferManager.createBuffer(ffnDim * fb);

        moeLayer = new boolean[blockCount];
        sharedOnGpu = new boolean[blockCount];
        boolean anyMoe = false, anySharedGpu = false;
        for (int i = 0; i < gpuLayerCount; i++) {
            Qwen35LayerWeights lw = weights.layers()[i];
            moeLayer[i] = lw.isMoe();
            anyMoe |= moeLayer[i];
            sharedOnGpu[i] = moeLayer[i] && lw.ffnGateShexp() instanceof CudaFloatTensor
                && lw.ffnUpShexp() instanceof CudaFloatTensor && lw.ffnDownShexp() instanceof CudaFloatTensor
                && (lw.ffnGateInpShexp() == null || lw.ffnGateInpShexp() instanceof CudaFloatTensor);
            anySharedGpu |= sharedOnGpu[i];
        }
        hasMoe = anyMoe;
        sharedFfn = config.expertSharedFeedForwardLength();
        if (hasMoe) {
            gpuRouted = bufferManager.createBuffer(dim * fb);
            hostMoe = cudaContext.allocPinnedHost(2L * dim * fb);
            moeXn = new float[dim];
            moeOut = new float[dim];
            gateAxpyFunc = cudaContext.compileKernel("kernels/cuda/batch_ops.cu", "axpy_sigmoid_gate");
            routedAxpyPB = new it.denzosoft.llmplayer.gpu.KernelParams(arena, 4);
            routedAxpyPB.setLong(0, gpuX).setLong(1, gpuRouted).setLong(2, 0L).setInt(3, dim);
            if (anySharedGpu) {
                gpuShG = bufferManager.createBuffer((long) sharedFfn * fb);
                gpuShU = bufferManager.createBuffer((long) sharedFfn * fb);
                gpuShOut = bufferManager.createBuffer(dim * fb);
                gpuShLogit = bufferManager.createBuffer(fb);
                mmShIn = new it.denzosoft.llmplayer.gpu.Dp4aMatmul(cudaContext, bufferManager, arena, dim);
                mmShDown = new it.denzosoft.llmplayer.gpu.Dp4aMatmul(cudaContext, bufferManager, arena, Math.max(32, sharedFfn));
                shSiluFunc = cudaContext.compileKernel("kernels/cuda/silu_mul.cu", "silu_mul");
                shSiluPB = new it.denzosoft.llmplayer.gpu.KernelParams(arena, 3);
                shSiluPB.setLong(0, gpuShG).setLong(1, gpuShU).setInt(2, sharedFfn);
                shAxpyPB = new it.denzosoft.llmplayer.gpu.KernelParams(arena, 4);
                shAxpyPB.setLong(0, gpuX).setLong(1, gpuShOut).setInt(3, dim);
            }
        }


        // DeltaNet buffers
        gpuQkv = bufferManager.createBuffer(deltaQkvDim * fb);
        gpuAlpha = bufferManager.createBuffer(timeStepRank * fb);
        gpuBeta = bufferManager.createBuffer(timeStepRank * fb);
        gpuGate = bufferManager.createBuffer(innerSize * fb);
        gpuDeltaOut = bufferManager.createBuffer(innerSize * fb);

        // Attention buffers
        gpuQ = bufferManager.createBuffer(qGateDim * fb);
        gpuQCompact = bufferManager.createBuffer(qDim * fb);
        gpuAttnGate = bufferManager.createBuffer(qDim * fb);
        gpuK = bufferManager.createBuffer(kvDim * fb);
        gpuV = bufferManager.createBuffer(kvDim * fb);
        gpuXb2 = bufferManager.createBuffer(qDim * fb);

        // Upload RoPE tables
        gpuCosTable = uploadFloatArray(rope.getCosTable());
        gpuSinTable = uploadFloatArray(rope.getSinTable());

        // Allocate persistent DeltaNet state and KV cache
        int histSize = convKernel - 1;
        gpuSsmState = new long[blockCount];
        gpuConvState = new long[blockCount];
        gpuKeyCache = new long[blockCount];
        gpuValueCache = new long[blockCount];
        for (int i = 0; i < gpuLayerCount; i++) {
            if (isDeltaNet[i]) {
                long stateBytes = (long) timeStepRank * stateSize * stateSize * fb;
                gpuSsmState[i] = bufferManager.createBuffer(stateBytes);
                cudaContext.fillBufferZero(gpuSsmState[i], stateBytes);
                long convBytes = (long) histSize * deltaQkvDim * fb;
                gpuConvState[i] = bufferManager.createBuffer(convBytes);
                cudaContext.fillBufferZero(gpuConvState[i], convBytes);
            } else {
                long kvLayerBytes = (long) maxSeqLen * kvDim * (useFp16Kv ? 2L : fb);
                gpuKeyCache[i] = bufferManager.createBuffer(kvLayerBytes);
                gpuValueCache[i] = bufferManager.createBuffer(kvLayerBytes);
                cudaContext.fillBufferZero(gpuKeyCache[i], kvLayerBytes);
                cudaContext.fillBufferZero(gpuValueCache[i], kvLayerBytes);
            }
        }

        // Compile CUDA kernels
        rmsnormFusedFunc = cudaContext.compileKernel("kernels/cuda/rmsnorm.cu", "rmsnorm_fused");
        siluFunc = cudaContext.compileKernel("kernels/cuda/silu.cu", "silu");
        siluMulFunc = cudaContext.compileKernel("kernels/cuda/silu_mul.cu", "silu_mul");
        ropeFunc = cudaContext.compileKernel("kernels/cuda/rope.cu", "rope_apply");
        kvCacheUpdateFunc = useFp16Kv
            ? cudaContext.compileKernel("kernels/cuda/attention_f16.cu", "kv_cache_update_f16")
            : cudaContext.compileKernel("kernels/cuda/attention.cu", "kv_cache_update");
        accumulateFunc = cudaContext.compileKernel("kernels/cuda/accumulate.cu", "accumulate");
        elementwiseMulFunc = cudaContext.compileKernel("kernels/cuda/elementwise_mul.cu", "elementwise_mul");
        perHeadNormFunc = cudaContext.compileKernel("kernels/cuda/rmsnorm_per_head.cu", "rmsnorm_per_head");
        conv1dFunc = cudaContext.compileKernel("kernels/cuda/conv1d_silu.cu", "conv1d_silu");
        // DeltaNet kernel: v2 uses float4 vectorized state access (enabled by default)
        {
            boolean useV2 = !"false".equals(System.getProperty("cuda.deltanet.v2", "true"));
            MemorySegment dFunc2 = null;
            if (useV2) {
                try { dFunc2 = cudaContext.compileKernel("kernels/cuda/deltanet_fused_v2.cu", "deltanet_fused_v2"); }
                catch (Exception e) { System.err.println("Qwen35 CUDA: deltanet v2 unavailable: " + e.getMessage()); }
            }
            deltanetFunc = (dFunc2 != null) ? dFunc2
                : cudaContext.compileKernel("kernels/cuda/deltanet_fused.cu", "deltanet_fused");
        }
        alphaBetaFunc = cudaContext.compileKernel("kernels/cuda/alpha_beta_gates.cu", "alpha_beta_gates");
        deinterleaveFunc = cudaContext.compileKernel("kernels/cuda/deinterleave_q_gate.cu", "deinterleave_q_gate");
        sigmoidMulFunc = cudaContext.compileKernel("kernels/cuda/sigmoid_elementwise_mul.cu", "sigmoid_elementwise_mul");

        // dp4a integer dot product path (llama.cpp-style optimization)
        boolean dp4aAvail = false;
        MemorySegment qFunc = null, dFunc = null, d5kFunc = null, d6kFunc = null, d3kFunc = null;
        MemorySegment d50Func = null, d80Func = null, dIq4nlFunc = null, dIq4xsFunc = null;
        try {
            qFunc = cudaContext.compileKernel("kernels/cuda/quantize_q8.cu", "quantize_q8");
            dFunc = cudaContext.compileKernel("kernels/cuda/matmul_q4_k_dp4a.cu", "matmul_q4_k_dp4a");
            dp4aAvail = true;
            // Also compile Q5_K and Q6_K dp4a variants
            try { d5kFunc = cudaContext.compileKernel("kernels/cuda/matmul_q5_k_dp4a.cu", "matmul_q5_k_dp4a"); }
            catch (Exception ignored) {}
            try { d6kFunc = cudaContext.compileKernel("kernels/cuda/matmul_q6_k_dp4a.cu", "matmul_q6_k_dp4a"); }
            catch (Exception ignored) {}
            try { d3kFunc = cudaContext.compileKernel("kernels/cuda/matmul_q3_k_dp4a.cu", "matmul_q3_k_dp4a"); }
            catch (Exception e) { System.err.println("Qwen35 CUDA: Q3_K dp4a unavailable: " + e.getMessage()); }
            // Extended dp4a kernels (mirrors CudaForwardPass extension)
            try { d50Func    = cudaContext.compileKernel("kernels/cuda/matmul_q5_0_dp4a.cu",   "matmul_q5_0_dp4a"); }
            catch (Exception e) { System.err.println("Qwen35 CUDA: Q5_0 dp4a unavailable: " + e.getMessage()); }
            try { d80Func    = cudaContext.compileKernel("kernels/cuda/matmul_q8_0_dp4a.cu",   "matmul_q8_0_dp4a"); }
            catch (Exception e) { System.err.println("Qwen35 CUDA: Q8_0 dp4a unavailable: " + e.getMessage()); }
            try { dIq4nlFunc = cudaContext.compileKernel("kernels/cuda/matmul_iq4_nl_dp4a.cu", "matmul_iq4_nl_dp4a"); }
            catch (Exception ignored) {}
            try { dIq4xsFunc = cudaContext.compileKernel("kernels/cuda/matmul_iq4_xs_dp4a.cu", "matmul_iq4_xs_dp4a"); }
            catch (Exception ignored) {}
        } catch (Exception e) {
            System.err.println("Qwen35 CUDA: dp4a kernels unavailable — " + e.getMessage());
        }
        // dp4a enabled by default — disable with -Dcuda.dp4a=false
        useDp4a = dp4aAvail && !"false".equals(System.getProperty("cuda.dp4a", "true"));
        quantizeFunc = qFunc;
        dp4aFunc = dFunc;
        dp4aQ3kFunc = d3kFunc;
        dp4aQ5kFunc = d5kFunc;
        dp4aQ6kFunc = d6kFunc;
        dp4aQ50Func = d50Func;
        dp4aQ80Func = d80Func;
        dp4aIq4nlFunc = dIq4nlFunc;
        dp4aIq4xsFunc = dIq4xsFunc;

        // T2.1: multi-row Q4_K dp4a kernel (opt-in)
        boolean mr4Req = "true".equals(System.getProperty("cuda.q4k.mr4", "false"));
        MemorySegment mr4Func = null;
        if (mr4Req && useDp4a) {
            try {
                mr4Func = cudaContext.compileKernel("kernels/cuda/matmul_q4_k_dp4a_mr4.cu", "matmul_q4_k_dp4a_mr4");
                System.err.println("Qwen35 CUDA: Q4_K multi-row (4 rows/warp) enabled (T2.1)");
            } catch (Exception e) {
                System.err.println("Qwen35 CUDA: Q4_K mr4 unavailable: " + e.getMessage());
            }
        }
        useDp4aQ4kMr4 = (mr4Func != null);
        dp4aQ4kMr4Func = mr4Func;
        if (useDp4a) {
            q8BufferSize = (dim / 32) * 40;
            gpuXbQ8 = bufferManager.createBuffer(q8BufferSize);
            int numQ8Blocks = dim / 32;
            quantizeGridDim = (numQ8Blocks + 7) / 8; // 8 warps per block (256 threads)
            quantizePB = new ParamBuffer(arena, 3);
            quantizePB.setLong(0, gpuXb);
            quantizePB.setLong(1, gpuXbQ8);
            quantizePB.setInt(2, dim);
            dp4aPB = new ParamBuffer(arena, 6);
            String dp4aTypes = "Q4_K";
            if (dp4aQ3kFunc != null) dp4aTypes += "+Q3_K";
            if (dp4aQ5kFunc != null) dp4aTypes += "+Q5_K";
            if (dp4aQ6kFunc != null && DP4A_Q6) dp4aTypes += "+Q6_K";
            System.err.println("Qwen35 CUDA: dp4a enabled (" + dp4aTypes + " × Q8_1)");
        } else {
            q8BufferSize = 0;
            gpuXbQ8 = 0;
            quantizeGridDim = 0;
            quantizePB = null;
            dp4aPB = null;
        }

        // T1.1: Fused rmsnorm+quantize kernel (opt-in, default false to preserve baseline)
        // Replaces: rmsnorm (writes xb) → quantize_q8 (xb→xbQ8) with a single kernel
        // that writes both outputs in one pass. Saves ~5-7% of GPU time when enabled.
        // Requires dp4a path (since the fused kernel writes Q8_1 output).
        boolean fnqRequested = "true".equals(System.getProperty("cuda.fused_norm_quantize", "false"));
        MemorySegment rnqFunc = null;
        if (fnqRequested && useDp4a) {
            try {
                rnqFunc = cudaContext.compileKernel("kernels/cuda/rmsnorm_quantize.cu", "rmsnorm_quantize_fused");
                System.err.println("Qwen35 CUDA: fused rmsnorm+quantize enabled (T1.1)");
            } catch (Exception e) {
                System.err.println("Qwen35 CUDA: fused rmsnorm+quantize unavailable: " + e.getMessage());
            }
        }
        useFusedNormQuantize = (rnqFunc != null);
        rmsnormQuantizeFunc = rnqFunc;
        // T1.2: dp4a for output projections — pre-quantize the input from gpuXb2/gpuDeltaOut/gpuHb
        // into a shared Q8 scratch buffer, then call the dp4a matmul kernel reading from scratch.
        // Trade-off: 1 extra quantize launch per output projection (~46µs each = 3 per layer)
        // vs ~85µs saving on the matmul kernel (dp4a is 2× faster than plain Q4_K matmul).
        // Net: ~-50% per output projection, ~10ms/token saving on Qwen3.5-4B.
        // T1.2: dp4a for output projections (ssmOut in DeltaNet / wo in full-attn / ffnDown in FFN).
        // Measured +9-11% on Qwen3.5 4B/9B with PPL within expected Q8_1 quantization envelope.
        // Flipped from opt-in to default-on in v1.14.0 — opt-out via -Dcuda.dp4a.outputs=false.
        boolean dp4aOutReq = !"false".equals(System.getProperty("cuda.dp4a.outputs", "true"));
        useDp4aOutputs = dp4aOutReq && useDp4a;
        if (useDp4aOutputs) {
            int maxInDim = Math.max(Math.max(qDim, innerSize), ffnDim);
            // Round up to 32-element boundary for Q8_1 blocks
            scratchQ8Size = ((maxInDim + 31) / 32) * 40;
            gpuScratchQ8 = bufferManager.createBuffer(scratchQ8Size);
            scratchQuantizePB = new ParamBuffer(arena, 3);
            // All 3 args set per-call by quantizeBuffer()
            System.err.println("Qwen35 CUDA: dp4a for output projections enabled (T1.2, scratch=" + scratchQ8Size + " bytes)");
        } else {
            gpuScratchQ8 = 0;
            scratchQ8Size = 0;
            scratchQuantizePB = null;
        }

        if (useFusedNormQuantize) {
            rmsnormQuantizePB = new ParamBuffer(arena, 6);
            // Always-the-same slots — set ONCE in constructor:
            //   0: normOut = gpuXb (FP32 normalized output, mirrors legacy rmsnormFusedFunc)
            //   1: qOut    = gpuXbQ8 (Q8_1 quantized output, mirrors legacy quantize_q8)
            //   2: input   = gpuX (input residual stream, mirrors legacy rmsnormFusedFunc)
            //   4: size    = dim
            //   5: eps     = normEps
            // Per-call slot:
            //   3: weights (set by normAndQuantize() per layer)
            rmsnormQuantizePB.setLong(0, gpuXb);
            rmsnormQuantizePB.setLong(1, gpuXbQ8);
            rmsnormQuantizePB.setLong(2, gpuX);
            rmsnormQuantizePB.setInt(4, dim);
            rmsnormQuantizePB.setFloat(5, normEps);
        } else {
            rmsnormQuantizePB = null;
        }

        // Upload per-layer weights
        gpuAttnNormWeights = new long[blockCount];
        gpuFfnNormWeights = new long[blockCount];
        gpuConvWeights = new long[blockCount];
        gpuSsmAWeights = new long[blockCount];
        gpuDtBiasWeights = new long[blockCount];
        gpuSsmNormWeights = new long[blockCount];
        gpuQNormWeights = new long[blockCount];
        gpuKNormWeights = new long[blockCount];

        for (int i = 0; i < gpuLayerCount; i++) {
            Qwen35LayerWeights lw = weights.layers()[i];
            gpuAttnNormWeights[i] = uploadNormWeights(lw.attnNorm(), dim);
            gpuFfnNormWeights[i] = uploadNormWeights(lw.postAttnNorm(), dim);

            if (isDeltaNet[i]) {
                gpuConvWeights[i] = uploadTensorAsFloats(lw.ssmConv1d(), deltaQkvDim * convKernel);
                gpuSsmAWeights[i] = uploadTensorAsFloats(lw.ssmA(), timeStepRank);
                gpuDtBiasWeights[i] = uploadTensorAsFloats(lw.ssmDtBias(), timeStepRank);
                gpuSsmNormWeights[i] = uploadNormWeights(lw.ssmNorm(), stateSize);
            } else {
                if (lw.qNorm() != null) {
                    gpuQNormWeights[i] = uploadNormWeights(lw.qNorm(), headSize);
                    gpuKNormWeights[i] = uploadNormWeights(lw.kNorm(), headSize);
                }
            }
        }

        // Output projection
        gpuOutputNormWeights = uploadNormWeights(weights.outputNorm(), dim);
        FloatTensor outputTensor = weights.output();
        if (outputTensor instanceof CudaFloatTensor) {
            gpuLogits = bufferManager.createBuffer((long) vocabSize * fb);
            gpuLogitsBytes = (long) vocabSize * fb;
            hostLogits = arena.allocate(ValueLayout.JAVA_FLOAT, vocabSize);
            outputMatmul = new MatmulLaunch((CudaFloatTensor) outputTensor, gpuXb, gpuLogits, vocabSize, dim, 0);
        } else {
            gpuLogits = 0;
            gpuLogitsBytes = 0;
            hostLogits = null;
            outputMatmul = null;
        }

        // GPU-side argmax
        MemorySegment argmaxPartialFuncLocal = cudaContext.compileKernel("kernels/cuda/argmax.cu", "argmax_partial");
        MemorySegment argmaxFinalFuncLocal = cudaContext.compileKernel("kernels/cuda/argmax.cu", "argmax_final");
        argmaxNumBlocks = Math.min(256, (vocabSize + 255) / 256);
        gpuArgmaxPartialVal = bufferManager.createBuffer((long) argmaxNumBlocks * fb);
        gpuArgmaxPartialIdx = bufferManager.createBuffer((long) argmaxNumBlocks * Integer.BYTES);
        gpuArgmaxResult = bufferManager.createBuffer(Integer.BYTES);
        hostArgmaxResult = arena.allocate(ValueLayout.JAVA_INT, 1);

        // Fused FFN gate+up (Q4_K only)
        MemorySegment fusedFunc = null;
        try { fusedFunc = cudaContext.compileKernel("kernels/cuda/matmul_q4_k_fused_gate_up.cu", "matmul_q4_k_fused_gate_up"); }
        catch (Exception ignored) {}
        boolean canFuse = fusedFunc != null;
        long[] fGateW = canFuse ? new long[gpuLayerCount] : null;
        long[] fUpW = canFuse ? new long[gpuLayerCount] : null;
        if (canFuse) {
            for (int i = 0; i < gpuLayerCount; i++) {
                Qwen35LayerWeights lw = weights.layers()[i];
                if (lw.ffnGate() instanceof CudaFloatTensor && lw.ffnUp() instanceof CudaFloatTensor) {
                    CudaFloatTensor gate = (CudaFloatTensor) lw.ffnGate();
                    CudaFloatTensor up = (CudaFloatTensor) lw.ffnUp();
                    if (gate.type() == it.denzosoft.llmplayer.tensor.GGMLType.Q4_K
                            && up.type() == it.denzosoft.llmplayer.tensor.GGMLType.Q4_K) {
                        fGateW[i] = gate.getGpuWeights();
                        fUpW[i] = up.getGpuWeights();
                    } else {
                        canFuse = false;
                        break;
                    }
                } else {
                    canFuse = false;
                    break;
                }
            }
        }
        useFusedGateUp = canFuse;
        fusedGateUpFunc = fusedFunc;
        fusedGateWeights = fGateW;
        fusedUpWeights = fUpW;
        if (canFuse) {
            int totalRows = ffnDim * 2;
            int fBlockDim = (int) Math.min(256, maxWg);
            fusedGateUpGridDim = (totalRows + fBlockDim / 32 - 1) / (fBlockDim / 32);
            fusedGateUpBlockDim = fBlockDim;
            System.err.println("Qwen35 CUDA: fused gate+up Q4_K kernel enabled");
        } else {
            fusedGateUpGridDim = 0;
            fusedGateUpBlockDim = 0;
        }

        // === Allocate param buffers ===
        matmulPB = new ParamBuffer(arena, 6);
        normPB = new ParamBuffer(arena, 5);
        normPB.setLong(0, gpuXb);
        normPB.setLong(1, gpuX);
        normPB.setInt(3, dim);
        normPB.setFloat(4, normEps);

        ropePB = new ParamBuffer(arena, 8);
        ropePB.setLong(1, gpuCosTable);
        ropePB.setLong(2, gpuSinTable);
        ropePB.setInt(4, headSize);
        ropePB.setInt(5, halfRope);
        ropePB.setLong(6, gpuTokenParams);
        ropePB.setInt(7, ropeType);

        kvPB = new ParamBuffer(arena, 6);
        kvPB.setLong(2, gpuK);
        kvPB.setLong(3, gpuV);
        kvPB.setInt(4, kvDim);
        kvPB.setLong(5, gpuTokenParams);

        maxBatch = initBatchedPrefill(weights);
        flashAttn = new FlashAttention(cudaContext, bufferManager, arena, headCount, headSize, maxSeqLen,
            useFp16Kv, gpuTokenParams, Math.max(1, maxBatch));

        siluPB = new ParamBuffer(arena, 2);
        siluMulPB = new ParamBuffer(arena, 3);
        siluMulPB.setLong(0, gpuHb);
        siluMulPB.setLong(1, gpuHb2);
        siluMulPB.setInt(2, ffnDim);

        conv1dPB = new ParamBuffer(arena, 6);
        conv1dPB.setLong(0, gpuQkv);
        conv1dPB.setInt(3, deltaQkvDim);
        conv1dPB.setInt(4, convKernel);
        conv1dPB.setLong(5, gpuTokenParams);

        // deltanet_fused: S, qkv, alpha, beta, gate, normW, output, normEps, nHeads, groupCount, dQK, dV
        deltanetPB = new ParamBuffer(arena, 12);
        deltanetPB.setLong(1, gpuQkv);
        deltanetPB.setLong(2, gpuAlpha);
        deltanetPB.setLong(3, gpuBeta);
        deltanetPB.setLong(4, gpuGate);      // raw gate (SiLU applied inside kernel)
        // [5] = gpuSsmNormWeights[layer] — set per launch
        deltanetPB.setLong(6, gpuDeltaOut);   // final output
        deltanetPB.setFloat(7, normEps);
        deltanetPB.setInt(8, timeStepRank);
        deltanetPB.setInt(9, groupCount);
        deltanetPB.setInt(10, stateSize); // dQK
        deltanetPB.setInt(11, stateSize); // dV

        alphaBetaPB = new ParamBuffer(arena, 5);
        alphaBetaPB.setLong(0, gpuAlpha);
        alphaBetaPB.setLong(1, gpuBeta);
        alphaBetaPB.setInt(4, timeStepRank);

        deinterleavePB = new ParamBuffer(arena, 5);
        deinterleavePB.setLong(0, gpuQ);
        deinterleavePB.setLong(1, gpuQCompact);
        deinterleavePB.setLong(2, gpuAttnGate);
        deinterleavePB.setInt(3, headCount);
        deinterleavePB.setInt(4, headSize);

        sigmoidMulPB = new ParamBuffer(arena, 3);
        sigmoidMulPB.setLong(0, gpuXb2);
        sigmoidMulPB.setLong(1, gpuAttnGate);
        sigmoidMulPB.setInt(2, qDim);

        perHeadNormPB = new ParamBuffer(arena, 4);
        perHeadNormPB.setInt(2, headSize);
        perHeadNormPB.setFloat(3, normEps);

        accPB = new ParamBuffer(arena, 3);
        accPB.setInt(2, dim);

        elemMulPB = new ParamBuffer(arena, 3);

        // Argmax param buffers
        ParamBuffer argmaxPartialPB = new ParamBuffer(arena, 4);
        argmaxPartialPB.setLong(1, gpuArgmaxPartialVal);
        argmaxPartialPB.setLong(2, gpuArgmaxPartialIdx);
        argmaxPartialPB.setInt(3, vocabSize);
        this.argmaxPartialPBuf = argmaxPartialPB;
        this.argmaxPartialFunc = argmaxPartialFuncLocal;

        ParamBuffer argmaxFinalPB = new ParamBuffer(arena, 4);
        argmaxFinalPB.setLong(0, gpuArgmaxPartialVal);
        argmaxFinalPB.setLong(1, gpuArgmaxPartialIdx);
        argmaxFinalPB.setLong(2, gpuArgmaxResult);
        argmaxFinalPB.setInt(3, argmaxNumBlocks);
        this.argmaxFinalPBuf = argmaxFinalPB;
        this.argmaxFinalFunc = argmaxFinalFuncLocal;

        // Fused gate+up param buffer (if applicable)
        if (useFusedGateUp) {
            fusedGateUpPB = new ParamBuffer(arena, 8);
            fusedGateUpPB.setLong(2, gpuXb);     // input
            fusedGateUpPB.setLong(3, gpuHb);     // gateOutput
            fusedGateUpPB.setLong(4, gpuHb2);    // upOutput
            fusedGateUpPB.setInt(5, ffnDim);     // gateRows
            fusedGateUpPB.setInt(6, dim);        // cols
            fusedGateUpPB.setInt(7, 0);          // addToOutput (write)
        } else {
            fusedGateUpPB = null;
        }

        // Pre-compute grid sizes
        int numWg = (int) ((dim + blockSize - 1) / blockSize);
        this.normNumWarps = (int) (blockSize / 32);
        this.normSharedMem = (normNumWarps + 1) * Float.BYTES;
        this.ropeQGridDim = (int) ((headCount * halfRope + blockSize - 1) / blockSize);
        this.ropeKGridDim = (int) ((headCountKV * halfRope + blockSize - 1) / blockSize);
        this.kvUpdateGridDim = (int) ((kvDim + blockSize - 1) / blockSize);
        this.siluQkvGridDim = (int) ((deltaQkvDim + blockSize - 1) / blockSize);
        this.siluGateGridDim = (int) ((innerSize + blockSize - 1) / blockSize);
        this.siluFFNGridDim = (int) ((ffnDim + blockSize - 1) / blockSize);
        this.conv1dGridDim = (int) ((deltaQkvDim + blockSize - 1) / blockSize);
        this.deinterleaveGridDim = (int) ((qDim + blockSize - 1) / blockSize);
        this.elemMulGridDim = (int) ((innerSize + blockSize - 1) / blockSize);
        this.accDimGridDim = numWg;
        this.perHeadNormBlockDim = (int) Math.min(Math.max(32, ((headSize + 31) / 32) * 32), blockSize);
        this.perHeadNormSharedMem = ((perHeadNormBlockDim / 32) + 1) * Float.BYTES;
        // DeltaNet fused shared mem: 2*dQK (Q,K) + numWarps*3 (reduce_q, reduce_k, reduce_norm)
        int numWarps = stateSize / 32; // 256/32 = 8
        this.deltanetSharedMem = (2 * stateSize + numWarps * 3) * Float.BYTES;

        // Build per-layer matmul descriptors
        layerMatmuls = new MatmulLaunch[gpuLayerCount][8];
        for (int i = 0; i < gpuLayerCount; i++) {
            Qwen35LayerWeights lw = weights.layers()[i];
            if (isDeltaNet[i]) {
                layerMatmuls[i][0] = new MatmulLaunch((CudaFloatTensor) lw.attnQkv(), gpuXb, gpuQkv, deltaQkvDim, dim, 0);
                layerMatmuls[i][1] = new MatmulLaunch((CudaFloatTensor) lw.ssmAlpha(), gpuXb, gpuAlpha, timeStepRank, dim, 0);
                layerMatmuls[i][2] = new MatmulLaunch((CudaFloatTensor) lw.ssmBeta(), gpuXb, gpuBeta, timeStepRank, dim, 0);
                layerMatmuls[i][3] = new MatmulLaunch((CudaFloatTensor) lw.attnGate(), gpuXb, gpuGate, innerSize, dim, 0);
                layerMatmuls[i][4] = new MatmulLaunch((CudaFloatTensor) lw.ssmOut(), gpuDeltaOut, gpuX, dim, innerSize, 1); // accumulate
            } else {
                layerMatmuls[i][0] = new MatmulLaunch((CudaFloatTensor) lw.wq(), gpuXb, gpuQ, qGateDim, dim, 0);
                layerMatmuls[i][1] = new MatmulLaunch((CudaFloatTensor) lw.wk(), gpuXb, gpuK, kvDim, dim, 0);
                layerMatmuls[i][2] = new MatmulLaunch((CudaFloatTensor) lw.wv(), gpuXb, gpuV, kvDim, dim, 0);
                layerMatmuls[i][3] = new MatmulLaunch((CudaFloatTensor) lw.wo(), gpuXb2, gpuX, dim, qDim, 1); // accumulate
            }
            // FFN (shared by both layer types; MoE layers go through forwardMoeFFN)
            if (!moeLayer[i]) {
                layerMatmuls[i][5] = new MatmulLaunch((CudaFloatTensor) lw.ffnGate(), gpuXb, gpuHb, ffnDim, dim, 0);
                layerMatmuls[i][6] = new MatmulLaunch((CudaFloatTensor) lw.ffnUp(), gpuXb, gpuHb2, ffnDim, dim, 0);
                layerMatmuls[i][7] = new MatmulLaunch((CudaFloatTensor) lw.ffnDown(), gpuHb, gpuX, dim, ffnDim, 1); // accumulate
            }
        }

        // CUDA graph availability
        this.graphAvailable = !hasMoe // the routed experts are a host callback inside every layer
                && !Boolean.getBoolean("cuda.nograph")
                && cudaContext.isGraphApiAvailable()
                && outputMatmul != null
                && flashAttn.graphCompatible();

        // Qwen3.5-MoE: no whole-token graph (the routed experts are a host callback inside every
        // layer), but each layer's device work — up to the FFN norm and the shared expert — replays
        // as its own graph from the layer's second call (F10); the callback stays outside.
        this.layerGraphExec = new MemorySegment[blockCount];
        this.layerRan = new boolean[blockCount];
        this.layerGraphsBroken = !hasMoe || !MoeAttentionCudaPass.LAYER_GRAPHS
            || !cudaContext.isGraphApiAvailable() || !flashAttn.graphCompatible();

        long totalVram = 0;
        for (int i = 0; i < gpuLayerCount; i++) {
            if (isDeltaNet[i]) {
                totalVram += (long) timeStepRank * stateSize * stateSize * fb; // S state
                totalVram += (long) (convKernel - 1) * deltaQkvDim * fb;       // conv state
            } else {
                totalVram += 2L * maxSeqLen * kvDim * fb; // KV cache
            }
        }
        // Count dp4a-eligible matmuls
        int dp4aCount = 0, totalMatmuls = 0;
        if (useDp4a) {
            for (int i = 0; i < gpuLayerCount; i++) {
                for (int j = 0; j < 8; j++) {
                    if (layerMatmuls[i][j] != null) {
                        totalMatmuls++;
                        if (layerMatmuls[i][j].dp4aType > 0) dp4aCount++;
                    }
                }
            }
        }

        System.err.println("Qwen35 CUDA: " + gpuLayerCount + "/" + blockCount + " layers on GPU"
                + " (state: " + (totalVram / 1024 / 1024) + " MB"
                + ", graph: " + (graphAvailable ? "available" : !layerGraphsBroken ? "per layer" : "unavailable")
                + (useDp4a ? ", dp4a: " + dp4aCount + "/" + totalMatmuls + " matmuls" : "") + ")");
    }

    // === Public API ===

    public static boolean isSupported(ModelConfig config, Qwen35Weights weights) {
        if (config.fullAttentionInterval() <= 0) return false;
        if (weights.layers().length == 0) return false;

        // Check both DeltaNet and attention layers have CUDA tensors
        for (int i = 0; i < Math.min(config.fullAttentionInterval(), weights.layers().length); i++) {
            Qwen35LayerWeights lw = weights.layers()[i];
            if (lw.isDeltaNet()) {
                if (!(lw.attnQkv() instanceof CudaFloatTensor)) return false;
                if (!(lw.ssmAlpha() instanceof CudaFloatTensor)) return false;
                if (!(lw.ssmOut() instanceof CudaFloatTensor)) return false;
                if (!lw.isMoe() && !(lw.ffnGate() instanceof CudaFloatTensor)) return false;
            } else {
                if (!(lw.wq() instanceof CudaFloatTensor)) return false;
                if (!(lw.wk() instanceof CudaFloatTensor)) return false;
                if (!(lw.wo() instanceof CudaFloatTensor)) return false;
                if (!lw.isMoe() && !(lw.ffnGate() instanceof CudaFloatTensor)) return false;
            }
        }
        return true;
    }

    public static int countGpuLayers(Qwen35Weights weights) {
        int count = 0;
        for (Qwen35LayerWeights lw : weights.layers()) {
            boolean onGpu = lw.isDeltaNet()
                    ? (lw.attnQkv() instanceof CudaFloatTensor)
                    : (lw.wq() instanceof CudaFloatTensor);
            if (onGpu) count++;
            else break;
        }
        return count;
    }

    public int getGpuLayerCount() {
        return gpuLayerCount;
    }

    /**
     * A token at position 0 starts a new sequence: zero the recurrent (SSM) states of the layers.
     * The kernels carry them from token to token and never read the position (the short conv does,
     * so its history needs no reset), so without this a second generation in the same process (the
     * web server, interactive mode, the placement calibrator) started from the previous sequence's
     * state.
     */
    private void resetRecurrentState() {
        long stateBytes = (long) timeStepRank * stateSize * stateSize * Float.BYTES;
        for (int i = 0; i < gpuLayerCount; i++) {
            if (gpuSsmState[i] != 0) cudaContext.fillBufferZero(gpuSsmState[i], stateBytes);
        }
    }

    public void uploadXAndUpdateParams(float[] x, int position) {
        if (position == 0) resetRecurrentState();
        long embBytes = (long) dim * Float.BYTES;
        MemorySegment.copy(x, 0, hostCombined, ValueLayout.JAVA_FLOAT, 0, dim);
        hostCombined.set(ValueLayout.JAVA_INT, embBytes, position);
        hostCombined.set(ValueLayout.JAVA_INT, embBytes + 4, position + 1);
        cudaContext.writeBuffer(gpuCombined, hostCombined, embBytes + 8);
    }

    public void downloadX(float[] x) {
        cudaContext.readBuffer(gpuX, hostX, (long) dim * Float.BYTES);
        MemorySegment.copy(hostX, ValueLayout.JAVA_FLOAT, 0, x, 0, dim);
    }

    /** Execute one layer on GPU. Dispatches to DeltaNet or attention based on layer type. */
    public void forwardLayer(int layerIdx, int position) {
        if (PROFILING) {
            forwardLayerProfiled(layerIdx, position);
            return;
        }
        if (!layerGraphsBroken) {
            MemorySegment g = layerGraphExec[layerIdx];
            if (g == null && layerRan[layerIdx]) g = captureLayer(layerIdx, position);
            if (g != null) {
                cudaContext.launchGraph(g);
            } else {
                deviceLayer(layerIdx, position);
                layerRan[layerIdx] = true;
            }
            if (moeLayer[layerIdx]) moeHost(layerIdx);
            return;
        }
        if (isDeltaNet[layerIdx]) {
            forwardDeltaNetLayer(layerIdx);
        } else {
            forwardAttentionLayer(layerIdx, position);
        }
    }

    /** The device work of a layer: everything but the routed-expert host callback of a MoE layer. */
    private void deviceLayer(int layerIdx, int position) {
        moeDeviceOnly = true;
        try {
            if (isDeltaNet[layerIdx]) forwardDeltaNetLayer(layerIdx);
            else forwardAttentionLayer(layerIdx, position);
        } finally {
            moeDeviceOnly = false;
        }
    }

    private MemorySegment captureLayer(int layerIdx, int position) {
        try {
            cudaContext.beginCapture();
            try {
                deviceLayer(layerIdx, position);
            } catch (RuntimeException e) {
                try { cudaContext.endCapture(); } catch (RuntimeException ignored) { }
                throw e;
            }
            MemorySegment graph = cudaContext.endCapture();
            MemorySegment exec = cudaContext.instantiateGraph(graph);
            cudaContext.destroyGraph(graph);
            layerGraphExec[layerIdx] = exec;
            return exec;
        } catch (RuntimeException e) {
            layerGraphsBroken = true;
            System.err.println("Qwen35 CUDA: per-layer graph capture failed (" + e.getMessage() + ") — running per launch");
            return null;
        }
    }

    // ---- -Dcuda.profile=true support (coarse per-layer-type timing) ----

    private static final boolean PROFILING = Boolean.getBoolean("cuda.profile");
    private long profDeltaNs, profAttnNs, profTotalNs;
    private int profTokens;

    /** Profiled variant of {@link #forwardLayer}. Adds a finish() barrier per layer to attribute
     *  time; prints a summary every 10 tokens.  The finish() calls add 15-25 % overhead, so
     *  profiling is opt-in via the {@code -Dcuda.profile=true} system property. */
    private void forwardLayerProfiled(int layerIdx, int position) {
        if (layerIdx == 0) {
            profTokens++;
            if (profTokens > 1 && profTokens % 10 == 0) printProfile();
        }
        long t0 = System.nanoTime();
        if (isDeltaNet[layerIdx]) {
            forwardDeltaNetLayer(layerIdx);
            cudaContext.finish();
            long dt = System.nanoTime() - t0;
            profDeltaNs += dt;
            profTotalNs += dt;
        } else {
            forwardAttentionLayer(layerIdx, position);
            cudaContext.finish();
            long dt = System.nanoTime() - t0;
            profAttnNs += dt;
            profTotalNs += dt;
        }
    }

    /** Print the rolling profile summary and reset counters. */
    public void printProfile() {
        if (profTokens == 0) return;
        double n = profTokens;
        System.err.printf("Qwen35 CUDA profile (%d tokens, %d GPU layers):%n", profTokens, gpuLayerCount);
        System.err.printf("  deltanet (per token total): %6.2f ms  |  full-attn (per token total): %6.2f ms  |  TOTAL: %6.2f ms/tok%n",
            profDeltaNs / 1e6 / n, profAttnNs / 1e6 / n, profTotalNs / 1e6 / n);
        profDeltaNs = profAttnNs = profTotalNs = 0;
        profTokens = 0;
    }

    /** Execute all GPU layers + output projection via CUDA graph. */
    public boolean forwardGraph(float[] logits) {
        if (!graphAvailable) return false;

        if (graphExec == null) {
            boolean capturing = false;
            try {
                cudaContext.beginCapture();
                capturing = true;
                for (int layer = 0; layer < gpuLayerCount; layer++) {
                    forwardLayerKernels(layer);
                }
                // Final norm + output
                normPB.setLong(2, gpuOutputNormWeights);
                launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);
                launchMatmul(outputMatmul);

                MemorySegment graph = cudaContext.endCapture();
                capturing = false;
                graphExec = cudaContext.instantiateGraph(graph);
                cudaContext.destroyGraph(graph);
                System.err.println("Qwen35 CUDA graph: captured " + gpuLayerCount + " layers");
            } catch (Exception e) {
                if (capturing) {
                    try { MemorySegment partial = cudaContext.endCapture(); if (partial != null && partial.address() != 0) cudaContext.destroyGraph(partial); } catch (Exception ignored) {}
                }
                System.err.println("Qwen35 CUDA graph: capture failed — " + e.getMessage());
                graphAvailable = false; // a failed capture would fail again: stay on per-layer mode
                graphExec = null;
                return false;
            }
        }

        cudaContext.launchGraph(graphExec);
        cudaContext.readBuffer(gpuLogits, hostLogits, gpuLogitsBytes);
        MemorySegment.copy(hostLogits, ValueLayout.JAVA_FLOAT, 0, logits, 0, vocabSize);
        return true;
    }

    /**
     * Prefill graph: all layers without output projection.
     * Saves ~5ms per prefill token by skipping the vocab-size output matmul.
     * Returns true if graph ran successfully.
     */
    public boolean forwardGraphPrefill() {
        if (!graphAvailable) return false;

        if (prefillGraphExec == null) {
            boolean capturing = false;
            try {
                cudaContext.beginCapture();
                capturing = true;
                for (int layer = 0; layer < gpuLayerCount; layer++) {
                    forwardLayerKernels(layer);
                }
                // NO output norm/matmul — just layers
                MemorySegment graph = cudaContext.endCapture();
                capturing = false;
                prefillGraphExec = cudaContext.instantiateGraph(graph);
                cudaContext.destroyGraph(graph);
                System.err.println("Qwen35 CUDA prefill graph: captured " + gpuLayerCount + " layers (no output)");
            } catch (Exception e) {
                if (capturing) {
                    try { MemorySegment partial = cudaContext.endCapture(); if (partial != null && partial.address() != 0) cudaContext.destroyGraph(partial); } catch (Exception ignored) {}
                }
                prefillGraphExec = null;
                return false;
            }
        }

        cudaContext.launchGraph(prefillGraphExec);
        return true;
    }

    /** Final RMSNorm + output matmul, download logits. */
    public boolean forwardFinalLogits(float[] logits) {
        if (outputMatmul == null) return false;
        normPB.setLong(2, gpuOutputNormWeights);
        launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);
        launchMatmul(outputMatmul);
        cudaContext.readBuffer(gpuLogits, hostLogits, gpuLogitsBytes);
        MemorySegment.copy(hostLogits, ValueLayout.JAVA_FLOAT, 0, logits, 0, vocabSize);
        return true;
    }

    /** GPU-side argmax: norm + output matmul + argmax, download only 4 bytes. Returns -1 if unavailable. */
    public int forwardFinalArgmax() {
        if (outputMatmul == null) return -1;
        normPB.setLong(2, gpuOutputNormWeights);
        launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);
        launchMatmul(outputMatmul);
        argmaxPartialPBuf.setLong(0, gpuLogits);
        launchKernel(argmaxPartialFunc, argmaxNumBlocks, 256, 0, argmaxPartialPBuf.ptrs);
        launchKernel(argmaxFinalFunc, 1, 256, 0, argmaxFinalPBuf.ptrs);
        cudaContext.readBuffer(gpuArgmaxResult, hostArgmaxResult, Integer.BYTES);
        return hostArgmaxResult.get(ValueLayout.JAVA_INT, 0);
    }

    /** CUDA graph + GPU argmax. Returns -1 to fall back. */
    public int forwardGraphArgmax() {
        if (!graphAvailable) return -1;
        if (graphExec == null) {
            // Capture graph (same as forwardGraph)
            boolean capturing = false;
            try {
                cudaContext.beginCapture();
                capturing = true;
                for (int layer = 0; layer < gpuLayerCount; layer++) {
                    forwardLayerKernels(layer);
                }
                normPB.setLong(2, gpuOutputNormWeights);
                launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);
                launchMatmul(outputMatmul);
                MemorySegment graph = cudaContext.endCapture();
                capturing = false;
                graphExec = cudaContext.instantiateGraph(graph);
                cudaContext.destroyGraph(graph);
                System.err.println("Qwen35 CUDA graph: captured " + gpuLayerCount + " layers (argmax)");
            } catch (Exception e) {
                if (capturing) { try { MemorySegment partial = cudaContext.endCapture(); if (partial != null && partial.address() != 0) cudaContext.destroyGraph(partial); } catch (Exception ignored) {} }
                System.err.println("Qwen35 CUDA graph: capture failed — " + e.getMessage());
                graphAvailable = false; // a failed capture would fail again: stay on per-layer mode
                graphExec = null;
                return -1;
            }
        }
        cudaContext.launchGraph(graphExec);
        // No finish() needed — same stream ordering guarantees graph completes before argmax
        argmaxPartialPBuf.setLong(0, gpuLogits);
        launchKernel(argmaxPartialFunc, argmaxNumBlocks, 256, 0, argmaxPartialPBuf.ptrs);
        launchKernel(argmaxFinalFunc, 1, 256, 0, argmaxFinalPBuf.ptrs);
        cudaContext.readBuffer(gpuArgmaxResult, hostArgmaxResult, Integer.BYTES);
        return hostArgmaxResult.get(ValueLayout.JAVA_INT, 0);
    }

    // === Batched prefill ===

    /** Allocate the chunk buffers and the GEMM helper; returns the chunk size, 0 when unavailable. */
    private int initBatchedPrefill(Qwen35Weights weights) {
        int nb = PREFILL_BATCH;
        if (nb <= 1 || gpuLayerCount != blockCount || !GemmF16.available()
                || "false".equals(System.getProperty("prefill.gpu.batched", "true"))) {
            return 0;
        }
        java.util.List<CudaFloatTensor> ws = new java.util.ArrayList<>();
        for (int i = 0; i < blockCount; i++) {
            Qwen35LayerWeights lw = weights.layers()[i];
            FloatTensor[] ts = isDeltaNet[i]
                ? new FloatTensor[] { lw.attnQkv(), lw.ssmAlpha(), lw.ssmBeta(), lw.attnGate(), lw.ssmOut() }
                : new FloatTensor[] { lw.wq(), lw.wk(), lw.wv(), lw.wo() };
            for (FloatTensor t : ts) {
                if (!(t instanceof CudaFloatTensor)) return 0;
                ws.add((CudaFloatTensor) t);
            }
            if (!moeLayer[i]) {
                for (FloatTensor t : new FloatTensor[] { lw.ffnGate(), lw.ffnUp(), lw.ffnDown() }) {
                    if (!(t instanceof CudaFloatTensor)) return 0;
                    ws.add((CudaFloatTensor) t);
                }
            }
        }
        long f = Float.BYTES;
        try {
            int maxCols = Math.max(Math.max(dim, innerSize), Math.max(qDim, ffnDim));
            gemm = new GemmF16(cudaContext, bufferManager, arena, nb, maxCols, 16L << 20, ws);
            bX = bufferManager.createBuffer((long) nb * dim * f);
            bXn = bufferManager.createBuffer((long) nb * dim * f);
            bQkv = bufferManager.createBuffer((long) nb * deltaQkvDim * f);
            bAlpha = bufferManager.createBuffer((long) nb * timeStepRank * f);
            bBeta = bufferManager.createBuffer((long) nb * timeStepRank * f);
            bGate = bufferManager.createBuffer((long) nb * innerSize * f);
            bOut = bufferManager.createBuffer((long) nb * innerSize * f);
            bQ = bufferManager.createBuffer((long) nb * qGateDim * f);
            bQc = bufferManager.createBuffer((long) nb * qDim * f);
            bAg = bufferManager.createBuffer((long) nb * qDim * f);
            bK = bufferManager.createBuffer((long) nb * kvDim * f);
            bV = bufferManager.createBuffer((long) nb * kvDim * f);
            bAtt = bufferManager.createBuffer((long) nb * qDim * f);
            boolean anyDense = false;
            for (int i = 0; i < blockCount; i++) anyDense |= !moeLayer[i];
            if (anyDense) {
                bHb = bufferManager.createBuffer((long) nb * ffnDim * f);
                bHb2 = bufferManager.createBuffer((long) nb * ffnDim * f);
            }
            if (hasMoe) {
                bRouted = bufferManager.createBuffer((long) nb * dim * f);
                bMoeIn = new float[nb][dim];
                bMoeOut = new float[nb][dim];
            }
            bTP = bufferManager.createBuffer((long) nb * 8);
            bHost = cudaContext.allocPinnedHost(2L * nb * dim * f + (long) nb * 8);
            String bo = "kernels/cuda/batch_ops.cu";
            rmsBatchFunc = cudaContext.compileKernel(bo, "rmsnorm_batch");
            ropeBatchFunc = cudaContext.compileKernel(bo, "rope_apply_batch");
            kvBatchFunc = cudaContext.compileKernel(bo, useFp16Kv ? "kv_cache_update_batch_f16" : "kv_cache_update_batch");
            bNormPB = new it.denzosoft.llmplayer.gpu.KernelParams(arena, 5);
            bRopePB = new it.denzosoft.llmplayer.gpu.KernelParams(arena, 9);
            bKvPB = new it.denzosoft.llmplayer.gpu.KernelParams(arena, 6);
            return nb;
        } catch (RuntimeException e) {
            System.err.println("Qwen35 CUDA: batched prefill unavailable — " + e.getMessage());
            if (gemm != null) { gemm.close(); gemm = null; }
            return 0;
        }
    }

    @Override
    public int maxBatchTokens() { return maxBatch; }

    @Override
    public void setMoeFfnBatch(LayerGpuForwardPass.MoeFfnBatch ffn) { this.moeFfnBatch = ffn; }

    @Override
    public void prefillBatch(float[][] x, int basePos, int n) {
        if (n < 1 || n > maxBatch) throw new IllegalArgumentException("batch " + n + " > " + maxBatch);
        if (basePos == 0) resetRecurrentState();
        long f = Float.BYTES;
        long xBytes = (long) n * dim * f;
        long tpOff = 2L * maxBatch * dim * f;
        for (int t = 0; t < n; t++) {
            MemorySegment.copy(x[t], 0, bHost, ValueLayout.JAVA_FLOAT, (long) t * dim * f, dim);
            bHost.set(ValueLayout.JAVA_INT, tpOff + t * 8L, basePos + t);
            bHost.set(ValueLayout.JAVA_INT, tpOff + t * 8L + 4, basePos + t + 1);
        }
        cudaContext.writeBufferAsync(bX, bHost, xBytes);
        cudaContext.writeBufferAsync(bTP, bHost.asSlice(tpOff, n * 8L), n * 8L);
        for (int layer = 0; layer < blockCount; layer++) {
            Qwen35LayerWeights lw = weights.layers()[layer];
            rmsnormB(bXn, bX, gpuAttnNormWeights[layer], n);
            gemm.toF16(bXn, n * dim);
            if (isDeltaNet[layer]) deltaNetBatch(layer, lw, n);
            else attentionBatch(layer, lw, n);
            ffnBatch(layer, lw, n);
        }
        // The last token's hidden state where forwardFinalLogits and the next decode step read it
        cudaContext.copyBufferDtoD(gpuX, bX + (long) (n - 1) * dim * f, (long) dim * f);
        cudaContext.finish();
    }

    private void deltaNetBatch(int layer, Qwen35LayerWeights lw, int n) {
        long f = Float.BYTES;
        gemm.gemm((CudaFloatTensor) lw.attnQkv(), deltaQkvDim, dim, bQkv, deltaQkvDim, n, false);
        gemm.gemm((CudaFloatTensor) lw.ssmAlpha(), timeStepRank, dim, bAlpha, timeStepRank, n, false);
        gemm.gemm((CudaFloatTensor) lw.ssmBeta(), timeStepRank, dim, bBeta, timeStepRank, n, false);
        gemm.gemm((CudaFloatTensor) lw.attnGate(), innerSize, dim, bGate, innerSize, n, false);
        conv1dPB.setLong(1, gpuConvState[layer]);
        conv1dPB.setLong(2, gpuConvWeights[layer]);
        alphaBetaPB.setLong(2, gpuSsmAWeights[layer]);
        alphaBetaPB.setLong(3, gpuDtBiasWeights[layer]);
        deltanetPB.setLong(0, gpuSsmState[layer]);
        deltanetPB.setLong(5, gpuSsmNormWeights[layer]);
        try {
            for (int t = 0; t < n; t++) {
                long qkv = bQkv + (long) t * deltaQkvDim * f;
                long al = bAlpha + (long) t * timeStepRank * f, be = bBeta + (long) t * timeStepRank * f;
                conv1dPB.setLong(0, qkv);
                conv1dPB.setLong(5, bTP + t * 8L);
                launchKernel(conv1dFunc, conv1dGridDim, (int) blockSize, 0, conv1dPB.ptrs);
                alphaBetaPB.setLong(0, al);
                alphaBetaPB.setLong(1, be);
                launchKernel(alphaBetaFunc, 1, (int) Math.max(32, timeStepRank), 0, alphaBetaPB.ptrs);
                deltanetPB.setLong(1, qkv);
                deltanetPB.setLong(2, al);
                deltanetPB.setLong(3, be);
                deltanetPB.setLong(4, bGate + (long) t * innerSize * f);
                deltanetPB.setLong(6, bOut + (long) t * innerSize * f);
                launchKernel(deltanetFunc, timeStepRank, stateSize, deltanetSharedMem, deltanetPB.ptrs);
            }
        } finally {
            // The per-token decode path reads these parameter blocks as set in the constructor
            conv1dPB.setLong(0, gpuQkv);
            conv1dPB.setLong(5, gpuTokenParams);
            alphaBetaPB.setLong(0, gpuAlpha);
            alphaBetaPB.setLong(1, gpuBeta);
            deltanetPB.setLong(1, gpuQkv);
            deltanetPB.setLong(2, gpuAlpha);
            deltanetPB.setLong(3, gpuBeta);
            deltanetPB.setLong(4, gpuGate);
            deltanetPB.setLong(6, gpuDeltaOut);
        }
        gemm.toF16(bOut, n * innerSize);
        gemm.gemm((CudaFloatTensor) lw.ssmOut(), dim, innerSize, bX, dim, n, true);
    }

    private void attentionBatch(int layer, Qwen35LayerWeights lw, int n) {
        gemm.gemm((CudaFloatTensor) lw.wq(), qGateDim, dim, bQ, qGateDim, n, false);
        gemm.gemm((CudaFloatTensor) lw.wk(), kvDim, dim, bK, kvDim, n, false);
        gemm.gemm((CudaFloatTensor) lw.wv(), kvDim, dim, bV, kvDim, n, false);
        int bs = (int) blockSize;
        try {
            // Packed rows are [q | gate] per head: the chunk's n tokens are n * headCount heads
            deinterleavePB.setLong(0, bQ);
            deinterleavePB.setLong(1, bQc);
            deinterleavePB.setLong(2, bAg);
            deinterleavePB.setInt(3, n * headCount);
            launchKernel(deinterleaveFunc, (n * qDim + bs - 1) / bs, bs, 0, deinterleavePB.ptrs);
            if (gpuQNormWeights[layer] != 0) {
                perHeadNormPB.setLong(0, bQc);
                perHeadNormPB.setLong(1, gpuQNormWeights[layer]);
                launchKernel(perHeadNormFunc, n * headCount, perHeadNormBlockDim, perHeadNormSharedMem, perHeadNormPB.ptrs);
                perHeadNormPB.setLong(0, bK);
                perHeadNormPB.setLong(1, gpuKNormWeights[layer]);
                launchKernel(perHeadNormFunc, n * headCountKV, perHeadNormBlockDim, perHeadNormSharedMem, perHeadNormPB.ptrs);
            }
            ropeB(bQc, headCount, qDim, n);
            ropeB(bK, headCountKV, kvDim, n);
            bKvPB.setLong(0, gpuKeyCache[layer]).setLong(1, gpuValueCache[layer]).setLong(2, bK).setLong(3, bV)
                 .setInt(4, kvDim).setLong(5, bTP);
            launch2(kvBatchFunc, (kvDim + bs - 1) / bs, n, bs, 0, bKvPB);
            flashAttn.launchBatch(defaultStream, bAtt, bQc, gpuKeyCache[layer], gpuValueCache[layer], headCountKV,
                headSize, kvDim, 0, (float) (1.0 / Math.sqrt(headSize)), 0f, n, bTP, qDim);
            sigmoidMulPB.setLong(0, bAtt);
            sigmoidMulPB.setLong(1, bAg);
            sigmoidMulPB.setInt(2, n * qDim);
            launchKernel(sigmoidMulFunc, (n * qDim + bs - 1) / bs, bs, 0, sigmoidMulPB.ptrs);
        } finally {
            deinterleavePB.setLong(0, gpuQ);
            deinterleavePB.setLong(1, gpuQCompact);
            deinterleavePB.setLong(2, gpuAttnGate);
            deinterleavePB.setInt(3, headCount);
            sigmoidMulPB.setLong(0, gpuXb2);
            sigmoidMulPB.setLong(1, gpuAttnGate);
            sigmoidMulPB.setInt(2, qDim);
        }
        gemm.toF16(bAtt, n * qDim);
        gemm.gemm((CudaFloatTensor) lw.wo(), dim, qDim, bX, dim, n, true);
    }

    private void ffnBatch(int layer, Qwen35LayerWeights lw, int n) {
        long f = Float.BYTES;
        rmsnormB(bXn, bX, gpuFfnNormWeights[layer], n);
        int bs = (int) blockSize;
        if (moeLayer[layer]) {
            // Routed experts (and the shared expert) on the CPU for the whole chunk: one sync
            long xBytes = (long) n * dim * f;
            cudaContext.readBufferAsync(bXn, bHost, xBytes);
            cudaContext.finish();
            for (int t = 0; t < n; t++) MemorySegment.copy(bHost, ValueLayout.JAVA_FLOAT, (long) t * dim * f, bMoeIn[t], 0, dim);
            if (moeFfnBatch != null) {
                moeFfnBatch.routedBatch(layer, bMoeIn, n, bMoeOut);
            } else {
                for (int t = 0; t < n; t++) moeFfn.routed(layer, bMoeIn[t], bMoeOut[t], true);
            }
            for (int t = 0; t < n; t++) MemorySegment.copy(bMoeOut[t], 0, bHost, ValueLayout.JAVA_FLOAT, (long) t * dim * f, dim);
            cudaContext.writeBufferAsync(bRouted, bHost, xBytes);
            try {
                accPB.setLong(0, bX);
                accPB.setLong(1, bRouted);
                accPB.setInt(2, n * dim);
                launchKernel(accumulateFunc, (n * dim + bs - 1) / bs, bs, 0, accPB.ptrs);
            } finally {
                accPB.setInt(2, dim);
            }
            return;
        }
        gemm.toF16(bXn, n * dim);
        gemm.gemm((CudaFloatTensor) lw.ffnGate(), ffnDim, dim, bHb, ffnDim, n, false);
        gemm.gemm((CudaFloatTensor) lw.ffnUp(), ffnDim, dim, bHb2, ffnDim, n, false);
        try {
            siluMulPB.setLong(0, bHb);
            siluMulPB.setLong(1, bHb2);
            siluMulPB.setInt(2, n * ffnDim);
            launchKernel(siluMulFunc, (n * ffnDim + bs - 1) / bs, bs, 0, siluMulPB.ptrs);
        } finally {
            siluMulPB.setLong(0, gpuHb);
            siluMulPB.setLong(1, gpuHb2);
            siluMulPB.setInt(2, ffnDim);
        }
        gemm.toF16(bHb, n * ffnDim);
        gemm.gemm((CudaFloatTensor) lw.ffnDown(), dim, ffnDim, bX, dim, n, true);
    }

    private void rmsnormB(long out, long in, long w, int n) {
        bNormPB.setLong(0, out).setLong(1, in).setLong(2, w).setInt(3, dim).setFloat(4, normEps);
        launch2(rmsBatchFunc, n, 1, (int) blockSize, normSharedMem, bNormPB);
    }

    private void ropeB(long vec, int heads, int stride, int n) {
        bRopePB.setLong(0, vec).setLong(1, gpuCosTable).setLong(2, gpuSinTable).setInt(3, heads).setInt(4, headSize)
               .setInt(5, halfRope).setLong(6, bTP).setInt(7, ropeType).setInt(8, stride);
        launch2(ropeBatchFunc, (int) ((heads * halfRope + blockSize - 1) / blockSize), n, (int) blockSize, 0, bRopePB);
    }

    private void launch2(MemorySegment fn, int gx, int gy, int block, int shared, it.denzosoft.llmplayer.gpu.KernelParams p) {
        int err = CudaBindings.launchKernel(fn, gx, gy, 1, block, 1, 1, shared, defaultStream, p.ptrs(), MemorySegment.NULL);
        if (err != CudaBindings.CUDA_SUCCESS) throw new RuntimeException("Qwen35 batched prefill launch error: " + err);
    }

    // === DeltaNet Layer ===

    private void forwardDeltaNetLayer(int layerIdx) {
        MatmulLaunch[] ml = layerMatmuls[layerIdx];

        // 1. Attention RMSNorm + Q8_1 quantize (T1.1 fused if enabled)
        normAndQuantize(gpuAttnNormWeights[layerIdx]);

        // 2. QKV projection: gpuXb -> gpuQkv (dp4a if available)
        launchMatmulDp4a(ml[0]);

        // 3. Fused causal conv1d + SiLU (reads position from tokenParams[0])
        conv1dPB.setLong(1, gpuConvState[layerIdx]);
        conv1dPB.setLong(2, gpuConvWeights[layerIdx]);
        launchKernel(conv1dFunc, conv1dGridDim, (int) blockSize, 0, conv1dPB.ptrs);

        // 5. Alpha/beta projections: gpuXb -> gpuAlpha, gpuBeta (dp4a)
        launchMatmulDp4a(ml[1]); // ssmAlpha
        launchMatmulDp4a(ml[2]); // ssmBeta

        // 6. Compute alpha/beta gates
        alphaBetaPB.setLong(2, gpuSsmAWeights[layerIdx]);
        alphaBetaPB.setLong(3, gpuDtBiasWeights[layerIdx]);
        launchKernel(alphaBetaFunc, 1, (int) Math.max(32, timeStepRank), 0, alphaBetaPB.ptrs);

        // 7. Gate projection: gpuXb -> gpuGate (raw, SiLU applied inside fused kernel) (dp4a)
        launchMatmulDp4a(ml[3]);

        // 8. Fused: DeltaNet recurrence + per-head RMSNorm + SiLU(gate) * output
        deltanetPB.setLong(0, gpuSsmState[layerIdx]);
        deltanetPB.setLong(5, gpuSsmNormWeights[layerIdx]);
        launchKernel(deltanetFunc, timeStepRank, stateSize, deltanetSharedMem, deltanetPB.ptrs);

        // 11. Output projection: deltaOut -> gpuX (accumulate)
        if (useDp4aOutputs) {
            quantizeBuffer(gpuDeltaOut, gpuScratchQ8, innerSize);
            launchMatmulOutputDp4a(ml[4], gpuScratchQ8);
        } else {
            launchMatmul(ml[4]);
        }

        // 12. FFN
        forwardFFN(layerIdx);
    }

    // === Attention Layer ===

    private void forwardAttentionLayer(int layerIdx, int position) {
        MatmulLaunch[] ml = layerMatmuls[layerIdx];

        // 1. Attention RMSNorm + Q8_1 quantize (T1.1 fused if enabled)
        normAndQuantize(gpuAttnNormWeights[layerIdx]);

        // 2. Q projection (packed Q+gate): gpuXb -> gpuQ [qGateDim] (dp4a)
        launchMatmulDp4a(ml[0]);

        // 3. Deinterleave Q+gate -> gpuQCompact + gpuAttnGate
        launchKernel(deinterleaveFunc, deinterleaveGridDim, (int) blockSize, 0, deinterleavePB.ptrs);

        // 4. K, V projections (dp4a)
        launchMatmulDp4a(ml[1]); // wk -> gpuK
        launchMatmulDp4a(ml[2]); // wv -> gpuV

        // 5. QK-norm
        if (gpuQNormWeights[layerIdx] != 0) {
            perHeadNormPB.setLong(0, gpuQCompact);
            perHeadNormPB.setLong(1, gpuQNormWeights[layerIdx]);
            launchKernel(perHeadNormFunc, headCount, perHeadNormBlockDim, perHeadNormSharedMem, perHeadNormPB.ptrs);
            perHeadNormPB.setLong(0, gpuK);
            perHeadNormPB.setLong(1, gpuKNormWeights[layerIdx]);
            launchKernel(perHeadNormFunc, headCountKV, perHeadNormBlockDim, perHeadNormSharedMem, perHeadNormPB.ptrs);
        }

        // 6. RoPE on Q and K
        ropePB.setLong(0, gpuQCompact);
        ropePB.setInt(3, headCount);
        launchKernel(ropeFunc, ropeQGridDim, (int) blockSize, 0, ropePB.ptrs);
        ropePB.setLong(0, gpuK);
        ropePB.setInt(3, headCountKV);
        launchKernel(ropeFunc, ropeKGridDim, (int) blockSize, 0, ropePB.ptrs);

        // 7. KV cache update
        kvPB.setLong(0, gpuKeyCache[layerIdx]);
        kvPB.setLong(1, gpuValueCache[layerIdx]);
        launchKernel(kvCacheUpdateFunc, kvUpdateGridDim, (int) blockSize, 0, kvPB.ptrs);

        // 8. Full attention
        flashAttn.launch(defaultStream, gpuXb2, gpuQCompact, gpuKeyCache[layerIdx], gpuValueCache[layerIdx],
            headCountKV, headSize, kvDim, 0, (float) (1.0 / Math.sqrt(headSize)), 0f, position);

        // 9. Sigmoid gate: xb2 *= sigmoid(attnGate)
        launchKernel(sigmoidMulFunc, deinterleaveGridDim, (int) blockSize, 0, sigmoidMulPB.ptrs);

        // 10. Wo projection: xb2 -> gpuX (accumulate)
        if (useDp4aOutputs) {
            quantizeBuffer(gpuXb2, gpuScratchQ8, qDim);
            launchMatmulOutputDp4a(ml[3], gpuScratchQ8);
        } else {
            launchMatmul(ml[3]);
        }

        // 11. FFN
        forwardFFN(layerIdx);
    }

    // === FFN (shared) ===

    private void forwardFFN(int layerIdx) {
        if (moeLayer[layerIdx]) {
            forwardMoeFFN(layerIdx);
            return;
        }
        MatmulLaunch[] ml = layerMatmuls[layerIdx];

        // FFN RMSNorm + Q8_1 quantize (T1.1 fused if enabled)
        normAndQuantize(gpuFfnNormWeights[layerIdx]);

        // Gate + Up projections (fused Q4_K, or dp4a, or separate)
        if (useFusedGateUp) {
            fusedGateUpPB.setLong(0, fusedGateWeights[layerIdx]);
            fusedGateUpPB.setLong(1, fusedUpWeights[layerIdx]);
            launchKernel(fusedGateUpFunc, fusedGateUpGridDim, fusedGateUpBlockDim, 0, fusedGateUpPB.ptrs);
        } else {
            launchMatmulDp4a(ml[5]); // ffnGate -> gpuHb
            launchMatmulDp4a(ml[6]); // ffnUp -> gpuHb2
        }

        // SiLU(gate) * up
        launchKernel(siluMulFunc, siluFFNGridDim, (int) blockSize, 0, siluMulPB.ptrs);

        // Down projection: gpuHb -> gpuX (accumulate)
        if (useDp4aOutputs) {
            quantizeBuffer(gpuHb, gpuScratchQ8, ffnDim);
            launchMatmulOutputDp4a(ml[7], gpuScratchQ8);
        } else {
            launchMatmul(ml[7]);
        }
    }

    @Override
    public void setMoeFfn(LayerGpuForwardPass.MoeFfn ffn) { this.moeFfn = ffn; }

    /**
     * MoE FFN of one layer: norm, then the gated shared expert queued on the GPU, then the normed
     * input downloaded (the one sync), the routed experts from the engine, uploaded and added.
     */
    private void forwardMoeFFN(int layerIdx) {
        normAndQuantize(gpuFfnNormWeights[layerIdx]);
        Qwen35LayerWeights lw = weights.layers()[layerIdx];
        boolean shared = sharedOnGpu[layerIdx];
        if (shared) {
            mmShIn.invalidate();
            mmShIn.matmul((CudaFloatTensor) lw.ffnGateShexp(), 0, gpuXb, gpuShG, sharedFfn, dim, false);
            mmShIn.matmul((CudaFloatTensor) lw.ffnUpShexp(), 0, gpuXb, gpuShU, sharedFfn, dim, false);
            launchKernel(shSiluFunc, (sharedFfn + (int) blockSize - 1) / (int) blockSize, (int) blockSize, 0, shSiluPB.ptrs());
            mmShDown.invalidate();
            mmShDown.matmul((CudaFloatTensor) lw.ffnDownShexp(), 0, gpuShG, gpuShOut, dim, sharedFfn, false);
            long logit = 0;
            if (lw.ffnGateInpShexp() != null) {
                mmShIn.matmul((CudaFloatTensor) lw.ffnGateInpShexp(), 0, gpuXb, gpuShLogit, 1, dim, false);
                logit = gpuShLogit;
            }
            shAxpyPB.setLong(2, logit);
            launchKernel(gateAxpyFunc, (dim + (int) blockSize - 1) / (int) blockSize, (int) blockSize, 0, shAxpyPB.ptrs());
        }
        if (!moeDeviceOnly) moeHost(layerIdx);
    }

    /**
     * Host part of a MoE layer: download the normed input (the layer's one synchronisation), run
     * the routed experts in the engine, upload their sum and add it (with the shared expert's gate
     * already applied on the device) to the residual.
     */
    private void moeHost(int layerIdx) {
        boolean shared = sharedOnGpu[layerIdx];
        long fb = Float.BYTES;
        cudaContext.readBuffer(gpuXb, hostMoe, (long) dim * fb);
        MemorySegment.copy(hostMoe, ValueLayout.JAVA_FLOAT, 0, moeXn, 0, dim);
        moeFfn.routed(layerIdx, moeXn, moeOut, !shared);
        MemorySegment routedHost = hostMoe.asSlice((long) dim * fb, (long) dim * fb);
        MemorySegment.copy(moeOut, 0, routedHost, ValueLayout.JAVA_FLOAT, 0, dim);
        cudaContext.writeBufferAsync(gpuRouted, routedHost, (long) dim * fb);
        launchKernel(gateAxpyFunc, (dim + (int) blockSize - 1) / (int) blockSize, (int) blockSize, 0, routedAxpyPB.ptrs());
    }

    // === CUDA Graph layer (fixed shared mem) ===

    private void forwardLayerKernels(int layerIdx) {
        if (isDeltaNet[layerIdx]) {
            forwardDeltaNetLayerKernels(layerIdx);
        } else {
            forwardAttentionLayerKernels(layerIdx);
        }
    }

    private void forwardDeltaNetLayerKernels(int layerIdx) {
        // Same as forwardDeltaNetLayer but for graph capture
        forwardDeltaNetLayer(layerIdx);
    }

    private void forwardAttentionLayerKernels(int layerIdx) {
        MatmulLaunch[] ml = layerMatmuls[layerIdx];

        normPB.setLong(2, gpuAttnNormWeights[layerIdx]);
        launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);

        launchMatmul(ml[0]); // wq
        launchKernel(deinterleaveFunc, deinterleaveGridDim, (int) blockSize, 0, deinterleavePB.ptrs);
        launchMatmul(ml[1]); // wk
        launchMatmul(ml[2]); // wv

        if (gpuQNormWeights[layerIdx] != 0) {
            perHeadNormPB.setLong(0, gpuQCompact);
            perHeadNormPB.setLong(1, gpuQNormWeights[layerIdx]);
            launchKernel(perHeadNormFunc, headCount, perHeadNormBlockDim, perHeadNormSharedMem, perHeadNormPB.ptrs);
            perHeadNormPB.setLong(0, gpuK);
            perHeadNormPB.setLong(1, gpuKNormWeights[layerIdx]);
            launchKernel(perHeadNormFunc, headCountKV, perHeadNormBlockDim, perHeadNormSharedMem, perHeadNormPB.ptrs);
        }

        ropePB.setLong(0, gpuQCompact);
        ropePB.setInt(3, headCount);
        launchKernel(ropeFunc, ropeQGridDim, (int) blockSize, 0, ropePB.ptrs);
        ropePB.setLong(0, gpuK);
        ropePB.setInt(3, headCountKV);
        launchKernel(ropeFunc, ropeKGridDim, (int) blockSize, 0, ropePB.ptrs);

        kvPB.setLong(0, gpuKeyCache[layerIdx]);
        kvPB.setLong(1, gpuValueCache[layerIdx]);
        launchKernel(kvCacheUpdateFunc, kvUpdateGridDim, (int) blockSize, 0, kvPB.ptrs);

        flashAttn.launch(defaultStream, gpuXb2, gpuQCompact, gpuKeyCache[layerIdx], gpuValueCache[layerIdx],
            headCountKV, headSize, kvDim, 0, (float) (1.0 / Math.sqrt(headSize)), 0f, -1);

        launchKernel(sigmoidMulFunc, deinterleaveGridDim, (int) blockSize, 0, sigmoidMulPB.ptrs);
        launchMatmul(ml[3]);
        forwardFFN(layerIdx);
    }

    // === Launch helpers ===

    private void launchMatmul(MatmulLaunch ml) {
        matmulPB.setLong(0, ml.weightPtr);
        matmulPB.setLong(1, ml.inputPtr);
        matmulPB.setLong(2, ml.outputPtr);
        matmulPB.setInt(3, ml.rows);
        matmulPB.setInt(4, ml.cols);
        matmulPB.setInt(5, ml.addToOutput);
        launchKernel(ml.function, ml.gridDim, ml.blockDim, ml.sharedMem, matmulPB.ptrs);
    }

    /** Launch matmul using dp4a kernel with Q8_1 quantized input (if available). */
    private void launchMatmulDp4a(MatmulLaunch ml) {
        if (!useDp4a || ml.dp4aType == 0) { launchMatmul(ml); return; }
        // Select the correct dp4a kernel for this quantization type
        MemorySegment func;
        int gridDim = ml.gridDim;
        int blockDim = ml.blockDim;
        switch (ml.dp4aType) {
            case 3: func = dp4aQ3kFunc; break;
            case 5: func = dp4aQ5kFunc; break;
            case 6:
                // Measured ~3% slower than the FP32 Q6_K kernel (byte loads on the unaligned 210-byte
                // block); opt-in like CudaForwardPass, -Dcuda.dp4a.q6=true.
                if (!DP4A_Q6) { launchMatmul(ml); return; }
                func = dp4aQ6kFunc; break;
            case 50: func = dp4aQ50Func; break;
            case 80: func = dp4aQ80Func; break;
            case 41: func = dp4aIq4nlFunc; break;
            case 42: func = dp4aIq4xsFunc; break;
            default:
                // T2.1: prefer multi-row Q4_K kernel when enabled
                if (useDp4aQ4kMr4 && ml.rows % 4 == 0) {
                    func = dp4aQ4kMr4Func;
                    // mr4 kernel layout: blockDim=128 (4 warps), each warp = 4 rows = 16 rows/block
                    blockDim = 128;
                    gridDim = (ml.rows + 15) / 16;
                } else {
                    func = dp4aFunc;
                }
                break;
        }
        if (func == null) { launchMatmul(ml); return; } // kernel not available
        dp4aPB.setLong(0, ml.weightPtr);
        dp4aPB.setLong(1, gpuXbQ8);    // Q8_1 input instead of FP32
        dp4aPB.setLong(2, ml.outputPtr);
        dp4aPB.setInt(3, ml.rows);
        dp4aPB.setInt(4, ml.cols);
        dp4aPB.setInt(5, ml.addToOutput);
        launchKernel(func, gridDim, blockDim, 0, dp4aPB.ptrs);
    }

    /** Quantize gpuXb to gpuXbQ8 (Q8_1 format). Call once after RMSNorm, before matmuls. */
    private void quantizeInput() {
        if (!useDp4a) return;
        launchKernel(quantizeFunc, quantizeGridDim, 256, 0, quantizePB.ptrs);
    }

    /**
     * T1.2: Quantize an arbitrary FP32 buffer to Q8_1 in scratch.
     * Used to prepare output-projection inputs (gpuXb2/gpuDeltaOut/gpuHb) for dp4a matmul.
     */
    private void quantizeBuffer(long inputPtr, long outputPtr, int size) {
        scratchQuantizePB.setLong(0, inputPtr);
        scratchQuantizePB.setLong(1, outputPtr);
        scratchQuantizePB.setInt(2, size);
        int gridDim = ((size / 32) + 7) / 8;
        launchKernel(quantizeFunc, gridDim, 256, 0, scratchQuantizePB.ptrs);
    }

    /**
     * T1.2: dp4a output-projection matmul. Caller has pre-quantized the input into `inputQ8`.
     * Falls back to plain {@link #launchMatmul} if dp4a unavailable for this tensor type.
     */
    private void launchMatmulOutputDp4a(MatmulLaunch ml, long inputQ8) {
        if (!useDp4aOutputs || ml.dp4aType == 0) { launchMatmul(ml); return; }
        MemorySegment func;
        switch (ml.dp4aType) {
            case 3: func = dp4aQ3kFunc; break;
            case 5: func = dp4aQ5kFunc; break;
            case 6:
                // Measured ~3% slower than the FP32 Q6_K kernel (byte loads on the unaligned 210-byte
                // block); opt-in like CudaForwardPass, -Dcuda.dp4a.q6=true.
                if (!DP4A_Q6) { launchMatmul(ml); return; }
                func = dp4aQ6kFunc; break;
            case 50: func = dp4aQ50Func; break;
            case 80: func = dp4aQ80Func; break;
            case 41: func = dp4aIq4nlFunc; break;
            case 42: func = dp4aIq4xsFunc; break;
            default: func = dp4aFunc; break;  // Q4_K (most output projections)
        }
        if (func == null) { launchMatmul(ml); return; }
        dp4aPB.setLong(0, ml.weightPtr);
        dp4aPB.setLong(1, inputQ8);
        dp4aPB.setLong(2, ml.outputPtr);
        dp4aPB.setInt(3, ml.rows);
        dp4aPB.setInt(4, ml.cols);
        dp4aPB.setInt(5, ml.addToOutput);
        launchKernel(func, ml.gridDim, ml.blockDim, 0, dp4aPB.ptrs);
    }

    /**
     * T1.1: rmsnorm(gpuX → gpuXb) + quantize_q8(gpuXb → gpuXbQ8).
     * If `cuda.fused_norm_quantize=true`, single fused kernel; otherwise legacy
     * two-kernel path. Identical mathematical output (no PPL change).
     *
     * Replaces the call pattern at the start of every layer:
     *   normPB.setLong(2, weights);
     *   launchKernel(rmsnormFusedFunc, ...);
     *   quantizeInput();
     *
     * Use this method instead — it dispatches to the fused or legacy path.
     */
    private void normAndQuantize(long weights) {
        if (useFusedNormQuantize) {
            // Fused: 1 launch, 1 HBM round-trip (writes both gpuXb and gpuXbQ8)
            rmsnormQuantizePB.setLong(3, weights);
            // slots 0 (gpuXb), 1 (gpuXbQ8), 2 (gpuX), 4 (dim), 5 (eps) set in constructor
            launchKernel(rmsnormQuantizeFunc, 1, (int) blockSize, normSharedMem, rmsnormQuantizePB.ptrs);
        } else {
            // Legacy: 2 launches (rmsnorm → gpuXb, then quantize → gpuXbQ8)
            normPB.setLong(2, weights);
            launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);
            quantizeInput();
        }
    }

    // === Profiling (opt-in via -Dqwen35.profile=true) ===
    // Per-kernel timing aggregation. Synchronizes after each kernel launch when enabled
    // — destroys throughput but reveals where the time goes per category.
    private static final boolean PROFILE_ENABLED =
        "true".equals(System.getProperty("qwen35.profile", "false"));
    private final java.util.LinkedHashMap<String, long[]> profileStats = new java.util.LinkedHashMap<>();
    private int profileTokenCount = 0;
    private static final int PROFILE_REPORT_EVERY = 30;

    private String labelForFunction(MemorySegment function) {
        // Order matters: most-frequent first
        if (function == dp4aFunc) return "matmul.dp4a.Q4_K";
        if (function == dp4aQ4kMr4Func) return "matmul.dp4a.Q4_K.mr4";
        if (function == dp4aQ5kFunc) return "matmul.dp4a.Q5_K";
        if (function == dp4aQ6kFunc) return "matmul.dp4a.Q6_K";
        if (function == fusedGateUpFunc) return "matmul.fused_gate_up";
        if (function == rmsnormFusedFunc) return "rmsnorm";
        if (function == quantizeFunc) return "quantize_input";
        if (function == rmsnormQuantizeFunc) return "rmsnorm+quantize (T1.1 fused)";
        if (function == deltanetFunc) return "deltanet (recurrence+norm+gate)";
        if (function == conv1dFunc) return "conv1d+silu";
        if (function == alphaBetaFunc) return "alpha_beta";
        if (function == deinterleaveFunc) return "deinterleave_q_gate";
        if (function == sigmoidMulFunc) return "sigmoid_mul";
        if (function == siluMulFunc) return "silu_mul (FFN)";
        if (function == ropeFunc) return "rope";
        if (function == kvCacheUpdateFunc) return "kv_cache_update";
        if (function == perHeadNormFunc) return "qk_norm_per_head";
        if (function == argmaxPartialFunc) return "argmax_partial";
        if (function == argmaxFinalFunc) return "argmax_final";
        return "matmul.other_fp32";
    }

    private void launchKernel(MemorySegment function, int gridDim, int blockDim,
                               int sharedMem, MemorySegment params) {
        long t0 = PROFILE_ENABLED ? System.nanoTime() : 0L;
        int err = CudaBindings.launchKernel(function,
            gridDim, 1, 1, blockDim, 1, 1,
            sharedMem, defaultStream, params, MemorySegment.NULL);
        if (err != CudaBindings.CUDA_SUCCESS) {
            throw new RuntimeException("CUDA error in Qwen35CudaForwardPass: " + err);
        }
        if (PROFILE_ENABLED) {
            cudaContext.finish();
            long elapsed = System.nanoTime() - t0;
            String label = labelForFunction(function);
            long[] stats = profileStats.computeIfAbsent(label, k -> new long[2]);
            stats[0] += elapsed;
            stats[1]++;
        }
    }

    /** Call this from the public forward methods after each token completes. */
    public void profileTokenComplete() {
        if (!PROFILE_ENABLED) return;
        profileTokenCount++;
        if (profileTokenCount % PROFILE_REPORT_EVERY == 0) {
            printProfileReport();
        }
    }

    private void printProfileReport() {
        long total = 0;
        for (long[] s : profileStats.values()) total += s[0];
        System.err.println();
        System.err.println("[qwen35.profile] token=" + profileTokenCount + ", total kernel time=" + (total / 1_000_000) + " ms");
        System.err.printf("  %-35s %10s %12s %10s %8s%n", "kernel", "calls", "total(ms)", "avg(us)", "%");
        // Sort by total time descending
        java.util.List<java.util.Map.Entry<String, long[]>> sorted =
            new java.util.ArrayList<>(profileStats.entrySet());
        sorted.sort((a, b) -> Long.compare(b.getValue()[0], a.getValue()[0]));
        for (var e : sorted) {
            long ns = e.getValue()[0];
            long n = e.getValue()[1];
            double pct = total > 0 ? 100.0 * ns / total : 0.0;
            System.err.printf("  %-35s %10d %12d %10d %7.1f%%%n",
                e.getKey(), n, ns / 1_000_000, n > 0 ? (ns / n) / 1_000 : 0L, pct);
        }
        System.err.println();
    }

    // === Upload helpers ===

    private long uploadNormWeights(FloatTensor tensor, int size) {
        float[] w = new float[size];
        for (int i = 0; i < size; i++) w[i] = tensor.getFloat(i);
        return bufferManager.uploadNormWeights(w);
    }

    private long uploadTensorAsFloats(FloatTensor tensor, int size) {
        float[] w = new float[size];
        for (int i = 0; i < size; i++) w[i] = tensor.getFloat(i);
        return uploadFloatArray(w);
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

    @Override
    public void close() {
        if (gemm != null) gemm.close();
        if (bHost != null) try { cudaContext.freePinnedHost(bHost); } catch (Exception ignored) {}
        if (graphExec != null) {
            try { cudaContext.destroyGraphExec(graphExec); } catch (Exception ignored) {}
        }
        for (MemorySegment g : layerGraphExec) {
            if (g != null) try { cudaContext.destroyGraphExec(g); } catch (Exception ignored) {}
        }
        if (hostMoe != null) try { cudaContext.freePinnedHost(hostMoe); } catch (Exception ignored) {}
        arena.close();
    }
}
