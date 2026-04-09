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
public class Qwen35CudaForwardPass implements AutoCloseable {

    private final CudaContext cudaContext;
    private final CudaBufferManager bufferManager;
    private final Arena arena;

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
    private final MemorySegment dp4aQ5kFunc;  // Q5_K dp4a kernel
    private final MemorySegment dp4aQ6kFunc;  // Q6_K dp4a kernel
    private final ParamBuffer quantizePB;   // 3 args: input, output, size
    private final ParamBuffer dp4aPB;       // 6 args: weights, input_q8, output, rows, cols, addToOutput
    private final int quantizeGridDim;
    private final int q8BufferSize;         // (dim/32) * 40 bytes

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
    private final MemorySegment attentionFullFunc;
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
        final int dp4aType;  // 0=none, 4=Q4_K, 5=Q5_K, 6=Q6_K

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
            else dp4aType = 0;
        }
    }

    // Reusable param buffers
    private final ParamBuffer matmulPB;       // 6 args
    private final ParamBuffer normPB;         // 5 args: out, in, weights, size, eps
    private final ParamBuffer ropePB;         // 8 args
    private final ParamBuffer kvPB;           // 6 args
    private final ParamBuffer attnPB;         // 9 args
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
    private final boolean graphAvailable;
    private final int graphAttnSharedMem;

    private final int gpuLayerCount;

    public Qwen35CudaForwardPass(ModelConfig config, Qwen35Weights weights,
                                  CudaBufferManager bufferManager, int maxSeqLen) {
        this.bufferManager = bufferManager;
        this.cudaContext = bufferManager.getCudaContext();
        this.arena = Arena.ofShared();
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
                long kvLayerBytes = (long) maxSeqLen * kvDim * fb;
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
        kvCacheUpdateFunc = cudaContext.compileKernel("kernels/cuda/attention.cu", "kv_cache_update");
        attentionFullFunc = cudaContext.compileKernel("kernels/cuda/attention.cu", "attention_full");
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
        MemorySegment qFunc = null, dFunc = null, d5kFunc = null, d6kFunc = null;
        try {
            qFunc = cudaContext.compileKernel("kernels/cuda/quantize_q8.cu", "quantize_q8");
            dFunc = cudaContext.compileKernel("kernels/cuda/matmul_q4_k_dp4a.cu", "matmul_q4_k_dp4a");
            dp4aAvail = true;
            // Also compile Q5_K and Q6_K dp4a variants
            try { d5kFunc = cudaContext.compileKernel("kernels/cuda/matmul_q5_k_dp4a.cu", "matmul_q5_k_dp4a"); }
            catch (Exception ignored) {}
            try { d6kFunc = cudaContext.compileKernel("kernels/cuda/matmul_q6_k_dp4a.cu", "matmul_q6_k_dp4a"); }
            catch (Exception ignored) {}
        } catch (Exception e) {
            System.err.println("Qwen35 CUDA: dp4a kernels unavailable — " + e.getMessage());
        }
        // dp4a enabled by default — disable with -Dcuda.dp4a=false
        useDp4a = dp4aAvail && !"false".equals(System.getProperty("cuda.dp4a", "true"));
        quantizeFunc = qFunc;
        dp4aFunc = dFunc;
        dp4aQ5kFunc = d5kFunc;
        dp4aQ6kFunc = d6kFunc;
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
            if (dp4aQ5kFunc != null) dp4aTypes += "+Q5_K";
            if (dp4aQ6kFunc != null) dp4aTypes += "+Q6_K";
            System.err.println("Qwen35 CUDA: dp4a enabled (" + dp4aTypes + " × Q8_1)");
        } else {
            q8BufferSize = 0;
            gpuXbQ8 = 0;
            quantizeGridDim = 0;
            quantizePB = null;
            dp4aPB = null;
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

        attnPB = new ParamBuffer(arena, 9);
        attnPB.setLong(0, gpuXb2);
        attnPB.setLong(1, gpuQCompact);
        attnPB.setInt(4, headCount);
        attnPB.setInt(5, headCountKV);
        attnPB.setInt(6, headSize);
        attnPB.setInt(7, kvDim);
        attnPB.setLong(8, gpuTokenParams);

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
            // FFN (shared by both layer types)
            layerMatmuls[i][5] = new MatmulLaunch((CudaFloatTensor) lw.ffnGate(), gpuXb, gpuHb, ffnDim, dim, 0);
            layerMatmuls[i][6] = new MatmulLaunch((CudaFloatTensor) lw.ffnUp(), gpuXb, gpuHb2, ffnDim, dim, 0);
            layerMatmuls[i][7] = new MatmulLaunch((CudaFloatTensor) lw.ffnDown(), gpuHb, gpuX, dim, ffnDim, 1); // accumulate
        }

        // CUDA graph availability
        this.graphAttnSharedMem = (maxSeqLen + 32) * Float.BYTES;
        this.graphAvailable = !Boolean.getBoolean("cuda.nograph")
                && cudaContext.isGraphApiAvailable()
                && outputMatmul != null
                && graphAttnSharedMem <= 48 * 1024;

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
                + ", graph: " + (graphAvailable ? "available" : "unavailable")
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
                if (!(lw.ffnGate() instanceof CudaFloatTensor)) return false;
            } else {
                if (!(lw.wq() instanceof CudaFloatTensor)) return false;
                if (!(lw.wk() instanceof CudaFloatTensor)) return false;
                if (!(lw.wo() instanceof CudaFloatTensor)) return false;
                if (!(lw.ffnGate() instanceof CudaFloatTensor)) return false;
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

    public void uploadXAndUpdateParams(float[] x, int position) {
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
        if (isDeltaNet[layerIdx]) {
            forwardDeltaNetLayer(layerIdx);
        } else {
            forwardAttentionLayer(layerIdx, position);
        }
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
                    try { cudaContext.endCapture(); } catch (Exception ignored) {}
                }
                System.err.println("Qwen35 CUDA graph: capture failed — " + e.getMessage());
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
                    try { cudaContext.endCapture(); } catch (Exception ignored) {}
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
                if (capturing) { try { cudaContext.endCapture(); } catch (Exception ignored) {} }
                System.err.println("Qwen35 CUDA graph: capture failed — " + e.getMessage());
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

    // === DeltaNet Layer ===

    private void forwardDeltaNetLayer(int layerIdx) {
        MatmulLaunch[] ml = layerMatmuls[layerIdx];

        // 1. Attention RMSNorm: gpuX -> gpuXb
        normPB.setLong(2, gpuAttnNormWeights[layerIdx]);
        launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);
        quantizeInput(); // Quantize gpuXb → gpuXbQ8 for dp4a matmuls

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
        launchMatmul(ml[4]);

        // 12. FFN
        forwardFFN(layerIdx);
    }

    // === Attention Layer ===

    private void forwardAttentionLayer(int layerIdx, int position) {
        MatmulLaunch[] ml = layerMatmuls[layerIdx];

        // 1. Attention RMSNorm: gpuX -> gpuXb
        normPB.setLong(2, gpuAttnNormWeights[layerIdx]);
        launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);
        quantizeInput(); // Quantize gpuXb → gpuXbQ8 for dp4a matmuls

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
        attnPB.setLong(2, gpuKeyCache[layerIdx]);
        attnPB.setLong(3, gpuValueCache[layerIdx]);
        int attnSharedMem = (position + 1 + 32) * Float.BYTES;
        launchKernel(attentionFullFunc, headCount, Math.min(256, (int) blockSize), attnSharedMem, attnPB.ptrs);

        // 9. Sigmoid gate: xb2 *= sigmoid(attnGate)
        launchKernel(sigmoidMulFunc, deinterleaveGridDim, (int) blockSize, 0, sigmoidMulPB.ptrs);

        // 10. Wo projection: xb2 -> gpuX (accumulate)
        launchMatmul(ml[3]);

        // 11. FFN
        forwardFFN(layerIdx);
    }

    // === FFN (shared) ===

    private void forwardFFN(int layerIdx) {
        MatmulLaunch[] ml = layerMatmuls[layerIdx];

        // FFN RMSNorm: gpuX -> gpuXb (using postAttnNorm weights)
        normPB.setLong(2, gpuFfnNormWeights[layerIdx]);
        launchKernel(rmsnormFusedFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);
        quantizeInput(); // Quantize gpuXb → gpuXbQ8 for FFN matmuls

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
        launchMatmul(ml[7]);
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

        attnPB.setLong(2, gpuKeyCache[layerIdx]);
        attnPB.setLong(3, gpuValueCache[layerIdx]);
        // Fixed shared mem for graph capture
        launchKernel(attentionFullFunc, headCount, Math.min(256, (int) blockSize), graphAttnSharedMem, attnPB.ptrs);

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
        switch (ml.dp4aType) {
            case 5: func = dp4aQ5kFunc; break;
            case 6: func = dp4aQ6kFunc; break;
            default: func = dp4aFunc; break; // Q4_K
        }
        if (func == null) { launchMatmul(ml); return; } // kernel not available
        dp4aPB.setLong(0, ml.weightPtr);
        dp4aPB.setLong(1, gpuXbQ8);    // Q8_1 input instead of FP32
        dp4aPB.setLong(2, ml.outputPtr);
        dp4aPB.setInt(3, ml.rows);
        dp4aPB.setInt(4, ml.cols);
        dp4aPB.setInt(5, ml.addToOutput);
        launchKernel(func, ml.gridDim, ml.blockDim, 0, dp4aPB.ptrs);
    }

    /** Quantize gpuXb to gpuXbQ8 (Q8_1 format). Call once after RMSNorm, before matmuls. */
    private void quantizeInput() {
        if (!useDp4a) return;
        launchKernel(quantizeFunc, quantizeGridDim, 256, 0, quantizePB.ptrs);
    }

    private void launchKernel(MemorySegment function, int gridDim, int blockDim,
                               int sharedMem, MemorySegment params) {
        int err = CudaBindings.launchKernel(function,
            gridDim, 1, 1, blockDim, 1, 1,
            sharedMem, defaultStream, params, MemorySegment.NULL);
        if (err != CudaBindings.CUDA_SUCCESS) {
            throw new RuntimeException("CUDA error in Qwen35CudaForwardPass: " + err);
        }
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
        if (graphExec != null) {
            try { cudaContext.destroyGraphExec(graphExec); } catch (Exception ignored) {}
        }
        arena.close();
    }
}
